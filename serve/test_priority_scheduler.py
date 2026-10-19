import queue
import unittest
from serve.priority_scheduler import (BackfillAdmission, PriorityProtocolError,
                                      PriorityScheduler, PriorityUnavailable, Role)


class Clock:
    now = 0.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.sent = Clock(), []
        self.s = PriorityScheduler(self.sent.append, clock=self.clock, timer=False)
    def bg(self): return self.s.submit('GEN 100 seed=424242 1,2,3')
    def fg(self): return self.s.submit('GEN 10 4,5', Role.HERMES)
    def pause(self, r, phase='decode'):
        self.s.feed(f'PAUSED {r.id} {phase} 3 1')
    def done(self, r): self.s.feed(f'R {r.id} DONE 2 {r.prompt_tokens} 1 1 stop 0 0 0')
    def park(self):
        b = self.bg(); h = self.fg(); self.pause(b); self.done(h)
        return b
    def test_default_full_pause_and_same_request_after_lease(self):
        b = self.park()
        self.clock.advance(299); self.s.tick()
        self.assertEqual(self.sent, ['PGEN 1 100 seed=424242 1,2,3','PPAUSE 1','PGEN 2 10 4,5'])
        self.clock.advance(1); self.s.tick()
        self.assertEqual(self.sent[-1], 'PRESUME 1')
        self.s.feed('RESUMED 1'); self.s.feed('R 1 T 65'); self.done(b)
        self.assertEqual(list(b.events.queue), ['T 65','DONE 2 3 1 1 stop 0 0 0'])
    def test_299_second_renewal_preserves_original_lease(self):
        b = self.park(); self.clock.advance(299)
        h = self.fg(); self.done(h)
        self.assertEqual(self.s.lease_until, 599)
        self.clock.advance(299); self.s.tick()
        self.assertEqual(b.state, 'parked')
        self.clock.advance(1); self.s.tick()
        self.assertEqual(self.sent[-1], 'PRESUME 1')
    def test_internal_continuation_uses_original_http_arrival(self):
        h=self.s.submit('GEN 10 4,5',Role.HERMES,arrival=0);self.done(h)
        self.clock.advance(299)
        h=self.s.submit('GEN 10 4,5',Role.HERMES,arrival=0);self.done(h)
        self.assertEqual(self.s.lease_until,300)
        self.fg();self.assertEqual(self.s.lease_until,599)
    def test_same_response_remains_open_across_four_renewals(self):
        b=self.park()
        for _ in range(4):
            self.clock.advance(299);h=self.fg();self.done(h);self.s.tick()
            self.assertEqual(b.state,'parked');self.assertTrue(b.events.empty())
        self.clock.advance(300);self.s.tick();self.s.feed('RESUMED 1');self.done(b)
        self.assertEqual(sum(x.startswith('PGEN 1 ') for x in self.sent),1)
    def test_prefill_ack_validates_committed_position(self):
        b=self.bg();self.fg();self.s.feed('PAUSED 1 prefill 2 0')
        self.assertEqual(self.s.parked,b.id);self.assertIsNone(self.s.dead)
    def test_cancel_races_pause_refusal_without_killing_other_owner(self):
        b=self.bg();h=self.fg();self.s.cancel(b);self.s.feed('PREFUSED 1 pause_unavailable')
        self.assertIsNone(self.s.dead);self.assertEqual(h.state,'failed')
        self.s.feed('R 1 DONE 1 3 0 0 cancel 0 0 0')
        self.assertEqual(b.state,'done')
    def test_cancel_races_resume_refusal_without_hang(self):
        b=self.park();self.clock.advance(300);self.s.tick();self.s.cancel(b)
        self.s.feed('PREFUSED 1 resume_unavailable')
        self.assertIsNone(self.s.dead)
        self.s.feed('R 1 DONE 1 3 0 0 cancel 0 0 0');self.assertEqual(b.state,'done')
    def test_expiry_and_foreground_arrival_never_run_background(self):
        b = self.park(); self.clock.advance(300)
        h = self.fg()
        self.assertEqual(self.s.active, h.id)
        self.assertEqual(b.state, 'parked')
        self.assertNotIn('PRESUME 1',self.sent)
    def test_arrival_while_resume_in_flight_immediately_requests_repause(self):
        b=self.park(); self.clock.advance(300); self.s.tick()
        h=self.fg(); self.s.feed('RESUMED 1')
        self.assertEqual(self.sent[-1],'PPAUSE 1')
        self.pause(b); self.assertEqual(self.s.active,h.id)
    def test_foreground_fifo_and_background_fifo(self):
        b=self.bg(); b2=self.bg(); h1=self.fg(); h2=self.fg()
        self.pause(b); self.assertEqual(self.s.active,h1.id)
        self.done(h1); self.assertEqual(self.s.active,h2.id)
        self.done(h2); self.clock.advance(300); self.s.tick()
        self.s.feed('RESUMED 1'); self.done(b)
        self.assertEqual(self.s.active,b2.id)
    def test_transition_output_allowed_but_after_ack_forbidden(self):
        b=self.bg(); self.fg(); self.s.feed('R 1 T 65')
        self.assertIsNone(self.s.dead)
        self.pause(b); self.s.feed('R 1 T 66')
        self.assertIsInstance(self.s.dead,PriorityProtocolError)
    def test_parked_cancel_does_not_stop_foreground(self):
        b=self.bg(); h=self.fg(); self.pause(b); self.s.cancel(b)
        self.assertEqual(self.sent[-1],'PCANCEL 1')
        self.s.feed('R 1 DONE 1 3 0 0 cancel 0 0 0')
        self.assertEqual(self.s.active,h.id)
        self.s.feed(f'R {h.id} T 72'); self.done(h)
        self.assertIsNone(self.s.dead)
    def test_queued_cancel_never_reaches_engine(self):
        self.bg(); b=self.bg(); self.s.cancel(b); self.s.release(b)
        self.assertEqual(len(self.sent),1)
        self.assertNotIn(b.id,self.s.requests)
    def test_cancel_during_pause_acknowledges_once(self):
        b=self.bg(); h=self.fg(); self.s.cancel(b); self.pause(b)
        self.s.feed('R 1 DONE 1 3 0 0 cancel 0 0 0'); self.done(h)
        self.assertEqual(sum(x.startswith('DONE ') for x in b.events.queue),1)
    def test_engine_death_releases_every_waiter(self):
        b=self.bg(); h=self.fg(); self.pause(b); self.s.feed(None)
        self.assertEqual((b.state,h.state),('failed','failed'))
        self.assertIsInstance(b.events.get_nowait(),Exception)
    def test_transfer_fatal_is_global(self):
        b=self.bg(); h=self.fg(); self.s.feed('R 1 FATAL transfer')
        self.assertEqual((b.state,h.state),('failed','failed'))
    def test_pause_refusal_keeps_coding_and_fails_foreground(self):
        b=self.bg(); h=self.fg(); self.s.feed('PREFUSED 1 pause_unavailable')
        self.assertEqual((b.state,h.state),('running','failed'))
        self.assertIsInstance(h.events.get_nowait(),PriorityUnavailable)
        self.assertEqual(self.s.lease_until,300)
    def test_suspended_watchdog_does_not_require_fake_progress(self):
        b=self.park(); self.clock.advance(299); self.s.tick()
        self.assertEqual(b.state,'parked'); self.assertIsNone(self.s.dead)
    def test_active_watchdog_expires_even_with_transport_heartbeats(self):
        self.bg(); self.clock.advance(121); self.s.tick()
        self.assertIsInstance(self.s.dead,PriorityProtocolError)
    def test_transfer_allowance_finite(self):
        self.bg(); self.s.feed('R 1 BLOCK state_transfer 180')
        self.clock.advance(179); self.s.tick(); self.assertIsNone(self.s.dead)
        self.clock.advance(2); self.s.tick(); self.assertIsNotNone(self.s.dead)
    def test_native_resume_transfer_notice_precedes_ack_without_renewing_lease(self):
        b=self.park();self.clock.advance(301);self.s.tick()
        self.assertEqual(self.sent[-1],'PRESUME 1')
        self.s.feed('R 1 BLOCK state_transfer 180')
        self.assertIsNone(self.s.dead)
        self.assertEqual((b.state,b.control,b.heard,b.allowance),('resuming','resume',301,180))
        self.assertEqual(self.s.lease_until,300)
        self.clock.advance(179);self.s.tick();self.assertIsNone(self.s.dead)
        self.s.feed('RESUMED 1');self.s.feed('R 1 T 65');self.done(b)
        self.assertEqual(list(b.events.queue),['BLOCK state_transfer 180','T 65','DONE 2 3 1 1 stop 0 0 0'])
    def test_only_bounded_owned_transfer_progress_allowed_before_resume_ack(self):
        for cancelling in (False,True):
            for payload in ('T 65','PP 2 3 1 1','REUSED 2','BLOCK other 180','BLOCK state_transfer',
                    'BLOCK state_transfer 0','BLOCK state_transfer 181','BLOCK state_transfer inf',
                    'BLOCK state_transfer nan','BLOCK state_transfer 180 extra'):
                with self.subTest(cancelling=cancelling,payload=payload):
                    clock=Clock();s=PriorityScheduler(lambda _:None,clock=clock,timer=False)
                    b=s.submit('GEN 100 1,2,3');h=s.submit('GEN 10 4,5',Role.HERMES)
                    s.feed('PAUSED 1 decode 3 1');s.feed('R 2 DONE 2 2 1 1 stop 0 0 0')
                    clock.advance(301);s.tick()
                    if cancelling:s.cancel(b)
                    s.feed('R 1 '+payload)
                    self.assertIsInstance(s.dead,PriorityProtocolError)
    def test_cancel_and_fatal_remain_owned_during_native_resume_transfer(self):
        b=self.park();self.clock.advance(301);self.s.tick();self.s.cancel(b)
        self.s.feed('R 1 BLOCK state_transfer 180')
        self.assertIsNone(self.s.dead);self.assertTrue(b.cancel_sent)
        self.s.feed('RESUMED 1');self.s.feed('R 1 DONE 1 3 0 0 cancel 0 0 0')
        self.assertEqual(b.state,'done');self.assertEqual(self.sent.count('PCANCEL 1'),1)
        self.assertIsNone(self.s.active);self.assertIsNone(self.s.parked)
        clock=Clock();s=PriorityScheduler(lambda _:None,clock=clock,timer=False)
        b=s.submit('GEN 100 1,2,3');s.submit('GEN 10 4,5',Role.HERMES)
        s.feed('PAUSED 1 decode 3 1');s.feed('R 2 DONE 2 2 1 1 stop 0 0 0')
        clock.advance(301);s.tick();s.feed('R 1 BLOCK state_transfer 180');s.feed('R 1 FATAL transfer')
        self.assertIsInstance(s.dead,PriorityProtocolError);self.assertEqual(b.state,'failed')
    def test_foreign_generation_cleanup_cannot_remove_reused_numeric_id(self):
        old=self.bg();self.s.fail(PriorityProtocolError('old generation ended'))
        new=PriorityScheduler(lambda _:None,clock=self.clock,timer=False)
        current=new.submit('GEN 100 1,2,3')
        self.assertEqual(old.id,current.id)
        with self.assertRaises(PriorityProtocolError):new.release(old)
        with self.assertRaises(PriorityProtocolError):new.cancel(old)
        self.s.release(old);self.s.release(old)  # same-origin outer scope is idempotent
        self.assertIs(new.requests[1],current);new.tick();self.assertIsNone(new.dead)
    def test_http_turn_keeps_priority_between_continuations_after_lease_expiry(self):
        b=self.bg();turn=self.s.begin_turn(arrival=0)
        h=self.s.submit('GEN 10 4,5',Role.HERMES,turn=turn);self.pause(b)
        self.clock.advance(301);self.done(h);self.s.release(h)
        self.assertIsNone(self.s.active);self.assertEqual(b.state,'parked')
        self.assertNotIn('PRESUME 1',self.sent)
        continuation=self.s.submit('GEN 10 4,5',Role.HERMES,turn=turn)
        self.assertEqual(self.s.active,continuation.id);self.assertEqual(self.s.lease_until,300)
        self.done(continuation);self.s.release(continuation)
        self.assertNotIn('PRESUME 1',self.sent)
        self.s.end_turn(turn);self.s.end_turn(turn)
        self.assertEqual(self.sent[-1],'PRESUME 1')
    def test_http_turn_release_cancels_only_its_native_request(self):
        b=self.bg();turn=self.s.begin_turn()
        h=self.s.submit('GEN 10 4,5',Role.HERMES,turn=turn);self.pause(b)
        self.s.end_turn(turn)
        self.assertEqual(self.sent[-1],f'PCANCEL {h.id}')
        self.s.feed(f'R {h.id} DONE 1 2 0 0 cancel 0 0 0')
        self.assertEqual(b.state,'parked');self.assertNotIn('PCANCEL 1',self.sent)
    def test_foreign_or_closed_http_turn_cannot_admit_work(self):
        turn=self.s.begin_turn();new=PriorityScheduler(lambda _:None,timer=False)
        with self.assertRaises(PriorityProtocolError):new.submit('GEN 10 1',Role.HERMES,turn=turn)
        self.s.end_turn(turn)
        with self.assertRaises(PriorityProtocolError):self.s.submit('GEN 10 1',Role.HERMES,turn=turn)
        self.assertEqual(self.s.lease_until,0)
    def test_backfill_refusal_carries_through_same_rolling_lease_renewal(self):
        b=self.park();self.s.admit_backfill(BackfillAdmission('a'*64,True,1,(1,1,1)),'a'*64)
        self.clock.advance(10);self.s.tick();self.s.feed('R 1 BLOCK state_transfer 180')
        self.s.feed('PREFUSED 1 backfill_unavailable')
        self.clock.advance(289);h=self.fg();self.done(h)
        self.assertEqual((self.s.lease_until,self.s.backfill_disabled_until),(599,599))
        self.clock.advance(10);self.s.tick();self.assertEqual(b.state,'parked')
        self.assertEqual(self.sent.count('PRESUME 1 backfill=1'),1)
        self.clock.advance(290);self.s.tick();self.assertEqual(self.sent[-1],'PRESUME 1')
    def test_backfill_refusal_can_reset_after_actual_lease_lapse(self):
        self.park();self.s.admit_backfill(BackfillAdmission('a'*64,True,1,(1,1,1)),'a'*64)
        self.clock.advance(10);self.s.tick();self.s.feed('PREFUSED 1 backfill_unavailable')
        self.clock.advance(291);h=self.fg();self.done(h)
        self.clock.advance(10);self.s.tick()
        self.assertEqual(self.sent[-1],'PRESUME 1 backfill=1')
    def test_unbounded_allowance_and_unknown_owner_fail_closed(self):
        for line in ('R 1 BLOCK transfer inf','R 1 BLOCK transfer nan','R 99 T 1','T 1','RESUMED 1'):
            s=PriorityScheduler(lambda _:None,timer=False); s.submit('GEN 1 1')
            s.feed(line); self.assertIsNotNone(s.dead,line)
    def test_cancel_ack_timeout_is_bounded(self):
        b=self.park(); self.s.cancel(b); self.clock.advance(181); self.s.tick()
        self.assertIsNotNone(self.s.dead)
    def test_backfill_only_after_measured_admission_and_ten_seconds(self):
        b=self.park(); profile='a'*64
        self.s.admit_backfill(BackfillAdmission(profile,True,1234,(1,2,3)),profile)
        self.clock.advance(9); self.s.tick(); self.assertEqual(b.state,'parked')
        self.clock.advance(1); self.s.tick()
        self.assertEqual(self.sent[-1],'PRESUME 1 backfill=1')
        self.assertEqual(self.s.lease_until,300)
        self.s.feed('RESUMED 1'); h=self.fg()
        self.assertEqual(self.sent[-1],'PPAUSE 1')
        self.pause(b); self.assertEqual(self.s.active,h.id)
    def test_backfill_refusal_disables_only_current_lease(self):
        b=self.park(); self.s.admit_backfill(BackfillAdmission('a'*64,True,1,(1,1,1)),'a'*64)
        self.clock.advance(10); self.s.tick(); self.s.feed('PREFUSED 1 backfill_unavailable')
        self.clock.advance(280); self.s.tick(); self.assertEqual(b.state,'parked')
        self.clock.advance(10); self.s.tick(); self.assertEqual(self.sent[-1],'PRESUME 1')
    def test_unmeasured_or_foreign_backfill_admission_rejected(self):
        for admission in (None,BackfillAdmission('a'*64,False,1,(1,1,1)),
            BackfillAdmission('a'*64,True,513*1024**2,(1,1,1)),
            BackfillAdmission('a'*64,True,1,(1,31,1))):
            with self.assertRaises(ValueError): self.s.admit_backfill(admission,'a'*64)
        self.assertFalse(self.s.backfill)
    def test_roles_cannot_be_strings_or_headers(self):
        for role in ('hermes', {'X-XTX-Client':'hermes'},None):
            with self.assertRaises(ValueError): self.s.submit('GEN 1 1',role)
        self.assertEqual(self.s.lease_until,0)
    def test_duplicate_terminal_is_fatal(self):
        b=self.bg(); self.done(b); self.done(b)
        self.assertIsNotNone(self.s.dead)


if __name__=='__main__': unittest.main()
