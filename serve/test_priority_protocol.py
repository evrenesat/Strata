"""Real StrataEngine pipe reader against a bounded synthetic engine, no GPU."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from serve.priority_scheduler import Role, PriorityScheduler, PriorityUnavailable, PriorityProtocolError
from serve.test_priority_scheduler import Clock
from serve.server import StrataEngine, engine_args, Service, ByteTokenizer, EngineDied
from serve.frontend import ChatTemplate

FAKE_PRIORITY = r'''
import json,queue,sys,threading,time
args=sys.argv[1:]
reply=json.loads(open(args[args.index('--reply')+1]).read())
log=open(args[args.index('--events')+1],'a')
priority='--priority-serving' in args
print('INFO engine=fake priority_protocol=1 priority_backfill=1',flush=True)
print('READY 262144 stop',flush=True)
q=queue.Queue()
def reader():
    for line in sys.stdin:q.put(line.strip())
    q.put('QUIT')
threading.Thread(target=reader,daemon=True).start()
active=parked=None
def terminal(r,finish='stop'):
    print(('R %d '%r['id'] if priority else '')+f"DONE {r['at']} {r['prompt']} 1 1 {finish} 0 0 0",flush=True)
while True:
    try:line=q.get(timeout=.004 if active else None)
    except queue.Empty:line=''
    if line:
        log.write(line.split(' ',2)[0]+' '+line.split(' ',2)[1]+'\n' if ' ' in line else line+'\n');log.flush()
    if line=='QUIT':break
    if line.startswith(('PGEN ','PGENI ','GEN ','GENI ')):
        assert active is None
        fields=line.split();rid=int(fields[1]) if priority else 0
        payload=list(b'HERMES')+[257] if priority and rid%2==0 else reply
        active=dict(id=rid,at=0,tokens=payload,prompt=len(fields[-1].split(',')),limit=int(fields[2 if priority else 1]))
    elif line.startswith('PPAUSE '):
        assert active and active['id']==int(line.split()[1]) and parked is None
        print(f"R {active['id']} BLOCK state_transfer 180",flush=True)
        parked,active=active,None
        print(f"PAUSED {parked['id']} decode {parked['prompt']-1+parked['at']} {parked['at']}",flush=True)
    elif line.startswith('PRESUME '):
        assert not active and parked and parked['id']==int(line.split()[1])
        print(f"R {parked['id']} BLOCK state_transfer 180",flush=True)
        active,parked=parked,None
        print(f"RESUMED {active['id']}",flush=True)
    elif line.startswith('PCANCEL '):
        rid=int(line.split()[1])
        if active and active['id']==rid:terminal(active,'cancel');active=None
        elif parked and parked['id']==rid:terminal(parked,'cancel');parked=None
        else:raise RuntimeError('cancel without owner')
    elif line=='STOP':
        if active:terminal(active,'cancel');active=None
    if active:
        t=active['tokens'][active['at']];active['at']+=1
        print(('R %d '%active['id'] if priority else '')+f'T {t}',flush=True)
        if t==257 or active['at']>=active['limit']:terminal(active);active=None
'''


class PipeFixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        root=Path(self.tmp.name)
        self.script,self.reply,self.events=root/'engine.py',root/'reply.json',root/'events'
        self.script.write_text('#!'+sys.executable+'\n'+FAKE_PRIORITY)
        self.script.chmod(0o700)
        self.engine=None
    def start(self, payload=b'background response',priority=True,lazy=False):
        self.reply.write_text(json.dumps(list(payload)+[257]))
        args=['--reply',str(self.reply),'--events',str(self.events)]
        if priority:args+=['--priority-serving']
        self.engine=StrataEngine(str(self.script),args,lazy=lazy)
        return self.engine
    def tearDown(self):
        if self.engine:self.engine.close()
        self.tmp.cleanup()
    def wait(self,predicate,timeout=3):
        end=time.monotonic()+timeout
        while not predicate() and time.monotonic()<end:time.sleep(.001)
        self.assertTrue(predicate(),'bounded fixture timed out')


class ProtocolTests(PipeFixture):
    def test_central_reader_routes_foreground_and_resumes_same_seed_request(self):
        e=self.start(b'x'*100); out=[]
        t=threading.Thread(target=lambda:out.extend(e.generate([1,2,3],200,{'seed':424242},threading.Event())))
        t.start();self.wait(lambda:len(out)>3)
        hermes=list(e.generate([4,5],20,{},threading.Event(),_priority_role=Role.HERMES))
        self.assertEqual(bytes(hermes[:-1]),b'HERMES')
        self.assertTrue(t.is_alive());self.assertEqual(e.last['prompt_tokens'],2)
        e.priority.lease_until=0;e.priority.tick();t.join(3)
        self.assertFalse(t.is_alive());self.assertEqual(out,list(b'x'*100)+[257])
        verbs=self.events.read_text().splitlines()
        self.assertEqual(verbs.count('PGEN 1'),1)
        self.assertEqual(verbs.count('PPAUSE 1'),1)
        self.assertEqual(verbs.count('PRESUME 1'),1)
        self.assertFalse(any(x.startswith('STOP') for x in verbs))
    def test_parked_client_disconnect_cancels_only_itself(self):
        e=self.start(b'x'*100);out=[];cancel=threading.Event()
        t=threading.Thread(target=lambda:out.extend(e.generate([1,2,3],200,{},cancel)))
        t.start();self.wait(lambda:len(out)>2)
        h=e.priority.submit('GEN 20 4,5',Role.HERMES)
        self.wait(lambda:e.priority.parked==1)
        cancel.set();t.join(3)
        self.assertFalse(t.is_alive());self.assertIsNone(e.priority.dead)
        self.wait(lambda:h.state=='done');e.priority.release(h)
        self.assertIn('PCANCEL 1',self.events.read_text())
    def test_priority_disabled_legacy_protocol_unchanged(self):
        e=self.start(priority=False)
        self.assertEqual(list(e.generate([1,2,3],100,{},threading.Event())),list(b'background response')+[257])
        self.assertIsNone(e.priority)
        self.assertTrue(self.events.read_text().startswith('GEN 100'))
    def test_engine_death_ends_parked_request(self):
        e=self.start(b'x'*100);out=[];errors=[]
        def run():
            try:out.extend(e.generate([1,2,3],200,{},threading.Event()))
            except Exception as error:errors.append(error)
        t=threading.Thread(target=run);t.start();self.wait(lambda:len(out)>2)
        list(e.generate([4,5],20,{},threading.Event(),_priority_role=Role.HERMES))
        e.proc.kill();t.join(3)
        self.assertFalse(t.is_alive());self.assertEqual(len(errors),1)
    def test_mode_rejects_incompatible_configuration_before_launch(self):
        for arg in ('--batch','--slots','--layer-split','--pipeline-windows','--lookup-chain'):
            with self.assertRaises(ValueError):engine_args({'args':[arg,'2'],'priority':{'enabled':True}})
        with self.assertRaises(ValueError):engine_args({'args':[],'parallel':2,'priority':{'enabled':True}})
        with self.assertRaises(ValueError):engine_args({'args':[],'priority':{'idle_backfill':True}})
        self.assertEqual(engine_args({'args':[]}),[])
        self.assertIn('--priority-serving',engine_args({'args':[],'priority':{'enabled':True}}))
    def test_priority_control_commands_cannot_interleave(self):
        e=self.start()
        with self.assertRaises(ValueError):e.session_file('save','/unused')
        with self.assertRaises(ValueError):e.vram(None)
    def test_foreground_reservation_consumed_once_without_second_renewal(self):
        e=self.start();cancel=threading.Event()
        svc=Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        with svc.trusted_priority_role(Role.HERMES):
            with svc.foreground_reservation([1,2,3],100,{},cancel) as r:
                lease=e.priority.lease_until
                events=list(svc.run([1,2,3],False,[],100,{},cancel))
                self.assertTrue(r.claimed)
                self.assertEqual(e.priority.lease_until,lease)
        self.assertEqual(self.events.read_text().count('PGEN 1\n'),1)
        self.assertEqual(len([x for k,x in events if k=='done']),1)
        self.assertEqual(e.priority.requests,{})
    def test_preheader_refusal_is_typed_and_never_enters_header_scope(self):
        e=StrataEngine('unused',[],lazy=True)
        def send(line):
            if line.startswith('PPAUSE'):e.priority.feed('PREFUSED 1 pause_unavailable')
        e.priority=PriorityScheduler(send,timer=False)
        b=e.priority.submit('GEN 100 1,2,3');headers=[]
        with self.assertRaises(PriorityUnavailable):
            r=e.reserve_priority([4,5],10,{},threading.Event())
            headers.append('committed')
            e.release_priority(r)
        self.assertEqual(headers,[])
        self.assertEqual(e.priority.active,b.id)
        self.assertEqual(len(e.priority.requests),1)
    def test_preheader_disconnect_releases_exact_reservation(self):
        e=self.start(b'x'*100);cancel=threading.Event();cancel.set()
        with self.assertRaises(PriorityUnavailable):e.reserve_priority([1,2,3],200,{},cancel)
        self.assertEqual(e.priority.requests,{})
        self.assertIn('PCANCEL 1',self.events.read_text())
        self.assertNotIn('STOP',self.events.read_text())
    def test_native_failure_after_headers_ends_same_stream(self):
        e=self.start(b'x'*100);cancel=threading.Event()
        svc=Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        with svc.trusted_priority_role(Role.HERMES):
            with svc.foreground_reservation([1,2,3],200,{},cancel):
                headers_committed=True
                stream=svc.run([1,2,3],False,[],200,{},cancel)
                next(stream);e.proc.kill()
                with self.assertRaises(EngineDied):list(stream)
                self.assertTrue(headers_committed)
        self.assertEqual(e.priority.requests,{})
    def test_old_generator_cleanup_after_restart_preserves_new_request(self):
        e=self.start(b'x'*100)
        old_stream=e.generate([1,2,3],200,{},threading.Event());next(old_stream)
        old=e.priority;old.fail(PriorityProtocolError('fixture generation ended'))
        e.restart(tries=1)
        current=e.priority.submit('GEN 200 4,5,6')
        self.assertEqual(current.id,1)
        old_stream.close()
        self.assertIs(e.priority.requests[1],current)
        e.priority.tick();self.assertIsNone(e.priority.dead)
        self.wait(lambda:current.state=='done');e.release_priority(current)
    def test_old_reservation_scope_after_restart_preserves_new_request(self):
        e=self.start(b'x'*100);svc=Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        with svc.trusted_priority_role(Role.HERMES):
            with svc.foreground_reservation([1,2,3],200,{},threading.Event()) as old_request:
                old=e.priority;old.fail(PriorityProtocolError('fixture generation ended'))
                e.restart(tries=1);current=e.priority.submit('GEN 200 4,5,6')
                self.assertEqual(current.id,old_request.id)
            self.assertIs(e.priority.requests[1],current)
            e.release_priority(old_request)
        self.assertIs(e.priority.requests[1],current)
        e.priority.tick();self.assertIsNone(e.priority.dead)
        self.wait(lambda:current.state=='done');e.release_priority(current)
    def test_first_lazy_background_request_can_be_preempted(self):
        e=self.start(b'x'*200,lazy=True)
        self.assertIsNone(e.priority);self.assertTrue(e.priority_configured)
        svc=Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        out=[];errors=[];cancel=threading.Event()
        def background():
            try:out.extend(svc.run([1,2,3],False,[],512,{},cancel))
            except Exception as error:errors.append(error)
        t=threading.Thread(target=background,daemon=True);t.start()
        try:
            self.wait(lambda:e.priority is not None and e.priority.active==1 and len(out)>2)
            self.assertFalse(svc.fifo.locked())
            with svc.trusted_priority_role(Role.HERMES):
                with svc.foreground_reservation([4,5],20,{},threading.Event(),timeout=2):
                    foreground=list(svc.run([4,5],False,[],20,{},threading.Event()))
            self.assertTrue(t.is_alive());self.assertEqual(e.priority.parked,1)
            self.assertEqual(len([x for k,x in foreground if k=='done']),1)
            e.priority.lease_until=0;e.priority.tick();t.join(3)
            self.assertFalse(t.is_alive());self.assertEqual(errors,[])
            self.assertIn('PPAUSE 1',self.events.read_text())
        finally:
            cancel.set();e.close();t.join(3)
    def test_service_reasoning_continuation_holds_http_owner_after_expiry(self):
        e=self.start(b'x'*100);svc=Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        svc.reasoning_close_retry=True
        clock=Clock();clock.now=time.monotonic()+1;e.priority.clock=clock
        original_feed=e.priority.feed
        def feed(line):
            if line and line.startswith('R 2 DONE '):clock.now=e.priority.lease_until+1
            original_feed(line)
        e.priority.feed=feed
        background=[];errors=[];cancel=threading.Event()
        def bg():
            try:background.extend(e.generate([1,2,3],200,{},cancel))
            except Exception as error:errors.append(error)
        t=threading.Thread(target=bg,daemon=True);t.start();self.wait(lambda:len(background)>2)
        try:
            with svc.trusted_priority_role(Role.HERMES):
                with svc.foreground_reservation([4,5],256,{},threading.Event()):
                    foreground=list(svc.run([4,5],True,[],256,{},threading.Event()))
                self.assertTrue(t.is_alive());self.assertEqual(e.priority.parked,1)
                self.assertGreater(clock.now,e.priority.lease_until)
                self.assertNotIn('PRESUME 1',self.events.read_text())
                self.assertIn('PGEN 3',self.events.read_text())
            t.join(3);self.assertFalse(t.is_alive());self.assertEqual(errors,[])
            self.assertEqual(len([x for k,x in foreground if k=='done']),1)
            commands=self.events.read_text()
            self.assertLess(commands.index('PGEN 3'),commands.index('PRESUME 1'))
        finally:
            cancel.set();e.close();t.join(3)


if __name__=='__main__':unittest.main()
