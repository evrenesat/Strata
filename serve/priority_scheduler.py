"""One resident engine owner, one protected coding request, monotonic priority leases.

Roles are Python enum values supplied by a trusted caller, never request fields.
The HTTP layer deliberately supplies BACKGROUND until the CP4 auth boundary exists.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import queue
import math
import threading
import time


class Role(Enum):
    BACKGROUND = 'background'
    HERMES = 'hermes'


class PriorityUnavailable(RuntimeError):
    pass


class PriorityQueueFull(PriorityUnavailable):
    """Bounded queue admission refused a new submission (503, Retry-After 5)."""


class PriorityResumeRefusedPersistent(PriorityUnavailable):
    """The engine persistently refused a parked request's resume: its HTTP stream is closed
    honestly, but its native capsule was neither cancelled nor stopped."""
    code = 'priority_resume_refused_persistent'


class PriorityProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackfillAdmission:
    profile_sha256: str
    native_hot_image: bool
    completed_image_bytes: int
    cold_first_answer_seconds: tuple[float, float, float]  # idle, coding prefill, coding decode

    def validate(self, profile):
        if (self.profile_sha256 != profile or len(profile) != 64 or
                any(c not in '0123456789abcdef' for c in profile) or not self.native_hot_image or
                not 0 < self.completed_image_bytes <= 512*1024**2 or
                len(self.cold_first_answer_seconds) != 3 or
                any(not 0 < n <= 30 for n in self.cold_first_answer_seconds)):
            raise ValueError('idle backfill requires matching measured native admission')


@dataclass
class Request:
    id: int
    command: str
    role: Role
    state: str = 'queued'
    events: queue.Queue = field(default_factory=queue.Queue)
    heard: float = 0
    allowance: float = 120
    cancel_sent: bool = False
    started: bool = False
    backfill: bool = False
    max_new: int = 0
    prompt_tokens: int = 0
    admitted: threading.Event = field(default_factory=threading.Event)
    failure: Exception | None = None
    terminal: str | None = None
    claimed: bool = False
    control: str | None = None
    origin: object = field(default=None,repr=False,compare=False)
    turn: object = field(default=None,repr=False,compare=False)


@dataclass(eq=False)
class ForegroundTurn:
    """One trusted HTTP request, possibly several native continuations."""
    origin: object = field(repr=False)
    arrival: float
    admitted: bool = False
    closed: bool = False


class PriorityScheduler:
    def __init__(self, send, clock=time.monotonic, lease_seconds=300, silence_seconds=120,
                 fatal=None, timer=True, resume_retry_deadline_seconds=300,
                 max_queued_requests=64, max_queued_command_bytes=8*1024*1024):
        if not 1 <= lease_seconds <= 3600 or silence_seconds <= 0:
            raise ValueError('priority requires finite positive timing bounds')
        # A refused resume retries every 5 s; past this deadline the parked stream is closed
        # honestly instead of keeping the request (and the engine's background work) suspended.
        if not 5 <= resume_retry_deadline_seconds <= 86400:
            raise ValueError('priority requires a bounded resume retry deadline (5..86400 s)')
        if (isinstance(max_queued_requests, bool) or not isinstance(max_queued_requests, int) or
                max_queued_requests < 1 or isinstance(max_queued_command_bytes, bool) or
                not isinstance(max_queued_command_bytes, int) or max_queued_command_bytes < 1):
            raise ValueError('priority queue bounds must be positive whole numbers')
        self.send, self.clock, self.fatal = send, clock, fatal
        self.lease_seconds, self.silence = lease_seconds, silence_seconds
        self.resume_retry_deadline = resume_retry_deadline_seconds
        self.max_queued_requests = max_queued_requests
        self.max_queued_command_bytes = max_queued_command_bytes
        self.lock = threading.RLock()
        self.requests = {}
        self.turns = set()
        self.foreground, self.background = deque(), deque()
        self.active = self.parked = None
        self.retiring = {}   # id -> (Request, ack deadline): stranded capsules whose PCANCEL terminal is still owed
        self.retire_ack_seconds = 120   # a PCANCEL acknowledgment is a fast pipe round trip; past this the engine is stuck
        self.lease_until = 0.0
        self.lease_admitted = False     # a foreground request actually ran under the current lease
        self.idle_since = clock()
        self.next_id = 1
        self.dead = None
        self.closed = threading.Event()
        # Native completed-Hermes hot-image capability must be admitted before
        # any idle backfill. Full pause is the conservative default.
        self.backfill = False
        self.backfill_disabled_until = 0
        self.resume_retry_at = 0
        # The parked request's refusal episode: plain (lease-expiry) resume refusals repeat
        # every 5 s, so they are counted and bounded; backfill refusals are already damped
        # to one attempt per lease (backfill_disabled_until) and do not burn this deadline.
        self.resume_refusals = 0
        self.resume_first_refusal = None
        self.resume_last_reason = None
        # Bounded protocol evidence: refusal reason codes the engine actually sent.
        self.protocol_evidence = deque(maxlen=64)
        if timer:
            threading.Thread(target=self._timer, daemon=True).start()

    def _write(self, line):
        try:
            self.send(line)
        except Exception:
            self.fail(PriorityProtocolError('priority engine pipe failed'))

    def begin_turn(self, arrival=None):
        with self.lock:
            if self.dead:
                raise self.dead
            at=self.clock() if arrival is None else arrival
            if not isinstance(at,(float,int)) or not math.isfinite(at) or at>self.clock():
                raise ValueError('invalid trusted request arrival')
            turn=ForegroundTurn(self,at)
            self.turns.add(turn)
            return turn

    def end_turn(self, turn):
        with self.lock:
            if not isinstance(turn,ForegroundTurn) or turn.origin is not self:
                raise PriorityProtocolError('foreign foreground turn')
            if turn.closed:
                return
            turn.closed=True
            for r in list(self.requests.values()):
                if r.turn is turn and r.state not in ('done','failed'):
                    self.cancel(r)
            self.turns.discard(turn)
            self._dispatch()

    def submit(self, command, role=Role.BACKGROUND, arrival=None, turn=None):
        if not isinstance(role, Role):
            raise ValueError('priority role must come from the trusted boundary')
        if not command.startswith(('GEN ', 'GENI ')) or '\n' in command:
            raise ValueError('invalid priority generation command')
        fields = command.split()
        if (len(fields)<3 or not fields[1].isdecimal() or not 0<int(fields[1])<2**31 or
                any(not x.isdecimal() for x in fields[-1].split(','))):
            raise ValueError('invalid priority request budget or token IDs')
        with self.lock:
            if self.dead:
                raise self.dead
            if turn is not None and (not isinstance(turn,ForegroundTurn) or turn.origin is not self or
                    turn.closed or turn not in self.turns or role is not Role.HERMES):
                raise PriorityProtocolError('foreign or closed foreground turn')
            if self.next_id >= 2**63:
                raise PriorityProtocolError('priority request IDs exhausted')
            queued = [q for q in self.requests.values() if q.state == 'queued']
            if (len(queued) >= self.max_queued_requests or
                    sum(len(q.command) for q in queued) + len(command) > self.max_queued_command_bytes):
                # Only new submissions are refused; running and parked requests never count.
                raise PriorityQueueFull('priority_queue_full; retry after 5 seconds')
            r = Request(self.next_id, command, role, heard=self.clock(), allowance=self.silence)
            r.origin,r.turn=self,turn
            r.max_new, r.prompt_tokens = int(fields[1]), len(fields[-1].split(','))
            self.next_id += 1
            self.requests[r.id] = r
            if role is Role.HERMES:
                at = turn.arrival if turn else self.clock() if arrival is None else arrival
                if not isinstance(at, (float,int)) or not math.isfinite(at) or at>self.clock():
                    self.requests.pop(r.id)
                    raise ValueError('invalid trusted request arrival')
                if turn is None or not turn.admitted:
                    now=self.clock()
                    fallback_active=now<self.lease_until and self.backfill_disabled_until>=self.lease_until
                    if now >= self.lease_until:
                        self.lease_admitted = False   # a fresh lease starts with nothing admitted under it
                    self.lease_until = max(self.lease_until, at + self.lease_seconds)
                    if fallback_active:
                        self.backfill_disabled_until=self.lease_until
                    if turn:turn.admitted=True
                self.foreground.append(r.id)
                if self.active is not None:
                    running = self.requests[self.active]
                    if running.role is Role.BACKGROUND and running.state == 'running':
                        running.state = 'pausing'
                        running.control = 'pause'
                        self._write(f'PPAUSE {running.id}')
            else:
                self.background.append(r.id)
            self._dispatch()
            return r

    def _dispatch(self):
        if self.dead or self.active is not None or self.retiring:
            return   # let the engine finish a pending PCANCEL before another PPAUSE goes out
        for pending in (self.foreground, self.background):
            while pending and (pending[0] not in self.requests or self.requests[pending[0]].state != 'queued' or
                    (self.requests[pending[0]].turn is not None and self.requests[pending[0]].turn.closed)):
                pending.popleft()
        if self.foreground:
            r = self.requests[self.foreground.popleft()]
        elif any(turn.admitted and not turn.closed for turn in self.turns):
            return  # The admitted HTTP owner is between native continuations.
        else:
            if self.clock() < self.lease_until and not self.lease_admitted:
                # No foreground ever ran under this lease, none is queued and none is in
                # flight: it protects nobody, so it expires now and background proceeds
                # when the engine is next idle instead of after the full lease.
                self.lease_until = self.clock()
            if self.clock() < self.lease_until:
                if (self.parked is None or not self.backfill or self.clock() < self.backfill_disabled_until or
                        self.clock()-self.idle_since < 10):
                    return
                r = self.requests[self.parked]
                if r.state != 'parked' or self.clock() < self.resume_retry_at:
                    return
                self.active = r.id
                r.state, r.heard, r.allowance, r.backfill = 'resuming', self.clock(), 180.0, True
                r.control = 'resume'
                self._write(f'PRESUME {r.id} backfill=1')
                return
            elif self.parked is not None:
                r = self.requests[self.parked]
                if r.state != 'parked' or self.clock() < self.resume_retry_at:
                    return
                if self._resume_refused_past_deadline(r):
                    return
                r.backfill = False
                self.active = r.id
                r.state, r.heard, r.allowance = 'resuming', self.clock(), 180.0
                r.control = 'resume'
                self._write(f'PRESUME {r.id}')
                return
            elif self.background:
                r = self.requests[self.background.popleft()]
            else:
                return
        self.active = r.id
        r.state, r.heard, r.started = 'running', self.clock(), True
        if r.role is Role.HERMES:
            self.lease_admitted = True   # this lease has now actually served a foreground request
        verb, rest = r.command.split(' ', 1)
        self._write(f'P{verb} {r.id} {rest}')
        r.admitted.set()

    def queue_full(self):
        """Count-based pre-header admission check: True when one more queued submission
        would exceed max_queued_requests.  The byte bound is enforced at submit time (the
        command's size is known only then); this lets the HTTP layer refuse with 503 before
        streaming headers are committed."""
        with self.lock:
            return sum(1 for q in self.requests.values() if q.state == 'queued') >= self.max_queued_requests

    def cancel(self, r):
        with self.lock:
            if r.origin is not self or (self.requests.get(r.id) is not r and r.state not in ('done','failed')):
                raise PriorityProtocolError('foreign priority request cancellation')
            if r.state in ('done', 'failed'):
                return
            if r.state == 'queued':
                r.state = 'done'
                # Complete the admission state so a thread waiting in reserve_priority wakes
                # at once with a typed cancellation instead of spinning to its deadline.
                r.failure = PriorityUnavailable('priority reservation cancelled before admission')
                r.events.put('DONE 0 0 0 0 cancel 0 0 0')
                r.admitted.set()
                self._dispatch()
            elif not r.cancel_sent:
                r.cancel_sent = True
                r.state = 'cancelling'
                r.heard, r.allowance = self.clock(), 180.0
                self._write(f'PCANCEL {r.id}')

    @staticmethod
    def _reason_token(text):
        """The engine's refusal reason, reduced to a safe comment token (never client data)."""
        if 0 < len(text) <= 64 and all(c == '_' or '0' <= c <= '9' or 'a' <= c <= 'z' for c in text):
            return text
        return 'unspecified'

    def _resume_refused_past_deadline(self, r):
        """True once the parked request's plain-resume refusals passed the bounded policy."""
        if (self.resume_first_refusal is None or r.cancel_sent or
                self.clock() - self.resume_first_refusal < self.resume_retry_deadline):
            return False
        self._strand_parked(r)
        return True

    def _strand_parked(self, r):
        """Persistent resume refusal: close the parked request's stream with a typed error,
        stop retrying, and RETIRE the native capsule with PCANCEL.  The engine supports one
        parked capsule and refuses nested pauses, so an orphaned capsule would permanently
        disable preemption for every later background request; and with the HTTP stream
        already closed by the typed failure, a later resume could no longer deliver output
        to anyone.  Retiring frees the single capsule slot, lets background work proceed
        and keeps PPAUSE available.  The engine answers the cancellation with this
        request's terminal line, which is consumed silently below (the client already has
        its typed error)."""
        reason = self.resume_last_reason or 'unspecified'
        attempts = self.resume_refusals
        self.protocol_evidence.append((self.clock(), 'resume_refused_persistent', r.id, reason))
        self.resume_refusals, self.resume_first_refusal, self.resume_last_reason = 0, None, None
        self.parked = None
        r.state, r.control, r.stranded = 'failed', None, True
        r.failure = PriorityResumeRefusedPersistent(
            f'{PriorityResumeRefusedPersistent.code}: the engine refused resume {attempts} time(s) over '
            f'{self.resume_retry_deadline:.0f} s (last reason {reason}); the request and its parked '
            'capsule were retired (PCANCEL) so priority serving continues for new work')
        r.events.put(r.failure)
        r.admitted.set()
        r.cancel_sent = True
        self.retiring[r.id] = (r, self.clock() + self.retire_ack_seconds)   # identity survives the HTTP side's release until the terminal
        self._write(f'PCANCEL {r.id}')
        self.protocol_evidence.append((self.clock(), 'stranded_capsule_retired', r.id, 'pcancel'))
        self._dispatch()

    @staticmethod
    def _done_fields(r, fields):
        """The ordinary DONE validation shared by live and retiring terminals: complete
        field structure, sane counts, finite nonnegative timings."""
        return (len(fields) < 6 or not fields[1].isdecimal() or not fields[2].isdecimal() or
                int(fields[1]) > r.max_new or int(fields[2]) != r.prompt_tokens or
                any(not isinstance(x, str) or x == '' or not math.isfinite(float(x)) or float(x) < 0
                    for x in fields[3:5]))

    def _finish(self, r, line):
        if getattr(r, 'stranded', False) and r.state == 'failed':
            # The retired capsule's terminal (from the stranding PCANCEL): the client
            # already received the typed persistent-refusal error, so it is only recorded.
            self.protocol_evidence.append((self.clock(), 'stranded_capsule_terminal', r.id, line.split()[0] if line.split() else ''))
            return
        if r.state in ('done', 'failed'):
            raise PriorityProtocolError('duplicate priority terminal event')
        r.state = 'done'
        r.terminal = line
        r.admitted.set()
        r.events.put(line)
        if self.active == r.id:
            self.active = None
        if self.parked == r.id:
            self.parked = None
        self.idle_since = self.clock()
        self._dispatch()

    def feed(self, line):
        with self.lock:
            if self.dead:
                return
            try:
                self._feed(line)
            except (ValueError, KeyError, IndexError, PriorityProtocolError):
                self.fail(PriorityProtocolError('priority engine lost request ownership'))

    def _feed(self, line):
        if line is None:
            self.fail(PriorityProtocolError('priority engine ended'))
            return
        parts = line.strip().split(' ', 2)
        if parts[0] not in ('R', 'PAUSED', 'RESUMED', 'PREFUSED') or len(parts) < 2:
            raise PriorityProtocolError('untagged engine output in priority mode')
        if not parts[1].isdecimal() or not 0 < int(parts[1]) < 2**63:
            raise PriorityProtocolError('bad request ID')
        if int(parts[1]) in self.retiring:
            retired, _deadline = self.retiring[int(parts[1])]
            payload = parts[2] if len(parts) > 2 else ''
            if parts[0] != 'R':
                raise PriorityProtocolError('unexpected event for a retiring request')
            if payload.startswith('FATAL '):
                raise PriorityProtocolError('priority retirement ended fatal')
            if not payload.startswith('DONE '):
                raise PriorityProtocolError('retiring request emitted nonterminal output')
            fields = payload.split()
            if len(fields) != 9 or self._done_fields(retired, fields) or fields[5] != 'cancel':
                raise PriorityProtocolError('invalid retirement terminal')
            self.retiring.pop(int(parts[1]))
            self.protocol_evidence.append((self.clock(), 'stranded_capsule_terminal', retired.id, 'cancel'))
            self._dispatch()
            return
        r = self.requests[int(parts[1])]
        if parts[0] == 'PAUSED':
            if self.active != r.id or r.state not in ('pausing', 'cancelling') or self.parked is not None:
                raise PriorityProtocolError('unexpected pause acknowledgement')
            fields = parts[2].split()
            if (len(fields) not in (3,4) or fields[0] not in ('prefill','decode') or
                    not all(x.isdecimal() for x in fields[1:3])):
                raise PriorityProtocolError('bad pause boundary')
            position, produced = map(int, fields[1:3])
            if (produced>=r.max_new or
                    (fields[0]=='prefill' and (produced or not 0<position<r.prompt_tokens)) or
                    (fields[0]=='decode' and position!=r.prompt_tokens-1+produced)):
                raise PriorityProtocolError('pause boundary budget/position mismatch')
            if len(fields) == 4:
                if fields[3] != 'backfill=0':
                    raise PriorityProtocolError('bad pause capability')
                self.backfill_disabled_until = self.lease_until
            r.state = 'cancelling' if r.cancel_sent else 'parked'
            r.control = None
            self.parked, self.active = r.id, None
            self.resume_refusals, self.resume_first_refusal, self.resume_last_reason = 0, None, None
            self._dispatch()
        elif parts[0] == 'RESUMED':
            if self.active != r.id or r.state not in ('resuming', 'cancelling'):
                raise PriorityProtocolError('unexpected resume acknowledgement')
            self.parked = None
            r.state = 'cancelling' if r.cancel_sent else 'running'
            r.control = None
            r.heard, r.allowance = self.clock(), self.silence
            self.resume_refusals, self.resume_first_refusal, self.resume_last_reason = 0, None, None
            if self.foreground and not r.cancel_sent:
                r.state = 'pausing'
                r.control = 'pause'
                self._write(f'PPAUSE {r.id}')
        elif parts[0] == 'PREFUSED':
            reason = self._reason_token(parts[2] if len(parts) > 2 else '')
            if self.active == r.id and r.control=='resume' and r.state in ('resuming','cancelling'):
                was_backfill = r.backfill
                self.active, r.state = None, 'cancelling' if r.cancel_sent else 'parked'
                r.control = None
                self.resume_retry_at = self.clock()+5
                if was_backfill:
                    self.backfill_disabled_until = self.lease_until
                r.backfill = False
                kind = 'backfill_refused' if was_backfill else 'resume_refused'
                self.protocol_evidence.append((self.clock(), kind, r.id, reason))
                if was_backfill:
                    # Backfill refusals are damped to one attempt per lease; the comment
                    # still tells the parked client about the memory pressure.
                    r.events.put(f'WAIT priority backfill_refused reason={reason}')
                else:
                    # A plain resume refusal retries every 5 s: the attempt is counted and
                    # the whole episode is bounded by resume_retry_deadline_seconds.
                    self.resume_refusals += 1
                    if self.resume_first_refusal is None:
                        self.resume_first_refusal = self.clock()
                    self.resume_last_reason = reason
                    r.events.put(f'WAIT priority resume_refused attempt={self.resume_refusals} '
                                 f'reason={reason}')
                if not r.cancel_sent and not was_backfill and self._resume_refused_past_deadline(r):
                    return
                self._dispatch()
                return
            if self.active != r.id or r.control!='pause' or r.state not in ('pausing','cancelling'):
                raise PriorityProtocolError('unexpected pause refusal')
            self.protocol_evidence.append((self.clock(), 'pause_refused', r.id, reason))
            r.state = 'cancelling' if r.cancel_sent else 'running'
            r.control = None
            if r.backfill:
                self.backfill_disabled_until=self.lease_until
            while self.foreground:
                waiting = self.requests[self.foreground.popleft()]
                if waiting.state == 'queued':
                    waiting.state = 'failed'
                    waiting.failure=PriorityUnavailable('priority_unavailable; retry after 5 seconds')
                    waiting.events.put(waiting.failure)
                    waiting.admitted.set()
        else:
            payload = parts[2]
            if payload.startswith('FATAL '):
                self.fail(PriorityProtocolError('native priority state transfer failed'))
                return
            if payload.startswith('DONE ') or payload.startswith('ERR '):
                if not (self.active==r.id or (self.parked==r.id and r.cancel_sent) or
                        getattr(r, 'stranded', False)):
                    raise PriorityProtocolError('terminal output without owned request')
                if payload.startswith('DONE '):
                    f=payload.split()
                    if (self._done_fields(r, f) or f[5] not in ('stop','length','cancel')):
                        raise PriorityProtocolError('invalid terminal counts')
                self._finish(r, payload)
                return
            # Native PRESUME announces its bounded host/device transfer before
            # RESUMED. Only this control event is legal during that transition;
            # cancellation must not accidentally authorize tokens before ACK.
            resuming = r.control=='resume' and r.state in ('resuming','cancelling')
            transfer = payload.startswith('BLOCK ')
            if (self.active != r.id or (resuming and not transfer) or
                    (not resuming and r.state not in ('running','pausing','cancelling'))):
                raise PriorityProtocolError('output without active ownership')
            r.heard = self.clock()
            if transfer:
                values = payload.split()
                if len(values)!=3 or values[1]!='state_transfer':
                    raise PriorityProtocolError('invalid transfer progress')
                allowance = float(values[2])
                if not math.isfinite(allowance) or not 0 < allowance <= 180:
                    raise PriorityProtocolError('unbounded transfer allowance')
                r.allowance = allowance
            else:
                r.allowance = self.silence
            r.events.put(payload)

    def tick(self):
        with self.lock:
            if self.dead:
                return
            if self.active is not None:
                r = self.requests[self.active]
                if self.clock() - r.heard > r.allowance:
                    self.fail(PriorityProtocolError('active priority engine stopped making progress'))
                    return
            # A cancelled parked request must acknowledge without waking it.
            for r in self.requests.values():
                if r.state == 'cancelling' and self.clock() - r.heard > r.allowance:
                    self.fail(PriorityProtocolError('priority cancellation was not acknowledged'))
                    return
            for rid, (retired, deadline) in list(self.retiring.items()):
                if self.clock() > deadline:
                    # A living engine that never acknowledges the retirement leaves the
                    # single park slot (and this scheduler) stuck forever: fail honestly so
                    # the normal restart mechanism recovers both.
                    self.fail(PriorityProtocolError('priority retirement was not acknowledged'))
                    return
            self._dispatch()

    def fail(self, error):
        with self.lock:
            if self.dead:
                return
            self.dead = error
            for r in self.requests.values():
                if r.state not in ('done', 'failed'):
                    r.state = 'failed'
                    r.failure = error
                    r.events.put(error)
                    r.admitted.set()
            self.active = self.parked = None
            self.closed.set()
            if self.fatal:
                self.fatal(error)

    def _timer(self):
        while not self.closed.wait(.1):
            self.tick()

    def close(self):
        self.fail(PriorityProtocolError('priority engine closed'))

    def release(self, r):
        with self.lock:
            if r.origin is not self:
                raise PriorityProtocolError('foreign priority request release')
            current=self.requests.get(r.id)
            if current is not None and current is not r:
                raise PriorityProtocolError('priority request identity collision')
            if r.state not in ('done','failed'):
                raise PriorityProtocolError('cannot release a live request')
            self.foreground = deque(i for i in self.foreground if i != r.id)
            self.background = deque(i for i in self.background if i != r.id)
            if getattr(r, 'stranded', False) and self.retiring.get(r.id) is r:
                # The engine still owes this request's PCANCEL terminal; keep the identity
                # registered (under the retirement record) so the acknowledgment finds it.
                self.retiring[r.id] = r
            self.requests.pop(r.id, None)

    def admit_backfill(self, admission: BackfillAdmission, profile_sha256: str):
        if not isinstance(admission, BackfillAdmission):
            raise ValueError('backfill admission must be trusted measured evidence')
        admission.validate(profile_sha256)
        with self.lock:
            self.backfill = True

    def snapshot(self):
        with self.lock:
            return dict(active=self.active, parked=self.parked, lease_until=self.lease_until,
                        backfill=self.backfill, requests={i:r.state for i,r in self.requests.items()})
