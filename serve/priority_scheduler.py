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
                 fatal=None, timer=True):
        if not 1 <= lease_seconds <= 3600 or silence_seconds <= 0:
            raise ValueError('priority requires finite positive timing bounds')
        self.send, self.clock, self.fatal = send, clock, fatal
        self.lease_seconds, self.silence = lease_seconds, silence_seconds
        self.lock = threading.RLock()
        self.requests = {}
        self.turns = set()
        self.foreground, self.background = deque(), deque()
        self.active = self.parked = None
        self.lease_until = 0.0
        self.idle_since = clock()
        self.next_id = 1
        self.dead = None
        self.closed = threading.Event()
        # Native completed-Hermes hot-image capability must be admitted before
        # any idle backfill. Full pause is the conservative default.
        self.backfill = False
        self.backfill_disabled_until = 0
        self.resume_retry_at = 0
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
        if self.dead or self.active is not None:
            return
        for pending in (self.foreground, self.background):
            while pending and (pending[0] not in self.requests or self.requests[pending[0]].state != 'queued' or
                    (self.requests[pending[0]].turn is not None and self.requests[pending[0]].turn.closed)):
                pending.popleft()
        if self.foreground:
            r = self.requests[self.foreground.popleft()]
        elif any(turn.admitted and not turn.closed for turn in self.turns):
            return  # The admitted HTTP owner is between native continuations.
        elif self.clock() < self.lease_until:
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
        verb, rest = r.command.split(' ', 1)
        self._write(f'P{verb} {r.id} {rest}')
        r.admitted.set()

    def cancel(self, r):
        with self.lock:
            if r.origin is not self or (self.requests.get(r.id) is not r and r.state not in ('done','failed')):
                raise PriorityProtocolError('foreign priority request cancellation')
            if r.state in ('done', 'failed'):
                return
            if r.state == 'queued':
                r.state = 'done'
                r.events.put('DONE 0 0 0 0 cancel 0 0 0')
                self._dispatch()
            elif not r.cancel_sent:
                r.cancel_sent = True
                r.state = 'cancelling'
                r.heard, r.allowance = self.clock(), 180.0
                self._write(f'PCANCEL {r.id}')

    def _finish(self, r, line):
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
            self._dispatch()
        elif parts[0] == 'RESUMED':
            if self.active != r.id or r.state not in ('resuming', 'cancelling'):
                raise PriorityProtocolError('unexpected resume acknowledgement')
            self.parked = None
            r.state = 'cancelling' if r.cancel_sent else 'running'
            r.control = None
            r.heard, r.allowance = self.clock(), self.silence
            if self.foreground and not r.cancel_sent:
                r.state = 'pausing'
                r.control = 'pause'
                self._write(f'PPAUSE {r.id}')
        elif parts[0] == 'PREFUSED':
            if self.active == r.id and r.control=='resume' and r.state in ('resuming','cancelling'):
                self.active, r.state = None, 'cancelling' if r.cancel_sent else 'parked'
                r.control = None
                self.resume_retry_at = self.clock()+5
                if r.backfill:
                    self.backfill_disabled_until = self.lease_until
                r.backfill = False
                r.events.put('WAIT resume_unavailable')
                self._dispatch()
                return
            if self.active != r.id or r.control!='pause' or r.state not in ('pausing','cancelling'):
                raise PriorityProtocolError('unexpected pause refusal')
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
                if not (self.active==r.id or (self.parked==r.id and r.cancel_sent)):
                    raise PriorityProtocolError('terminal output without owned request')
                if payload.startswith('DONE '):
                    f=payload.split()
                    if (len(f)<6 or not f[1].isdecimal() or not f[2].isdecimal() or
                            int(f[1])>r.max_new or int(f[2])!=r.prompt_tokens or
                            any(not math.isfinite(float(x)) or float(x)<0 for x in f[3:5]) or
                            f[5] not in ('stop','length','cancel')):
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
