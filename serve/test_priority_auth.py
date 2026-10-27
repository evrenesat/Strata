"""Actual HTTP auth -> validation -> logical turn -> native reservation ordering."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import time
import urllib.error
import urllib.request
import unittest
from unittest import mock
from serve.priority_auth import PriorityAuth,configured_auth,validate_chat
from serve.priority_scheduler import Role,PriorityUnavailable
from serve.server import Service,ByteTokenizer,MockEngine,EngineDied,serve
from serve.frontend import ChatTemplate
from serve.test_priority_protocol import PipeFixture

OWNER='owner-fixture-credential'
HERMES='hermes-fixture-credential'


class CredentialTests(unittest.TestCase):
    def test_file_schema_duplicates_empty_owner_and_permissions_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'roles.json'
            valid={'schema':1,'clients':[{'role':'hermes','key':HERMES}]}
            p.write_text(json.dumps(valid));p.chmod(0o640)
            auth=PriorityAuth.load(str(p),OWNER)
            self.assertIs(auth.role(HERMES),Role.HERMES)
            self.assertIsNone(auth.role(OWNER));self.assertIsNone(auth.role('wrong'))
            for value in ({'schema':True,'clients':valid['clients']},{'schema':1,'clients':[]},
                          {'schema':1,'clients':[{'role':'hermes','key':''}]},
                          {'schema':1,'clients':valid['clients']*2},
                          {'schema':1,'clients':[{'role':'hermes','key':OWNER}]},
                          {'schema':1,'clients':[{'role':'admin','key':HERMES}]}):
                p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):PriorityAuth.load(str(p),OWNER)
            p.write_text('{"schema":1,"schema":1,"clients":[]}')
            with self.assertRaises(ValueError):PriorityAuth.load(str(p),OWNER)
            p.write_text(json.dumps(valid));p.chmod(0o644)
            with self.assertRaises(ValueError):PriorityAuth.load(str(p),OWNER)
            p.chmod(0o640)
            with self.assertRaises(ValueError):PriorityAuth.load(str(p),'')
            link=Path(d)/'link';link.symlink_to(p)
            with self.assertRaises(OSError):PriorityAuth.load(str(link),OWNER)
    def test_priority_disabled_does_not_read_role_file(self):
        with mock.patch.object(PriorityAuth,'load') as load:
            self.assertIsNone(configured_auth({'priority':{'enabled':False,'clients_file':'bad'}},OWNER))
            self.assertIsNone(configured_auth({'priority':{'enabled':True}},OWNER))
            load.assert_not_called()
    def test_exact_endpoint_allowlist(self):
        for method,path in PriorityAuth.ALLOWED:self.assertTrue(PriorityAuth.allowed(method,path+'?x=1'))
        for method,path in [('POST','/settings'),('POST','/v1/messages'),('GET','/api/health'),
                            ('GET','/v1/models/'),('GET','/health/'),('DELETE','/v1/chat/completions')]:
            self.assertFalse(PriorityAuth.allowed(method,path))
    def test_all_role_keys_use_constant_time_comparison(self):
        auth=PriorityAuth(OWNER,[{'role':'hermes','key':HERMES},{'role':'hermes','key':HERMES+'2'}])
        import hmac
        original=hmac.compare_digest
        with mock.patch('serve.priority_auth.hmac.compare_digest',wraps=original) as compare:
            self.assertIs(auth.role(HERMES),Role.HERMES)
            self.assertEqual(compare.call_count,2)


class AuthHTTPTests(PipeFixture):
    def service(self,priority=True):
        self.e=self.start(b'background answer',priority=priority)
        self.svc=Service(self.e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        self.svc.api_key=OWNER
        self.svc.priority_auth=PriorityAuth(OWNER,[{'role':'hermes','key':HERMES}]) if priority else None
        self.httpd=serve(self.svc,port=0)
        self.addCleanup(self.httpd.server_close);self.addCleanup(self.httpd.shutdown)
        return self.svc
    def request(self,path='/v1/chat/completions',key=HERMES,body=None,method='POST',headers=None):
        if body is None and method=='POST':body={'messages':[{'role':'user','content':'fixture'}],
                                                  'max_tokens':128,'stream':True,'reasoning_effort':'none'}
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key,**(headers or {})}
        req=urllib.request.Request(f'http://127.0.0.1:{self.httpd.server_port}'+path,
                                   data=json.dumps(body).encode() if body is not None else None,
                                   method=method,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=5) as r:return r.status,dict(r.headers),r.read()
        except urllib.error.HTTPError as r:return r.code,dict(r.headers),r.read()
    def test_hermes_restricted_endpoints_and_discovery_do_not_renew(self):
        self.service()
        for path in ('/health','/v1/models'):
            self.assertEqual(self.request(path,method='GET')[0],200)
        for method,path in [('GET','/status'),('GET','/settings'),('GET','/config'),('GET','/metrics'),
                            ('GET','/api/requests'),('GET','/slots'),('GET','/props'),('GET','/'),
                            ('POST','/settings'),('POST','/load'),('POST','/unload'),
                            ('POST','/config'),('POST','/slots/0?action=save'),
                            ('POST','/v1/responses'),('POST','/v1/messages')]:
            self.assertEqual(self.request(path,method=method)[0],403,(method,path))
        self.assertEqual(self.e.priority.lease_until,0)
        self.assertFalse(self.e.priority.requests)
    def test_invalid_auth_and_malformed_body_do_not_reserve_or_renew(self):
        self.service()
        with mock.patch.object(self.svc,'foreground_reservation',wraps=self.svc.foreground_reservation) as reserve:
            self.assertEqual(self.request(key='wrong')[0],401)
            for patch in ({'messages':[]},{'stream':'yes'},{'max_tokens':True},{'temperature':float('nan')},
                          {'chat_template_kwargs':[]},{'strata_mcp':True},{'model':'unknown'}):
                body={'messages':[{'role':'user','content':'fixture'}],'max_tokens':128,**patch}
                self.assertEqual(self.request(body=body)[0],400,patch)
            reserve.assert_not_called()
        self.assertEqual(self.e.priority.lease_until,0)
    def test_owner_and_forged_labels_remain_background(self):
        self.service()
        body={'messages':[{'role':'user','content':'fixture'}],'max_tokens':128,'stream':True,
              'reasoning_effort':'none','role':'hermes','priority':'hermes'}
        code,_,wire=self.request(key=OWNER,body=body,headers={'X-XTX-Client':'hermes','X-Priority':'hermes'})
        self.assertEqual(code,200)
        chunks=[json.loads(line[5:]) for line in wire.splitlines() if line.startswith(b'data: {')]
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content','') for c in chunks),'background answer')
        self.assertEqual(self.e.priority.lease_until,0)
        self.assertEqual(self.request('/settings',key=OWNER,method='GET')[0],200)
    def test_foreground_reservation_before_headers_exactly_once_and_whole_turn(self):
        self.service();order=[];calls=[]
        original=self.svc.foreground_reservation
        @contextmanager
        def record(*args,**kwargs):
            order.append('reserve');calls.append(getattr(self.svc._priority_role,'value',None))
            with original(*args,**kwargs) as r:
                order.append('admitted');yield r
            order.append('released')
        handler=self.httpd.RequestHandlerClass
        original_sse=handler._sse
        def sse(request):
            order.append('headers')
            self.assertEqual(len(self.e.priority.turns),1)
            return original_sse(request)
        # The actual HTTP handler consumes the already-started native request.
        with mock.patch.object(self.svc,'foreground_reservation',side_effect=record),mock.patch.object(handler,'_sse',sse):
            code,headers,wire=self.request()
        self.assertEqual(code,200);self.assertEqual(order,['reserve','admitted','headers','released'])
        self.assertEqual(calls,[Role.HERMES]);self.assertGreater(self.e.priority.lease_until,time.monotonic()+295)
        self.assertEqual(self.events.read_text().count('PGEN '),1)
        self.assertEqual(wire.count(b'data: [DONE]'),1)
        self.assertFalse(self.e.priority.turns)
    def test_real_authenticated_http_preempts_background_without_reenqueue(self):
        self.service();out=[];cancel=threading.Event()
        # A long first request provides time for the real HTTP auth boundary.
        self.reply.write_text(json.dumps(list(b'x'*1500)+[257]))
        # The synthetic engine reads the reply at startup, so restart it with
        # the same reviewed startup path before either request is admitted.
        self.e.restart()
        thread=threading.Thread(target=lambda:out.extend(self.e.generate([1,2,3],2048,{},cancel)))
        thread.start();self.wait(lambda:len(out)>3)
        try:
            code,_,wire=self.request()
            self.assertEqual(code,200)
            self.assertIsNotNone(self.e.priority.parked)
            commands=self.events.read_text()
            self.assertEqual(commands.count('PGEN 2\n'),1)
            self.assertIn('PPAUSE 1\n',commands)
            self.assertNotIn('STOP',commands)
        finally:
            cancel.set();thread.join(4)
        self.assertFalse(thread.is_alive());self.assertIsNone(self.e.priority.dead)
    def test_preheader_refusal_is_503_with_retry_after_and_no_sse(self):
        self.service()
        @contextmanager
        def refuse(*_,**__):raise PriorityUnavailable('not parked');yield
        with mock.patch.object(self.svc,'foreground_reservation',side_effect=refuse):
            code,headers,wire=self.request()
        self.assertEqual(code,503);self.assertEqual(headers['Retry-After'],'5')
        self.assertIn(b'priority_unavailable',wire);self.assertNotIn(b'data:',wire)
    def test_failure_after_headers_is_error_stream_not_retroactive_503(self):
        self.service()
        def fail(*_,**__):
            yield 'ping',None
            raise EngineDied('fixture engine failed')
        with mock.patch.object(self.svc,'run',side_effect=fail):
            code,headers,wire=self.request()
        self.assertEqual(code,200);self.assertEqual(headers['Content-Type'],'text/event-stream')
        self.assertIn(b'"error"',wire);self.assertEqual(wire.count(b'data: [DONE]'),1)
        self.assertNotIn(b'finish_reason": "stop"',wire)
        self.assertFalse(self.e.priority.turns)
    def test_disabled_priority_preserves_owner_legacy_and_rejects_role_key(self):
        self.service(priority=False)
        self.assertEqual(self.request(key=OWNER)[0],200)
        self.assertEqual(self.request()[0],401)
        self.assertIsNone(self.e.priority)


if __name__=='__main__':unittest.main()
