"""A parser stays alive across a real fake-engine pause; no replay/finish."""
import threading
import json
import time
import urllib.error
import urllib.request
import unittest
from pathlib import Path
from unittest import mock
from serve.frontend import ChatTemplate
from serve.priority_scheduler import Role
from serve.server import ByteTokenizer, Service, MockEngine, serve
from serve.test_priority_protocol import PipeFixture


class StreamTests(PipeFixture):
    def interrupted(self,payload,cut,thinking=False,tools=None,sampling=None):
        e=self.start(payload);tok=ByteTokenizer()
        svc=Service(e,tok,ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        reached,release=threading.Event(),threading.Event()
        original=e.generate
        def gate(*args,**kwargs):
            for i,token in enumerate(original(*args,**kwargs)):
                if kwargs.get('_priority_role',Role.BACKGROUND) is Role.BACKGROUND and i==cut:
                    reached.set();release.wait(3)
                yield token
        e.generate=gate;events=[];errors=[]
        def background():
            try:events.extend(svc.run([1,2,3],thinking,tools or [],2048,sampling or {},threading.Event()))
            except Exception as error:errors.append(error)
        t=threading.Thread(target=background);t.start();self.wait(reached.is_set)
        h=e.priority.submit('GEN 20 4,5',Role.HERMES)
        self.wait(lambda:e.priority.parked==1);release.set()
        self.wait(lambda:h.state=='done');e.priority.release(h)
        self.assertTrue(t.is_alive())
        e.priority.lease_until=0;e.priority.tick();t.join(4)
        self.assertFalse(t.is_alive());self.assertEqual(errors,[])
        done=[x for k,x in events if k=='done']
        self.assertEqual(len(done),1)
        baseline=Service(MockEngine(tok,payload.decode()),tok,svc.template)
        expected=list(baseline.run([1,2,3],thinking,tools or [],2048,sampling or {},threading.Event()))
        def normalized(items):
            return [(e.kind,e.text,e.call.name if e.call else None,e.call.arguments if e.call else None)
                    for k,e in items if k=='event']
        self.assertEqual(normalized(events),normalized(expected))
        self.assertEqual(done[0]['completion_tokens'],[x for k,x in expected if k=='done'][0]['completion_tokens'])
        return [x for k,x in events if k=='event'],done[0],self.events.read_text()
    def test_incomplete_utf8_and_reasoning_survive_one_response(self):
        payload='long thought </think>\n\nHello € and goodbye'.encode()
        events,done,commands=self.interrupted(payload,payload.index('€'.encode())+1,thinking=True)
        self.assertEqual(''.join(e.text for e in events if e.kind=='reasoning'),'long thought ')
        self.assertEqual(''.join(e.text for e in events if e.kind=='content'),'Hello € and goodbye')
        self.assertEqual(done['completion_tokens'],len(payload)+1)
        self.assertEqual(commands.count('PGEN 1\n'),1)
    def test_incomplete_tool_arguments_keep_one_call_identity(self):
        payload=b'<tool_call>\n<function=lookup>\n<parameter=query>ocean heat evidence</parameter>\n</function>\n</tool_call>'
        tools=[{'name':'lookup','description':'lookup','parameters':{'type':'object','properties':{'query':{'type':'string'}}}}]
        events,done,_=self.interrupted(payload,65,tools=tools)
        calls=[e.call for e in events if e.kind=='tool_call']
        self.assertEqual(len(calls),1)
        self.assertEqual((calls[0].name,calls[0].arguments),('lookup',{'query':'ocean heat evidence'}))
        ids={e.call.id for e in events if e.call is not None}
        self.assertEqual(ids,{calls[0].id})
        self.assertEqual(done['finish'],'stop')  # HTTP maps the completed call to tool_calls.
    def test_stop_prefix_survives_pause_without_leaking(self):
        payload=b'answer ENDSTOP trailing'
        events,done,_=self.interrupted(payload,8,sampling={'stop':['ENDSTOP']})
        self.assertEqual(''.join(e.text for e in events if e.kind=='content'),'answer ')
        self.assertEqual(done['finish'],'stop')
    def test_trusted_context_defaults_background_and_restores(self):
        e=self.start();svc=Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        with self.assertRaises(ValueError):
            with svc.trusted_priority_role('hermes'):pass
        with svc.trusted_priority_role(Role.HERMES):
            self.assertIs(svc._priority_role.value,Role.HERMES)
        self.assertIs(svc._priority_role.value,Role.BACKGROUND)
        self.assertEqual(e.priority.lease_until,0)
    def test_immediate_headers_and_real_flushed_comment_during_full_pause(self):
        e=self.start(b'wire response')
        svc=Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        httpd=serve(svc,port=0)
        # A trusted test lease holds an ordinary background HTTP request. Forged
        # request role fields/headers must neither renew it nor create Hermes work.
        # The lease is planted as a used one: an admitted owner's lease is never
        # retracted, while a lease nothing was admitted under expires at once.
        lease=time.monotonic()+300;e.priority.lease_until=lease;e.priority.lease_admitted=True
        body={'messages':[{'role':'user','content':'hi'}],'max_tokens':128,'stream':True,
              'reasoning_effort':'none','priority':'hermes','role':'hermes'}
        request=urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-XTX-Client':'hermes-interactive',
                                                 'X-Priority':'hermes'})
        began=time.monotonic()
        try:
            with urllib.request.urlopen(request,timeout=14) as response:
                self.assertLess(time.monotonic()-began,2)
                self.assertEqual(response.headers['Content-Type'],'text/event-stream')
                lines=[]
                while True:
                    line=response.readline();self.assertTrue(line)
                    lines.append(line)
                    if line.startswith(b':'):
                        elapsed=time.monotonic()-began
                        self.assertGreaterEqual(elapsed,9)
                        self.assertLess(elapsed,13)
                        break
                self.assertIsNone(e.priority.active)
                self.assertEqual(e.priority.lease_until,lease)
                self.assertTrue(all(r.role is Role.BACKGROUND for r in e.priority.requests.values()))
                e.priority.lease_until=0;e.priority.tick()
                lines.append(response.read())
                wire=b''.join(lines)
                self.assertEqual(wire.count(b'data: [DONE]'),1)
                data=[json.loads(l[5:]) for l in wire.splitlines() if l.startswith(b'data: {')]
                self.assertEqual(''.join(c['choices'][0]['delta'].get('content','') for c in data),'wire response')
                self.assertEqual(sum(c['choices'][0]['finish_reason'] is not None for c in data),1)
                self.assertEqual(len({c['id'] for c in data}),1)
        finally:
            httpd.shutdown();httpd.server_close()

    def test_logprobs_request_refused_and_engine_survives(self):
        e=self.start(b'wire response')
        svc=Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        httpd=serve(svc,port=0)
        try:
            body={'messages':[{'role':'user','content':'hi'}],'max_tokens':128,'logprobs':True,'top_logprobs':5,
                  'reasoning_effort':'none','priority':'hermes','role':'hermes'}
            request=urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-XTX-Client':'hermes-interactive',
                                                       'X-Priority':'hermes'})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request,timeout=10)
            self.assertEqual(caught.exception.code,400)
            error=json.loads(caught.exception.read())
            self.assertEqual(error['error']['type'],'invalid_request_error')
            self.assertIn('priority',error['error']['message'])
            self.assertIsNone(e.proc.poll())                   # the engine process survived the refused request
            self.assertIsNone(e.priority.dead)                 # the scheduler never saw an untagged line
            follow={'messages':[{'role':'user','content':'hi'}],'max_tokens':128,'reasoning_effort':'none'}
            request=urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                data=json.dumps(follow).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=14) as response:
                self.assertEqual(response.code,200)
                answer=json.loads(response.read())
                self.assertEqual(answer['choices'][0]['message']['content'],'wire response')
        finally:
            httpd.shutdown();httpd.server_close()


if __name__=='__main__':unittest.main()
