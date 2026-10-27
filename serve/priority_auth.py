"""Protected in-process role credentials; HTTP labels never grant priority."""
from __future__ import annotations
import hmac
import json
import math
import os
import stat
from urllib.parse import urlsplit
from serve.priority_scheduler import Role


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate priority credential field')
        result[key]=value
    return result


class PriorityAuth:
    """Owner remains BACKGROUND/full access; distinct Hermes keys are restricted."""
    ALLOWED=frozenset({('GET','/health'),('GET','/v1/models'),('POST','/v1/chat/completions')})

    def __init__(self,owner_key,clients):
        if not isinstance(owner_key,str) or not owner_key:raise ValueError('priority credentials require owner authentication')
        if not isinstance(clients,list) or not 1<=len(clients)<=16:raise ValueError('invalid priority credential entries')
        keys=[]
        for entry in clients:
            if (not isinstance(entry,dict) or set(entry)!={'role','key'} or entry.get('role')!='hermes'):
                raise ValueError('invalid priority credential entry')
            key=entry.get('key')
            if (not isinstance(key,str) or not 16<=len(key)<=256 or
                    any(not 33<=ord(c)<=126 for c in key)):
                raise ValueError('invalid priority credential key')
            if hmac.compare_digest(key.encode(),owner_key.encode()) or key in keys:
                raise ValueError('duplicate priority credential key')
            keys.append(key)
        self._keys=tuple(k.encode() for k in keys)

    @classmethod
    def load(cls,path,owner_key):
        if not isinstance(path,str) or not os.path.isabs(path):raise ValueError('priority credential path must be absolute')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or stat.S_IMODE(info.st_mode) not in (0o600,0o640):
                raise ValueError('priority credential file must be root-owned mode0600 or0640')
            if not 0<info.st_size<=65536:raise ValueError('priority credential file size is invalid')
            with os.fdopen(fd,'r',encoding='utf-8') as stream:
                fd=-1
                value=json.load(stream,object_pairs_hook=_object)
            if not isinstance(value,dict) or set(value)!={'schema','clients'} or type(value['schema']) is not int or value['schema']!=1:
                raise ValueError('invalid priority credential schema')
            return cls(owner_key,value['clients'])
        finally:
            if fd>=0:os.close(fd)

    def role(self,given):
        raw=given.encode('latin-1','replace')
        found=False
        for key in self._keys:found |= hmac.compare_digest(raw,key)
        return Role.HERMES if found else None

    @classmethod
    def allowed(cls,method,target):return (method,urlsplit(target).path) in cls.ALLOWED


def configured_auth(config,owner_key):
    priority=config.get('priority')
    if not isinstance(priority,dict) or priority.get('enabled') is not True:return None
    path=priority.get('clients_file')
    if path is None:return None # background-only HTTP until an explicit protected map is configured
    return PriorityAuth.load(path,owner_key)


def validate_chat(req,models):
    """Reject malformed or server-control input before foreground admission/lease."""
    if not isinstance(req,dict):raise ValueError('send a JSON object')
    messages=req.get('messages')
    if not isinstance(messages,list) or not messages:raise ValueError('messages must be a nonempty list')
    if any(not isinstance(m,dict) or m.get('role') not in ('system','developer','user','assistant','tool') for m in messages):
        raise ValueError('invalid chat message role')
    if 'model' in req and req['model'] not in models:raise ValueError('model not found')
    if 'stream' in req and type(req['stream']) is not bool:raise ValueError('stream must be boolean')
    for key in ('max_tokens','max_completion_tokens'):
        if key in req and (type(req[key]) is not int or not 0<=req[key]<=262144):raise ValueError('invalid output budget')
    if req.get('n',1)!=1:raise ValueError('only one completion is supported')
    for key,low,high in (('temperature',0,100),('top_p',0,1),('min_p',0,1),
                         ('repetition_penalty',0,100),('presence_penalty',-100,100),('frequency_penalty',-100,100)):
        value=req.get(key)
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or
                                  not math.isfinite(value) or not low<=value<=high):raise ValueError('invalid sampling value')
    for key in ('top_k','seed','penalty_last_n'):
        value=req.get(key)
        if value is not None and (type(value) is not int or not 0<=value<2**63):raise ValueError('invalid sampling integer')
    if req.get('strata_mcp') or 'strata_tune' in req:raise ValueError('server tools/tuning are unavailable to this credential')
    if 'chat_template_kwargs' in req and not isinstance(req['chat_template_kwargs'],dict):
        raise ValueError('chat_template_kwargs must be an object')
