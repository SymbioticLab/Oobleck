from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HostInfo(_message.Message):
    __slots__ = ("ip", "devices", "port", "status")
    IP_FIELD_NUMBER: _ClassVar[int]
    DEVICES_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ip: str
    devices: str
    port: int
    status: str
    def __init__(self, ip: _Optional[str] = ..., devices: _Optional[str] = ..., port: _Optional[int] = ..., status: _Optional[str] = ...) -> None: ...

class DistInfo(_message.Message):
    __slots__ = ("hosts",)
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedCompositeFieldContainer[HostInfo]
    def __init__(self, hosts: _Optional[_Iterable[_Union[HostInfo, _Mapping]]] = ...) -> None: ...

class CodeInfo(_message.Message):
    __slots__ = ("path", "args")
    PATH_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    path: str
    args: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, path: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ...) -> None: ...

class PortInfo(_message.Message):
    __slots__ = ("port",)
    PORT_FIELD_NUMBER: _ClassVar[int]
    port: int
    def __init__(self, port: _Optional[int] = ...) -> None: ...

class AgentInfo(_message.Message):
    __slots__ = ("agent_index",)
    AGENT_INDEX_FIELD_NUMBER: _ClassVar[int]
    agent_index: int
    def __init__(self, agent_index: _Optional[int] = ...) -> None: ...
