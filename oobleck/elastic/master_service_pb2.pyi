from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HostInfo(_message.Message):
    __slots__ = ("ip", "slots", "port")
    IP_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    ip: str
    slots: int
    port: int
    def __init__(self, ip: _Optional[str] = ..., slots: _Optional[int] = ..., port: _Optional[int] = ...) -> None: ...

class DistInfo(_message.Message):
    __slots__ = ("hosts",)
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedCompositeFieldContainer[HostInfo]
    def __init__(self, hosts: _Optional[_Iterable[_Union[HostInfo, _Mapping]]] = ...) -> None: ...

class CodeInfo(_message.Message):
    __slots__ = ("code",)
    CODE_FIELD_NUMBER: _ClassVar[int]
    code: bytes
    def __init__(self, code: _Optional[bytes] = ...) -> None: ...

class PortInfo(_message.Message):
    __slots__ = ("port",)
    PORT_FIELD_NUMBER: _ClassVar[int]
    port: int
    def __init__(self, port: _Optional[int] = ...) -> None: ...
