import asyncio
import enum
import pickle
from dataclasses import dataclass
from typing import Any

TIMEOUT = 5


@dataclass
class DistributionInfo:
    agent_ips: list[str]
    world_size: int


class RequestType(enum.Enum):
    UNDEFINED = 0
    LAUNCH_JOB = 1
    GET_DIST_INFO = 2
    REGISTER_AGENT = 3
    PING = 4
    FORWARD_RANK0_PORT = 5


class Response(enum.Enum):
    SUCCESS = 1
    FAILURE = 2
    PONG = 3
    RECONFIGURATION = 4
    FORWARD_RANK0_PORT = 5


async def send(
    w: asyncio.StreamWriter,
    msg: Any,
    need_pickle: bool = True,
    drain: bool = True,
    close: bool = False,
):
    if need_pickle:
        msg = pickle.dumps(msg)

    w.write(len(msg).to_bytes(4, "little"))
    w.write(msg)

    if drain or close:
        await w.drain()

    if close:
        w.close()
        await w.wait_closed()


async def send_request_type(w: asyncio.StreamWriter, type: RequestType):
    w.write(type.value.to_bytes(1, "little"))
    await w.drain()


async def recv_request_type(r: asyncio.StreamReader) -> RequestType:
    return RequestType(int.from_bytes(await r.readexactly(1), "little"))


async def send_response(
    w: asyncio.StreamWriter,
    request: RequestType,
    response: Response,
    close: bool = True,
):
    w.write(response.value.to_bytes(1, "little"))
    w.write(request.value.to_bytes(1, "little"))
    await w.drain()
    if close:
        w.close()
        await w.wait_closed()


async def recv_response(
    r: asyncio.StreamReader, timeout=TIMEOUT
) -> tuple[Response, RequestType]:
    response = await asyncio.wait_for(r.readexactly(2), timeout=timeout)
    return Response(int.from_bytes(response[:1], "little")), RequestType(
        int.from_bytes(response[1:], "little")
    )


async def recv(
    r: asyncio.StreamReader, need_pickle: bool = True, timeout=TIMEOUT
) -> Any:
    len = int.from_bytes(
        await asyncio.wait_for(r.readexactly(4), timeout=timeout), "little"
    )
    msg = await asyncio.wait_for(r.readexactly(len), timeout=timeout)
    return pickle.loads(msg) if need_pickle else msg
