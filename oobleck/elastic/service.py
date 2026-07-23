"""Public membership service with bounded broadcast failure handling."""

from __future__ import annotations

import asyncio

from oobleck.elastic.service_base import MasterControlService as _MasterControlService
from oobleck.elastic.service_public_base import (
    ControlStatus,
    NodeAgentClient,
    inspect_membership,
    inspect_status,
    request_drain,
)
from oobleck.elastic.transport import MessageEnvelope


class MasterControlService(_MasterControlService):
    async def _broadcast_message(self, message: MessageEnvelope) -> None:
        # A peer can close during a proposal or activation broadcast. That
        # failed writer is a detection input, not permission to stall the
        # membership state machine indefinitely.
        try:
            await asyncio.wait_for(super()._broadcast_message(message), timeout=1.0)
        except TimeoutError:
            return


__all__ = [
    "ControlStatus",
    "MasterControlService",
    "NodeAgentClient",
    "inspect_membership",
    "inspect_status",
    "request_drain",
]
