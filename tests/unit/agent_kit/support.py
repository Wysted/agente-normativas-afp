"""Helpers shared by the node tests."""

from typing import Any

from langgraph.runtime import Runtime

from agintis.agent_kit.runtime import AgentContext
from agintis.shared_kernel.ids import RunId, TenantId

TENANT = TenantId("acme")


def runtime_for(
    tenant: TenantId = TENANT, events: list[Any] | None = None
) -> Runtime[AgentContext]:
    """A node `Runtime` whose `emit()` events are appended to `events`."""
    sink = events if events is not None else []
    return Runtime(context=AgentContext(tenant, RunId("run-1")), stream_writer=sink.append)
