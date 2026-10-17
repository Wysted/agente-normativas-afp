from dataclasses import dataclass

from agintis.modules.agents_catalog.application.registry import AgentRegistry
from agintis.modules.agents_catalog.application.run_agent import RunAgent


@dataclass(frozen=True, slots=True)
class AgentsCatalogServices:
    """What the HTTP routes need, built by the composition root."""

    registry: AgentRegistry
    run_agent: RunAgent
    max_message_chars: int
