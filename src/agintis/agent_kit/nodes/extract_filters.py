"""ExtractFiltersNode: turns the topic's query into metadata filters for the topic."""

from collections.abc import Mapping
from typing import Any

from langgraph.runtime import Runtime

from agintis.agent_kit.nodes.filters import FilterExtractor
from agintis.agent_kit.nodes.state import TopicState
from agintis.agent_kit.runtime import AgentContext


class ExtractFiltersNode:
    """Delegates to the `FilterExtractor` of the task's topic (each built for ITS profile).

    One node serves every topic of the shared topic subgraph: the `TopicTask` carries the topic
    name. Always writes `filters` (an empty list when nothing was extracted or the profile has no
    filterable field, in which case no model is called). The state is private to the subgraph run,
    so nothing can leak between topics or turns.
    """

    def __init__(self, extractors: Mapping[str, FilterExtractor]) -> None:
        self._extractors = dict(extractors)

    async def __call__(self, state: TopicState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        return {"filters": await self._extractors[state["topic"]].extract(state["query"])}
