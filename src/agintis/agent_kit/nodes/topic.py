"""The topic subgraph: ONE compiled graph that every `Send` of the router runs.

    extract_filters -> retrieve

Its state is private (`TopicState`: the task's topic and query, the extracted filters); the task
(`TopicTask`) picks the topic's profile and extractor from the mappings given at build time, so a
single compiled subgraph serves all topics. The only thing it hands back to the parent graph is
a `topic_results` entry (`TopicOutput`): `output_schema` is what stops `query` and `filters`
from reaching the parent's channels of the same name (N parallel branches writing one
last-value channel is an error). It never emits `Sources`: the merge does.
"""

from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel

from agintis.agent_kit.nodes.extract_filters import ExtractFiltersNode
from agintis.agent_kit.nodes.filters import FilterExtractor
from agintis.agent_kit.nodes.retrieve import RetrieveNode, TopicRetrieval
from agintis.agent_kit.nodes.sources import SourceLimits
from agintis.agent_kit.nodes.state import TopicOutput, TopicState, TopicTask
from agintis.agent_kit.runtime import AgentContext
from agintis.modules.knowledge.api import DEFAULT_MAX_QUERY_CHARS, RagProfile

EXTRACT_FILTERS = "extract_filters"
RETRIEVE = "retrieve"


def build_topic_graph(
    retrieval: TopicRetrieval,
    profiles: Mapping[str, RagProfile],
    extractors: Mapping[str, FilterExtractor],
    limits: SourceLimits = SourceLimits(),  # noqa: B008 (frozen)
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
) -> Pregel[Any, Any, Any, Any]:
    """Compiled with `checkpointer=False`: a subgraph used as a node would otherwise INHERIT the
    parent's saver and write a checkpoint per branch (namespace `topic:<task id>`), useless here
    (the parent's checkpoint already holds `topic_results` and a failed branch simply re-runs)
    and pure write amplification. Consequence: no `interrupt()` inside this subgraph (the
    clarification policy lives in the parent, after `analyze`)."""
    builder = StateGraph(
        TopicState,
        context_schema=AgentContext,
        input_schema=TopicTask,
        output_schema=TopicOutput,
    )
    builder.add_node(EXTRACT_FILTERS, ExtractFiltersNode(extractors))
    builder.add_node(RETRIEVE, RetrieveNode(retrieval, profiles, limits, max_query_chars))
    builder.add_edge(START, EXTRACT_FILTERS)
    builder.add_edge(EXTRACT_FILTERS, RETRIEVE)
    builder.add_edge(RETRIEVE, END)
    return builder.compile(checkpointer=False)
