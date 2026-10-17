"""MergeSourcesNode: the sources of every routed topic as ONE numbered list.

Scores of different topics (profiles) are on different scales and not comparable, so topics are
merged by RANK, not by score, with a WEIGHTED round-robin (`interleave`), deterministic:
- groups in router order (primary topic first), the evidence kept from earlier rewrite attempts
  as the LAST group (lowest priority in each round);
- each round takes `weight` consecutive items of a group: the primary topic `primary_weight`
  (default 2), every other group 1. A plain 1:1 round-robin lets an irrelevant secondary topic
  push the right topic's hits down; the weight keeps the primary's best hits in front while a
  second topic still contributes;
- duplicates are dropped, first occurrence wins (chunk id; items without one by source, locator
  and text). Across topics this is DEFENSIVE: profiles are unique per agent spec, so two topics
  never return the same chunk; it matters for the kept previous evidence, which a new attempt
  can find again.
Then the usual `select_sources` limits cut the list.
"""

import logging
from collections.abc import Callable, Hashable, Sequence
from typing import Any

from langgraph.runtime import Runtime

from agintis.agent_kit.nodes.sources import SourceLimits, select_sources
from agintis.agent_kit.nodes.state import RagState, SourceItem
from agintis.agent_kit.runtime import AgentContext

logger = logging.getLogger(__name__)

# Items per round the primary topic contributes (the others contribute one).
DEFAULT_PRIMARY_WEIGHT = 2


def interleave[T](
    groups: Sequence[Sequence[T]],
    key: Callable[[T], Hashable],
    weights: Sequence[int] | None = None,
) -> list[T]:
    """Weighted round-robin merge of ranked lists (best first each); `weights[i]` = items of
    group i per round (default 1 each). The first occurrence of a `key` wins."""
    weight = list(weights) if weights is not None else [1] * len(groups)
    if len(weight) != len(groups) or any(w < 1 for w in weight):
        raise ValueError("weights need one positive integer per group")
    merged: list[T] = []
    seen: set[Hashable] = set()
    position = [0] * len(groups)
    while any(position[i] < len(group) for i, group in enumerate(groups)):
        for i, group in enumerate(groups):
            taken = group[position[i] : position[i] + weight[i]]
            position[i] += weight[i]
            for item in taken:
                if (item_key := key(item)) not in seen:
                    seen.add(item_key)
                    merged.append(item)
    return merged


def primary_weights(groups: int, primary_weight: int) -> list[int]:
    """`primary_weight` for the first group, 1 for the rest."""
    return [primary_weight] + [1] * (groups - 1) if groups else []


def source_key(item: SourceItem) -> Hashable:
    chunk_id = item.get("chunk_id")
    if chunk_id:
        return ("chunk", chunk_id)
    return ("text", item["source"], item["locator"], item["text"])


def merge_sources(
    results: Sequence[tuple[str, list[SourceItem]]],
    limits: SourceLimits,
    previous: Sequence[SourceItem] = (),
    primary_weight: int = DEFAULT_PRIMARY_WEIGHT,
) -> list[SourceItem]:
    """`results` = (topic, sources) pairs in router order (primary first); `previous` = evidence
    of earlier attempts, merged last."""
    groups = [sources for _, sources in results]
    weights = primary_weights(len(groups), primary_weight)
    if previous:
        groups.append(list(previous))
        weights.append(1)
    return select_sources(interleave(groups, source_key, weights), limits)


class MergeSourcesNode:
    """Writes `sources` (the merged, capped selection) and `previous_sources` (the same list, the
    evidence a rewrite attempt must not discard). Always rewrites `sources` (a thread keeps the
    previous turn's). The `Sources` event is NOT sent here: `SynthesizeNode` emits it from the
    list it numbers, because not every path to the answer goes through a merge.
    """

    def __init__(self, limits: SourceLimits, primary_weight: int = DEFAULT_PRIMARY_WEIGHT) -> None:
        if primary_weight < 1:
            raise ValueError("primary_weight must be at least 1")
        self._limits = limits
        self._primary_weight = primary_weight

    async def __call__(self, state: RagState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        by_topic = {r["topic"]: r["sources"] for r in state.get("topic_results") or []}
        order = [r["topic"] for r in state.get("routed_topics") or [] if r["topic"] in by_topic]
        order += [t for t in by_topic if t not in order]  # defensive: results without a route
        sources = merge_sources(
            [(t, by_topic[t]) for t in order],
            self._limits,
            state.get("previous_sources") or [],
            self._primary_weight,
        )
        return {"sources": sources, "previous_sources": sources}
