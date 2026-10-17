"""Typed per-request data handed to every node through `Runtime.context`.

Per-request data travels here (`astream(context=...)`), never in `config["configurable"]`:
it is typed, validated by the graph's `context_schema`, and not persisted with checkpoints.
Graphs built for `LangGraphRunner` declare `StateGraph(State, context_schema=AgentContext)`.
"""

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agintis.modules.agents_catalog.api import RunInput
from agintis.shared_kernel.ids import RunId, TenantId

# Query embeddings a run keeps: one per routed attempt (the original query plus its rewrites).
MAX_QUERY_VECTORS = 8


class QueryVectors:
    """Per-run, in-memory memo of query embeddings (text -> vector), bounded (oldest dropped).

    A router that already embedded the query (the evidence router probes the corpus with it)
    stores the vector here and the retrieval of the same attempt reads it instead of embedding
    again. It lives in the run's context, never in the graph state: a 1024-float list per attempt
    would be written to every checkpoint of a thread, and losing it only costs one embedding.
    """

    def __init__(self, max_entries: int = MAX_QUERY_VECTORS) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._vectors: OrderedDict[str, Sequence[float]] = OrderedDict()

    def get(self, text: str) -> Sequence[float] | None:
        return self._vectors.get(text)

    def put(self, text: str, vector: Sequence[float]) -> None:
        self._vectors[text] = vector
        self._vectors.move_to_end(text)
        while len(self._vectors) > self._max_entries:
            self._vectors.popitem(last=False)

    def __len__(self) -> int:
        return len(self._vectors)


@dataclass(frozen=True, slots=True)
class AgentContext:
    tenant_id: TenantId
    run_id: RunId
    # Fresh per context (= per run). Mutable on purpose: it is a memo, not identity.
    query_vectors: QueryVectors = field(default_factory=QueryVectors, compare=False, repr=False)


# Strategy: builds the context object of a run (a graph may need a richer subclass).
ContextFactory = Callable[[RunInput], Any]


def agent_context(run_input: RunInput) -> AgentContext:
    """Default `ContextFactory`."""
    return AgentContext(tenant_id=run_input.tenant_id, run_id=run_input.run_id)
