"""Which retrieved passages the model sees: ONE selection shared by merge and synthesis.

`RetrieveNode` selects per topic, `MergeSourcesNode` selects the merged list before it stores
it (it emits nothing); `SynthesizeNode` selects again (idempotent) and both emits `Sources` and
renders from that one selection, so the client's source list is exactly what the model saw and
citation number n always names the n-th source of both.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from agintis.agent_kit.nodes.state import SourceItem

# Sources kept, characters kept of each one, and characters of source text in one prompt
# (about 3k tokens).
MAX_SOURCES = 8
MAX_SOURCE_CHARS = 1500
MAX_CONTEXT_CHARS = 12000


@dataclass(frozen=True, slots=True)
class SourceLimits:
    max_sources: int = MAX_SOURCES
    max_source_chars: int = MAX_SOURCE_CHARS
    max_context_chars: int = MAX_CONTEXT_CHARS

    def __post_init__(self) -> None:
        if min(self.max_sources, self.max_source_chars, self.max_context_chars) < 1:
            raise ValueError("Source limits must be at least 1")


def select_sources(items: Sequence[SourceItem], limits: SourceLimits) -> list[SourceItem]:
    """At most `max_sources` passages in order, each cut to `max_source_chars`, together within
    `max_context_chars` (the passage that crosses the budget is cut, the rest dropped).
    Passages without text are skipped, so numbering follows the returned list."""
    selected: list[SourceItem] = []
    budget = limits.max_context_chars
    for item in items:
        if len(selected) >= limits.max_sources or budget <= 0:
            break
        text = item["text"][: min(limits.max_source_chars, budget)]
        if not text.strip():
            continue
        budget -= len(text)
        selected.append(SourceItem({**item, "text": text}))
    return selected
