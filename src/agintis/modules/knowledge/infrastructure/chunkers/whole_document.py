"""Whole-document chunker (key "whole_document"): short documents stay in ONE chunk.

An FAQ answer or a notice loses meaning when cut by its sub-headings. The document is rendered
as `title`, then each section (`## sub-heading` when it has one) and, when all of it fits the
token budget, emitted as a single chunk. A longer document falls back to the structured
chunker (same params), which cuts at headings and paragraphs.
"""

import asyncio
from collections.abc import Mapping
from typing import Final

from agintis.modules.knowledge.application import ChunkDraft, Chunker, ParsedDocument, ParsedSection
from agintis.modules.knowledge.domain import MetadataValue, RagProfile
from agintis.modules.knowledge.infrastructure.chunkers.splitting import (
    MIN_CHUNK_MAX_TOKENS,
    chunk_budget,
)
from agintis.modules.knowledge.infrastructure.chunkers.structured import (
    STRUCTURED_CHUNKER_KEY,
    StructuredChunker,
    StructuredParams,
)
from agintis.modules.knowledge.infrastructure.chunkers.tokens import TokenCounter
from agintis.modules.knowledge.profiles import validate_params

WHOLE_DOCUMENT_CHUNKER_KEY: Final = "whole_document"
BLOCK_SEPARATOR: Final = "\n\n"


class WholeDocumentParams(StructuredParams):
    """Same knobs as the structured chunker it falls back to."""


class WholeDocumentChunker(Chunker):
    version = 1
    min_chunk_max_tokens = MIN_CHUNK_MAX_TOKENS
    params_model = WholeDocumentParams

    def __init__(self, counter: TokenCounter, fallback: StructuredChunker | None = None) -> None:
        self._counter = counter
        self._fallback = fallback if fallback is not None else StructuredChunker(counter)

    def dependency_versions(self) -> Mapping[str, int]:
        # A long document is cut by the fallback: its changes must re-index these profiles too.
        return {f"chunker:{STRUCTURED_CHUNKER_KEY}": self._fallback.version}

    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        budget = chunk_budget(profile.chunk_max_tokens)
        params = validate_params(
            WholeDocumentParams, profile.chunker_params, where="whole_document chunker params"
        )
        draft = await asyncio.to_thread(self._whole, parsed, params, budget)
        return [draft] if draft is not None else await self._fallback.chunk(parsed, profile)

    def _whole(
        self, parsed: ParsedDocument, params: WholeDocumentParams, budget: int
    ) -> ChunkDraft | None:
        if not parsed.sections:
            return None
        title = next((s.headings[0] for s in parsed.sections if s.headings), "")
        body = BLOCK_SEPARATOR.join(_render(section) for section in parsed.sections)
        text = f"{title}{BLOCK_SEPARATOR}{body}" if title and params.include_heading else body
        if self._counter.count(text) > budget:
            return None
        metadata: dict[str, MetadataValue] = {}
        pages = [s.page for s in parsed.sections if s.page is not None]
        if pages:
            metadata["page"], metadata["page_end"] = min(pages), max(pages)
        if title:
            metadata["heading"] = title
        return ChunkDraft(text=text, metadata=metadata, body=body)


def _render(section: ParsedSection) -> str:
    # The first heading is the document title (printed once above); deeper ones label sections.
    sub = section.headings[-1] if len(section.headings) > 1 else ""
    return f"## {sub}{BLOCK_SEPARATOR[:1]}{section.text}" if sub else section.text
