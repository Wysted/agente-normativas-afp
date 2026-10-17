"""Parser "simple_index_card": text documents as usual, scanned ones as their index card."""

from typing import Any

import pytest

from agintis.modules.knowledge.api import ScannedDocumentError, SourceFile
from agintis.modules.knowledge.infrastructure.parsers import IndexCardParser, SimpleParser
from agintis.modules.knowledge.infrastructure.registry import build_parsers
from tests.support.documents import make_blank_pdf, make_pdf

CARD = {"title": " Resolución N° 1266 ", "description": "Complementa la individualización."}


def _pdf(content: bytes, **meta: Any) -> SourceFile:
    return SourceFile("u.pdf", "application/pdf", content, document_metadata=meta)


class TestIndexCardParser:
    async def test_a_scanned_pdf_becomes_its_card(self) -> None:
        parsed = await IndexCardParser().parse(_pdf(make_blank_pdf(3), **CARD))
        assert parsed.title == "Resolución N° 1266"
        assert [(s.text, s.page) for s in parsed.sections] == [
            ("Resolución N° 1266\n\nComplementa la individualización.", None)
        ]
        assert parsed.collected["content_mode"] == ("index_card",)

    async def test_the_description_is_optional(self) -> None:
        parsed = await IndexCardParser().parse(_pdf(make_blank_pdf(1), title="Only a title"))
        assert parsed.sections[0].text == "Only a title"

    async def test_a_scanned_pdf_without_a_title_still_fails(self) -> None:
        for meta in ({}, {"title": "   "}, {"title": 5}):
            with pytest.raises(ScannedDocumentError):
                await IndexCardParser().parse(_pdf(make_blank_pdf(1), **meta))

    async def test_a_pdf_with_text_is_parsed_normally(self) -> None:
        content = make_pdf(["Page one has plenty of text.", "Page two as well."])
        parsed = await IndexCardParser().parse(_pdf(content, **CARD))
        plain = await SimpleParser().parse(_pdf(content))
        assert parsed.sections == plain.sections
        assert parsed.collected["content_mode"] == ("full_text",)

    async def test_the_plain_parser_still_refuses_scanned_pdfs(self) -> None:
        with pytest.raises(ScannedDocumentError):
            await SimpleParser().parse(_pdf(make_blank_pdf(2), **CARD))

    async def test_text_sources_get_the_content_mode_too(self) -> None:
        parsed = await IndexCardParser().parse(SourceFile("a.txt", "text/plain", b"hello"))
        assert parsed.collected["content_mode"] == ("full_text",)

    def test_it_follows_the_simple_parser_version(self) -> None:
        assert IndexCardParser.version == SimpleParser.version

    def test_both_parsers_are_registered(self) -> None:
        parsers = build_parsers(
            max_bytes=10**6, max_pages=10, max_uncompressed_bytes=10**6, max_compression_ratio=10.0
        )
        assert isinstance(parsers["simple_index_card"], IndexCardParser)
        assert type(parsers["simple"]) is SimpleParser
