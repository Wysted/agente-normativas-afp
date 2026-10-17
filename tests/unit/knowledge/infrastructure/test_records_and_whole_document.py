"""Records chunker (index-like documents) and whole-document chunker (short documents)."""

from typing import Any

import pytest

from agintis.modules.knowledge.api import (
    ChunkDraft,
    InvalidProfileError,
    ParsedDocument,
    ParsedSection,
    RagProfile,
)
from agintis.modules.knowledge.infrastructure.chunkers import (
    RecordsChunker,
    RecordsParams,
    StructuredChunker,
    WholeDocumentChunker,
)
from agintis.modules.knowledge.infrastructure.chunkers.splitting import SPECIAL_TOKENS_RESERVE
from agintis.modules.knowledge.profiles import validate_params
from tests.unit.knowledge.infrastructure.test_chunker import WordCounter

REF = r"(?:Of\.\s*Ord\.|NI)\s*N[°º]"
RECORDS: dict[str, Any] = {
    "group_levels": [
        {
            "name": "materia",
            "pattern": r"^(?P<materia>[IVX]+)\.\s+(?P<materia_nombre>[A-Z][A-Z ]+)$",
        },
        {"name": "rama", "pattern": r"^(?P<rama>[A-D])\.\s+(?P<rama_nombre>[A-Z][A-Z ]+)$"},
        {
            "name": "tema",
            "pattern": (r"^(?P<tema>\d{1,2})\.?\s+(?P<tema_nombre>[^\d].*?)\s*(?=" + REF + r"|$)"),
        },
    ],
    "record_pattern": (
        r"^(?P<topic_num>\d{1,2}(?:\.\d{1,2}){1,2})\.?\s+(?P<topic>.+?)\s*(?P<refs>(?:"
        + REF
        + ".*)?)$"
    ),
    "group_by": "tema",
    "header_groups": ["materia_nombre", "rama_nombre", "tema_nombre"],
    "repeat_group_header": True,
}

INDEX = """\
I. MATERIA ONE
A. RAMA A
7. Tema siete Of. Ord. N° 100 de 01-02-03
7.1. Sub one Of. Ord. N° 101 de 01-02-03
7.2. Sub two NI N° 5 de 02-03-04
8. Tema ocho
8.1. Sub eight Of. Ord. N° 9 de 01-01-01
II. MATERIA TWO
9. Tema nueve
9.1. Sub nine Of. Ord. N° 1 de 01-01-01"""


def _profile(budget: int = 60, **params: Any) -> RagProfile:
    return RagProfile(
        "p",
        3,
        budget + SPECIAL_TOKENS_RESERVE,
        "simple",
        "records",
        chunker_params={**RECORDS, **params},
    )


async def _chunk(
    text: str, profile: RagProfile, *, pages: tuple[str, ...] = ()
) -> list[ChunkDraft]:
    sections = (
        tuple(ParsedSection(t, page=i) for i, t in enumerate(pages, start=1))
        if pages
        else (ParsedSection(text),)
    )
    return await RecordsChunker(WordCounter()).chunk(ParsedDocument("t", sections), profile)


class TestRecords:
    async def test_one_chunk_per_group_with_the_context_header(self) -> None:
        drafts = await _chunk(INDEX, _profile())
        assert [d.text.split("\n")[0] for d in drafts] == [
            "MATERIA ONE > RAMA A > Tema siete",
            "MATERIA ONE > RAMA A > Tema ocho",
            "MATERIA TWO > Tema nueve",
        ]
        first = drafts[0]
        assert first.body == (
            "7. Tema siete Of. Ord. N° 100 de 01-02-03\n"
            "7.1. Sub one Of. Ord. N° 101 de 01-02-03\n"
            "7.2. Sub two NI N° 5 de 02-03-04"
        )
        assert first.metadata["heading"] == "MATERIA ONE > RAMA A > Tema siete"

    async def test_context_groups_are_recorded_and_deeper_levels_reset(self) -> None:
        drafts = await _chunk(INDEX, _profile())
        assert drafts[0].groups["materia_nombre"] == ("MATERIA ONE",)
        assert drafts[0].groups["rama_nombre"] == ("RAMA A",)
        assert drafts[0].groups["tema_nombre"] == ("Tema siete",)
        # A new materia drops the rama of the previous one.
        assert "rama_nombre" not in drafts[2].groups
        assert drafts[2].groups["materia_nombre"] == ("MATERIA TWO",)

    async def test_the_header_can_be_left_out_of_the_text(self) -> None:
        drafts = await _chunk(INDEX, _profile(repeat_group_header=False))
        assert drafts[0].text == drafts[0].body
        assert drafts[0].metadata["heading"] == "MATERIA ONE > RAMA A > Tema siete"

    async def test_a_big_group_is_split_and_every_piece_repeats_the_header(self) -> None:
        drafts = await _chunk(INDEX, _profile(budget=20))
        header = "MATERIA ONE > RAMA A > Tema siete"
        pieces = [d for d in drafts if d.text.startswith(header)]
        assert len(pieces) == 3
        assert all(WordCounter().count(d.text) <= 20 for d in drafts)
        assert [d.body.split("\n")[0].split()[0] for d in pieces if d.body] == [
            "7.",
            "7.1.",
            "7.2.",
        ]

    async def test_a_line_over_budget_is_cut(self) -> None:
        long_line = "7.3. " + " ".join(f"word{i}" for i in range(50))
        text = "A. RAMA\n7. Tema siete\n" + long_line
        drafts = await _chunk(text, _profile(budget=15))
        assert len(drafts) > 3
        assert all(WordCounter().count(d.text) <= 15 for d in drafts)

    async def test_records_before_any_group_are_kept(self) -> None:
        drafts = await _chunk("7.1. Orphan Of. Ord. N° 1 de 01-01-01\n7. Tema siete", _profile())
        assert drafts[0].body == "7.1. Orphan Of. Ord. N° 1 de 01-01-01"

    async def test_continuation_lines_join_the_current_group(self) -> None:
        drafts = await _chunk("7. Tema siete\ncontinuation text\n7.1. Sub", _profile())
        assert drafts[0].body == "7. Tema siete\ncontinuation text\n7.1. Sub"

    async def test_pages_come_from_the_line_offsets(self) -> None:
        drafts = await _chunk(
            "", _profile(), pages=("7. Tema siete\n7.1. Sub one", "7.2. Sub two\n8. Tema ocho")
        )
        assert (drafts[0].metadata["page"], drafts[0].metadata["page_end"]) == (1, 2)
        assert (drafts[1].metadata["page"], drafts[1].metadata["page_end"]) == (2, 2)

    async def test_the_preamble_before_the_first_group_is_kept_by_default(self) -> None:
        text = "Introduction line one\nintroduction line two\n" + INDEX
        drafts = await _chunk(text, _profile(preamble_heading="Intro"))
        assert drafts[0].text == "Intro\nIntroduction line one\nintroduction line two"
        assert drafts[0].groups == {}
        assert drafts[0].metadata["heading"] == "Intro"
        assert len(drafts) == 4  # the three groups follow

    async def test_the_preamble_without_a_heading_is_plain_text(self) -> None:
        drafts = await _chunk("Just an introduction\n" + INDEX, _profile())
        assert drafts[0].text == "Just an introduction"
        assert "heading" not in drafts[0].metadata

    async def test_the_preamble_can_be_dropped(self) -> None:
        drafts = await _chunk("Intro text\n7.1. Orphan\n" + INDEX, _profile(preamble="drop"))
        assert [d.text.split("\n")[0] for d in drafts][:1] == ["MATERIA ONE > RAMA A > Tema siete"]
        assert not any("Intro text" in d.text or "Orphan" in d.text for d in drafts)

    async def test_records_and_text_before_the_first_group_share_the_preamble(self) -> None:
        drafts = await _chunk("Intro\n7.1. Orphan Of. Ord. N° 1 de 01-01-01\n7. Tema", _profile())
        assert drafts[0].body == "Intro\n7.1. Orphan Of. Ord. N° 1 de 01-01-01"

    async def test_an_oversize_header_is_capped_so_the_lines_always_fit(self) -> None:
        words = " ".join(f"h{i}" for i in range(30))
        text = f"TEMA {words}\n1. a b c d e f\n2. g h i"
        params: dict[str, Any] = {
            "group_levels": [{"name": "tema", "pattern": r"^TEMA (?P<tema>.+)$"}],
            "record_pattern": r"^\d+\.\s",
            "group_by": "tema",
            "header_groups": ["tema"],
        }
        profile = RagProfile(
            "r", 3, 20 + SPECIAL_TOKENS_RESERVE, "simple", "records", chunker_params=params
        )
        drafts = await RecordsChunker(WordCounter()).chunk(
            ParsedDocument("T", (ParsedSection(text),)), profile
        )
        assert len(drafts) > 1
        assert all(WordCounter().count(d.text) <= 20 for d in drafts)
        assert all(WordCounter().count(d.text.split("\n")[0]) <= 10 for d in drafts)
        assert drafts[0].metadata["heading"] == words  # the heading metadata keeps it whole

    def test_params_are_validated(self) -> None:
        def check(**overrides: Any) -> RecordsParams:
            return validate_params(RecordsParams, {**RECORDS, **overrides}, where="records")

        assert check().group_names() >= {"materia_nombre", "topic_num", "refs"}
        for overrides, message in (
            ({"group_by": "nope"}, "group_by"),
            ({"header_groups": ["nope"]}, "unknown regex groups"),
            ({"record_pattern": "(a+)+"}, "nested quantifier"),
            ({"group_levels": []}, "at least 1 item"),
            ({"unknown": 1}, "Extra inputs"),
        ):
            with pytest.raises(InvalidProfileError, match=message):
                check(**overrides)


def _whole_profile(budget: int, **params: Any) -> RagProfile:
    return RagProfile(
        "p", 3, budget + SPECIAL_TOKENS_RESERVE, "simple", "whole_document", chunker_params=params
    )


def _article(*bodies: tuple[tuple[str, ...], str, int | None]) -> ParsedDocument:
    return ParsedDocument(
        "Title", tuple(ParsedSection(text, page, headings) for headings, text, page in bodies)
    )


class TestWholeDocument:
    def _chunker(self) -> WholeDocumentChunker:
        counter = WordCounter()
        return WholeDocumentChunker(counter, StructuredChunker(counter))

    async def test_a_short_document_is_one_chunk_with_its_sections(self) -> None:
        doc = _article(
            (("Q?",), "The short answer.", None),
            (("Q?", "Detail"), "More detail.", None),
        )
        (draft,) = await self._chunker().chunk(doc, _whole_profile(50))
        assert draft.text == "Q?\n\nThe short answer.\n\n## Detail\nMore detail."
        assert draft.body == "The short answer.\n\n## Detail\nMore detail."
        assert draft.metadata == {"heading": "Q?"}

    async def test_pages_are_the_hull_of_its_sections(self) -> None:
        doc = _article((("Q?",), "a", 2), (("Q?", "S"), "b", 3))
        (draft,) = await self._chunker().chunk(doc, _whole_profile(50))
        assert (draft.metadata["page"], draft.metadata["page_end"]) == (2, 3)

    async def test_without_include_heading_the_title_is_not_prepended(self) -> None:
        doc = _article((("Q?",), "The answer.", None))
        (draft,) = await self._chunker().chunk(doc, _whole_profile(50, include_heading=False))
        assert draft.text == "The answer."

    async def test_a_long_document_falls_back_to_the_structured_chunker(self) -> None:
        long_text = " ".join(f"w{i}" for i in range(40))
        doc = _article((("Guide",), "intro", None), (("Guide", "Topic"), long_text, None))
        drafts = await self._chunker().chunk(doc, _whole_profile(30))
        assert len(drafts) > 1
        assert drafts[0].text == "Guide\n\nintro"
        assert all(WordCounter().count(d.text) <= 30 for d in drafts)

    def test_it_reports_the_structured_version_it_falls_back_to(self) -> None:
        versions = self._chunker().dependency_versions()
        assert versions == {"chunker:structured": StructuredChunker.version}

    def test_a_structured_version_bump_changes_the_profile_fingerprint(self) -> None:
        profile = _whole_profile(40)
        now = profile.fingerprint_for(
            {"chunker:whole_document": 1, **self._chunker().dependency_versions()}
        )
        bumped = profile.fingerprint_for({"chunker:whole_document": 1, "chunker:structured": 99})
        assert now != bumped

    async def test_an_empty_document_has_no_chunks(self) -> None:
        assert await self._chunker().chunk(ParsedDocument("t", ()), _whole_profile(50)) == []

    async def test_params_are_those_of_the_structured_chunker(self) -> None:
        with pytest.raises(InvalidProfileError, match="Extra inputs"):
            await self._chunker().chunk(_article((("Q",), "a", None)), _whole_profile(50, nope=1))
