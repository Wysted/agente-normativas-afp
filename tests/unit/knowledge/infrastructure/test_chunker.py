import re
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from agintis.modules.knowledge.api import ParsedDocument, ParsedSection, RagProfile
from agintis.modules.knowledge.infrastructure.chunkers import (
    HfTokenCounter,
    StructuredChunker,
)
from agintis.modules.knowledge.infrastructure.chunkers.splitting import SPECIAL_TOKENS_RESERVE
from agintis.modules.knowledge.infrastructure.chunkers.tokens import enforce_budget


class WordCounter:
    """One token per whitespace-separated word."""

    def count(self, text: str) -> int:
        return len(text.split())

    def split(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


BUDGET = 20
PROFILE = RagProfile(
    "p",
    3,
    BUDGET + SPECIAL_TOKENS_RESERVE,
    "simple",
    "structured",
    chunker_params={"include_heading": False},
)
HEADED = RagProfile("p", 3, BUDGET + SPECIAL_TOKENS_RESERVE, "simple", "structured")


def _doc(*sections: ParsedSection) -> ParsedDocument:
    return ParsedDocument(title="t", sections=sections)


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


async def _chunk(*sections: ParsedSection) -> list[tuple[str, dict[str, object]]]:
    drafts = await StructuredChunker(WordCounter()).chunk(_doc(*sections), PROFILE)
    return [(d.text, dict(d.metadata)) for d in drafts]


async def test_small_paragraphs_are_merged_up_to_budget() -> None:
    text = "\n\n".join([_words(8, "a"), _words(8, "b"), _words(8, "c")])
    chunks = await _chunk(ParsedSection(text=text))
    assert [c[0].count("\n\n") for c in chunks] == [1, 0]
    assert all(WordCounter().count(c[0]) <= BUDGET for c in chunks)


async def test_never_crosses_section_boundaries_and_keeps_metadata() -> None:
    chunks = await _chunk(
        ParsedSection(text="short a", page=1),
        ParsedSection(text="short b", page=2, headings=("Intro", "Scope")),
    )
    assert chunks == [
        ("short a", {"page": 1, "page_end": 1}),
        ("short b", {"page": 2, "page_end": 2, "heading": "Intro > Scope"}),
    ]


async def test_long_paragraph_is_packed_by_sentences() -> None:
    sentences = [f"{_words(9, f's{i}_')}." for i in range(4)]
    chunks = await _chunk(ParsedSection(text=" ".join(sentences)))
    assert [c[0] for c in chunks] == [" ".join(sentences[:2]), " ".join(sentences[2:])]


async def test_oversized_sentence_is_cut_at_token_boundaries() -> None:
    chunks = await _chunk(ParsedSection(text=_words(45)))
    sizes = [WordCounter().count(c[0]) for c in chunks]
    assert sizes == [20, 20, 5]


async def test_every_chunk_respects_budget_for_mixed_content() -> None:
    text = "\n\n".join(
        [_words(5), _words(60, "long"), f"{_words(12)}. {_words(12)}. {_words(30)}.", _words(3)]
    )
    chunks = await _chunk(ParsedSection(text=text))
    assert chunks
    assert all(WordCounter().count(c[0]) <= BUDGET for c in chunks)
    original = re.sub(r"\s+", " ", text)
    rebuilt = re.sub(r"\s+", " ", " ".join(c[0] for c in chunks))
    assert rebuilt == original


async def test_empty_sections_yield_no_chunks() -> None:
    assert await _chunk(ParsedSection(text="  \n\n  ")) == []
    assert await _chunk() == []


async def test_profile_budget_must_exceed_special_tokens_reserve() -> None:
    tiny = RagProfile("p", 1, SPECIAL_TOKENS_RESERVE, "simple", "structured")
    with pytest.raises(ValueError, match="special tokens"):
        await StructuredChunker(WordCounter()).chunk(_doc(ParsedSection(text="x")), tiny)


class TestHfTokenCounter:
    @pytest.fixture
    def counter(self, tmp_path: Path) -> HfTokenCounter:
        vocab = {f"w{i}": i for i in range(100)} | {"[UNK]": 100}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))
        return HfTokenCounter(str(path))

    def test_counts_tokens(self, counter: HfTokenCounter) -> None:
        assert counter.count("w1 w2 w3") == 3
        assert counter.count("") == 0

    def test_split_cuts_on_token_boundaries(self, counter: HfTokenCounter) -> None:
        pieces = counter.split("w1 w2 w3 w4 w5", 2)
        assert pieces == ["w1 w2", "w3 w4", "w5"]
        assert all(counter.count(p) <= 2 for p in pieces)

    def test_split_rejects_non_positive_size(self, counter: HfTokenCounter) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            counter.split("w1", 0)

    async def test_chunker_with_real_tokenizer(self, counter: HfTokenCounter) -> None:
        drafts = await StructuredChunker(counter).chunk(
            _doc(ParsedSection(text=" ".join(f"w{i}" for i in range(50)))), PROFILE
        )
        assert [counter.count(d.text) for d in drafts] == [20, 20, 10]


class TestIncludeHeading:
    """D4: the heading path is part of the embedded/stored text and of the token budget."""

    @staticmethod
    async def _chunk(
        section: ParsedSection, profile: RagProfile = HEADED
    ) -> list[tuple[str, dict[str, object]]]:
        drafts = await StructuredChunker(WordCounter()).chunk(_doc(section), profile)
        return [(d.text, dict(d.metadata)) for d in drafts]

    async def test_heading_is_prepended_by_default_and_metadata_is_kept(self) -> None:
        chunks = await self._chunk(ParsedSection(text="body text", headings=("Intro", "Scope")))
        assert chunks == [("Intro > Scope\n\nbody text", {"heading": "Intro > Scope"})]

    async def test_heading_tokens_count_against_the_budget(self) -> None:
        section = ParsedSection(
            text=_words(60, "b"), headings=("Alpha", "Beta")
        )  # 3 heading tokens
        chunks = await self._chunk(section)
        assert len(chunks) > 3
        assert all(text.startswith("Alpha > Beta\n\n") for text, _ in chunks)
        assert all(WordCounter().count(text) <= BUDGET for text, _ in chunks)

    async def test_param_off_keeps_the_body_only(self) -> None:
        chunks = await self._chunk(ParsedSection(text="body", headings=("H",)), PROFILE)
        assert chunks == [("body", {"heading": "H"})]

    async def test_sections_without_headings_are_unchanged(self) -> None:
        assert await self._chunk(ParsedSection(text="plain", page=2)) == [
            ("plain", {"page": 2, "page_end": 2})
        ]

    async def test_overlong_heading_is_truncated_to_half_the_budget(self) -> None:
        section = ParsedSection(text=_words(30, "b"), headings=(_words(50, "h"),))
        chunks = await self._chunk(section)
        assert all(WordCounter().count(text) <= BUDGET for text, _ in chunks)
        assert all(text.startswith("h0 h1") for text, _ in chunks)

    async def test_the_param_changes_the_profile_fingerprint(self) -> None:
        assert HEADED.fingerprint != PROFILE.fingerprint

    async def test_unknown_params_fail_loudly(self) -> None:
        bad = RagProfile("p", 1, 100, "simple", "structured", chunker_params={"overlap": 10})
        with pytest.raises(ValueError, match="overlap"):
            await StructuredChunker(WordCounter()).chunk(_doc(ParsedSection(text="x")), bad)

    async def test_string_param_is_rejected(self) -> None:
        bad = RagProfile(
            "p", 1, 100, "simple", "structured", chunker_params={"include_heading": "no"}
        )
        with pytest.raises(ValueError, match="include_heading"):
            await StructuredChunker(WordCounter()).chunk(_doc(ParsedSection(text="x")), bad)

    async def test_budget_holds_when_the_tokenizer_merges_across_the_boundary(self) -> None:
        class MergingCounter(WordCounter):
            """Superadditive: joining the heading to a body costs one extra token."""

            def count(self, text: str) -> int:
                return len(text.split()) + (1 if "\n\n" in text and not text.startswith("w") else 0)

        section = ParsedSection(text=_words(40), headings=("Head",))
        drafts = await StructuredChunker(MergingCounter()).chunk(_doc(section), HEADED)
        assert all(MergingCounter().count(d.text) <= BUDGET for d in drafts)

    async def test_an_overflowing_chunk_is_resplit_alone(self) -> None:
        class PrefixPenaltyCounter(WordCounter):
            """Only the composed chunk (heading + body) containing BAD costs extra tokens."""

            def count(self, text: str) -> int:
                penalty = 9 if text.startswith("H") and "BAD" in text else 0
                return len(text.split()) + penalty

        a, b, c, d = _words(8, "a"), _words(8, "b"), "BAD " + _words(7, "c"), _words(8, "d")
        section = ParsedSection(text="\n\n".join([a, b, c, d]), headings=("H",))
        drafts = await StructuredChunker(PrefixPenaltyCounter()).chunk(_doc(section), HEADED)
        # The untouched chunk keeps both paragraphs; only the offender is split in two.
        assert [x.text for x in drafts] == [
            f"H\n\n{a}\n\n{b}",
            f"H\n\n{c}",
            f"H\n\n{d}",
        ]
        assert all(PrefixPenaltyCounter().count(x.text) <= BUDGET for x in drafts)


class TestCrossPages:
    CROSSING = RagProfile(
        "p",
        3,
        BUDGET + SPECIAL_TOKENS_RESERVE,
        "simple",
        "structured",
        chunker_params={"include_heading": False, "cross_pages": True},
    )

    @staticmethod
    async def _chunk(
        profile: RagProfile, *sections: ParsedSection
    ) -> list[tuple[str, dict[str, object]]]:
        drafts = await StructuredChunker(WordCounter()).chunk(_doc(*sections), profile)
        return [(d.text, dict(d.metadata)) for d in drafts]

    def _pages(self) -> tuple[ParsedSection, ...]:
        return tuple(
            ParsedSection(text=_words(8, f"p{n}_"), page=n, headings=("H",)) for n in (1, 2, 3)
        )

    async def test_default_never_crosses_pages_and_page_end_equals_page(self) -> None:
        chunks = await _chunk(*self._pages())
        assert [c[1] for c in chunks] == [
            {"page": 1, "page_end": 1, "heading": "H"},
            {"page": 2, "page_end": 2, "heading": "H"},
            {"page": 3, "page_end": 3, "heading": "H"},
        ]

    async def test_small_pages_merge_and_report_their_page_range(self) -> None:
        chunks = await self._chunk(self.CROSSING, *self._pages())
        # 8 + 8 words fit the 20-word budget; the third page starts a new chunk.
        assert [c[1] for c in chunks] == [
            {"page": 1, "page_end": 2, "heading": "H"},
            {"page": 3, "page_end": 3, "heading": "H"},
        ]
        assert chunks[0][0].count("\n\n") == 1

    async def test_each_chunk_gets_the_pages_of_its_own_text(self) -> None:
        big = ParsedSection(text="\n\n".join(_words(8, f"a{i}_") for i in range(3)), page=1)
        nxt = ParsedSection(text=_words(8, "b_"), page=2)
        chunks = await self._chunk(self.CROSSING, big, nxt)
        pages = [(c[1]["page"], c[1]["page_end"]) for c in chunks]
        assert pages == [(1, 1), (1, 2)]
        assert "b_0" in chunks[-1][0]
        assert "a2_0" in chunks[-1][0]

    async def test_oversize_paragraph_pieces_keep_the_paragraph_page(self) -> None:
        long_page = ParsedSection(text=_words(45, "x"), page=4)
        first = ParsedSection(text="short", page=3)
        chunks = await self._chunk(self.CROSSING, first, long_page)
        assert {(c[1]["page"], c[1]["page_end"]) for c in chunks[1:]} == {(4, 4)}

    async def test_never_crosses_a_heading_boundary(self) -> None:
        a = ParsedSection(text=_words(5, "a"), page=1, headings=("A",))
        b = ParsedSection(text=_words(5, "b"), page=2, headings=("B",))
        chunks = await self._chunk(self.CROSSING, a, b)
        assert [(c[1]["page"], c[1]["page_end"], c[1]["heading"]) for c in chunks] == [
            (1, 1, "A"),
            (2, 2, "B"),
        ]

    async def test_pageless_sections_get_no_page_metadata(self) -> None:
        chunks = await self._chunk(
            self.CROSSING, ParsedSection(text="one", headings=("H",)), ParsedSection(text="two")
        )
        assert all("page" not in c[1] and "page_end" not in c[1] for c in chunks)

    async def test_heading_prefix_still_applies_and_budget_holds(self) -> None:
        profile = RagProfile(
            "p",
            3,
            BUDGET + SPECIAL_TOKENS_RESERVE,
            "simple",
            "structured",
            chunker_params={"cross_pages": True},
        )
        chunks = await self._chunk(profile, *self._pages())
        assert all(text.startswith("H\n\n") for text, _ in chunks)
        assert all(WordCounter().count(text) <= BUDGET for text, _ in chunks)
        assert chunks[0][1]["page"] == 1
        assert chunks[-1][1]["page_end"] == 3

    async def test_the_param_changes_the_fingerprint_and_must_be_boolean(self) -> None:
        assert self.CROSSING.fingerprint != PROFILE.fingerprint
        bad = RagProfile("p", 1, 100, "simple", "structured", chunker_params={"cross_pages": "yes"})
        with pytest.raises(ValueError, match="cross_pages"):
            await StructuredChunker(WordCounter()).chunk(_doc(ParsedSection(text="x")), bad)


class TestEnforceBudget:
    """S1: a split piece that re-encodes above the budget is shrunk, backing off to whitespace."""

    @staticmethod
    def _superadditive(text: str) -> int:
        words = text.split()
        return len(words) + sum(1 for w in words if w.endswith(","))  # commas cost one more

    def test_fitting_text_is_returned_whole(self) -> None:
        assert enforce_budget("a b c", 5, lambda t: len(t.split())) == ["a b c"]

    def test_oversized_pieces_are_split_on_whitespace(self) -> None:
        text = "a, b, c, d, e, f"
        pieces = enforce_budget(text, 4, self._superadditive)
        assert all(self._superadditive(p) <= 4 for p in pieces)
        assert " ".join(pieces) == text
        assert all(not p.startswith(",") and p == p.strip() for p in pieces)

    def test_words_are_not_cut_when_whitespace_exists(self) -> None:
        pieces = enforce_budget("alpha beta gamma delta", 2, lambda t: len(t.split()))
        assert pieces == ["alpha beta", "gamma delta"]

    def test_single_unbreakable_overlong_word_is_cut_by_characters(self) -> None:
        pieces = enforce_budget("x" * 10, 3, lambda t: -(-len(t) // 4))  # 4 chars per token
        assert "".join(pieces) == "x" * 10
        assert all(len(p) <= 12 for p in pieces)

    def test_empty_text(self) -> None:
        assert enforce_budget("  ", 3, len) == []
