"""Heading templates and the budget splitter shared by the structural chunkers."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from agintis.modules.knowledge.infrastructure.chunkers import HfTokenCounter
from agintis.modules.knowledge.infrastructure.chunkers.splitting import BudgetSplitter
from agintis.modules.knowledge.infrastructure.templates import (
    check_template_syntax,
    render_template,
    template_names,
)
from tests.unit.knowledge.infrastructure.test_chunker import WordCounter


class TestTemplates:
    @pytest.mark.parametrize(
        ("template", "values", "expected"),
        [
            ("{a}", {"a": "x"}, "x"),
            ("{a} - {b}", {"a": "x"}, "x -"),  # missing values vanish, spaces collapse
            ("{a:5}", {"a": "abcdefgh"}, "abcd…"),
            ("{a:5}", {"a": "abcde"}, "abcde"),
            ("T{a}{b?, L{b}}", {"a": "I", "b": "A"}, "TI, LA"),
            ("T{a}{b?, L{b}}", {"a": "I"}, "TI"),
            ("T{a}{b?, L{b}}", {"a": "I", "b": ""}, "TI"),
            ("{v?{v} }{h:6}", {"v": "add", "h": "head text"}, "add head…"),
            ("{n}.", {"n": 3}, "3."),
            ("{zero?kept}", {"zero": 0}, "kept"),  # 0 is a value; only None and "" are empty
            ("{n?N {n}}", {"n": 0}, "N 0"),
            ("{x:1}", {"x": "abc"}, "…"),
            ("no placeholders", {}, "no placeholders"),
            ("{a}   {b}", {"a": "x", "b": "y"}, "x y"),
        ],
    )
    def test_render(self, template: str, values: dict[str, object], expected: str) -> None:
        assert render_template(template, values) == expected

    def test_values_are_never_scanned_for_placeholders_again(self) -> None:
        values = {"title": "{secret}", "secret": "LEAKED", "n": "{title}"}
        assert render_template("{title?Doc: {title}}", values) == "Doc: {secret}"
        assert render_template("{title} {n}", values) == "{secret} {title}"
        assert render_template("{n?{n:4}}", values) == "{ti…"

    def test_a_zero_limit_is_not_a_placeholder(self) -> None:
        assert render_template("{title:0}", {"title": "abcdef"}) == "{title:0}"
        with pytest.raises(ValueError, match="malformed placeholder"):
            check_template_syntax("{title:0}")

    @pytest.mark.parametrize("template", ["{a", "a}", "{a?{b?c}}", "{}", "{a b}"])
    def test_malformed_templates_are_rejected_at_boot(self, template: str) -> None:
        with pytest.raises(ValueError, match="malformed placeholder"):
            check_template_syntax(template)

    def test_well_formed_templates_pass_and_report_their_names(self) -> None:
        template = "T{a}{b?, L{b}{c:5}} {d:70}"
        assert check_template_syntax(template) == template
        assert template_names(template) == {"a", "b", "c", "d"}


class TestBudgetSplitter:
    def _words(self, n: int, prefix: str = "w") -> str:
        return " ".join(f"{prefix}{i}" for i in range(n))

    def test_text_within_budget_is_untouched(self) -> None:
        assert BudgetSplitter(WordCounter()).split("a b c", 5) == ["a b c"]

    def test_blank_text_has_no_pieces(self) -> None:
        assert BudgetSplitter(WordCounter()).split("  \n ", 5) == []

    def test_paragraphs_are_the_first_cut(self) -> None:
        text = f"{self._words(6, 'a')}\n\n{self._words(6, 'b')}\n\n{self._words(6, 'c')}"
        pieces = BudgetSplitter(WordCounter()).split(text, 13)
        assert pieces == [f"{self._words(6, 'a')}\n\n{self._words(6, 'b')}", self._words(6, "c")]

    def test_a_line_ending_a_sentence_is_preferred_over_any_line(self) -> None:
        text = "one two three\nfour five.\nsix seven eight"
        assert BudgetSplitter(WordCounter()).split(text, 5) == [
            "one two three\nfour five.",
            "six seven eight",
        ]

    def test_sentences_then_words_then_tokens(self) -> None:
        splitter = BudgetSplitter(WordCounter())
        sentences = "aa bb cc. dd ee ff. gg hh ii."
        assert splitter.split(sentences, 4) == ["aa bb cc.", "dd ee ff.", "gg hh ii."]
        assert splitter.split(self._words(10), 4) == [
            "w0 w1 w2 w3",
            "w4 w5 w6 w7",
            "w8 w9",
        ]

    def test_every_piece_fits(self) -> None:
        text = "\n\n".join(self._words(n) for n in (3, 40, 9, 100))
        pieces = BudgetSplitter(WordCounter()).split(text, 12)
        assert all(WordCounter().count(piece) <= 12 for piece in pieces)
        assert " ".join(" ".join(pieces).split()) == " ".join(text.split())

    def test_a_single_overlong_word_falls_back_to_the_token_splitter(self) -> None:
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
        counter = HfTokenCounter("unused")
        counter._tokenizer = tokenizer
        pieces = BudgetSplitter(counter).split("abcdefghij", 4)
        assert pieces == ["abcd", "efgh", "ij"]

    def test_pieces_are_reported_as_offsets_of_the_original_text(self) -> None:
        text = f"{self._words(6, 'a')}\n\n{self._words(6, 'b')}\n\n{self._words(6, 'c')}"
        spans = BudgetSplitter(WordCounter()).split_spans(text, 13)
        assert [text[start:end] for start, end in spans] == [
            f"{self._words(6, 'a')}\n\n{self._words(6, 'b')}",
            self._words(6, "c"),
        ]
        assert spans[1].start == text.index("c0")

    def test_offsets_point_at_each_piece_even_when_it_is_cut_inside_a_paragraph(self) -> None:
        text = "x " + self._words(10) + "\n\n" + self._words(10, "z")
        splitter = BudgetSplitter(WordCounter())
        spans = splitter.split_spans(text, 4)
        assert all(WordCounter().count(text[a:b]) <= 4 for a, b in spans)
        # in order, disjoint, and covering every word
        assert [a for a, _ in spans] == sorted(a for a, _ in spans)
        assert all(spans[i].end <= spans[i + 1].start for i in range(len(spans) - 1))
        assert " ".join(text[a:b] for a, b in spans).split() == text.split()

    def test_a_token_cut_keeps_exact_offsets(self) -> None:
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
        counter = HfTokenCounter("unused")
        counter._tokenizer = tokenizer
        spans = BudgetSplitter(counter).split_spans("abcdefghij", 4)
        assert spans == [(0, 4), (4, 8), (8, 10)]

    def test_only_short_pieces_are_cached(self) -> None:
        counter = WordCounter()
        splitter = BudgetSplitter(counter)
        long_text = self._words(400)  # > 1000 characters: never kept
        splitter.count(long_text)
        splitter.count("a b c")
        assert set(splitter._cache) == {"a b c"}

    def test_the_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            BudgetSplitter(WordCounter()).split("a", 0)
