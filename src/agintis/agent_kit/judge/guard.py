"""Deterministic verbatim guard of the LLM judges.

The judges ask the model to COPY spans: the claim (or sentence) from the answer and the evidence
from a source. A model that paraphrases while copying hides the very error the judge exists to
find (the claim is silently repaired toward the source before it is compared), so after parsing
every quote is checked to be a span of the text it claims to come from.

The exact comparison is lenient only on typography: Unicode NFC, whitespace collapsed, curly
quotes turned straight, and the case of the quote's FIRST word (a quote that starts mid-sentence
is often capitalized). Nothing else is folded: no accent stripping, no other case folding. A quote
must start and end at word boundaries ("compatible con" is not a span of "incompatible con"). The
judges show the model neutralized text (`<` / `>` as `&lt;` / `&gt;`), so a quote copied from that
rendering is also accepted when its entities are undone. For claim / sentence quotes the `[n]`
citation markers are removed from BOTH sides first (the prompt lets the model omit them),
together with the space they leave before punctuation.

Two tolerant fallbacks run ONLY after the exact check fails, are counted apart (`claims_fuzzy`,
`evidence_fuzzy`) and are fully deterministic (difflib over tokens, no model). Both work on an
ALLOWLIST: the only tokens whose change is tolerated are the function words of
`TOLERATED_TOKENS` (articles, demonstratives, clitics, simple prepositions, "que"). An open
denylist of "dangerous" words cannot work: the real errors are content words (a number word, an
actor, an antonym, "y" vs "o"), and no list of them is ever complete.

- Claim quotes (`match_quote`): token alignment against the best window of the answer (anchored
  on the longest matching block). Accepted iff at most 2 tokens differ AND at most 10% of the
  quote's tokens (a quote under 10 tokens allows none: it can differ in punctuation / markdown
  only), AND every difference is tolerated: the INSERTION or DELETION of an allowlisted word, or
  the SUBSTITUTION of a word by another of its closed equivalence class
  (`SUBSTITUTION_CLASSES`: "el" / "la", "este" / "éste", "ese" / "dicho"...). A swap between
  prepositions ("a" / "de"), between indefinite and definite articles ("unos" / "los"), and any
  content word ("treinta" / "sesenta", "otorga" / "deniega", "y" / "o", a dropped "no") is the
  paraphrase the guard is there to reject.
- Citation sentences (`match_sentence`): the same match against the sentence the code enumerated
  for that index (not the whole answer), and the quote must cover at least 90% of that
  sentence's tokens, so a quote cannot be a harmless fragment of another sentence.
- Evidence fragments (`match_evidence`): a quote is split at ellipses (`\\.{2,}`, `…`, `[...]`)
  and " | ", each fragment stripped of edge punctuation. Fragments of at least 12 chars must be
  exact spans of the cited sources, or pass the shingle rule: at least 5 words, at least 80% of
  the word 5-gram shingles present in the sources, and EVERY token not covered by a present
  shingle tolerated (so an inserted "no", a prefixed "nunca" or a swapped edge word, whose
  shingles are all absent, is refused) and every digit token covered. A PDF artifact such as an
  injected footnote only breaks the shingles that cross it; its neighbours stay covered. Shorter
  fragments are ignored unless they carry a digit, a negation, a quantifier, a number word, an
  ordinal or a condition marker (`WATCH_TOKENS`): then they must be exact spans, never ignored.
  At least one fragment of 12 chars must exist, and a whole quote shorter than that is accepted
  only when it carries such a watched token.

`uncovered_protected` is REPORT-only (it changes no verdict): the watched tokens of the answer
that fall outside every claim quote that matched, because a quote that is a truncation of a
sentence ("podrá retirar sus fondos") drops the "no" before it and still matches exactly.
"""

import re
import string
import unicodedata
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from typing import Any, Literal

from agintis.agent_kit.judge.citations import CITATION

_WHITESPACE = re.compile(r"\s+")
# Curly quotes (code points: U+2018..U+201B single, U+201C..U+201D double) -> straight ones.
_QUOTES = {0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'", 0x201C: '"', 0x201D: '"'}
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
# Fragment separators of an evidence quote: ellipses of 2+ dots, "…", "[...]" and " | ".
_FRAGMENT_SPLIT = re.compile(r"\[\.\.\.\]|\.{2,}|…|\s\|\s")
# "_" is a word character: stripping it blindly would cut `"__main__"` and break the `(?!\w)`
# boundary, so it is trimmed only as a paired emphasis delimiter (`_a b_`, `__a b__`; see `_trim`).
_EDGE = string.punctuation.replace("_", "") + string.whitespace + "¿¡«»…"
_EMPHASIS = re.compile(r"(_{1,2})(\S(?:.*\S)?)\1")
# An inline footnote mark glued to a word ("equivalente3"): 1-2 digits right after a letter.
_FOOTNOTE = re.compile(r"(?<=[^\W\d])\d{1,2}\b")
_TOKEN = re.compile(r"\[\s*\d[^\]]*\]|\w+")  # words and [n] markers
_WORD = re.compile(r"\w+")
MIN_FRAGMENT_CHARS = 12  # shorter fragments are too weak to prove anything and are ignored
MAX_QUOTE_EDITS = 2
MAX_QUOTE_EDIT_RATIO = 0.10
MIN_SENTENCE_COVERAGE = 0.90
SHINGLE_WORDS = 5
MIN_SHINGLE_COVERAGE = 0.8

Match = Literal["exact", "fuzzy"]

# The ONLY tokens whose insertion or deletion a tolerant match accepts (compared lowercase):
# function words. "y", "e", "o", "u" (conjunctions: "y" vs "o" flips a condition) are NOT here.
TOLERATED_TOKENS = frozenset(
    {
        *("el", "la", "los", "las", "un", "una", "unos", "unas", "lo"),  # articles
        *("este", "esta", "estos", "estas", "éste", "ésta", "éstos", "éstas"),
        *("ese", "esa", "esos", "esas", "dicho", "dicha", "dichos", "dichas"),
        *("se", "le", "les"),  # clitics (la, lo, las, los are above)
        *("de", "del", "a", "al", "en", "por", "para", "con"),  # prepositions
        "que",
    }
)

# A SUBSTITUTION is tolerated only between members of one class: the same function word in
# another gender / accent / register. Never between prepositions, never indefinite <-> definite.
SUBSTITUTION_CLASSES: tuple[frozenset[str], ...] = (
    frozenset({"el", "la", "lo"}),
    frozenset({"los", "las"}),
    frozenset({"este", "éste"}),
    frozenset({"esta", "ésta"}),
    frozenset({"estos", "éstos"}),
    frozenset({"estas", "éstas"}),
    frozenset({"ese", "dicho"}),
    frozenset({"esa", "dicha"}),
    frozenset({"esos", "dichos"}),
    frozenset({"esas", "dichas"}),
    frozenset({"se", "le"}),
)

# Negations, quantifiers, conditions, temporal markers, number words and ordinals (digits count
# too). They drive the `uncovered_protected` report AND the short-evidence-fragment rule, which
# DOES gate verdicts: a short fragment with one of them must be an exact span. Not a tolerance
# list ("un" / "una" are articles, not watched).
WATCH_TOKENS = frozenset(
    {
        *("no", "ni", "nunca", "jamás", "tampoco", "sin", "nadie", "nada"),
        *("ningún", "ninguna", "ninguno"),
        *("todo", "toda", "todos", "todas", "cualquier", "cualquiera", "algunos", "algunas"),
        *("solo", "sólo", "solamente", "únicamente", "exclusivamente", "siempre", "cada"),
        *("ambos", "ambas", "más", "menos", "mayor", "menor", "máximo", "mínimo"),
        *("si", "salvo", "excepto", "cuando", "mientras", "antes", "después", "desde", "hasta"),
        *("dentro", "durante", "ya", "aún", "todavía", "vez", "previa", "previo", "posterior"),
        # number words
        *("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"),
        *("diez", "once", "doce", "trece", "catorce", "quince", "veinte"),
        *("treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"),
        *("cien", "ciento", "mil", "millón", "millones"),
        *("medio", "media", "mitad", "doble", "triple", "tercio"),
        # ordinals and relative position
        *("primer", "primero", "primera", "segundo", "segunda", "tercer", "tercero", "tercera"),
        *("cuarto", "cuarta", "quinto", "quinta", "sexto", "sexta", "séptimo", "séptima"),
        *("octavo", "octava", "noveno", "novena", "décimo", "décima", "undécimo", "undécima"),
        *("duodécimo", "duodécima", "vigésimo", "vigésima"),
        *("último", "última", "penúltimo", "penúltima", "anterior", "siguiente"),
    }
)
# dieciséis..diecinueve, veintiuno..veintinueve, doscientos, trescientas...
_WATCH_PATTERNS = (re.compile(r"(?:dieci|veinti)\w+"), re.compile(r"\w+cient[oa]s"))


def parameters() -> dict[str, Any]:
    """Everything that decides what the guard accepts (hashed into the report meta, so two runs
    are comparable only when it is equal). Read at call time."""
    return {
        "tolerated_tokens": sorted(TOLERATED_TOKENS),
        "substitution_classes": sorted(sorted(c) for c in SUBSTITUTION_CLASSES),
        "watch_tokens": sorted(WATCH_TOKENS),
        "min_fragment_chars": MIN_FRAGMENT_CHARS,
        "max_quote_edits": MAX_QUOTE_EDITS,
        "max_quote_edit_ratio": MAX_QUOTE_EDIT_RATIO,
        "min_sentence_coverage": MIN_SENTENCE_COVERAGE,
        "shingle_words": SHINGLE_WORDS,
        "min_shingle_coverage": MIN_SHINGLE_COVERAGE,
        "patterns": [
            p.pattern
            for p in (
                _FRAGMENT_SPLIT,
                _FOOTNOTE,
                _EMPHASIS,
                _TOKEN,
                _SPACE_BEFORE_PUNCT,
                CITATION,
                *_WATCH_PATTERNS,
            )
        ],
        "edge_chars": _EDGE,
        "quote_map": sorted(_QUOTES.items()),
    }


def normalize(text: str) -> str:
    """The light normalization both sides of a comparison go through."""
    text = unicodedata.normalize("NFC", text).translate(_QUOTES)
    return _WHITESPACE.sub(" ", text).strip()


def _unescape(text: str) -> str:
    return text.replace("&lt;", "<").replace("&gt;", ">")


def strip_markers(text: str) -> str:
    """`text` without `[n]` citation markers nor the space they leave before punctuation."""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", CITATION.sub("", text))


def _trim(text: str) -> str:
    """`text` without edge punctuation nor a paired markdown emphasis delimiter around several
    words (`_el plazo_`). shortcut: a single emphasized word (`_plazo_`) is left alone since it
    cannot be told from an identifier (`_private_`); upgrade if a case quotes those. Inner text with
    `_` (`__init__ y __main__`) is never unwrapped."""
    text = text.strip(_EDGE)
    match = _EMPHASIS.fullmatch(text)
    if match is not None and " " in match.group(2) and "_" not in match.group(2):
        return match.group(2).strip(_EDGE)
    return text


def _span_pattern(wanted: str) -> re.Pattern[str]:
    """`wanted` (normalized) as a pattern: literal, starting and ending at word boundaries, with
    the case of its first word ignored."""
    first = _WORD.search(wanted)
    if first is None:
        return re.compile(re.escape(wanted))
    start = r"(?<!\w)" if first.start() == 0 else ""
    end = r"(?!\w)" if _WORD.match(wanted[-1]) else ""
    return re.compile(
        re.escape(wanted[: first.start()])
        + start
        + f"(?i:{re.escape(first.group())})"
        + re.escape(wanted[first.end() :])
        + end
    )


def is_verbatim(quote: str, haystack: str | Iterable[str]) -> bool:
    """True iff the (normalized) quote is a non-empty span of the (normalized) haystack, at word
    boundaries, or of any of them when several texts are given."""
    wanted = normalize(quote)
    if not wanted:
        return False
    texts = [normalize(t) for t in ([haystack] if isinstance(haystack, str) else haystack)]
    patterns = {_span_pattern(c) for c in (wanted, normalize(_unescape(quote))) if c}
    return any(p.search(text) for text in texts for p in patterns)


def is_tolerated(token: str) -> bool:
    return token.lower() in TOLERATED_TOKENS


def _substitutable(a: str, b: str) -> bool:
    low_a, low_b = a.lower(), b.lower()
    if low_a == low_b:
        return is_tolerated(low_a)
    return any(low_a in c and low_b in c for c in SUBSTITUTION_CLASSES)


def _has_digit(token: str) -> bool:
    return any(c.isdigit() for c in token)


def is_watched(token: str) -> bool:
    low = token.lower()
    return (
        _has_digit(token) or low in WATCH_TOKENS or any(p.fullmatch(low) for p in _WATCH_PATTERNS)
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def _blocks(wanted: Sequence[str], haystack: Sequence[str]) -> list[Any]:
    matcher = SequenceMatcher(None, wanted, haystack, autojunk=False)
    return [b for b in matcher.get_matching_blocks() if b.size]


def _window(wanted: Sequence[str], haystack: Sequence[str]) -> tuple[int, int] | None:
    """The span of `haystack` the quote best aligns with, or None when nothing matches.

    Anchored on the LONGEST matching block (a stray match of an article in an earlier sentence
    cannot drag the window): the quote is realigned only inside the haystack around the anchor
    (the quote's length plus the edit budget on both sides) and the window is extrapolated from
    the first and last blocks found there."""
    blocks = _blocks(wanted, haystack)
    if not blocks:
        return None
    anchor = max(blocks, key=lambda b: b.size)
    origin = anchor.b - anchor.a
    low = max(0, origin - MAX_QUOTE_EDITS)
    region = haystack[low : min(len(haystack), origin + len(wanted) + MAX_QUOTE_EDITS)]
    found = _blocks(wanted, region)
    first, last = found[0], found[-1]
    start = max(0, first.b - first.a)
    end = min(len(region), last.b + last.size + len(wanted) - last.a - last.size)
    return low + start, low + end


def _within_budget(wanted: Sequence[str], window: Sequence[str]) -> bool:
    """At most the edit budget of differences, each one tolerated: an insertion / deletion of a
    function word, or a substitution inside one equivalence class."""
    budget = min(MAX_QUOTE_EDITS, int(len(wanted) * MAX_QUOTE_EDIT_RATIO))
    edits = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, wanted, window, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        edits += max(i2 - i1, j2 - j1)
        if tag == "replace":
            pairs = zip(wanted[i1:i2], window[j1:j2], strict=False)
            tolerated = i2 - i1 == j2 - j1 and all(_substitutable(a, b) for a, b in pairs)
        else:
            tolerated = all(is_tolerated(t) for t in (*wanted[i1:i2], *window[j1:j2]))
        if not tolerated:
            return False
    return edits <= budget


def _spellings(wanted: Sequence[str], pool: Iterable[str]) -> list[list[str]]:
    """The quote as it is, then with its FIRST word spelled like each case variant found in
    `pool` (only the first word is compared without regard to case)."""
    if not wanted:
        return [[]]
    first = wanted[0]
    variants = sorted({t for t in pool if t != first and t.casefold() == first.casefold()})
    return [[spelling, *wanted[1:]] for spelling in (first, *variants)]


def _aligned(wanted: Sequence[str], haystack: Sequence[str]) -> tuple[list[str], tuple[int, int]]:
    """`(spelling, span)` of the quote within the edit budget, `([], (0, 0))` if it fits nowhere."""
    for candidate in _spellings(wanted, haystack):
        span = _window(candidate, haystack)
        if span is not None and _within_budget(candidate, haystack[span[0] : span[1]]):
            return candidate, span
    return [], (0, 0)


def match_quote(quote: str, answer: str) -> Match | None:
    """How a claim quote relates to the answer (markers ignored): `exact` (a span after the light
    normalization), `fuzzy` (the tolerant token match of the module docstring) or None."""
    clean_quote, clean_answer = strip_markers(quote), strip_markers(answer)
    if is_verbatim(clean_quote, clean_answer):
        return "exact"
    wanted = _tokens(_unescape(clean_quote))
    if not wanted:
        return None
    return "fuzzy" if _aligned(wanted, _tokens(clean_answer))[0] else None


def match_sentence(quote: str, sentence: str) -> Match | None:
    """`match_quote` against ONE enumerated sentence, which the quote must also cover for at
    least `MIN_SENTENCE_COVERAGE` of its (non-marker) tokens."""
    match = match_quote(quote, sentence)
    if match is None:
        return None
    wanted = _tokens(_unescape(strip_markers(quote)))
    target = _tokens(strip_markers(sentence))
    matched = max(sum(b.size for b in _blocks(w, target)) for w in _spellings(wanted, target))
    return match if target and matched >= MIN_SENTENCE_COVERAGE * len(target) else None


def uncovered_protected(answer: str, quotes: Iterable[str]) -> list[str]:
    """Watched tokens (digits, negations, quantifiers, conditions, number words, ordinals) of the
    answer outside the span of every quote that MATCHED the answer (exact or fuzzy): what a
    truncating quote silently dropped. Markers are not tokens here."""
    tokens = _tokens(strip_markers(answer))
    covered: set[int] = set()
    for quote in quotes:
        wanted = _tokens(_unescape(strip_markers(quote)))
        if wanted and match_quote(quote, answer) is not None:
            covered.update(range(*_aligned(wanted, tokens)[1]))
    return [t for i, t in enumerate(tokens) if i not in covered and is_watched(t)]


def _has_token_span(part: str, texts: Iterable[str]) -> bool:
    """The words of `part` appear contiguously, as whole words, in some text (a short fragment
    such as "no" must not match inside "ninguno"); the first word ignores case."""
    wanted = _tokens(part)
    n = len(wanted)
    return bool(n) and any(
        haystack[i].casefold() == wanted[0].casefold() and haystack[i + 1 : i + n] == wanted[1:]
        for haystack in map(_tokens, texts)
        for i in range(len(haystack) - n + 1)
    )


def _footnote_free(texts: Iterable[str]) -> list[str]:
    """Each source as is, then without footnote digits."""
    return [t for source in texts for t in (source, _FOOTNOTE.sub("", source))]


def _shingles(tokens: Sequence[str]) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + SHINGLE_WORDS]) for i in range(len(tokens) - SHINGLE_WORDS + 1)}


def _passes_shingles(words: Sequence[str], available: set[tuple[str, ...]]) -> bool:
    covered: set[int] = set()
    present = 0
    total = len(words) - SHINGLE_WORDS + 1
    for i in range(total):
        if tuple(words[i : i + SHINGLE_WORDS]) in available:
            present += 1
            covered.update(range(i, i + SHINGLE_WORDS))
    uncovered = [t for i, t in enumerate(words) if i not in covered]
    return (
        present >= MIN_SHINGLE_COVERAGE * total
        and all(is_tolerated(t) for t in uncovered)
        and not any(_has_digit(t) for t in uncovered)
    )


def _fuzzy_fragment(fragment: str, sources: list[str]) -> bool:
    words = _tokens(fragment)
    if len(words) < SHINGLE_WORDS:
        return False
    pools = [_tokens(t) for t in sources]
    available = set().union(*(_shingles(p) for p in pools))
    seen = {t for p in pools for t in p}
    return any(_passes_shingles(spelling, available) for spelling in _spellings(words, seen))


def _exact_whole(quote: str, texts: list[str]) -> bool:
    """The whole quote is a span of a source AND is strong enough to prove anything: at least
    `MIN_FRAGMENT_CHARS` long, or short with a watched token that is a whole-word span."""
    if not is_verbatim(quote, texts):
        return False
    whole = _trim(normalize(quote))
    return len(whole) >= MIN_FRAGMENT_CHARS or (
        any(is_watched(t) for t in _tokens(whole)) and _has_token_span(whole, texts)
    )


def match_evidence(quote: str, sources: Iterable[str]) -> Match | None:
    """How an evidence quote relates to the cited sources: `exact` (the whole quote, or every
    fragment, is a span of some source, footnote digits tolerated), `fuzzy` (some fragment
    passed only the shingle rule) or None."""
    texts = _footnote_free(sources)
    if _exact_whole(quote, texts):
        return "exact"
    parts = [_trim(normalize(f)) for f in _FRAGMENT_SPLIT.split(quote)]
    long_parts = [p for p in parts if len(p) >= MIN_FRAGMENT_CHARS]
    if not long_parts:
        return None
    for part in parts:  # a short fragment that carries a number / negation is never ignored
        short = 0 < len(part) < MIN_FRAGMENT_CHARS
        bare = part.strip("_")  # `_no_` is emphasis here, not an identifier
        if short and any(is_watched(t) for t in _tokens(bare)) and not _has_token_span(bare, texts):
            return None
    fuzzy = False
    for part in long_parts:
        if is_verbatim(part, texts):
            continue
        if not _fuzzy_fragment(part, texts):
            return None
        fuzzy = True
    return "fuzzy" if fuzzy else "exact"
