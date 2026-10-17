"""Citation markers of an answer: parsing, sentence splitting and the deterministic precheck.

The synthesis prompt asks the model to cite with `[n]` (n = the number of a source, restarting at
1 every turn). Models also write `[1, 2]`, `[1-3]` (hyphen or en dash; ranges are expanded) and
padded numbers such as `[0001]`; all of them name sources, so a dangling `[1, 7]` is caught.
Everything here is text processing, no model involved.
"""

import re
from dataclasses import dataclass

# At most 3 digits, leading zeros allowed (`[0001]` is 1, `[2024]` is not a citation): the same
# reach as `SynthesizeNode`'s own `[n]` pattern (`\[\d{1,3}\]`), widened only to lists and ranges.
_NUMBER = r"0*\d{1,3}"
# `[3]`, `[1, 2]`, `[1;2]`, `[1-3]` (hyphen or en dash): numbers joined by , ; or a range dash.
MARKER = rf"\[\s*{_NUMBER}(?:\s*(?:[,;]|[-\u2013])\s*{_NUMBER})*\s*\]"
CITATION = re.compile(MARKER)
_ITEM = re.compile(rf"({_NUMBER})(?:\s*[-\u2013]\s*({_NUMBER}))?")
MAX_RANGE = 100  # a range wider than this is not a citation list; only its ends are kept

# A sentence ends at . ! ? followed by blanks, or at a line break (bullets, list items).
_BOUNDARY = re.compile(r"(?<=[.!?])[ \t]+|\n+")
# Markers (and the punctuation around them) that open a piece: "[1] Next sentence" after a full
# stop belongs to the sentence BEFORE.
_LEADING_MARKERS = re.compile(rf"^(?:{MARKER}[\s.,;:!?]*)+")
# Abbreviations whose full stop does not end a sentence (compared lowercase, without the dot).
ABBREVIATIONS = frozenset(
    [
        "art",
        "arts",
        "inc",
        "incs",
        "núm",
        "num",
        "nro",
        "nº",
        "cap",
        "caps",
        "párr",
        "pág",
        "págs",
        "pp",
        "vs",
        "cfr",
        "ej",
        "ref",
        "fig",
        "tít",
        "ss",
        "sr",
        "sra",
        "sres",
        "srta",
        "dr",
        "dra",
        "lic",
        "ing",
        "atte",
        "dto",
        "dtos",
    ]
)
# Dotted initialisms: D.L. / D.S. / D.F.L. / S.A.
_INITIALISM = re.compile(r"^(?:[A-Za-zÁÉÍÓÚÑáéíóúñ]\.){2,}$")


@dataclass(frozen=True, slots=True)
class CitedSentence:
    """A sentence of the answer that carries at least one `[n]` marker."""

    index: int  # 1-based, among the CITED sentences
    text: str
    numbers: tuple[int, ...]  # distinct, in order of appearance


def _marker_numbers(marker: str) -> list[int]:
    numbers: list[int] = []
    for match in _ITEM.finditer(marker):
        low = int(match.group(1))
        if match.group(2) is None:
            numbers.append(low)
            continue
        low, high = sorted((low, int(match.group(2))))
        numbers.extend(range(low, high + 1) if high - low <= MAX_RANGE else (low, high))
    return numbers


_BACKTICK_RUN = re.compile(r"`+")


# shortcut: quadratic on pathological backtick runs (~3.5 s at 40k chars), fine for real answers;
# index the runs by width if answers grow.
def _code_span_ranges(text: str) -> list[tuple[int, int]]:
    """Inline code spans of `text` as (start, end): CommonMark, a backtick run closed by the next
    run of the SAME length; an unclosed run is literal, and a backslash-escaped backtick does not
    open a span (the rest of its run does)."""
    runs = list(_BACKTICK_RUN.finditer(text))
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(runs):
        opener = runs[i]
        start = opener.start()
        backslashes = len(text[:start]) - len(text[:start].rstrip("\\"))
        if backslashes % 2:
            start += 1
        width = opener.end() - start
        close = next(
            (j for j in range(i + 1, len(runs)) if runs[j].end() - runs[j].start() == width), None
        )
        if width == 0 or close is None:
            i += 1
            continue
        ranges.append((start, runs[close].end()))
        i = close + 1
    return ranges


def _blank_code_spans(text: str) -> str:
    """`text` with inline code spans replaced by a space."""
    out: list[str] = []
    pos = 0
    for start, end in _code_span_ranges(text):
        out += [text[pos:start], " "]
        pos = end
    out.append(text[pos:])
    return "".join(out)


def prose_only(text: str) -> str:
    """`text` without fenced code blocks and inline code spans: markers there (`a[1]`, `[5, 2]`)
    are code, not citations. Spans are paired inside one prose run (never across a fence), the
    same unit `split_sentences` works on."""
    return "\n".join(_blank_code_spans(part) for is_code, part in _segments(text) if not is_code)


def _prose_numbers(text: str) -> tuple[int, ...]:
    found = (n for m in CITATION.finditer(prose_only(text)) for n in _marker_numbers(m.group()))
    return tuple(dict.fromkeys(found))


def _ends_with_abbreviation(piece: str) -> bool:
    words = piece.split()
    last = words[-1].lstrip("(\"'«“[") if words else ""
    return bool(
        last.endswith(".")
        and (_INITIALISM.match(last) or last.rstrip(".").lower() in ABBREVIATIONS)
    )


# A boundary character inside an inline code span is swapped for a private-use char while the text
# is split, so a span (and the marker-looking text in it) always stays in ONE sentence.
_MASK = {".": "\ue000", "!": "\ue001", "?": "\ue002", "\n": "\ue003"}
_UNMASK = {ord(v): k for k, v in _MASK.items()}


def _mask_code_spans(text: str) -> str:
    chars = list(text)
    for start, end in _code_span_ranges(text):
        for i in range(start, end):
            chars[i] = _MASK.get(chars[i], chars[i])
    return "".join(chars)


def _pieces(answer: str) -> list[str]:
    """Pieces cut at sentence ends, NOT after an abbreviation ("art. 5", "D.L. 3.500") nor when
    the next piece starts in lowercase (a sentence does not)."""
    pieces: list[str] = []
    joined_to_previous = False
    start = 0
    for match in [*_BOUNDARY.finditer(answer), None]:
        end = len(answer) if match is None else match.start()
        piece = answer[start:end].strip()
        if piece:
            if joined_to_previous and pieces:
                pieces[-1] = f"{pieces[-1]} {piece}"
            else:
                pieces.append(piece)
        if match is None:
            break
        start = match.end()
        next_char = answer[start : start + 1]
        joined_to_previous = (
            "\n" not in match.group()
            and bool(pieces)
            and (_ends_with_abbreviation(pieces[-1]) or next_char.islower())
        )
    return pieces


# shortcut: fences indented 0-3 spaces only (4+ inside list items falls back to the line-by-line
# split), upgrade when a case indents code in lists.
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")


def _fence_close(line: str, opener: str) -> tuple[int, str] | None:
    """(end of the fence, text after it) when `line` closes a block opened with `opener` (same
    char, at least as long), else None. Text on the closing line ("``` Esto asigna x [2].") is
    not part of the block: it continues as prose. A fence glued to a word ("```js") is an opening
    with an info string, never a close."""
    match = _FENCE.match(line)
    if match is None or match.group(1)[0] != opener[0] or len(match.group(1)) < len(opener):
        return None
    if match.group(2)[:1].isalnum():
        return None
    return match.end(1), match.group(2).strip()


def _segments(answer: str) -> list[tuple[bool, str]]:
    """`answer` as (is_code, text) runs: fenced blocks (opening to closing fence, or to the end
    when unclosed) are code; markers on the closing line go to the prose that follows."""
    segments: list[tuple[bool, str]] = []
    prose: list[str] = []
    code: list[str] = []
    opener = ""
    for line in answer.split("\n"):
        if opener:
            code.append(line)
            close = _fence_close(line, opener)
            if close is not None:
                end, rest = close
                code[-1] = line[:end]
                segments.append((True, "\n".join(code)))
                code, opener = [], ""
                if rest.strip(" .,;:!?"):  # a bare "```." leaves no prose
                    prose.append(rest)
            continue
        match = _FENCE.match(line)
        if match is not None and not (match.group(1)[0] == "`" and "`" in match.group(2)):
            if prose:
                segments.append((False, "\n".join(prose)))
            prose, code, opener = [], [line], match.group(1)
        else:
            prose.append(line)
    if code:
        segments.append((True, "\n".join(code)))
    if prose:
        segments.append((False, "\n".join(prose)))
    return segments


def _introduces_block(sentence: str) -> bool:
    """True when `sentence` leads into a code block: it ends with ':' (markers after it are
    ignored: "Por ejemplo: [1]") or has no final punctuation."""
    return not CITATION.sub("", sentence).rstrip().endswith((".", "!", "?"))


def split_sentences(answer: str) -> list[str]:
    """The sentences of `answer`. Markers that follow the full stop ("... plazo. [1] Siguiente")
    are glued back to the sentence they belong to; blank pieces are dropped.

    A fenced code block (``` or ~~~) is atomic: never split inside it. It is glued to the previous
    sentence only when that sentence introduces it (ends with ':' or has no final punctuation,
    markers aside) and is not itself a code block; otherwise ("A [1].\n```...") the block is its
    own unit. Markers right after the closing fence (same line or next) are glued to the unit:
    "Por ejemplo:\n```py\n...\n```\n[1]" is ONE sentence. A unit with no marker stays uncited."""
    sentences: list[str] = []
    last_is_code = False
    for is_code, text in _segments(answer):
        if is_code:
            if sentences and not last_is_code and _introduces_block(sentences[-1]):
                sentences[-1] = f"{sentences[-1]}\n{text}"
            else:
                sentences.append(text)
            last_is_code = True
            continue
        for piece in _pieces(_mask_code_spans(text)):
            piece = piece.translate(_UNMASK)
            leading = _LEADING_MARKERS.match(piece) if sentences else None
            if leading is not None:
                sentences[-1] = f"{sentences[-1]} {leading.group().strip()}"
                piece = piece[leading.end() :].strip()
            if piece:
                sentences.append(piece)
                last_is_code = False
    return sentences


def cited_sentences(answer: str) -> list[CitedSentence]:
    """The sentences of `answer` with at least one marker, numbered 1.. among themselves."""
    cited: list[CitedSentence] = []
    for sentence in split_sentences(answer):
        numbers = _prose_numbers(sentence)
        if numbers:
            cited.append(CitedSentence(len(cited) + 1, sentence, numbers))
    return cited


def citation_numbers(text: str) -> tuple[int, ...]:
    """Distinct source numbers named by the markers of `text` outside code, in order of
    appearance: the union of `cited_sentences`' numbers, so a precheck (dangling, refusal) and the
    per-sentence enumeration never disagree (a lone fence line parses differently from the same
    text inside a longer one)."""
    found = (n for s in split_sentences(text) for n in _prose_numbers(s))
    return tuple(dict.fromkeys(found))


def dangling_citations(answer: str, source_count: int) -> tuple[int, ...]:
    """Markers that name a source that does not exist (0, or above the number of sources)."""
    return tuple(n for n in citation_numbers(answer) if not 1 <= n <= source_count)
