"""Token counting/splitting with a Hugging Face `tokenizers` tokenizer (Rust, no torch)."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

_WHITESPACE = " \n\t"


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Cut `text` at token boundaries into pieces of at most `max_tokens` tokens."""
        ...


def enforce_budget(text: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Post-condition of every split: pieces of at most `max_tokens` according to `count`.

    Slicing a text at token offsets and re-encoding the slice can yield more tokens than the
    window (merges across the cut, normalization). Oversized pieces are shrunk (longest prefix
    that fits, found by bisection), backing off to the last whitespace so words stay whole.
    A single character that alone exceeds the budget is emitted as-is (nothing smaller exists).
    """
    pieces: list[str] = []
    rest = text.strip()
    while rest:
        if len(rest) == 1 or count(rest) <= max_tokens:
            pieces.append(rest)
            break
        low, high = 1, len(rest) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if count(rest[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        cut = low
        if not rest[cut].isspace():
            back = max(rest.rfind(ws, 1, cut) for ws in _WHITESPACE)
            if back > 0:
                cut = back
        piece = rest[:cut].strip()
        if piece:
            pieces.append(piece)
        rest = rest[cut:].strip()
    return pieces


class HfTokenCounter:
    """Counts WITHOUT special tokens (the chunker reserves room for them).

    `source` is a local tokenizer.json path or a hub id (e.g. "BAAI/bge-m3", cached in
    ~/.cache/huggingface after the first download, so later runs are offline-friendly). Loading
    is lazy and thread-safe so building the object never touches the network or the disk.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._tokenizer: Tokenizer | None = None
        self._lock = threading.Lock()

    def _load(self) -> Tokenizer:
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    path = Path(self._source)
                    tokenizer = (
                        Tokenizer.from_file(str(path))
                        if path.is_file()
                        else Tokenizer.from_pretrained(self._source)
                    )
                    tokenizer.no_truncation()
                    tokenizer.no_padding()
                    self._tokenizer = tokenizer
        return self._tokenizer

    def warm_up(self) -> None:
        """Load the tokenizer now (raises when it cannot be loaded): fail fast at boot."""
        self._load()

    def count(self, text: str) -> int:
        return len(self._load().encode(text, add_special_tokens=False).ids)

    def split(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        offsets = self._load().encode(text, add_special_tokens=False).offsets
        windows = (
            text[offsets[start][0] : offsets[min(start + max_tokens, len(offsets)) - 1][1]]
            for start in range(0, len(offsets), max_tokens)
        )
        return [
            fitted
            for window in windows
            for fitted in enforce_budget(window, max_tokens, self.count)
        ]
