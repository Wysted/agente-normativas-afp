"""Common machinery of the extractors and transforms: typed params and cached regexes.

Params are validated by a Pydantic `ParamsModel` per key (at boot by `validate_profiles`, and
again, cached per params object, when the strategy runs). Regexes are compiled once per
(pattern, flags) with the scan cap of `MAX_REGEX_SCAN_CHARS`.
"""

import re
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping
from datetime import date
from functools import lru_cache
from typing import ClassVar, cast

from agintis.modules.knowledge.application import (
    ExtractionContext,
    Extractor,
    RawValue,
    ValueTransform,
)
from agintis.modules.knowledge.domain import ParamValue
from agintis.modules.knowledge.profiles import (
    MAX_REGEX_SCAN_CHARS,
    SCAN_WINDOW_CHARS,
    ParamsModel,
    ProfileRegex,
    compile_profile_regex,
    validate_params,
)

_TYPED_CACHE_SIZE = 512


class NoParams(ParamsModel):
    """For strategies that take no parameters."""


@lru_cache(maxsize=512)
def scan_regex(pattern: str, flags: int = 0) -> ProfileRegex:
    return compile_profile_regex(pattern, flags, max_chars=MAX_REGEX_SCAN_CHARS)


def regex_flags(flags: str) -> int:
    return re.IGNORECASE if "i" in flags else 0


def scan_windows(text: str, size: int = SCAN_WINDOW_CHARS) -> Iterator[str]:
    """`text` in windows of at most `size` characters, cut at a blank line when possible.

    Bounds the work of one regex call (a polynomial pattern costs the square of its input) and
    keeps patterns that wrap over a line break working: only a window edge can split a match,
    and edges prefer paragraph breaks, then line breaks.
    """
    position, end = 0, len(text)
    while position < end:
        stop = min(position + size, end)
        if stop < end:
            for separator in ("\n\n", "\n"):
                cut = text.rfind(separator, position + size // 2, stop)
                if cut >= 0:
                    stop = cut + len(separator)
                    break
        yield text[position:stop]
        position = stop


def memoized[T](
    context: ExtractionContext, namespace: str, params: ParamsModel, compute: Callable[[], T]
) -> T:
    """A document-level value computed once per document and params (`ExtractionContext.memo`).

    Only for results that depend on the document and `params`, never on the current chunk.
    """
    key = f"{namespace}:{params.model_dump_json()}"
    if key not in context.memo:
        context.memo[key] = compute()
    return cast(T, context.memo[key])


class _TypedParams[P: ParamsModel]:
    """Validates a params mapping once per mapping object (they are immutable and long-lived)."""

    def __init__(self, model: type[P], where: str) -> None:
        self._model = model
        self._where = where
        self._cache: dict[int, tuple[Mapping[str, ParamValue], P]] = {}

    def get(self, params: Mapping[str, ParamValue]) -> P:
        hit = self._cache.get(id(params))
        if hit is not None and hit[0] is params:  # the reference keeps the id from being reused
            return hit[1]
        typed = validate_params(self._model, params, where=self._where)
        if len(self._cache) >= _TYPED_CACHE_SIZE:
            self._cache.clear()
        self._cache[id(params)] = (params, typed)
        return typed


class ParamsExtractor[P: ParamsModel](Extractor):
    key: ClassVar[str]
    params_model: type[P]

    def __init__(self) -> None:
        self._typed = _TypedParams(self.params_model, f"extractor {self.key!r} params")

    def extract(self, context: ExtractionContext, params: Mapping[str, ParamValue]) -> RawValue:
        return self._extract(context, self._typed.get(params))

    @abstractmethod
    def _extract(self, context: ExtractionContext, params: P) -> RawValue: ...


class ParamsTransform[P: ParamsModel](ValueTransform):
    key: ClassVar[str]
    params_model: type[P]

    def __init__(self) -> None:
        self._typed = _TypedParams(self.params_model, f"transform {self.key!r} params")

    def apply(self, value: RawValue, params: Mapping[str, ParamValue], today: date) -> RawValue:
        return self._apply(value, self._typed.get(params), today)

    @abstractmethod
    def _apply(self, value: RawValue, params: P, today: date) -> RawValue: ...
