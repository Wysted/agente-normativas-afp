"""Resource limits and per-call context shared by the lightweight parsers."""

from dataclasses import dataclass, field
from typing import Final

MIB: Final = 1024 * 1024
# Single source of the default limits: `ParserSettings` reuses them.
DEFAULT_MAX_BYTES: Final = 50 * MIB
DEFAULT_MAX_PAGES: Final = 2000
DEFAULT_MAX_UNCOMPRESSED_BYTES: Final = 200 * MIB
DEFAULT_MAX_COMPRESSION_RATIO: Final = 100.0


@dataclass(frozen=True)
class ParseLimits:
    """Bootstrap passes the configured values; the defaults serve direct construction."""

    max_bytes: int = DEFAULT_MAX_BYTES
    max_pages: int = DEFAULT_MAX_PAGES
    # DOCX (zip) pre-check, before any member is decompressed.
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO

    def __post_init__(self) -> None:
        if min(self.max_bytes, self.max_pages, self.max_uncompressed_bytes) <= 0:
            raise ValueError("parser limits must be greater than 0")
        if self.max_compression_ratio <= 1:
            raise ValueError("max_compression_ratio must be greater than 1")


@dataclass(frozen=True)
class ParseContext:
    """What a parse function may need besides the bytes: MIME charset and limits."""

    charset: str | None = None
    limits: ParseLimits = field(default_factory=ParseLimits)
