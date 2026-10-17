"""Text decoding and sanitizing shared by the parsers and the cleaners."""

import codecs
import logging
import re
from email.message import Message

from agintis.modules.knowledge.domain import strip_unstorable as strip_unstorable  # re-export

logger = logging.getLogger(__name__)

_UTF8_BOM = codecs.BOM_UTF8
_UTF16_BOMS = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)
# HTML5 prescan window: the declaration must appear in the first 1024 bytes.
META_SNIFF_BYTES = 1024
_META_CHARSET = re.compile(
    rb"""<meta[^>]+?charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE
)


def mime_charset(mime_type: str) -> str | None:
    """`charset` parameter of a Content-Type value (`text/plain; charset="latin-1"`)."""
    message = Message()
    message["content-type"] = mime_type
    charset = message.get_content_charset()
    return charset.strip() if charset else None


def _known_codec(charset: str) -> str | None:
    try:
        return codecs.lookup(charset).name
    except LookupError:
        return None


def sniff_meta_charset(content: bytes) -> str | None:
    """Charset declared by `<meta charset=...>` / `<meta http-equiv content="...charset=...">`
    in the first `META_SNIFF_BYTES` bytes (ASCII-compatible encodings only), else None."""
    match = _META_CHARSET.search(content[:META_SNIFF_BYTES])
    return match.group(1).decode("ascii") if match else None


def decode_bytes(content: bytes, charset: str | None = None, *, sniff_meta: bool = False) -> str:
    """Best-effort decoding, never raises.

    1. The charset declared by the MIME type, when it is a known codec and decodes strictly.
    2. A BOM (UTF-8 / UTF-16).
    3. Strict UTF-8.
    4. With `sniff_meta` (HTML), the charset the markup declares in its first 1024 bytes, when
       it is a known codec and decodes strictly.
    5. cp1252 (the usual encoding of legacy Spanish/Western text; undecodable bytes become
       U+FFFD).
    A heuristic detector (charset-normalizer) was rejected: it guesses wrong on the short,
    mostly-ASCII texts typical here and adds a dependency for a rare case.
    """
    if charset is not None:
        codec = _known_codec(charset)
        if codec is None:
            logger.warning("Unknown charset %r declared; detecting the encoding", charset)
        else:
            try:
                return content.decode(codec)
            except UnicodeDecodeError:
                logger.warning("Content is not valid %s as declared; detecting the encoding", codec)
    if content.startswith(_UTF8_BOM):
        return content.decode("utf-8-sig", errors="replace")
    if content.startswith(_UTF16_BOMS):
        return content.decode("utf-16", errors="replace")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass
    declared = sniff_meta_charset(content) if sniff_meta else None
    if declared is not None and (codec := _known_codec(declared)) is not None:
        try:
            return content.decode(codec)
        except UnicodeDecodeError:
            logger.warning("Content is not valid %s as the markup declares", codec)
    return content.decode("cp1252", errors="replace")


def decode_text(content: bytes, charset: str | None = None, *, sniff_meta: bool = False) -> str:
    """Decoded text with a single leading BOM removed and `\\n` line endings."""
    text = decode_bytes(content, charset, sniff_meta=sniff_meta).removeprefix("﻿")
    return text.replace("\r\n", "\n").replace("\r", "\n")
