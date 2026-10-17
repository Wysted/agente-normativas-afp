"""DOCX via python-docx (MIT). Sections split at Title/Heading N paragraphs; tables as text rows.

Extracted: body paragraphs and top-level tables, in document order (merged cells appear once).
IGNORED: headers and footers, footnotes/endnotes, comments, text boxes and shapes, nested
tables inside cells, images and their alt text, tracked-change deletions.

The zip container is checked BEFORE python-docx opens it (zip bombs): entry count, declared
uncompressed size, per-entry and global compression ratio. Declared sizes come from the zip
directory; Python's zipfile never reads past an entry's declared size, so a lying directory
cannot inflate more than it declares.
"""

import re
import zipfile
from io import BytesIO
from typing import Final

from docx import Document as open_docx
from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from agintis.modules.knowledge.application import (
    DocumentTooLargeError,
    ParsedDocument,
    ParsedSection,
    UnparseableDocumentError,
)
from agintis.modules.knowledge.infrastructure.parsers._title import title_from_uri
from agintis.modules.knowledge.infrastructure.parsers.limits import ParseContext, ParseLimits

_HEADING_STYLE = re.compile(r"^(?:heading|título|titulo)\s*(\d+)$", re.IGNORECASE)
_TITLE_STYLE = "title"
_TITLE_LEVEL = 1
_CELL_SEPARATOR = " | "
MAX_ZIP_ENTRIES: Final = 10_000


def check_zip(content: bytes, uri: str, limits: ParseLimits) -> None:
    """Reject zip bombs from the central directory, without decompressing anything."""
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise UnparseableDocumentError(f"Cannot read DOCX {uri!r}: {exc}") from exc
    if len(entries) > MAX_ZIP_ENTRIES:
        raise DocumentTooLargeError(f"{uri!r} has {len(entries)} archive entries")
    total = sum(entry.file_size for entry in entries)
    if total > limits.max_uncompressed_bytes:
        raise DocumentTooLargeError(
            f"{uri!r} expands to {total} bytes (limit {limits.max_uncompressed_bytes})"
        )
    for entry in entries:
        if entry.file_size and entry.file_size > limits.max_compression_ratio * max(
            entry.compress_size, 1
        ):
            raise DocumentTooLargeError(
                f"{uri!r}: entry {entry.filename!r} has a suspicious compression ratio"
            )
    if total > limits.max_compression_ratio * max(len(content), 1):
        raise DocumentTooLargeError(f"{uri!r} has a suspicious overall compression ratio")


def _open(content: bytes, uri: str) -> DocxDocument:
    try:
        return open_docx(BytesIO(content))
    except Exception as exc:  # python-docx raises PackageNotFoundError, KeyError, BadZipFile...
        raise UnparseableDocumentError(f"Cannot read DOCX {uri!r}: {exc}") from exc


def _heading_level(paragraph: Paragraph) -> int | None:
    name = (paragraph.style.name if paragraph.style is not None else "") or ""
    if name.strip().lower() == _TITLE_STYLE:
        return _TITLE_LEVEL
    match = _HEADING_STYLE.match(name.strip())
    return int(match.group(1)) if match else None


def _table_text(table: Table) -> str:
    lines: list[str] = []
    for row in table.rows:
        seen: set[object] = set()  # a merged cell is returned once per grid column it spans
        values: list[str] = []
        for cell in row.cells:
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            if text := cell.text.strip():
                values.append(text)
        if values:
            lines.append(_CELL_SEPARATOR.join(values))
    return "\n".join(lines)


def parse_docx(content: bytes, uri: str, context: ParseContext) -> ParsedDocument:
    check_zip(content, uri, context.limits)
    document = _open(content, uri)
    sections: list[ParsedSection] = []
    path: list[tuple[int, str]] = []
    body: list[str] = []
    title = (document.core_properties.title or "").strip()

    def flush() -> None:
        text = "\n\n".join(body).strip()
        body.clear()
        if text:
            sections.append(ParsedSection(text=text, headings=tuple(h for _, h in path)))

    for block in document.iter_inner_content():
        if isinstance(block, Table):
            table_text = _table_text(block)
            if table_text:
                body.append(table_text)
            continue
        text = block.text.strip()
        level = _heading_level(block)
        if level is None or not text:
            if text:
                body.append(text)
            continue
        flush()
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, text))
        title = title or text
    flush()
    return ParsedDocument(title=title or title_from_uri(uri), sections=tuple(sections))
