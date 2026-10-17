"""Boot check: the lexical query configuration is the one `chunks.tsv` is generated with.

`tsv` is a generated column, so its text search configuration is fixed by the migration. A query
parsed with another configuration (other stemmer, other stop words, no `unaccent`) silently
matches little or nothing, with no error anywhere: this check turns that into a startup failure.
"""

import re

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

# The generation expression as the catalog deparses it: `to_tsvector('cfg'::regconfig, text)`.
_GENERATION_CONFIG = re.compile(r"to_tsvector\('((?:[^']|'')+)'::regconfig\s*,")

_GENERATION_EXPRESSION = text(
    "SELECT pg_get_expr(d.adbin, d.adrelid) "
    "FROM pg_attrdef d "
    "JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
    "WHERE a.attrelid = to_regclass('chunks') AND a.attname = 'tsv'"
)


class TextSearchConfigError(RuntimeError):
    """The configured query text search configuration does not match `chunks.tsv`."""


async def _regconfig_oid(conn: AsyncConnection, name: str) -> int | None:
    """OID of the text search configuration `name`, None when it does not exist."""
    try:
        async with conn.begin_nested():  # a failed cast must not poison the transaction
            oid = await conn.scalar(text("SELECT CAST(:name AS regconfig)::oid"), {"name": name})
    except DBAPIError:
        return None
    return int(oid) if oid is not None else None


async def verify_text_search_config(engine: AsyncEngine, config: str) -> None:
    """Raise `TextSearchConfigError` unless `config` resolves to the configuration of `tsv`."""
    async with engine.connect() as conn:
        wanted = await _regconfig_oid(conn, config)
        if wanted is None:
            raise TextSearchConfigError(f"Text search configuration {config!r} does not exist")
        expression = await conn.scalar(_GENERATION_EXPRESSION)
        if expression is None:
            raise TextSearchConfigError("Column chunks.tsv has no generation expression")
        match = _GENERATION_CONFIG.search(str(expression))
        if match is None:
            raise TextSearchConfigError(
                f"Cannot find the text search configuration in chunks.tsv: {expression}"
            )
        used = match.group(1).replace("''", "'")
        generated = await _regconfig_oid(conn, used)
        if generated != wanted:
            raise TextSearchConfigError(
                f"KNOWLEDGE_TEXT_SEARCH_CONFIG={config!r} differs from the configuration "
                f"chunks.tsv is generated with ({used!r}): lexical search would not match"
            )
