"""Expression indexes of `indexed` metadata fields: expected set, diff against the catalog, sync.

`MetadataField.indexed` creates nothing by itself: an index is schema, and the set a deployment
needs depends on the profiles it loads, so it is managed by `MetadataIndexSync` (driven by
`python -m agintis.cli.indexes`) instead of an Alembic migration.

WHAT is indexed is derived from the SQL converter, never duplicated by hand: a scalar field gets a
partial btree on `ordering_expression(metadata_spec(field))` (`WHERE <expr> IS NOT NULL`), the
exact expression its range / ordering predicates compare (typed under a `jsonb_typeof` guard,
`COLLATE "C"` for text), so the planner matches query and index; the predicate keeps chunks
without the field (or with another type) out of the index, and every range comparison implies it.
Only RANGE / ORDER BY need such an index. Equality and list containment use `metadata @> ...`,
which the table-wide GIN `ix_chunks_metadata` (jsonb_path_ops) already serves; a per-field index
cannot be matched by that operator, so do not flag equality-only fields `indexed`, and list fields
get no per-field index at all.

The catalog is the source of truth for what exists. Each index created here carries a comment
with the fingerprint of its definition (expression + predicate) and a hash of the definition
Postgres stored (`pg_get_indexdef`), so an index built from another expression is STALE and one
altered, built by hand or marked by someone else is UNMANAGED. Indexes are only ever dropped by an
explicit `prune` (orphans, stale, unmanaged and invalid ones, never one that is still building).
A sync run holds one advisory lock, so two runs never interleave their DDL.
"""

import hashlib
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import Any

from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.pool import NullPool

from agintis.modules.knowledge.domain import MetadataField
from agintis.modules.knowledge.infrastructure.postgres.fields import metadata_spec
from agintis.modules.knowledge.infrastructure.postgres.locks import index_sync_lock_keys
from agintis.modules.knowledge.infrastructure.postgres.tables import METADATA_INDEX_PREFIX
from agintis.platform.db import ordering_expression

MAX_IDENTIFIER_LENGTH = 63  # Postgres NAMEDATALEN - 1
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
_DIGEST_LENGTH = 8
_FINGERPRINT_LENGTH = 12
_COMMENT_PREFIX = "agintis:expr:"
_COMMENT = re.compile(rf"{re.escape(_COMMENT_PREFIX)}([0-9a-f]{{12}}):def:([0-9a-f]{{12}})")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")

logger = logging.getLogger(__name__)


class ConflictingIndexError(ValueError):
    """Two declared fields share an index name but would need different expressions."""


class IndexStatus(StrEnum):
    MISSING = "missing"  # expected, not in the database
    PRESENT = "present"  # expected, valid and exactly as this tool created it
    STALE = "stale"  # exists, but was built from another definition than the converter emits
    UNMANAGED = "unmanaged"  # our name, but no (or a foreign / out-of-date) fingerprint comment
    INVALID = "invalid"  # exists but unusable (a failed CREATE INDEX CONCURRENTLY)
    BUILDING = "building"  # still being built by some session: never touched
    ORPHAN = "orphan"  # managed prefix, but no loaded profile expects it


@dataclass(frozen=True)
class MetadataIndexSpec:
    """An expected index: stable name plus the SQL expression the converter emits."""

    name: str
    field: str
    expression: str

    @property
    def predicate(self) -> str:
        return f"({self.expression}) IS NOT NULL"

    @property
    def definition(self) -> str:
        return f"({self.expression}) WHERE {self.predicate}"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.definition.encode()).hexdigest()[:_FINGERPRINT_LENGTH]

    def comment_for(self, indexdef: str) -> str:
        """Comment recording this spec's fingerprint and the definition Postgres stored."""
        return f"{_COMMENT_PREFIX}{self.fingerprint}:def:{_definition_hash(indexdef)}"

    @property
    def create_sql(self) -> str:
        return (
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{_identifier(self.name)}" '
            f"ON chunks (({self.expression})) WHERE {self.predicate}"
        )


@dataclass(frozen=True)
class IndexEntry:
    name: str
    status: IndexStatus
    spec: MetadataIndexSpec | None = None  # None for orphans


@dataclass(frozen=True)
class IndexPlan:
    entries: tuple[IndexEntry, ...]

    def with_status(self, *statuses: IndexStatus) -> list[IndexEntry]:
        return [e for e in self.entries if e.status in statuses]

    @property
    def in_sync(self) -> bool:
        return all(e.status is IndexStatus.PRESENT for e in self.entries)


@dataclass(frozen=True)
class ApplyResult:
    created: tuple[str, ...]
    dropped: tuple[str, ...]
    # Created, but the fingerprint COMMENT failed: UNMANAGED until rebuilt with --prune.
    unmarked: tuple[str, ...] = ()
    # Being built by another session when the run started: left alone.
    skipped: tuple[str, ...] = ()


class IndexSyncError(RuntimeError):
    """A sync run failed; `result` says what it had already done (nothing is rolled back)."""

    def __init__(self, message: str, result: ApplyResult | None = None) -> None:
        super().__init__(message)
        self.result = result or ApplyResult((), ())


@dataclass(frozen=True)
class CatalogIndex:
    name: str
    valid: bool
    comment: str | None
    building: bool = False  # a CREATE INDEX is running for it (pg_stat_progress_create_index)
    indexdef: str | None = None  # pg_get_indexdef; None = not compared


def _definition_hash(indexdef: str) -> str:
    return hashlib.sha256(indexdef.encode()).hexdigest()[:_FINGERPRINT_LENGTH]


def _identifier(name: str) -> str:
    if _SAFE_IDENTIFIER.fullmatch(name) is None:  # defense in depth: names come from validated
        raise ValueError(f"Unsafe index identifier {name!r}")  # fields, never from user input
    return name


def metadata_index_name(field: MetadataField) -> str:
    """Stable, length-safe name of the index backing `field` (`ix_chunks_meta_<name>`).

    A name that would exceed Postgres' 63 characters is truncated and suffixed with a digest of
    the full field name, so two long names never collide.
    """
    name = f"{METADATA_INDEX_PREFIX}{field.name}"
    if len(name) <= MAX_IDENTIFIER_LENGTH:
        return name
    digest = hashlib.sha256(field.name.encode()).hexdigest()[:_DIGEST_LENGTH]
    keep = MAX_IDENTIFIER_LENGTH - len(METADATA_INDEX_PREFIX) - _DIGEST_LENGTH - 1
    return f"{METADATA_INDEX_PREFIX}{field.name[:keep]}_{digest}"


def _render(field: MetadataField) -> str:
    expression = ordering_expression(metadata_spec(field))
    dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
    return str(expression.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))


def expected_metadata_indexes(fields: Iterable[MetadataField]) -> list[MetadataIndexSpec]:
    """The per-field indexes `indexed` fields need, sorted by name, one per distinct name.

    Fields declared by several profiles collapse into one index when their expressions agree
    (`int` / `float`, or `str` / `date`, share one); `ConflictingIndexError` when they do not.
    """
    specs: dict[str, MetadataIndexSpec] = {}
    for field in fields:
        if not field.indexed or field.is_list:
            continue
        spec = MetadataIndexSpec(metadata_index_name(field), field.name, _render(field))
        known = specs.setdefault(spec.name, spec)
        if known.expression != spec.expression:
            raise ConflictingIndexError(
                f"Metadata field {field.name!r} is declared with incompatible types: index "
                f"{spec.name} cannot serve both ({known.expression} vs {spec.expression})"
            )
    return [specs[name] for name in sorted(specs)]


def _status_of(spec: MetadataIndexSpec, row: CatalogIndex) -> IndexStatus:
    if not row.valid:
        return IndexStatus.BUILDING if row.building else IndexStatus.INVALID
    if row.comment is None or not row.comment.startswith(_COMMENT_PREFIX):
        return IndexStatus.UNMANAGED  # built by hand, or commented by someone else
    parsed = _COMMENT.fullmatch(row.comment)
    if parsed is None or parsed.group(1) != spec.fingerprint:
        return IndexStatus.STALE  # ours, but an older / different definition
    if row.indexdef is not None and parsed.group(2) != _definition_hash(row.indexdef):
        return IndexStatus.UNMANAGED  # the stored definition is not the one we created
    return IndexStatus.PRESENT


def diff_indexes(
    expected: Sequence[MetadataIndexSpec], existing: Iterable[CatalogIndex]
) -> IndexPlan:
    """Pure diff of the expected set against the catalog rows of the managed prefix."""
    present = {row.name: row for row in existing}
    entries: list[IndexEntry] = []
    for spec in expected:
        row = present.get(spec.name)
        status = IndexStatus.MISSING if row is None else _status_of(spec, row)
        entries.append(IndexEntry(spec.name, status, spec))
    wanted = {spec.name for spec in expected}
    for name in sorted(present):
        if name not in wanted:
            row = present[name]
            building = row.building and not row.valid
            entries.append(
                IndexEntry(name, IndexStatus.BUILDING if building else IndexStatus.ORPHAN)
            )
    return IndexPlan(tuple(entries))


_CATALOG = text(
    "SELECT c.relname AS name, i.indisvalid AS valid, "
    "obj_description(c.oid, 'pg_class') AS comment, "
    "pg_get_indexdef(c.oid) AS indexdef, "
    "EXISTS (SELECT 1 FROM pg_stat_progress_create_index p WHERE p.index_relid = c.oid) "
    "AS building "
    "FROM pg_index i "
    "JOIN pg_class c ON c.oid = i.indexrelid "
    "JOIN pg_class t ON t.oid = i.indrelid "
    "JOIN pg_namespace n ON n.oid = t.relnamespace "
    "WHERE t.relname = 'chunks' AND n.nspname = current_schema() "
    "AND starts_with(c.relname, :prefix) ORDER BY c.relname"
)

# Relations named like an index we are about to create that are NOT indexes of `chunks`.
_NAME_COLLISIONS = text(
    "SELECT c.relname FROM pg_class c "
    "JOIN pg_namespace n ON n.oid = c.relnamespace "
    "LEFT JOIN pg_index i ON i.indexrelid = c.oid "
    "LEFT JOIN pg_class t ON t.oid = i.indrelid "
    "WHERE n.nspname = current_schema() AND c.relname = ANY(:names) "
    "AND (c.relkind <> 'i' OR t.relname IS DISTINCT FROM 'chunks') ORDER BY c.relname"
)


async def _read_catalog(conn: AsyncConnection) -> list[CatalogIndex]:
    rows: Any = (await conn.execute(_CATALOG, {"prefix": METADATA_INDEX_PREFIX})).all()
    return [
        CatalogIndex(r.name, r.valid, r.comment, building=r.building, indexdef=r.indexdef)
        for r in rows
    ]


@dataclass
class _Progress:
    created: list[str] = dataclass_field(default_factory=list)
    dropped: list[str] = dataclass_field(default_factory=list)
    unmarked: list[str] = dataclass_field(default_factory=list)
    skipped: list[str] = dataclass_field(default_factory=list)

    def result(self) -> ApplyResult:
        return ApplyResult(
            tuple(self.created), tuple(self.dropped), tuple(self.unmarked), tuple(self.skipped)
        )


class MetadataIndexSync:
    """Plans and applies the metadata index set.

    `engine` reads the catalog. DDL runs on `ddl_engine`, which MUST be unpooled (the run's
    session settings and its advisory lock die with the connection, nothing to reset), in
    AUTOCOMMIT because `CREATE INDEX CONCURRENTLY` cannot run inside a transaction, with the
    statement timeout lifted: building an index legitimately outlives the request-oriented one.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        ddl_engine: AsyncEngine,
        *,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(ddl_engine.pool, NullPool):
            raise ValueError("ddl_engine must be unpooled (create_unpooled_engine)")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be greater than 0")
        self._engine = engine
        self._ddl_engine = ddl_engine
        self._lock_timeout = lock_timeout_seconds

    async def plan(self, fields: Iterable[MetadataField]) -> IndexPlan:
        expected = expected_metadata_indexes(fields)
        async with self._engine.connect() as conn:
            return diff_indexes(expected, await _read_catalog(conn))

    async def apply(
        self,
        plan: IndexPlan,
        *,
        prune: bool = False,
        lock_timeout_seconds: float | None = None,
    ) -> ApplyResult:
        """Create the missing indexes; with `prune` also drop (and rebuild) the unusable ones.

        The run holds the sync advisory lock (waiting at most `lock_timeout_seconds`) and diffs
        the catalog again under it, so a plan computed before a concurrent run is never applied
        stale. Without `prune` nothing is ever dropped: stale / unmanaged / invalid indexes are
        left as they are and orphans untouched. An index that is still building is never touched.
        A failure raises `IndexSyncError` carrying what the run had already done.
        """
        specs = [e.spec for e in plan.entries if e.spec is not None]
        progress = _Progress()
        timeout = self._lock_timeout if lock_timeout_seconds is None else lock_timeout_seconds
        async with self._ddl_engine.connect() as conn:
            conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            await self._take_lock(conn, timeout)
            try:
                await self._run(conn, specs, prune, progress)
            except IndexSyncError as exc:
                exc.result = progress.result()
                raise
            except Exception as exc:
                raise IndexSyncError(f"{type(exc).__name__}: {exc}", progress.result()) from exc
        return progress.result()

    async def _take_lock(self, conn: AsyncConnection, timeout_seconds: float) -> None:
        """Session-level advisory lock; released when the (unpooled) connection closes."""
        millis = str(max(1, int(timeout_seconds * 1000)))
        high, low = index_sync_lock_keys()
        try:
            await conn.execute(text("SELECT set_config('lock_timeout', :v, false)"), {"v": millis})
            await conn.execute(
                text("SELECT pg_advisory_lock(:high, :low)"), {"high": high, "low": low}
            )
        except DBAPIError as exc:
            raise IndexSyncError(
                f"Another index sync is running (no lock after {timeout_seconds:g}s): {exc.orig}"
            ) from exc
        # The index builds wait for old transactions by design: only the sync lock is bounded.
        await conn.execute(text("SELECT set_config('lock_timeout', '0', false)"))
        await conn.execute(text("SELECT set_config('statement_timeout', '0', false)"))

    async def _run(
        self,
        conn: AsyncConnection,
        specs: Sequence[MetadataIndexSpec],
        prune: bool,
        progress: _Progress,
    ) -> None:
        plan = diff_indexes(specs, await _read_catalog(conn))
        progress.skipped.extend(e.name for e in plan.with_status(IndexStatus.BUILDING))
        unusable = (
            IndexStatus.STALE,
            IndexStatus.UNMANAGED,
            IndexStatus.INVALID,
        )
        to_create = plan.with_status(IndexStatus.MISSING, *(unusable if prune else ()))
        await self._assert_names_free(conn, [e.name for e in plan.with_status(IndexStatus.MISSING)])
        if prune:
            for entry in plan.with_status(IndexStatus.ORPHAN, *unusable):
                await conn.execute(
                    text(f'DROP INDEX CONCURRENTLY IF EXISTS "{_identifier(entry.name)}"')
                )
                progress.dropped.append(entry.name)
        for entry in to_create:
            if entry.spec is not None:  # only orphans lack a spec, and they are never created
                await self._create(conn, entry.spec, progress)

    async def _assert_names_free(self, conn: AsyncConnection, names: list[str]) -> None:
        """`IF NOT EXISTS` would skip silently over a table / view / foreign index of that name."""
        if not names:
            return
        taken = (await conn.execute(_NAME_COLLISIONS, {"names": names})).scalars().all()
        if taken:
            raise IndexSyncError(
                f"Cannot create {', '.join(taken)}: the name is used by another relation "
                "(not an index of chunks); rename or drop it first"
            )

    async def _create(
        self, conn: AsyncConnection, spec: MetadataIndexSpec, progress: _Progress
    ) -> None:
        await conn.execute(text(spec.create_sql))
        row = next((r for r in await _read_catalog(conn) if r.name == spec.name), None)
        if row is None or not row.valid or row.indexdef is None:
            raise IndexSyncError(f"Index {spec.name} is missing or invalid after its build")
        progress.created.append(spec.name)
        try:
            await conn.execute(
                text(
                    f'COMMENT ON INDEX "{_identifier(spec.name)}" '
                    f"IS '{spec.comment_for(row.indexdef)}'"
                )
            )
        except DBAPIError:
            logger.warning(
                "Index %s was created but its fingerprint comment failed; it will show as "
                "unmanaged until rebuilt with --prune",
                spec.name,
                exc_info=True,
            )
            progress.unmarked.append(spec.name)


_PROBLEM_STATUSES = (
    IndexStatus.MISSING,
    IndexStatus.INVALID,
    IndexStatus.STALE,
    IndexStatus.UNMANAGED,
)


async def metadata_index_problems(
    engine: AsyncEngine, fields: Iterable[MetadataField]
) -> list[IndexEntry]:
    """Expected indexes that are missing, unusable, stale or unmanaged (what a boot should warn).

    Declaring `indexed` in a profile creates nothing: run `python -m agintis.cli.indexes --apply`.
    An index still building is not a problem.
    """
    expected = expected_metadata_indexes(fields)
    async with engine.connect() as conn:
        plan = diff_indexes(expected, await _read_catalog(conn))
    return plan.with_status(*_PROBLEM_STATUSES)
