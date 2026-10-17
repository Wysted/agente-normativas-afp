"""Shared SQLAlchemy MetaData: every module's tables register here (Alembic reads it)."""

from sqlalchemy import MetaData

# Deterministic constraint names keep migrations stable across environments.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

# Tables owned by other libraries that live in the same database. Alembic must neither
# autogenerate nor drop them (LangGraph's AsyncPostgresSaver creates and migrates its own).
EXTERNAL_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)
