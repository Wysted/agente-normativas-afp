"""Database platform: async engine, sessions, Unit of Work, Criteria->SQL converter."""

from agintis.platform.db.checks import DatabaseCheckError, check_database, parse_version
from agintis.platform.db.criteria import (
    ColumnField,
    FieldSpec,
    JsonbField,
    apply_criteria,
    compile_filter,
    compile_order,
    jsonb_typed_expression,
    ordering_expression,
)
from agintis.platform.db.engine import create_engine, create_session_factory, create_unpooled_engine
from agintis.platform.db.metadata import EXTERNAL_TABLES, metadata
from agintis.platform.db.unit_of_work import UnitOfWork

__all__ = [
    "EXTERNAL_TABLES",
    "ColumnField",
    "DatabaseCheckError",
    "FieldSpec",
    "JsonbField",
    "UnitOfWork",
    "apply_criteria",
    "check_database",
    "compile_filter",
    "compile_order",
    "create_engine",
    "create_session_factory",
    "create_unpooled_engine",
    "jsonb_typed_expression",
    "metadata",
    "ordering_expression",
    "parse_version",
]
