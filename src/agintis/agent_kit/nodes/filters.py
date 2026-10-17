"""Query-side filter extraction: an LLM fills the metadata schema of ONE RagProfile.

The structured-output schema is built from the profile's declared, filterable metadata fields,
so nothing here knows any case: a field is `eq` (scalar, keyword and list fields: a list
field matches when it contains the value) or `range` (a `date`, or an `indexed` int/float: two
inclusive bounds `<name>_gte` / `<name>_lte`).

OpenAI strict `json_schema` rejects open objects and wants every property required: the
schema has only explicit typed properties, all required and nullable (null = "not stated").
`extra="forbid"` makes it `additionalProperties: false`.

The model output is never trusted. Each value is conformed with the knowledge module's own
`conform_metadata` (the code that types values at ingest), so a filter compares exactly what is
stored; whatever does not conform is dropped with a debug log. Filters are hints: nothing here
fails the run on bad values.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, Strict, create_model

from agintis.agent_kit.nodes.prompts import DEFAULT_EXTRACT_FILTERS_PROMPT
from agintis.agent_kit.nodes.state import FilterClause
from agintis.agent_kit.nodes.structured import bind_structured, invoke_structured
from agintis.modules.knowledge.api import (
    InvalidMetadataError,
    MetadataField,
    RagProfile,
    conform_metadata,
)
from agintis.platform.llm import ChatModels, ChatRole
from agintis.shared_kernel.criteria import AllOf, Filter, FilterExpr

logger = logging.getLogger(__name__)

GTE_SUFFIX = "_gte"
LTE_SUFFIX = "_lte"
_SCHEMA_NAME = "QueryFilters"
_SCHEMA_DOC = (
    "Metadata filters stated in the question. A field the question does not explicitly "
    "state MUST be null."
)
# Strict types: a lax int/float field would coerce `True` (and numeric strings) before the node
# could look at them; a value that does not fit makes the WHOLE object fail to validate, and the
# extractor then salvages the raw JSON field by field (see `FilterExtractor.extract`).
_PYTHON_TYPES: Mapping[str, Any] = {
    "str": Annotated[str, Strict()],
    "int": Annotated[int, Strict()],
    "float": Annotated[float, Strict()],
    "date": Annotated[str, Strict()],  # ISO date text
    "list[str]": Annotated[str, Strict()],  # one element of the list
    "list[int]": Annotated[int, Strict()],
}

type FilterKind = Literal["eq", "range"]


@dataclass(frozen=True, slots=True)
class FilterFieldPlan:
    """How one filterable field is asked for: its metadata declaration, kind and hint text."""

    field: MetadataField
    kind: FilterKind
    description: str | None = None

    @property
    def name(self) -> str:
        return self.field.name

    @property
    def scalar_type(self) -> str:
        """The metadata type of ONE value (the element type for list fields)."""
        return self.field.type.removeprefix("list[").removesuffix("]")


def field_kind(declared: MetadataField) -> FilterKind:
    if declared.is_list:
        return "eq"
    if declared.type == "date" or (declared.type in ("int", "float") and declared.indexed):
        return "range"
    return "eq"


def plan_filter_fields(
    profile: RagProfile, descriptions: Mapping[str, str] | None = None
) -> tuple[FilterFieldPlan, ...]:
    """The fields the extractor may fill, in profile order.

    `descriptions` (field name -> hint for the model) is also an allowlist: only those fields
    are asked for, and each must be a filterable field the profile declares (ValueError
    otherwise). `None` = every declared filterable field, without hints.
    """
    filterable = {f.name: f for f in profile.metadata_fields if f.filterable}
    if descriptions is not None:
        unknown = sorted(set(descriptions) - set(filterable))
        if unknown:
            raise ValueError(
                f"profile {profile.name!r} has no filterable metadata field(s) {unknown} "
                f"(filterable: {sorted(filterable)})"
            )
        chosen = [f for f in filterable.values() if f.name in descriptions]
    else:
        chosen = list(filterable.values())
    plans = tuple(
        FilterFieldPlan(f, field_kind(f), (descriptions or {}).get(f.name)) for f in chosen
    )
    names = [name for plan in plans for name in _property_names(plan)]
    if len(set(names)) != len(names):
        raise ValueError(f"profile {profile.name!r}: filter field names collide: {sorted(names)}")
    return plans


def _property_names(plan: FilterFieldPlan) -> tuple[str, ...]:
    if plan.kind == "eq":
        return (plan.name,)
    return (plan.name + GTE_SUFFIX, plan.name + LTE_SUFFIX)


def _describe(plan: FilterFieldPlan, bound: str | None) -> str:
    kind_hint = {
        "date": "ISO date YYYY-MM-DD",
        "int": "integer",
        "float": "number",
        "str": "text",
    }[plan.scalar_type]
    parts = [plan.description or f"Value of the metadata field '{plan.name}'"]
    if bound == "gte":
        parts.append(f"Inclusive LOWER bound ({kind_hint}).")
    elif bound == "lte":
        parts.append(f"Inclusive UPPER bound ({kind_hint}).")
    else:
        parts.append(f"Exactly one value ({kind_hint}).")
    return " ".join(parts)


def build_filter_schema(plans: tuple[FilterFieldPlan, ...]) -> type[BaseModel]:
    """Pydantic model with one required, nullable property per eq field and two per range field.

    Attributes get safe internal names (`f0`, `f1`, ...) and the metadata name is the ALIAS, so
    the wire name is exact while a metadata field called `model_config`, `model_dump` or `_x`
    can never clash with pydantic. Parse and dump BY ALIAS (`model_dump(by_alias=True)`).
    """
    definitions: dict[str, Any] = {}
    for plan in plans:
        python_type = _PYTHON_TYPES[plan.field.type]
        if plan.kind == "eq":
            bounds: tuple[tuple[str, str | None], ...] = ((plan.name, None),)
        else:
            bounds = ((plan.name + GTE_SUFFIX, "gte"), (plan.name + LTE_SUFFIX, "lte"))
        for wire_name, bound in bounds:
            definitions[f"f{len(definitions)}"] = (
                python_type | None,
                Field(..., alias=wire_name, description=_describe(plan, bound)),
            )
    return create_model(
        _SCHEMA_NAME,
        __config__=ConfigDict(extra="forbid"),
        __doc__=_SCHEMA_DOC,
        **definitions,
    )


def _conform(plan: FilterFieldPlan, value: object) -> str | int | float | None:
    """`value` typed exactly as `plan.field` is stored, or None (logged) if it does not conform."""
    if value is None or isinstance(value, bool):
        return None
    candidate: Any = value.strip() if isinstance(value, str) else value
    if candidate == "":
        return None
    declared = plan.field
    try:
        conformed = conform_metadata(
            {declared.name: [candidate] if declared.is_list else candidate}, [declared]
        )
    except InvalidMetadataError as exc:
        logger.debug("Dropped filter value for %r: %s", declared.name, exc)
        return None
    typed = conformed[declared.name]
    if isinstance(typed, tuple):  # a list field: the single element asked for
        return typed[0] if typed else None
    return typed


def clauses_from_values(
    plans: tuple[FilterFieldPlan, ...], values: Mapping[str, object]
) -> list[FilterClause]:
    """Typed clauses for the non-null, conforming values; invalid ones are dropped."""
    clauses: list[FilterClause] = []
    for plan in plans:
        if plan.kind == "eq":
            if (value := _conform(plan, values.get(plan.name))) is not None:
                clauses.append(FilterClause(field=plan.name, op="eq", value=value))
            continue
        lower = _conform(plan, values.get(plan.name + GTE_SUFFIX))
        upper = _conform(plan, values.get(plan.name + LTE_SUFFIX))
        if lower is not None and upper is not None and lower > upper:  # type: ignore[operator]
            logger.debug("Dropped inverted range for %r: %r > %r", plan.name, lower, upper)
            continue
        if lower is not None:
            clauses.append(FilterClause(field=plan.name, op="gte", value=lower))
        if upper is not None:
            clauses.append(FilterClause(field=plan.name, op="lte", value=upper))
    return clauses


def normalize_clauses(clauses: list[FilterClause]) -> list[FilterClause]:
    """A `gte` and a `lte` on the same field with the same value are one `eq` (a point range)."""
    lower = {c["field"]: c["value"] for c in clauses if c["op"] == "gte"}
    upper = {c["field"]: c["value"] for c in clauses if c["op"] == "lte"}
    points = {f for f, v in lower.items() if f in upper and upper[f] == v}
    normalized: list[FilterClause] = []
    for clause in clauses:
        if clause["field"] not in points:
            normalized.append(clause)
        elif clause["op"] == "gte":
            normalized.append(FilterClause(field=clause["field"], op="eq", value=clause["value"]))
    return normalized


def clauses_to_filter(clauses: list[FilterClause]) -> FilterExpr | None:
    """The knowledge `FilterExpr` (AND of every normalized clause); None when nothing filters."""
    parts: list[FilterExpr] = []
    for clause in normalize_clauses(clauses):
        match clause["op"]:
            case "eq":
                parts.append(Filter.eq(clause["field"], clause["value"]))
            case "gte":
                parts.append(Filter.gte(clause["field"], clause["value"]))
            case "lte":
                parts.append(Filter.lte(clause["field"], clause["value"]))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else AllOf(*parts)


STRONG_FIELD_TYPES = frozenset({"int", "float", "date", "list[int]"})


def strong_clauses(clauses: list[FilterClause], profile: RagProfile) -> list[FilterClause]:
    """Clauses on numeric (int/float) and date fields: identifiers and dates the question states
    precisely and a model rarely invents, unlike free-text fields (kind, topic, system...)."""
    types = {f.name: f.type for f in profile.metadata_fields}
    return [c for c in clauses if types.get(c["field"]) in STRONG_FIELD_TYPES]


class FilterExtractor:
    """Asks the analyzer model for the filters a question states (no graph, no state).

    Shared by `ExtractFiltersNode` and the eval CLI, so a measurement runs the very same code
    as the agent. Without filterable fields (`enabled` is False) the model is never called.
    """

    def __init__(
        self,
        chat_models: ChatModels,
        profile: RagProfile,
        descriptions: Mapping[str, str] | None = None,
        prompt: str = DEFAULT_EXTRACT_FILTERS_PROMPT,
    ) -> None:
        self._chat_models = chat_models
        self._prompt = prompt
        self._plans = plan_filter_fields(profile, descriptions)
        self._schema = build_filter_schema(self._plans) if self._plans else None
        self._structured: Runnable[Any, Any] | None = None  # bound lazily, once

    @property
    def enabled(self) -> bool:
        return self._schema is not None

    @property
    def plans(self) -> tuple[FilterFieldPlan, ...]:
        return self._plans

    @property
    def schema(self) -> type[BaseModel] | None:
        return self._schema

    async def extract(self, query: str) -> list[FilterClause]:
        """Typed clauses for what `query` states. Each field is validated on its own: a value that
        does not fit one field drops that field only. Unparseable output yields `[]`; provider
        and network errors propagate (see `structured.py`)."""
        if self._schema is None or not query.strip():
            return []
        if self._structured is None:
            self._structured = bind_structured(
                self._chat_models.get(ChatRole.ANALYZER), self._schema
            )
        outcome = await invoke_structured(
            self._structured, [SystemMessage(self._prompt), HumanMessage(query)]
        )
        if outcome.parsed is not None:
            values: Mapping[str, object] = outcome.parsed.model_dump(by_alias=True)
        else:
            values = outcome.raw_values or {}
        return clauses_from_values(self._plans, values)
