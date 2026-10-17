"""Registry of named RagProfiles."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from agintis.modules.knowledge.application import (
    NO_RERANKER_KEY,
    Extractor,
    InvalidProfileError,
    UnknownStrategyError,
)
from agintis.modules.knowledge.domain import (
    DENSE_RETRIEVAL_MODE,
    MetadataField,
    ParamValue,
    RagProfile,
)
from agintis.modules.knowledge.profiles.params import ParamsModel, validate_params
from agintis.shared_kernel.errors import DomainError, NotFoundError

# 512 tokens fits bge-m3 comfortably; the structured chunker respects headings, no overlap.
# "simple" + "structured" are the lightweight (no Docling) strategies; reranking is off by default.
# include_heading is explicit so the default behavior is part of the profile fingerprint.
DEFAULT_PROFILE = RagProfile(
    name="default",
    top_k=5,
    chunk_max_tokens=512,
    parser_key="simple",
    chunker_key="structured",
    chunker_params={"include_heading": True},
    reranker_key="none",
)


class RagProfileRegistry:
    def __init__(self, profiles: Iterable[RagProfile] = (DEFAULT_PROFILE,)) -> None:
        self._profiles: dict[str, RagProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: RagProfile) -> None:
        if profile.name in self._profiles:
            raise DomainError(f"RagProfile {profile.name!r} is already registered")
        self._profiles[profile.name] = profile

    def get(self, name: str) -> RagProfile:
        try:
            return self._profiles[name]
        except KeyError:
            raise NotFoundError(f"RagProfile {name!r} not found") from None

    def all(self) -> tuple[RagProfile, ...]:
        return tuple(self._profiles.values())


def _param_problems(
    label: str,
    schema: type[ParamsModel] | None,
    params: Mapping[str, ParamValue],
) -> list[str]:
    return _validated(label, schema, params)[1]


def _validated(
    label: str,
    schema: type[ParamsModel] | None,
    params: Mapping[str, ParamValue],
) -> tuple[ParamsModel | None, list[str]]:
    if schema is None:
        return None, ([f"{label}: takes no parameters"] if params else [])
    try:
        return validate_params(schema, params, where=label), []
    except InvalidProfileError as exc:
        return None, [str(exc)]


def _group_problems(
    profile: RagProfile, chunker_params: ParamsModel | None, declared: bool
) -> list[str]:
    """`group:<name>` sources must name a regex group of the profile's chunker.

    `chunker_params` None with `declared` = the params were invalid (reported elsewhere).
    """
    sources = [
        (f.name, f.source.removeprefix("group:"))
        for f in profile.metadata_fields
        if f.source and f.source.startswith("group:")
    ]
    if not sources or (chunker_params is None and declared):
        return []
    group_names = getattr(chunker_params, "group_names", None)
    available = group_names() if callable(group_names) else frozenset()
    return [
        f"profile {profile.name!r}: metadata field {name!r} reads group {group!r}, which "
        f"chunker {profile.chunker_key!r} does not produce"
        for name, group in sources
        if group not in available
    ]


def _field_problems(
    profile: RagProfile,
    declared: MetadataField,
    extractors: Mapping[str, Extractor],
    extractor_schemas: Mapping[str, type[ParamsModel]],
    transform_schemas: Mapping[str, type[ParamsModel]],
    readable: frozenset[str] | None,
) -> list[str]:
    """`readable`: names an extractor may read (chunker groups + fields declared earlier);
    None = the chunker's groups are unknown (invalid params, reported elsewhere)."""
    where = f"profile {profile.name!r}: metadata field {declared.name!r}"
    problems: list[str] = []
    if declared.source and declared.source.startswith("extractor:"):
        name, _, selector = declared.source.removeprefix("extractor:").partition(".")
        named = profile.extractors.get(name)
        if named is not None and declared.params:
            problems.append(f"{where}: params belong on the named extractor {name!r}, not here")
        key, params = (named.key, named.params) if named is not None else (name, declared.params)
        issues: list[str] = []
        if named is None:  # named extractors are validated once, with the profile's own list
            _, issues = _validated(
                f"{where} extractor {key!r} params", extractor_schemas.get(key), params
            )
            problems += issues
        extractor = extractors.get(key)
        if extractor is not None and not issues:
            selectors = extractor.selectors_for(params)
            if selector and selector not in selectors:
                problems.append(
                    f"{where}: extractor {key!r} has no selector {selector!r} "
                    f"(available: {sorted(selectors) or 'none'})"
                )
            if selectors and not selector:
                problems.append(f"{where}: extractor {key!r} needs a selector {sorted(selectors)}")
            if readable is not None and (
                unknown := sorted(extractor.references(params) - readable)
            ):
                problems.append(
                    f"{where}: extractor {key!r} reads {unknown}, which are neither chunker "
                    "groups nor metadata fields declared before this one"
                )
    for position, spec in enumerate(declared.transforms, start=1):
        problems += _param_problems(
            f"{where} transform #{position} {spec.key!r} params",
            transform_schemas.get(spec.key),
            spec.params,
        )
    return problems


@dataclass(frozen=True)
class CleanerOrdering:
    """Which cleaner keys constrain the pipeline order.

    `rebuilds_text`: cleaners that rebuild the sections from the raw source (HTML main content):
    whatever text edits ran before them would be silently lost, so only `read_only` cleaners
    (they leave the sections alone, e.g. `html_collect`) may precede one.
    """

    rebuilds_text: frozenset[str] = frozenset()
    read_only: frozenset[str] = frozenset()


def _ordering_problems(profile: RagProfile, ordering: CleanerOrdering) -> list[str]:
    problems: list[str] = []
    for position, spec in enumerate(profile.cleaners):
        if spec.key not in ordering.rebuilds_text:
            continue
        culprits = [
            f"#{index} {earlier.key!r}"
            for index, earlier in enumerate(profile.cleaners[:position], start=1)
            if earlier.key not in ordering.read_only
        ]
        if culprits:
            problems.append(
                f"profile {profile.name!r}: cleaner #{position + 1} {spec.key!r} rebuilds the text "
                f"from the source, so it must be the first text-producing cleaner "
                f"(only {sorted(ordering.read_only)} may precede it), but it comes after "
                + ", ".join(culprits)
            )
    return problems


def _metadata_strategy_refs(
    profile: RagProfile,
    extractor_schemas: Mapping[str, type[ParamsModel]] | None,
    transform_schemas: Mapping[str, type[ParamsModel]] | None,
) -> list[tuple[str, str, Iterable[str]]]:
    refs: list[tuple[str, str, Iterable[str]]] = []
    if extractor_schemas is not None:
        refs.extend(
            ("extractor", s.key, extractor_schemas.keys()) for s in profile.extractors.values()
        )
    for declared in profile.metadata_fields:
        source = declared.source or ""
        if extractor_schemas is not None and source.startswith("extractor:"):
            name = source.removeprefix("extractor:").partition(".")[0]
            if name not in profile.extractors:
                refs.append(("extractor", name, extractor_schemas.keys()))
        if transform_schemas is not None:
            refs.extend(("transform", t.key, transform_schemas.keys()) for t in declared.transforms)
    return refs


def validate_profiles(
    registry: RagProfileRegistry,
    parser_keys: Iterable[str],
    chunker_keys: Iterable[str],
    reranker_keys: Iterable[str] = (NO_RERANKER_KEY,),
    *,
    chunker_schemas: Mapping[str, type[ParamsModel]] | None = None,
    chunker_min_tokens: Mapping[str, int] | None = None,
    cleaner_schemas: Mapping[str, type[ParamsModel]] | None = None,
    cleaner_ordering: CleanerOrdering | None = None,
    extractors: Mapping[str, Extractor] | None = None,
    extractor_schemas: Mapping[str, type[ParamsModel]] | None = None,
    transform_schemas: Mapping[str, type[ParamsModel]] | None = None,
    retrieval_mode_keys: Iterable[str] = (DENSE_RETRIEVAL_MODE,),
) -> None:
    """Fail fast (call from bootstrap) when a profile cannot run.

    1. Every strategy key must be registered (`UnknownStrategyError`, all problems at once):
       parser, chunker, reranker, retrieval mode and each cleaner (`cleaner_schemas` keys are
       the registered cleaners; none given = none registered).
    2. When `chunker_schemas` is given, each chunker's `chunker_params` and each cleaner's
       `params` must satisfy that key's Pydantic schema (`InvalidProfileError`, unknown keys
       included). A registered key without a schema accepts no parameters.
    3. When `cleaner_ordering` is given, a cleaner that rebuilds the text from the source may
       only be preceded by read-only cleaners (`InvalidProfileError`).
    4. When `extractor_schemas` / `transform_schemas` are given, every metadata field's
       extractor (key, params, selector) and transforms (key, params) must exist and validate
       (`extractors` are only needed to check selectors and the names they read: chunker regex
       groups and metadata fields declared earlier). With `chunker_schemas`, every
       `group:<name>` source must name a regex group the profile's chunker declares.
    5. When `chunker_min_tokens` is given, `chunk_max_tokens` must reach the chunker's minimum
       (below it the heading prefix leaves no room for the body).
    """
    parsers, chunkers, rerankers = set(parser_keys), set(chunker_keys), set(reranker_keys)
    modes = set(retrieval_mode_keys)
    cleaners = cleaner_schemas if cleaner_schemas is not None else {}
    unknown = [
        f"profile {p.name!r}: {kind} {key!r}"
        for p in registry.all()
        for kind, key, known in _metadata_strategy_refs(p, extractor_schemas, transform_schemas)
        if key not in known
    ]
    unknown += [
        f"profile {p.name!r}: {kind} {key!r}"
        for p in registry.all()
        for kind, key, known in (
            ("parser", p.parser_key, parsers),
            ("chunker", p.chunker_key, chunkers),
            ("reranker", p.reranker_key, rerankers),
            ("retrieval mode", p.retrieval_mode, modes),
            *(("cleaner", c.key, cleaners.keys()) for c in p.cleaners),
        )
        if key not in known
    ]
    if unknown:
        raise UnknownStrategyError("Unregistered strategies: " + "; ".join(unknown))

    problems: list[str] = []
    for p in registry.all():
        minimum = (chunker_min_tokens or {}).get(p.chunker_key, 1)
        if p.chunk_max_tokens < minimum:
            problems.append(
                f"profile {p.name!r}: chunk_max_tokens {p.chunk_max_tokens} is below the "
                f"minimum {minimum} of chunker {p.chunker_key!r}"
            )
        group_names: frozenset[str] | None = None  # unknown without chunker schemas
        if chunker_schemas is not None:
            typed, issues = _validated(
                f"profile {p.name!r}: chunker {p.chunker_key!r} params",
                chunker_schemas.get(p.chunker_key),
                p.chunker_params,
            )
            problems += issues
            problems += _group_problems(p, typed, declared=bool(issues))
            names = getattr(typed, "group_names", None)
            group_names = None if issues else (names() if callable(names) else frozenset())
        if extractor_schemas is not None or transform_schemas is not None:
            for name, named in p.extractors.items():
                problems += _param_problems(
                    f"profile {p.name!r}: extractor {name!r} ({named.key!r}) params",
                    (extractor_schemas or {}).get(named.key),
                    named.params,
                )
            earlier: set[str] = set()
            for declared in p.metadata_fields:
                problems += _field_problems(
                    p,
                    declared,
                    extractors or {},
                    extractor_schemas or {},
                    transform_schemas or {},
                    None if group_names is None else group_names | earlier,
                )
                earlier.add(declared.name)
        for position, spec in enumerate(p.cleaners, start=1):
            problems += _param_problems(
                f"profile {p.name!r}: cleaner #{position} {spec.key!r} params",
                cleaners.get(spec.key),
                spec.params,
            )
        if cleaner_ordering is not None:
            problems += _ordering_problems(p, cleaner_ordering)
    if problems:
        raise InvalidProfileError("Invalid strategy parameters: " + "; ".join(problems))
