"""Pick the profile a document should be ingested with (`ProfileSelector` implementation).

Every rule of every `active` profile is evaluated (a `draft` profile loads and validates but is
never selected). Among the rules that match, the highest `priority` wins, regardless of file or
declaration order: a narrow rule outranks the broad one it overlaps. If rules of more than one
profile tie at the top priority the document is NOT routed (reason `ambiguous: a, b`): a silent
order-dependent choice is how documents end up in the wrong profile. The decision uses the
document metadata the caller supplied at ingest (a crawler's index row, say), so it happens
before any download or parsing.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from agintis.modules.knowledge.application import (
    DocumentMetadata,
    DocumentValue,
    ProfileSelection,
    ProfileSelector,
)
from agintis.modules.knowledge.domain import RagProfile, RoutingCondition, RoutingRule
from agintis.modules.knowledge.profiles.regex import ProfileRegex, compile_profile_regex


def _candidates(value: DocumentValue) -> list[str]:
    """The texts a condition tests: the value, or each item of a list-valued field."""
    if value is None:
        return []
    items = value if isinstance(value, tuple) else (value,)
    return [str(item) for item in items if item is not None]


@dataclass(frozen=True)
class _CompiledCondition:
    field: str
    values: frozenset[str]
    regex: ProfileRegex | None

    @classmethod
    def of(cls, condition: RoutingCondition) -> "_CompiledCondition":
        regex = compile_profile_regex(condition.pattern) if condition.pattern is not None else None
        return cls(condition.field, frozenset(condition.any_of), regex)

    def holds(self, document: DocumentMetadata) -> bool:
        texts = _candidates(document.get(self.field))
        if self.regex is not None:
            return any(self.regex.search(text) for text in texts)
        return any(text in self.values for text in texts)


@dataclass(frozen=True)
class _CompiledRule:
    profile: RagProfile
    match: tuple[_CompiledCondition, ...]
    exclude: tuple[_CompiledCondition, ...]
    priority: int

    @classmethod
    def of(cls, profile: RagProfile, rule: RoutingRule) -> "_CompiledRule":
        return cls(
            profile,
            tuple(_CompiledCondition.of(c) for c in rule.match),
            tuple(_CompiledCondition.of(c) for c in rule.exclude),
            rule.priority,
        )

    def accepts(self, document: DocumentMetadata) -> bool:
        return all(c.holds(document) for c in self.match) and not any(
            c.holds(document) for c in self.exclude
        )


class ProfileRouter(ProfileSelector):
    def __init__(self, profiles: Iterable[RagProfile]) -> None:
        self._rules = [
            _CompiledRule.of(profile, rule)
            for profile in profiles
            if profile.status == "active"
            for rule in profile.routing
        ]

    def select(self, document: DocumentMetadata) -> ProfileSelection:
        matched = [rule for rule in self._rules if rule.accepts(document)]
        if not matched:
            return ProfileSelection(
                None, "no active profile has a routing rule matching the document"
            )
        top = max(rule.priority for rule in matched)
        winners = {rule.profile.name: rule.profile for rule in matched if rule.priority == top}
        if len(winners) > 1:
            return ProfileSelection(None, f"ambiguous: {', '.join(sorted(winners))}")
        (profile,) = winners.values()
        return ProfileSelection(profile, f"routed to {profile.name!r}")
