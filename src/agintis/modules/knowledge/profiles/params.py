"""Pydantic plumbing shared by strategy parameter schemas (chunkers, cleaners).

The domain keeps parameters as immutable nested mappings; each strategy key owns a
`ParamsModel` subclass describing what it accepts. Validation runs at boot (`validate_profiles`)
and again, cheaply, when the strategy executes. Unknown keys are errors everywhere.
"""

from collections.abc import Mapping
from typing import Annotated

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, ValidationError

from agintis.modules.knowledge.application import InvalidProfileError
from agintis.modules.knowledge.domain import ParamValue, thaw_params
from agintis.modules.knowledge.profiles.regex import check_profile_regex


class ParamsModel(BaseModel):
    """Base of every strategy params schema: strict types, no unknown keys, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _flag(value: object) -> object:
    # Booleans or the 0/1 integers (profiles written by hand); never strings like "no".
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return value


Flag = Annotated[bool, BeforeValidator(_flag)]

# A regular expression kept as text (params stay JSON-like and fingerprintable) but compiled
# and checked once at validation time to fail fast.
Pattern = Annotated[str, AfterValidator(check_profile_regex)]


def describe_errors(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )


def validate_params[M: ParamsModel](
    schema: type[M], params: Mapping[str, ParamValue], *, where: str
) -> M:
    """Validate `params` against `schema`; `InvalidProfileError` names `where` and every issue."""
    try:
        return schema.model_validate(thaw_params(params))
    except ValidationError as exc:
        raise InvalidProfileError(f"{where}: {describe_errors(exc)}") from exc
