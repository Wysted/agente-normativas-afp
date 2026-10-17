from datetime import UTC, datetime, timedelta

from agintis.shared_kernel.ports import Clock


class FixedClock(Clock):
    def __init__(self, instant: datetime | None = None) -> None:
        self._instant = instant or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._instant


class StepClock(Clock):
    """A clock the test moves forward."""

    def __init__(self) -> None:
        self.instant = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.instant

    def advance(self, **delta: float) -> None:
        self.instant += timedelta(**delta)
