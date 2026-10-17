"""System implementation of the Clock port."""

from datetime import UTC, datetime

from agintis.shared_kernel.ports import Clock


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)
