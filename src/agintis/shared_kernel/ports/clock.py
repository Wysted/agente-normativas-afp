"""Clock port: lets use cases get the current time without touching the system clock."""

from abc import ABC, abstractmethod
from datetime import datetime


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime:
        """Return the current timezone-aware datetime in UTC."""
