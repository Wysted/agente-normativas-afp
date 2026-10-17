"""Small Unit of Work around one AsyncSession and one transaction."""

from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class UnitOfWork:
    """`async with UnitOfWork(factory) as uow:` -> `uow.session` inside ONE transaction.

    Commits when the block ends normally and rolls back on any exception (cancellation
    included); the session is always closed. One UoW per task: AsyncSession is not
    concurrency-safe. Repositories take the factory and open a UoW per operation; callers that
    need several operations to be atomic can share `uow.session` with lower-level helpers.
    """

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("UnitOfWork is not active; use it as an async context manager")
        return self._session

    async def __aenter__(self) -> Self:
        self._session = self._factory()
        await self._session.begin()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        session = self.session
        try:
            if exc_type is None:
                await session.commit()
            else:
                await session.rollback()
        finally:
            await session.close()
            self._session = None
