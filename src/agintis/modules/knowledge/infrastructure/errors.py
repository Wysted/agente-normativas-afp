"""Classification of the transient failures retrieval can hit (client-safe, retryable)."""

import psycopg
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from agintis.modules.knowledge.infrastructure.tei import TeiUnavailableError
from agintis.shared_kernel.errors import FailureInfo

RETRIEVAL_UNAVAILABLE_CODE = "retrieval_unavailable"
_RETRIEVAL_UNAVAILABLE = FailureInfo(
    RETRIEVAL_UNAVAILABLE_CODE,
    "The knowledge base is temporarily unavailable; retry shortly.",
    True,
)


class RetrievalErrorClassifier:
    """`classify(exc)` -> a retryable `retrieval_unavailable` for outages, None for anything else.

    Outages: the embedding service stayed unreachable/overloaded (`TeiUnavailableError`), the
    database refused or dropped the connection, a statement hit its timeout (`QueryCanceled` is
    an `OperationalError`) or the connection pool timed out. A programming or data error
    (`ProgrammingError`, `DataError`, ...) is NOT one and stays `internal`. It is meant to sit in
    a chain before the generic classifier (see `bootstrap.adapters.ChainedErrorClassifier`).
    """

    def classify(self, exc: BaseException) -> FailureInfo | None:
        if isinstance(
            exc, TeiUnavailableError | OperationalError | InterfaceError | PoolTimeoutError
        ):
            return _RETRIEVAL_UNAVAILABLE
        if isinstance(exc, DBAPIError) and exc.connection_invalidated:
            return _RETRIEVAL_UNAVAILABLE
        if isinstance(exc, psycopg.OperationalError | psycopg.InterfaceError):
            return _RETRIEVAL_UNAVAILABLE
        return None
