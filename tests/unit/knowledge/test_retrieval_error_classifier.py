import psycopg
import pytest
from sqlalchemy.exc import (
    DataError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from agintis.bootstrap.adapters import ChainedErrorClassifier
from agintis.modules.knowledge.infrastructure.errors import (
    RETRIEVAL_UNAVAILABLE_CODE,
    RetrievalErrorClassifier,
)
from agintis.modules.knowledge.infrastructure.tei import TeiError, TeiUnavailableError
from agintis.platform.llm import ProviderErrorClassifier

CLASSIFIER = RetrievalErrorClassifier()


@pytest.mark.parametrize(
    "error",
    [
        TeiUnavailableError("overloaded"),
        OperationalError("SELECT", {}, Exception("canceling statement due to statement timeout")),
        InterfaceError("SELECT", {}, Exception("connection closed")),
        PoolTimeoutError("QueuePool limit reached"),
        psycopg.OperationalError("server closed the connection"),
        psycopg.errors.QueryCanceled("statement timeout"),
    ],
)
def test_outages_are_retryable_and_leak_no_exception_text(error: Exception) -> None:
    failure = CLASSIFIER.classify(error)

    assert failure is not None
    assert (failure.code, failure.retryable) == (RETRIEVAL_UNAVAILABLE_CODE, True)
    assert str(error) not in failure.message


@pytest.mark.parametrize(
    "error",
    [
        ProgrammingError("SELECT", {}, Exception("syntax error")),
        DataError("SELECT", {}, Exception("bad value")),
        TeiError("unexpected answer"),  # a bad answer is a bug, not an outage
        ValueError("x"),
    ],
)
def test_bugs_and_data_errors_are_not_outages(error: Exception) -> None:
    assert CLASSIFIER.classify(error) is None


def test_the_chain_asks_the_specific_classifiers_first_then_the_generic_one() -> None:
    chain = ChainedErrorClassifier(ProviderErrorClassifier(), RetrievalErrorClassifier())

    assert chain.classify(TeiUnavailableError("x")).code == RETRIEVAL_UNAVAILABLE_CODE
    assert chain.classify(ValueError("x")).code == "internal"
