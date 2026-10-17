"""Create or migrate the LangGraph checkpoint tables (conversation state of threads).

    uv run python -m agintis.cli.checkpointer_setup

The app never creates these tables at boot: it refuses to start until this command has run
(and again after an upgrade that bumps the checkpointer schema). Idempotent. Needs only the
database (DATABASE_URL): no embedding service and no case files (profiles, agent specs).
CHECKPOINTER_ENABLED=false is rejected: there is nothing to set up for a deployment without
threads.

Exit codes: 0 tables ready; 2 the checkpointer is disabled; 3 aborted (database unavailable,
missing privileges).
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from agintis.bootstrap.checkpointer import build_checkpointer
from agintis.bootstrap.settings import get_settings
from agintis.platform.checkpointer import EXPECTED_SCHEMA_VERSION

EXIT_OK = 0
EXIT_DISABLED = 2
EXIT_ABORTED = 3
logger = logging.getLogger("agintis.cli.checkpointer_setup")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="python -m agintis.cli.checkpointer_setup",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


async def run() -> bool:
    """True when the tables are ready, False when the checkpointer is disabled."""
    # Only the pool is built (no Container): the case files (profiles, agent specs, LLM profiles)
    # and the embedding service must not be able to block the schema setup.
    settings = get_settings()
    if not settings.checkpointer.enabled:
        return False
    checkpointer = build_checkpointer(settings)
    try:
        await checkpointer.open(wait=True)
        await checkpointer.setup()
        await checkpointer.verify()
        return True
    finally:
        await checkpointer.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        ready = asyncio.run(run())
    except Exception as exc:  # database down, not enough privileges, bad settings
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        return EXIT_ABORTED
    if not ready:
        print("CHECKPOINTER_ENABLED is false: nothing to set up", file=sys.stderr)
        return EXIT_DISABLED
    print(f"checkpoint tables ready (schema version {EXPECTED_SCHEMA_VERSION})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
