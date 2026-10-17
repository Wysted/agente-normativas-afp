"""CI hand-copies the pgvector and TEI images from docker-compose.yml; keep them in sync."""

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEI_IMAGE = re.compile(r"ghcr\.io/huggingface/text-embeddings-inference:\S+")


def _load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    return data


def test_ci_pgvector_image_matches_compose() -> None:
    compose = _load("docker-compose.yml")["services"]["db"]["image"]
    ci = _load(".github/workflows/ci.yml")["jobs"]["integration"]["services"]["db"]["image"]
    assert ci == compose, f"ci.yml db image {ci!r} != docker-compose.yml {compose!r}"


def test_ci_tei_image_matches_compose() -> None:
    compose = _load("docker-compose.yml")["services"]["embeddings"]["image"]
    steps = _load(".github/workflows/ci.yml")["jobs"]["integration"]["steps"]
    found = [m.group(0) for s in steps if (m := TEI_IMAGE.search(s.get("run", "")))]
    assert found, "no TEI image found in any ci.yml integration step"
    assert found == [compose], f"ci.yml TEI image {found!r} != docker-compose.yml {compose!r}"
