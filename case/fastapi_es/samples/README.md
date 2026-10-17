# Samples: FastAPI documentation in Spanish

- Source: https://github.com/fastapi/fastapi, `docs/es/docs/`, commit
  `bb7f314287bf36e04539c572a2256d94ea45f0d6` (tag 0.143.3). Published at https://fastapi.tiangolo.com/es/.
- License: MIT, Copyright (c) 2018 Sebastián Ramírez (upstream `LICENSE`, copied next to this file).
- The Spanish text is an LLM-generated translation reviewed by native speakers
  (https://fastapi.tiangolo.com/translations/): wording quirks are upstream's, not ours.
- Subset: `tutorial/` (every page under `docs/es/docs/tutorial/`, family `tutorial`) and 13
  top-level pages (family `guia`). `_llm-test.md` and `translation-banner.md` are excluded.
- Preprocessed: each `{* ../../docs_src/<file> hl[..] *}` include line is replaced by a fenced
  `python` block with the referenced file (highlight modifiers dropped); `ln[...]` keeps only
  those line ranges, with `# ...` inserted between non-contiguous ranges. Nothing else is changed.

Files are not versioned (gitignored, only this README, `_manifest.json` and `LICENSE` are). Rebuild them,
together with `../index/documents.jsonl`, with:

    uv run python -I case/fastapi_es/fetch/fetch.py
