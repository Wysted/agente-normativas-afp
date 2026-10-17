#!/bin/bash
exec ~/.local/bin/uv run --with httpx --with beautifulsoup4 --with lxml python -I "$@"
