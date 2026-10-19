#!/bin/bash
exec uv run --with httpx --with beautifulsoup4 --with lxml python -I "$@"
