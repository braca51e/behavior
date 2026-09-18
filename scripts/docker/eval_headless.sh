#!/usr/bin/env bash
# Back-compat alias: headless is already the default in eval.sh.
exec "$(cd "$(dirname "$0")" && pwd)/eval.sh" "$@"
