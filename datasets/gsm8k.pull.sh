#!/usr/bin/env bash
# A convenience wrapper. The pull logic lives in datasets/pull.py, which uses the
# huggingface_hub library from the dev shell - one implementation, two ways to reach it.
#
#   bash datasets/gsm8k.pull.sh   ==   python3 datasets/pull.py gsm8k
#
# The parameterised tool replaces the per-dataset shell scripts this file used to be: the
# manifest already knows each dataset's licence and patterns, so a second copy of that
# knowledge in shell was a second place to be wrong.

set -euo pipefail

exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pull.py" gsm8k
