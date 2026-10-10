#!/usr/bin/env bash
# The gate for this repository: run it before every commit, and in CI.
#
#   bash scripts/verify.sh
#
# Four steps, each of which must pass on this repository (not on a hypothetical one):
#
#   1. the shared layer's tests
#   2. P1's tests
#   3. P1's end-to-end check on fixtures (no network, no GPU)
#   4. repository checks: characters, payloads, secrets
#
# Exit status is non-zero if anything fails, so this is usable as a CI step unchanged.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

status=0
run() {
  local label="$1"
  shift
  echo "== ${label} =="
  if "$@"; then
    echo "   ok"
  else
    echo "   FAILED: ${label}"
    status=1
  fi
}

# The two test suites have different discovery roots: the shared layer is a package
# under `common/`, while `p1-verifiers` has a hyphen in its name and cannot be one.
run "common tests" python3 -m unittest discover -s common/tests -t .
run "p1 tests" python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests
run "p1 check (fixtures)" python3 p1-verifiers/checks/check_p1.py
run "repository checks" python3 scripts/check_repo.py

if [ "$status" -eq 0 ]; then
  echo
  echo "all checks passed"
else
  echo
  echo "verify.sh: FAILED - fix the steps marked above"
fi
exit "$status"
