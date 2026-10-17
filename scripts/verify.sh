#!/usr/bin/env bash
# The gate for this repository: run it before every commit, and in CI.
#
#   bash scripts/verify.sh
#
# Five steps, each of which must pass on this repository (not on a hypothetical one):
#
#   1. the shared layer's tests
#   2. P1's tests
#   3. P1's end-to-end check on fixtures (no network, no GPU)
#   4. P2's tests
#   5. P2's loop check: every task solvable, the loop closes, no entropy cliff, conflicts logged
#   6. the dataset tool's tests (offline: the download is injected)
#   7. repository checks: characters, payloads, secrets, imports
#
# Nothing here touches the network. `datasets/pull.py` does, and it is not part of the gate
# for exactly that reason - the gate must be able to tell you that your environment is fine
# without asking anything of the world.
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

# Four test suites, four discovery roots. The shared layer is a package under `common/`;
# `p1-verifiers` and `p2-reagent` have hyphens in their names and cannot be packages; `datasets/`
# holds a tool rather than a package, so its tests are a root of their own.
run "common tests" python3 -m unittest discover -s common/tests -t .
run "p1 tests" python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests
run "p1 check (fixtures)" python3 p1-verifiers/checks/check_p1.py
run "p2 tests" python3 -m unittest discover -s p2-reagent/tests -t p2-reagent/tests
run "p2 check (loop)" python3 p2-reagent/checks/check_p2.py
run "datasets tests" python3 -m unittest discover -s datasets/tests -t datasets/tests

# The checks are checked first: a scan that has stopped being able to fail would report
# success for the rest of the gate too.
run "repository checks (selftest)" python3 scripts/check_repo.py --selftest
run "repository checks" python3 scripts/check_repo.py

if [ "$status" -eq 0 ]; then
  echo
  echo "all checks passed"
else
  echo
  echo "verify.sh: FAILED - fix the steps marked above"
fi
exit "$status"
