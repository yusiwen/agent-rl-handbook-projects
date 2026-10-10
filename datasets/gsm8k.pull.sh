#!/usr/bin/env bash
# Pull GSM8K into datasets/payloads/ and record exactly what was pulled.
#
#   bash datasets/gsm8k.pull.sh
#
# The revision is read from the Hugging Face API before the download and written to the
# digest file afterwards, so a run record can cite something other than "main, in
# October".  Nothing lands in git: payloads/ is ignored.
#
# This script is also the pattern for the others named in datasets/manifest.yaml.

set -euo pipefail

DATASET="openai/gsm8k"
LICENCE="MIT (per the dataset card, checked 2026-10-09)"
NAME="gsm8k"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/datasets/payloads/$NAME"
mkdir -p "$OUT"

if ! command -v hf >/dev/null 2>&1; then
  echo "the 'hf' CLI is not on PATH; install it with: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

echo "reading the current revision of $DATASET"
REVISION="$(python3 - "$DATASET" <<'PY'
import json, sys, urllib.request
url = f"https://huggingface.co/api/datasets/{sys.argv[1]}"
with urllib.request.urlopen(url, timeout=30) as response:
    print(json.load(response)["sha"])
PY
)"
echo "  revision $REVISION"
echo "  licence  $LICENCE"

echo "downloading into $OUT"
hf download "$DATASET" --repo-type dataset --revision "$REVISION" --local-dir "$OUT" --quiet

python3 - "$OUT" "$NAME" "$DATASET" "$REVISION" "$LICENCE" <<'PY'
import hashlib, json, pathlib, sys

out, name, dataset, revision, licence = sys.argv[1:6]
root = pathlib.Path(out)
entries = {}
total = 0
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    # Skip the CLI's own bookkeeping: .cache/, .huggingface/, lock files. They are not
    # the dataset, and a digest that changes when a client updates is not a pin.
    relative_parts = path.relative_to(root).parts
    if any(part.startswith(".") for part in relative_parts):
        continue
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    size = path.stat().st_size
    total += size
    entries[path.relative_to(root).as_posix()] = {
        "sha256": hasher.hexdigest()[:16],
        "bytes": size,
    }

digest = {
    "dataset": dataset,
    "revision": revision,
    "licence": licence,
    "files": entries,
    "total_bytes": total,
}
# The digest is provenance, so it is committed: datasets/digests/ is exempted from the
# payload ignore rules. The data itself never is.
digests = pathlib.Path(out).parents[1] / "digests"
digests.mkdir(parents=True, exist_ok=True)
digest_path = digests / f"{name}.digest.json"
digest_path.write_text(json.dumps(digest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"  {len(entries)} file(s), {total} bytes")
print(f"  digest written to {digest_path}")
PY

echo "done. Cite datasets/digests/$NAME.digest.json in any run record that uses this data."
