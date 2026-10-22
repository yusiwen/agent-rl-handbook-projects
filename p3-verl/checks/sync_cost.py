"""What does weight sync cost, and how do you know it happened?

Two measurements and one check, all against a running vLLM that has LoRA enabled:

1. **Restart cost** - stop the server, start it again, poll /health. This is the baseline every
   "just restart with the new weights" design pays per round.
2. **Adapter cost** - POST /v1/load_lora_adapter for each adapter and time it. This is what a round
   pays if the base model stays resident and the update arrives as an adapter.
3. **Did it apply?** - the same prompt to the base model and to the adapter, comparing the top-5
   token log-probabilities. Comparing the generated text alone is not evidence: on three confident
   prompts in this project the argmax token was identical with and without the adapter while the
   distributions differed by more than a nat. Same argmax is not same weights.

Run on the reference machine (a DGX Spark) from the repository root:

    python3 p3-verl/checks/sync_cost.py \
        --adapter policy-sft:/models/sft/adapter \
        --adapter policy-r6:/models/rl-c/adapter-r6

The adapter paths are the *container's* view of the volume: the trainer writes `/work/rl-c/adapter-r6`,
the server reads `/models/rl-c/adapter-r6`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE_MESSAGES = [{"role": "user", "content": "Reply with exactly one line: ACTION: ping"}]


def post(url: str, payload: dict, timeout: float = 120.0) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            text = handle.read().decode()
            # The LoRA endpoints answer with a sentence, not JSON; the completions endpoint with JSON.
            try:
                return handle.status, json.loads(text or "{}")
            except json.JSONDecodeError:
                return handle.status, {"text": text.strip()}
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read().decode()[:400]}


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=5) as handle:
            return handle.status == 200
    except Exception:  # noqa: BLE001 - a probe reports reachability, not a stack
        return False


def restart_cost(base_url: str) -> int:
    """Restart through the same script a reader would use, and read the number it printed."""
    result = subprocess.run(
        ["bash", str(ROOT / "p3-verl" / "env" / "serve_lora.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"healthy after (\d+) ms", result.stdout)
    if not match:
        print(result.stdout[-800:], result.stderr[-800:], file=sys.stderr)
        raise SystemExit("could not start the server through serve_lora.sh")
    if not healthy(base_url):  # the script polls its own port; make sure we agree
        raise SystemExit(f"{base_url} does not answer /health after the restart")
    return int(match.group(1))


def top_logprobs(base_url: str, model: str, top: int = 5) -> dict[str, float]:
    status, payload = post(
        base_url + "/v1/chat/completions",
        {
            "model": model,
            "messages": PROBE_MESSAGES,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": top,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    if status != 200:
        raise SystemExit(f"{model}: {status} {payload}")
    content = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return {entry["token"]: float(entry["logprob"]) for entry in content}


def spread(a: dict[str, float], b: dict[str, float]) -> float:
    """Largest log-probability difference over the union of the two top-k sets."""
    keys = set(a) | set(b)
    return max(abs(a.get(k, -30.0) - b.get(k, -30.0)) for k in keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure what weight sync costs, and whether it applied")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--base-name", default="qwen3-1.7b-base")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="an adapter to load, as name:path-inside-the-container",
    )
    parser.add_argument("--skip-restart", action="store_true", help="use the server that is already up")
    parser.add_argument("--json-out", default="", help="write the measurements here as JSON")
    args = parser.parse_args(argv)

    adapters: list[tuple[str, str]] = []
    for item in args.adapter:
        if ":" not in item:
            parser.error(f"--adapter wants NAME:PATH, got {item!r}")
        name, path = item.split(":", 1)
        adapters.append((name, path))

    record: dict[str, object] = {}

    restart_ms = None
    if not args.skip_restart:
        restart_ms = restart_cost(args.base_url)
        record["restart_ms"] = restart_ms
        print(f"restart to /health 200 : {restart_ms:,} ms")

    base = top_logprobs(args.base_url, args.base_name)
    record["base_top"] = base

    loads = {}
    probes = []
    for name, path in adapters:
        started = time.time()
        status, payload = post(
            args.base_url + "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}
        )
        elapsed_ms = int((time.time() - started) * 1000)
        loads[name] = {"status": status, "ms": elapsed_ms, "response": payload}
        said = payload.get("text") or payload.get("error") or payload
        print(f"load {name:<14} : {elapsed_ms:,} ms ({status}) - {said}")

        after = top_logprobs(args.base_url, name)
        delta = spread(base, after)
        probes.append({"adapter": name, "spread": round(delta, 4), "top": after})
        verdict = "changed" if delta > 0.01 else "NO CHANGE"
        print(f"  distribution vs base : max delta {delta:.3f} nats -> {verdict}")

        started = time.time()
        unload_status, _ = post(args.base_url + "/v1/unload_lora_adapter", {"lora_name": name})
        unload_ms = int((time.time() - started) * 1000)
        print(f"  unload               : {unload_ms:,} ms ({unload_status})")

        if restart_ms:
            print(f"  sync cost vs restart : {elapsed_ms / restart_ms * 100:.2f}% of one restart")

    record["loads"] = loads
    record["probes"] = probes

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    if not adapters:
        print("no --adapter given: measured the base model only")
        return 0
    if any(probe["spread"] <= 0.01 for probe in probes):
        print("FAIL: an adapter did not change the distribution - it was not applied", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
