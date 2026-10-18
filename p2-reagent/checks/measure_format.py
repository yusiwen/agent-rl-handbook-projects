"""How often does the served model reply with a usable ACTION line?

    python3 p2-reagent/checks/measure_format.py --base-url http://127.0.0.1:8000 --model qwen3-1.7b

This is the number the fine-tune is judged against, and it has to exist before the fine-tune does:
"format compliance reached 90%" means nothing without the reading it started from, on the same
prompts, the same parser and the same tasks.

It measures the *first turn* of each task, which is where format compliance is decided - a model
that cannot produce one legal action on turn one will not produce one on turn three. Temperature
defaults to 0 so the baseline is a reading rather than a sample.

No GPU, no container: one HTTP client against an OpenAI-compatible endpoint, so it runs from any
machine that can reach the server.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent.parent))

from p2_reagent import actions, env, prompt  # noqa: E402


def ask(base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen3 opens a thinking block unless it is told not to; the baseline is about the answer
        # format, so the reasoning is switched off the same way the serving command does it.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"].get("content") or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure format compliance of a served model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument("--tasks", nargs="*", default=None, help="task ids; default is all of them")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    wanted = args.tasks or [task.task_id for task in env.all_tasks()]
    reasons: Counter[str] = Counter()
    rows = []
    for task_id in wanted:
        attempt = env.start(task_id)
        messages = prompt.render(attempt.task, attempt.observation, attempt.menu())
        try:
            reply = ask(args.base_url, args.model, messages, args.max_tokens, args.temperature)
        except (urllib.error.URLError, KeyError, TimeoutError) as exc:
            print(f"FAIL {task_id}: {type(exc).__name__}: {exc}")
            return 2
        parsed = actions.parse_action(reply, attempt.menu())
        reasons[parsed.reason] += 1
        shown = parsed.raw.replace("\n", " ")[:70]
        rows.append((task_id, parsed.reason, shown))
        print(f"{'ok  ' if parsed.ok else 'FAIL'} {task_id:22s} {parsed.reason:15s} {shown!r}")

    total = len(rows)
    ok = reasons["ok"]
    print()
    for reason, count in sorted(reasons.items(), key=lambda pair: -pair[1]):
        print(f"  {reason:15s} {count}/{total}")
    print()
    print(
        f"FORMAT_BASELINE model={args.model} tasks={total} ok={ok} "
        f"fraction={ok / total:.2f} temperature={args.temperature}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
