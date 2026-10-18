"""Run whole episodes with a served model as the policy, and count what it solves.

    python3 p2-reagent/checks/measure_policy.py --base-url http://127.0.0.1:8000 --model qwen3-1.7b
    python3 p2-reagent/checks/measure_policy.py --base-url http://127.0.0.1:8001 --model qwen3-1.7b-sft

The counterpart to `measure_format.py`, and the number the fine-tune is actually for. Format
compliance says the model can answer in the contract; this says whether the answers are right.

What it measures, and what it does not: the model is asked one turn at a time, its reply is parsed
with the same parser the loop uses, and the episode continues until the environment says done or
the turn budget runs out. An unparsed reply is passed through as the text the model produced, which
the environment rejects as an action outside the menu - so a model that stops following the format
loses the task, visibly, rather than having its reply quietly repaired.

Temperature 0 by default: this is a reading, not a sample. `--samples N` repeats each task at
temperature 1.0 to show how much the reading depends on sampling.
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
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"].get("content") or ""


def play(base_url: str, model: str, task_id: str, max_turns: int, max_tokens: int, temperature: float) -> dict:
    attempt = env.start(task_id)
    history: list[tuple[str, str, str]] = []
    reasons: Counter[str] = Counter()
    actions_taken: list[str] = []
    while not attempt.finished and len(actions_taken) < max_turns:
        messages = prompt.render(attempt.task, attempt.observation, attempt.menu(), history)
        reply = ask(base_url, model, messages, max_tokens, temperature)
        parsed = actions.parse_action(reply, attempt.menu())
        reasons[parsed.reason] += 1
        action = parsed.action if parsed.ok else (parsed.raw or "<unparsed>")
        result = attempt.step(action)
        history.append((action, result.signal, result.observation))
        actions_taken.append(action)
    return {
        "task_id": task_id,
        "solved": attempt.solved(),
        "turns": len(actions_taken),
        "reasons": dict(reasons),
        "actions": actions_taken,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run episodes with a served model as the policy")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args(argv)

    wanted = args.tasks or [task.task_id for task in env.all_tasks()]
    solved_total = 0
    runs = 0
    turn_counts: list[int] = []
    format_ok = 0
    format_total = 0
    for sample in range(args.samples):
        for task_id in wanted:
            try:
                outcome = play(
                    args.base_url, args.model, task_id, args.max_turns, args.max_tokens,
                    args.temperature if args.samples == 1 else 1.0,
                )
            except (urllib.error.URLError, KeyError, TimeoutError) as exc:
                print(f"FAIL {task_id}: {type(exc).__name__}: {exc}")
                return 2
            runs += 1
            solved_total += int(outcome["solved"])
            turn_counts.append(outcome["turns"])
            format_total += sum(outcome["reasons"].values())
            format_ok += outcome["reasons"].get("ok", 0)
            flag = "ok  " if outcome["solved"] else "FAIL"
            print(f"{flag} {task_id:22s} turns={outcome['turns']} {' -> '.join(outcome['actions'])[:90]}")
    print()
    print(f"  solved {solved_total}/{runs} ({solved_total / runs:.0%})")
    print(f"  format {format_ok}/{format_total} usable turns ({format_ok / format_total:.0%})")
    print(f"  mean turns {sum(turn_counts) / len(turn_counts):.2f}")
    print()
    print(
        f"POLICY_RESULT model={args.model} tasks={len(wanted)} samples={args.samples} "
        f"solved={solved_total}/{runs} fraction={solved_total / runs:.2f} "
        f"format={format_ok / format_total:.2f} temperature={args.temperature}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
