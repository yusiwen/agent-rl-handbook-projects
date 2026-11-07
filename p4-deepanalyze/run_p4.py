"""Run P4 end to end: question -> code -> sandbox -> report, over a served model.

    PYTHONPATH=. python3 p4-deepanalyze/run_p4.py \
        --base-url http://172.17.0.1:8003 --model qwen3-1.7b-base \
        --code-sha git:abc1234 --sandbox docker

The record it writes is the one Chapter 10 quotes: a header naming the model, the sandbox
image, the code revision and the dataset digest, then one line per task with every turn, every
observation and the verdict. ``summary.json`` is derived from those lines, never from the
console - the printed table is a convenience, the file is the claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.runs import RunHeader, RunWriter, digest_of  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from p4_deepanalyze import tasks as task_module  # noqa: E402
from p4_deepanalyze.agent import run_episode  # noqa: E402
from p4_deepanalyze.client import ChatClient  # noqa: E402
from p4_deepanalyze.sandbox import DockerSandbox, LocalSandbox, python_available_in  # noqa: E402

PROJECT = "p4-deepanalyze"


def summarise(episodes: list[dict]) -> dict:
    total = len(episodes)
    solved = sum(1 for e in episodes if e["solved"])
    protocol_errors = sum(1 for e in episodes if e["protocol_errors"])
    sandbox_errors = sum(e["sandbox_errors"] for e in episodes)
    timeouts = sum(e["timeouts"] for e in episodes)
    code_runs = sum(1 for e in episodes for t in e["turns"] if t["kind"] == "code")
    return {
        "tasks": total,
        "solved": solved,
        "solve_rate": round(solved / total, 4) if total else 0.0,
        "protocol_error_tasks": protocol_errors,
        "protocol_compliance": round((total - protocol_errors) / total, 4) if total else 0.0,
        "code_runs": code_runs,
        "sandbox_errors": sandbox_errors,
        "timeouts": timeouts,
        "mean_turns": round(sum(len(e["turns"]) for e in episodes) / total, 2) if total else 0.0,
        "seconds": round(sum(e["seconds"] for e in episodes), 1),
        "per_task": {e["task_id"]: {"solved": e["solved"], "turns": len(e["turns"]), "reason": e["reason"]} for e in episodes},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the P4 data-analysis agent")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", default="qwen3-1.7b-base")
    parser.add_argument("--tasks", default=str(HERE / "fixtures" / "tasks.jsonl"))
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--code-sha", default="git:none")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sandbox", choices=("docker", "local"), default="docker")
    parser.add_argument("--sandbox-timeout", type=float, default=20.0)
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    task_list = task_module.load(args.tasks)
    if args.limit:
        task_list = task_list[: args.limit]

    sandbox = (
        DockerSandbox(image=args.image, timeout_s=args.sandbox_timeout)
        if args.sandbox == "docker"
        else LocalSandbox(timeout_s=args.sandbox_timeout)
    )
    health = python_available_in(sandbox) if args.sandbox == "docker" else {"executor": "local-python", "present": True}
    if args.sandbox == "docker" and not health["present"]:
        print(f"sandbox is not usable: {health}", file=sys.stderr)
        return 2

    client = ChatClient(
        base_url=args.base_url, model=args.model,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )

    config = {
        "base_url": args.base_url, "model": args.model, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "max_turns": args.max_turns,
        "sandbox": args.sandbox, "image": args.image, "sandbox_timeout": args.sandbox_timeout,
        "tasks": Path(args.tasks).name,
    }
    config_hash = "sha256:" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:32]
    header = RunHeader(
        run_id="p4-" + time.strftime("%Y%m%d-%H%M%S"),
        env_version=f"executor={health.get('executor')};model={args.model}",
        config_hash=config_hash,
        code_sha=args.code_sha,
        project=PROJECT,
        dataset=Path(args.tasks).name,
        dataset_digest="sha256:" + digest_of(args.tasks)[:32],
        notes=args.notes or f"max_turns={args.max_turns}; temperature={args.temperature}",
    )
    writer = RunWriter(header, args.runs_dir)
    print(f"run: {writer.dir}")

    episodes: list[dict] = []
    started = time.monotonic()
    for index, task in enumerate(task_list, start=1):
        episode = run_episode(
            task, client, sandbox,
            max_turns=args.max_turns, model=args.model, code_sha=header.code_sha,
        )
        row = episode.to_dict()
        row["task_question"] = task.question
        row["expected"] = task.expected
        episodes.append(row)
        mark = "ok" if episode.solved else "no"
        print(f"  [{index}/{len(task_list)}] {task.id:26} {mark:2}  turns={len(episode.turns)}  {episode.reason}")

    with open(writer.dir / "episodes.jsonl", "w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarise(episodes)
    summary["wall_seconds"] = round(time.monotonic() - started, 1)
    summary["model_calls"] = client.calls
    summary["sandbox"] = health
    (writer.dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"\nsolved {summary['solved']}/{summary['tasks']}  "
        f"protocol compliance {summary['protocol_compliance']:.0%}  "
        f"code runs {summary['code_runs']}  sandbox errors {summary['sandbox_errors']}  "
        f"timeouts {summary['timeouts']}  wall {summary['wall_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
