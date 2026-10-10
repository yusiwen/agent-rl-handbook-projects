#!/usr/bin/env python3
# pyright: reportMissingImports=false

"""Run P1 against a served model: the same pipeline the fixtures exercise, with a real model.

    python3 p1-verifiers/checks/run_p1_live.py --base-url http://host:8000 --model the-id

``check_p1.py`` proves the *pipeline* offline; this proves the *loop*. Five things it does that
a naive script would not, each because the naive version cost something:

* **Preflight the model name** against ``/v1/models`` before spending a round on it. The
  failure this prevents is the expensive one: a run that discovers a typo after the server has
  loaded 20 GB of weights.
* **Report coverage first**, because the covered subset is the ceiling. If three verifiers are
  still stubs, the honest thing is to say so before the tokens are spent, not after.
* **Count tokens with the model's own tokeniser**, over HTTP, and put its name in every row's
  ``env_version``. A row whose token counts came from a word counter is not a measurement.
* **Check the rows it just produced** with ``common/checks.py`` before printing any conclusion.
  A run that records its own bug is worth more than one that reports a number.
* **Record the environment**: endpoint, model, tokeniser, task-file digest, commit. Without
  them the record is a screenshot.

Exit status is 0 only when the run finished and every check passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "src"))

from common.checks import (  # noqa: E402
    check_advantage_spread,
    check_boundaries,
    check_mask,
    check_terms,
)
from common.runs import RunHeader, RunWriter, digest_of, round_record  # noqa: E402

from p1_verifiers import (  # noqa: E402
    default_registry,
    load_tasks,
    partition_by_coverage,
    report,
    run_episode,
)
from p1_verifiers.harness import Provenance  # noqa: E402
from p1_verifiers.remote import (  # noqa: E402
    OpenAIChatResponder,
    RemoteError,
    ServerTokeniser,
    list_models,
)

DEFAULT_TASKS = REPO / "p1-verifiers" / "fixtures" / "tasks.jsonl"


def code_sha() -> str:
    """The current commit, or ``git:none`` outside a repository."""
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                            capture_output=True, text=True, check=False)
    return f"git:{result.stdout.strip()}" if result.returncode == 0 else "git:none"


def default_runs_root() -> Path:
    """Where records go: ``--runs-dir``, else ``$ARL_RUNS_DIR``, else ``runs/`` - the same
    order the fixture check uses, so one environment variable moves both."""
    return Path(os.environ.get("ARL_RUNS_DIR") or REPO / "runs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P1 against a served model.")
    parser.add_argument("--base-url", required=True,
                        help="the server's root, e.g. http://192.168.2.40:8000")
    parser.add_argument("--model", required=True,
                        help="the served model id, exactly as the server spells it")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="a JSONL task file")
    parser.add_argument("--samples", type=int, default=1,
                        help="attempts per task (the GRPO group size)")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N tasks")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per attempt")
    parser.add_argument("--api-key-env", default=None,
                        help="name of an environment variable holding the key, if the server wants one")
    parser.add_argument("--runs-dir", default=None, help="where run records go")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = Path(args.runs_dir) if args.runs_dir else default_runs_root()

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    registry = default_registry()

    coverage = registry.coverage(task.as_mapping() for task in tasks)
    print(coverage.render())
    covered, uncovered = partition_by_coverage(tasks, registry)
    for task, reason in uncovered:
        print(f"  not scoreable: {task.id} - {reason}")
    if not covered:
        print("error: nothing in this task set can be scored, so a run would produce no signal",
              file=sys.stderr)
        return 2

    # Preflight: the model name first, so a typo costs a second rather than a round.
    try:
        models = list_models(args.base_url)
    except RemoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.model not in models:
        print(f"error: the server does not serve {args.model!r}. It serves: "
              f"{', '.join(models) or '(nothing)'}", file=sys.stderr)
        return 2

    responder = OpenAIChatResponder(
        args.base_url, args.model, api_key_env=args.api_key_env, temperature=args.temperature,
        top_p=args.top_p, max_tokens=args.max_tokens, timeout=args.timeout,
    )
    tokeniser = ServerTokeniser(args.base_url, args.model, timeout=args.timeout)

    try:  # the tokeniser is a second endpoint, and it fails for its own reasons
        probe = tokeniser.count("a token count from the server")
    except RemoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    provenance = Provenance(env_version=tokeniser.name, config_hash=digest_of(args.tasks),
                            code_sha=code_sha())
    print(f"model    : {responder.describe()}")
    print(f"tokeniser: {tokeniser.name} (probe: {probe} tokens)")
    print(f"tasks    : {len(covered)} covered, x{args.samples} samples = "
          f"{len(covered) * args.samples} attempts")
    print()

    rows = []
    for task in covered:
        for sample_index in range(args.samples):
            try:
                response = responder.respond(task, sample_index)
                row = run_episode(task, response, registry, sample_index, provenance, tokeniser)
            except RemoteError as exc:
                # One failed attempt must not destroy the records of the ones that worked.
                print(f"  ERROR {task.id}-s{sample_index}: {exc}", file=sys.stderr)
                continue
            rows.append(row)
            mark = "ok  " if row.extras.get("passed") else "MISS"
            finish = f", {responder.last_finish_reason}" if responder.last_finish_reason else ""
            print(f"  {mark} {row.episode_id}  {row.model_tokens} tokens trained{finish}")

    if not rows:
        print("error: no attempt completed", file=sys.stderr)
        return 1

    print()
    print(report(rows))

    results = [check_mask(rows), check_boundaries(rows), check_terms(rows),
               check_advantage_spread(rows)]
    print()
    for result in results:
        print(result.render())

    run_id = (f"p1-live-{args.model.replace('/', '_')}-"
              f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    writer = RunWriter(
        RunHeader(
            run_id=run_id,
            env_version=tokeniser.name,
            config_hash=digest_of(args.tasks),
            code_sha=code_sha(),
            project="p1-verifiers",
            dataset=Path(args.tasks).name,
            dataset_digest=digest_of(args.tasks),
            notes=f"{responder.describe()}; samples={args.samples}; "
                  f"covered {len(covered)}/{len(tasks)} tasks",
        ),
        root=runs_root,
    )
    writer.append_round(round_record(0, rows))
    writer.append_rollouts(0, rows)
    print(f"\nrun record: {writer.dir}")

    failed = [result.name for result in results if not result.ok]
    if failed:
        print(f"FAILED: {', '.join(failed)} - the numbers above describe a broken run",
              file=sys.stderr)
        return 1
    print("all checks passed on this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
