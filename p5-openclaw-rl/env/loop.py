"""The asynchronous loop: rollout workers, a trainer, and an honest record of how stale it got.

    python3 p5-openclaw-rl/env/loop.py --mode sync  --rounds 12 --out /rl/sync
    python3 p5-openclaw-rl/env/loop.py --mode async --rounds 12 --out /rl/async --workers 3

What is being measured is not a score.  Two loops, the same policy, the same tasks, the same number
of updates; one waits for fresh samples, the other keeps sampling while it trains.  The second wins
on wall clock and loses on freshness, and the only reason the comparison is worth anything is that
every sample records the policy version that produced it.

The mechanics that make that possible:

* the rollout workers talk to vLLM over HTTP, so generation never contends with the trainer's GPU
  memory, and a worker can be killed without touching the model;
* each update saves the adapter and hot-loads it under a new name (`p5-v<k>`), which is the weight
  sync: workers pick the new name up on their next episode, at no restart cost;
* every sample carries `version`, so the trainer can compute, per batch, how far behind the policy the
  samples were - and `--stale-bound` can drop the ones that are too far behind, counted, not silently.

A caveat worth stating in the record: the environment is in-process and resets in microseconds, so
the sampling rate here is limited by the served model, not by tool latency.  A container-based world
would make staleness much worse, which is exactly why the number is recorded rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "src"))

from p5_openclaw import credit, protocol  # noqa: E402
from p5_openclaw.world import TASKS, World, solved  # noqa: E402

TASKS_BY_ID = {task.id: task for task in TASKS}


# ----------------------------------------------------------------------------------------------
# serving
# ----------------------------------------------------------------------------------------------


def chat(base_url: str, model: str, messages: list[dict], temperature: float, max_tokens: int = 120) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"]


def served_models(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=30) as response:
        return [row["id"] for row in json.loads(response.read())["data"]]


def load_adapter(base_url: str, name: str, path: str) -> None:
    # load_inplace makes a rerun of the same experiment work: the server refuses to add a second
    # adapter under a name it already knows, which is correct behaviour and a nuisance when the
    # previous attempt was interrupted halfway through a version sequence.
    payload = json.dumps({"lora_name": name, "lora_path": path, "load_inplace": True}).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/load_lora_adapter", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"adapter load failed for {name}: {exc.code} {exc.read()[:300]!r}") from exc


# ----------------------------------------------------------------------------------------------
# one episode
# ----------------------------------------------------------------------------------------------


def episode(task_id: str, base_url: str, model: str, temperature: float) -> dict:
    """Play one episode and return everything the trainer and the record need."""
    task = TASKS_BY_ID[task_id]
    world = World(task=task)
    messages = [
        {"role": "system", "content": protocol.SYSTEM_PROMPT},
        {"role": "user", "content": protocol.first_message(task.question)},
    ]
    turns: list[dict] = []
    started = time.monotonic()
    for _ in range(task.max_turns):
        reply = chat(base_url, model, messages, temperature)
        entry: dict = {"messages": list(messages), "reply": reply}
        try:
            call = protocol.parse(reply)
        except protocol.ProtocolError as exc:
            entry["protocol_error"] = exc.reason
            turns.append(entry)
            break
        observation = world.step(call.tool, call.args)
        entry["tool"] = call.tool
        entry["args"] = call.args
        entry["observation"] = observation.text
        entry["error"] = observation.error
        turns.append(entry)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": protocol.format_observation(observation.text, observation.error, observation.done)})
        if observation.done:
            break
    return {
        "task_id": task_id,
        "model": model,
        "turns": turns,
        "solved": solved(world),
        "steps": len(turns),
        "tool_errors": world.errors,
        "protocol_errors": sum(1 for turn in turns if turn.get("protocol_error")),
        "seconds": round(time.monotonic() - started, 2),
    }


# ----------------------------------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------------------------------


class Version:
    """The current policy name, shared by the workers and bumped by the trainer."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        with self._lock:
            return self._name

    def bump(self, name: str) -> None:
        with self._lock:
            self._name = name


def rollout_worker(index: int, args, version: Version, samples: queue.Queue, stop: threading.Event) -> None:
    order = list(TASKS_BY_ID)
    offset = index
    while not stop.is_set():
        task_id = order[offset % len(order)]
        offset += 1
        name = version.name
        record = episode(task_id, args.base_url, name, args.temperature)
        record["version"] = name
        record["arrived"] = time.time()
        samples.put(record)


def _by_task(batch: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for sample in batch:
        groups.setdefault(sample["task_id"], []).append(sample)
    return groups


def train_step(model, tokeniser, optimiser, batch: list[dict], reward_of) -> tuple[float, int]:
    """One policy-gradient step over a batch, with the group mean as the baseline."""
    import torch

    by_task: dict[str, list[dict]] = {}
    for sample in batch:
        by_task.setdefault(sample["task_id"], []).append(sample)
    used = []
    for rows in by_task.values():
        rewards = [reward_of(row) for row in rows]
        mean = sum(rewards) / len(rewards)
        for row, reward in zip(rows, rewards):
            row["reward"] = reward
            row["advantage"] = round(reward - mean, 6)
            if row["advantage"] != 0.0 and row["turns"]:
                used.append(row)

    loss_value = 0.0
    for row in used:
        losses = _turn_losses(model, tokeniser, row)
        if not losses:
            continue
        loss = sum(losses) / len(losses)
        weighted = -row["advantage"] * loss
        optimiser.zero_grad(set_to_none=True)
        weighted.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimiser.step()
        loss_value += loss.item()
    return (loss_value / len(used) if used else 0.0), len(used)


def _turn_losses(model, tokeniser, row: dict) -> list:
    import torch

    losses = []
    for turn in row["turns"]:
        text = tokeniser.apply_chat_template(turn["messages"], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokeniser(text, add_special_tokens=False)["input_ids"]
        reply_ids = tokeniser(turn["reply"] + tokeniser.eos_token, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt_ids + reply_ids], device="cuda")
        labels = torch.tensor([[-100] * len(prompt_ids) + reply_ids], device="cuda")
        losses.append(model(input_ids=ids, labels=labels).loss)
    return losses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P5 asynchronous multi-turn loop")
    parser.add_argument("--mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--base-url", default="http://172.17.0.1:8003")
    parser.add_argument("--adapter", required=True, help="starting adapter, served under --adapter-name")
    parser.add_argument("--adapter-name", default="p5-v0")
    parser.add_argument("--model-path-prefix", default="/models/p5", help="where new adapters are written")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--samples-per-task", type=int, default=2,
                        help="2 or more: a group of one sample has no baseline and teaches nothing")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--stale-bound", type=int, default=1)
    parser.add_argument(
        "--reward",
        choices=("outcome", "process"),
        default="outcome",
        help=(
            "outcome: solved = 1, else 0. process: the outcome plus a small credit for each tool call "
            "that went through and a penalty for each that errored - a rule-based stand-in for a "
            "process reward model, kept cheap so the loop is what gets measured"
        ),
    )
    parser.add_argument("--sync-eval", type=int, default=1, help="run a greedy eval every N rounds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokeniser = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        json.loads((Path(args.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"],
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.train()
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def reward_of(row: dict) -> float:
        if args.reward == "outcome":
            return 1.0 if row["solved"] else 0.0
        valid = sum(1 for turn in row["turns"] if turn.get("tool") and not turn.get("error"))
        return round((1.0 if row["solved"] else 0.0) + 0.1 * min(valid, 3) - 0.3 * row["tool_errors"], 4)

    # Preflight: a worker that gets 404s dies in its thread, the queue stays empty, and the loop
    # reports "0 rounds" as if it had run. The name has to be served before anything else starts.
    available = served_models(args.base_url)
    if args.adapter_name not in available:
        raise SystemExit(
            f"{args.adapter_name} is not served at {args.base_url} (available: {available}). "
            f"Load it first: POST /v1/load_lora_adapter with lora_path={args.adapter}"
        )

    version = Version(args.adapter_name)
    samples: queue.Queue = queue.Queue()
    stop = threading.Event()
    workers = [
        threading.Thread(target=rollout_worker, args=(i, args, version, samples, stop), daemon=True)
        for i in range(args.workers if args.mode == "async" else 1)
    ]
    for worker in workers:
        worker.start()

    sample_log = (out / "samples.jsonl").open("w", encoding="utf-8")
    update_log = (out / "updates.jsonl").open("w", encoding="utf-8")
    started = time.monotonic()
    trained = 0
    dropped = 0
    history: list[dict] = []

    for round_index in range(args.rounds):
        wanted = args.batch
        if args.mode == "sync":
            # Fresh samples only: wait for a full batch per task, so staleness is zero by construction.
            wanted = max(args.batch, args.samples_per_task * len(TASKS_BY_ID))
        batch: list[dict] = []
        while len(batch) < wanted:
            try:
                batch.append(samples.get(timeout=240))
            except queue.Empty:
                print("  no samples for 240 s; stopping")
                break
        if not batch:
            break

        current = version.name
        for sample in batch:
            sample_log.write(json.dumps({k: v for k, v in sample.items() if k != "turns"}, sort_keys=True) + "\n")
        keep, too_old = credit.partition(batch, current, args.stale_bound)
        freshness = [row["staleness"] for row in batch]
        dropped += len(too_old)
        if not keep:
            print(f"  round {round_index}: every sample was too stale; nothing trained")
            continue

        loss, used = train_step(model, tokeniser, optimiser, keep, reward_of)
        trained += used
        singleton = sum(1 for rows in _by_task(keep).values() if len(rows) < 2)
        record = {
            "round": round_index,
            "singleton_groups": singleton,
            "trained_on": len(keep),
            "used": used,
            "dropped": len(batch) - len(keep),
            "mean_staleness": round(statistics.mean(freshness), 2),
            "max_staleness": max(freshness),
            "solved_in_batch": sum(1 for s in keep if s["solved"]),
            "loss": round(loss, 4),
            "rounds_seconds": round(time.monotonic() - started, 1),
        }
        history.append(record)
        update_log.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"round {round_index:2d}  batch {len(keep):2d}  used {used:2d}  dropped {record['dropped']}  "
            f"singletons {singleton}  "
            f"stale mean {record['mean_staleness']:.2f} max {record['max_staleness']}  loss {loss:.4f}  "
            f"t {record['rounds_seconds']:.1f}s"
        )

        # weight sync: save, load under a new name, and let the workers pick it up
        next_index = credit.version_index(current) + 1
        next_name = f"p5-v{next_index}"
        target = Path(f"{args.model_path_prefix}-v{next_index}")
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target / "adapter")
        tokeniser.save_pretrained(target / "adapter")
        # The served path is the saved path: both live under the volume mounted at /models.  Deriving
        # it from the same variable is the only way the two cannot disagree.
        load_adapter(args.base_url, next_name, f"{target.as_posix()}/adapter")
        version.bump(next_name)

    stop.set()
    sample_log.close()
    update_log.close()

    summary = credit.summarise(history, time.monotonic() - started)
    summary.update({"mode": args.mode, "reward": args.reward, "workers": args.workers,
                    "stale_bound": args.stale_bound, "final_adapter": version.name})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
