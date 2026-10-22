"""The join: sample from a served policy, train on those samples, push the result back.

This is the one loop neither half of Project 2 had.

- `run_p2_served.py` samples over HTTP and never updates: it can prove the service works, not that
  feedback reaches the weights.
- `rl_step.py` updates where it generates, so there is no weight-sync step to get wrong - and no
  weight sync to get right either. It is not the shape a real system has.

`rl_served.py` runs the real shape in the smallest honest version of it:

    round N    sample  -> HTTP, model = policy-rN (an adapter registered on the server)
               score   -> the rule critic, as in every other stage
               train   -> in this process, from the same adapter the server is serving
               sync    -> save adapter-rN+1, POST /v1/load_lora_adapter as policy-rN+1
    round N+1  sample  -> HTTP, model = policy-rN+1

Two copies of the base model are alive at once - the server's and this process's - which is what the
colocated trainer/server pattern costs. The base weights are shared on disk; only the adapter moves
over the wire (16-18 ms against a 156 s restart, measured in this project's README).

The honesty notes that belong with the numbers:

* The reply is **re-tokenised** in this process rather than taken from the server's token ids. With
  one tokenizer on both sides the sequence agrees; a different tokenizer would make the
  log-probabilities silently wrong, which is why the script prints the tokenizer it loaded.
* The update is on-policy only in the loose sense: the samples came from the served weights and the
  gradient is computed by the same weights loaded here. If the server had drifted to a different
  adapter, nothing in this script would notice. Chaining `--serve-policy` to the name this script
  registers is what keeps them together, and the per-round line prints the name that was actually
  used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.runs import RunHeader, RunWriter, round_record
from p2_reagent import actions, env, prompt
from p2_reagent.agent import run_episode
from p2_reagent.loop import advantage_for, progress_reached, summarise_groups

#: The run record is attributed to this project, not to the one whose machinery it reuses.
PROJECT = "p3-verl"
from p2_reagent.train import turn_weights


def http_json(url: str, payload: dict | None = None, timeout: float = 180.0) -> dict:
    """POST when given a payload, GET otherwise; the LoRA endpoints answer with a sentence."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            text = handle.read().decode()
            try:
                return json.loads(text or "{}")
            except json.JSONDecodeError:
                return {"text": text.strip()}
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{url}: {error.code} {error.read().decode()[:300]}") from error


class ServedPolicy:
    """Samples from whatever adapter the server serves, and remembers the text it used."""

    def __init__(self, base_url, tokeniser, model_name, temperature, max_tokens):
        self.base_url = base_url.rstrip("/")
        self.tokeniser = tokeniser
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.version = model_name
        self.per_episode: list[list[tuple[str, str, float]]] = []
        self._current: list[tuple[str, str, float]] = []
        self.unparsed = 0
        self.calls = 0

    def begin(self) -> None:
        self._current = []

    def finish(self) -> None:
        self.per_episode.append(self._current)

    def choose(self, attempt) -> str:
        history = [(action, result.signal, result.observation) for action, result in attempt.history]
        messages = prompt.render(attempt.task, attempt.observation, attempt.menu(), history)
        prompt_text = self.tokeniser.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": 0.95 if self.temperature > 0 else 1.0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = http_json(self.base_url + "/v1/chat/completions", payload)
        self.calls += 1
        reply = (response["choices"][0]["message"]["content"] or "").strip()
        self._current.append((prompt_text, reply, 1.0))
        parsed = actions.parse_action(reply, attempt.menu())
        if not parsed.ok:
            self.unparsed += 1
        return parsed.action if parsed.ok else (parsed.raw or "<unparsed>")


def reply_logprobs(model, tokeniser, full_ids, reply_ids):
    """log p(reply tokens | prompt), with a graph attached."""
    inputs = torch.tensor([list(full_ids)], device="cuda")
    logits = model(inputs).logits.float()
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    start = len(full_ids) - len(reply_ids)
    targets = torch.tensor(list(reply_ids), device="cuda")
    positions = torch.arange(start - 1, start - 1 + len(reply_ids), device="cuda")
    return log_probs[positions, targets]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="served sampling, in-process update, HTTP sync")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--start", default="/work/sft/adapter", help="the adapter this process loads")
    parser.add_argument("--serve-url", default="http://127.0.0.1:8003")
    parser.add_argument("--serve-policy", default="policy-sft", help="a name the server already serves")
    parser.add_argument("--serve-path-root", default="/work", help="this container's view of the volume")
    parser.add_argument("--server-path-root", default="/models", help="the server's view of the volume")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="/work/rl-serve")
    parser.add_argument("--runs-dir", default="/work/runs")
    parser.add_argument("--code-sha", default="git:unknown")
    args = parser.parse_args(argv)

    tokeniser = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.start, is_trainable=True)
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    print(f"trainer loaded {args.start}: {sum(p.numel() for p in trainable):,} trainable parameters")
    print(f"tokenizer: {tokeniser.name_or_path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config_hash = hashlib.sha256(
        json.dumps(vars(args), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    writer = RunWriter(
        RunHeader(
            run_id="p3-rl-served",
            env_version=env.ENV_VERSION,
            config_hash="sha256:" + config_hash,
            code_sha=args.code_sha,
            project=PROJECT,
            dataset=",".join(args.tasks),
            dataset_digest="sha256:" + hashlib.sha256(",".join(args.tasks).encode()).hexdigest()[:16],
            notes=(
                f"samples={args.samples}; rounds={args.rounds}; epochs={args.epochs}; lr={args.lr}; "
                f"served={args.serve_policy}; sync=load_lora_adapter"
            ),
        ),
        root=args.runs_dir,
    )
    print(f"run record: {writer.dir}")

    served_name = args.serve_policy
    for round_index in range(args.rounds):
        started = time.time()
        policy = ServedPolicy(
            args.serve_url, tokeniser, served_name, args.temperature, args.max_new_tokens
        )

        episodes = []
        per_task: dict[str, list] = {}
        for task in args.tasks:
            group = []
            for sample in range(args.samples):
                policy.begin()
                group.append(run_episode(policy, task, sample_index=sample, policy_version=served_name))
                policy.finish()
            per_task[task] = group
            episodes.extend(group)

        writer.append_rollouts(round_index, [episode.rollout for episode in episodes])
        writer.append_round(round_record(round_index, [episode.rollout for episode in episodes]))

        rewards = {episode.rollout.episode_id: episode.rollout.reward() for episode in episodes}
        advantages = {}
        for task, group in per_task.items():
            summaries = summarise_groups(group, rewards)
            for episode in group:
                advantages[episode.rollout.episode_id] = advantage_for(summaries, episode, rewards)

        pairs = []
        weighted_turns = 0
        for episode, captured in zip(episodes, policy.per_episode):
            advantage = advantages[episode.rollout.episode_id]
            weights = turn_weights(episode, advantage, mode="trajectory")
            episode.rollout.reward_terms["rule_turn"] = progress_reached(episode)
            for (prompt_text, reply_text, _), weight in zip(captured, weights):
                if weight == 0.0 or not reply_text:
                    continue
                prompt_ids = tokeniser(prompt_text, add_special_tokens=False)["input_ids"]
                reply_ids = tokeniser(reply_text, add_special_tokens=False)["input_ids"]
                if not reply_ids:
                    continue
                weighted_turns += 1
                pairs.append((weight, tuple(prompt_ids + reply_ids), tuple(reply_ids)))

        losses = []
        for _ in range(args.epochs if pairs else 0):
            terms = [
                -weight * reply_logprobs(model, tokeniser, full_ids, reply_ids).mean()
                for weight, full_ids, reply_ids in pairs
            ]
            loss = torch.stack(terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            losses.append(loss.item())

        next_name = f"policy-r{round_index + 1}"
        adapter_dir = out / f"adapter-r{round_index + 1}"
        model.save_pretrained(adapter_dir)
        server_path = str(adapter_dir).replace(args.serve_path_root, args.server_path_root, 1)
        started_sync = time.time()
        loaded = http_json(
            args.serve_url + "/v1/load_lora_adapter",
            {"lora_name": next_name, "lora_path": server_path},
        )
        sync_ms = int((time.time() - started_sync) * 1000)

        solved = sum(1 for episode in episodes if episode.solved)
        print(
            f"round {round_index}: served={served_name} solved={solved}/{len(episodes)} "
            f"loss={(losses[-1] if losses else 0.0):.5f} weighted_turns={weighted_turns} "
            f"calls={policy.calls} sync->{next_name} {sync_ms}ms {loaded.get('text', '')} "
            f"total={time.time() - started:.0f}s"
        )
        served_name = next_name

    served = http_json(args.serve_url + "/v1/models").get("data", [])
    print("served models now:", [entry["id"] for entry in served])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
