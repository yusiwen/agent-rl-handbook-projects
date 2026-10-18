"""Step A: one task, five samples, two rounds - does a gradient reach the next round's sampling?

    python3 /env/rl_step.py --task files-r06 --samples 5 --rounds 2

This is the smallest run that can answer the only question step A asks. Not "did the policy improve"
- that is step B - but: does the weight the credit machinery computes reach the tokens the model
actually produced, change the adapter, and get used by the next round's generation?

Why generation happens in this process rather than over HTTP: a served policy would need the new
weights pushed to the server between rounds, which is Chapter 9's weight-sync problem. In-process,
the model that generates round N+1 is the model that stepped in round N by construction, so the
property is structural instead of something to verify.

What is reused rather than reimplemented: the environment, the prompt, the strict action parser, the
group advantages, the turn weights. What is new: a policy that generates locally and keeps the
prompt/response pairs it produced, a re-scoring pass that turns those tokens into log-probabilities
with a graph attached, and the step that consumes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2_reagent import actions, env, prompt
from p2_reagent.agent import run_episode
from p2_reagent.loop import advantage_for, progress_reached, summarise_groups
from p2_reagent.train import turn_weights


class LocalPolicy:
    """Generates in this process, and remembers every (prompt, reply) pair it produced."""

    def __init__(self, model, tokeniser, temperature: float, max_new_tokens: int) -> None:
        self.model = model
        self.tokeniser = tokeniser
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.version = "local"
        self.per_episode: list[list[tuple[str, tuple[int, ...]]]] = []
        self._current: list[tuple[str, tuple[int, ...]]] = []
        self.unparsed = 0

    def begin(self) -> None:
        self._current = []

    def finish(self) -> None:
        self.per_episode.append(self._current)

    def choose(self, attempt) -> str:
        history = [(action, result.signal, result.observation) for action, result in attempt.history]
        messages = prompt.render(attempt.task, attempt.observation, attempt.menu(), history)
        text = self.tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokeniser(text, add_special_tokens=False)["input_ids"]
        inputs = torch.tensor([prompt_ids], device="cuda")
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.tokeniser.pad_token_id or self.tokeniser.eos_token_id,
            )
        reply_ids = tuple(int(token) for token in output[0, inputs.shape[1] :].tolist())
        reply = self.tokeniser.decode(list(reply_ids), skip_special_tokens=True)
        self._current.append((text, reply_ids))
        parsed = actions.parse_action(reply, attempt.menu())
        if not parsed.ok:
            self.unparsed += 1
        return parsed.action if parsed.ok else (parsed.raw or "<unparsed>")


def reply_logprobs(model, tokeniser, prompt_text: str, reply_ids: tuple[int, ...]):
    """The mean log-probability this model gives that reply, with its graph attached."""
    prompt_ids = tokeniser(prompt_text, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prompt_ids + list(reply_ids)], device="cuda")
    logits = model(input_ids=ids).logits.float()[0]
    # logits[t] predicts token t+1, so the reply's first token is predicted by the last prompt token.
    start = len(prompt_ids) - 1
    window = logits[start : start + len(reply_ids)]
    chosen = torch.log_softmax(window, dim=-1)
    targets = torch.tensor(list(reply_ids), device="cuda").unsqueeze(1)
    return chosen.gather(1, targets).squeeze(1)


def adapter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" in name:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return "sha256:" + digest.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="step A: does a gradient reach the next round?")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--start", default="/work/sft/adapter")
    parser.add_argument("--task", default="files-r06")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", default="/work/rl")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokeniser = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.start, is_trainable=True)
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    print(f"resumed {args.start}: {sum(p.numel() for p in trainable):,} trainable parameter(s)")

    policy = LocalPolicy(model, tokeniser, args.temperature, args.max_new_tokens)
    digests = {"r0": adapter_digest(model)}

    for round_index in range(args.rounds):
        started = time.time()
        episodes = []
        for sample in range(args.samples):
            policy.begin()
            episodes.append(
                run_episode(policy, args.task, sample_index=sample, policy_version=f"local-r{round_index}")
            )
            policy.finish()

        rewards = {episode.rollout.episode_id: episode.rollout.reward() for episode in episodes}
        summaries = summarise_groups(episodes, rewards)
        terms = []
        weighted_turns = 0
        for episode, captured in zip(episodes, policy.per_episode):
            advantage = advantage_for(summaries, episode, rewards)
            weights = turn_weights(episode, advantage, mode="trajectory")
            episode.rollout.reward_terms["rule_turn"] = progress_reached(episode)
            for (text, reply_ids), weight in zip(captured, weights):
                if weight == 0.0 or not reply_ids:
                    continue
                weighted_turns += 1
                terms.append(-weight * reply_logprobs(model, tokeniser, text, reply_ids).mean())

        loss_value = 0.0
        if terms:
            loss = torch.stack(terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            loss_value = loss.item()

        digests[f"r{round_index + 1}"] = adapter_digest(model)
        model.save_pretrained(out / f"adapter-r{round_index + 1}")
        solved = sum(1 for episode in episodes if episode.solved)
        print(
            f"round {round_index}: solved {solved}/{len(episodes)} "
            f"mean reward {sum(rewards.values()) / len(rewards):+.3f} loss {loss_value:.5f} "
            f"weighted_turns {weighted_turns} unparsed {policy.unparsed} "
            f"{time.time() - started:.0f}s digest {digests[f'r{round_index + 1}']}"
        )
        policy.unparsed = 0

    print()
    print(
        f"RL_STEP task={args.task} samples={args.samples} rounds={args.rounds} lr={args.lr} "
        f"digests={json.dumps(digests)} digest_changed={len(set(digests.values())) > 1}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
