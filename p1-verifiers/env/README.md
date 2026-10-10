# P1's environment

One served model, on the reference machine. That is the whole thing: P1 needs an
OpenAI-compatible chat endpoint and a `/tokenize` endpoint, and this directory is what provides
them.

Where it runs: a DGX Spark (GB10, 128 GB unified memory, aarch64). The GPU stanza in
`docker-compose.yaml` is the form that works there - it resolves through CDI, because this host
has no `nvidia` container runtime. Nothing here is Spark-specific except that stanza and the two
cache paths.

## Start it, stop it

```bash
cd p1-verifiers/env
docker compose up -d          # no --build: the image is pinned, not built
docker compose logs -f p1-model
docker compose down           # stops and removes the container; weights stay in the cache
```

The first start takes minutes: vLLM loads the weights and compiles kernels. The second start is
quick because the compile cache is a mounted volume. Do not read "container started" as "model
ready" - check it:

## Is it up?

```bash
BASE=http://192.168.2.40:8000

curl -s $BASE/v1/models | python3 -m json.tool        # 1. the name, exactly as clients must use it
curl -s -X POST $BASE/tokenize -H 'Content-Type: application/json' \
     -d '{"model":"qwen3-8b-nvfp4","prompt":"one two three"}'   # 2. the tokeniser P1 counts with
curl -s -X POST $BASE/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"qwen3-8b-nvfp4","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":64}'
```

All three matter, for different reasons:

1. **`/v1/models`** tells you the served name. Client code must use that string and no other.
2. **`/tokenize`** is not optional here. P1 records a token count next to the name of what counted
   it; if this endpoint is missing, `run_p1_live.py` refuses to run rather than quietly falling
   back to counting words - a row with a made-up token count is worse than no row.
3. **One chat completion** proves the path P1 actually uses, including that the model emits
   something the verifier can extract.

`run_p1_live.py` performs all three checks itself before it spends a token, so a manual run of the
above is mainly for when something is wrong and you want to see which of the three it is.

## Which model

The default is **`Qwen/Qwen3-1.7B`**: in the Qwen3 family the plain name is the chat model and
`-Base` is the pretrained sibling. Instruction tuning is not a nicety here - P1's system prompt
asks for the final answer on its own line, and a base checkpoint does not follow instructions.
The number you get from one is a mixture of "cannot do the task" and "did not do as asked", which
is exactly the distinction P1 exists to make. 1.7B is the family's nearest size to the 1.5B this
environment was asked for, and it starts in seconds, which matters when every task is sampled
twice.

So:

| Option | Weights | What to expect |
|---|---|---|
| `Qwen/Qwen3-1.7B` (default) | ~3.4 GB to download | instruction tuned, same architecture as everything else here; a real small-model pass rate |
| `nvidia/Qwen3-8B-NVFP4` | 6 GB, already cached | bigger but a *base* checkpoint: the loop works, the pass rate will be low, and that is not the verifier's fault |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3.1 GB | literally 1.5B, and no reasoning block at all - the simpler option if the thinking switch below is unwelcome |
| `Qwen3.6-35B-A3B-NVFP4` | 22 GB, cached | the fastest setup measured on this machine (MoE, 3B active); it runs under SGLang on port 30000, not under this compose file |

Switching is one variable:

```bash
ARL_MODEL=Qwen/Qwen2.5-1.5B-Instruct ARL_SERVED_NAME=qwen2.5-1.5b docker compose up -d
```

## Fetch the weights without starting a server

The same image does the downloading, so the cache lands exactly where the container will look
for it:

```bash
docker run --rm --entrypoint hf \
  -v /home/yusiwen/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.27.1 download Qwen/Qwen3-1.7B
```

Otherwise the first `docker compose up -d` downloads it, and `docker compose logs -f` is where
you watch that happen.

## One flag on the P1 side: `--no-thinking`

Qwen3's chat models reason before answering unless the chat template is told not to. For a
single-turn task that is tokens spent on a scratchpad nobody reads - and P1's verifier only ever
looks at the last line - so the runner has a switch:

```bash
python3 p1-verifiers/checks/run_p1_live.py --base-url ... --model qwen3-1.7b --no-thinking
```

It sends `chat_template_kwargs={"enable_thinking": false}` with every request and writes the mode
into the run header, because a run with thinking on is not comparable to one with it off.

## What this deliberately does not do

- **No training and no GPU sharing.** One model per machine: the pool is unified, so a second
  model is a second set of weights in the same 128 GB, not a second GPU.
- **No autostart.** `restart: "no"` - this machine is powered off daily, and a model that comes
  back by itself is a surprise rather than a convenience.
- **No proxy, no auth, no TLS.** P1 talks to it directly on the LAN. Putting a gateway in front
  is a different exercise, and adding one here would hide the endpoint the tokens are actually
  counted by.
- **No healthcheck.** See comment 4 at the top of `docker-compose.yaml`; the reason is a real
  afternoon lost to a healthcheck that was wrong while the service was right.

## The four traps, all in the compose file

They are commented where they bite rather than collected here: the entrypoint that already says
`vllm serve`, the `~` that becomes `/root`, the served name that must match exactly, and the
healthcheck that should not exist. `P1`'s run record cites this environment as
`server-tokenize/<model>@<host>`, so a row produced against a differently-named server is
distinguishable from one produced against this one.
