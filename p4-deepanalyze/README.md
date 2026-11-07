# P4 - a data-analysis agent, end to end

**Status: measured.** The loop runs end to end on the Spark; Chapter 10's Project 4 lessons are
written from two runs of it.

| run | executor | solved | sandbox errors | code runs |
| --- | --- | --- | --- | --- |
| `p4-20261107-190738` | `python:3.12-slim` | 1/6 | **6** (all `No module named 'pandas'`) | 6 |
| `p4-20261107-190831` | `arlh-p4-sandbox:1` (pandas 2.2.3) | **4/6** | 0 | 6 |
| `p4-20261107-204926` | plus forced repair and the "new process" prompt | **5/6** | 0 | 7 |

## The cold start

Twelve supervised turns built by running the oracles; held-out evaluation on the same six schemas
over different rows (`fixtures/tasks-heldout.jsonl`). Adapter: `env/sft.py`, LoRA r=16.

| policy | training | held-out | compliance | code runs |
| --- | --- | --- | --- | --- |
| `qwen3-1.7b-base` | - | 5/6 | 100% | 7 |
| `p4-coldstart` | 12 epochs, lr 2e-4, loss 0.0001 | 5/6 | **83%** | 5 |
| `p4-coldstart-v2` | 4 epochs, lr 1e-4, loss 0.076 | **6/6** | 100% | 6 |

The fine-tune fixes the behaviour the demonstrations taught - the base model's reversed
`stock - reorder_level` disappears on rows it never saw - and training to loss 0.0001 costs the
marker format (v1 reproduced the oracle verbatim, without `<code>`). Stopping at 0.076 keeps both.

Same policy (`qwen3-1.7b-base`, temperature 0), same six fixtures, same prompt, `code_sha
git:62fc34e`. Only the executor changed. The 1/6 is the more instructive number: every program
asked for a library that was not installed, so the environment - not the model - decided the score,
and only `sandbox_errors` in the summary makes that visible.
Book chapter: Chapter 10 (end-to-end projects), project 4 in the book's appendix.
Compute tier: ⚠️ - a 0.6B policy end to end; an 8B model only for QLoRA supervised
fine-tuning (`configs/rented-*.yaml`), never for the RL stage on the reference machine.

## What you build

An agent that reads a table, writes and runs code, and answers - inside a sandbox:

```
question + CSV -> agent (writes code) -> sandbox executes -> observation -> answer
                                            |
                                            +-- this is the verifier: it runs, so it
                                                cannot be talked out of the truth
```

The interesting parts are the two the book keeps returning to:

* **The sandbox is a reward source, not a safety detail.** A container that hangs, or one
  whose failure the pipeline records as a wrong answer, silently changes what the model
  learns. Timeouts must be distinguishable in the row.
* **The trace format is the contract.** A code-execution turn is a `model` segment
  followed by a `tool` segment; the mask follows from that, and `../common/mask.py`
  refuses a row where it does not.

## Data

`xlangai/DS-1000` (CC-BY-SA-4.0 - share-alike, so nothing from it may be committed) and
`stanfordnlp/wikitablequestions` / `wenhu/tab_fact` (CC-BY-4.0). See
`../datasets/README.md`; the digests under `../datasets/digests/` are what a run record
should cite.

## What is here

- `src/p4_deepanalyze/`: the marker protocol, the Docker and local sandboxes, the episode loop,
  the verifier, and the CLI (`run_p4.py`).
- `fixtures/build_tasks.py`: six tables whose expected answers are recomputed from their own CSV,
  so a fixture cannot disagree with its oracle. `--check` fails if `tasks.jsonl` is stale.
- `checks/check_p4.py`: the gate's six checks - fixtures reproduced through the sandbox, an oracle
  episode that solves, prose that counts as a protocol error, a timeout that is not a wrong answer.
- `checks/p4_claims.py`: the two run numbers Chapter 10 quotes, regenerated from `episodes.jsonl`.
- `sandbox/Dockerfile`: the executor, with the reason it is not `python:3.12-slim`.

## What is not here yet

- The supervised stage (the cold start). The demonstrations exist - the oracles in `check_p4.py` -
  but no run trains on them; Chapter 7 is the subject.
- The RL stage: designed in Chapter 10, Lesson 4, and blocked on the training runtime that
  Chapter 9 could not start on this hardware.
- Real tables from DS-1000 and the table-question corpora; the fixtures are committed because the
  share-alike corpora cannot be.
