# P4 - a data-analysis agent, end to end

**Status: not started.** Written with Chapter 10.
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

## What is not here yet

- The Docker sandbox and its health checks.
- The supervised stage: this project needs a cold start before RL will do anything, which
  is Chapter 7's subject rather than Chapter 10's.
