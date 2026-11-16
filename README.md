# nemotron-streaming-loader

An expert-granular streaming loader for running NVIDIA's
Nemotron-3-Nano-30B mixture-of-experts checkpoint on a machine whose
memory cannot hold it.

## The problem

The checkpoint is about 62 GB on disk. On a small machine (the runs
here were measured on a home machine with about 16 GB of memory and
a 4 GiB graphics card), the model runs with its experts offloaded to
disk. The shipped offload path streams whole expert banks through
memory at every forward pass — including a pass over the experts the
router did not choose, whose only effect on the output is a
constant. At the full shape below, one decode step took 131.20
seconds, and almost all of it was that disk traffic.

## The fetch rule

Once the router has chosen, fetch exactly those experts — 8 of 128
per expert layer — in large sequential reads (32 MiB units), serve
them to the model code unchanged, and let the unchosen experts stay
on disk. The loader changes how the tensors arrive, never what is
computed.

## What is here

- `loader-rs/` — the loader daemon, in Rust. It indexes the
  checkpoint's shards once, then serves per-layer expert tensors
  over a local socket in route order. Build it with
  `cargo build --release` (see `loader-rs/Cargo.toml`).
- `loadercell*.py`, `streamprobe*.py` — the run cells and probes
  that drive the model against the daemon, with per-forward timers.
- `verify_reader.py` — the byte-check: it compares the daemon's
  served tensors against the checkpoint files directly.
- `run-stage2*.sh` — the drivers for the measured runs.
- `runs/`, `logs/` — the run records: identity outputs, timers,
  stage logs, daemon logs, and the byte-check output.
- `modeling/modeling_nemotron_h.py` — the patched modeling file the
  measured runs used (the cache-routing repair for this checkpoint,
  posted in full in the discussion thread linked below). The loader
  itself is modeling-agnostic; the runs paired it with this file.

## Measured results

Same machine, same shape as the repair runs in the thread (the full
30B shape, experts offloaded), the short pair measured in the same
session:

| | The ordinary way | With the loader |
| --- | --- | --- |
| One decode step | 131.20 s | 9.51 s |
| The whole 6-token pass | 841.15 s | 121.47 s |
| Major page faults | 722,011 | 6,945 |

That is about 14× faster per decode step. The tokens stayed the
model's own: 6 of 6 identical on the short pass, and 32 of 32
against the patched 32-token run from the repair. The reader
byte-check passed 213 of 213 sampled tensors. At a smaller test
shape the same pattern held: 2.7× faster per step, tokens again
identical.

One honest negative: keeping fetched experts in memory between
steps (retention) never engaged at the full shape under the memory
budget the shape could afford — zero retained hits — so the gain
above is the fetch path alone. Where a smaller shape left room,
retention did engage and cut each step's fetch traffic from
2.754 GB to between 1.26 and 1.70 GB. The identity comparison so
far reaches 32 tokens; a longer run is the obvious next test.

## The discussion

The full story — the cache repair this work follows on from, and
the follow-on post these files belong to — is in the model card's
discussion thread:

https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions/65
