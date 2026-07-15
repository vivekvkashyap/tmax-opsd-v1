# tmax-opsd-v1

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers)-v1 environment that packages the
[TMax](https://arxiv.org/abs/2606.23321) terminal-agent dataset (14,601 Docker-image tasks) with a
TMax-faithful **vanillux** harness, plus a set of per-task **hints** for on-policy self-distillation
(OPSD) training.

It supports two training signals over the same tasks:
- **GRPO** — the binary solved/not-solved reward from each task's hidden verifier.
- **OPSD** — dense per-token distillation from the policy conditioned on a method-level **hint**
  (the `demo` field). This repo ships the generated hints.

## What's in here

| Path | What |
|------|------|
| `tmax_opsd_v1/` | the environment package (taskset, harness, reward, dataset builder) |
| `data/demos/<task_id>.md` | **9,959** generated OPSD hints, one file per task |
| `scripts/hintgen/` | the hint-generation pipeline + `RESUME.md` (how to generate the rest) |
| `scripts/prepare_data.py` | build `tasks.parquet` ahead of time |
| `docs/superpowers/specs/` | the hint-generation design spec |

`data/tasks.parquet` and `data/raw/` are **not committed** (regenerable — see [Data](#data)).

## The environment

- **Taskset** `TMaxTaskset` — joins Ai2's two TMax releases (`allenai/TMax-15K` and
  `allenai/tmax-15k-open-instruct`); each task runs in its own Docker image. Built lazily on first
  `load()` if `tasks.parquet` is absent.
- **Harness** `VanilluxHarness` — a single persistent-shell `bash` tool, the
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` submit marker, 10k head/tail observation truncation, and a
  format-error nudge — matching TMax's mini-SWE-agent-derived harness.
- **Reward** `solved` — binary 0/1 from the task's hidden pytest verifier
  (`/logs/verifier/reward.txt`), run once at scoring time. Plus `@metric pass_fraction` and
  `@metric hacked` (an oracle-copy detector), which are logged but never summed into the reward.
- **`validate()`** — a pre-flight fixture check that flags dead/broken images (a known ~15% of the
  corpus); pair with `blocklist_path` to skip them.
- Public API: `from tmax_opsd_v1 import TMaxTaskset, VanilluxHarness`.

Each task row (`TMaxData`) carries `task_id`, taxonomy fields (`domain`, `skill_type`, …), the
`verifier_kind`, and a `demo` slot for the OPSD hint. `truth` (the answer key) and `test_script`
(the verifier) are excluded from every serialization — they are never shown to the policy.

## The OPSD hints

OPSD (on-policy self-distillation) teaches the policy from a *hint-conditioned* version of itself:
the hint is prepended, the policy's own rollout is prefill-scored under it, and the resulting
per-token log-probs become the distillation target. The hint must teach the **method** without
leaking the answer.

The hints in `data/demos/<task_id>.md` are exactly that — short, method-level demonstrations
generated with **Claude Sonnet 5** from each task's `truth`, with the literal answer and any
oracle/shortcut path redacted (the discover-vs-given rule; see the design spec). Each is a single
paragraph of imperative, terminal-oriented guidance.

**Coverage: 9,959 / 14,601 tasks.** The remainder are either not yet generated or a small set of
offensive-security-framed tasks that Claude's safety filter declines — those stay GRPO-only.

### Wiring hints into a training run

The env exposes the hint via `TMaxData.demo`, which OPSD reads through its `demo_key`. In an
orchestrator config, point the OPSD env entry at this field and restrict it to hinted tasks:

```toml
[[orchestrator.train.env]]
id = "tmax-opsd-v1"                                 # GRPO — all tasks

[[orchestrator.train.env]]
id = "tmax-opsd-v1"
algo = { type = "opsd", demo_key = "demo" }         # OPSD — hinted subset
tasks = [ ... ]                                      # task_ids that have a demo
```

To load the hints onto the tasks, join `data/demos/<task_id>.md` into the `demo` column by
`task_id` at dataset-build time.

### Generating / resuming the hints

The full set is generated with Claude subagents (no API key needed — runs on a Claude plan), in
resumable chunks with a mapping-integrity gate. See **`scripts/hintgen/RESUME.md`** for the exact
loop (slice → generate → gate → repair → repeat).

## Data

`tasks.parquet` is built from the two public allenai releases and cached under
`$TMAX_OPSD_DATA_DIR` (default `~/.cache/tmax-opsd-v1/`). Build it ahead of time:

```bash
uv run python scripts/prepare_data.py
```

…or let `TMaxTaskset.load()` build it lazily on first use. It is not committed because it contains
synthetic secret-shaped strings (fake keys/tokens in the security-themed task content) that trip
GitHub's secret scanning.

## Install & test

```bash
uv sync --all-extras
uv run pytest            # docker-marked tests are skipped by default
```

Runtime: use the `prime` runtime (Prime Sandboxes) or a Linux `docker` host. Set
`TMAX_OPSD_DATA_DIR` to control where the built parquet is cached.
