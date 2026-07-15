# OPSD Hint Generation — Design Spec

**Date:** 2026-07-15
**Status:** Approved design, pre-implementation
**Scope:** `tmax-opsd-v1/` only. No changes to `prime-rl` or `verifiers`.

## 1. Goal

Produce a `demo` (hint) for every one of the 14,601 tasks in `tmax-opsd-v1`, so
that a mixed **GRPO + OPSD** training run works. OPSD (on-policy
self-distillation) requires a per-task demonstration string; today the field is
empty, so the OPSD half of a mixed run crashes on the first rollout. Filling it
is pure data work in our own repo.

Each hint is a **method-level demonstration**: it teaches *how an expert
approaches the task* (so the hinted self-teacher produces better trajectories to
distill from) while withholding the literal answer and any copyable shortcut.

## 2. Background

A mixed run is two env entries pointing at the same env — one `type = "grpo"`,
one `type = "opsd"`. The trainer packs their samples into one batch and sums
three independently-normalized loss components (`rl`, `ce`, `ref_kl`); GRPO
writes `rl`, OPSD writes `ref_kl`. This is already supported by prime-rl
(`orchestrator/algo/`, per-env `algo` config, `trainer/rl/`). The only reason a
mixed run fails today:

1. OPSD looks up the hint under `demo_key`, default `"demonstration"`; our field
   is `demo` (`configs/algorithm.py:268`, `tmax_opsd_v1/taskset.py:66`).
2. The `demo` value is always `None` — the column isn't in `build.py`'s
   `COLUMNS`, and `taskset.py`'s row-builder never reads it. OPSD raises
   `ValueError` on `None` (`orchestrator/algo/opsd.py:53-62`).

This spec covers generating the hint data and wiring it through. The
`demo_key = "demo"` config fix and the two-entry TOML are documented here but are
trivial and not the substance of the work.

### What `truth` contains (the generation source)

`truth` (avg ~2,000 chars, present on all 14,601 tasks) is the task author's
answer key. It has three kinds of content:

- **Verifier** — how success is graded (`verifier_kind`, metric/threshold,
  verification code).
- **Fixture/setup** — how the environment and ground-truth data are built
  (setup scripts, hidden generation parameters, input files).
- **Oracle/reference solution** — the correct answer, in one of two forms:
  - a **pre-built binary** at a path (e.g. `/app/oracle_parser`,
    `/app/legacy_router`) — ~1,510 tasks reference one (see §3);
  - **inline in the text** — the setup script and/or the literal expected
    answers (e.g. `rolling_avg = {"Alice": 30.0, ...}`).

The hint is manufactured from `truth`: keep the method, strip the answer and any
oracle path.

## 3. Scope

- **All 14,601 tasks get a hint.** Oracle-binary tasks (~1,510: 253 via the
  `oracle_path` column + more detectable only by scanning `truth` — `oracle`
  mentions, "Oracle/Reference/Legacy Program:" lines, `fuzz_equivalence`
  verifier) are **kept**. They are hintable — `truth` describes the method even
  when the answer is a binary — and stay in both the GRPO and OPSD sets.
- **Open-loop generation.** Generate once with an LLM, apply static
  leak/length checks, write. No per-task container run to validate; hint impact
  is measured in aggregate via OPSD run metrics.
- **Reward-hacking note (out of scope, tracked):** for `fuzz_equivalence`
  tasks the oracle binary sits in the container and copying it passes the
  verifier; `detect_hack` (`parsing.py:72`) logs but does not gate reward. This
  is a GRPO reward-hacking risk independent of hint generation. Not addressed
  here; watch the `detect_hack` metric on the first run.

## 4. Architecture

Pipeline: **generator (Batch API + system prompt) → leak-guard → `demos.parquet`
(keyed by `task_id`) → `build.py` left-join → `demo` column → `taskset.py` →
OPSD.**

### 4.1 Generator

- **Engine:** Anthropic **Batch API**, one request per task (async, ~50%
  discount, resumable).
- **Model:** `claude-sonnet-5` — committed choice for the run (hint quality is
  the whole ballgame). Recorded per-row in `gen_model` for provenance.
- **Input per request:** the fixed system prompt (§4.2) + user message
  `DESCRIPTION:\n{description}\n\nTRUTH:\n{truth}`.
- **Output:** the hint text only.

### 4.2 System prompt

```
You are writing a short METHOD DEMONSTRATION for a terminal-agent task. Your
output will be shown to a coding agent as expert guidance before it attempts
the task. It must teach HOW an expert approaches the problem so the agent
learns the method — never hand it the answer.

You are given:
- DESCRIPTION: what the agent itself sees.
- TRUTH: an internal answer key (setup scripts, expected outputs, verification
  logic, and sometimes a reference solution or a pre-built "oracle" binary).
  Use TRUTH only to understand the correct approach. It is NOT shown to the agent.

Write the demonstration so that it:
- Describes the METHOD: the key steps, the core technique, and the one or two
  places a naive attempt goes wrong. Assume the agent works in a Linux terminal,
  issuing shell commands and editing files.
- Is CONCISE: one focused paragraph or a short ordered list (aim under 180 words).
- Is IMPERATIVE and CONCRETE ("Read /home/user/data.csv…", "FFT the signal to
  find the dominant frequencies…"), grounded only in the given task. Do not
  invent details the inputs don't support.

NEVER include (these turn the demonstration into a cheat):
- Any path to a pre-built solution or oracle/reference/legacy binary
  (e.g. /app/oracle_parser, /app/legacy_router). Never tell the agent to run,
  copy, link, or read such an artifact — it must build its own solution.
- Literal expected answers, or any value the agent is meant to compute or
  discover: final numeric results, exact expected JSON/output, hidden
  fixture-generation parameters, or secrets/text the agent must recover
  (e.g. via OCR). Describe HOW to derive or recover them instead.

You MAY state values that are part of the task's own specification and that the
agent legitimately needs: input file paths, the required output path / entry
point, service ports, endpoint routes, required headers, and the success
metric/threshold.

Output ONLY the demonstration text — no preamble, no headings, no "Here is a
hint", no mention of TRUTH or oracle.
```

**Key rule — discover-vs-given.** Values the agent must *recover* (OCR'd text,
hidden signal params, computed answers) are taught as method, never stated.
Values that are part of the *task spec* the agent needs (input/output paths,
ports, routes, thresholds) may be stated. This split is encoded in the "NEVER"
vs "MAY" sections above.

### 4.3 Leak-guard (deterministic backstop)

The prompt is layer one; the LLM can silently violate it, so a deterministic
post-filter is the hard E0 guarantee. For each generated hint, keyed to its
task:

1. Extract candidate shortcut tokens from *that task's* `truth`: any executable
   path under `/app`, `/opt`, `/srv`, `/usr/local`, and any name following an
   "Oracle/Reference/Legacy Program|Binary|Path:" label.
2. If the hint contains any such token → **regenerate once**; if it still leaks,
   write the hint with `leak_flag = true` and **exclude that task_id from the
   OPSD `tasks:` list** (it stays in GRPO).
3. Enforce a length cap (reject/regenerate hints beyond the cap; default derived
   from the 180-word target, e.g. 1,400 chars).

Note: the leak-guard keys off the task's *actual `truth` text*, not the parsed
`oracle_path` column (which undercounts oracle references ~6×).

### 4.4 Storage — `data/demos.parquet`, keyed by `task_id`

Hints are stored **out of** `tasks.parquet`, in a sidecar where **each row is
one datapoint's hint, keyed by that datapoint's `task_id`** (unique across all
14,601 rows — verified):

```
data/demos.parquet
  task_id         str   primary key — the datapoint name (e.g. task_000004_b4949f3f)
  demo            str   the hint text (null if generation failed)
  gen_model       str   e.g. claude-sonnet-5
  prompt_version  str   system-prompt version tag (e.g. v1)
  leak_flag       bool  true if the leak-guard could not produce a clean hint
```

Rationale for a sidecar over an in-place column:
- **Cheap, independent regeneration** — iterating the prompt rewrites a small
  file, never rebuilds the 46 MB `tasks.parquet`.
- **Provenance per datapoint** — `gen_model`/`prompt_version`/`leak_flag` let us
  regenerate just the flagged or stale rows.
- **Resumable** — the generator upserts by `task_id`; a 14.6k Batch run re-runs
  only missing/flagged rows.

### 4.5 Wiring (two one-line code touches)

- `tmax_opsd_v1/build.py`: add `"demo"` to `COLUMNS`; after building base rows,
  left-join `demos.parquet` on `task_id` and fill `demo` (null where absent).
- `tmax_opsd_v1/taskset.py`: add `demo=row.get("demo")` to the `TaskData`
  construction (~line 193). Without this, `task.data.demo` stays `None` even
  when the column exists.

### 4.6 Mixed-run config (documented, trivial)

```toml
[[orchestrator.train.env]]
id = "tmax-opsd-v1"                                   # GRPO — all 14,601 tasks

[[orchestrator.train.env]]
id = "tmax-opsd-v1"
algo = { type = "opsd", demo_key = "demo" }           # demo_key fix
tasks = [ ... ]                                        # task_ids with a clean (non-leak_flag) demo
```

## 5. Data flow

1. Read `tasks.parquet` → for each `task_id`, build one Batch request
   (`description` + `truth`).
2. Submit Batch → collect responses.
3. Leak-guard each hint against its `truth`; regenerate-once or `leak_flag`.
4. Upsert rows into `data/demos.parquet` keyed by `task_id`.
5. `build.py` left-joins → `tasks.parquet` gains a `demo` column.
6. `taskset.py` reads `demo` onto `TaskData`; OPSD consumes it via
   `demo_key = "demo"`.

## 6. Error handling / edge cases

- **Generation failure / empty output for a task:** write `demo = null`; the
  task is absent from the OPSD `tasks:` list, stays in GRPO. No crash.
- **Persistent leak:** `leak_flag = true`, excluded from OPSD, stays in GRPO.
- **Thin `truth`:** if `truth` yields no usable method, the hint may be weak;
  open-loop accepts this and relies on aggregate OPSD metrics. (No per-task
  quality gate — that was the closed-loop option, explicitly not chosen.)
- **`demo_key` mismatch:** guarded by the config line; without it OPSD looks up
  `"demonstration"` and crashes — covered by §4.6.

## 7. Testing

Per repo convention (`AGENTS.md`: test pure logic only, conservative additions):
- Unit-test the **leak-guard** shortcut-token extraction and the containment
  check on a few hand-built `(truth, hint)` pairs (pure function, no I/O).
- Unit-test the **`build.py` left-join** fills `demo` correctly and leaves
  uncovered tasks `null` (small in-memory frames).
- Do **not** test the Batch API call or the prompt (runtime/framework glue).

## 8. Decisions made (for the record)

- Open-loop generation (not closed-loop validated). — user
- Keep all oracle-binary tasks in both GRPO and OPSD. — user
- Claude Batch API, `claude-sonnet-5` (committed, not a fallback). — user
- Sidecar `demos.parquet` keyed by `task_id`, not an in-place column. — recommended
- Reward-hacking on oracle tasks: out of scope, monitor `detect_hack`. — noted
```
