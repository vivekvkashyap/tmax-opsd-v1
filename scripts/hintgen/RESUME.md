# Resuming OPSD hint generation

This directory holds everything needed to generate the per-task OPSD hints (the
`demo` slot) and resume the run at any time. Hints are written **one file per
datapoint** at `data/demos/<task_id>.md`, containing only the hint text.

## Current state (2026-07-15)

- **~9,960 / 14,601 hints generated (~68%)**, mapping-verified clean.
- Remaining **~4,600 tasks** not yet generated.
- A handful (~40) of offensive-security-framed tasks are **permanently blocked**
  by Claude's cyber-safety filter (both Sonnet 5 and Opus reject them). They get
  no hint and stay GRPO-only by design. Add them to a blocklist (below) so they
  aren't retried each chunk.

## Why generation needs Claude Code (not a plain script)

The hints are written by **Claude Sonnet 5 subagents**, driven either by the
Claude Code `Workflow` tool or by dispatched subagents — this runs on your
Claude Max plan, no API key. The Python scripts here only do the plumbing
(slicing tasks, the mapping gate, repair). The actual writing is the model.

## The loop (repeat until coverage is complete)

Run each step from the repo root (`tmax-opsd-v1/`).

### 1. Slice the remaining tasks into batches
```
uv run python scripts/hintgen/slice_remaining.py 12
```
Writes `.hintgen/batches/batch_0000.json …`, skipping every task_id that already
has a non-empty `data/demos/*.md` file (and anything in `.hintgen/blocklist.txt`).
Note the printed **batches written** count.

### 2. Generate a chunk (in a Claude Code session)
Ask Claude to launch the workflow, capped at ~300 batches per launch (stays under
the 1000-agents-per-workflow limit; ~300 batches ≈ 3,600 tasks):
```
Workflow({
  scriptPath: "<abs repo>/scripts/hintgen/hint_workflow.js",
  args: { repo: "<abs repo>", count: 300 }
})
```
It runs ~16 Sonnet 5 subagents concurrently, each writing 12 hints, skipping
already-done files, and falling back to per-task singles if a cyber-flagged task
kills a batch. It's resumable — if it stops mid-way, just re-slice and relaunch.

### 3. Run the mapping gate + repair
```
uv run python scripts/hintgen/gate.py
```
This scans every hint on disk, finds swaps (a hint written into the wrong
task's file — the one silent failure mode of batch writing), fixes mutual swaps
in place, and writes any that need regeneration to `.hintgen/regen_batch.json`.

### 4. Regenerate the few swapped ones (if any)
If `.hintgen/regen_batch.json` is non-empty, ask Claude to run **one** hint agent
over it (a singleton batch can't swap):
> "Read scripts/hintgen/hint_instructions.md and follow it. Batch file is
> .hintgen/regen_batch.json. Write each hint to data/demos/<task_id>.md."

### 5. Repeat from step 1
Re-slicing now excludes everything just written. Loop until `slice_remaining.py`
reports only the permanently-blocked tasks remaining.

## Handling the permanently cyber-blocked tasks

When a chunk's leftover "missing" tasks are all cyber-blocked (they fail on every
model), stop retrying them: append their task_ids (one per line) to
`.hintgen/blocklist.txt`. `slice_remaining.py` will then exclude them, so the
final chunks aren't wasted re-attempting known-blocked tasks.

To find still-missing task_ids at any point:
```
uv run python -c "
import pandas as pd, glob, os
df=pd.read_parquet('data/tasks.parquet')
done={os.path.basename(f)[:-3] for f in glob.glob('data/demos/*.md')}
print('\n'.join(t for t in df['task_id'] if t not in done))"
```

## Cost / pacing

Full run ≈ 40–50M output tokens (~14k hints). It spans multiple Max quota
windows — each ~300-batch chunk is ~16–18M tokens and ~75 min. If a window taps
out, agents just fail to null and the next `slice → workflow` picks up from disk.

## After all hints exist — wire them into the env (not yet done)

1. `tmax_opsd_v1/build.py`: add `"demo"` to `COLUMNS`, and left-join the hints
   (built from `data/demos/<task_id>.md`) onto the rows by `task_id`.
2. `tmax_opsd_v1/taskset.py`: add `demo=row.get("demo")` to the `TaskData`
   construction (~line 193). Without this, `task.data.demo` stays `None`.
3. In the orchestrator TOML, add a second env entry:
   `algo = { type = "opsd", demo_key = "demo" }` with a `tasks = [...]` list of
   the covered (hinted, non-blocklisted) task_ids. The GRPO entry uses all tasks.

See `docs/superpowers/specs/2026-07-15-opsd-hint-generation-design.md` for the
full design.
