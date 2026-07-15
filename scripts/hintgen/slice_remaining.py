"""Slice not-yet-hinted tasks into batch files under <repo>/.hintgen/batches/.
Resumable: skips any task_id that already has a non-empty demo file, and any
task_id listed in <repo>/.hintgen/blocklist.txt (one per line — use for the
permanently cyber-blocked tasks so they aren't retried every chunk).

Usage:  uv run python scripts/hintgen/slice_remaining.py [BATCH_SIZE]
"""
import sys, os, glob, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OUT = os.path.join(REPO, '.hintgen', 'batches')

os.makedirs(OUT, exist_ok=True)
for f in glob.glob(f'{OUT}/batch_*.json'):
    os.remove(f)

df = pd.read_parquet(f'{REPO}/data/tasks.parquet')

done = set()
for f in glob.glob(f'{REPO}/data/demos/*.md'):
    if os.path.getsize(f) > 0:
        done.add(os.path.basename(f)[:-3])

blockfile = os.path.join(REPO, '.hintgen', 'blocklist.txt')
if os.path.exists(blockfile):
    done |= {ln.strip() for ln in open(blockfile) if ln.strip()}

remaining = df[~df['task_id'].isin(done)][['task_id', 'description', 'truth']].reset_index(drop=True)
n = 0
for i in range(0, len(remaining), BATCH):
    chunk = remaining.iloc[i:i + BATCH]
    recs = [{'task_id': r.task_id, 'description': str(r.description), 'truth': str(r.truth)}
            for r in chunk.itertuples()]
    with open(f'{OUT}/batch_{n:04d}.json', 'w') as fh:
        json.dump(recs, fh)
    n += 1

print('already done (or blocklisted):', len(done))
print('remaining tasks:', len(remaining))
print('batches written:', n, '(size', BATCH, ')')
print('batch dir:', OUT)
