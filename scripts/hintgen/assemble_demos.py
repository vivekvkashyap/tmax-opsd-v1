"""Assemble data/demos/<task_id>.md into data/demos.parquet (task_id, demo),
the sidecar that build.py left-joins into tasks.parquet. Also writes
data/hinted_task_ids.txt (one per line) for the OPSD `tasks` selector.

Usage:  uv run python scripts/hintgen/assemble_demos.py
"""
import os, glob
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DEMOS = os.path.join(REPO, 'data', 'demos')

rows = []
for f in sorted(glob.glob(f'{DEMOS}/*.md')):
    text = open(f).read().strip()
    if not text:
        continue
    rows.append({'task_id': os.path.basename(f)[:-3], 'demo': text})

# Canonical location is INSIDE the package so it ships in the wheel / Hub env
# (build.py's load_demos reads it there). Also mirror to data/ for local builds.
table = pa.Table.from_pylist(rows)
pkg_out = os.path.join(REPO, 'tmax_opsd_v1', 'demos.parquet')
pq.write_table(table, pkg_out)
pq.write_table(table, os.path.join(REPO, 'data', 'demos.parquet'))

ids_path = os.path.join(REPO, 'data', 'hinted_task_ids.txt')
with open(ids_path, 'w') as fh:
    fh.write('\n'.join(r['task_id'] for r in rows) + '\n')

print(f'assembled {len(rows)} hints -> tmax_opsd_v1/demos.parquet (+ data/demos.parquet)')
print(f'wrote {len(rows)} task_ids -> data/hinted_task_ids.txt')
