"""Mapping gate + repair over ALL hints in <repo>/data/demos/.

Detects:
  - MISSING  : task with no non-empty demo file (not counted here; use slice to (re)generate)
  - MISMATCH : hint shares zero real filenames/paths with its own task's
               description+truth (a likely swap)

For each MISMATCH, decides SWAP (hint's paths match another task markedly
better) vs false-positive (hint just uses placeholder names — left alone).
Repairs:
  - mutual swaps (A<->B) : exchange the two files (no model call)
  - other swaps          : delete the corrupted file and add it to
                           <repo>/.hintgen/regen_batch.json for a singleton
                           regeneration agent to rewrite.

Usage:  uv run python scripts/hintgen/gate.py
Then, if regen_batch.json is non-empty, run one hint-generation agent over it
(see RESUME.md).
"""
import os, glob, re, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DEMOS = f'{REPO}/data/demos'
FNAME = re.compile(r'[A-Za-z0-9_-]+\.(?:sh|py|c|cpp|h|go|rs|sql|csv|json|txt|log|png|jpg|mp4|wav|joblib|md|ini|jwt|elf|pcap|db|yaml|yml|toml|conf|cfg|xml|html|js|ts)\b')
ABS = re.compile(r'/(?:home|app|opt|srv|usr|tmp|var|etc)/[A-Za-z0-9_./-]+')


def strong(s):
    return set(FNAME.findall(s)) | set(ABS.findall(s))


df = pd.read_parquet(f'{REPO}/data/tasks.parquet').set_index('task_id')
ctx = {tid: strong(str(r.description) + ' ' + str(r.truth)) for tid, r in df.iterrows()}

demo_ids = [os.path.basename(f)[:-3] for f in glob.glob(f'{DEMOS}/*.md')]
print(f'hints on disk: {len(demo_ids)} / {len(df)}')

# find mismatches (0 filename overlap with own task)
mismatch = []
for tid in demo_ids:
    if tid not in ctx:
        continue
    ht = strong(open(f'{DEMOS}/{tid}.md').read())
    if len(ht) < 2:
        continue
    if len(ht & ctx[tid]) == 0:
        mismatch.append(tid)


def best_other(ht, self_tid):
    b, bn = None, 0
    for otid, ot in ctx.items():
        if otid == self_tid:
            continue
        n = len(ht & ot)
        if n > bn:
            bn, b = n, otid
    return b, bn


swaps = []
for tid in mismatch:
    ht = strong(open(f'{DEMOS}/{tid}.md').read())
    own = len(ht & ctx[tid])
    b, bn = best_other(ht, tid)
    if bn >= 2 and bn > own:
        swaps.append(tid)

print(f'mismatches flagged: {len(mismatch)} | genuine swaps: {len(swaps)} | false positives: {len(mismatch) - len(swaps)}')

# repair: mutual pairs -> swap files; others -> regen
owner = {}
for tid in swaps:
    ht = strong(open(f'{DEMOS}/{tid}.md').read())
    owner[tid] = best_other(ht, tid)[0]

handled, regen = set(), []
for tid in swaps:
    if tid in handled:
        continue
    partner = owner[tid]
    if partner in swaps and owner.get(partner) == tid:
        a, b = f'{DEMOS}/{tid}.md', f'{DEMOS}/{partner}.md'
        ca, cb = open(a).read(), open(b).read()
        open(a, 'w').write(cb)
        open(b, 'w').write(ca)
        handled.update({tid, partner})
        print(f'SWAPPED pair {tid} <-> {partner}')
    else:
        os.remove(f'{DEMOS}/{tid}.md')
        regen.append(tid)
        handled.add(tid)
        print(f'REGEN   {tid} (content belonged to {partner})')

os.makedirs(f'{REPO}/.hintgen', exist_ok=True)
recs = [{'task_id': t, 'description': str(df.loc[t, 'description']), 'truth': str(df.loc[t, 'truth'])} for t in regen]
json.dump(recs, open(f'{REPO}/.hintgen/regen_batch.json', 'w'))
print(f'\nmutual swaps fixed: {len(handled) - len(regen)} | need regen: {len(regen)} -> .hintgen/regen_batch.json')
