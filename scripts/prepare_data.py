"""CLI wrapper: build data/tasks.parquet via tmax_opsd_v1.build.build_dataset.

Usage: uv run python scripts/prepare_data.py [--data-dir data]
Downloads raw inputs only when missing (pre-seeded data/raw/ skips the network)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tmax_opsd_v1.build import build_dataset
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent.parent / "data")
    args = parser.parse_args()
    out = build_dataset(args.data_dir / "tasks.parquet", data_dir=args.data_dir)
    rows = pq.read_table(out).to_pylist()
    kinds = {k: sum(1 for r in rows if r["verifier_kind"] == k) for k in {r["verifier_kind"] for r in rows}}
    print(f"wrote {out}: {len(rows)} rows, {len(set(r['image'] for r in rows))} unique images")
    print(f"verifier kinds: {kinds}")
    print(f"rows with parsed oracle_path: {sum(1 for r in rows if r['oracle_path'])}")


if __name__ == "__main__":
    main()
