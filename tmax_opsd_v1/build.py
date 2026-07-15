"""Build data/tasks.parquet by joining the two Ai2 TMax releases + task-data test
scripts. Lives in the package (not scripts/) so an installed env can bootstrap its
own dataset on first load. See TMaxTaskset.load()."""

import tarfile
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tmax_opsd_v1.parsing import parse_truth

# Canonical task data (truth, taxonomy, description, verifier) — the official
# allenai release the paper's github points to (14,601 rows); the documented,
# reproducible source.
TAXONOMY_URL = "https://huggingface.co/datasets/allenai/TMax-15K/resolve/main/data/train-00000-of-00001.parquet"
# Still required for the runnable image: TMax-15K's container_def is a Singularity
# build recipe pointing at the authors' internal cluster, unbuildable by us.
# open-instruct is the only source of a pre-built, pullable Docker image.
OPEN_INSTRUCT_URL = "https://huggingface.co/datasets/allenai/tmax-15k-open-instruct/resolve/main/data/train-00000-of-00001.parquet"
TASKDATA_URL = "https://huggingface.co/datasets/allenai/tmax-15k-open-instruct/resolve/main/task-data.tar.gz"

COLUMNS = [
    "task_id", "image", "description", "scenario", "domain", "skill_type",
    "primitive_skills", "task_complexity", "command_complexity",
    "verifier_kind", "oracle_path", "agent_entry_point", "truth", "test_script",
]

EXPECTED_ROWS = 14601


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def build_rows(taxonomy_rows: list[dict], image_by_task: dict, test_scripts: dict) -> list[dict]:
    rows = []
    for t in taxonomy_rows:
        task_id = t["task_id"]
        if task_id not in image_by_task:
            raise ValueError(f"no image for {task_id} in open-instruct release")
        if task_id not in test_scripts:
            raise ValueError(f"no tests/test.sh for {task_id} in task-data")
        info = parse_truth(t.get("truth") or "")
        rows.append({
            "task_id": task_id,
            "image": image_by_task[task_id],
            "description": t.get("description") or "",
            "scenario": t.get("scenario") or "",
            "domain": t.get("domain") or "",
            "skill_type": t.get("skill_type") or "",
            "primitive_skills": list(t.get("primitive_skills") or []),
            "task_complexity": t.get("task_complexity") or "",
            "command_complexity": t.get("command_complexity") or "",
            "verifier_kind": info.verifier_kind,
            "oracle_path": info.oracle_path,
            "agent_entry_point": info.agent_entry_point,
            "truth": t.get("truth") or "",
            "test_script": test_scripts[task_id],
        })
    return rows


def build_dataset(out_path: Path, data_dir: Path | None = None) -> Path:
    """Download the raw releases (cached under data_dir/raw), join them, and write
    tasks.parquet to out_path. data_dir defaults to out_path's parent. Returns out_path."""
    out_path = Path(out_path)
    data_dir = Path(data_dir) if data_dir is not None else out_path.parent
    raw = data_dir / "raw"

    taxonomy = pq.read_table(fetch(TAXONOMY_URL, raw / "taxonomy.parquet")).to_pylist()
    oi = pq.read_table(
        fetch(OPEN_INSTRUCT_URL, raw / "open_instruct.parquet"), columns=["env_config"]
    ).to_pylist()
    image_by_task = {r["env_config"]["task_id"]: r["env_config"]["image"] for r in oi}

    taskdata_dir = raw / "taskdata"
    if not taskdata_dir.is_dir():
        archive = fetch(TASKDATA_URL, raw / "task-data.tar.gz")
        taskdata_dir.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(taskdata_dir)
    test_scripts = {
        d.name: (d / "tests" / "test.sh").read_text()
        for d in sorted(taskdata_dir.iterdir())
        if (d / "tests" / "test.sh").is_file()
    }

    rows = build_rows(taxonomy, image_by_task, test_scripts)
    if len(rows) != EXPECTED_ROWS:
        raise AssertionError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows).select(COLUMNS), out_path)
    return out_path
