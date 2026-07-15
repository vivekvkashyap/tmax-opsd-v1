import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROWS = [
    {
        "task_id": f"task_{i:06d}_test", "image": f"example/img:{i}",
        "description": f"Fix bug {i}.", "scenario": "You are a dev.",
        "domain": "sys", "skill_type": "debug", "primitive_skills": ["git"],
        "task_complexity": "hard", "command_complexity": "medium",
        "verifier_kind": "fuzz_equivalence" if i == 0 else None,
        "oracle_path": "/app/oracle" if i == 0 else None,
        "agent_entry_point": "/home/user/fixed" if i == 0 else None,
        "truth": "SECRET-TRUTH", "test_script": 'echo "1" > /logs/verifier/reward.txt',
        "demo": None,
    }
    for i in range(3)
]

@pytest.fixture
def tasks_parquet(tmp_path):
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), path)
    return path
