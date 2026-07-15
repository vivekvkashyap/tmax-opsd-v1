from tmax_opsd_v1.build import build_rows

TAXONOMY = [{
    "task_id": "task_x", "domain": "sys", "skill_type": "debug",
    "primitive_skills": ["git"], "task_complexity": "hard", "command_complexity": "medium",
    "scenario": "You are a dev.", "description": "Fix the parser.",
    "truth": '# Verifier Configuration: fuzz_equivalence\noracle_path: "/app/o"\nagent_entry_point: "/home/user/f"',
}]

def test_build_rows_joins_and_parses():
    rows = build_rows(TAXONOMY, {"task_x": "img:tag"}, {"task_x": "echo 1 > r.txt"})
    assert len(rows) == 1
    r = rows[0]
    assert r["image"] == "img:tag"
    assert r["test_script"] == "echo 1 > r.txt"
    assert r["verifier_kind"] == "fuzz_equivalence"
    assert r["oracle_path"] == "/app/o"

def test_build_rows_raises_on_missing_join():
    import pytest
    with pytest.raises(ValueError, match="task_x"):
        build_rows(TAXONOMY, {}, {"task_x": "s"})
    with pytest.raises(ValueError, match="task_x"):
        build_rows(TAXONOMY, {"task_x": "img"}, {})
