from pathlib import Path

import pytest

from tmax_opsd_v1.taskset import DEFAULT_DATA, TMaxTaskset, TMaxTasksetConfig

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_DATA).exists(), reason="run scripts/prepare_data.py first"
)

def test_full_corpus_loads():
    tasks = TMaxTaskset(TMaxTasksetConfig()).select()
    assert len(tasks) == 14601
    images = {t.data.image for t in tasks}
    assert len(images) == 14490
    assert all(t.data.image.startswith("hamishi740/swerl-tmax-v3:") for t in tasks[:100])

def test_known_task_zero_fields():
    cfg = TMaxTasksetConfig(tasks=["task_000000_c19dda5b"])
    (t,) = TMaxTaskset(cfg).select()
    assert t.data.verifier_kind == "fuzz_equivalence"
    assert t.data.oracle_path == "/app/oracle_parser"
    assert t.data.agent_entry_point == "/home/user/fixed_parser"
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in t.data.prompt
    # truth not leaked: these substrings appear in this task's truth field but
    # are not legitimately part of the task description
    for leaked in ("fuzz_distribution", "ground_truth_text", "Verifier Configuration"):
        assert leaked not in t.data.prompt
        assert leaked not in (t.data.system_prompt or "")
