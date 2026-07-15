# tests/test_taskset.py
import pytest
from pathlib import Path
from types import SimpleNamespace
from tmax_opsd_v1.taskset import TMaxTaskset, TMaxTasksetConfig

def test_load_builds_typed_tasks(tasks_parquet):
    ts = TMaxTaskset(TMaxTasksetConfig(data_path=tasks_parquet))
    tasks = ts.select()
    assert len(tasks) == 3
    t = tasks[0]
    assert t.data.task_id == "task_000000_test"
    assert t.data.image == "example/img:0"
    assert t.data.oracle_path == "/app/oracle"
    assert t.data.timeout.finalize == 600.0
    assert t.data.resources.cpu == 1 and t.data.resources.memory == 2

def test_prompt_rendered_and_truth_never_leaks(tasks_parquet):
    ts = TMaxTaskset(TMaxTasksetConfig(data_path=tasks_parquet))
    t = ts.select()[0]
    assert "Fix bug 0." in t.data.prompt          # instance template got the description
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in t.data.prompt
    assert t.data.system_prompt and "THOUGHT" in t.data.system_prompt
    for text in (t.data.prompt, t.data.system_prompt):
        assert "SECRET-TRUTH" not in text
    # excluded from every serialization (traces, /task channel)
    assert "SECRET-TRUTH" not in t.data.model_dump_json()
    assert "reward.txt" not in t.data.model_dump_json()

def test_tasks_filter_and_blocklist(tasks_parquet, tmp_path):
    block = tmp_path / "block.txt"
    block.write_text("task_000001_test\n")
    cfg = TMaxTasksetConfig(data_path=tasks_parquet, blocklist_path=block)
    ids = [t.data.task_id for t in TMaxTaskset(cfg).select()]
    assert ids == ["task_000000_test", "task_000002_test"]
    cfg = TMaxTasksetConfig(data_path=tasks_parquet, tasks=["task_000002_test"])
    ids = [t.data.task_id for t in TMaxTaskset(cfg).select()]
    assert ids == ["task_000002_test"]


class FakeRuntime:
    """Collects writes; scripted run() results; read() returns the reward file."""
    def __init__(self, reward_file=b"1\n", run_output="= 3 passed in 1s =", exits=None):
        self.reward_file, self.run_output = reward_file, run_output
        self.exits = list(exits or [])
        self.writes, self.runs = {}, []
    async def write(self, path, data):
        self.writes[path] = data
    async def run(self, argv, env):
        self.runs.append(argv)
        code = self.exits.pop(0) if self.exits else 0
        return SimpleNamespace(exit_code=code, stdout=self.run_output, stderr="")
    async def read(self, path):
        return self.reward_file


class FakeTrace:
    def __init__(self):
        self.info, self.branches = {}, []


@pytest.fixture
def task(tasks_parquet):
    return TMaxTaskset(TMaxTasksetConfig(data_path=tasks_parquet)).select()[0]


def _verifier_runs(rt):
    """Runs whose argv actually invokes the uploaded test.sh (excludes the sentinel probe)."""
    return [argv for argv in rt.runs if any("vf_test.sh" in str(a) for a in argv)]


async def test_finalize_runs_verifier_once_and_stashes(task):
    # exits: [0] -> sentinel probe succeeds (submitted); verifier run defaults to exit 0 too.
    rt, trace = FakeRuntime(reward_file=b"1\n", exits=[0]), FakeTrace()
    await task.finalize(trace, rt)
    assert trace.info["submitted"] is True
    assert trace.info["verifier_reward"] == 1.0
    assert "3 passed" in trace.info["verifier_output"]
    assert any(b"reward.txt" in v or b"echo" in v for v in rt.writes.values())  # test.sh uploaded
    assert len(_verifier_runs(rt)) == 1  # verifier ran exactly once
    assert len(rt.runs) == 2  # sentinel probe + verifier run, nothing else

async def test_finalize_missing_reward_file_scores_zero(task):
    class NoRead(FakeRuntime):
        async def read(self, path):
            raise OSError("gone")
    trace = FakeTrace()
    await task.finalize(trace, NoRead(exits=[0]))
    assert trace.info["submitted"] is True
    assert trace.info["verifier_reward"] == 0.0

async def test_finalize_not_submitted_skips_verifier_entirely(task):
    """TMax invariant: no submit -> reward 0, verifier never uploaded or run (F0.5/F1.3 budget)."""
    rt, trace = FakeRuntime(exits=[1]), FakeTrace()  # sentinel probe fails: not submitted
    await task.finalize(trace, rt)
    assert trace.info["submitted"] is False
    assert trace.info["verifier_reward"] == 0.0
    assert trace.info["verifier_output"] == ""
    assert rt.writes == {}          # test.sh never uploaded
    assert len(_verifier_runs(rt)) == 0
    assert len(rt.runs) == 1         # only the sentinel probe ran
    assert await task.submitted(trace) == 0.0

async def test_finalize_clamps_out_of_range_reward(task):
    trace_high = FakeTrace()
    await task.finalize(trace_high, FakeRuntime(reward_file=b"3\n", exits=[0]))
    assert trace_high.info["verifier_reward"] == 1.0

    trace_low = FakeTrace()
    await task.finalize(trace_low, FakeRuntime(reward_file=b"-1\n", exits=[0]))
    assert trace_low.info["verifier_reward"] == 0.0

async def test_submitted_metric_reads_stash(task):
    trace = FakeTrace()
    trace.info = {"submitted": True}
    assert await task.submitted(trace) == 1.0
    trace.info = {"submitted": False}
    assert await task.submitted(trace) == 0.0
    trace.info = {}
    assert await task.submitted(trace) == 0.0

async def test_solved_and_pass_fraction_read_stash(task):
    trace = FakeTrace()
    trace.info = {"verifier_reward": 1.0, "verifier_output": "= 1 failed, 3 passed ="}
    assert await task.solved(trace) == 1.0
    assert await task.pass_fraction(trace) == {"pass_fraction": 0.75}

async def test_hacked_metric_on_oracle_copy(task):
    trace = FakeTrace()
    call = SimpleNamespace(name="bash", arguments='{"command": "cp /app/oracle /home/user/fixed"}')
    msg = SimpleNamespace(tool_calls=[call])
    trace.branches = [SimpleNamespace(messages=[msg])]
    assert await task.hacked(trace) == 1.0

async def test_validate_fails_on_missing_fixture(tasks_parquet):
    cfg = TMaxTasksetConfig(data_path=tasks_parquet)
    task = TMaxTaskset(cfg).select()[0]
    # task 0's test_script has no fixture paths -> vacuously valid
    assert await task.validate(FakeRuntime()) is True
    # inject a script that requires a fixture; runtime says it's missing (exit 1)
    object.__setattr__(task.data, "test_script", 'test -x "/app/oracle_parser"')
    assert await task.validate(FakeRuntime(exits=[1])) is False


def test_load_lazy_builds_when_parquet_missing(tmp_path, monkeypatch):
    """If data_path doesn't exist, load() builds it via build_dataset (no network in test)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tmax_opsd_v1 import build as build_mod
    from tmax_opsd_v1.taskset import TMaxTaskset, TMaxTasksetConfig

    target = tmp_path / "nested" / "tasks.parquet"
    called = {}

    def fake_build(out_path, data_dir=None):
        called["path"] = Path(out_path)
        row = {
            "task_id": "task_lazy", "image": "img:lazy", "description": "d", "scenario": "s",
            "domain": "", "skill_type": "", "primitive_skills": [], "task_complexity": "",
            "command_complexity": "", "verifier_kind": None, "oracle_path": None,
            "agent_entry_point": None, "truth": "", "test_script": "echo 1",
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row]), out_path)
        return Path(out_path)

    monkeypatch.setattr(build_mod, "build_dataset", fake_build)
    tasks = TMaxTaskset(TMaxTasksetConfig(data_path=target)).select()
    assert called["path"] == target        # lazy build was invoked with the missing path
    assert len(tasks) == 1 and tasks[0].data.task_id == "task_lazy"


def test_load_does_not_build_when_parquet_exists(tasks_parquet, monkeypatch):
    """When the parquet already exists, load() must NOT call build_dataset."""
    from tmax_opsd_v1 import build as build_mod
    from tmax_opsd_v1.taskset import TMaxTaskset, TMaxTasksetConfig

    def boom(*a, **k):
        raise AssertionError("build_dataset must not be called when parquet exists")

    monkeypatch.setattr(build_mod, "build_dataset", boom)
    tasks = TMaxTaskset(TMaxTasksetConfig(data_path=tasks_parquet)).select()
    assert len(tasks) == 3
