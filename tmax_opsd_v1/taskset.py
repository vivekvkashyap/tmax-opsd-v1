"""TMax terminal tasks as a verifiers-v1 taskset.

Built from Ai2's own releases (see scripts/prepare_data.py), not Prime's tmax-v1
repackaging — that one cannot run OPSD (no demo slot), uses an ipython harness,
and inherits the 120s verifier-timeout bug. Design: experiment.md Parts 3-4."""

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from pydantic import Field

from verifiers.v1.decorators import metric, reward
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskData, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace

from tmax_opsd_v1.parsing import detect_hack, parse_pass_fraction, required_fixture_paths

# Written by program.py's run_agent_loop on submit; program.py can't be imported here
# (it would drag the `openai` import into the env package), so each side defines its
# own copy of this constant — see program.py::SUBMIT_SENTINEL_PATH.
SUBMIT_SENTINEL_PATH = "/tmp/.vanillux_submitted"

PROMPTS = yaml.safe_load((Path(__file__).parent / "prompts.yaml").read_text())
# Writable cache location — the source-tree data/ dir doesn't exist on an installed
# wheel. Override with TMAX_OPSD_DATA_DIR (e.g. a pod's shared disk).
DEFAULT_DATA_DIR = Path(os.environ.get("TMAX_OPSD_DATA_DIR") or (Path.home() / ".cache" / "tmax-opsd-v1"))
DEFAULT_DATA = DEFAULT_DATA_DIR / "tasks.parquet"


class TMaxTasksetConfig(TasksetConfig):
    data_path: Path = DEFAULT_DATA
    """tasks.parquet. Built lazily by load() on first use if absent (or ahead of time
    via scripts/prepare_data.py); defaults to the DEFAULT_DATA cache path."""
    tasks: list[str] | None = None
    """Optional task_id subset (None = all 14,601)."""
    blocklist_path: Path | None = None
    """Optional file of task_ids to skip, one per line (audits/blocklist_dead_tasks.txt)."""
    require_demo: bool = False
    """When True, drop tasks whose `demo` (OPSD hint) is empty. Set this on an
    OPSD env entry so only hinted tasks are routed to it (opsd raises on a
    missing demo). Leave False for GRPO, which needs no hint."""
    verifier_timeout: float = 600.0
    """Finalize (verifier) timeout. TMax's own env enforced a 600s floor; task.toml's
    120s false-zeros fuzz verifiers (experiment.md F0.5/F1.3). Do not lower."""


class TMaxData(TaskData):
    """One TMax task row. `truth` and `test_script` are excluded from every dump:
    truth is the privileged hint material (never the student's), test_script is
    the verifier (uploaded only at scoring time)."""

    task_id: str
    domain: str = ""
    skill_type: str = ""
    primitive_skills: list[str] = []
    task_complexity: str = ""
    command_complexity: str = ""
    scenario: str = ""
    verifier_kind: str | None = None
    oracle_path: str | None = None
    agent_entry_point: str | None = None
    truth: str = Field("", exclude=True)
    test_script: str = Field("", exclude=True)
    demo: str | None = None
    """OPSD hint/demo slot (opsd.py demo_key). Filled offline, never by the env."""


def bash_commands(trace: Trace) -> list[str]:
    """Every bash command the agent issued, from the trace's final branch."""
    commands = []
    branches = trace.branches
    for message in branches[-1].messages if branches else []:
        for call in getattr(message, "tool_calls", None) or []:
            if call.name != "bash":
                continue
            args = call.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if isinstance(args, dict) and isinstance(args.get("command"), str):
                commands.append(args["command"])
    return commands


class TMaxTask(Task[TMaxData]):
    NEEDS_CONTAINER = True

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        """TMax runs the verifier ONLY when the agent submitted (vllm_utils.py: no
        submit -> _run_tests never reached -> reward stays 0, no test run). Probe the
        sentinel first; an unsubmitted rollout returns immediately at reward 0 without
        paying for the upload or up to 600s of verifier runtime. If submitted, upload
        the verifier now (the agent is done — experiment.md invariant 1) and run it
        exactly once; reward + metrics read the stash from trace.info."""
        probe = await runtime.run(["sh", "-c", f"test -e {SUBMIT_SENTINEL_PATH}"], {})
        submitted = probe.exit_code == 0
        trace.info["submitted"] = submitted
        if not submitted:
            trace.info["verifier_reward"] = 0.0
            trace.info["verifier_output"] = ""
            return
        await runtime.write("/tmp/.vf_test.sh", self.data.test_script.encode())
        result = await runtime.run(
            ["sh", "-c", "mkdir -p /logs/verifier && bash /tmp/.vf_test.sh"], {}
        )
        trace.info["verifier_output"] = (result.stdout + result.stderr)[-20000:]
        try:
            raw = (await runtime.read("/logs/verifier/reward.txt")).decode().strip()
            # Clamp like TMax's own _parse_reward (~line 519): a misbehaving test.sh
            # (or a fuzz harness reporting >1 hits) must never inflate/invert reward.
            trace.info["verifier_reward"] = max(0.0, min(1.0, float(raw or 0)))
        except (SandboxError, OSError, ValueError):
            trace.info["verifier_reward"] = 0.0

    @reward(weight=1.0)
    async def solved(self, trace: Trace) -> float:
        """Binary outcome, identical to TMax (experiment.md §3.2)."""
        return float(trace.info.get("verifier_reward", 0.0))

    @metric
    async def pass_fraction(self, trace: Trace) -> dict[str, float]:
        """Graded diagnostics from the pytest summary — logged, never summed."""
        return parse_pass_fraction(trace.info.get("verifier_output", ""))

    @metric
    async def hacked(self, trace: Trace) -> float:
        """E0's instrument: did the agent copy the oracle into place? (F0.2)"""
        return detect_hack(
            bash_commands(trace), self.data.oracle_path, self.data.agent_entry_point
        )

    @metric
    async def submitted(self, trace: Trace) -> float:
        """E-ladder diagnostic: how often do rollouts end without ever submitting?"""
        return float(trace.info.get("submitted", False))

    async def validate(self, runtime: Runtime) -> bool:
        """F1.5 pre-flight: every fixture the verifier asserts must exist in the
        image; ~15% of sampled tasks shipped broken (dead) and would train as
        permanent zeros. Run via the `vf validate` CLI to build the blocklist."""
        for path in required_fixture_paths(self.data.test_script):
            result = await runtime.run(["sh", "-c", f'test -e "{path}"'], {})
            if result.exit_code != 0:
                return False
        return True


class TMaxTaskset(Taskset[TMaxTask, TMaxTasksetConfig]):
    def load(self):
        blocked: set[str] = set()
        if self.config.blocklist_path is not None:
            blocked = {
                line.strip()
                for line in Path(self.config.blocklist_path).read_text().splitlines()
                if line.strip()
            }
        wanted = set(self.config.tasks) if self.config.tasks is not None else None
        if not Path(self.config.data_path).exists():
            from tmax_opsd_v1.build import build_dataset
            build_dataset(self.config.data_path)
        table = pq.read_table(self.config.data_path)
        instance = PROMPTS["instance_template"]
        system = PROMPTS["system_template"]
        idx = 0
        for row in table.to_pylist():
            if row["task_id"] in blocked:
                continue
            if wanted is not None and row["task_id"] not in wanted:
                continue
            if self.config.require_demo and not (row.get("demo") if isinstance(row, dict) else None):
                continue
            data = TMaxData(
                idx=idx,
                name=row["task_id"],
                prompt=instance.replace("{{task}}", row["description"]),
                system_prompt=system,
                image=row["image"],
                timeout=TaskTimeout(
                    finalize=self.config.verifier_timeout,
                    scoring=60.0,
                ),
                resources=TaskResources(cpu=1, memory=2),
                task_id=row["task_id"],
                domain=row["domain"],
                skill_type=row["skill_type"],
                primitive_skills=list(row["primitive_skills"] or []),
                task_complexity=row["task_complexity"],
                command_complexity=row["command_complexity"],
                scenario=row["scenario"],
                verifier_kind=row["verifier_kind"],
                oracle_path=row["oracle_path"],
                agent_entry_point=row["agent_entry_point"],
                truth=row["truth"],
                test_script=row["test_script"],
                demo=row.get("demo"),
            )
            yield TMaxTask(data, self.config.task)
            idx += 1
