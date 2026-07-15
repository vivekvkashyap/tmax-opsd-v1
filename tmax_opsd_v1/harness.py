"""Vanillux harness: launches program.py's agent loop, which reproduces TMax's
RL rollout semantics — swerl_vanillux_sandbox.py's env (wrapper, truncation,
tool schema) plus vllm_utils.py's rollout loop (step accounting, multi-call
turns, format-error feedback). format_error_feedback/last_step_warning default
OFF, matching the paper's published run. All prompt CONTENT lives on the
taskset (TaskData.system_prompt/prompt render prompts.yaml); this class only
launches the loop program with TMax's knobs."""

from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class VanilluxHarnessConfig(HarnessConfig):
    max_steps: int = 64
    """Tool calls (+ injected format-error feedback) before the rollout is cut —
    TMax's RL step accounting, not assistant turns (rollout.step_count in
    vllm_utils.py)."""
    command_timeout: float = 120.0
    """Per-bash-command timeout in seconds (TMax: 120)."""
    temperature: float = 1.0
    """Sampling temperature, forwarded to the client per turn (TMax RL scripts:
    --temperature 1.0)."""
    per_turn_max_tokens: int = 16384
    """Max tokens per assistant turn (TMax RL scripts: --per_turn_max_tokens 16384)."""
    format_error_feedback: bool = False
    """Inject TOOL_CALL_FORMAT_ERROR_MESSAGE and continue when a turn has no valid
    bash call, instead of ending the rollout. OFF by default — the paper's run had
    this off (tool_call_format_error_feedback=False in swerl_vanillux_sandbox.py)."""
    last_step_warning: bool = False
    """Append LAST_STEP_WARNING to the observation on the penultimate step. OFF by
    default — the paper's run had this off (last_step_warning=False)."""


class VanilluxHarness(Harness[VanilluxHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if not isinstance(prompt, str):
            raise ValueError("vanillux requires a string prompt")
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt or ''}",
            f"--prompt={prompt}",
            f"--max-steps={self.config.max_steps}",
            f"--command-timeout={self.config.command_timeout}",
            f"--temperature={self.config.temperature}",
            f"--per-turn-max-tokens={self.config.per_turn_max_tokens}",
        ]
        if self.config.format_error_feedback:
            args.append("--format-error-feedback")
        if self.config.last_step_warning:
            args.append("--last-step-warning")
        return await runtime.run_program(
            [*program, *args], {**self.config.resolved_env}
        )
