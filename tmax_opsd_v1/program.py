# /// script
# requires-python = ">=3.10"
# dependencies = ["openai"]
# ///
"""Vanillux agent loop — reproduces TMax's RL rollout semantics, NOT the offline
solver: one persistent-shell `bash` tool, TMax's disk-backed wrapper and cwd/env
files, mini-swe-agent-style head/tail truncation, and `vllm_utils.py`'s rollout
step accounting (steps = executed tool calls + injected format-error feedback,
not assistant turns). Format-error feedback and the last-step warning are
flag-gated and OFF by default, matching the paper's published run. Ground
truth: open_instruct/environments/swerl_vanillux_sandbox.py (env) and
vllm_utils.py ~1180-1330 (rollout loop). Runs inside the task container as a
self-contained uv script; state files match TMax's exactly (fidelity,
experiment.md §3.1)."""

import argparse
import asyncio
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from openai import AsyncOpenAI

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Written by this program on submit; read by tmax_opsd_v1/taskset.py::finalize (Task 2).
# program.py can't be imported by the package (it would drag the `openai` import into
# the env package), so each side defines its own copy of this constant.
SUBMIT_SENTINEL_PATH = "/tmp/.vanillux_submitted"

WRAPPER_PATH = "/tmp/.swerl_vanillux_bash_wrapper.sh"
_CWD_PATH = "/tmp/.swerl_vanillux_cwd"
_ENV_PATH = "/tmp/.swerl_vanillux_env"

# TMax's disk-backed persistent shell: env + cwd survive across bash calls.
# Byte-exact with swerl_vanillux_sandbox.py's _BASH_WRAPPER (~line 53); the paths
# there are shlex.quote()d but contain nothing that needs quoting, so this f-string
# reproduces the same bytes.
BASH_WRAPPER = f"""#!/bin/bash
set -a
source {_ENV_PATH} 2>/dev/null || true
set +a
_cwd="$(cat {_CWD_PATH} 2>/dev/null || echo /app)"
cd "$_cwd" 2>/dev/null || cd /workspace || exit 1
eval "$1"
_exit_code=$?
export -p > {_ENV_PATH}
pwd > {_CWD_PATH}
exit $_exit_code
"""

# Replicates _prepare_vanillux_runtime (~line 398): seeds cwd=/app, truncates any
# stale env file, and guarantees /app exists (symlinked to /workspace on images
# that lack it). Run once before the loop starts, then the wrapper file is written.
RUNTIME_PREP_CMD = (
    "mkdir -p /workspace /root && "
    "cd /workspace && "
    '[ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; } && '
    f"printf '%s\\n' /app > {_CWD_PATH} && "
    f": > {_ENV_PATH}"
)

MAX_OBS_CHARS = 10_000
HEAD_CHARS = 5_000
TAIL_CHARS = 5_000

# Must equal prompts.yaml's observation.too_long_hint, INCLUDING the trailing
# newline the yaml `|` block scalar produces (see test_too_long_hint_matches_
# prompts_yaml). TMax loads this from vanillux_prompts.yaml at runtime; program.py
# can't read package files, so we embed the same bytes here.
TOO_LONG_HINT = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a\n"
    "smaller number of lines selectively. If you're using grep or find and it\n"
    "produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you\n"
    "can redirect output to a file and then search in that file.\n"
)

# Mirrors prompts.yaml's format_error_template (mini-swe-agent's format_error_
# template), including its trailing newline. format_error_message() substitutes
# {{error}} the same way TMax's format_error_message() does (plain .replace(),
# not str.format(), so error text can contain literal braces safely).
FORMAT_ERROR_TEMPLATE = (
    "Format error: {{error}}\n\n"
    "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
    f"end the task, please issue the command `echo {SUBMIT_MARKER}`\n"
    "via the `bash` tool, with no other content in the command.\n"
)


def format_error_message(error: str) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)


# Byte-exact with swerl_vanillux_sandbox.py's TOOL_CALL_FORMAT_ERROR_MESSAGE
# (~line 91) — a plain Python constant there, not yaml-sourced, hence no
# trailing newline (unlike FORMAT_ERROR_TEMPLATE above).
TOOL_CALL_FORMAT_ERROR_MESSAGE = (
    "Format error: Your last response did not include a valid `bash` tool call.\n\n"
    "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
    f"end the task, please issue the command `echo {SUBMIT_MARKER}`\n"
    "via the `bash` tool, with no other content in the command."
)

# Byte-exact with swerl_sandbox.py's LAST_STEP_WARNING (lines 39-42).
LAST_STEP_WARNING = (
    "Warning: you only have one more tool call remaining. "
    f"You must end your next tool call with `echo {SUBMIT_MARKER}`"
)

# Byte-exact with swerl_vanillux_sandbox.py's _BASH_TOOL (~line 66).
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. "
            "Working directory and environment variables are preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute."}
            },
            "required": ["command"],
        },
    },
}


def truncate_observation(output: str) -> str:
    """mini-swe-agent's head/tail truncation, byte-exact with truncate_observation
    in swerl_vanillux_sandbox.py (~line 108)."""
    if len(output) <= MAX_OBS_CHARS:
        return output
    elided = len(output) - HEAD_CHARS - TAIL_CHARS
    return (
        f"{TOO_LONG_HINT}\n\n"
        f"---- HEAD ({HEAD_CHARS} chars) ----\n"
        f"{output[:HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({TAIL_CHARS} chars) ----\n"
        f"{output[-TAIL_CHARS:]}"
    )


def build_observation(output: str, exit_code: int) -> str:
    observation = truncate_observation(output) if output else "(no output)"
    return f"{observation}\n\n(exit_code={exit_code})"


def apply_last_step_warning(observation: str, *, enabled: bool, steps: int, max_steps: int) -> str:
    """Only applied to bash-call-driven observations (real execution or the
    missing-command error) — matching TMax's _with_last_step_warning, which wraps
    every env.step() result. Format-error feedback (1f) is injected upstream of
    env.step() in vllm_utils.py's rollout loop and never receives this suffix, even
    though it also consumes a step; see run_agent_loop."""
    if enabled and steps == max_steps - 1:
        return f"{observation}\n\n{LAST_STEP_WARNING}"
    return observation


def merge_output(stdout: str, stderr: str) -> str:
    """Byte-exact with swerl_vanillux_sandbox.py's _execute_bash stderr merge
    (~line 459)."""
    output = stdout
    if stderr:
        output += f"\n{stderr}" if output else stderr
    return output


# Per-stream output cap, byte-exact with DockerBackend/ApptainerBackend's
# `_MAX_OUTPUT_BYTES` (backends.py ~169, 581): each of stdout/stderr is sliced to
# 1,000,000 RAW BYTES before decoding and before the stderr-merge, so a >1MB
# stream truncates exactly the way TMax's does and can't hide/reveal a
# SUBMIT_MARKER past the cap that TMax would miss (or vice versa). run_bash
# captures raw bytes (no text=True) precisely so this slice-then-decode order
# matches backends.py ~322-323 / ~723-724.
MAX_OUTPUT_BYTES = 1_000_000


def build_bash_result(
    stdout: bytes | None, stderr: bytes | None, exit_code: int, timeout: float
) -> tuple[str, int]:
    """Post-processing shared by run_bash's normal-return and timeout paths.

    Byte-exact with backends.py's DockerBackend/ApptainerBackend.run_command
    (~lines 320-326, 722-727): normalize None to b"" (~320-321), slice each raw
    stream to MAX_OUTPUT_BYTES and decode with errors="replace" (~322-323 — never
    raise on invalid UTF-8; the agent sees mojibake plus the real exit code),
    THEN — only for exit_code 124 — prepend the exact timeout message to (the
    now-capped) stderr (~324-325), THEN merge. The order matters: TMax prepends
    the message AFTER truncating, so the message doesn't get truncated away with
    the rest of a huge stderr.
    """
    stdout_text = (stdout or b"")[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    stderr_text = (stderr or b"")[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if exit_code == 124:
        # backends.py ~324-326 / ~725-726: f"Command timed out after {effective_timeout}s.\n" + stderr
        stderr_text = f"Command timed out after {int(timeout)}s.\n" + stderr_text
    return merge_output(stdout_text, stderr_text), exit_code


def run_bash(command: str, timeout: float) -> tuple[str, int]:
    try:
        # Bytes capture (no text=True), matching backends.py: invalid UTF-8 in the
        # output must NOT raise (TMax decodes errors="replace"); with text=True a
        # binary cat would raise UnicodeDecodeError inside communicate() and turn
        # into a bogus `error: ...`/exit-1 observation.
        result = subprocess.run(
            ["bash", WRAPPER_PATH, command],
            capture_output=True, timeout=timeout,
        )
        return build_bash_result(result.stdout, result.stderr, result.returncode, timeout)
    except subprocess.TimeoutExpired as e:
        # backends.py ~275-278/711-714: TMax wraps the command in
        # `timeout --signal=TERM --kill-after=10 <N> bash -c <command>`, so a
        # timeout is observed as the wrapped process exiting 124 with whatever
        # partial stdout/stderr the command produced before being killed — NOT as
        # an exception that discards output. Recover the partial bytes (None when
        # nothing was captured) so they flow through the same cap/decode/merge
        # path as a normal return.
        return build_bash_result(e.stdout, e.stderr, 124, timeout)
    except Exception as e:  # noqa: BLE001 — surface anything as a tool error
        return f"error: {e}", 1


def parse_command_arg(arguments: str | None) -> str | None:
    """Parses a bash tool call's JSON arguments; returns the command string or
    None for missing/empty/non-string command or unparseable JSON (all handled
    identically per 1g)."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    command = parsed.get("command")
    return command if isinstance(command, str) and command else None


def prepare_runtime(wrapper_path: str = WRAPPER_PATH) -> None:
    """Replicates _prepare_vanillux_runtime (~lines 398-408): run the mkdir/symlink/
    seed prep command, write the wrapper file, then chmod +x it. Same side effects,
    same order."""
    subprocess.run(["bash", "-c", RUNTIME_PREP_CMD])
    Path(wrapper_path).write_text(BASH_WRAPPER)
    subprocess.run(["chmod", "+x", wrapper_path])


async def run_agent_loop(
    client,
    messages: list[dict],
    *,
    model: str,
    max_steps: int,
    command_timeout: float,
    temperature: float = 1.0,
    per_turn_max_tokens: int = 16384,
    format_error_feedback: bool = False,
    last_step_warning: bool = False,
    run_bash_fn: Callable[[str, float], tuple[str, int]] = run_bash,
    submit_sentinel_path: str = SUBMIT_SENTINEL_PATH,
) -> list[dict]:
    """The RL rollout loop (vllm_utils.py ~1186-1339), adapted to the OpenAI chat
    completions API. `messages` is mutated in place and also returned.

    Step accounting (1d): `steps` increments once per executed bash call and once
    per format-error-feedback injection — never for the accepted-divergence
    "unknown tool" responses below. The outer loop runs while steps < max_steps;
    the inner per-call loop stops executing further bash calls once the budget is
    exhausted mid-turn.
    """
    steps = 0
    while steps < max_steps:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[BASH_TOOL],
            temperature=temperature,
            max_tokens=per_turn_max_tokens,
        )
        message = completion.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        tool_calls = list(message.tool_calls or [])
        bash_calls = [tc for tc in tool_calls if tc.function.name == "bash"]
        wrong_calls = [tc for tc in tool_calls if tc.function.name != "bash"]

        # 1f: format-error feedback, mirroring vllm_utils.py:1233-1242. TMax
        # computes has_wrong_tool_call over the UNFILTERED parsed calls, so when
        # the flag is on, a turn mixing valid bash calls with non-bash calls STILL
        # gets the shared feedback message (and its one-step charge) in addition
        # to normal dispatch of the valid bash calls.
        feedback_role: str | None = None
        if format_error_feedback:
            if not tool_calls:
                # No tool call at all reads like a fresh user instruction.
                feedback_role = "user"
            elif wrong_calls or not bash_calls:
                feedback_role = "tool"

        if not bash_calls and feedback_role is None:
            # Nothing to execute and no feedback to inject (vllm_utils.py:1249):
            # matches the paper's run — end the rollout immediately, unsubmitted.
            break

        # Feedback observations are appended BEFORE tool-call observations
        # (vllm_utils.py:1253's feedback loop runs before the :1264 dispatch loop)
        # and consume one step total, no matter how many ids they answer.
        if feedback_role == "user":
            messages.append({"role": "user", "content": TOOL_CALL_FORMAT_ERROR_MESSAGE})
            steps += 1
        elif feedback_role == "tool":
            # TMax's text-parsing path emits ONE shared tool-role message; the
            # OpenAI API requires a response per tool_call_id, so the same text
            # answers each offending id (accepted divergence). In mixed turns the
            # offending ids' responses ARE the feedback — the model-visible trace
            # stays identical to TMax's (feedback, then bash observations) with no
            # extra filler message.
            for tc in wrong_calls:
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": TOOL_CALL_FORMAT_ERROR_MESSAGE,
                })
            steps += 1
        elif wrong_calls:
            # Accepted divergence (flag off): the OpenAI API still requires a tool
            # message per id, a constraint TMax's text-parsing path never faced —
            # TMax simply filters non-bash calls out with no response. Rendered
            # with the env's format_error_template; does not consume a step (it's
            # neither an executed bash call nor 1f's format-error feedback).
            for tc in wrong_calls:
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": format_error_message(
                        f"Unknown tool '{tc.function.name}'. The only available tool is `bash`."
                    ),
                })

        if not bash_calls:
            continue

        submitted = False
        for tc in bash_calls:
            if steps >= max_steps:
                break  # budget exhausted mid-turn; stop executing further calls

            steps += 1
            command = parse_command_arg(tc.function.arguments)
            if command is None:
                # 1g: empty/missing `command` (or unparseable JSON) is env-level
                # and always on, regardless of format_error_feedback.
                content = format_error_message("'command' parameter is required.")
            else:
                output, exit_code = await asyncio.to_thread(run_bash_fn, command, command_timeout)
                submitted = SUBMIT_MARKER in output
                if submitted:
                    Path(submit_sentinel_path).touch()
                content = build_observation(output, exit_code)
            content = apply_last_step_warning(
                content, enabled=last_step_warning, steps=steps, max_steps=max_steps
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            if submitted:
                break

        if submitted:
            break

    return messages


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--per-turn-max-tokens", type=int, default=16384)
    parser.add_argument("--format-error-feedback", action="store_true")
    parser.add_argument("--last-step-warning", action="store_true")
    args = parser.parse_args()

    prepare_runtime()

    client = AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key, timeout=1800.0, max_retries=0
    )
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    await run_agent_loop(
        client,
        messages,
        model=args.model,
        max_steps=args.max_steps,
        command_timeout=args.command_timeout,
        temperature=args.temperature,
        per_turn_max_tokens=args.per_turn_max_tokens,
        format_error_feedback=args.format_error_feedback,
        last_step_warning=args.last_step_warning,
    )


if __name__ == "__main__":
    asyncio.run(main())
