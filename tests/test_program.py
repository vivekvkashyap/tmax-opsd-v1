"""TDD for the vanillux agent loop (program.py) — TMax RL rollout semantics.

Ground truth: tmax/training/open-instruct/open_instruct/environments/
swerl_vanillux_sandbox.py (+swerl_sandbox.py for LAST_STEP_WARNING/SUBMIT_MARKER)
and vllm_utils.py's rollout loop (~1180-1330) for step accounting.
"""

import subprocess
from pathlib import Path

import yaml
import pytest

from tmax_opsd_v1.program import (
    BASH_TOOL,
    BASH_WRAPPER,
    FORMAT_ERROR_TEMPLATE,
    LAST_STEP_WARNING,
    MAX_OUTPUT_BYTES,
    RUNTIME_PREP_CMD,
    SUBMIT_MARKER,
    SUBMIT_SENTINEL_PATH,
    TOO_LONG_HINT,
    TOOL_CALL_FORMAT_ERROR_MESSAGE,
    WRAPPER_PATH,
    apply_last_step_warning,
    build_bash_result,
    build_observation,
    format_error_message,
    merge_output,
    parse_command_arg,
    prepare_runtime,
    run_agent_loop,
    run_bash,
    truncate_observation,
)

PROMPTS = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "tmax_opsd_v1" / "prompts.yaml").read_text()
)


# ---------------------------------------------------------------------------
# Fakes for driving run_agent_loop without network / subprocess.
# ---------------------------------------------------------------------------


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id, name, arguments="{}"):
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeCompletion:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeCompletions:
    def __init__(self, completions):
        self._queue = list(completions)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._queue.pop(0)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, messages):
        completions = FakeCompletions([FakeCompletion(m) for m in messages])
        self.chat = FakeChat(completions)

    @property
    def calls(self):
        return self.chat.completions.calls


def fake_run_bash(output="ok", exit_code=0):
    def _run(command, timeout):
        return output, exit_code

    return _run


# ---------------------------------------------------------------------------
# 1a. Wrapper byte-exact
# ---------------------------------------------------------------------------


def test_wrapper_path_is_tmax_faithful():
    assert WRAPPER_PATH == "/tmp/.swerl_vanillux_bash_wrapper.sh"


def test_bash_wrapper_byte_exact():
    expected = (
        "#!/bin/bash\n"
        "set -a\n"
        "source /tmp/.swerl_vanillux_env 2>/dev/null || true\n"
        "set +a\n"
        '_cwd="$(cat /tmp/.swerl_vanillux_cwd 2>/dev/null || echo /app)"\n'
        'cd "$_cwd" 2>/dev/null || cd /workspace || exit 1\n'
        'eval "$1"\n'
        "_exit_code=$?\n"
        "export -p > /tmp/.swerl_vanillux_env\n"
        "pwd > /tmp/.swerl_vanillux_cwd\n"
        "exit $_exit_code\n"
    )
    assert BASH_WRAPPER == expected


def test_bash_wrapper_no_longer_falls_back_to_root():
    # The divergence being removed: `cd /workspace 2>/dev/null || cd /`.
    assert "|| cd /\n" not in BASH_WRAPPER
    assert "|| exit 1" in BASH_WRAPPER


# ---------------------------------------------------------------------------
# 1b. Runtime preparation
# ---------------------------------------------------------------------------


def test_runtime_prep_cmd_byte_exact():
    expected = (
        "mkdir -p /workspace /root && "
        "cd /workspace && "
        '[ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; } && '
        "printf '%s\\n' /app > /tmp/.swerl_vanillux_cwd && "
        ": > /tmp/.swerl_vanillux_env"
    )
    assert RUNTIME_PREP_CMD == expected


def test_prepare_runtime_prep_then_write_then_chmod(tmp_path, monkeypatch):
    # Records each subprocess.run call along with whether the wrapper file existed
    # at that moment — proving prep runs BEFORE the write and chmod runs AFTER,
    # matching _prepare_vanillux_runtime's order (~lines 398-408).
    wrapper_path = tmp_path / "wrapper.sh"
    events = []
    monkeypatch.setattr(
        "tmax_opsd_v1.program.subprocess.run",
        lambda args, **k: events.append((args, wrapper_path.exists())),
    )
    prepare_runtime(wrapper_path=str(wrapper_path))

    assert events == [
        (["bash", "-c", RUNTIME_PREP_CMD], False),  # prep before the wrapper exists
        (["chmod", "+x", str(wrapper_path)], True),  # chmod after it's written
    ]
    assert wrapper_path.read_text() == BASH_WRAPPER


# ---------------------------------------------------------------------------
# 1c already covered via run_agent_loop client-kwargs test below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1h. stderr merge
# ---------------------------------------------------------------------------


def test_merge_output_stdout_only():
    assert merge_output("hello", "") == "hello"


def test_merge_output_stderr_only():
    assert merge_output("", "boom") == "boom"


def test_merge_output_both():
    assert merge_output("out", "err") == "out\nerr"


def test_merge_output_neither():
    assert merge_output("", "") == ""


# ---------------------------------------------------------------------------
# Final-review fix 1: timeout observation keeps partial output + prepends
# TMax's exact message (backends.py ~275-278, 324-326, 711-714, 725-726).
# Final-review fix 2: per-stream 1,000,000-BYTE slice before decode and merge
# (backends.py ~169, 322-323, 581, 723-724).
# Final-review fix 3: bytes capture + decode(errors="replace") — invalid UTF-8
# yields mojibake with the real exit code, never an `error:` observation
# (backends.py ~322-323 decode semantics).
# ---------------------------------------------------------------------------


def test_build_bash_result_caps_each_stream_before_merge():
    stdout = b"o" * 1_000_100
    stderr = b"e" * 1_000_100
    output, exit_code = build_bash_result(stdout, stderr, 0, timeout=30.0)
    expected_stdout = "o" * MAX_OUTPUT_BYTES
    expected_stderr = "e" * MAX_OUTPUT_BYTES
    assert output == expected_stdout + "\n" + expected_stderr
    assert exit_code == 0


def test_build_bash_result_leaves_short_streams_untouched():
    output, exit_code = build_bash_result(b"out", b"err", 0, timeout=30.0)
    assert output == "out\nerr"
    assert exit_code == 0


def test_build_bash_result_timeout_prepends_exact_message():
    output, exit_code = build_bash_result(b"partial out", b"partial err", 124, timeout=30.0)
    assert exit_code == 124
    assert output == "partial out\nCommand timed out after 30s.\npartial err"


def test_build_bash_result_timeout_message_uses_int_seconds():
    output, exit_code = build_bash_result(b"", b"", 124, timeout=45.7)
    assert exit_code == 124
    assert output == "Command timed out after 45s.\n"


def test_build_bash_result_timeout_message_prepended_after_cap_not_before():
    # TMax truncates stderr to MAX_OUTPUT_BYTES FIRST, then prepends the timeout
    # message — so the final stderr is (message + capped stderr), which is LONGER
    # than MAX_OUTPUT_BYTES. If the cap were applied after prepending instead, the
    # tail of the message+stderr blob would be sliced off — a divergence.
    huge_stderr = b"e" * (MAX_OUTPUT_BYTES + 500)
    output, exit_code = build_bash_result(b"", huge_stderr, 124, timeout=10.0)
    expected = "Command timed out after 10s.\n" + ("e" * MAX_OUTPUT_BYTES)
    assert output == expected
    assert exit_code == 124


def test_build_bash_result_non_timeout_exit_code_no_message():
    output, exit_code = build_bash_result(b"out", b"err", 1, timeout=30.0)
    assert output == "out\nerr"
    assert exit_code == 1


def test_build_bash_result_invalid_utf8_decodes_with_replacement():
    # backends.py ~322-323: .decode("utf-8", errors="replace") — invalid bytes
    # become U+FFFD, never an exception.
    output, exit_code = build_bash_result(b"ok \xff\xfe done", b"\x80 err", 2, timeout=30.0)
    assert exit_code == 2
    assert output == "ok �� done\n� err"


def test_build_bash_result_byte_cap_can_split_a_multibyte_char():
    # Slicing RAW BYTES at the cap (TMax's semantics) may cut a multi-byte char
    # in half; errors="replace" turns the dangling prefix into U+FFFD. A
    # chars-based cap would keep the full "é" — pinning bytes semantics exactly.
    stdout = b"a" * (MAX_OUTPUT_BYTES - 1) + "é".encode()  # 2-byte char straddles the cap
    output, exit_code = build_bash_result(stdout, b"", 0, timeout=30.0)
    assert exit_code == 0
    assert output == "a" * (MAX_OUTPUT_BYTES - 1) + "�"


def test_run_bash_timeout_keeps_partial_output_via_build_bash_result(monkeypatch):
    # subprocess.TimeoutExpired.stdout/.stderr are the raw bytes captured before
    # the kill (or None); run_bash must route them through build_bash_result
    # rather than discarding the partial output.
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"], output=b"partial stdout\n", stderr=b"partial stderr\n")

    monkeypatch.setattr("tmax_opsd_v1.program.subprocess.run", fake_run)
    output, exit_code = run_bash("sleep 5", 2.0)
    assert exit_code == 124
    assert output == "partial stdout\n\nCommand timed out after 2s.\npartial stderr\n"


def test_run_bash_timeout_with_no_partial_output(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"], output=None, stderr=None)

    monkeypatch.setattr("tmax_opsd_v1.program.subprocess.run", fake_run)
    output, exit_code = run_bash("sleep 5", 3.0)
    assert exit_code == 124
    assert output == "Command timed out after 3s.\n"


def test_run_bash_success_path_applies_the_output_cap(monkeypatch):
    class FakeResult:
        stdout = b"x" * 1_000_050
        stderr = b""
        returncode = 0

    monkeypatch.setattr("tmax_opsd_v1.program.subprocess.run", lambda *a, **k: FakeResult())
    output, exit_code = run_bash("echo hi", 30.0)
    assert exit_code == 0
    assert output == "x" * MAX_OUTPUT_BYTES


def test_run_bash_captures_bytes_not_text(monkeypatch):
    # run_bash must NOT pass text=True: raw-bytes capture is what lets the
    # byte-cap and the errors="replace" decode match backends.py — and what
    # keeps invalid UTF-8 from raising inside communicate().
    captured_kwargs = {}

    class FakeResult:
        stdout = b"ok"
        stderr = b""
        returncode = 0

    def fake_run(args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeResult()

    monkeypatch.setattr("tmax_opsd_v1.program.subprocess.run", fake_run)
    run_bash("echo hi", 30.0)
    assert "text" not in captured_kwargs
    assert captured_kwargs.get("universal_newlines") is None


def _run_bash_with_wrapper(tmp_path, wrapper_body, command, timeout):
    """Drive the real run_bash with a temp wrapper script (no mocks)."""
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(wrapper_body)
    wrapper.chmod(0o755)
    import tmax_opsd_v1.program as program_mod

    original = program_mod.WRAPPER_PATH
    program_mod.WRAPPER_PATH = str(wrapper)
    try:
        return run_bash(command, timeout)
    finally:
        program_mod.WRAPPER_PATH = original


def test_run_bash_real_timeout_keeps_partial_output_end_to_end(tmp_path):
    # A real (short) timeout, exercising the actual subprocess.run TimeoutExpired
    # path end-to-end rather than a mock — kept well under 1s.
    output, exit_code = _run_bash_with_wrapper(
        tmp_path, '#!/bin/bash\necho "before timeout"\nsleep 5\n', "ignored", 0.2
    )
    assert exit_code == 124
    assert "Command timed out after 0s.\n" in output
    assert "before timeout" in output


def test_run_bash_real_invalid_utf8_yields_replacement_chars_and_real_exit_code(tmp_path):
    # An agent cat-ing a binary is a real occurrence; TMax's decode(errors=
    # "replace") shows mojibake plus the real exit code and the rollout
    # continues. With text=True this raised UnicodeDecodeError inside
    # communicate() and surfaced as `error: 'utf-8' codec can't decode...`/exit 1.
    output, exit_code = _run_bash_with_wrapper(
        tmp_path, "#!/bin/bash\nprintf 'A\\xff\\xfeB'\nexit 3\n", "ignored", 5.0
    )
    assert exit_code == 3
    assert output == "A��B"
    assert not output.startswith("error:")


# ---------------------------------------------------------------------------
# 1i. Truncation byte-exact
# ---------------------------------------------------------------------------


def test_short_output_untouched():
    assert truncate_observation("hello") == "hello"


def test_too_long_hint_matches_prompts_yaml():
    assert TOO_LONG_HINT == PROMPTS["observation"]["too_long_hint"]


def test_long_output_truncation_byte_exact():
    output = "A" * 6000 + "B" * 6000
    result = truncate_observation(output)
    elided = len(output) - 5000 - 5000
    expected = (
        f"{TOO_LONG_HINT}\n\n"
        f"---- HEAD (5000 chars) ----\n"
        f"{output[:5000]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL (5000 chars) ----\n"
        f"{output[-5000:]}"
    )
    assert result == expected


def test_build_observation_no_output():
    assert build_observation("", 0) == "(no output)\n\n(exit_code=0)"


def test_build_observation_with_output_and_exit_code():
    obs = build_observation("done", 7)
    assert obs == "done\n\n(exit_code=7)"


# ---------------------------------------------------------------------------
# format_error_message / FORMAT_ERROR_TEMPLATE / TOOL_CALL_FORMAT_ERROR_MESSAGE
# ---------------------------------------------------------------------------


def test_format_error_template_matches_prompts_yaml():
    assert FORMAT_ERROR_TEMPLATE == PROMPTS["format_error_template"]


def test_format_error_message_substitutes_error():
    msg = format_error_message("'command' parameter is required.")
    assert msg == (
        "Format error: 'command' parameter is required.\n\n"
        "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
        f"end the task, please issue the command `echo {SUBMIT_MARKER}`\n"
        "via the `bash` tool, with no other content in the command.\n"
    )


def test_tool_call_format_error_message_byte_exact():
    assert TOOL_CALL_FORMAT_ERROR_MESSAGE == (
        "Format error: Your last response did not include a valid `bash` tool call.\n\n"
        "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
        f"end the task, please issue the command `echo {SUBMIT_MARKER}`\n"
        "via the `bash` tool, with no other content in the command."
    )


# ---------------------------------------------------------------------------
# Tool schema byte-exact (1l)
# ---------------------------------------------------------------------------


def test_bash_tool_schema_byte_exact():
    assert BASH_TOOL == {
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


# ---------------------------------------------------------------------------
# parse_command_arg
# ---------------------------------------------------------------------------


def test_parse_command_arg_valid():
    assert parse_command_arg('{"command": "ls"}') == "ls"


def test_parse_command_arg_missing_key():
    assert parse_command_arg("{}") is None


def test_parse_command_arg_empty_string_command():
    assert parse_command_arg('{"command": ""}') is None


def test_parse_command_arg_unparseable_json():
    assert parse_command_arg("not json") is None


def test_parse_command_arg_non_dict_json():
    assert parse_command_arg("[1, 2]") is None


def test_parse_command_arg_none_arguments():
    assert parse_command_arg(None) is None


# ---------------------------------------------------------------------------
# apply_last_step_warning (1k)
# ---------------------------------------------------------------------------


def test_last_step_warning_text_exact():
    assert LAST_STEP_WARNING == (
        "Warning: you only have one more tool call remaining. "
        f"You must end your next tool call with `echo {SUBMIT_MARKER}`"
    )


def test_last_step_warning_appended_when_enabled_and_penultimate():
    obs = apply_last_step_warning("obs text", enabled=True, steps=3, max_steps=4)
    assert obs == f"obs text\n\n{LAST_STEP_WARNING}"


def test_last_step_warning_not_appended_when_disabled():
    obs = apply_last_step_warning("obs text", enabled=False, steps=3, max_steps=4)
    assert obs == "obs text"


def test_last_step_warning_not_appended_when_not_penultimate():
    obs = apply_last_step_warning("obs text", enabled=True, steps=1, max_steps=4)
    assert obs == "obs text"


# ---------------------------------------------------------------------------
# run_agent_loop — 1d/1e/1f/1g/1j/1k end-to-end via a fake client.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tool_call_breaks_when_feedback_off():
    client = FakeClient([FakeMessage(tool_calls=None)])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=5, command_timeout=1.0,
        format_error_feedback=False,
        run_bash_fn=fake_run_bash(),
    )
    assert len(client.calls) == 1
    # Only the assistant turn was appended; no feedback message; loop ended.
    assert result[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_no_tool_call_injects_user_feedback_when_flag_on():
    client = FakeClient([FakeMessage(tool_calls=None)])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=1, command_timeout=1.0,
        format_error_feedback=True,
        run_bash_fn=fake_run_bash(),
    )
    assert result[-1] == {"role": "user", "content": TOOL_CALL_FORMAT_ERROR_MESSAGE}


@pytest.mark.asyncio
async def test_tool_calls_but_none_valid_break_when_feedback_off():
    client = FakeClient([FakeMessage(tool_calls=[FakeToolCall("id1", "edit")])])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=5, command_timeout=1.0,
        format_error_feedback=False,
        run_bash_fn=fake_run_bash(),
    )
    assert len(client.calls) == 1
    assert result[-1]["role"] == "assistant"  # no tool responses appended; loop ended


@pytest.mark.asyncio
async def test_tool_calls_but_none_valid_inject_tool_feedback_when_flag_on():
    client = FakeClient([FakeMessage(tool_calls=[FakeToolCall("id1", "edit")])])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=1, command_timeout=1.0,
        format_error_feedback=True,
        run_bash_fn=fake_run_bash(),
    )
    assert result[-1] == {
        "role": "tool", "tool_call_id": "id1", "content": TOOL_CALL_FORMAT_ERROR_MESSAGE,
    }


@pytest.mark.asyncio
async def test_multi_call_execution_order_and_step_accounting():
    calls_seen = []

    def recording_run_bash(command, timeout):
        calls_seen.append(command)
        return f"ran {command}", 0

    client = FakeClient([
        FakeMessage(tool_calls=[
            FakeToolCall("id1", "bash", '{"command": "echo a"}'),
            FakeToolCall("id2", "bash", '{"command": "echo b"}'),
        ]),
    ])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=2, command_timeout=1.0,
        run_bash_fn=recording_run_bash,
    )
    assert calls_seen == ["echo a", "echo b"]
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["id1", "id2"]
    assert tool_msgs[0]["content"] == "ran echo a\n\n(exit_code=0)"
    assert tool_msgs[1]["content"] == "ran echo b\n\n(exit_code=0)"


@pytest.mark.asyncio
async def test_budget_cutoff_mid_turn():
    calls_seen = []

    def recording_run_bash(command, timeout):
        calls_seen.append(command)
        return "ok", 0

    client = FakeClient([
        FakeMessage(tool_calls=[
            FakeToolCall("id1", "bash", '{"command": "one"}'),
            FakeToolCall("id2", "bash", '{"command": "two"}'),
            FakeToolCall("id3", "bash", '{"command": "three"}'),
        ]),
    ])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=2, command_timeout=1.0,
        run_bash_fn=recording_run_bash,
    )
    assert calls_seen == ["one", "two"]  # third never executed — budget exhausted
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["id1", "id2"]


@pytest.mark.asyncio
async def test_mixed_valid_and_unknown_tool_calls_flag_off():
    client = FakeClient([
        FakeMessage(tool_calls=[
            FakeToolCall("id1", "bash", '{"command": "echo hi"}'),
            FakeToolCall("id2", "str_replace_editor", "{}"),
        ]),
    ])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=1, command_timeout=1.0,
        run_bash_fn=fake_run_bash("hi", 0),
    )
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    # Wrong-call responses precede bash observations (TMax appends non-dispatch
    # observations before the dispatch loop's). No step charged for the filler:
    # the single available step went to the bash call.
    assert tool_msgs[0]["tool_call_id"] == "id2"
    assert tool_msgs[0]["content"] == format_error_message(
        "Unknown tool 'str_replace_editor'. The only available tool is `bash`."
    )
    assert tool_msgs[1]["tool_call_id"] == "id1"
    assert tool_msgs[1]["content"] == "hi\n\n(exit_code=0)"


@pytest.mark.asyncio
async def test_mixed_turn_flag_on_charges_feedback_step_and_executes_bash():
    # vllm_utils.py:1235 computes has_wrong_tool_call over the UNFILTERED calls:
    # with the flag on, a mixed turn gets the shared tool-role feedback (one step)
    # AND the valid bash call still executes (one more step). max_steps=2 means
    # both steps are consumed by this single turn, so exactly one API call happens.
    calls_seen = []

    def recording_run_bash(command, timeout):
        calls_seen.append(command)
        return "hi", 0

    client = FakeClient([
        FakeMessage(tool_calls=[
            FakeToolCall("id1", "bash", '{"command": "echo hi"}'),
            FakeToolCall("id2", "str_replace_editor", "{}"),
        ]),
    ])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=2, command_timeout=1.0,
        format_error_feedback=True,
        run_bash_fn=recording_run_bash,
    )
    assert calls_seen == ["echo hi"]  # the valid bash call still executed
    assert len(client.calls) == 1  # feedback + bash = 2 steps -> budget exhausted
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    # Feedback first (vllm_utils appends feedback before dispatch observations),
    # answering the offending id; then the bash observation.
    assert tool_msgs == [
        {"role": "tool", "tool_call_id": "id2", "content": TOOL_CALL_FORMAT_ERROR_MESSAGE},
        {"role": "tool", "tool_call_id": "id1", "content": "hi\n\n(exit_code=0)"},
    ]


@pytest.mark.asyncio
async def test_mixed_turn_flag_on_feedback_consumes_last_step_before_bash():
    # With only one step left, the feedback charge exhausts the budget and the
    # bash call must NOT execute (TMax's dispatch loop checks the budget before
    # each call, vllm_utils.py:1265).
    calls_seen = []

    def recording_run_bash(command, timeout):
        calls_seen.append(command)
        return "hi", 0

    client = FakeClient([
        FakeMessage(tool_calls=[
            FakeToolCall("id1", "bash", '{"command": "echo hi"}'),
            FakeToolCall("id2", "str_replace_editor", "{}"),
        ]),
    ])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=1, command_timeout=1.0,
        format_error_feedback=True,
        run_bash_fn=recording_run_bash,
    )
    assert calls_seen == []  # budget went to the feedback; bash never ran
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert tool_msgs == [
        {"role": "tool", "tool_call_id": "id2", "content": TOOL_CALL_FORMAT_ERROR_MESSAGE},
    ]


@pytest.mark.asyncio
async def test_empty_command_always_errors_regardless_of_flag():
    for flag in (False, True):
        client = FakeClient([
            FakeMessage(tool_calls=[FakeToolCall("id1", "bash", "{}")]),
        ])
        messages = [{"role": "user", "content": "hi"}]
        result = await run_agent_loop(
            client, messages,
            model="m", max_steps=1, command_timeout=1.0,
            format_error_feedback=flag,
            run_bash_fn=fake_run_bash(),
        )
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert tool_msgs[-1]["content"] == format_error_message(
            "'command' parameter is required."
        )


@pytest.mark.asyncio
async def test_submit_marker_touches_sentinel_and_stops_loop(tmp_path):
    sentinel = tmp_path / ".vanillux_submitted"
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall("id1", "bash", '{"command": "echo done"}')]),
    ])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=64, command_timeout=1.0,
        run_bash_fn=fake_run_bash(f"{SUBMIT_MARKER}\n", 0),
        submit_sentinel_path=str(sentinel),
    )
    assert sentinel.exists()
    assert len(client.calls) == 1  # loop stopped, no further turns requested
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert tool_msgs[-1]["content"] == f"{SUBMIT_MARKER}\n\n\n(exit_code=0)"


@pytest.mark.asyncio
async def test_last_step_warning_gated_and_exact_text():
    # max_steps=2: the first (and only) bash call lands on step 1 == max_steps-1,
    # i.e. the penultimate step, so it must carry the warning. A second turn with
    # no tool call then ends the rollout cleanly (feedback off by default).
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall("id1", "bash", '{"command": "x"}')]),
        FakeMessage(tool_calls=None),
    ])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=2, command_timeout=1.0,
        last_step_warning=True,
        run_bash_fn=fake_run_bash("out", 0),
    )
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert tool_msgs[-1]["content"] == f"out\n\n(exit_code=0)\n\n{LAST_STEP_WARNING}"


@pytest.mark.asyncio
async def test_last_step_warning_off_by_default():
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall("id1", "bash", '{"command": "x"}')]),
        FakeMessage(tool_calls=None),
    ])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=2, command_timeout=1.0,
        run_bash_fn=fake_run_bash("out", 0),
    )
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert tool_msgs[-1]["content"] == "out\n\n(exit_code=0)"


@pytest.mark.asyncio
async def test_format_error_feedback_does_not_get_last_step_warning():
    # Format-error feedback is injected upstream of TMax's env.step() (which is the
    # only thing _with_last_step_warning wraps), so it never receives the suffix
    # even when it consumes the penultimate step. See program.py's comment.
    client = FakeClient([FakeMessage(tool_calls=None)])
    messages = [{"role": "user", "content": "hi"}]
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=1, command_timeout=1.0,
        format_error_feedback=True,
        last_step_warning=True,
        run_bash_fn=fake_run_bash(),
    )
    assert result[-1]["content"] == TOOL_CALL_FORMAT_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_temperature_and_max_tokens_forwarded_to_client():
    client = FakeClient([FakeMessage(tool_calls=None)])
    messages = [{"role": "user", "content": "hi"}]
    await run_agent_loop(
        client, messages,
        model="m", max_steps=1, command_timeout=1.0,
        temperature=0.3, per_turn_max_tokens=1234,
        run_bash_fn=fake_run_bash(),
    )
    assert client.calls[0]["temperature"] == 0.3
    assert client.calls[0]["max_tokens"] == 1234
    assert client.calls[0]["model"] == "m"
    assert client.calls[0]["tools"] == [BASH_TOOL]


@pytest.mark.asyncio
async def test_no_bash_calls_at_all_stops_before_any_step_increment():
    # steps never incremented when flag off and there's truly nothing to do.
    client = FakeClient([FakeMessage(tool_calls=None)])
    messages = [{"role": "user", "content": "hi"}]
    # max_steps=0 would never even enter the loop; use 1 to prove the break
    # happens without consuming the single available step.
    result = await run_agent_loop(
        client, messages,
        model="m", max_steps=1, command_timeout=1.0,
        format_error_feedback=False,
        run_bash_fn=fake_run_bash(),
    )
    assert len(client.calls) == 1
