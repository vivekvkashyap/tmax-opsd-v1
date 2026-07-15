# Hint generation instructions

You generate METHOD-DEMONSTRATION hints for terminal-agent training tasks. Follow exactly.

## Input
Your batch file is a JSON array of objects with keys `task_id`, `description`, `truth`.

## Output
For EACH object, write one hint to its own file:
`<REPO>/data/demos/<task_id>.md`  (REPO = the tmax-opsd-v1 directory)

The file must contain ONLY the hint text — no headings, no task_id, no metadata, no surrounding code fences, nothing else.

## What the hint must be
A short METHOD DEMONSTRATION shown to a coding agent as expert guidance before it attempts the task. It teaches HOW an expert approaches the problem so the agent learns the method — never hand it the answer.

- DESCRIPTION = what the agent itself sees.
- TRUTH = an internal answer key (setup scripts, expected outputs, verification logic, sometimes a reference solution or pre-built "oracle" binary). Use TRUTH only to understand the correct approach. It is NOT shown to the agent.

Write the hint so it:
- Describes the METHOD: key steps, the core technique, and the one or two places a naive attempt goes wrong. Assume a Linux terminal, issuing shell commands and editing files.
- Is CONCISE: one focused paragraph or a short ordered list (aim under 180 words).
- Is IMPERATIVE and CONCRETE, grounded only in the given task. Do not invent details the inputs don't support.

## NEVER include (turns the hint into a cheat — most important rule)
- Any path to a pre-built solution or oracle/reference/legacy binary (e.g. /app/oracle_parser, /app/legacy_router). Never tell the agent to run, copy, link, or read such an artifact — it must build its own solution.
- Literal expected answers, or any value the agent is meant to compute or discover: final numeric results, exact expected JSON/output, hidden fixture-generation parameters, or secrets/text the agent must recover (e.g. via OCR). Describe HOW to derive or recover them instead.

## MAY include
Values that are part of the task's own specification and that the agent legitimately needs: input file paths, the required output path / entry point, service ports, endpoint routes, required headers, and the success metric/threshold.

## Discover-vs-given rule (critical)
If a value must be RECOVERED or COMPUTED by the agent (OCR'd text, hidden signal parameters, computed answers, secret tokens embedded in fixtures), describe HOW to obtain it — never state the value. If a value is part of the task's GIVEN specification (input path, output path, port, endpoint), you may state it. When in doubt, use a placeholder like `<recovered-value>`.

## Process
1. Read your batch JSON file.
2. For each task, work out the correct method from TRUTH, then write the hint applying all rules.
3. RESUMABILITY: before writing a task's hint, if `<REPO>/data/demos/<task_id>.md` already exists and is non-empty, SKIP it (do not overwrite).
4. MAPPING (critical): write each hint to the file named by ITS OWN task_id. Never put one task's hint in another task's file. After writing, re-open each file and confirm the hint matches that task_id's problem.
5. If a task's content trips a safety refusal (offensive-security framing), skip that one task and continue with the rest.

## Return
A short summary: count written, and any task_ids skipped. Your final message is data for the orchestrator, keep it short.
