# tmax-opsd-v1

A verifiers-v1 environment packaging the [TMax](https://arxiv.org/abs/2606.23321) terminal-agent
dataset (14,601 Docker-image tasks) with a TMax-faithful **vanillux** harness, built for
on-policy self-distillation (OPSD) experiments.

- **Taskset** `TMaxTaskset`: joins Ai2's two TMax releases (`allenai/TMax-15K`
  and `allenai/tmax-15k-open-instruct`); each task runs its own Docker image. The dataset is
  **built lazily on first load** (downloaded + joined into a cached `tasks.parquet`), so a fresh
  install needs no manual data step.
- **Harness** `VanilluxHarness`: single persistent-shell `bash` tool, `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  submit marker, 10k head/tail observation truncation, format-error nudge — matching TMax's
  mini-SWE-agent-derived harness.
- **Reward** `solved`: binary 0/1 from the task's hidden pytest verifier (`/logs/verifier/reward.txt`),
  run once at scoring time. Plus `@metric pass_fraction` and `@metric hacked` (oracle-copy detector),
  logged, never summed into reward.
- **`validate()`**: pre-flight fixture check that flags dead/broken images (a known ~15% of the corpus).

Runtime: use the `prime` runtime (Prime Sandboxes) or a Linux `docker` host. Set
`TMAX_OPSD_DATA_DIR` to control where the built parquet is cached.
