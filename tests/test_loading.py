from verifiers.v1.loaders import default_harness_id, harness_class, taskset_class

from tmax_opsd_v1.harness import VanilluxHarness, VanilluxHarnessConfig
from tmax_opsd_v1.taskset import TMaxTaskset


def test_verifiers_resolves_both_plugins():
    assert taskset_class("tmax_opsd_v1") is TMaxTaskset
    assert harness_class("tmax_opsd_v1") is VanilluxHarness
    assert default_harness_id("tmax_opsd_v1") == "tmax_opsd_v1"


def test_harness_defaults_match_tmax():
    config = VanilluxHarnessConfig(id="tmax_opsd_v1")
    assert config.max_steps == 64
    assert config.command_timeout == 120.0
    harness = VanilluxHarness(config)
    assert harness.APPENDS_SYSTEM_PROMPT is True
