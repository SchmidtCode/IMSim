from __future__ import annotations

from dataclasses import replace

from imsim.models import TrainingProfile
from imsim.performance import (
    PerformanceOptions,
    compact_dashboard_snapshot,
    format_result,
    result_to_dict,
    run_performance_test,
)
from imsim.repository import create_session_repository
from imsim.services.training import build_simulator_state, unlock_all_academy_levels


def _options(scenario: str) -> PerformanceOptions:
    return PerformanceOptions(
        scenario=scenario,
        users=2,
        ticks=2,
        duration_seconds=None,
        tick_rate=0.0,
        throttle=False,
        workers=2,
        render_profile="tick-only",
        render_every=1,
        split_panel="all",
        auto_po=True,
    )


def test_performance_harness_runs_simulator_users(test_config, tmp_path):
    config = replace(test_config, session_dir=tmp_path / "perf-sessions", database_url=None)
    repository = create_session_repository(config)

    result = run_performance_test(repository, _options("simulator"), session_dir=config.session_dir)

    assert result.total_ticks == 4
    assert result.total_errors == 0
    assert result.total_conflicts == 0
    assert "Measured throughput" in format_result(result)
    assert result.stage_latency["load_state"].count == 4
    assert result_to_dict(result)["options"]["render_profile"] == "tick-only"


def test_performance_harness_runs_academy_users(test_config, tmp_path):
    config = replace(test_config, session_dir=tmp_path / "academy-perf", database_url=None)
    repository = create_session_repository(config)

    result = run_performance_test(repository, _options("academy"), session_dir=config.session_dir)

    assert result.total_ticks == 4
    assert result.total_errors == 0
    assert result.total_conflicts == 0


def test_performance_harness_payload_profile_records_payloads(test_config, tmp_path):
    config = replace(test_config, session_dir=tmp_path / "payload-perf", database_url=None)
    repository = create_session_repository(config)
    options = replace(_options("simulator"), render_profile="payload-only")

    result = run_performance_test(repository, options, session_dir=config.session_dir)

    assert result.total_ticks == 4
    assert result.total_renders == 4
    assert result.payload.count == 4
    assert result.stage_latency["build_payload"].count == 4
    assert result.stage_latency["json_serialize"].count == 4


def test_compact_dashboard_snapshot_is_plain_json_shape():
    profile = TrainingProfile()
    unlock_all_academy_levels(profile)
    state = build_simulator_state(profile)

    snapshot = compact_dashboard_snapshot(state)

    assert snapshot["schema"] == "imsim.dashboard.snapshot.v1"
    assert snapshot["view"] == "simulator"
    assert snapshot["items"]
    assert snapshot["history"]
    assert "figure" not in snapshot
    assert "grid" not in snapshot
