from __future__ import annotations

from dataclasses import replace

from imsim.performance import PerformanceOptions, format_result, run_performance_test
from imsim.repository import create_session_repository


def _options(scenario: str) -> PerformanceOptions:
    return PerformanceOptions(
        scenario=scenario,
        users=2,
        ticks=2,
        duration_seconds=None,
        tick_rate=0.0,
        throttle=False,
        workers=2,
        render=False,
        render_every=1,
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


def test_performance_harness_runs_academy_users(test_config, tmp_path):
    config = replace(test_config, session_dir=tmp_path / "academy-perf", database_url=None)
    repository = create_session_repository(config)

    result = run_performance_test(repository, _options("academy"), session_dir=config.session_dir)

    assert result.total_ticks == 4
    assert result.total_errors == 0
    assert result.total_conflicts == 0
