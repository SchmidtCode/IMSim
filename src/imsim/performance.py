from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import threading
import time
import tracemalloc
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from plotly.utils import PlotlyJSONEncoder

from .config import IMSimConfig
from .models import TrainingProfile
from .repository import SessionConflictError, SessionRepository, create_session_repository
from .services.simulation import place_custom_orders, place_purchase_orders, tick_state
from .services.training import (
    academy_levels,
    active_level,
    build_level_state,
    build_simulator_state,
    is_action_allowed,
    record_custom_order,
    record_guided_order,
    unlock_all_academy_levels,
)
from .ui.components import (
    build_exception_center,
    build_inventory_table,
    build_kpi_strip,
    costs_card_children,
    refresh_inventory_figure,
    sales_card_children,
    service_card_children,
)

SCENARIOS = ("simulator", "academy", "mixed")


@dataclass(frozen=True, slots=True)
class PerformanceOptions:
    scenario: str
    users: int
    ticks: int
    duration_seconds: float | None
    tick_rate: float
    throttle: bool
    workers: int
    render: bool
    render_every: int
    auto_po: bool


@dataclass(slots=True)
class UserResult:
    session_id: str
    scenario: str
    ticks: int = 0
    renders: int = 0
    errors: int = 0
    conflicts: int = 0
    behind_schedule: int = 0
    latency_seconds: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latency_seconds is None:
            self.latency_seconds = []


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class PerformanceResult:
    options: PerformanceOptions
    wall_seconds: float
    total_ticks: int
    total_renders: int
    total_errors: int
    total_conflicts: int
    total_behind_schedule: int
    latency: LatencySummary
    peak_allocated_mb: float
    session_dir: Path | None

    @property
    def ticks_per_second(self) -> float:
        return self.total_ticks / max(self.wall_seconds, 1e-9)

    @property
    def target_ticks_per_second(self) -> float | None:
        if not self.options.throttle:
            return None
        return self.options.users * self.options.tick_rate


class _VirtualUser:
    def __init__(
        self,
        *,
        index: int,
        scenario: str,
        repository: SessionRepository,
        options: PerformanceOptions,
    ) -> None:
        self.index = index
        self.repository = repository
        self.options = options
        self.scenario = self._assigned_scenario(scenario, index)
        self.session_id = f"perf-{self.scenario}-{index}-{uuid.uuid4().hex[:8]}"
        self.level_index = (index % len(academy_levels())) + 1
        self.current_figure: Any = None

    @staticmethod
    def _assigned_scenario(scenario: str, index: int) -> str:
        if scenario != "mixed":
            return scenario
        return "simulator" if index % 2 == 0 else "academy"

    def run(self, deadline: float | None) -> UserResult:
        result = UserResult(session_id=self.session_id, scenario=self.scenario)
        interval = 1.0 / self.options.tick_rate if self.options.tick_rate > 0 else 0.0
        next_due = time.perf_counter()

        while self._should_continue(result.ticks, deadline):
            if self.options.throttle and interval > 0:
                now = time.perf_counter()
                if now < next_due:
                    time.sleep(next_due - now)
                elif result.ticks > 0:
                    result.behind_schedule += 1
                next_due += interval

            started = time.perf_counter()
            try:
                rendered = self._tick_once(result.ticks)
            except SessionConflictError:
                result.conflicts += 1
                continue
            except Exception:
                result.errors += 1
                continue
            elapsed = time.perf_counter() - started
            result.ticks += 1
            result.renders += int(rendered)
            assert result.latency_seconds is not None
            result.latency_seconds.append(elapsed)
        return result

    def _should_continue(self, ticks_done: int, deadline: float | None) -> bool:
        if deadline is not None:
            return time.perf_counter() < deadline
        return ticks_done < self.options.ticks

    def _tick_once(self, tick_index: int) -> bool:
        state = self.repository.get_or_create(self.session_id)
        if self.scenario == "simulator":
            state = self._ready_simulator_state(state)
        else:
            state = self._ready_academy_state(state)

        if self.scenario == "academy":
            self._script_academy_actions(state)

        tick_state(state)
        self.repository.save(self.session_id, state)

        should_render = (
            self.options.render
            and self.options.render_every > 0
            and tick_index % self.options.render_every == 0
        )
        if should_render:
            self.current_figure = _render_dashboard_snapshot(state, self.current_figure)
        return should_render

    def _ready_simulator_state(self, state):
        if state.training.current_view != "simulator" or not state.items:
            profile = TrainingProfile()
            unlock_all_academy_levels(profile)
            next_state = build_simulator_state(profile)
            next_state.revision = state.revision
            state = next_state
        state.training.simulator_unlocked = True
        state.training.auto_po_reward_unlocked = True
        state.global_settings.auto_po_enabled = self.options.auto_po
        state.is_initialized = True
        return state

    def _ready_academy_state(self, state):
        needs_next_lesson = (
            state.training.current_view != "lesson"
            or not state.items
            or state.training.lesson_status in {"passed", "failed"}
        )
        if needs_next_lesson:
            profile = state.training
            unlock_all_academy_levels(profile)
            level = academy_levels()[self.level_index - 1]
            next_state = build_level_state(level.level_id, profile)
            next_state.revision = state.revision
            state = next_state
            self.level_index = (self.level_index % len(academy_levels())) + 1
        state.is_initialized = True
        return state

    def _script_academy_actions(self, state) -> None:
        level = active_level(state)
        if level is None:
            return
        if is_action_allowed(state, "guided_po"):
            below_op = any(item.pna < item.op for item in state.items)
            summary = place_purchase_orders(state)
            if int(summary["lines"]) > 0:
                record_guided_order(state, below_op=below_op)
        if is_action_allowed(state, "custom_order"):
            quantities = [item.soq for item in state.items]
            if place_custom_orders(state, quantities):
                record_custom_order(state)


def _json_roundtrip(value: Any) -> None:
    json.dumps(value, cls=PlotlyJSONEncoder)


def _render_dashboard_snapshot(state, current_figure: Any) -> Any:
    figure = refresh_inventory_figure(state, "light", current_figure)
    payload = {
        "day": f"Day: {state.day}",
        "figure": figure,
        "service": service_card_children(state),
        "costs": costs_card_children(state),
        "sales": sales_card_children(state),
        "kpi": build_kpi_strip(state),
        "grid": build_inventory_table(state, "light"),
        "exceptions": build_exception_center(state),
    }
    _json_roundtrip(payload)
    return figure.to_plotly_json() if hasattr(figure, "to_plotly_json") else current_figure


def run_performance_test(
    repository: SessionRepository,
    options: PerformanceOptions,
    *,
    session_dir: Path | None = None,
) -> PerformanceResult:
    users = [
        _VirtualUser(index=index, scenario=options.scenario, repository=repository, options=options)
        for index in range(options.users)
    ]
    deadline = (
        time.perf_counter() + options.duration_seconds
        if options.duration_seconds is not None
        else None
    )
    started = time.perf_counter()
    tracemalloc.start()

    results: list[UserResult] = []
    result_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_run_user_into,
            args=(user, deadline, results, result_lock),
            daemon=True,
        )
        for user in users[: options.workers]
    ]
    queued_users = iter(users[options.workers :])

    for thread in threads:
        thread.start()
    while threads:
        for thread in tuple(threads):
            if thread.is_alive():
                continue
            thread.join()
            threads.remove(thread)
            next_user = next(queued_users, None)
            if next_user is not None:
                next_thread = threading.Thread(
                    target=_run_user_into,
                    args=(next_user, deadline, results, result_lock),
                    daemon=True,
                )
                threads.append(next_thread)
                next_thread.start()
        if threads:
            time.sleep(0.01)

    current_allocated, peak_allocated = tracemalloc.get_traced_memory()
    del current_allocated
    tracemalloc.stop()
    wall_seconds = time.perf_counter() - started
    return _summarize_results(options, results, wall_seconds, peak_allocated, session_dir)


def _run_user_into(
    user: _VirtualUser,
    deadline: float | None,
    results: list[UserResult],
    result_lock: threading.Lock,
) -> None:
    result = user.run(deadline)
    with result_lock:
        results.append(result)


def _summarize_results(
    options: PerformanceOptions,
    results: Iterable[UserResult],
    wall_seconds: float,
    peak_allocated: int,
    session_dir: Path | None,
) -> PerformanceResult:
    result_list = list(results)
    latencies = [
        latency
        for result in result_list
        for latency in (result.latency_seconds or [])
    ]
    return PerformanceResult(
        options=options,
        wall_seconds=wall_seconds,
        total_ticks=sum(result.ticks for result in result_list),
        total_renders=sum(result.renders for result in result_list),
        total_errors=sum(result.errors for result in result_list),
        total_conflicts=sum(result.conflicts for result in result_list),
        total_behind_schedule=sum(result.behind_schedule for result in result_list),
        latency=_latency_summary(latencies),
        peak_allocated_mb=peak_allocated / (1024 * 1024),
        session_dir=session_dir,
    )


def _latency_summary(latencies: list[float]) -> LatencySummary:
    if not latencies:
        return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(latencies)
    return LatencySummary(
        count=len(ordered),
        mean_ms=statistics.fmean(ordered) * 1000.0,
        p50_ms=_percentile_ms(ordered, 50),
        p90_ms=_percentile_ms(ordered, 90),
        p95_ms=_percentile_ms(ordered, 95),
        p99_ms=_percentile_ms(ordered, 99),
        max_ms=max(ordered) * 1000.0,
    )


def _percentile_ms(ordered_seconds: list[float], percentile: int) -> float:
    raw_index = math.ceil(percentile / 100 * len(ordered_seconds)) - 1
    index = max(0, min(len(ordered_seconds) - 1, raw_index))
    return ordered_seconds[index] * 1000.0


def _build_options(args: argparse.Namespace) -> PerformanceOptions:
    users = max(1, int(args.users))
    workers = max(1, min(int(args.workers or users), users))
    return PerformanceOptions(
        scenario=args.scenario,
        users=users,
        ticks=max(1, int(args.ticks)),
        duration_seconds=None if args.duration is None else max(0.1, float(args.duration)),
        tick_rate=max(0.0, float(args.tick_rate)),
        throttle=not args.no_throttle,
        workers=workers,
        render=not args.no_render,
        render_every=max(1, int(args.render_every)),
        auto_po=not args.no_auto_po,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an in-process IMSim load test that exercises simulation ticks, "
            "session persistence, and optional dashboard rendering."
        )
    )
    parser.add_argument("--scenario", choices=SCENARIOS, default="simulator")
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument(
        "--ticks",
        type=int,
        default=60,
        help="Ticks per user when no duration is set.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Run for this many seconds instead of fixed ticks.",
    )
    parser.add_argument(
        "--tick-rate",
        type=float,
        default=1.0,
        help="Ticks per virtual user per second. Use 6 for the dashboard 6x setting.",
    )
    parser.add_argument("--no-throttle", action="store_true", help="Run as fast as possible.")
    parser.add_argument("--workers", type=int, help="Concurrent worker threads. Defaults to users.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip dashboard component/figure rendering and measure tick + persistence only.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N ticks when rendering is enabled.",
    )
    parser.add_argument(
        "--no-auto-po",
        action="store_true",
        help="Disable simulator auto purchasing during the run.",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Use this file-session directory instead of an auto-deleted temp directory.",
    )
    parser.add_argument(
        "--database-url",
        help="Use this database URL for session persistence instead of file sessions.",
    )
    parser.add_argument(
        "--use-env-repository",
        action="store_true",
        help="Use IMSIM_DATABASE_URL/DATABASE_URL from the environment if present.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    options = _build_options(args)
    base_config = IMSimConfig.from_env()

    if args.database_url:
        config = replace(base_config, database_url=args.database_url)
        result = _run_with_config(config, options, None)
    elif args.use_env_repository:
        session_dir = None if base_config.database_url else base_config.session_dir
        result = _run_with_config(base_config, options, session_dir)
    elif args.session_dir is not None:
        session_dir = args.session_dir.expanduser().resolve()
        config = replace(base_config, database_url=None, session_dir=session_dir)
        result = _run_with_config(config, options, session_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="imsim-perf-") as session_dir_name:
            session_dir = Path(session_dir_name)
            config = replace(base_config, database_url=None, session_dir=session_dir)
            result = _run_with_config(config, options, session_dir)

    print(format_result(result))


def _run_with_config(
    config: IMSimConfig,
    options: PerformanceOptions,
    session_dir: Path | None,
) -> PerformanceResult:
    repository = create_session_repository(config)
    return run_performance_test(repository, options, session_dir=session_dir)


def format_result(result: PerformanceResult) -> str:
    latency = result.latency
    target = result.target_ticks_per_second
    target_line = (
        "Target throughput: unthrottled"
        if target is None
        else f"Target throughput: {target:.2f} ticks/sec"
    )
    session_line = (
        f"Session store: {result.session_dir}"
        if result.session_dir is not None
        else "Session store: configured database repository"
    )
    return "\n".join(
        [
            "IMSim performance run",
            f"Scenario: {result.options.scenario}",
            f"Users: {result.options.users} virtual users, {result.options.workers} workers",
            (
                f"Tick rate: {result.options.tick_rate:.2f}/user/sec, "
                f"throttle={result.options.throttle}"
            ),
            (
                f"Render dashboard: {result.options.render} "
                f"every {result.options.render_every} tick(s)"
            ),
            f"Auto purchasing: {result.options.auto_po}",
            session_line,
            target_line,
            f"Wall time: {result.wall_seconds:.2f}s",
            f"Completed ticks: {result.total_ticks}",
            f"Dashboard renders: {result.total_renders}",
            f"Measured throughput: {result.ticks_per_second:.2f} ticks/sec",
            (
                "Latency: "
                f"mean {latency.mean_ms:.2f}ms, p50 {latency.p50_ms:.2f}ms, "
                f"p90 {latency.p90_ms:.2f}ms, p95 {latency.p95_ms:.2f}ms, "
                f"p99 {latency.p99_ms:.2f}ms, max {latency.max_ms:.2f}ms"
            ),
            f"Behind-schedule ticks: {result.total_behind_schedule}",
            f"Save conflicts: {result.total_conflicts}",
            f"Errors: {result.total_errors}",
            f"Peak traced allocations: {result.peak_allocated_mb:.2f} MiB",
        ]
    )


if __name__ == "__main__":
    main()
