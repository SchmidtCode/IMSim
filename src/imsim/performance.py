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
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from plotly.utils import PlotlyJSONEncoder

from .config import IMSimConfig
from .models import SimulationState, TrainingProfile
from .repository import SessionConflictError, SessionRepository, create_session_repository
from .services.planning import item_on_order
from .services.simulation import place_custom_orders, place_purchase_orders, tick_state
from .services.training import (
    academy_levels,
    active_layout_variant,
    active_level,
    after_overhead_pct,
    average_inventory_value,
    build_level_state,
    build_simulator_state,
    fill_rate,
    is_action_allowed,
    lesson_days_remaining,
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
RENDER_PROFILES = ("full-server", "tick-only", "payload-only", "split-panels")
SPLIT_PANELS = ("all", "figure", "cards", "grid", "exceptions")
STAGE_NAMES = (
    "load_state",
    "tick",
    "save_state",
    "build_payload",
    "build_figure",
    "build_cards",
    "build_grid",
    "build_exceptions",
    "json_serialize",
)


@dataclass(frozen=True, slots=True)
class PerformanceOptions:
    scenario: str
    users: int
    ticks: int
    duration_seconds: float | None
    tick_rate: float
    throttle: bool
    workers: int
    render_profile: str
    render_every: int
    split_panel: str
    auto_po: bool

    @property
    def render(self) -> bool:
        return self.render_profile != "tick-only"


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
    stage_seconds: dict[str, list[float]] | None = None
    payload_bytes: list[int] | None = None
    error_messages: list[str] | None = None

    def __post_init__(self) -> None:
        if self.latency_seconds is None:
            self.latency_seconds = []
        if self.stage_seconds is None:
            self.stage_seconds = {stage: [] for stage in STAGE_NAMES}
        if self.payload_bytes is None:
            self.payload_bytes = []
        if self.error_messages is None:
            self.error_messages = []


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
class PayloadSummary:
    count: int
    total_bytes: int
    mean_bytes: float
    p95_bytes: int
    max_bytes: int


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
    stage_latency: dict[str, LatencySummary]
    payload: PayloadSummary
    peak_allocated_mb: float
    session_dir: Path | None
    error_messages: tuple[str, ...]

    @property
    def ticks_per_second(self) -> float:
        return self.total_ticks / max(self.wall_seconds, 1e-9)

    @property
    def target_ticks_per_second(self) -> float | None:
        if not self.options.throttle or self.options.tick_rate <= 0:
            return None
        return self.options.users * self.options.tick_rate

    @property
    def target_90_ticks_per_second(self) -> float | None:
        target = self.target_ticks_per_second
        return None if target is None else target * 0.9

    @property
    def meets_capacity_target(self) -> bool | None:
        target_90 = self.target_90_ticks_per_second
        if target_90 is None:
            return None
        return (
            self.ticks_per_second >= target_90
            and self.latency.p95_ms <= 1500.0
            and self.total_errors == 0
            and self.total_conflicts == 0
        )


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
                rendered = self._tick_once(result.ticks, result)
            except SessionConflictError:
                result.conflicts += 1
                continue
            except Exception as exc:
                result.errors += 1
                assert result.error_messages is not None
                if len(result.error_messages) < 5:
                    result.error_messages.append(f"{type(exc).__name__}: {exc}")
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

    def _tick_once(self, tick_index: int, result: UserResult) -> bool:
        state = _timed(result, "load_state", lambda: self.repository.get_or_create(self.session_id))
        state = _timed(result, "tick", lambda: self._prepare_and_tick(state))
        _timed(result, "save_state", lambda: self.repository.save(self.session_id, state))

        should_render = (
            self.options.render
            and self.options.render_every > 0
            and tick_index % self.options.render_every == 0
        )
        if should_render:
            self.current_figure = _render_dashboard_snapshot(
                state,
                self.current_figure,
                result,
                self.options,
            )
        return should_render

    def _prepare_and_tick(self, state: SimulationState) -> SimulationState:
        if self.scenario == "simulator":
            state = self._ready_simulator_state(state)
        else:
            state = self._ready_academy_state(state)

        if self.scenario == "academy":
            self._script_academy_actions(state)

        tick_state(state)
        return state

    def _ready_simulator_state(self, state: SimulationState) -> SimulationState:
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

    def _ready_academy_state(self, state: SimulationState) -> SimulationState:
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

    def _script_academy_actions(self, state: SimulationState) -> None:
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


def _timed(result: UserResult, stage: str, action: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    try:
        return action()
    finally:
        assert result.stage_seconds is not None
        result.stage_seconds.setdefault(stage, []).append(time.perf_counter() - started)


def _serialize_payload(result: UserResult, value: Any) -> int:
    started = time.perf_counter()
    encoded = json.dumps(value, cls=PlotlyJSONEncoder)
    elapsed = time.perf_counter() - started
    size = len(encoded.encode("utf-8"))
    assert result.stage_seconds is not None
    assert result.payload_bytes is not None
    result.stage_seconds["json_serialize"].append(elapsed)
    result.payload_bytes.append(size)
    return size


def compact_dashboard_snapshot(state: SimulationState, theme: str = "light") -> dict[str, Any]:
    level = active_level(state)
    fill = fill_rate(state)
    after = after_overhead_pct(state)
    avg_inventory = average_inventory_value(state)
    revenue = state.sales.revenue
    gross = revenue - state.sales.cogs
    turns = None if avg_inventory <= 0 else state.sales.cogs / avg_inventory
    gmroi = None if avg_inventory <= 0 else gross / avg_inventory
    return {
        "schema": "imsim.dashboard.snapshot.v1",
        "theme": theme,
        "view": state.training.current_view,
        "layout_variant": active_layout_variant(state),
        "day": state.day,
        "running": state.is_initialized,
        "level": None
        if level is None
        else {
            "id": level.level_id,
            "index": level.index,
            "title": level.title,
            "days_remaining": lesson_days_remaining(state),
            "lesson_status": state.training.lesson_status,
        },
        "metrics": {
            "fill_rate": fill,
            "after_overhead_pct": after,
            "revenue": state.sales.revenue,
            "cogs": state.sales.cogs,
            "gross": gross,
            "cost_total": state.costs.total,
            "cost_ordering": state.costs.ordering,
            "cost_holding": state.costs.holding,
            "cost_stockout": state.costs.stockout,
            "cost_expedite": state.costs.expedite,
            "cost_purchases": state.costs.purchases,
            "avg_inventory": avg_inventory,
            "turns": turns,
            "gmroi": gmroi,
            "orders_today": state.service_today.orders,
            "stockouts_today": state.service_today.orders_stockout,
            "orders_total": state.service_totals.orders,
            "stockouts_total": state.service_totals.orders_stockout,
            "units_sold": state.sales.units_sold,
        },
        "items": [
            _compact_item_row(index, item)
            for index, item in enumerate(state.items, start=1)
        ],
        "history": [
            {
                "day": point.day,
                "on_hand": point.total_on_hand,
                "on_order": point.total_on_order,
                "backorder": point.total_backorder,
                "pna": point.total_pna,
            }
            for point in state.history
        ],
        "exceptions": [
            {
                "day": exception.day,
                "item": exception.item_index + 1,
                "code": exception.code,
                "message": exception.message,
                "op": exception.op,
                "asq": exception.asq,
            }
            for exception in state.exception_center[-12:]
        ],
    }


def _compact_item_row(index: int, item) -> dict[str, float | int | bool]:
    return {
        "item": index,
        "usage_rate": item.usage_rate,
        "hits_per_month": item.hits_per_month,
        "item_cost": item.item_cost,
        "lead_time": item.lead_time,
        "op": item.op,
        "lp": item.lp,
        "eoq": item.eoq,
        "oq": item.oq,
        "pna": item.pna,
        "on_hand": item.on_hand,
        "on_order": item_on_order(item),
        "backorder": item.backorder,
        "soq": item.soq,
        "standard_pack": item.standard_pack,
        "safety_allowance": item.safety_allowance,
        "cp": item.cp,
        "surplus_line": item.surplus_line,
        "stockout_today": item.stockout_today,
    }


def _render_dashboard_snapshot(
    state: SimulationState,
    current_figure: Any,
    result: UserResult,
    options: PerformanceOptions,
) -> Any:
    if options.render_profile == "payload-only":
        payload = _timed(result, "build_payload", lambda: compact_dashboard_snapshot(state))
        _serialize_payload(result, payload)
        return current_figure
    if options.render_profile == "split-panels":
        return _render_split_panels(state, current_figure, result, options.split_panel)
    return _render_full_server_snapshot(state, current_figure, result)


def _render_full_server_snapshot(
    state: SimulationState,
    current_figure: Any,
    result: UserResult,
) -> Any:
    figure = _timed(
        result,
        "build_figure",
        lambda: refresh_inventory_figure(state, "light", current_figure),
    )
    cards = _timed(result, "build_cards", lambda: _server_cards_payload(state))
    grid = _timed(result, "build_grid", lambda: build_inventory_table(state, "light"))
    exceptions = _timed(result, "build_exceptions", lambda: build_exception_center(state))
    payload = {
        "day": f"Day: {state.day}",
        "figure": figure,
        **cards,
        "grid": grid,
        "exceptions": exceptions,
    }
    _serialize_payload(result, payload)
    return figure.to_plotly_json() if hasattr(figure, "to_plotly_json") else current_figure


def _render_split_panels(
    state: SimulationState,
    current_figure: Any,
    result: UserResult,
    split_panel: str,
) -> Any:
    next_figure = current_figure
    if split_panel in {"all", "figure"}:
        figure = _timed(
            result,
            "build_figure",
            lambda: refresh_inventory_figure(state, "light", current_figure),
        )
        _serialize_payload(result, {"figure": figure})
        next_figure = (
            figure.to_plotly_json()
            if hasattr(figure, "to_plotly_json")
            else current_figure
        )
    if split_panel in {"all", "cards"}:
        cards = _timed(result, "build_cards", lambda: _server_cards_payload(state))
        _serialize_payload(result, cards)
    if split_panel in {"all", "grid"}:
        grid = _timed(result, "build_grid", lambda: build_inventory_table(state, "light"))
        _serialize_payload(result, {"grid": grid})
    if split_panel in {"all", "exceptions"}:
        exceptions = _timed(result, "build_exceptions", lambda: build_exception_center(state))
        _serialize_payload(result, {"exceptions": exceptions})
    return next_figure


def _server_cards_payload(state: SimulationState) -> dict[str, Any]:
    return {
        "service": service_card_children(state),
        "costs": costs_card_children(state),
        "sales": sales_card_children(state),
        "kpi": build_kpi_strip(state),
    }


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
    stage_latency = {
        stage: _latency_summary(
            [
                latency
                for result in result_list
                for latency in ((result.stage_seconds or {}).get(stage) or [])
            ]
        )
        for stage in STAGE_NAMES
    }
    payload_bytes = [
        size
        for result in result_list
        for size in (result.payload_bytes or [])
    ]
    error_messages = tuple(
        message for result in result_list for message in (result.error_messages or [])
    )
    return PerformanceResult(
        options=options,
        wall_seconds=wall_seconds,
        total_ticks=sum(result.ticks for result in result_list),
        total_renders=sum(result.renders for result in result_list),
        total_errors=sum(result.errors for result in result_list),
        total_conflicts=sum(result.conflicts for result in result_list),
        total_behind_schedule=sum(result.behind_schedule for result in result_list),
        latency=_latency_summary(latencies),
        stage_latency=stage_latency,
        payload=_payload_summary(payload_bytes),
        peak_allocated_mb=peak_allocated / (1024 * 1024),
        session_dir=session_dir,
        error_messages=error_messages[:10],
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


def _payload_summary(payloads: list[int]) -> PayloadSummary:
    if not payloads:
        return PayloadSummary(0, 0, 0.0, 0, 0)
    ordered = sorted(payloads)
    return PayloadSummary(
        count=len(ordered),
        total_bytes=sum(ordered),
        mean_bytes=statistics.fmean(ordered),
        p95_bytes=ordered[_percentile_index(ordered, 95)],
        max_bytes=max(ordered),
    )


def _percentile_ms(ordered_seconds: list[float], percentile: int) -> float:
    return ordered_seconds[_percentile_index(ordered_seconds, percentile)] * 1000.0


def _percentile_index(ordered_values: list[Any], percentile: int) -> int:
    raw_index = math.ceil(percentile / 100 * len(ordered_values)) - 1
    return max(0, min(len(ordered_values) - 1, raw_index))


def _build_options(args: argparse.Namespace) -> PerformanceOptions:
    users = max(1, int(args.users))
    workers = max(1, min(int(args.workers or users), users))
    render_profile = "tick-only" if args.no_render else args.render_profile
    return PerformanceOptions(
        scenario=args.scenario,
        users=users,
        ticks=max(1, int(args.ticks)),
        duration_seconds=None if args.duration is None else max(0.1, float(args.duration)),
        tick_rate=max(0.0, float(args.tick_rate)),
        throttle=not args.no_throttle,
        workers=workers,
        render_profile=render_profile,
        render_every=max(1, int(args.render_every)),
        split_panel=args.split_panel,
        auto_po=not args.no_auto_po,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an in-process IMSim load test that exercises simulation ticks, "
            "session persistence, and dashboard render profiles."
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
        "--render-profile",
        choices=RENDER_PROFILES,
        default="full-server",
        help="Dashboard render workload to model.",
    )
    parser.add_argument(
        "--split-panel",
        choices=SPLIT_PANELS,
        default="all",
        help="Panel to isolate when --render-profile split-panels is used.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Alias for --render-profile tick-only.",
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
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write machine-readable run results to this JSON file.",
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

    if args.json_output:
        _write_json_result(args.json_output, result)
    print(format_result(result))


def _run_with_config(
    config: IMSimConfig,
    options: PerformanceOptions,
    session_dir: Path | None,
) -> PerformanceResult:
    repository = create_session_repository(config)
    return run_performance_test(repository, options, session_dir=session_dir)


def _write_json_result(path: Path, result: PerformanceResult) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result_to_dict(result), indent=2, cls=PlotlyJSONEncoder),
        encoding="utf-8",
    )


def result_to_dict(result: PerformanceResult) -> dict[str, Any]:
    return {
        "options": asdict(result.options),
        "wall_seconds": result.wall_seconds,
        "target_ticks_per_second": result.target_ticks_per_second,
        "target_90_ticks_per_second": result.target_90_ticks_per_second,
        "meets_capacity_target": result.meets_capacity_target,
        "ticks_per_second": result.ticks_per_second,
        "total_ticks": result.total_ticks,
        "total_renders": result.total_renders,
        "total_errors": result.total_errors,
        "total_conflicts": result.total_conflicts,
        "total_behind_schedule": result.total_behind_schedule,
        "latency": asdict(result.latency),
        "stages": {
            stage: asdict(summary)
            for stage, summary in result.stage_latency.items()
            if summary.count > 0
        },
        "payload": asdict(result.payload),
        "peak_allocated_mb": result.peak_allocated_mb,
        "session_dir": None if result.session_dir is None else str(result.session_dir),
        "error_messages": list(result.error_messages),
    }


def format_result(result: PerformanceResult) -> str:
    latency = result.latency
    target = result.target_ticks_per_second
    target_line = (
        "Target throughput: unthrottled"
        if target is None
        else f"Target throughput: {target:.2f} ticks/sec"
    )
    capacity = result.meets_capacity_target
    capacity_line = (
        "Capacity target: n/a"
        if capacity is None
        else (
            "Capacity target: PASS"
            if capacity
            else f"Capacity target: FAIL (needs {result.target_90_ticks_per_second:.2f} ticks/sec)"
        )
    )
    session_line = (
        f"Session store: {result.session_dir}"
        if result.session_dir is not None
        else "Session store: configured database repository"
    )
    lines = [
        "IMSim performance run",
        f"Scenario: {result.options.scenario}",
        f"Users: {result.options.users} virtual users, {result.options.workers} workers",
        (
            f"Tick rate: {result.options.tick_rate:.2f}/user/sec, "
            f"throttle={result.options.throttle}"
        ),
        (
            f"Render profile: {result.options.render_profile} "
            f"every {result.options.render_every} tick(s)"
        ),
        f"Split panel: {result.options.split_panel}",
        f"Auto purchasing: {result.options.auto_po}",
        session_line,
        target_line,
        capacity_line,
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
        _format_stage_line(result),
        _format_payload_line(result.payload),
        f"Behind-schedule ticks: {result.total_behind_schedule}",
        f"Save conflicts: {result.total_conflicts}",
        f"Errors: {result.total_errors}",
        f"Peak traced allocations: {result.peak_allocated_mb:.2f} MiB",
    ]
    if result.error_messages:
        lines.append("Sample errors: " + " | ".join(result.error_messages[:3]))
    return "\n".join(lines)


def _format_stage_line(result: PerformanceResult) -> str:
    parts = []
    labels = {
        "load_state": "load",
        "tick": "tick",
        "save_state": "save",
        "build_payload": "payload",
        "build_figure": "figure",
        "build_cards": "cards",
        "build_grid": "grid",
        "build_exceptions": "exceptions",
        "json_serialize": "json",
    }
    for stage in STAGE_NAMES:
        summary = result.stage_latency.get(stage)
        if summary is not None and summary.count:
            parts.append(f"{labels[stage]} p95 {summary.p95_ms:.2f}ms")
    return "Stage timings: " + (", ".join(parts) if parts else "n/a")


def _format_payload_line(payload: PayloadSummary) -> str:
    if payload.count <= 0:
        return "Payload bytes: n/a"
    mean_kib = payload.mean_bytes / 1024.0
    p95_kib = payload.p95_bytes / 1024.0
    total_mib = payload.total_bytes / (1024.0 * 1024.0)
    return (
        f"Payload bytes: mean {mean_kib:.2f} KiB, p95 {p95_kib:.2f} KiB, "
        f"total {total_mib:.2f} MiB"
    )


if __name__ == "__main__":
    main()
