from __future__ import annotations

from dataclasses import asdict, dataclass

from ..models import GlobalSettings, SimulationState, TrainingProfile
from .planning import create_inventory_item, format_money


@dataclass(frozen=True, slots=True)
class ScenarioItem:
    usage_rate: float
    lead_time: float
    item_cost: float
    initial_pna: float
    safety_allowance_pct: float
    standard_pack: float
    hits_per_month: float


@dataclass(frozen=True, slots=True)
class LevelDefinition:
    index: int
    level_id: str
    title: str
    summary: str
    formula: str
    tutorial_steps: tuple[str, ...]
    locked_features: tuple[str, ...]
    demand_mode: str
    day_window: int
    scenario: tuple[ScenarioItem, ...]
    visible_panels: frozenset[str]
    visible_columns: tuple[str, ...]
    allowed_actions: frozenset[str]
    win_conditions: dict[str, float | int | bool]
    global_settings: GlobalSettings


@dataclass(frozen=True, slots=True)
class LessonEvaluation:
    completed: bool
    passed: bool | None
    title: str
    message: str
    metric_rows: tuple[str, ...]


FULL_SIMULATOR_PANELS = frozenset(
    {
        "kpi",
        "graph",
        "service",
        "costs",
        "sales",
        "inventory",
        "exceptions",
        "actions",
        "policy",
        "session",
        "imports",
        "advanced_sandbox",
    }
)

FULL_SIMULATOR_COLUMNS = (
    "item",
    "usage_rate",
    "lead_time",
    "op",
    "lp",
    "oq",
    "pna",
    "on_hand",
    "on_order",
    "backorder",
    "soq",
)


def _settings(
    *,
    r_cycle: int = 14,
    r_cost: float = 8.0,
    k_cost: float = 0.18,
    stockout_penalty: float = 5.0,
    expedite_rate: float = 0.03,
    gm: float = 0.15,
    realization: float = 1.0,
    asq_enabled: bool = False,
) -> GlobalSettings:
    settings = GlobalSettings(
        r_cycle=r_cycle,
        r_cost=r_cost,
        k_cost=k_cost,
        stockout_penalty=stockout_penalty,
        expedite_rate=expedite_rate,
        gm=gm,
        realization=realization,
        auto_po_enabled=False,
    )
    settings.asq.enabled = asq_enabled
    return settings


LESSON_DEFINITIONS: tuple[LevelDefinition, ...] = (
    LevelDefinition(
        index=1,
        level_id="level-1",
        title="On Hand and Usage",
        summary=(
            "Watch one item drain down day by day so you can see how usage "
            "pulls physical inventory toward zero."
        ),
        formula="Tomorrow's on hand = today's on hand - daily usage",
        tutorial_steps=(
            "On hand is the physical stock available to ship right now.",
            "Usage consumes that stock each day, even before you learn reordering controls.",
            (
                "Once on hand reaches zero, the next demand becomes backorder "
                "because nothing is left to ship."
            ),
        ),
        locked_features=(
            (
                "Reordering stays locked in this lesson "
                "so the focus stays on the inventory drain curve."
            ),
            "Fill rate, order points, costs, and advanced controls unlock later.",
        ),
        demand_mode="deterministic",
        day_window=20,
        scenario=(
            ScenarioItem(
                usage_rate=60,
                lead_time=15,
                item_cost=35,
                initial_pna=34,
                safety_allowance_pct=0,
                standard_pack=1,
                hits_per_month=30,
            ),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session"}),
        visible_columns=("item", "on_hand", "daily_usage", "backorder"),
        allowed_actions=frozenset(),
        win_conditions={"on_hand_zero_close": True, "backorder_min": 2.0},
        global_settings=_settings(),
    ),
    LevelDefinition(
        index=2,
        level_id="level-2",
        title="Basic Ordering and PNA",
        summary=(
            "Manage one item with a guided reorder button and learn how "
            "on hand, on order, and PNA move together."
        ),
        formula="PNA = On Hand - Reserved - Committed - Backordered + On Order + Received",
        tutorial_steps=(
            "PNA shows your full inventory position, not just what is sitting on the shelf.",
            (
                "When you place an order, on order goes up immediately "
                "even before the receipt arrives."
            ),
            "Use the guided reorder when the item drifts near its order point.",
        ),
        locked_features=(
            "Custom quantities stay locked until the ordering-controls lesson.",
            "Safety stock, costs, and ASQ stay hidden so the focus stays on basic replenishment.",
        ),
        demand_mode="deterministic",
        day_window=18,
        scenario=(
            ScenarioItem(
                usage_rate=60,
                lead_time=10,
                item_cost=28,
                initial_pna=24,
                safety_allowance_pct=0,
                standard_pack=5,
                hits_per_month=30,
            ),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions"}),
        visible_columns=(
            "item",
            "daily_usage",
            "on_hand",
            "on_order",
            "backorder",
            "pna",
            "op",
            "soq",
        ),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={"guided_order_min": 1, "close_at_or_above_op": True},
        global_settings=_settings(),
    ),
    LevelDefinition(
        index=3,
        level_id="level-3",
        title="Simple Order Point",
        summary=(
            "Manage a small group of items with a guided reorder "
            "button and a clean order-point signal."
        ),
        formula="Order when on hand is near OP",
        tutorial_steps=(
            "Usage and lead time combine into a reorder point for each item.",
            "In this perfect-world lesson, reordering early enough is usually enough.",
            "Fill rate now matters because missed demand becomes lost service.",
        ),
        locked_features=(
            "Custom quantities stay locked until the ordering-controls lesson.",
            "Safety stock and PNA stay hidden until real-world OP.",
        ),
        demand_mode="deterministic",
        day_window=20,
        scenario=(
            ScenarioItem(
                usage_rate=60,
                lead_time=10,
                item_cost=28,
                initial_pna=24,
                safety_allowance_pct=0,
                standard_pack=1,
                hits_per_month=30,
            ),
            ScenarioItem(
                usage_rate=90,
                lead_time=12,
                item_cost=42,
                initial_pna=42,
                safety_allowance_pct=0,
                standard_pack=1,
                hits_per_month=30,
            ),
            ScenarioItem(
                usage_rate=30,
                lead_time=15,
                item_cost=20,
                initial_pna=20,
                safety_allowance_pct=0,
                standard_pack=1,
                hits_per_month=30,
            ),
        ),
        visible_panels=frozenset({"service", "inventory", "session", "actions"}),
        visible_columns=("item", "on_hand", "usage_rate", "lead_time", "op", "days_to_op"),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={"fill_rate_min": 0.97, "close_at_or_above_op": True},
        global_settings=_settings(),
    ),
    LevelDefinition(
        index=4,
        level_id="level-4",
        title="Ordering Controls",
        summary=(
            "Introduce custom quantities, on-order inventory, and "
            "backorders so the player learns the controls."
        ),
        formula="On order arrives after lead time and protects future demand",
        tutorial_steps=(
            "On-order units are committed but not received yet.",
            "Backorders mean demand arrived faster than stock and receipts.",
            "This lesson requires at least one manual custom order, not just the guided button.",
        ),
        locked_features=(
            "Safety stock, PNA, and ASQ are still locked.",
            "The full signal map and cost stack arrive later.",
        ),
        demand_mode="deterministic",
        day_window=20,
        scenario=(
            ScenarioItem(
                usage_rate=60,
                lead_time=9,
                item_cost=30,
                initial_pna=18,
                safety_allowance_pct=0,
                standard_pack=1,
                hits_per_month=30,
            ),
            ScenarioItem(
                usage_rate=45,
                lead_time=12,
                item_cost=18,
                initial_pna=18,
                safety_allowance_pct=0,
                standard_pack=1,
                hits_per_month=30,
            ),
            ScenarioItem(
                usage_rate=75,
                lead_time=15,
                item_cost=55,
                initial_pna=28,
                safety_allowance_pct=0,
                standard_pack=1,
                hits_per_month=30,
            ),
            ScenarioItem(
                usage_rate=30,
                lead_time=8,
                item_cost=25,
                initial_pna=14,
                safety_allowance_pct=0,
                standard_pack=1,
                hits_per_month=30,
            ),
        ),
        visible_panels=frozenset({"service", "inventory", "session", "actions"}),
        visible_columns=(
            "item",
            "on_hand",
            "on_order",
            "backorder",
            "usage_rate",
            "lead_time",
            "op",
            "soq",
        ),
        allowed_actions=frozenset({"guided_po", "custom_order"}),
        win_conditions={
            "fill_rate_min": 0.96,
            "manual_custom_order_min": 1,
            "zero_backorder_close": True,
        },
        global_settings=_settings(),
    ),
    LevelDefinition(
        index=5,
        level_id="level-5",
        title="Real-World OP",
        summary=(
            "Bring in variability, safety stock, and PNA so the "
            "learner sees why simple OP breaks down."
        ),
        formula="PNA = on hand + on order - backorder",
        tutorial_steps=(
            "Safety stock protects service when demand is not perfectly smooth.",
            "Projected net available combines stock on hand with inbound supply and backorders.",
            "This is the point where inventory position matters more than physical stock alone.",
        ),
        locked_features=(
            "Review-cycle controls and ASQ stay locked one more lesson.",
            "Auto purchase orders remain disabled for training.",
        ),
        demand_mode="stochastic",
        day_window=25,
        scenario=(
            ScenarioItem(
                usage_rate=48,
                lead_time=21,
                item_cost=95,
                initial_pna=52,
                safety_allowance_pct=25,
                standard_pack=5,
                hits_per_month=10,
            ),
            ScenarioItem(
                usage_rate=72,
                lead_time=18,
                item_cost=34,
                initial_pna=58,
                safety_allowance_pct=20,
                standard_pack=2,
                hits_per_month=12,
            ),
            ScenarioItem(
                usage_rate=30,
                lead_time=14,
                item_cost=64,
                initial_pna=28,
                safety_allowance_pct=30,
                standard_pack=1,
                hits_per_month=8,
            ),
            ScenarioItem(
                usage_rate=54,
                lead_time=28,
                item_cost=22,
                initial_pna=60,
                safety_allowance_pct=15,
                standard_pack=4,
                hits_per_month=16,
            ),
        ),
        visible_panels=frozenset({"service", "inventory", "session", "actions"}),
        visible_columns=(
            "item",
            "on_hand",
            "on_order",
            "backorder",
            "usage_rate",
            "lead_time",
            "safety_allowance",
            "pna",
            "op",
        ),
        allowed_actions=frozenset({"guided_po", "custom_order"}),
        win_conditions={"fill_rate_min": 0.95, "avg_inventory_value_max": 10500.0},
        global_settings=_settings(stockout_penalty=6.5, gm=0.18),
    ),
    LevelDefinition(
        index=6,
        level_id="level-6",
        title="Review Cycle and PO Management",
        summary=(
            "Open the signal map, SOQ logic, and PO actions so the "
            "learner manages an actual replenishment flow."
        ),
        formula="LP = OP + review-cycle demand",
        tutorial_steps=(
            "LP and OQ shape when to order and how much to buy.",
            "Use the PO overview to inspect, expedite, or cancel inbound receipts.",
            "Margin still matters, so service cannot be bought at any inventory cost.",
        ),
        locked_features=(
            "ASQ and the final certification dashboard are still ahead.",
            "Auto purchase orders remain locked until simulator unlock.",
        ),
        demand_mode="stochastic",
        day_window=30,
        scenario=(
            ScenarioItem(
                usage_rate=48,
                lead_time=21,
                item_cost=95,
                initial_pna=40,
                safety_allowance_pct=35,
                standard_pack=5,
                hits_per_month=10,
            ),
            ScenarioItem(
                usage_rate=90,
                lead_time=45,
                item_cost=22,
                initial_pna=120,
                safety_allowance_pct=20,
                standard_pack=10,
                hits_per_month=18,
            ),
            ScenarioItem(
                usage_rate=16,
                lead_time=14,
                item_cost=145,
                initial_pna=25,
                safety_allowance_pct=50,
                standard_pack=1,
                hits_per_month=4,
            ),
            ScenarioItem(
                usage_rate=60,
                lead_time=18,
                item_cost=60,
                initial_pna=55,
                safety_allowance_pct=20,
                standard_pack=5,
                hits_per_month=12,
            ),
            ScenarioItem(
                usage_rate=72,
                lead_time=24,
                item_cost=42,
                initial_pna=76,
                safety_allowance_pct=25,
                standard_pack=4,
                hits_per_month=14,
            ),
            ScenarioItem(
                usage_rate=36,
                lead_time=16,
                item_cost=75,
                initial_pna=32,
                safety_allowance_pct=30,
                standard_pack=2,
                hits_per_month=8,
            ),
        ),
        visible_panels=frozenset(
            {"kpi", "graph", "service", "costs", "sales", "inventory", "session", "actions"}
        ),
        visible_columns=(
            "item",
            "usage_rate",
            "lead_time",
            "op",
            "lp",
            "oq",
            "pna",
            "on_hand",
            "on_order",
            "backorder",
            "soq",
        ),
        allowed_actions=frozenset(
            {"guided_po", "custom_order", "po_overview", "expedite_receipt", "cancel_receipt"}
        ),
        win_conditions={"fill_rate_min": 0.96, "after_overhead_min": 0.0},
        global_settings=_settings(
            r_cycle=14, r_cost=8.0, k_cost=0.18, stockout_penalty=5.5, expedite_rate=0.03, gm=0.18
        ),
    ),
    LevelDefinition(
        index=7,
        level_id="level-7",
        title="Certification",
        summary=(
            "Run the full dashboard and prove you can balance service, "
            "cost, and replenishment decisions."
        ),
        formula="Use the full IM dashboard to hit service and margin together",
        tutorial_steps=(
            "This is the full training dashboard, including ASQ and the exception center.",
            "The simulator stays locked until this lesson is passed.",
            "Auto purchase orders are still disabled so the certification remains hands-on.",
        ),
        locked_features=(
            "Imports remain locked in certification so everyone solves the same scenario.",
            "Auto purchase orders unlock only after passing this lesson.",
        ),
        demand_mode="stochastic",
        day_window=30,
        scenario=(
            ScenarioItem(
                usage_rate=48,
                lead_time=21,
                item_cost=95,
                initial_pna=40,
                safety_allowance_pct=35,
                standard_pack=5,
                hits_per_month=10,
            ),
            ScenarioItem(
                usage_rate=90,
                lead_time=45,
                item_cost=22,
                initial_pna=120,
                safety_allowance_pct=20,
                standard_pack=10,
                hits_per_month=18,
            ),
            ScenarioItem(
                usage_rate=16,
                lead_time=14,
                item_cost=145,
                initial_pna=25,
                safety_allowance_pct=50,
                standard_pack=1,
                hits_per_month=4,
            ),
            ScenarioItem(
                usage_rate=60,
                lead_time=18,
                item_cost=60,
                initial_pna=55,
                safety_allowance_pct=20,
                standard_pack=5,
                hits_per_month=12,
            ),
            ScenarioItem(
                usage_rate=72,
                lead_time=24,
                item_cost=42,
                initial_pna=76,
                safety_allowance_pct=25,
                standard_pack=4,
                hits_per_month=14,
            ),
            ScenarioItem(
                usage_rate=36,
                lead_time=16,
                item_cost=75,
                initial_pna=32,
                safety_allowance_pct=30,
                standard_pack=2,
                hits_per_month=8,
            ),
        ),
        visible_panels=FULL_SIMULATOR_PANELS - {"imports", "advanced_sandbox"},
        visible_columns=FULL_SIMULATOR_COLUMNS,
        allowed_actions=frozenset(
            {
                "guided_po",
                "custom_order",
                "po_overview",
                "expedite_receipt",
                "cancel_receipt",
                "update_parameters",
                "apply_asq",
            }
        ),
        win_conditions={
            "fill_rate_min": 0.97,
            "after_overhead_min": 0.0,
            "zero_backorder_close": True,
        },
        global_settings=_settings(
            r_cycle=14,
            r_cost=8.0,
            k_cost=0.18,
            stockout_penalty=5.0,
            expedite_rate=0.03,
            gm=0.18,
            asq_enabled=True,
        ),
    ),
)

LEVELS_BY_ID = {level.level_id: level for level in LESSON_DEFINITIONS}


def clone_training_profile(profile: TrainingProfile) -> TrainingProfile:
    return TrainingProfile.from_dict(asdict(profile))


def academy_levels() -> tuple[LevelDefinition, ...]:
    return LESSON_DEFINITIONS


def academy_level(level_id: str | None) -> LevelDefinition | None:
    if not level_id:
        return None
    return LEVELS_BY_ID.get(level_id)


def active_level(state: SimulationState) -> LevelDefinition | None:
    if state.training.current_view != "lesson":
        return None
    return academy_level(state.training.active_level_id)


def academy_level_status(profile: TrainingProfile, level: LevelDefinition) -> str:
    if level.level_id in profile.completed_levels:
        return "completed"
    if level.index <= profile.highest_unlocked_level:
        return "unlocked"
    return "locked"


def visible_panels(state: SimulationState) -> frozenset[str]:
    if state.training.current_view == "simulator":
        return FULL_SIMULATOR_PANELS
    level = active_level(state)
    if level is None:
        return frozenset()
    return level.visible_panels


def visible_columns(state: SimulationState) -> tuple[str, ...]:
    if state.training.current_view == "simulator":
        return FULL_SIMULATOR_COLUMNS
    level = active_level(state)
    if level is None:
        return ()
    return level.visible_columns


def simulator_view_allowed(profile: TrainingProfile) -> bool:
    return profile.simulator_unlocked


def is_action_allowed(state: SimulationState, action: str) -> bool:
    if state.training.current_view == "main_menu":
        return False
    if state.training.current_view == "simulator":
        if action == "auto_po":
            return state.training.auto_po_reward_unlocked
        return True
    level = active_level(state)
    if level is None:
        return False
    if action == "auto_po":
        return False
    return action in level.allowed_actions


def lesson_elapsed_days(state: SimulationState) -> int:
    return max(0, int(state.day) - 1)


def lesson_days_remaining(state: SimulationState) -> int:
    level = active_level(state)
    if level is None:
        return 0
    return max(0, int(level.day_window) - lesson_elapsed_days(state))


def demand_mode(state: SimulationState) -> str:
    level = active_level(state)
    if level is None:
        return "stochastic"
    return level.demand_mode


def fill_rate(state: SimulationState) -> float | None:
    totals = state.service_totals
    if totals.orders <= 0:
        return None
    return (totals.orders - totals.orders_stockout) / totals.orders


def after_overhead_pct(state: SimulationState) -> float | None:
    if state.sales.revenue <= 0:
        return None
    gross = state.sales.revenue - state.sales.cogs
    after_overhead = gross - state.costs.total
    return after_overhead / state.sales.revenue


def average_inventory_value(state: SimulationState) -> float:
    elapsed = max(1, lesson_elapsed_days(state))
    return state.analytics.inv_value_daysum / elapsed


def _progress_metric_rows(state: SimulationState, level: LevelDefinition) -> tuple[str, ...]:
    rows: list[str] = []
    current_fill = fill_rate(state)
    if "fill_rate_min" in level.win_conditions:
        target = float(level.win_conditions["fill_rate_min"])
        current = "n/a" if current_fill is None else f"{current_fill * 100:.1f}%"
        rows.append(f"Fill rate: {current} / target {target * 100:.1f}%")
    if level.win_conditions.get("on_hand_zero_close"):
        rows.append(
            "On hand reached zero: "
            + ("yes" if sum(item.on_hand for item in state.items) <= 0 else "not yet")
        )
    if "backorder_min" in level.win_conditions:
        target = float(level.win_conditions["backorder_min"])
        backorder_total = sum(item.backorder for item in state.items)
        rows.append(f"Backorder observed: {backorder_total:.0f} / target {target:.0f}")
    if level.win_conditions.get("no_final_stockout"):
        rows.append(
            "Final-day stockout: "
            + (
                "clear"
                if not any(item.stockout_today for item in state.items)
                else "stockout recorded"
            )
        )
    if level.win_conditions.get("close_at_or_above_op"):
        safe_count = sum(1 for item in state.items if item.on_hand >= item.op)
        rows.append(f"Items closing at/above OP: {safe_count}/{len(state.items)}")
    if "guided_order_min" in level.win_conditions:
        needed = int(level.win_conditions["guided_order_min"])
        rows.append(f"Guided reorders used: {state.training.guided_orders_placed}/{needed}")
    if "manual_custom_order_min" in level.win_conditions:
        needed = int(level.win_conditions["manual_custom_order_min"])
        rows.append(f"Manual custom orders used: {state.training.custom_orders_placed}/{needed}")
    if level.win_conditions.get("zero_backorder_close"):
        backorder_total = sum(item.backorder for item in state.items)
        rows.append(f"Backorder at close: {backorder_total:.0f}")
    if "avg_inventory_value_max" in level.win_conditions:
        ceiling = float(level.win_conditions["avg_inventory_value_max"])
        rows.append(
            f"Avg inventory value: {format_money(average_inventory_value(state))} "
            f"/ cap {format_money(ceiling)}"
        )
    if "after_overhead_min" in level.win_conditions:
        current_after = after_overhead_pct(state)
        current = "n/a" if current_after is None else f"{current_after * 100:.1f}%"
        rows.append(f"After-overhead GM: {current} / target 0.0%")
    return tuple(rows)


def evaluate_active_lesson(state: SimulationState) -> LessonEvaluation | None:
    level = active_level(state)
    if level is None:
        return None
    rows = _progress_metric_rows(state, level)
    if lesson_elapsed_days(state) < level.day_window:
        remaining = lesson_days_remaining(state)
        title = f"{level.title} in progress"
        message = f"{remaining} simulated day(s) remaining."
        return LessonEvaluation(False, None, title, message, rows)

    checks: list[bool] = []
    if "fill_rate_min" in level.win_conditions:
        current_fill = fill_rate(state) or 0.0
        checks.append(current_fill >= float(level.win_conditions["fill_rate_min"]))
    if level.win_conditions.get("on_hand_zero_close"):
        checks.append(sum(item.on_hand for item in state.items) <= 0)
    if "backorder_min" in level.win_conditions:
        checks.append(
            sum(item.backorder for item in state.items)
            >= float(level.win_conditions["backorder_min"])
        )
    if level.win_conditions.get("no_final_stockout"):
        checks.append(not any(item.stockout_today for item in state.items))
    if level.win_conditions.get("close_at_or_above_op"):
        checks.append(all(item.on_hand >= item.op for item in state.items))
    if "guided_order_min" in level.win_conditions:
        checks.append(
            state.training.guided_orders_placed >= int(level.win_conditions["guided_order_min"])
        )
    if "manual_custom_order_min" in level.win_conditions:
        checks.append(
            state.training.custom_orders_placed
            >= int(level.win_conditions["manual_custom_order_min"])
        )
    if level.win_conditions.get("zero_backorder_close"):
        checks.append(sum(item.backorder for item in state.items) <= 0)
    if "avg_inventory_value_max" in level.win_conditions:
        checks.append(
            average_inventory_value(state) <= float(level.win_conditions["avg_inventory_value_max"])
        )
    if "after_overhead_min" in level.win_conditions:
        checks.append(
            (after_overhead_pct(state) or -1.0) >= float(level.win_conditions["after_overhead_min"])
        )

    passed = all(checks)
    if passed:
        message = "Next content is now unlocked in the academy menu."
        if level.level_id == "level-1":
            message = (
                "On hand is gone and demand is now turning into backorder. "
                "Restart the lesson or return to the academy for level 2."
            )
        return LessonEvaluation(
            True,
            True,
            f"{level.title} complete",
            message,
            rows,
        )
    return LessonEvaluation(
        True,
        False,
        f"{level.title} failed",
        "Retry the lesson to improve the tracked objectives.",
        rows,
    )


def apply_lesson_evaluation(
    state: SimulationState, evaluation: LessonEvaluation
) -> LessonEvaluation:
    level = active_level(state)
    if level is None or not evaluation.completed:
        return evaluation
    profile = state.training
    profile.lesson_status = "passed" if evaluation.passed else "failed"
    profile.last_result_title = evaluation.title
    profile.last_result_message = evaluation.message
    state.is_initialized = False
    if not evaluation.passed:
        return evaluation
    if level.level_id not in profile.completed_levels:
        profile.completed_levels.append(level.level_id)
    profile.completed_levels = sorted(
        set(profile.completed_levels),
        key=lambda level_id: LEVELS_BY_ID[level_id].index,
    )
    profile.highest_unlocked_level = min(
        len(LESSON_DEFINITIONS), max(profile.highest_unlocked_level, level.index + 1)
    )
    if level.index == len(LESSON_DEFINITIONS):
        profile.simulator_unlocked = True
        profile.auto_po_reward_unlocked = True
    return evaluation


def record_guided_order(state: SimulationState) -> None:
    state.training.guided_orders_placed += 1


def record_custom_order(state: SimulationState) -> None:
    state.training.custom_orders_placed += 1


def _state_for_scenario(
    *,
    profile: TrainingProfile,
    scenario: tuple[ScenarioItem, ...],
    settings: GlobalSettings,
) -> SimulationState:
    state = SimulationState(
        global_settings=GlobalSettings.from_dict(asdict(settings)),
        items=[],
        day=1,
        is_initialized=False,
        training=profile,
    )
    for item in scenario:
        state.items.append(
            create_inventory_item(
                usage_rate=item.usage_rate,
                lead_time=item.lead_time,
                item_cost=item.item_cost,
                pna=item.initial_pna,
                safety_allowance=item.safety_allowance_pct / 100.0,
                standard_pack=item.standard_pack,
                global_settings=state.global_settings,
                hits_per_month=item.hits_per_month,
            )
        )
    state.record_history()
    return state


def build_level_state(level_id: str, profile: TrainingProfile | None = None) -> SimulationState:
    level = LEVELS_BY_ID[level_id]
    progress = clone_training_profile(profile or TrainingProfile())
    progress.current_view = "lesson"
    progress.active_level_id = level_id
    progress.lesson_status = "ready"
    progress.lesson_intro_dismissed = False
    progress.last_result_title = ""
    progress.last_result_message = ""
    progress.guided_orders_placed = 0
    progress.custom_orders_placed = 0
    return _state_for_scenario(
        profile=progress, scenario=level.scenario, settings=level.global_settings
    )


def build_simulator_state(profile: TrainingProfile | None = None) -> SimulationState:
    progress = clone_training_profile(profile or TrainingProfile())
    progress.current_view = "simulator"
    progress.active_level_id = None
    progress.lesson_status = "idle"
    progress.lesson_intro_dismissed = False
    progress.last_result_title = ""
    progress.last_result_message = ""
    progress.guided_orders_placed = 0
    progress.custom_orders_placed = 0
    certification = LEVELS_BY_ID["level-7"]
    return _state_for_scenario(
        profile=progress,
        scenario=certification.scenario,
        settings=certification.global_settings,
    )


def reset_progress_state() -> SimulationState:
    return SimulationState()
