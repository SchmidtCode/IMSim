from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from ..models import GlobalSettings, SimulationState, TrainingProfile
from .planning import create_inventory_item, effective_review_cycle, format_money


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
    layout_variant: str
    teaching_goal: str = ""
    concept_tags: tuple[str, ...] = ()
    advanced_note: str = ""
    csd_mapping_note: str = ""
    success_hint: str = ""


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
    day_basis: int = 30,
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
        day_basis=day_basis,
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


def _item(
    usage_rate: float,
    lead_time: float,
    item_cost: float,
    initial_pna: float,
    *,
    safety_allowance_pct: float = 0,
    standard_pack: float = 1,
    hits_per_month: float = 12,
) -> ScenarioItem:
    return ScenarioItem(
        usage_rate=usage_rate,
        lead_time=lead_time,
        item_cost=item_cost,
        initial_pna=initial_pna,
        safety_allowance_pct=safety_allowance_pct,
        standard_pack=standard_pack,
        hits_per_month=hits_per_month,
    )


def _advanced_scenario() -> tuple[ScenarioItem, ...]:
    return (
        _item(48, 21, 95, 40, safety_allowance_pct=35, standard_pack=5, hits_per_month=10),
        _item(90, 45, 22, 120, safety_allowance_pct=20, standard_pack=10, hits_per_month=18),
        _item(16, 14, 145, 25, safety_allowance_pct=50, standard_pack=1, hits_per_month=4),
        _item(60, 18, 60, 55, safety_allowance_pct=20, standard_pack=5, hits_per_month=12),
        _item(72, 24, 42, 76, safety_allowance_pct=25, standard_pack=4, hits_per_month=14),
        _item(36, 16, 75, 32, safety_allowance_pct=30, standard_pack=2, hits_per_month=8),
    )


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
            "Once on hand reaches zero, the next demand becomes backorder.",
        ),
        locked_features=(
            "Reordering stays locked so the focus stays on the inventory drain curve.",
            "Fill rate, order points, costs, and advanced controls unlock later.",
        ),
        demand_mode="deterministic",
        day_window=19,
        scenario=(_item(60, 15, 35, 34, hits_per_month=30),),
        visible_panels=frozenset({"graph", "service", "inventory", "session"}),
        visible_columns=("item", "on_hand", "daily_usage", "backorder"),
        allowed_actions=frozenset(),
        win_conditions={"on_hand_zero_close": True, "backorder_min": 2.0},
        global_settings=_settings(),
        layout_variant="intro_trend",
        teaching_goal="See inventory leave the shelf before any planning math appears.",
        concept_tags=("on hand", "usage"),
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
            "When you place an order, on order goes up before the receipt arrives.",
            "Wait for PNA to fall below OP, then use the guided reorder.",
        ),
        locked_features=(
            "Custom quantities stay locked until later ordering lessons.",
            "Safety stock, costs, and ASQ stay hidden so the focus stays on basic replenishment.",
        ),
        demand_mode="deterministic",
        day_window=24,
        scenario=(_item(60, 10, 28, 24, standard_pack=5, hits_per_month=30),),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions"}),
        visible_columns=("item", "on_hand", "on_order", "backorder", "pna", "op"),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={
            "guided_order_below_op_min": 1,
            "close_pna_at_or_above_op": True,
        },
        global_settings=_settings(),
        layout_variant="intro_pna",
        teaching_goal="Connect the buying action to inventory position.",
        concept_tags=("PNA", "on order"),
    ),
    LevelDefinition(
        index=3,
        level_id="level-3",
        title="Customer Promise: Fill Rate",
        summary="Use a guided reorder and watch how complete lines protect fill rate.",
        formula="Fill rate = complete stock lines / total stock lines",
        tutorial_steps=(
            "Fill rate is a customer-service measurement, not an inventory quantity.",
            "A partially filled stock line counts as missed service in this model.",
            "Use a guided reorder to protect upcoming customer lines before stock runs out.",
            "A guided reorder creates inbound supply first; on-hand inventory only increases "
            "after the simulated lead time passes and the order is received.",
        ),
        locked_features=(
            "Custom quantities remain locked until later ordering lessons.",
            "Order point math waits until usage and lead time are introduced.",
        ),
        demand_mode="deterministic",
        day_window=12,
        scenario=(_item(60, 4, 24, 8, standard_pack=5, hits_per_month=30),),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions"}),
        visible_columns=("item", "on_hand", "on_order", "daily_usage", "pna", "backorder"),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={"fill_rate_min": 0.80, "guided_order_min": 1},
        global_settings=_settings(stockout_penalty=6.0),
        layout_variant="workspace_basic",
        teaching_goal="Make service failure concrete before introducing more formulas.",
        concept_tags=("fill rate", "backorder"),
    ),
    LevelDefinition(
        index=4,
        level_id="level-4",
        title="Hits vs Usage",
        summary="Compare items that sell the same total units but show up on orders differently.",
        formula="Hits count how often an item is requested; usage counts how many units sell.",
        tutorial_steps=(
            "Two items can have the same usage but very different customer popularity.",
            "High-hit items are the ones customers most expect to be available.",
            "Hits help humans decide where attention matters most.",
        ),
        locked_features=(
            "Ordering stays guided in the background.",
            "Lead time and order point unlock after the data basics are clear.",
        ),
        demand_mode="deterministic",
        day_window=5,
        scenario=(
            _item(60, 10, 18, 20, hits_per_month=2),
            _item(60, 10, 18, 20, hits_per_month=30),
            _item(15, 10, 80, 8, hits_per_month=1),
        ),
        visible_panels=frozenset({"service", "inventory", "session"}),
        visible_columns=("item", "usage_rate", "hits_per_month", "on_hand", "daily_usage"),
        allowed_actions=frozenset(),
        win_conditions={"fill_rate_min": 1.0},
        global_settings=_settings(),
        layout_variant="workspace_basic",
        teaching_goal="Separate popularity from quantity before item ranking appears.",
        concept_tags=("hits", "usage"),
    ),
    LevelDefinition(
        index=5,
        level_id="level-5",
        title="Usage Rate",
        summary="Turn monthly usage into a daily planning pace.",
        formula="Daily usage = monthly usage rate / day basis",
        tutorial_steps=(
            "Usage rate is the monthly demand pace used in the simulator.",
            "The simulator converts monthly usage into a daily drain using the current day basis.",
            "Later lessons multiply usage by lead-time and review-cycle windows.",
        ),
        locked_features=(
            "Lead time and replenishment triggers remain hidden.",
            "The lesson keeps demand deterministic so the slope is easy to read.",
        ),
        demand_mode="deterministic",
        day_window=6,
        scenario=(
            _item(30, 10, 16, 12, hits_per_month=8),
            _item(60, 10, 20, 20, hits_per_month=16),
            _item(90, 10, 28, 28, hits_per_month=24),
        ),
        visible_panels=frozenset({"service", "inventory", "session"}),
        visible_columns=("item", "usage_rate", "daily_usage", "on_hand", "backorder"),
        allowed_actions=frozenset(),
        win_conditions={"no_final_stockout": True},
        global_settings=_settings(),
        layout_variant="workspace_basic",
        teaching_goal="Build the daily-rate intuition needed for OP, LP, and EOQ.",
        concept_tags=("usage rate", "daily usage"),
    ),
    LevelDefinition(
        index=6,
        level_id="level-6",
        title="Lead Time",
        summary=(
            "Place a guided order and watch why replenishment has to start "
            "before shelves are empty."
        ),
        formula="Lead time is PO creation day to receipt day",
        tutorial_steps=(
            "A purchase order creates on-order inventory, not shelf inventory.",
            "Lead time is the delay between buying and receiving.",
            "The earlier lessons showed usage; this lesson shows the supplier clock.",
        ),
        locked_features=(
            "Custom quantities and PO actions remain locked.",
            "Order point calculations unlock after lead time is visible.",
        ),
        demand_mode="deterministic",
        day_window=6,
        scenario=(_item(45, 8, 30, 10, standard_pack=5, hits_per_month=18),),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions"}),
        visible_columns=("item", "on_hand", "on_order", "lead_time", "pna", "soq"),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={"guided_order_min": 1, "on_order_min": 1.0},
        global_settings=_settings(),
        layout_variant="intro_pna",
        teaching_goal="Show that buying is a timed decision, not an instant refill.",
        concept_tags=("lead time", "on order"),
    ),
    LevelDefinition(
        index=7,
        level_id="level-7",
        title="PNA as the Replenishment Signal",
        summary="Use PNA, not only on hand, to decide whether an item is still protected.",
        formula="PNA = On Hand - Reserved - Committed - Backordered + On Order + Received",
        tutorial_steps=(
            "PNA combines current stock, inbound supply, and unresolved demand.",
            "An item can look low on the shelf but still be covered by an inbound PO.",
            "The replenishment model watches PNA because it sees more of the story.",
        ),
        locked_features=(
            "Safety stock and line point remain hidden.",
            "Custom quantities stay locked until SOQ is introduced.",
        ),
        demand_mode="deterministic",
        day_window=8,
        scenario=(
            _item(60, 10, 28, 18, standard_pack=5, hits_per_month=24),
            _item(30, 8, 18, 10, standard_pack=2, hits_per_month=8),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions"}),
        visible_columns=("item", "on_hand", "on_order", "backorder", "pna", "op"),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={"guided_order_min": 1, "on_order_min": 1.0},
        global_settings=_settings(),
        layout_variant="intro_pna",
        teaching_goal="Make PNA the learner's default replenishment signal.",
        concept_tags=("PNA", "replenishment"),
        csd_mapping_note=(
            "CSD mapping: PNA is the replenishment signal buyers use because it includes "
            "more than shelf stock. This simulator shows the full business formula, while "
            "the app math simplifies Reserved, Committed, and Received to zero."
        ),
    ),
    LevelDefinition(
        index=8,
        level_id="level-8",
        title="Order Point in a Perfect World",
        summary="Manage a small group of items with deterministic demand and no safety stock.",
        formula="OP = monthly usage x lead-time days / day basis",
        tutorial_steps=(
            "Usage rate and lead time combine into a reorder point for each item.",
            "In this simplified lesson, demand is smooth and there is no safety stock.",
            "Use guided reorders before PNA falls too far below OP.",
        ),
        locked_features=(
            "Custom quantities stay locked until later ordering controls.",
            "Safety stock remains hidden until the real-world OP lesson.",
        ),
        demand_mode="deterministic",
        day_window=20,
        scenario=(
            _item(60, 10, 28, 24, hits_per_month=30),
            _item(90, 12, 42, 42, hits_per_month=30),
            _item(30, 15, 20, 20, hits_per_month=30),
        ),
        visible_panels=frozenset({"service", "inventory", "session", "actions"}),
        visible_columns=(
            "item",
            "on_hand",
            "on_order",
            "usage_rate",
            "lead_time",
            "op",
            "days_to_op",
        ),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={"fill_rate_min": 0.97, "close_at_or_above_op": True, "guided_order_min": 2},
        global_settings=_settings(),
        layout_variant="workspace_basic",
        teaching_goal="Answer the first inventory question: when should we buy?",
        concept_tags=("OP", "lead time"),
    ),
    LevelDefinition(
        index=9,
        level_id="level-9",
        title="Safety Stock",
        summary="Compare items with different safety allowances and see how the OP rises.",
        formula="Safety stock = monthly usage x lead-time days / day basis x safety allowance",
        tutorial_steps=(
            "Safety stock is a buffer for normal variation in usage or lead time.",
            "A higher safety allowance raises OP, which triggers earlier buying.",
            "The buffer protects service, but it also increases inventory investment.",
        ),
        locked_features=(
            "Review cycle and line point remain locked.",
            "Cost controls stay hidden until the tradeoff lessons.",
        ),
        demand_mode="deterministic",
        day_window=10,
        scenario=(
            _item(60, 15, 28, 42, safety_allowance_pct=0, hits_per_month=20),
            _item(60, 15, 28, 42, safety_allowance_pct=50, hits_per_month=20),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session"}),
        visible_columns=("item", "usage_rate", "lead_time", "safety_allowance", "op", "on_hand"),
        allowed_actions=frozenset(),
        win_conditions={"fill_rate_min": 1.0},
        global_settings=_settings(),
        layout_variant="workspace_signal",
        teaching_goal="Show why real-world OP is higher than perfect-world OP.",
        concept_tags=("safety stock", "safety allowance"),
    ),
    LevelDefinition(
        index=10,
        level_id="level-10",
        title="Real-World Order Point",
        summary="Bring in variability, safety stock, and PNA so simple OP gets more realistic.",
        formula="OP = lead-time demand + safety stock",
        tutorial_steps=(
            "Demand is no longer perfectly smooth.",
            "PNA combines shelf stock, inbound supply, and unresolved demand.",
            "Order before PNA falls through the real-world OP.",
        ),
        locked_features=(
            "Review-cycle controls and ASQ stay locked one more lesson.",
            "Auto purchase orders remain disabled for training.",
        ),
        demand_mode="stochastic",
        day_window=25,
        scenario=(
            _item(48, 21, 95, 52, safety_allowance_pct=25, standard_pack=5, hits_per_month=10),
            _item(72, 18, 34, 58, safety_allowance_pct=20, standard_pack=2, hits_per_month=12),
            _item(30, 14, 64, 28, safety_allowance_pct=30, standard_pack=1, hits_per_month=8),
            _item(54, 28, 22, 60, safety_allowance_pct=15, standard_pack=4, hits_per_month=16),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions"}),
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
        layout_variant="workspace_signal",
        teaching_goal="Use OP as a service-protection signal under uncertain demand.",
        concept_tags=("real-world OP", "PNA"),
    ),
    LevelDefinition(
        index=11,
        level_id="level-11",
        title="Long Tail and Item Ranking",
        summary="Use hits to see why popular items and long-tail items need different attention.",
        formula="Rank by hits so human attention follows customer demand",
        tutorial_steps=(
            "A few items usually create most customer requests.",
            "Long-tail items still matter, but rules can manage more of their routine work.",
            "Good inventory management uses the same model with different attention levels.",
        ),
        locked_features=(
            "Ranking is shown conceptually; automatic rank assignment is not part of this lesson.",
            "Review-cycle and cost tuning unlock in the next section.",
        ),
        demand_mode="deterministic",
        day_window=8,
        scenario=(
            _item(90, 15, 22, 80, safety_allowance_pct=20, hits_per_month=45),
            _item(48, 20, 95, 50, safety_allowance_pct=30, hits_per_month=10),
            _item(12, 12, 180, 20, safety_allowance_pct=40, hits_per_month=2),
            _item(6, 30, 45, 12, safety_allowance_pct=50, hits_per_month=0.5),
        ),
        visible_panels=frozenset({"service", "inventory", "session"}),
        visible_columns=("item", "hits_per_month", "usage_rate", "item_cost", "pna", "op"),
        allowed_actions=frozenset(),
        win_conditions={"no_final_stockout": True},
        global_settings=_settings(),
        layout_variant="workspace_basic",
        teaching_goal="Connect customer popularity to management-by-exception thinking.",
        concept_tags=("hits", "long tail"),
    ),
    LevelDefinition(
        index=12,
        level_id="level-12",
        title="Product Lines and Review Cycle",
        summary=(
            "Learn why a supplier/product line is reviewed as a group once one item "
            "triggers action."
        ),
        formula="Review cycle = planned days between P-line purchases",
        tutorial_steps=(
            "A product line is a group of items that can ride the same supplier PO.",
            "The review cycle describes how long we intend to wait before buying again.",
            "A buyer uses the line view to decide what else should go on today's PO.",
        ),
        locked_features=(
            "Line point math unlocks next.",
            "PO expedite/cancel controls remain locked until PO management.",
        ),
        demand_mode="stochastic",
        day_window=18,
        scenario=(
            _item(48, 21, 95, 40, safety_allowance_pct=25, standard_pack=5, hits_per_month=10),
            _item(72, 18, 34, 52, safety_allowance_pct=20, standard_pack=2, hits_per_month=12),
            _item(30, 14, 64, 28, safety_allowance_pct=30, standard_pack=1, hits_per_month=8),
            _item(54, 28, 22, 56, safety_allowance_pct=15, standard_pack=4, hits_per_month=16),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions"}),
        visible_columns=("item", "usage_rate", "lead_time", "pna", "op", "lp", "soq"),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={"guided_order_min": 1, "fill_rate_min": 0.92},
        global_settings=_settings(r_cycle=14, stockout_penalty=6.0, gm=0.18),
        layout_variant="workspace_signal",
        teaching_goal="Shift from isolated-item thinking to line-level replenishment.",
        concept_tags=("P-line", "review cycle"),
        csd_mapping_note=(
            "CSD mapping: Product lines group items that may be reviewed together for a "
            "supplier buy. Review cycle represents how often the line is expected to be "
            "reviewed or purchased."
        ),
    ),
    LevelDefinition(
        index=13,
        level_id="level-13",
        title="Line Point",
        summary="Use LP to decide which near-trigger items should ride along on today's PO.",
        formula="LP = OP + review-cycle demand",
        tutorial_steps=(
            "LP is an early warning level above OP.",
            "If PNA is at or below LP during a product-line buy, the item may need to be included.",
            "The guided order now follows SOQ recommendations across the line.",
        ),
        locked_features=(
            "EOQ and cost tuning are still simplified.",
            "PO overview controls unlock after the line-point lesson.",
        ),
        demand_mode="deterministic",
        day_window=40,
        scenario=(
            _item(60, 18, 40, 60, safety_allowance_pct=20, standard_pack=5, hits_per_month=18),
            _item(45, 15, 22, 42, safety_allowance_pct=20, standard_pack=5, hits_per_month=12),
            _item(30, 20, 75, 38, safety_allowance_pct=25, standard_pack=1, hits_per_month=7),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions"}),
        visible_columns=("item", "pna", "op", "lp", "days_to_op", "soq"),
        allowed_actions=frozenset({"guided_po"}),
        win_conditions={
            "guided_order_below_op_item_min": 1,
            "guided_order_below_lp_item_min": 3,
            "guided_order_below_lp_min": 2,
        },
        global_settings=_settings(r_cycle=14),
        layout_variant="workspace_signal",
        teaching_goal="Show that the trigger item is not the only item to buy.",
        concept_tags=("LP", "SOQ"),
        csd_mapping_note=(
            "CSD mapping: LP is the level where near-trigger items may be included during a "
            "product-line review. Some setups order when PNA is less than or equal to LP; "
            "others wait until PNA falls below LP."
        ),
    ),
    LevelDefinition(
        index=14,
        level_id="level-14",
        title="Carrying Cost vs Replenishment Cost",
        summary="Balance the cost of holding inventory with the cost of replenishing too often.",
        formula="Too much raises K-cost; too little raises R-cost",
        tutorial_steps=(
            "Carrying cost grows with inventory value and time.",
            "Replenishment cost is paid each time a PO line is created.",
            "The best quantity balances those two pressures instead of maximizing service alone.",
        ),
        locked_features=(
            "EOQ calculation appears in the next lesson.",
            "ASQ and imports remain locked until certification.",
        ),
        demand_mode="stochastic",
        day_window=24,
        scenario=_advanced_scenario(),
        visible_panels=frozenset(
            {"kpi", "graph", "service", "costs", "sales", "inventory", "session", "actions"}
        ),
        visible_columns=("item", "item_cost", "usage_rate", "pna", "op", "lp", "soq"),
        allowed_actions=frozenset({"guided_po", "custom_order"}),
        win_conditions={"fill_rate_min": 0.94, "after_overhead_min": -0.05},
        global_settings=_settings(
            r_cycle=14, r_cost=8.0, k_cost=0.18, stockout_penalty=5.5, gm=0.18
        ),
        layout_variant="workspace_advanced",
        teaching_goal="Introduce the profitability side of inventory decisions.",
        concept_tags=("K-cost", "R-cost"),
    ),
    LevelDefinition(
        index=15,
        level_id="level-15",
        title="EOQ and Order Quantity",
        summary="Use EOQ as the starting point for how much to buy.",
        formula="EOQ balances annual replenishment cost and carrying cost",
        tutorial_steps=(
            "EOQ is calculated per item because usage, item cost, K-cost, and R-cost differ.",
            "Order quantity starts from EOQ but is bounded by practical minimums and maximums.",
            (
                "Place at least one custom order to feel the difference between "
                "deciding and accepting."
            ),
        ),
        locked_features=(
            "Standard pack rounding and PO overview unlock next.",
            "Parameter tuning remains locked until the tradeoff bridge.",
        ),
        demand_mode="stochastic",
        day_window=24,
        scenario=_advanced_scenario(),
        visible_panels=frozenset(
            {"kpi", "graph", "service", "costs", "sales", "inventory", "session", "actions"}
        ),
        visible_columns=("item", "usage_rate", "item_cost", "eoq", "oq", "pna", "op", "lp"),
        allowed_actions=frozenset({"guided_po", "custom_order"}),
        win_conditions={"manual_custom_order_min": 1, "fill_rate_min": 0.92},
        global_settings=_settings(
            r_cycle=14, r_cost=8.0, k_cost=0.18, stockout_penalty=5.5, gm=0.18
        ),
        layout_variant="workspace_advanced",
        teaching_goal="Answer the second inventory question: how much should we buy?",
        concept_tags=("EOQ", "OQ"),
    ),
    LevelDefinition(
        index=16,
        level_id="level-16",
        title="Suggested Order Quantity and Standard Pack",
        summary="See how SOQ recovers below OP, adds OQ, and rounds to supplier packs.",
        formula="SOQ = OQ + shortage below OP, rounded to standard pack",
        tutorial_steps=(
            "If PNA is below OP, suggested order quantity (SOQ) first recovers the shortfall.",
            "Then it adds the order quantity needed for the next cycle.",
            "The final recommendation is rounded to the supplier standard pack.",
        ),
        locked_features=(
            "Policy tuning and ASQ are still ahead.",
            "Auto purchase orders remain locked until the simulator reward.",
        ),
        demand_mode="stochastic",
        day_window=28,
        scenario=_advanced_scenario(),
        visible_panels=frozenset(
            {"kpi", "graph", "service", "costs", "sales", "inventory", "session", "actions"}
        ),
        visible_columns=("item", "pna", "op", "lp", "oq", "standard_pack", "soq", "on_order"),
        allowed_actions=frozenset(
            {"guided_po", "custom_order", "po_overview", "expedite_receipt", "cancel_receipt"}
        ),
        win_conditions={
            "manual_custom_order_min": 1,
            "fill_rate_min": 0.94,
            "after_overhead_min": -0.05,
        },
        global_settings=_settings(
            r_cycle=14, r_cost=8.0, k_cost=0.18, stockout_penalty=5.5, expedite_rate=0.03, gm=0.18
        ),
        layout_variant="workspace_advanced",
        teaching_goal="Turn the model's buy/no-buy signal into a practical PO quantity.",
        concept_tags=("SOQ", "standard pack"),
        advanced_note=(
            "Simulator note: SOQ is simplified for training. In a real ERP environment, "
            "recommended order quantity can also depend on order method, product-line "
            "settings, standard pack or buying unit rounding, min/max logic, "
            "ROQ/use-order-quantity settings, and buyer overrides."
        ),
        csd_mapping_note=(
            "CSD mapping: PO RRAR quantity recommendations may vary by order method, "
            "Use ROQ/order quantity behavior, standard pack or buying unit rounding, "
            "and buyer changes."
        ),
    ),
    LevelDefinition(
        index=17,
        level_id="level-17",
        title="Exceptions: Critical Point and Surplus",
        summary=(
            "Identify urgent stockout risk and overstock before planning emergency bridge buys."
        ),
        formula=(
            "Critical point = monthly usage x lead-time days / day basis; surplus "
            "threshold = LP + OQ"
        ),
        tutorial_steps=(
            "Critical point means the item is in urgent replenishment territory.",
            "Surplus threshold marks the model's high-water point.",
            "Good inventory work means managing shortages and overstock without losing "
            "the whole system view.",
        ),
        locked_features=(
            "Certification opens ASQ and the exception center.",
            "Imports and auto purchase orders stay locked until the simulator is earned.",
        ),
        demand_mode="deterministic",
        day_window=1,
        scenario=(
            _item(90, 30, 18, 60, safety_allowance_pct=50, standard_pack=10, hits_per_month=24),
            _item(24, 15, 150, 180, safety_allowance_pct=25, standard_pack=1, hits_per_month=3),
            _item(54, 18, 42, 65, safety_allowance_pct=20, standard_pack=6, hits_per_month=12),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session"}),
        visible_columns=("item", "pna", "cp", "op", "lp", "oq", "surplus_line"),
        allowed_actions=frozenset(),
        win_conditions={"critical_item_min": 1, "surplus_item_min": 1},
        global_settings=_settings(r_cycle=14),
        layout_variant="workspace_signal",
        teaching_goal="Teach the two exception boundaries that frame urgent action and overbuying.",
        concept_tags=("critical point", "surplus"),
        csd_mapping_note=(
            "CSD mapping: Critical point is a priority/exception signal. Short-term surplus "
            "is generally inventory available above line point plus order quantity, with "
            "additional details depending on order method and report logic."
        ),
    ),
    LevelDefinition(
        index=18,
        level_id="level-18",
        title="Emergency PLine Bridge Buy",
        summary=(
            "Use a temporary review-cycle bridge to pull near-term demand into "
            "one emergency buy without changing normal setup."
        ),
        formula="Bridge buy = normal line buy recalculated with a temporary review cycle",
        tutorial_steps=(
            (
                "Use a bridge only when today's signal is urgent but the normal line "
                "buy is still too small to cover the near-term risk."
            ),
            (
                "The override widens the look-ahead window, so items close to line "
                "point can join this buy instead of waiting for the next cycle."
            ),
            (
                "Accept the bridge lines that solve the emergency; do not treat the "
                "larger buy as the new normal."
            ),
            (
                "After the PO is created, restore the normal review cycle so future "
                "recommendations return to the planned cadence."
            ),
        ),
        locked_features=(
            (
                "This lesson uses the simulator Review Cycle Override as the "
                "Demand Center one-time recalculation action."
            ),
            "Imports and ASQ remain locked until certification.",
        ),
        demand_mode="deterministic",
        day_window=1,
        scenario=(
            _item(90, 20, 90, 50, safety_allowance_pct=25, standard_pack=5, hits_per_month=30),
            _item(60, 15, 35, 50, safety_allowance_pct=20, standard_pack=5, hits_per_month=12),
            _item(30, 30, 120, 45, safety_allowance_pct=20, standard_pack=1, hits_per_month=5),
            _item(45, 10, 22, 15, safety_allowance_pct=30, standard_pack=4, hits_per_month=18),
        ),
        visible_panels=frozenset({"graph", "service", "inventory", "session", "actions", "policy"}),
        visible_columns=("item", "pna", "op", "lp", "usage_rate", "soq", "on_order", "backorder"),
        allowed_actions=frozenset(
            {"guided_po", "custom_order", "po_overview", "update_parameters"}
        ),
        win_conditions={
            "emergency_review_cycle_min": 11,
            "emergency_normal_review_cycle": 7,
            "emergency_bridge_order_min": 1,
        },
        global_settings=_settings(
            r_cycle=7, r_cost=1.5, k_cost=0.30, stockout_penalty=6.0, gm=0.18
        ),
        layout_variant="workspace_advanced",
        teaching_goal=(
            "Practice a controlled emergency buy without turning it into a permanent policy change."
        ),
        concept_tags=("review cycle", "bridge buy", "line point", "one-time only"),
        advanced_note=(
            "Theory: review cycle is the planned time until the next line review. "
            "Extending it temporarily increases expected demand coverage, which can "
            "pull close-to-trigger items into the current recommendation. The tradeoff "
            "is higher inventory investment, so the override should be visible, "
            "one-time, and cleared after the bridge order."
        ),
    ),
    LevelDefinition(
        index=19,
        level_id="level-19",
        title="Certification: Balanced Inventory Management",
        summary="Run the full dashboard and balance service, cost, and replenishment decisions.",
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
        scenario=_advanced_scenario(),
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
        layout_variant="workspace_certification",
        teaching_goal="Prove the learner can balance customer experience with profitability.",
        concept_tags=("certification", "balanced objectives"),
        csd_mapping_note=(
            "CSD mapping: This dashboard is a training workspace. It combines "
            "replenishment, service, cost, exception, and ordering signals into one "
            "exercise rather than copying any ERP screen."
        ),
    ),
)

LEVELS_BY_ID = {level.level_id: level for level in LESSON_DEFINITIONS}


def clone_training_profile(profile: TrainingProfile) -> TrainingProfile:
    return TrainingProfile.from_dict(asdict(profile))


def academy_levels() -> tuple[LevelDefinition, ...]:
    return LESSON_DEFINITIONS


def cheat_unlock_password() -> str:
    return os.environ.get("IMSIM_CHEAT_UNLOCK_PASSWORD", "spreadsheets rule")


def cheat_unlock_password_matches(password: str | None) -> bool:
    normalized = " ".join(str(password or "").strip().casefold().split())
    return normalized == " ".join(cheat_unlock_password().strip().casefold().split())


def unlock_all_academy_levels(profile: TrainingProfile) -> TrainingProfile:
    profile.highest_unlocked_level = len(LESSON_DEFINITIONS)
    profile.simulator_unlocked = True
    profile.auto_po_reward_unlocked = True
    profile.lesson_status = "idle"
    profile.last_result_title = "Academy unlocked"
    profile.last_result_message = (
        "All lessons, simulator mode, and the sandbox reward controls are now available."
    )
    return profile


def final_academy_level() -> LevelDefinition:
    return LESSON_DEFINITIONS[-1]


def academy_level(level_id: str | None) -> LevelDefinition | None:
    if not level_id:
        return None
    return LEVELS_BY_ID.get(level_id)


def active_level(state: SimulationState) -> LevelDefinition | None:
    if state.training.current_view != "lesson":
        return None
    return academy_level(state.training.active_level_id)


def active_layout_variant(state: SimulationState) -> str:
    if state.training.current_view == "simulator":
        return "simulator"
    level = active_level(state)
    return level.layout_variant if level is not None else "simulator"


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
    if level.win_conditions.get("close_pna_at_or_above_op"):
        safe_count = sum(1 for item in state.items if item.pna >= item.op)
        rows.append(f"Close with PNA at/above OP: {safe_count}/{len(state.items)}")
    if "guided_order_below_op_min" in level.win_conditions:
        needed = int(level.win_conditions["guided_order_below_op_min"])
        rows.append(
            "Reorder triggered after PNA fell below OP: "
            f"{state.training.guided_orders_below_op}/{needed}"
        )
    if "guided_order_min" in level.win_conditions:
        needed = int(level.win_conditions["guided_order_min"])
        rows.append(f"Guided reorders used: {state.training.guided_orders_placed}/{needed}")
    if "guided_order_below_lp_min" in level.win_conditions:
        needed_op_items = int(level.win_conditions.get("guided_order_below_op_item_min", 0))
        needed_items = int(level.win_conditions.get("guided_order_below_lp_item_min", 2))
        needed_orders = int(level.win_conditions["guided_order_below_lp_min"])
        rows.append(
            f"Guided reorders while {needed_op_items}+ item below OP and "
            f"{needed_items}+ total below LP: {state.training.guided_orders_below_lp}/"
            f"{needed_orders}"
        )
    if "on_order_min" in level.win_conditions:
        target = float(level.win_conditions["on_order_min"])
        on_order_total = sum(sum(receipt.qty for receipt in item.pipeline) for item in state.items)
        rows.append(f"On-order inventory: {on_order_total:.0f} / target {target:.0f}")
    if "manual_custom_order_min" in level.win_conditions:
        needed = int(level.win_conditions["manual_custom_order_min"])
        rows.append(f"Manual custom orders used: {state.training.custom_orders_placed}/{needed}")
    if "parameter_update_min" in level.win_conditions:
        needed = int(level.win_conditions["parameter_update_min"])
        rows.append(f"Parameter updates used: {state.training.parameter_updates_applied}/{needed}")
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
        target = float(level.win_conditions["after_overhead_min"])
        current = "n/a" if current_after is None else f"{current_after * 100:.1f}%"
        rows.append(f"After-overhead GM: {current} / target {target * 100:.1f}%")
    if "below_lp_min" in level.win_conditions:
        needed = int(level.win_conditions["below_lp_min"])
        current = sum(1 for item in state.items if item.pna <= item.lp)
        rows.append(f"Items at/below LP: {current}/{needed}")
    if "critical_item_min" in level.win_conditions:
        needed = int(level.win_conditions["critical_item_min"])
        current = sum(1 for item in state.items if item.pna <= item.cp)
        rows.append(f"Items at/below critical point: {current}/{needed}")
    if "surplus_item_min" in level.win_conditions:
        needed = int(level.win_conditions["surplus_item_min"])
        current = sum(1 for item in state.items if item.on_hand >= item.surplus_line)
        rows.append(f"Items above surplus threshold: {current}/{needed}")
    if "emergency_review_cycle_min" in level.win_conditions:
        emergency_cycle = int(level.win_conditions["emergency_review_cycle_min"])
        normal_cycle = int(level.win_conditions.get("emergency_normal_review_cycle", 21))
        override_used = state.training.emergency_review_cycle_applied or (
            effective_review_cycle(state.global_settings) >= emergency_cycle
        )
        rows.append(
            "Temporary review cycle override used: "
            + ("yes" if override_used else "not yet")
            + f" / target {emergency_cycle} days"
        )
        rows.append(
            "Bridge PO created with override: "
            f"{state.training.emergency_bridge_orders_placed}/"
            f"{int(level.win_conditions.get('emergency_bridge_order_min', 1))}"
        )
        rows.append(
            "Normal review cycle restored: "
            + ("yes" if state.training.emergency_review_cycle_restored else "not yet")
            + f" / target {normal_cycle} days"
        )
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
    if level.win_conditions.get("close_pna_at_or_above_op"):
        checks.append(all(item.pna >= item.op for item in state.items))
    if "guided_order_below_op_min" in level.win_conditions:
        checks.append(
            state.training.guided_orders_below_op
            >= int(level.win_conditions["guided_order_below_op_min"])
        )
    if "guided_order_min" in level.win_conditions:
        checks.append(
            state.training.guided_orders_placed >= int(level.win_conditions["guided_order_min"])
        )
    if "guided_order_below_lp_min" in level.win_conditions:
        checks.append(
            state.training.guided_orders_below_lp
            >= int(level.win_conditions["guided_order_below_lp_min"])
        )
    if "on_order_min" in level.win_conditions:
        on_order_total = sum(sum(receipt.qty for receipt in item.pipeline) for item in state.items)
        checks.append(on_order_total >= float(level.win_conditions["on_order_min"]))
    if "manual_custom_order_min" in level.win_conditions:
        checks.append(
            state.training.custom_orders_placed
            >= int(level.win_conditions["manual_custom_order_min"])
        )
    if "parameter_update_min" in level.win_conditions:
        checks.append(
            state.training.parameter_updates_applied
            >= int(level.win_conditions["parameter_update_min"])
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
    if "below_lp_min" in level.win_conditions:
        checks.append(
            sum(1 for item in state.items if item.pna <= item.lp)
            >= int(level.win_conditions["below_lp_min"])
        )
    if "critical_item_min" in level.win_conditions:
        checks.append(
            sum(1 for item in state.items if item.pna <= item.cp)
            >= int(level.win_conditions["critical_item_min"])
        )
    if "surplus_item_min" in level.win_conditions:
        checks.append(
            sum(1 for item in state.items if item.on_hand >= item.surplus_line)
            >= int(level.win_conditions["surplus_item_min"])
        )
    if "emergency_review_cycle_min" in level.win_conditions:
        checks.append(state.training.emergency_review_cycle_applied)
        checks.append(
            state.training.emergency_bridge_orders_placed
            >= int(level.win_conditions.get("emergency_bridge_order_min", 1))
        )
        checks.append(state.training.emergency_review_cycle_restored)

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


def _record_emergency_bridge_if_applicable(state: SimulationState) -> None:
    level = active_level(state)
    if level is None or "emergency_review_cycle_min" not in level.win_conditions:
        return
    emergency_cycle = int(level.win_conditions["emergency_review_cycle_min"])
    if effective_review_cycle(state.global_settings) >= emergency_cycle:
        state.training.emergency_review_cycle_applied = True
        state.training.emergency_review_cycle_restored = False
        state.training.emergency_bridge_orders_placed += 1


def record_guided_order(
    state: SimulationState,
    *,
    below_op: bool = False,
    below_op_count: int = 0,
    below_lp_count: int = 0,
) -> None:
    state.training.guided_orders_placed += 1
    if below_op:
        state.training.guided_orders_below_op += 1
    level = active_level(state)
    if level is not None and "guided_order_below_lp_min" in level.win_conditions:
        needed_op = int(level.win_conditions.get("guided_order_below_op_item_min", 0))
        needed_lp = int(level.win_conditions.get("guided_order_below_lp_item_min", 2))
        if below_op_count >= needed_op and below_lp_count >= needed_lp:
            state.training.guided_orders_below_lp += 1
    _record_emergency_bridge_if_applicable(state)


def record_custom_order(state: SimulationState) -> None:
    state.training.custom_orders_placed += 1
    _record_emergency_bridge_if_applicable(state)


def record_parameter_update(state: SimulationState) -> None:
    state.training.parameter_updates_applied += 1
    level = active_level(state)
    if level is None or "emergency_review_cycle_min" not in level.win_conditions:
        return
    emergency_cycle = int(level.win_conditions["emergency_review_cycle_min"])
    normal_cycle = int(level.win_conditions.get("emergency_normal_review_cycle", 21))
    if state.global_settings.r_cycle >= emergency_cycle:
        state.training.emergency_review_cycle_applied = True
        state.training.emergency_review_cycle_restored = False
    elif (
        state.training.emergency_review_cycle_applied
        and state.global_settings.r_cycle == normal_cycle
    ):
        state.training.emergency_review_cycle_restored = True


def record_review_cycle_override_applied(state: SimulationState) -> None:
    level = active_level(state)
    if level is None or "emergency_review_cycle_min" not in level.win_conditions:
        return
    emergency_cycle = int(level.win_conditions["emergency_review_cycle_min"])
    if effective_review_cycle(state.global_settings) >= emergency_cycle:
        state.training.emergency_review_cycle_applied = True
        state.training.emergency_review_cycle_restored = False


def record_review_cycle_override_cleared(state: SimulationState) -> None:
    level = active_level(state)
    if level is None or "emergency_review_cycle_min" not in level.win_conditions:
        return
    normal_cycle = int(level.win_conditions.get("emergency_normal_review_cycle", 21))
    if (
        state.training.emergency_review_cycle_applied
        and state.global_settings.r_cycle == normal_cycle
    ):
        state.training.emergency_review_cycle_restored = True


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
    progress.guided_orders_below_op = 0
    progress.guided_orders_below_lp = 0
    progress.custom_orders_placed = 0
    progress.parameter_updates_applied = 0
    progress.emergency_review_cycle_applied = False
    progress.emergency_bridge_orders_placed = 0
    progress.emergency_review_cycle_restored = False
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
    progress.guided_orders_below_op = 0
    progress.guided_orders_below_lp = 0
    progress.custom_orders_placed = 0
    progress.parameter_updates_applied = 0
    progress.emergency_review_cycle_applied = False
    progress.emergency_bridge_orders_placed = 0
    progress.emergency_review_cycle_restored = False
    certification = final_academy_level()
    return _state_for_scenario(
        profile=progress,
        scenario=certification.scenario,
        settings=certification.global_settings,
    )


def reset_progress_state() -> SimulationState:
    return SimulationState()
