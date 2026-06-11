from __future__ import annotations

from dataclasses import asdict

import dash_bootstrap_components as dbc
from dash import html

from ..models import GlobalSettings, InventoryItem, SimulationState
from ..services.planning import format_money, item_on_order, update_gs_related_values
from ..services.training import (
    academy_level_status,
    academy_levels,
    active_level,
    evaluate_active_lesson,
    fill_rate,
    lesson_days_remaining,
)
from .panels import _kpi_card, _lesson_snapshot_disclosure, _lesson_snapshot_table

_GLOSSARY_ENTRIES = (
    (
        "PNA",
        "Projected net available. The inventory position used for replenishment decisions. "
        "Full training formula: on hand - reserved - committed - backordered + on "
        "order + received.",
    ),
    (
        "On hand",
        "Physical stock currently available in the warehouse before considering future "
        "receipts or unresolved demand.",
    ),
    ("On order", "Quantity already ordered but not yet received."),
    ("Backorder", "Demand that could not be filled from available stock."),
    ("Usage rate", "The monthly demand pace used by the simulator."),
    ("Daily usage", "Usage rate divided by day basis."),
    ("Lead time", "The time between placing an order and receiving it."),
    ("OP", "Order point. The level where replenishment should begin."),
    (
        "Safety stock",
        "Extra stock added to protect against normal demand or lead-time variation.",
    ),
    (
        "LP",
        "Line point. The higher review threshold used to decide what else should ride "
        "along during a line review.",
    ),
    ("OQ", "Order quantity. The practical quantity to buy when replenishment is needed."),
    (
        "EOQ",
        "Economic order quantity. A calculated quantity that balances replenishment cost "
        "and carrying cost.",
    ),
    (
        "SOQ",
        "Suggested order quantity. The simulator's recommendation after considering PNA, "
        "OP/LP, OQ, and standard pack rounding.",
    ),
    ("Standard pack", "Supplier pack increment used to round suggested buys."),
    ("Critical point", "Urgent replenishment threshold below normal protection."),
    (
        "Surplus threshold",
        "LP + OQ in the simulator's short-term overstock model.",
    ),
    (
        "Fill rate",
        "Percent of stock lines completed without a stockout miss.",
    ),
    (
        "Hits",
        "How often an item appears on demand lines. Used to show customer frequency/popularity.",
    ),
    (
        "ASQ",
        "Average sales quantity. Keep locked until advanced and certification content.",
    ),
)


def _compact_lesson_pill_text(row: str) -> str:
    replacements = (
        ("Current fill rate: ", "Fill rate: "),
        ("Current after-overhead GM: ", "After-OH GM: "),
        ("After-overhead GM: ", "After-OH GM: "),
        ("Items closing at/above OP: ", "At/above OP: "),
        ("Close with PNA at/above OP: ", "PNA at/above OP: "),
        ("Reorder triggered after PNA fell below OP: ", "Below-OP reorder: "),
        ("Guided reorders used: ", "Guided reorders: "),
        ("Manual custom orders used: ", "Custom orders: "),
        ("Parameter updates used: ", "Policy updates: "),
        ("On-order inventory: ", "On order: "),
        ("Avg inventory value: ", "Avg inv value: "),
        ("Items at/below critical point: ", "At/below CP: "),
        ("Items above surplus threshold: ", "Above surplus: "),
        ("Backorder at close: ", "Close backorder: "),
    )
    for old, new in replacements:
        if row.startswith(old):
            return row.replace(old, new, 1)
    return row


def _lesson_uses_day_basis(level) -> bool:
    day_basis_phrase = "day basis"
    return day_basis_phrase in level.formula.casefold() or any(
        day_basis_phrase in step.casefold() for step in level.tutorial_steps
    )


def _lesson_day_basis_helper(state: SimulationState) -> html.Div:
    return html.Div(
        f"Simulator simplified month: {state.global_settings.day_basis} days",
        className="helper-copy mb-2",
    )


def _lesson_secondary_note(
    title: str,
    hint: str,
    copy: str,
) -> html.Details:
    return _lesson_snapshot_disclosure(
        html.Div(copy, className="helper-copy mb-0"),
        label=title,
        hint=hint,
    )


def _lesson_secondary_notes(level) -> list[html.Details]:
    notes: list[html.Details] = []
    if level.advanced_note:
        notes.append(
            _lesson_secondary_note(
                "Advanced note",
                "Optional deeper context",
                level.advanced_note,
            )
        )
    if level.csd_mapping_note:
        notes.append(
            _lesson_secondary_note(
                "CSD mapping",
                "Optional system mapping",
                level.csd_mapping_note,
            )
        )
    return notes


def _projected_line_summary(
    state: SimulationState,
    *,
    review_cycle: int,
) -> tuple[float, int, float, float]:
    settings = GlobalSettings.from_dict(asdict(state.global_settings))
    settings.r_cycle = int(review_cycle)
    settings.review_cycle_override_days = None
    total_soq = 0.0
    line_count = 0
    total_lp = 0.0
    total_cost = 0.0
    for source in state.items:
        item = InventoryItem.from_dict(source.to_dict())
        update_gs_related_values(item, settings)
        total_soq += item.soq
        total_lp += item.lp
        total_cost += item.soq * item.item_cost
        if item.soq > 0:
            line_count += 1
    return total_soq, line_count, total_lp, total_cost


def _emergency_replenishment_panel(state: SimulationState, level) -> html.Div:
    normal_cycle = int(level.win_conditions.get("emergency_normal_review_cycle", 7))
    emergency_cycle = int(level.win_conditions.get("emergency_review_cycle_min", 14))
    normal_qty, normal_lines, normal_lp, normal_cost = _projected_line_summary(
        state, review_cycle=normal_cycle
    )
    emergency_qty, emergency_lines, emergency_lp, emergency_cost = _projected_line_summary(
        state, review_cycle=emergency_cycle
    )
    return html.Div(
        [
            html.Div(
                (
                    "A bridge buy is a temporary timing decision. Raising review cycle "
                    "days makes the recommendation look farther ahead, which can bring "
                    "near-line-point items into the same PO. Use it to cover the gap, "
                    "then return to the regular cadence."
                ),
                className="helper-copy mb-2",
            ),
            _lesson_snapshot_table(
                "Before / After RRAR",
                ("Control", "Normal Buy", "Emergency Buy"),
                (
                    ("Review Cycle", f"{normal_cycle} days", f"{emergency_cycle} days"),
                    ("Included Lines", str(normal_lines), str(emergency_lines)),
                    ("Line Point Total", f"{normal_lp:.1f}", f"{emergency_lp:.1f}"),
                    ("Suggested Qty", f"{normal_qty:.1f}", f"{emergency_qty:.1f}"),
                    ("Extended Cost", format_money(normal_cost), format_money(emergency_cost)),
                    ("Tradeoff", "Standard line coverage", "Higher bridge coverage"),
                ),
            ),
            _lesson_snapshot_table(
                "Demand Center Header",
                ("Buyer", "Vendor", "Warehouse", "PLine", "Target", "Merge"),
                (
                    (
                        "RSC",
                        "100 - Apex Electrical",
                        "MAIN",
                        "ELEC",
                        "$7,500",
                        "Review after accept",
                    ),
                ),
            ),
        ],
        className="lesson-snapshot-stack",
    )


def _reference_definition_block(term: str, definition: str) -> html.Div:
    return html.Div(
        [
            html.Div(term, className="lesson-snapshot-label"),
            html.Div(definition, className="helper-copy mb-0"),
        ],
        className="reference-entry",
    )


def reference_modal_children() -> list:
    glossary = html.Div(
        [
            html.Div("Glossary", className="panel-label"),
            html.H3("Simulator terms", className="panel-title-small"),
            html.Div(
                [
                    _reference_definition_block(term, definition)
                    for term, definition in _GLOSSARY_ENTRIES
                ],
                className="reference-entry-list",
            ),
        ],
        className="reference-panel",
    )
    csd_mapping = html.Div(
        [
            html.Div("CSD mapping", className="panel-label"),
            html.H3("General context", className="panel-title-small"),
            html.Div(
                (
                    "IMSim is an independent inventory management training project. It is "
                    "not a CSD screen, an Infor tool, or official ERP training."
                ),
                className="helper-copy mb-3",
            ),
            html.Ul(
                [
                    html.Li(
                        "CSD users may recognize concepts from Product Warehouse Product "
                        "Setup ordering controls, Demand Center/RRAR workflows, Product "
                        "Inquiry - Replenishment, and surplus reporting."
                    ),
                    html.Li(
                        "Exact behavior depends on company setup, product line setup, "
                        "order method, warehouse and product controls, security, and "
                        "customizations."
                    ),
                    html.Li(
                        "The simulator keeps its SOQ and exception logic intentionally "
                        "simple so the lessons stay teachable and comparable."
                    ),
                ],
                className="lesson-copy-list",
            ),
        ],
        className="reference-panel",
    )
    return [
        dbc.Tabs(
            [
                dbc.Tab(glossary, label="Glossary", tab_id="glossary"),
                dbc.Tab(csd_mapping, label="CSD Mapping", tab_id="csd-mapping"),
            ],
            id="reference-tabs",
            active_tab="glossary",
            class_name="reference-tabs",
        )
    ]


def academy_progress_children(state: SimulationState) -> list:
    total_levels = len(academy_levels())
    completed = len(state.training.completed_levels)
    simulator = "Unlocked" if state.training.simulator_unlocked else "Locked"
    return [
        dbc.Row(
            [
                dbc.Col(
                    _kpi_card(
                        "Lessons Complete",
                        f"{completed}/{total_levels}",
                        "neutral",
                        "Academy progress",
                    ),
                    md=4,
                ),
                dbc.Col(
                    _kpi_card(
                        "Next Unlock",
                        f"Level {min(total_levels, state.training.highest_unlocked_level)}",
                        "good" if completed else "neutral",
                        "Highest playable lesson",
                    ),
                    md=4,
                ),
                dbc.Col(
                    _kpi_card(
                        "Simulator",
                        simulator,
                        "good" if state.training.simulator_unlocked else "warn",
                        "Free-play sandbox",
                    ),
                    md=4,
                ),
            ],
            className="g-3",
        )
    ]


def academy_result_children(state: SimulationState):
    if not state.training.last_result_title and not state.training.last_result_message:
        return html.Div()
    tone = "success" if state.training.lesson_status == "passed" else "warning"
    return dbc.Alert(
        [
            html.Strong(state.training.last_result_title or "Academy update"),
            html.Div(state.training.last_result_message),
        ],
        color=tone,
        class_name="academy-result-alert",
    )


def lesson_tutorial_children(state: SimulationState) -> list:
    level = active_level(state)
    if level is None:
        return [dbc.Alert("Select a lesson from the academy menu.", color="secondary")]
    framing = []
    day_basis_helper = [_lesson_day_basis_helper(state)] if _lesson_uses_day_basis(level) else []
    secondary_notes = _lesson_secondary_notes(level)
    if level.teaching_goal:
        framing.append(html.Div(level.teaching_goal, className="helper-copy mb-2"))
    if level.concept_tags:
        framing.append(
            html.Div(
                [
                    dbc.Badge(tag, color="secondary", pill=True, class_name="me-1")
                    for tag in level.concept_tags
                ],
                className="mb-2",
            )
        )
    if level.index == 2 and state.items:
        item = state.items[0]
        on_order = item_on_order(item)
        reserved = 0.0
        committed = 0.0
        received = 0.0
        return [
            *framing,
            html.Div(level.formula, className="lesson-formula-chip"),
            html.Div(
                (
                    "This lesson keeps Reserved, Committed, and Received at 0 so the "
                    "purchasing view stays simple."
                ),
                className="helper-copy mb-2",
            ),
            html.Div(
                (
                    "Current PNA = "
                    f"{item.on_hand:.1f} - {reserved:.1f} - {committed:.1f} - "
                    f"{item.backorder:.1f} + {on_order:.1f} + {received:.1f} = {item.pna:.1f}"
                ),
                className="lesson-formula-chip",
            ),
            html.Ul([html.Li(step) for step in level.tutorial_steps], className="lesson-copy-list"),
            *secondary_notes,
        ]
    if level.index == 18 and state.items:
        return [
            *framing,
            html.Div(level.formula, className="lesson-formula-chip"),
            _emergency_replenishment_panel(state, level),
            html.Ul([html.Li(step) for step in level.tutorial_steps], className="lesson-copy-list"),
            *secondary_notes,
        ]
    return [
        *framing,
        html.Div(level.formula, className="lesson-formula-chip"),
        *day_basis_helper,
        html.Ul([html.Li(step) for step in level.tutorial_steps], className="lesson-copy-list"),
        *secondary_notes,
    ]


def lesson_objective_children(state: SimulationState) -> list:
    level = active_level(state)
    if level is None:
        return [dbc.Alert("No active lesson.", color="secondary")]
    evaluation = evaluate_active_lesson(state)
    headline = f"{lesson_days_remaining(state)} day(s) remaining"
    if evaluation is not None and evaluation.completed:
        headline = "Lesson window closed"
    rows = list(evaluation.metric_rows if evaluation is not None else ())
    if "fill_rate_min" in level.win_conditions:
        fill = fill_rate(state)
        fill_value = "n/a" if fill is None else f"{fill * 100:.1f}%"
        rows.insert(0, f"Current fill rate: {fill_value}")
    children: list = []
    if state.training.lesson_status in {"passed", "failed"} and (
        state.training.last_result_title or state.training.last_result_message
    ):
        children.append(
            dbc.Alert(
                [
                    html.Strong(state.training.last_result_title or "Lesson update"),
                    html.Div(state.training.last_result_message),
                ],
                color="success" if state.training.lesson_status == "passed" else "warning",
                class_name="academy-result-alert",
            )
        )
    children.extend(
        [
            html.Div(headline, className="lesson-objective-headline"),
            html.Ul([html.Li(row) for row in rows], className="lesson-copy-list"),
        ]
    )
    return children


def lesson_locked_children(state: SimulationState) -> list:
    level = active_level(state)
    if level is None:
        return [dbc.Alert("No active lesson.", color="secondary")]
    return [
        html.Ul(
            [html.Li(row) for row in level.locked_features],
            className="lesson-copy-list lesson-locked-list",
        )
    ]


def lesson_compact_summary_children(state: SimulationState) -> list:
    level = active_level(state)
    if level is None:
        return []
    evaluation = evaluate_active_lesson(state)
    headline = (
        "Lesson window closed"
        if evaluation is not None and evaluation.completed
        else f"{lesson_days_remaining(state)} day(s) remaining"
    )
    objective_rows = [
        _compact_lesson_pill_text(row)
        for row in list(evaluation.metric_rows if evaluation is not None else ())
    ]
    return [
        html.Div(level.formula, className="lesson-compact-chip"),
        html.Div(
            [
                html.Span(headline, className="lesson-compact-pill lesson-compact-pill-strong"),
                *[html.Span(row, className="lesson-compact-pill") for row in objective_rows],
            ],
            className="lesson-compact-inline",
        ),
    ]


def academy_level_card_children(level_index: int, state: SimulationState) -> list:
    level = academy_levels()[level_index - 1]
    status = academy_level_status(state.training, level)
    status_label = {
        "completed": "Completed",
        "unlocked": "Unlocked",
        "locked": "Locked",
    }[status]
    button_label = "Replay Lesson" if status == "completed" else "Start Lesson"
    return [
        html.Div(f"Level {level.index}", className="academy-card-kicker"),
        html.Div(level.title, className="academy-card-title"),
        html.P(level.summary, className="academy-card-copy"),
        dbc.Badge(
            status_label,
            color="success"
            if status == "completed"
            else ("primary" if status == "unlocked" else "secondary"),
            pill=True,
            class_name="academy-status-badge",
        ),
        html.Div(level.formula, className="academy-card-formula"),
        html.Button(
            button_label,
            id=f"academy-level-{level.index}-button",
            n_clicks=0,
            className="imsim-button button-primary button-block mt-3",
            disabled=status == "locked",
        ),
    ]


def simulator_unlock_children(state: SimulationState) -> list:
    unlocked = state.training.simulator_unlocked
    return [
        html.Div("Simulator", className="academy-card-kicker"),
        html.Div("Simulator Mode", className="academy-card-title"),
        html.P(
            "The full IM dashboard with imports, ASQ, and the sandbox reward controls."
            if unlocked
            else "Pass certification to unlock the full simulator and the sandbox reward controls.",
            className="academy-card-copy",
        ),
        dbc.Badge(
            "Unlocked" if unlocked else "Locked",
            color="primary" if unlocked else "secondary",
            pill=True,
            class_name="academy-status-badge",
        ),
        html.Div("Free-play sandbox", className="academy-card-formula"),
    ]
