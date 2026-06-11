from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from .primitives import (
    action_button,
    hero_card,
    number_field,
    optional_id,
    review_cycle_override_control,
    shell_card,
    toggle_field,
)

_GRAPH_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"],
}

_POLICY_CONTROLS = (
    ("number", "Review Cycle", "review-cycle-input", {"value": 14, "min_value": 1}),
    ("number", "R-Cost", "r-cost-input", {"value": 8, "min_value": 0, "step": 0.01}),
    (
        "number",
        "K-Cost %",
        "k-cost-input",
        {"value": 18, "min_value": 0, "step": 0.1},
    ),
    (
        "number",
        "Stockout $/unit",
        "stockout-penalty-input",
        {"value": 5, "min_value": 0, "step": 0.01},
    ),
    (
        "number",
        "Expedite %/day",
        "expedite-rate-input",
        {"value": 3, "min_value": 0, "step": 0.1},
    ),
    (
        "number",
        "GM %",
        "gm-input",
        {"value": 15, "min_value": 0, "max_value": 99, "step": 0.1},
    ),
    (
        "number",
        "Realization %",
        "realization-input",
        {"value": 100, "min_value": 50, "max_value": 100, "step": 0.1},
    ),
)

_ASQ_CONTROLS = (
    ("toggle", "Enable ASQ OP adjuster", "asq-enabled", {"enabled": True}),
    ("number", "Min Hits", "asq-min-hits", {"value": 3, "min_value": 0}),
    (
        "number",
        "Max $ Diff",
        "asq-max-diff",
        {"value": 2500, "min_value": 0, "step": 0.01},
    ),
    ("number", "Period Days", "asq-period-days", {"value": 30, "min_value": 1}),
    (
        "toggle",
        "Include transfers in ASQ",
        "asq-include-transfers",
        {"enabled": False},
    ),
)


def _control(kind: str, label: str, component_id: str, props: dict[str, Any]) -> html.Div:
    if kind == "toggle":
        return toggle_field(label, component_id, class_name="mb-2", **props)
    return number_field(label, component_id, class_name="mb-2", **props)


def _controls(definitions: tuple[tuple[str, str, str, dict[str, Any]], ...]) -> list[html.Div]:
    return [
        _control(kind, label, component_id, props)
        for kind, label, component_id, props in definitions
    ]


def _dashboard_panel(
    panel_id: str,
    children: list[Any],
    *,
    card_class_name: str = "dashboard-panel-card",
) -> html.Div:
    return html.Div(
        shell_card(children, class_name=card_class_name),
        id=panel_id,
        className="dashboard-panel",
    )


def _wrapped_action(
    label: str,
    component_id: str,
    variant: str,
    wrapper_id: str,
    *,
    class_name: str = "button-block mb-2",
) -> html.Div:
    return html.Div(
        action_button(label, component_id, variant, class_name=class_name),
        id=wrapper_id,
    )


def _command_rail_panel() -> html.Div:
    return _dashboard_panel(
        "actions-panel",
        [
            html.Div("Command rail", className="panel-label"),
            html.H3("Operations", className="panel-title"),
            html.Div(
                [
                    _wrapped_action(
                        "Place Guided Reorder",
                        "po-button",
                        "primary",
                        "guided-po-wrap",
                    ),
                    _wrapped_action(
                        "Custom Order",
                        "place-custom-order-button",
                        "light",
                        "custom-order-wrap",
                    ),
                    _wrapped_action(
                        "PO Overview",
                        "po-overview-button",
                        "dark",
                        "po-overview-wrap",
                    ),
                    _wrapped_action(
                        "Add / import items",
                        "add-item-button",
                        "light",
                        "add-item-wrap",
                        class_name="button-block",
                    ),
                    review_cycle_override_control(),
                ]
            ),
        ],
    )


def _auto_po_controls() -> html.Div:
    return html.Div(
        [
            html.Div(
                (
                    "Unlocked reward: auto purchase orders are a sandbox cheat and "
                    "stay off by default."
                ),
                id="advanced-sandbox-copy",
                className="helper-copy mb-2",
            ),
            toggle_field(
                "Auto purchase orders",
                "auto-po-enabled",
                enabled=False,
                class_name="mb-3",
            ),
        ],
        id="auto-po-shell",
    )


def _asq_controls() -> html.Div:
    return html.Div(
        [
            action_button(
                "ASQ Adjuster Settings",
                "toggle-asq-collapse",
                "secondary",
                class_name="button-block mb-2",
            ),
            dbc.Collapse(
                html.Div(
                    [
                        *_controls(_ASQ_CONTROLS),
                        action_button(
                            "Apply ASQ Now",
                            "apply-asq-button",
                            "info",
                            class_name="button-block",
                        ),
                    ]
                ),
                id="asq-collapse",
                is_open=False,
            ),
        ],
        id="asq-controls-shell",
    )


def _policy_panel() -> html.Div:
    return _dashboard_panel(
        "policy-panel",
        [
            html.Div("Policy", className="panel-label"),
            html.H3("Global parameters", className="panel-title"),
            *_controls(_POLICY_CONTROLS),
            _auto_po_controls(),
            _asq_controls(),
            html.Div(id="update-params-conf", className="mt-3"),
            action_button(
                "Update Parameters",
                "update-params-button",
                "primary",
                class_name="button-block mt-2",
            ),
        ],
    )


def _session_panel() -> html.Div:
    return _dashboard_panel(
        "session-panel",
        [
            html.Div(
                [
                    html.Div("Session", className="panel-label"),
                    action_button(
                        "Dark mode",
                        "theme-toggle",
                        "ghost",
                        class_name="theme-toggle-button button-sm",
                    ),
                ],
                className="session-card-header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Day: 1",
                                id="day-display",
                                className="status-value",
                            ),
                            html.Div(
                                "Status: Paused",
                                id="sim-status",
                                className="status-copy",
                            ),
                        ],
                        className="session-primary-block",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Simulation speed",
                                className="control-label session-speed-label",
                            ),
                            html.Div(
                                "1 tick/sec",
                                id="sim-speed-readout",
                                className="speed-readout",
                            ),
                            dcc.Slider(
                                id="sim-speed-slider",
                                min=0.5,
                                max=6.0,
                                step=0.5,
                                value=1.0,
                                marks=None,
                                updatemode="drag",
                                allow_direct_input=False,
                                className="sim-speed-slider",
                            ),
                        ],
                        className="session-speed-block",
                    ),
                ],
                className="session-metrics-grid",
            ),
            html.Div(id="asq-apply-feedback", className="mt-3"),
        ],
        card_class_name="dashboard-panel-card session-card mt-3",
    )


def _graph_panel() -> html.Div:
    return _dashboard_panel(
        "graph-panel",
        [
            html.Div("Signal map", className="panel-label"),
            html.H3(
                "Inventory signal map",
                className="panel-title",
                id="graph-panel-title",
            ),
            dcc.Graph(
                id="inventory-graph",
                figure={},
                className="inventory-graph",
                config=_GRAPH_CONFIG,
            ),
        ],
    )


def _metric_col(
    *,
    panel_id: str,
    title: str,
    card_id: str,
    title_id: str | None = None,
) -> dbc.Col:
    return dbc.Col(
        shell_card(
            [
                html.Div(
                    title,
                    className="panel-title-small",
                    **optional_id(title_id),
                ),
                html.Div(id=card_id),
            ],
            class_name="dashboard-panel-card h-100",
        ),
        id=panel_id,
        className="dashboard-panel",
        md=4,
    )


def _detail_col(
    *,
    panel_id: str,
    label: str,
    title: str,
    body_id: str,
    lg: int,
    title_id: str | None = None,
) -> dbc.Col:
    return dbc.Col(
        shell_card(
            [
                html.Div(label, className="panel-label"),
                html.H3(title, className="panel-title", **optional_id(title_id)),
                html.Div(id=body_id),
            ],
            class_name="dashboard-panel-card h-100",
        ),
        id=panel_id,
        className="dashboard-panel",
        lg=lg,
    )


def simulator_shell() -> html.Div:
    return html.Div(
        [
            hero_card(
                kicker="Simulator",
                title="Simulator workspace",
                title_id="simulator-title",
                copy=(
                    "Run the full inventory workspace with imports, policy controls, "
                    "signal maps, and sandbox reward controls."
                ),
                copy_id="simulator-copy",
                actions=[
                    action_button(
                        "Return to Academy",
                        "simulator-return-button",
                        "secondary",
                        class_name="button-pill",
                    ),
                    action_button(
                        "Reference",
                        "simulator-reference-button",
                        "ghost",
                        class_name="button-pill",
                    ),
                ],
                class_name="academy-shell mb-4",
            )
        ],
        id="simulator-shell",
        style={"display": "none"},
    )


def dashboard_shell() -> html.Div:
    return html.Div(
        [
            hero_card(
                kicker="Inventory Management Training",
                kicker_id="experience-kicker",
                title="Lesson workspace",
                title_id="experience-title",
                copy=(
                    "Adjust the live controls, read the signals, and close the objective window."
                ),
                copy_id="experience-copy",
                actions=[
                    action_button(
                        "Start Lesson",
                        "start-button",
                        "success",
                        class_name="button-pill",
                    ),
                    action_button(
                        "Restart Lesson",
                        "reset-button",
                        "secondary",
                        class_name="button-pill",
                    ),
                    action_button(
                        "Reference",
                        "experience-reference-button",
                        "ghost",
                        class_name="button-pill",
                    ),
                    html.Div(
                        action_button(
                            "Return to Academy",
                            "return-to-menu-button",
                            "ghost",
                            class_name="button-pill",
                        ),
                        id="lesson-return-wrap",
                    ),
                ],
                class_name="hero-banner-card experience-banner-card",
                extra_left=[
                    html.Div(
                        id="lesson-compact-summary",
                        className="lesson-compact-summary mt-3",
                    )
                ],
            ),
            html.Div(id="kpi-strip", className="dashboard-kpi-strip"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _command_rail_panel(),
                            _policy_panel(),
                            _session_panel(),
                        ],
                        xl=3,
                        className="control-rail-column dashboard-rail-column",
                    ),
                    dbc.Col(
                        [
                            _graph_panel(),
                            dbc.Row(
                                [
                                    _metric_col(
                                        panel_id="service-panel",
                                        title="Service",
                                        title_id="service-panel-title",
                                        card_id="service-card",
                                    ),
                                    _metric_col(
                                        panel_id="costs-panel",
                                        title="Costs",
                                        card_id="costs-card",
                                    ),
                                    _metric_col(
                                        panel_id="sales-panel",
                                        title="Sales",
                                        card_id="sales-card",
                                    ),
                                ],
                                className="g-3 dashboard-insights-row",
                            ),
                            dbc.Row(
                                [
                                    _detail_col(
                                        panel_id="inventory-panel",
                                        label="Inventory",
                                        title="Inventory planner",
                                        title_id="inventory-panel-title",
                                        body_id="inventory-table-shell",
                                        lg=7,
                                    ),
                                    _detail_col(
                                        panel_id="exceptions-panel",
                                        label="Exceptions",
                                        title="Exception center",
                                        body_id="exception-center-shell",
                                        lg=5,
                                    ),
                                ],
                                className="g-3 dashboard-detail-row",
                            ),
                        ],
                        xl=9,
                        className="dashboard-main-column",
                    ),
                ],
                className="g-3 dashboard-main-layout",
            ),
        ],
        id="dashboard-shell",
        className="dashboard-shell",
        style={"display": "none"},
    )
