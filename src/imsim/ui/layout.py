from __future__ import annotations

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html

from ..config import IMSimConfig
from ..models import default_state
from ..services.training import academy_levels
from .components import (
    academy_level_card_children,
    custom_order_grid_options,
    github_footer_card,
    po_overview_grid_options,
    simulator_unlock_children,
)


def _action_button(
    label: str,
    component_id: str,
    variant: str,
    *,
    disabled: bool = False,
    class_name: str = "",
) -> html.Button:
    classes = " ".join(
        part
        for part in [
            "imsim-button",
            f"button-{variant}",
            class_name,
        ]
        if part
    )
    return html.Button(
        label,
        id=component_id,
        n_clicks=0,
        className=classes,
        disabled=disabled,
    )


def _number_field(
    label: str,
    component_id: str,
    *,
    value: float | int | None = None,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    step: float | int | None = None,
    class_name: str = "",
) -> html.Div:
    return html.Div(
        [
            html.Label(label, htmlFor=component_id, className="control-label"),
            dbc.Input(
                id=component_id,
                type="number",
                value=value,
                min=min_value,
                max=max_value,
                step=step,
                className="control-input",
                inputMode="decimal",
            ),
        ],
        className=" ".join(part for part in ["control-field", class_name] if part),
    )


def _toggle_field(
    label: str,
    component_id: str,
    *,
    enabled: bool,
    class_name: str = "",
) -> html.Div:
    return html.Div(
        [
            html.Span(label, className="imsim-toggle-copy"),
            dbc.Switch(
                id=component_id,
                value=enabled,
                label="",
                class_name="imsim-toggle-switch",
                input_class_name="imsim-toggle-input",
            ),
        ],
        className=" ".join(part for part in ["toggle-field", class_name] if part),
    )


def build_layout(config: IMSimConfig):
    initial_state = default_state()
    return html.Div(
        dbc.Container(
            [
                dcc.Store(id="user-data-store", storage_type="local", data={}),
                dcc.Store(id="page-load", data=0),
                dcc.Store(id="gh-footer-store", storage_type="local", data=True),
                dcc.Store(id="upload-preview-data"),
                dcc.Store(id="theme-store", storage_type="local", data="light"),
                dcc.Store(id="session-revision", data=0),
                dcc.Store(id="dashboard-tick", data=0),
                dcc.Interval(id="interval-component", interval=1000, disabled=True),
                dcc.Interval(
                    id="shutdown-poll",
                    interval=1000,
                    n_intervals=0,
                    disabled=not (config.admin_token or config.allow_dev_shutdown),
                ),
                html.Div(id="maintenance-banner"),
                html.Div(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Div(
                                                        "Inventory Management Training",
                                                        className="hero-kicker",
                                                    ),
                                                    html.H1(
                                                        "IMSim Academy",
                                                        className="hero-title",
                                                    ),
                                                    html.P(
                                                        (
                                                            "Learn inventory management "
                                                            "a level at a time. Work "
                                                            "through guided lessons, "
                                                            "unlock controls, "
                                                            "and earn the full simulator."
                                                        ),
                                                        className="hero-copy",
                                                    ),
                                                ],
                                                lg=8,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        _action_button(
                                                            "Dark mode",
                                                            "academy-theme-toggle",
                                                            "ghost",
                                                            class_name=(
                                                                "theme-toggle-button button-sm"
                                                            ),
                                                        ),
                                                        _action_button(
                                                            "Reset Progress",
                                                            "academy-reset-progress-button",
                                                            "secondary",
                                                            class_name="button-pill",
                                                        ),
                                                    ],
                                                    className="hero-actions hero-actions-compact",
                                                ),
                                                lg=4,
                                                className="hero-actions-wrap",
                                            ),
                                        ],
                                        className="g-3 align-items-center",
                                    ),
                                    html.Div(id="academy-result-banner", className="mt-3"),
                                    html.Div(id="academy-progress-summary", className="mt-4"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        html.Div(
                                                            academy_level_card_children(
                                                                level.index, initial_state
                                                            ),
                                                            id=f"academy-level-{level.index}-card",
                                                        )
                                                    ),
                                                    className="shell-card academy-card h-100",
                                                ),
                                                lg=4,
                                                md=6,
                                            )
                                            for level in academy_levels()
                                        ]
                                        + [
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.Div(
                                                                simulator_unlock_children(
                                                                    initial_state
                                                                ),
                                                                id="academy-simulator-card",
                                                            ),
                                                            _action_button(
                                                                "Open Simulator",
                                                                "academy-simulator-button",
                                                                "dark",
                                                                disabled=True,
                                                                class_name="button-block mt-3",
                                                            ),
                                                        ]
                                                    ),
                                                    className="shell-card academy-card h-100",
                                                ),
                                                lg=4,
                                                md=6,
                                            )
                                        ],
                                        className="g-3 mt-2",
                                    ),
                                ]
                            ),
                            className="shell-card academy-shell mb-4",
                        )
                    ],
                    id="academy-menu-shell",
                ),
                html.Div(
                    [
                        dbc.Modal(
                            [
                                dbc.ModalHeader(
                                    html.Div(
                                        [
                                            html.Div("Lesson Brief", className="hero-kicker"),
                                            html.H2(
                                                "Academy Lesson",
                                                id="lesson-title",
                                                className="lesson-intro-title",
                                            ),
                                            html.P(
                                                "Start a lesson from the academy menu.",
                                                id="lesson-copy",
                                                className="lesson-intro-copy",
                                            ),
                                        ]
                                    ),
                                    close_button=False,
                                    class_name="lesson-intro-header",
                                ),
                                dbc.ModalBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div(
                                                                    "Tutorial",
                                                                    className="panel-label",
                                                                ),
                                                                html.Div(id="lesson-tutorial"),
                                                            ]
                                                        ),
                                                        className=(
                                                            "shell-card lesson-detail-card h-100"
                                                        ),
                                                    ),
                                                    lg=7,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div(
                                                                    "Objectives",
                                                                    className="panel-label",
                                                                ),
                                                                html.Div(id="lesson-objectives"),
                                                            ]
                                                        ),
                                                        className=(
                                                            "shell-card lesson-detail-card h-100"
                                                        ),
                                                    ),
                                                    lg=5,
                                                ),
                                            ],
                                            className="g-3 lesson-intro-grid",
                                        ),
                                        html.Div(
                                            id="lesson-locked",
                                            className="lesson-locked-cache",
                                        ),
                                    ],
                                    class_name="lesson-intro-body",
                                ),
                                dbc.ModalFooter(
                                    _action_button(
                                        "Enter Lesson",
                                        "lesson-intro-dismiss-button",
                                        "primary",
                                        class_name="button-pill",
                                    ),
                                    className="lesson-intro-footer",
                                ),
                            ],
                            id="lesson-intro-modal",
                            is_open=False,
                            centered=True,
                            scrollable=True,
                            size="xl",
                            class_name="lesson-intro-modal",
                            dialog_class_name="lesson-intro-dialog",
                            content_class_name="lesson-intro-content",
                            backdrop_class_name="lesson-intro-backdrop",
                        )
                    ],
                    id="lesson-shell",
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("Simulator", className="hero-kicker"),
                                                html.H1(
                                                    "Unlocked Simulator",
                                                    id="simulator-title",
                                                    className="hero-title",
                                                ),
                                                html.P(
                                                    (
                                                        "This is the full IM dashboard. Imports, "
                                                        "free play, and the sandbox "
                                                        "reward controls "
                                                        "are now available."
                                                    ),
                                                    id="simulator-copy",
                                                    className="hero-copy",
                                                ),
                                            ],
                                            lg=8,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    _action_button(
                                                        "Return to Academy",
                                                        "simulator-return-button",
                                                        "secondary",
                                                        class_name="button-pill",
                                                    ),
                                                ],
                                                className="hero-actions hero-actions-compact",
                                            ),
                                            lg=4,
                                            className="hero-actions-wrap",
                                        ),
                                    ],
                                    className="g-3 align-items-center",
                                )
                            ),
                            className="shell-card academy-shell mb-4",
                        )
                    ],
                    id="simulator-shell",
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    "Inventory Management Training",
                                                    id="experience-kicker",
                                                    className="hero-kicker",
                                                ),
                                                html.H1(
                                                    "Lesson control deck",
                                                    id="experience-title",
                                                    className="hero-title",
                                                ),
                                                html.P(
                                                    (
                                                        "Use the lesson controls and hit "
                                                        "the objective window."
                                                    ),
                                                    id="experience-copy",
                                                    className="hero-copy",
                                                ),
                                                html.Div(
                                                    id="lesson-compact-summary",
                                                    className="lesson-compact-summary mt-3",
                                                ),
                                            ],
                                            lg=8,
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                [
                                                    _action_button(
                                                        "Start Lesson",
                                                        "start-button",
                                                        "success",
                                                        class_name="button-pill",
                                                    ),
                                                    _action_button(
                                                        "Restart Lesson",
                                                        "reset-button",
                                                        "secondary",
                                                        class_name="button-pill",
                                                    ),
                                                    html.Div(
                                                        _action_button(
                                                            "Return to Academy",
                                                            "return-to-menu-button",
                                                            "ghost",
                                                            class_name="button-pill",
                                                        ),
                                                        id="lesson-return-wrap",
                                                    ),
                                                ],
                                                className="hero-actions hero-actions-compact",
                                            ),
                                            lg=4,
                                            className="hero-actions-wrap",
                                        ),
                                    ],
                                    className="g-3 align-items-center",
                                )
                            ),
                            className="shell-card hero-banner-card experience-banner-card mb-4",
                        ),
                        html.Div(id="kpi-strip", className="mt-4"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            "Command rail", className="panel-label"
                                                        ),
                                                        html.H3(
                                                            "Controls", className="panel-title"
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    _action_button(
                                                                        "Place Guided Reorder",
                                                                        "po-button",
                                                                        "primary",
                                                                        class_name=(
                                                                            "button-block mb-2"
                                                                        ),
                                                                    ),
                                                                    id="guided-po-wrap",
                                                                ),
                                                                html.Div(
                                                                    _action_button(
                                                                        "Custom Order",
                                                                        "place-custom-order-button",
                                                                        "light",
                                                                        class_name=(
                                                                            "button-block mb-2"
                                                                        ),
                                                                    ),
                                                                    id="custom-order-wrap",
                                                                ),
                                                                html.Div(
                                                                    _action_button(
                                                                        "PO Overview",
                                                                        "po-overview-button",
                                                                        "dark",
                                                                        class_name=(
                                                                            "button-block mb-2"
                                                                        ),
                                                                    ),
                                                                    id="po-overview-wrap",
                                                                ),
                                                                html.Div(
                                                                    _action_button(
                                                                        "Add or Import Items",
                                                                        "add-item-button",
                                                                        "warning",
                                                                        class_name="button-block",
                                                                    ),
                                                                    id="add-item-wrap",
                                                                ),
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                className="shell-card mb-3",
                                            ),
                                            id="actions-panel",
                                        ),
                                        html.Div(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div("Policy", className="panel-label"),
                                                        html.H3(
                                                            "Global parameters",
                                                            className="panel-title",
                                                        ),
                                                        _number_field(
                                                            "Review Cycle",
                                                            "review-cycle-input",
                                                            value=14,
                                                            min_value=1,
                                                            class_name="mb-2",
                                                        ),
                                                        _number_field(
                                                            "R-Cost",
                                                            "r-cost-input",
                                                            value=8,
                                                            min_value=0,
                                                            step=0.01,
                                                            class_name="mb-2",
                                                        ),
                                                        _number_field(
                                                            "K-Cost %",
                                                            "k-cost-input",
                                                            value=18,
                                                            min_value=0,
                                                            step=0.1,
                                                            class_name="mb-2",
                                                        ),
                                                        _number_field(
                                                            "Stockout $/unit",
                                                            "stockout-penalty-input",
                                                            value=5,
                                                            min_value=0,
                                                            step=0.01,
                                                            class_name="mb-2",
                                                        ),
                                                        _number_field(
                                                            "Expedite %/day",
                                                            "expedite-rate-input",
                                                            value=3,
                                                            min_value=0,
                                                            step=0.1,
                                                            class_name="mb-2",
                                                        ),
                                                        _number_field(
                                                            "GM %",
                                                            "gm-input",
                                                            value=15,
                                                            min_value=0,
                                                            max_value=99,
                                                            step=0.1,
                                                            class_name="mb-2",
                                                        ),
                                                        _number_field(
                                                            "Realization %",
                                                            "realization-input",
                                                            value=100,
                                                            min_value=50,
                                                            max_value=100,
                                                            step=0.1,
                                                            class_name="mb-2",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    (
                                                                        "Unlocked reward: auto "
                                                                        "purchase orders are a "
                                                                        "sandbox cheat and stay "
                                                                        "off by default."
                                                                    ),
                                                                    id="advanced-sandbox-copy",
                                                                    className="helper-copy mb-2",
                                                                ),
                                                                _toggle_field(
                                                                    "Auto purchase orders",
                                                                    "auto-po-enabled",
                                                                    enabled=False,
                                                                    class_name="mb-3",
                                                                ),
                                                            ],
                                                            id="auto-po-shell",
                                                        ),
                                                        html.Div(
                                                            [
                                                                _action_button(
                                                                    "ASQ Adjuster Settings",
                                                                    "toggle-asq-collapse",
                                                                    "secondary",
                                                                    class_name="button-block mb-2",
                                                                ),
                                                                dbc.Collapse(
                                                                    html.Div(
                                                                        [
                                                                            _toggle_field(
                                                                                (
                                                                                    "Enable ASQ "
                                                                                    "OP adjuster"
                                                                                ),
                                                                                "asq-enabled",
                                                                                enabled=True,
                                                                                class_name="mb-2",
                                                                            ),
                                                                            _number_field(
                                                                                "Min Hits",
                                                                                "asq-min-hits",
                                                                                value=3,
                                                                                min_value=0,
                                                                                class_name="mb-2",
                                                                            ),
                                                                            _number_field(
                                                                                "Max $ Diff",
                                                                                "asq-max-diff",
                                                                                value=2500,
                                                                                min_value=0,
                                                                                step=0.01,
                                                                                class_name="mb-2",
                                                                            ),
                                                                            _number_field(
                                                                                "Period Days",
                                                                                "asq-period-days",
                                                                                value=30,
                                                                                min_value=1,
                                                                                class_name="mb-2",
                                                                            ),
                                                                            _toggle_field(
                                                                                (
                                                                                    "Include "
                                                                                    "transfers "
                                                                                    "in ASQ"
                                                                                ),
                                                                                "asq-include-transfers",
                                                                                enabled=False,
                                                                                class_name="mb-2",
                                                                            ),
                                                                            _action_button(
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
                                                        ),
                                                        html.Div(
                                                            id="update-params-conf",
                                                            className="mt-3",
                                                        ),
                                                        _action_button(
                                                            "Update Parameters",
                                                            "update-params-button",
                                                            "primary",
                                                            class_name="button-block mt-2",
                                                        ),
                                                    ]
                                                ),
                                                className="shell-card",
                                            ),
                                            id="policy-panel",
                                        ),
                                        html.Div(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    "Session",
                                                                    className="panel-label",
                                                                ),
                                                                _action_button(
                                                                    "Dark mode",
                                                                    "theme-toggle",
                                                                    "ghost",
                                                                    class_name=(
                                                                        "theme-toggle-button "
                                                                        "button-sm"
                                                                    ),
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
                                                                            className=(
                                                                                "control-label "
                                                                                "session-speed-label"
                                                                            ),
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
                                                                            className=(
                                                                                "sim-speed-slider"
                                                                            ),
                                                                        ),
                                                                    ],
                                                                    className="session-speed-block",
                                                                ),
                                                            ],
                                                            className="session-metrics-grid",
                                                        ),
                                                        html.Div(
                                                            id="asq-apply-feedback",
                                                            className="mt-3",
                                                        ),
                                                    ]
                                                ),
                                                className="shell-card session-card mt-3",
                                            ),
                                            id="session-panel",
                                        ),
                                    ],
                                    xl=3,
                                    className="control-rail-column dashboard-rail-column",
                                ),
                                dbc.Col(
                                    [
                                        html.Div(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            "Visuals", className="panel-label"
                                                        ),
                                                        html.H3(
                                                            "Inventory signal map",
                                                            className="panel-title",
                                                            id="graph-panel-title",
                                                        ),
                                                        dcc.Graph(
                                                            id="inventory-graph",
                                                            figure={},
                                                            className="inventory-graph",
                                                            config={
                                                                "responsive": True,
                                                                "displaylogo": False,
                                                                "modeBarButtonsToRemove": [
                                                                    "lasso2d",
                                                                    "select2d",
                                                                    "toggleSpikelines",
                                                                ],
                                                            },
                                                        ),
                                                    ]
                                                ),
                                                className="shell-card mb-3",
                                            ),
                                            id="graph-panel",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div(
                                                                    "Service",
                                                                    className="panel-title-small",
                                                                    id="service-panel-title",
                                                                ),
                                                                html.Div(id="service-card"),
                                                            ]
                                                        ),
                                                        className="shell-card h-100",
                                                    ),
                                                    id="service-panel",
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div(
                                                                    "Costs",
                                                                    className="panel-title-small",
                                                                ),
                                                                html.Div(id="costs-card"),
                                                            ]
                                                        ),
                                                        className="shell-card h-100",
                                                    ),
                                                    id="costs-panel",
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div(
                                                                    "Sales",
                                                                    className="panel-title-small",
                                                                ),
                                                                html.Div(id="sales-card"),
                                                            ]
                                                        ),
                                                        className="shell-card h-100",
                                                    ),
                                                    id="sales-panel",
                                                    md=4,
                                                ),
                                            ],
                                            className="g-3 mb-3 dashboard-insights-row",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div(
                                                                    "Inventory",
                                                                    className="panel-label",
                                                                ),
                                                                html.H3(
                                                                    "Planner grid",
                                                                    className="panel-title",
                                                                    id="inventory-panel-title",
                                                                ),
                                                                html.Div(
                                                                    id="inventory-table-shell"
                                                                ),
                                                            ]
                                                        ),
                                                        className="shell-card h-100",
                                                    ),
                                                    id="inventory-panel",
                                                    lg=7,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div(
                                                                    "Exceptions",
                                                                    className="panel-label",
                                                                ),
                                                                html.H3(
                                                                    "ASQ exception center",
                                                                    className="panel-title",
                                                                ),
                                                                html.Div(
                                                                    id="exception-center-shell"
                                                                ),
                                                            ]
                                                        ),
                                                        className="shell-card h-100",
                                                    ),
                                                    id="exceptions-panel",
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
                            className="mt-3 g-3 dashboard-main-layout",
                        ),
                    ],
                    id="dashboard-shell",
                    className="dashboard-shell",
                    style={"display": "none"},
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Add Items"),
                        dbc.ModalBody(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.Div(
                                                        "Manual item", className="panel-label"
                                                    ),
                                                    _number_field(
                                                        "Usage Rate",
                                                        "usage-rate-input",
                                                        min_value=1,
                                                        class_name="mb-2",
                                                    ),
                                                    _number_field(
                                                        "Lead Time",
                                                        "lead-time-input",
                                                        min_value=1,
                                                        class_name="mb-2",
                                                    ),
                                                    _number_field(
                                                        "Item Cost",
                                                        "item-cost-input",
                                                        min_value=0.01,
                                                        step=0.01,
                                                        class_name="mb-2",
                                                    ),
                                                    _number_field(
                                                        "Initial PNA",
                                                        "pna-input",
                                                        min_value=0,
                                                        step=1,
                                                        class_name="mb-2",
                                                    ),
                                                    _number_field(
                                                        "Safety %",
                                                        "safety-allowance-input",
                                                        min_value=0.01,
                                                        step=0.1,
                                                        class_name="mb-2",
                                                    ),
                                                    _number_field(
                                                        "Standard Pack",
                                                        "standard-pack-input",
                                                        min_value=1,
                                                        step=1,
                                                        class_name="mb-2",
                                                    ),
                                                    _number_field(
                                                        "Hits / Month",
                                                        "hits-per-month-input",
                                                        min_value=1,
                                                        step=1,
                                                        class_name="mb-2",
                                                    ),
                                                    html.Div(id="add-item-error", className="mb-2"),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                _action_button(
                                                                    "Randomize",
                                                                    "randomize-button",
                                                                    "secondary",
                                                                    class_name="button-block",
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                _action_button(
                                                                    "Add Item",
                                                                    "submit-item-button",
                                                                    "primary",
                                                                    class_name="button-block",
                                                                )
                                                            ),
                                                        ],
                                                        className="g-2",
                                                    ),
                                                ]
                                            ),
                                            className="shell-card h-100",
                                        ),
                                        md=6,
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.Div("Import", className="panel-label"),
                                                    dbc.Alert(
                                                        (
                                                            "Upload .csv or .xlsx files "
                                                            "using the canonical 7-column "
                                                            "item format."
                                                        ),
                                                        color="light",
                                                    ),
                                                    dcc.Upload(
                                                        id="upload-item",
                                                        children=html.Div(
                                                            [
                                                                "Drop files here or ",
                                                                html.A("browse"),
                                                            ]
                                                        ),
                                                        multiple=True,
                                                        accept=".csv,.xlsx",
                                                        className="upload-shell",
                                                    ),
                                                    html.Div(
                                                        id="output-item-upload", className="mt-3"
                                                    ),
                                                    _action_button(
                                                        "Import Items",
                                                        "import-uploaded-items",
                                                        "primary",
                                                        class_name="mt-3",
                                                    ),
                                                    html.Div(
                                                        id="upload-feedback", className="mt-3"
                                                    ),
                                                ]
                                            ),
                                            className="shell-card h-100",
                                        ),
                                        md=6,
                                    ),
                                ],
                                className="g-3",
                            )
                        ),
                    ],
                    id="add-item-modal",
                    is_open=False,
                    size="xl",
                    content_class_name="imsim-modal-content",
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Place Custom Order"),
                        dbc.ModalBody(
                            dag.AgGrid(
                                id="custom-order-grid",
                                rowData=[],
                                columnDefs=[],
                                className="ag-theme-quartz imsim-ag-grid",
                                dashGridOptions=custom_order_grid_options(),
                                style={"height": "420px", "width": "100%"},
                            )
                        ),
                        dbc.ModalFooter(
                            [
                                _action_button(
                                    "Cancel",
                                    "cancel-custom-order-button",
                                    "secondary",
                                ),
                                _action_button(
                                    "Place Order",
                                    "place-order-button",
                                    "primary",
                                ),
                            ],
                            className="modal-actions",
                        ),
                    ],
                    id="place-custom-order-modal",
                    is_open=False,
                    size="xl",
                    content_class_name="imsim-modal-content",
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("PO Overview"),
                        dbc.ModalBody(
                            dag.AgGrid(
                                id="po-overview-grid",
                                rowData=[],
                                columnDefs=[],
                                className="ag-theme-quartz imsim-ag-grid",
                                dashGridOptions=po_overview_grid_options(),
                                style={"height": "420px", "width": "100%"},
                            )
                        ),
                        dbc.ModalFooter(
                            [
                                _action_button(
                                    "Expedite Selected",
                                    "po-expedite-button",
                                    "warning",
                                ),
                                _action_button(
                                    "Cancel Selected",
                                    "po-cancel-button",
                                    "danger",
                                ),
                                _action_button(
                                    "Close",
                                    "po-overview-close",
                                    "secondary",
                                ),
                            ],
                            className="modal-actions",
                        ),
                    ],
                    id="po-overview-modal",
                    is_open=False,
                    size="xl",
                    content_class_name="imsim-modal-content",
                ),
                github_footer_card(config.github_url),
            ],
            fluid=True,
            className="imsim-shell py-4",
        ),
        id="app-theme",
        className="imsim-theme theme-light",
        **{"data-bs-theme": "light"},
    )
