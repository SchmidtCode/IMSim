from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from ..config import IMSimConfig
from .components import github_footer_card


def build_layout(config: IMSimConfig):
    return html.Div(
        dbc.Container(
            [
                dcc.Store(id="user-data-store", storage_type="local", data={}),
                dcc.Store(id="page-load", data=0),
                dcc.Store(id="gh-footer-store", storage_type="local", data=True),
                dcc.Store(id="upload-preview-data"),
                dcc.Store(id="theme-store", storage_type="local", data="light"),
                dcc.Interval(id="interval-component", interval=1000, disabled=True),
                dcc.Interval(id="shutdown-poll", interval=1000, n_intervals=0),
                html.Div(id="maintenance-banner"),
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            "Inventory Management Simulator",
                                            className="hero-kicker",
                                        ),
                                        html.H1("IMSim control deck", className="hero-title"),
                                        html.P(
                                            (
                                                "Reorder policy controls, signal-map review, "
                                                "and exception handling in one simulator."
                                            ),
                                            className="hero-copy",
                                        ),
                                    ],
                                    lg=8,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Start Simulation",
                                                id="start-button",
                                                color="success",
                                                className="me-2",
                                            ),
                                            dbc.Button(
                                                "Reset",
                                                id="reset-button",
                                                color="secondary",
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
                    className="shell-card hero-banner-card mb-4",
                ),
                html.Div(id="kpi-strip", className="mt-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div("Command rail", className="panel-label"),
                                            html.H3("Controls", className="panel-title"),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Place SOQ Order",
                                                        id="po-button",
                                                        color="primary",
                                                        className="w-100 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "Custom Order",
                                                        id="place-custom-order-button",
                                                        color="light",
                                                        className="w-100 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "PO Overview",
                                                        id="po-overview-button",
                                                        color="dark",
                                                        className="w-100 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "Add or Import Items",
                                                        id="add-item-button",
                                                        color="warning",
                                                        className="w-100",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    className="shell-card mb-3",
                                ),
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div("Policy", className="panel-label"),
                                            html.H3("Global parameters", className="panel-title"),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Review Cycle"),
                                                    dbc.Input(
                                                        id="review-cycle-input",
                                                        type="number",
                                                        value=14,
                                                        min=1,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("R-Cost"),
                                                    dbc.Input(
                                                        id="r-cost-input",
                                                        type="number",
                                                        value=8,
                                                        min=0,
                                                        step=0.01,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("K-Cost %"),
                                                    dbc.Input(
                                                        id="k-cost-input",
                                                        type="number",
                                                        value=18,
                                                        min=0,
                                                        step=0.1,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Stockout $/unit"),
                                                    dbc.Input(
                                                        id="stockout-penalty-input",
                                                        type="number",
                                                        value=5,
                                                        min=0,
                                                        step=0.01,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Expedite %/day"),
                                                    dbc.Input(
                                                        id="expedite-rate-input",
                                                        type="number",
                                                        value=3,
                                                        min=0,
                                                        step=0.1,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("GM %"),
                                                    dbc.Input(
                                                        id="gm-input",
                                                        type="number",
                                                        value=15,
                                                        min=0,
                                                        max=99,
                                                        step=0.1,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Realization %"),
                                                    dbc.Input(
                                                        id="realization-input",
                                                        type="number",
                                                        value=100,
                                                        min=50,
                                                        max=100,
                                                        step=0.1,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.Switch(
                                                id="auto-po-enabled",
                                                label="Auto purchase orders",
                                                value=False,
                                                className="mb-3",
                                            ),
                                            dbc.Button(
                                                "ASQ Adjuster Settings",
                                                id="toggle-asq-collapse",
                                                color="secondary",
                                                className="w-100 mb-2",
                                            ),
                                            dbc.Collapse(
                                                html.Div(
                                                    [
                                                        dbc.Switch(
                                                            id="asq-enabled",
                                                            label="Enable ASQ OP adjuster",
                                                            value=True,
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Min Hits"),
                                                                dbc.Input(
                                                                    id="asq-min-hits",
                                                                    type="number",
                                                                    value=3,
                                                                    min=0,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Max $ Diff"),
                                                                dbc.Input(
                                                                    id="asq-max-diff",
                                                                    type="number",
                                                                    value=2500,
                                                                    min=0,
                                                                    step=0.01,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Period Days"),
                                                                dbc.Input(
                                                                    id="asq-period-days",
                                                                    type="number",
                                                                    value=30,
                                                                    min=1,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.Switch(
                                                            id="asq-include-transfers",
                                                            label="Include transfers in ASQ",
                                                            value=False,
                                                            className="mb-2",
                                                        ),
                                                        dbc.Button(
                                                            "Apply ASQ Now",
                                                            id="apply-asq-button",
                                                            color="info",
                                                            className="w-100",
                                                        ),
                                                    ]
                                                ),
                                                id="asq-collapse",
                                                is_open=False,
                                            ),
                                            html.Div(id="update-params-conf", className="mt-3"),
                                            dbc.Button(
                                                "Update Parameters",
                                                id="update-params-button",
                                                color="primary",
                                                className="w-100 mt-2",
                                            ),
                                        ]
                                    ),
                                    className="shell-card",
                                ),
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.Div("Session", className="panel-label"),
                                                    dbc.Button(
                                                        "Dark mode",
                                                        id="theme-toggle",
                                                        color="secondary",
                                                        outline=True,
                                                        size="sm",
                                                        className="theme-toggle-button",
                                                    ),
                                                ],
                                                className="session-card-header",
                                            ),
                                            html.Div(
                                                "Day: 1", id="day-display", className="status-value"
                                            ),
                                            html.Div(
                                                "Status: Paused",
                                                id="sim-status",
                                                className="status-copy",
                                            ),
                                            dbc.Label("Simulation speed", className="mt-3"),
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
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                            html.Div(id="asq-apply-feedback", className="mt-3"),
                                        ]
                                    ),
                                    className="shell-card session-card mt-3",
                                ),
                            ],
                            xl=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div("Visuals", className="panel-label"),
                                            html.H3(
                                                "Inventory signal map", className="panel-title"
                                            ),
                                            dcc.Graph(id="inventory-graph", figure={}),
                                        ]
                                    ),
                                    className="shell-card mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            "Service", className="panel-title-small"
                                                        ),
                                                        html.Div(id="service-card"),
                                                    ]
                                                ),
                                                className="shell-card h-100",
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            "Costs", className="panel-title-small"
                                                        ),
                                                        html.Div(id="costs-card"),
                                                    ]
                                                ),
                                                className="shell-card h-100",
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            "Sales", className="panel-title-small"
                                                        ),
                                                        html.Div(id="sales-card"),
                                                    ]
                                                ),
                                                className="shell-card h-100",
                                            ),
                                            md=4,
                                        ),
                                    ],
                                    className="g-3 mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            "Inventory", className="panel-label"
                                                        ),
                                                        html.H3(
                                                            "Planner grid", className="panel-title"
                                                        ),
                                                        html.Div(id="inventory-table-shell"),
                                                    ]
                                                ),
                                                className="shell-card h-100",
                                            ),
                                            lg=7,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            "Exceptions", className="panel-label"
                                                        ),
                                                        html.H3(
                                                            "ASQ exception center",
                                                            className="panel-title",
                                                        ),
                                                        html.Div(id="exception-center-shell"),
                                                    ]
                                                ),
                                                className="shell-card h-100",
                                            ),
                                            lg=5,
                                        ),
                                    ],
                                    className="g-3",
                                ),
                            ],
                            xl=8,
                        ),
                    ],
                    className="mt-4 g-4",
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
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("Usage Rate"),
                                                            dbc.Input(
                                                                id="usage-rate-input",
                                                                type="number",
                                                                min=1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("Lead Time"),
                                                            dbc.Input(
                                                                id="lead-time-input",
                                                                type="number",
                                                                min=1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("Item Cost"),
                                                            dbc.Input(
                                                                id="item-cost-input",
                                                                type="number",
                                                                min=0.01,
                                                                step=0.01,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("Initial PNA"),
                                                            dbc.Input(
                                                                id="pna-input",
                                                                type="number",
                                                                min=0,
                                                                step=1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("Safety %"),
                                                            dbc.Input(
                                                                id="safety-allowance-input",
                                                                type="number",
                                                                min=0.01,
                                                                step=0.1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("Standard Pack"),
                                                            dbc.Input(
                                                                id="standard-pack-input",
                                                                type="number",
                                                                min=1,
                                                                step=1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("Hits / Month"),
                                                            dbc.Input(
                                                                id="hits-per-month-input",
                                                                type="number",
                                                                min=1,
                                                                step=1,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    html.Div(id="add-item-error", className="mb-2"),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    "Randomize",
                                                                    id="randomize-button",
                                                                    color="secondary",
                                                                    className="w-100",
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    "Add Item",
                                                                    id="submit-item-button",
                                                                    color="primary",
                                                                    className="w-100",
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
                                                    dbc.Button(
                                                        "Import Items",
                                                        id="import-uploaded-items",
                                                        color="primary",
                                                        className="mt-3",
                                                        style={"display": "none"},
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
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Place Custom Order"),
                        dbc.ModalBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(html.Strong("Item"), width=1),
                                        dbc.Col(html.Strong("ATS")),
                                        dbc.Col(html.Strong("On-Order")),
                                        dbc.Col(html.Strong("Backorder")),
                                        dbc.Col(html.Strong("Usage")),
                                        dbc.Col(html.Strong("Lead Time")),
                                        dbc.Col(html.Strong("OP")),
                                        dbc.Col(html.Strong("LP")),
                                        dbc.Col(html.Strong("OQ")),
                                        dbc.Col(html.Strong("Order Qty"), width=2),
                                    ],
                                    className="custom-order-head",
                                ),
                                html.Div(id="custom-order-items-div"),
                            ]
                        ),
                        dbc.ModalFooter(
                            [
                                dbc.Button(
                                    "Cancel", id="cancel-custom-order-button", color="secondary"
                                ),
                                dbc.Button("Place Order", id="place-order-button", color="primary"),
                            ]
                        ),
                    ],
                    id="place-custom-order-modal",
                    is_open=False,
                    size="xl",
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("PO Overview"),
                        dbc.ModalBody(html.Div(id="po-overview-table")),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="po-overview-close", color="secondary")
                        ),
                    ],
                    id="po-overview-modal",
                    is_open=False,
                    size="xl",
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
