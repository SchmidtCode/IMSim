from __future__ import annotations

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html

from .components import custom_order_grid_options, po_overview_grid_options
from .primitives import action_button, modal_actions, number_field, shell_card, work_modal

_MANUAL_ITEM_FIELDS = (
    ("Usage Rate", "usage-rate-input", {"min_value": 1}),
    ("Lead Time", "lead-time-input", {"min_value": 1}),
    ("Item Cost", "item-cost-input", {"min_value": 0.01, "step": 0.01}),
    ("Initial PNA", "pna-input", {"min_value": 0, "step": 1}),
    ("Safety %", "safety-allowance-input", {"min_value": 0.01, "step": 0.1}),
    ("Standard Pack", "standard-pack-input", {"min_value": 1, "step": 1}),
    ("Hits / Month", "hits-per-month-input", {"min_value": 1, "step": 1}),
)


def _number_fields() -> list[html.Div]:
    return [
        number_field(label, component_id, class_name="mb-2", **props)
        for label, component_id, props in _MANUAL_ITEM_FIELDS
    ]


def _button_row(*buttons: html.Button) -> dbc.Row:
    return dbc.Row([dbc.Col(button) for button in buttons], className="g-2")


def _manual_item_card() -> dbc.Card:
    return shell_card(
        [
            html.Div("Manual item", className="panel-label"),
            *_number_fields(),
            html.Div(id="add-item-error", className="mb-2"),
            _button_row(
                action_button(
                    "Randomize",
                    "randomize-button",
                    "secondary",
                    class_name="button-block",
                ),
                action_button(
                    "Add Item",
                    "submit-item-button",
                    "primary",
                    class_name="button-block",
                ),
            ),
        ],
        class_name="h-100",
    )


def _import_card() -> dbc.Card:
    return shell_card(
        [
            html.Div("Import", className="panel-label"),
            dbc.Alert(
                "Upload .csv or .xlsx files using the canonical 7-column item format.",
                color="light",
            ),
            dcc.Upload(
                id="upload-item",
                children=html.Div(["Drop files here or ", html.A("browse")]),
                multiple=True,
                accept=".csv,.xlsx",
                className="upload-shell",
            ),
            html.Div(id="output-item-upload", className="mt-3"),
            action_button(
                "Import Items",
                "import-uploaded-items",
                "primary",
                class_name="mt-3",
            ),
            html.Div(id="upload-feedback", className="mt-3"),
        ],
        class_name="h-100",
    )


def _grid(grid_id: str, options: dict) -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id,
        rowData=[],
        columnDefs=[],
        className="ag-theme-quartz imsim-ag-grid",
        dashGridOptions=options,
        style={"height": "420px", "width": "100%"},
    )


def _grid_modal(
    *,
    title: str,
    component_id: str,
    grid_id: str,
    options: dict,
    footer_buttons: list[html.Button],
) -> dbc.Modal:
    return work_modal(
        title,
        component_id,
        _grid(grid_id, options),
        footer=modal_actions(footer_buttons),
        size="xl",
    )


def add_item_modal() -> dbc.Modal:
    return work_modal(
        "Add Items",
        "add-item-modal",
        dbc.Row(
            [
                dbc.Col(_manual_item_card(), md=6),
                dbc.Col(_import_card(), md=6),
            ],
            className="g-3",
        ),
        size="xl",
    )


def custom_order_modal() -> dbc.Modal:
    return _grid_modal(
        title="Place Custom Order",
        component_id="place-custom-order-modal",
        grid_id="custom-order-grid",
        options=custom_order_grid_options(),
        footer_buttons=[
            action_button("Cancel", "cancel-custom-order-button", "secondary"),
            action_button("Place Order", "place-order-button", "primary"),
        ],
    )


def po_overview_modal() -> dbc.Modal:
    return _grid_modal(
        title="PO Overview",
        component_id="po-overview-modal",
        grid_id="po-overview-grid",
        options=po_overview_grid_options(),
        footer_buttons=[
            action_button("Expedite Selected", "po-expedite-button", "warning"),
            action_button("Cancel Selected", "po-cancel-button", "danger"),
            action_button("Close", "po-overview-close", "secondary"),
        ],
    )
