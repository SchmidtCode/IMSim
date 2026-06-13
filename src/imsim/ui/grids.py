from __future__ import annotations

import dash_ag_grid as dag
import dash_bootstrap_components as dbc

from ..models import SimulationState
from ..services.planning import item_on_order
from ..services.training import active_layout_variant, active_level, visible_columns


def _workspace_grid_height(state: SimulationState, *, surface: str) -> str:
    variant = active_layout_variant(state)
    if surface == "inventory":
        return {
            "workspace_basic": "36rem",
            "workspace_signal": "32rem",
            "workspace_advanced": "28rem",
            "workspace_certification": "26rem",
            "simulator": "28rem",
        }.get(variant, "28rem")
    return {
        "workspace_basic": "32rem",
        "workspace_signal": "30rem",
        "workspace_advanced": "26rem",
        "workspace_certification": "26rem",
        "simulator": "26rem",
    }.get(variant, "26rem")


def _grid_theme_class(theme: str) -> str:
    return (
        "ag-theme-quartz-dark imsim-ag-grid" if theme == "dark" else "ag-theme-quartz imsim-ag-grid"
    )


def po_overview_grid_options() -> dict[str, object]:
    return {
        "animateRows": False,
        "rowSelection": {
            "mode": "multiRow",
            "checkboxes": True,
            "headerCheckbox": True,
            "enableClickSelection": True,
            "enableSelectionWithoutKeys": True,
        },
        "selectionColumnDef": {
            "width": 56,
            "maxWidth": 56,
            "resizable": False,
            "sortable": False,
            "pinned": "left",
        },
    }


def custom_order_grid_options() -> dict[str, object]:
    return {
        "animateRows": False,
        "singleClickEdit": True,
        "stopEditingWhenCellsLoseFocus": True,
        "enterNavigatesVerticallyAfterEdit": True,
    }


def build_po_overview_rows(state: SimulationState) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for item_index, item in enumerate(state.items):
        for receipt in item.pipeline:
            days_left = max(0, receipt.eta_day - state.day)
            rows.append(
                {
                    "item": item_index + 1,
                    "receipt_id": receipt.receipt_id,
                    "qty": int(receipt.qty),
                    "eta_day": int(receipt.eta_day),
                    "days_left": days_left,
                }
            )
    return rows


def build_po_overview_grid(state: SimulationState, theme: str = "light") -> dag.AgGrid | dbc.Alert:
    rows = build_po_overview_rows(state)
    if not rows:
        return dbc.Alert("No open purchase orders.", color="secondary")
    return dag.AgGrid(
        id="po-overview-grid",
        rowData=rows,
        columnDefs=[
            {"field": "item", "headerName": "Item", "pinned": "left", "maxWidth": 90},
            {"field": "receipt_id", "headerName": "PO Line"},
            {"field": "qty", "headerName": "Qty", "type": "numericColumn"},
            {"field": "eta_day", "headerName": "ETA", "type": "numericColumn"},
            {"field": "days_left", "headerName": "Days Left", "type": "numericColumn"},
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className=_grid_theme_class(theme),
        columnSize="sizeToFit",
        dashGridOptions=po_overview_grid_options(),
        style={"height": _workspace_grid_height(state, surface="modal"), "width": "100%"},
    )


def build_custom_order_rows(state: SimulationState) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for index, item in enumerate(state.items, start=1):
        rows.append(
            {
                "item_index": index - 1,
                "item": index,
                "on_hand": int(item.on_hand),
                "on_order": int(item_on_order(item)),
                "backorder": int(item.backorder),
                "usage_rate": round(item.usage_rate),
                "lead_time": round(item.lead_time),
                "op": round(item.op),
                "lp": round(item.lp),
                "oq": round(item.oq),
                "order_qty": int(round(item.soq)),
            }
        )
    return rows


def build_custom_order_grid(state: SimulationState, theme: str = "light") -> dag.AgGrid | dbc.Alert:
    rows = build_custom_order_rows(state)
    if not rows:
        return dbc.Alert("No items available.", color="warning")
    return dag.AgGrid(
        id="custom-order-grid",
        rowData=rows,
        columnDefs=[
            {"field": "item", "headerName": "Item", "pinned": "left", "maxWidth": 90},
            {"field": "on_hand", "headerName": "ATS (On Hand)", "type": "numericColumn"},
            {"field": "on_order", "headerName": "On-Order", "type": "numericColumn"},
            {"field": "backorder", "headerName": "Backorder", "type": "numericColumn"},
            {"field": "usage_rate", "headerName": "Usage", "type": "numericColumn"},
            {"field": "lead_time", "headerName": "Lead Time", "type": "numericColumn"},
            {"field": "op", "headerName": "OP", "type": "numericColumn"},
            {"field": "lp", "headerName": "LP", "type": "numericColumn"},
            {"field": "oq", "headerName": "OQ", "type": "numericColumn"},
            {
                "field": "order_qty",
                "headerName": "Order Qty",
                "type": "numericColumn",
                "editable": True,
                "cellClass": "custom-order-qty-cell",
                "cellEditor": "agNumberCellEditor",
                "cellEditorParams": {
                    "min": 0,
                    "step": 1,
                    "showStepperButtons": True,
                },
                "headerTooltip": (
                    "Current suggested order quantity (SOQ); adjust before placing "
                    "the custom order."
                ),
            },
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className=_grid_theme_class(theme),
        columnSize="sizeToFit",
        dashGridOptions=custom_order_grid_options(),
        style={"height": _workspace_grid_height(state, surface="modal"), "width": "100%"},
    )


def build_inventory_table(state: SimulationState, theme: str = "light"):
    level = active_level(state)
    column_config = {
        "item": ("item", "Item"),
        "usage_rate": ("usage_rate", "Usage"),
        "hits_per_month": ("hits_per_month", "Hits"),
        "item_cost": ("item_cost", "Cost"),
        "lead_time": ("lead_time", "Lead Time"),
        "op": ("op", "OP"),
        "lp": ("lp", "LP"),
        "eoq": ("eoq", "EOQ"),
        "oq": ("oq", "OQ"),
        "pna": ("pna", "PNA"),
        "on_hand": ("on_hand", "On Hand"),
        "on_order": ("on_order", "On Order"),
        "backorder": ("backorder", "Backorder"),
        "soq": ("soq", "SOQ"),
        "standard_pack": ("standard_pack", "Pack"),
        "safety_allowance": ("safety_allowance", "Safety %"),
        "cp": ("cp", "CP"),
        "surplus_line": ("surplus_line", "Surplus threshold"),
        "days_to_op": ("days_to_op", "Days to OP"),
        "daily_usage": ("daily_usage", "Daily Usage"),
    }
    rows = []
    for index, item in enumerate(state.items, start=1):
        rows.append(
            {
                "item": index,
                "usage_rate": round(item.usage_rate, 2),
                "hits_per_month": round(item.hits_per_month, 2),
                "item_cost": round(item.item_cost, 2),
                "lead_time": round(item.lead_time, 2),
                "op": round(item.op, 2),
                "lp": round(item.lp, 2),
                "eoq": round(item.eoq, 2),
                "oq": round(item.oq, 2),
                "pna": round(item.pna, 2),
                "on_hand": round(item.on_hand, 2),
                "on_order": round(item_on_order(item), 2),
                "backorder": round(item.backorder, 2),
                "soq": round(item.soq, 2),
                "standard_pack": round(item.standard_pack, 2),
                "safety_allowance": round(item.safety_allowance * 100.0, 1),
                "cp": round(item.cp, 2),
                "surplus_line": round(item.surplus_line, 2),
                "days_to_op": round(item.ats_days_frm_op, 2),
                "daily_usage": round(item.daily_ur, 2),
            }
        )
    if not rows:
        return dbc.Alert(
            "No items loaded yet. Add an item or import a sample workbook.", color="secondary"
        )
    selected_columns = visible_columns(state) or tuple(column_config.keys())
    compact_lesson_items = level is not None
    column_widths = {
        "item": {"minWidth": 72, "maxWidth": 90},
        "on_hand": {"minWidth": 120},
        "on_order": {"minWidth": 124},
        "backorder": {"minWidth": 128},
        "pna": {"minWidth": 108},
        "op": {"minWidth": 96},
        "lp": {"minWidth": 96},
        "oq": {"minWidth": 96},
        "eoq": {"minWidth": 96},
        "soq": {"minWidth": 96},
        "usage_rate": {"minWidth": 112},
        "hits_per_month": {"minWidth": 96},
        "item_cost": {"minWidth": 96},
        "lead_time": {"minWidth": 120},
        "standard_pack": {"minWidth": 112},
        "safety_allowance": {"minWidth": 118},
        "cp": {"minWidth": 96},
        "surplus_line": {"minWidth": 168},
        "days_to_op": {"minWidth": 132},
        "daily_usage": {"minWidth": 128},
    }
    column_defs = []
    for key in selected_columns:
        field, header = column_config[key]
        column_def: dict[str, object] = {"field": field, "headerName": header}
        if key == "soq":
            column_def["headerTooltip"] = "Suggested order quantity."
        if key == "item":
            column_def["pinned"] = "left"
        else:
            column_def["type"] = "numericColumn"
        column_def.update(column_widths.get(key, {}))
        column_defs.append(column_def)
    default_col_def = {"sortable": True, "filter": True, "resizable": True}
    dash_grid_options = {
        "pagination": True,
        "paginationPageSize": min(12, max(1, len(rows))),
        "paginationPageSizeSelector": False,
        "animateRows": False,
    }
    grid_style = {"height": _workspace_grid_height(state, surface="inventory"), "width": "100%"}
    if compact_lesson_items:
        default_col_def.update({"sortable": False, "filter": False})
        dash_grid_options = {
            "animateRows": False,
            "domLayout": "autoHeight",
            "pagination": False,
        }
        min_height = "7.5rem"
        if level is not None and level.index in (2, 3, 6):
            min_height = "5rem"
        grid_style = {"height": "auto", "minHeight": min_height, "width": "100%"}
    return dag.AgGrid(
        id="inventory-table-grid",
        rowData=rows,
        columnDefs=column_defs,
        defaultColDef=default_col_def,
        className=_grid_theme_class(theme),
        columnSize="sizeToFit",
        dashGridOptions=dash_grid_options,
        style=grid_style,
    )
