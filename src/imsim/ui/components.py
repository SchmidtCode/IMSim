from __future__ import annotations

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Patch, html

from ..models import InventoryItem, SimulationState
from ..services.planning import format_money, item_on_order
from ..services.training import (
    academy_level_status,
    academy_levels,
    active_level,
    after_overhead_pct,
    evaluate_active_lesson,
    fill_rate,
    lesson_days_remaining,
    visible_columns,
)


def _items_frame(items: list[InventoryItem]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    return pd.DataFrame([item.to_dict() for item in items]).reset_index(names="idx")


def _figure_theme(theme: str) -> dict[str, str]:
    if theme == "dark":
        return {
            "plot_bg": "rgba(13, 24, 38, 0.94)",
            "surface": "#17273d",
            "text": "#e6eefc",
            "muted": "#9aabc6",
            "line": "rgba(214, 228, 255, 0.14)",
            "guide": "rgba(148, 163, 184, 0.7)",
            "pna": "#2dd4bf",
            "proposed": "#fb923c",
            "ats": "#60a5fa",
            "zero": "#f87171",
        }
    return {
        "plot_bg": "rgba(255,255,255,0.55)",
        "surface": "#fffdf8",
        "text": "#132238",
        "muted": "#536277",
        "line": "rgba(19, 34, 56, 0.12)",
        "guide": "rgba(19, 34, 56, 0.55)",
        "pna": "#0f766e",
        "proposed": "#f97316",
        "ats": "#2563eb",
        "zero": "#dc2626",
    }


_ROOT_FONT_SIZE_PX = 16
_PLOT_TITLE_SIZE_REM = 1.5
_PLOT_MARGIN_REM = {"l": 1.5, "r": 1.5, "t": 3.5, "b": 1.5}
_PLOT_LINE_WIDTH_REM = 0.1875
_PLOT_MARKER_OUTLINE_WIDTH_REM = 0.125
_PLOT_MARKER_SIZE_REM = {
    "lesson": 0.5,
    "lesson_backorder": 0.4375,
    "signal": 0.875,
    "signal_proposed": 0.625,
    "signal_ats": 0.5625,
    "signal_zero": 0.5,
    "signal_stockout": 1.125,
}


def _rem_to_px(rem: float) -> float:
    return rem * _ROOT_FONT_SIZE_PX


def _plot_base_layout(
    title: str,
    colors: dict[str, str],
    *,
    include_margin: bool = True,
) -> dict[str, object]:
    layout: dict[str, object] = {
        "title": {
            "text": title,
            "font": {"color": colors["text"], "size": _rem_to_px(_PLOT_TITLE_SIZE_REM)},
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": colors["plot_bg"],
        "font": {"color": colors["text"]},
        "legend_title_text": "",
        "legend": {"font": {"color": colors["text"]}},
        "hoverlabel": {
            "bgcolor": colors["surface"],
            "bordercolor": colors["line"],
            "font": {"color": colors["text"]},
        },
    }
    if include_margin:
        layout["margin"] = {side: _rem_to_px(size) for side, size in _PLOT_MARGIN_REM.items()}
    return layout


def _plot_line(color: str, *, dash: str | None = None) -> dict[str, object]:
    line: dict[str, object] = {"width": _rem_to_px(_PLOT_LINE_WIDTH_REM), "color": color}
    if dash:
        line["dash"] = dash
    return line


def _plot_marker_outline(color: str) -> dict[str, object]:
    return {"width": _rem_to_px(_PLOT_MARKER_OUTLINE_WIDTH_REM), "color": color}


def _plot_marker(
    color: str,
    size_rem: float,
    *,
    symbol: str | None = None,
    line: dict[str, object] | None = None,
) -> dict[str, object]:
    marker: dict[str, object] = {"size": _rem_to_px(size_rem), "color": color}
    if symbol:
        marker["symbol"] = symbol
    if line:
        marker["line"] = line
    return marker


def _lesson_snapshot_table(
    label: str, headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="lesson-snapshot-label"),
            html.Table(
                [
                    html.Thead(html.Tr([html.Th(header) for header in headers])),
                    html.Tbody([html.Tr([html.Td(value) for value in row]) for row in rows]),
                ],
                className="lesson-service-table",
            ),
        ],
        className="lesson-snapshot-block",
    )


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _compact_lesson_pill_text(row: str) -> str:
    replacements = (
        ("Current fill rate: ", "Fill rate: "),
        ("Current after-overhead GM: ", "After-OH GM: "),
        ("Items closing at/above OP: ", "At/above OP: "),
        ("Guided reorders used: ", "Guided reorders: "),
        ("Manual custom orders used: ", "Custom orders: "),
        ("Avg inventory value: ", "Avg inv value: "),
    )
    for old, new in replacements:
        if row.startswith(old):
            return row.replace(old, new, 1)
    return row


def _lesson_item_snapshot_columns(level_index: int) -> tuple[str, ...]:
    return {
        1: ("item", "on_hand", "daily_usage", "backorder"),
        2: ("item", "on_hand", "on_order", "backorder", "pna"),
        3: ("item", "on_hand", "usage_rate", "op", "days_to_op"),
        4: ("item", "on_hand", "on_order", "backorder", "soq"),
        5: ("item", "on_hand", "on_order", "backorder", "pna"),
        6: ("item", "pna", "op", "lp", "soq"),
        7: ("item", "pna", "op", "lp", "soq"),
    }.get(level_index, ("item", "on_hand", "on_order", "backorder"))


def _lesson_item_snapshot_value(column: str, index: int, item: InventoryItem) -> str:
    if column == "item":
        return str(index)
    if column == "on_hand":
        return f"{item.on_hand:.1f}"
    if column == "daily_usage":
        return f"{item.daily_ur:.1f}"
    if column == "usage_rate":
        return f"{item.usage_rate:.1f}"
    if column == "on_order":
        return f"{item_on_order(item):.1f}"
    if column == "backorder":
        return f"{item.backorder:.1f}"
    if column == "pna":
        return f"{item.pna:.1f}"
    if column == "op":
        return f"{item.op:.1f}"
    if column == "lp":
        return f"{item.lp:.1f}"
    if column == "soq":
        return f"{item.soq:.1f}"
    if column == "days_to_op":
        return f"{item.ats_days_frm_op:.1f}"
    raise ValueError(f"Unsupported lesson snapshot column: {column}")


def _lesson_item_snapshot_block(level_index: int, items: list[InventoryItem]) -> html.Div:
    labels = {
        "item": "Item",
        "on_hand": "On Hand",
        "daily_usage": "Daily Usage",
        "usage_rate": "Usage",
        "on_order": "On Order",
        "backorder": "Backorder",
        "pna": "PNA",
        "op": "OP",
        "lp": "LP",
        "soq": "SOQ",
        "days_to_op": "Days to OP",
    }
    columns = _lesson_item_snapshot_columns(level_index)
    return _lesson_snapshot_table(
        "Inventory",
        tuple(labels[column] for column in columns),
        tuple(
            tuple(_lesson_item_snapshot_value(column, index, item) for column in columns)
            for index, item in enumerate(items, start=1)
        ),
    )


_FIGURE_KIND_KEY = "figure_kind"
_FIGURE_THEME_KEY = "theme"


def _grid_theme_class(theme: str) -> str:
    return (
        "ag-theme-quartz-dark imsim-ag-grid" if theme == "dark" else "ag-theme-quartz imsim-ag-grid"
    )


def _figure_meta(kind: str, theme: str) -> dict[str, str]:
    return {
        _FIGURE_KIND_KEY: kind,
        _FIGURE_THEME_KEY: theme,
    }


def _stable_history_points(state: SimulationState) -> list:
    return state.history or []


def _empty_marker_trace(
    *,
    name: str,
    marker: dict[str, object],
    hovertemplate: str,
    customdata: list | None = None,
) -> go.Scatter:
    trace = go.Scatter(
        x=[],
        y=[],
        mode="markers",
        name=name,
        marker=marker,
        hovertemplate=hovertemplate,
    )
    if customdata is not None:
        trace.customdata = customdata
    return trace


def _empty_line_trace(
    *,
    name: str,
    line: dict[str, object],
    marker: dict[str, object],
    hovertemplate: str,
) -> go.Scatter:
    return go.Scatter(
        x=[],
        y=[],
        mode="lines+markers",
        name=name,
        line=line,
        marker=marker,
        hovertemplate=hovertemplate,
    )


def _finalize_axes(
    fig: go.Figure,
    colors: dict[str, str],
    *,
    x_linear: bool = False,
    x_range: list[float] | None = None,
    y_tozero: bool = False,
) -> go.Figure:
    xaxis: dict[str, object] = {
        "color": colors["text"],
        "gridcolor": colors["line"],
        "linecolor": colors["line"],
        "title_font": {"color": colors["text"]},
        "tickfont": {"color": colors["text"]},
    }
    if x_linear:
        xaxis.update({"tickmode": "linear", "dtick": 1, "tick0": 1})
    if x_range is not None:
        xaxis["range"] = x_range
        xaxis["tickformat"] = "d"
    yaxis: dict[str, object] = {
        "color": colors["text"],
        "gridcolor": colors["line"],
        "linecolor": colors["line"],
        "title_font": {"color": colors["text"]},
        "tickfont": {"color": colors["text"]},
    }
    if y_tozero:
        yaxis["rangemode"] = "tozero"
    else:
        yaxis["zerolinecolor"] = colors["guide"]
    fig.update_xaxes(**xaxis)
    fig.update_yaxes(**yaxis)
    return fig


def _apply_signal_guides(
    fig: go.Figure,
    colors: dict[str, str],
    *,
    lower_label: str,
    upper_label: str,
    upper_y: float,
) -> go.Figure:
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=colors["guide"],
        annotation_text=lower_label,
        annotation_font_color=colors["text"],
    )
    fig.add_hline(
        y=upper_y,
        line_dash="dot",
        line_color=colors["guide"],
        annotation_text=upper_label,
        annotation_font_color=colors["text"],
    )
    return fig


def _apply_single_guide(
    fig: go.Figure,
    colors: dict[str, str],
    *,
    y: float,
    label: str,
) -> go.Figure:
    fig.add_hline(
        y=y,
        line_dash="dot",
        line_color=colors["guide"],
        annotation_text=label,
        annotation_font_color=colors["text"],
    )
    return fig


def _lesson_one_figure(state: SimulationState, theme: str, colors: dict[str, str]) -> go.Figure:
    history = _stable_history_points(state)
    days = [point.day for point in history]
    on_hand = [point.total_on_hand for point in history]
    backorder = [point.total_backorder for point in history]
    fig = go.Figure(
        data=[
            _empty_line_trace(
                name="On Hand",
                line=_plot_line(colors["pna"]),
                marker=_plot_marker(colors["pna"], _PLOT_MARKER_SIZE_REM["lesson"]),
                hovertemplate="On hand %{y:.1f}<extra></extra>",
            ),
            _empty_line_trace(
                name="Backorder",
                line=_plot_line(colors["zero"], dash="dash"),
                marker=_plot_marker(colors["zero"], _PLOT_MARKER_SIZE_REM["lesson_backorder"]),
                hovertemplate="Backorder %{y:.1f}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        **_plot_base_layout("On-hand inventory over time", colors),
        xaxis_title="Day",
        yaxis_title="Units",
        hovermode="x unified",
        uirevision=f"lesson-1:{theme}",
        meta=_figure_meta("lesson-1", theme),
    )
    fig.update_traces(x=days, selector={"name": "On Hand"})
    fig.update_traces(y=on_hand, selector={"name": "On Hand"})
    fig.update_traces(x=days, selector={"name": "Backorder"})
    fig.update_traces(y=backorder, selector={"name": "Backorder"})
    _apply_single_guide(fig, colors, y=0, label="Zero on hand")
    return _finalize_axes(fig, colors, x_linear=True, y_tozero=True)


def _lesson_two_figure(state: SimulationState, theme: str, colors: dict[str, str]) -> go.Figure:
    history = _stable_history_points(state)
    days = [point.day for point in history]
    fig = go.Figure(
        data=[
            _empty_line_trace(
                name="On Hand",
                line=_plot_line(colors["pna"]),
                marker=_plot_marker(colors["pna"], _PLOT_MARKER_SIZE_REM["lesson"]),
                hovertemplate="On hand %{y:.1f}<extra></extra>",
            ),
            _empty_line_trace(
                name="On Order",
                line=_plot_line(colors["ats"]),
                marker=_plot_marker(colors["ats"], _PLOT_MARKER_SIZE_REM["lesson"]),
                hovertemplate="On order %{y:.1f}<extra></extra>",
            ),
            _empty_line_trace(
                name="PNA",
                line=_plot_line(colors["proposed"]),
                marker=_plot_marker(colors["proposed"], _PLOT_MARKER_SIZE_REM["lesson"]),
                hovertemplate="PNA %{y:.1f}<extra></extra>",
            ),
            _empty_line_trace(
                name="Backorder",
                line=_plot_line(colors["zero"], dash="dash"),
                marker=_plot_marker(colors["zero"], _PLOT_MARKER_SIZE_REM["lesson_backorder"]),
                hovertemplate="Backorder %{y:.1f}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        **_plot_base_layout("Basic reorder quantities over time", colors),
        xaxis_title="Day",
        yaxis_title="Units",
        hovermode="x unified",
        uirevision=f"lesson-2:{theme}",
        meta=_figure_meta("lesson-2", theme),
    )
    fig.update_traces(x=days, selector={"name": "On Hand"})
    fig.update_traces(y=[point.total_on_hand for point in history], selector={"name": "On Hand"})
    fig.update_traces(x=days, selector={"name": "On Order"})
    fig.update_traces(
        y=[point.total_on_order for point in history],
        selector={"name": "On Order"},
    )
    fig.update_traces(x=days, selector={"name": "PNA"})
    fig.update_traces(y=[point.total_pna for point in history], selector={"name": "PNA"})
    fig.update_traces(x=days, selector={"name": "Backorder"})
    fig.update_traces(
        y=[point.total_backorder for point in history],
        selector={"name": "Backorder"},
    )
    _apply_single_guide(fig, colors, y=0, label="Zero")
    return _finalize_axes(fig, colors, x_linear=True, y_tozero=True)


def _signal_map_figure(state: SimulationState, theme: str, colors: dict[str, str]) -> go.Figure:
    rows = _items_frame(state.items)
    if rows.empty:
        rows = pd.DataFrame(
            columns=[
                "idx",
                "pna_days_frm_op",
                "pro_pna_days_frm_op",
                "pna",
                "lp",
                "ats_days_frm_op",
                "ats_days_to_stockout",
                "no_pna_days_frm_op",
                "stockout_today",
            ]
        )
    item_numbers = (rows["idx"] + 1).tolist() if "idx" in rows else []
    hover = "Item %{x}<br>%{y:.1f} days<extra>%{fullData.name}</extra>"
    proposed_mask = (
        (rows["pro_pna_days_frm_op"] != rows["pna_days_frm_op"]) & (rows["pna"] <= rows["lp"])
        if not rows.empty
        else pd.Series(dtype=bool)
    )
    proposed_rows = rows[proposed_mask] if not rows.empty else rows
    stockout_rows = rows[rows["stockout_today"]] if not rows.empty else rows
    fig = go.Figure(
        data=[
            _empty_marker_trace(
                name="PNA",
                marker=_plot_marker(colors["pna"], _PLOT_MARKER_SIZE_REM["signal"]),
                hovertemplate=hover,
            ),
            _empty_marker_trace(
                name="PNA + SOQ",
                marker=_plot_marker(
                    colors["proposed"],
                    _PLOT_MARKER_SIZE_REM["signal_proposed"],
                    symbol="circle-open",
                    line=_plot_marker_outline(colors["proposed"]),
                ),
                hovertemplate=hover,
            ),
            _empty_marker_trace(
                name="Available to Sell",
                marker=_plot_marker(
                    colors["ats"],
                    _PLOT_MARKER_SIZE_REM["signal_ats"],
                    symbol="x",
                ),
                hovertemplate="Item %{x}<br>%{customdata:.1f} days to stockout<extra>ATS</extra>",
                customdata=[],
            ),
            _empty_marker_trace(
                name="0 PNA",
                marker=_plot_marker(colors["zero"], _PLOT_MARKER_SIZE_REM["signal_zero"]),
                hovertemplate=hover,
            ),
            _empty_marker_trace(
                name="Stockout Today",
                marker=_plot_marker(
                    colors["zero"],
                    _PLOT_MARKER_SIZE_REM["signal_stockout"],
                    symbol="circle-open-dot",
                    line=_plot_marker_outline(colors["zero"]),
                ),
                hovertemplate=hover,
            ),
        ]
    )
    fig.update_layout(
        **_plot_base_layout(
            "Inventory Signal Map",
            colors,
            include_margin=not rows.empty,
        ),
        xaxis_title="Item",
        yaxis_title="Days from OP",
        hovermode="closest",
        uirevision=f"signal-map:{theme}",
        meta=_figure_meta("signal-map", theme),
    )
    fig.update_traces(x=item_numbers, selector={"name": "PNA"})
    fig.update_traces(y=rows["pna_days_frm_op"].tolist(), selector={"name": "PNA"})
    fig.update_traces(
        x=(proposed_rows["idx"] + 1).tolist() if not proposed_rows.empty else [],
        selector={"name": "PNA + SOQ"},
    )
    fig.update_traces(
        y=proposed_rows["pro_pna_days_frm_op"].tolist() if not proposed_rows.empty else [],
        selector={"name": "PNA + SOQ"},
    )
    fig.update_traces(x=item_numbers, selector={"name": "Available to Sell"})
    fig.update_traces(y=rows["ats_days_frm_op"].tolist(), selector={"name": "Available to Sell"})
    fig.update_traces(
        customdata=rows["ats_days_to_stockout"].tolist(),
        selector={"name": "Available to Sell"},
    )
    fig.update_traces(x=item_numbers, selector={"name": "0 PNA"})
    fig.update_traces(y=rows["no_pna_days_frm_op"].tolist(), selector={"name": "0 PNA"})
    fig.update_traces(
        x=(stockout_rows["idx"] + 1).tolist() if not stockout_rows.empty else [],
        selector={"name": "Stockout Today"},
    )
    fig.update_traces(
        y=stockout_rows["pna_days_frm_op"].tolist() if not stockout_rows.empty else [],
        selector={"name": "Stockout Today"},
    )
    _apply_signal_guides(
        fig,
        colors,
        lower_label="OP",
        upper_label="LP",
        upper_y=state.global_settings.r_cycle,
    )
    return _finalize_axes(
        fig,
        colors,
        x_linear=True,
        x_range=[0.5, max(1, len(rows)) + 0.5],
    )


def build_inventory_figure(state: SimulationState, theme: str = "light") -> go.Figure:
    colors = _figure_theme(theme)
    level = active_level(state)
    if level is not None and level.index == 1:
        return _lesson_one_figure(state, theme, colors)
    if level is not None and level.index == 2:
        return _lesson_two_figure(state, theme, colors)
    return _signal_map_figure(state, theme, colors)


def refresh_inventory_figure(
    state: SimulationState,
    theme: str = "light",
    current_figure: dict | None = None,
) -> go.Figure | Patch:
    target = build_inventory_figure(state, theme)
    layout = (current_figure or {}).get("layout") or {}
    current_meta = layout.get("meta") or {}
    target_meta = dict(target.layout.meta or {})
    if (
        current_meta.get(_FIGURE_KIND_KEY) != target_meta.get(_FIGURE_KIND_KEY)
        or current_meta.get(_FIGURE_THEME_KEY) != target_meta.get(_FIGURE_THEME_KEY)
        or len((current_figure or {}).get("data") or []) != len(target.data)
    ):
        return target

    patched = Patch()
    for index, trace in enumerate(target.data):
        patched["data"][index]["x"] = list(trace.x) if trace.x is not None else []
        patched["data"][index]["y"] = list(trace.y) if trace.y is not None else []
        if getattr(trace, "customdata", None) is not None:
            patched["data"][index]["customdata"] = list(trace.customdata)
        else:
            patched["data"][index]["customdata"] = []
        patched["data"][index]["hovertemplate"] = trace.hovertemplate
    patched["layout"]["title"] = target.layout.title.to_plotly_json()
    patched["layout"]["xaxis"] = target.layout.xaxis.to_plotly_json()
    patched["layout"]["yaxis"] = target.layout.yaxis.to_plotly_json()
    patched["layout"]["hovermode"] = target.layout.hovermode
    patched["layout"]["uirevision"] = target.layout.uirevision
    patched["layout"]["meta"] = target.layout.meta
    patched["layout"]["shapes"] = [shape.to_plotly_json() for shape in (target.layout.shapes or [])]
    patched["layout"]["annotations"] = [
        annotation.to_plotly_json() for annotation in (target.layout.annotations or [])
    ]
    return patched


def service_card_children(state: SimulationState) -> list:
    level = active_level(state)
    today = state.service_today
    totals = state.service_totals
    ats = sum(item.on_hand for item in state.items)
    on_order = sum(item_on_order(item) for item in state.items)
    backorder = sum(item.backorder for item in state.items)
    fill_today = (
        None if today.orders == 0 else 100.0 * (today.orders - today.orders_stockout) / today.orders
    )
    fill_total = (
        None
        if totals.orders == 0
        else 100.0 * (totals.orders - totals.orders_stockout) / totals.orders
    )
    if level is not None and level.index == 1 and state.items:
        item = state.items[0]
        status = "Depleted" if item.on_hand <= 0 else "Available"
        return [
            html.Div(
                [
                    _lesson_snapshot_table(
                        "Service",
                        ("On Hand", "Daily Usage", "Backorder", "Status"),
                        (
                            (
                                f"{item.on_hand:.1f} units",
                                f"{item.daily_ur:.1f} units/day",
                                f"{item.backorder:.1f} units",
                                status,
                            ),
                        ),
                    ),
                    _lesson_snapshot_table(
                        "Inventory",
                        ("Item", "On Hand", "Daily Usage", "Backorder"),
                        (
                            (
                                "1",
                                f"{item.on_hand:.0f}",
                                f"{item.daily_ur:.0f}",
                                f"{item.backorder:.0f}",
                            ),
                        ),
                    ),
                ],
                className="lesson-snapshot-stack",
            )
        ]
    if level is not None and state.items:
        return [
            html.Div(
                [
                    _lesson_snapshot_table(
                        "Service",
                        ("Orders", "Stockouts", "Today Fill", "Cum Fill", "Zero ATS"),
                        (
                            (
                                str(today.orders),
                                str(today.orders_stockout),
                                _format_pct(fill_today),
                                _format_pct(fill_total),
                                str(today.zero_on_hand_hits),
                            ),
                        ),
                    ),
                    _lesson_item_snapshot_block(level.index, state.items),
                ],
                className="lesson-snapshot-stack",
            )
        ]

    return [
        dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    f"Today: Orders {today.orders} • Stockouts "
                    f"{today.orders_stockout} • Zero ATS hits "
                    f"{today.zero_on_hand_hits}"
                ),
                dbc.ListGroupItem(
                    f"Fill Rate (cumulative): {_format_pct(fill_total)}  •  "
                    f"Today: {_format_pct(fill_today)}"
                ),
                dbc.ListGroupItem(
                    dbc.Badge(
                        f"ATS {int(ats)} • On-order {int(on_order)} • Backorder {int(backorder)}",
                        color="secondary",
                        pill=True,
                    )
                ),
            ],
            flush=True,
        )
    ]


def costs_card_children(state: SimulationState) -> list:
    costs = state.costs
    total = costs.ordering + costs.holding + costs.stockout + costs.expedite
    return [
        dbc.ListGroup(
            [
                dbc.ListGroupItem(f"Inventory Overhead: {format_money(total)}"),
                dbc.ListGroupItem(
                    " • ".join(
                        [
                            f"Ordering {format_money(costs.ordering)}",
                            f"Holding {format_money(costs.holding)}",
                            f"Stockout {format_money(costs.stockout)}",
                            f"Expedite {format_money(costs.expedite)}",
                        ]
                    )
                ),
                dbc.ListGroupItem(f"Receipts Booked: {format_money(costs.purchases)}"),
            ],
            flush=True,
        )
    ]


def sales_card_children(state: SimulationState) -> list:
    sales = state.sales
    costs = state.costs
    gross = sales.revenue - sales.cogs
    gross_pct = None if sales.revenue <= 0 else 100.0 * gross / sales.revenue
    after_overhead = gross - costs.total
    after_pct = None if sales.revenue <= 0 else 100.0 * after_overhead / sales.revenue

    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f}%"

    return [
        dbc.ListGroup(
            [
                dbc.ListGroupItem(f"Sales: {format_money(sales.revenue)}"),
                dbc.ListGroupItem(
                    f"COGS {format_money(sales.cogs)} • "
                    f"GM$ {format_money(gross)} • GM% {fmt(gross_pct)}"
                ),
                dbc.ListGroupItem(
                    f"After overhead {format_money(after_overhead)} • {fmt(after_pct)}"
                ),
            ],
            flush=True,
        )
    ]


def _kpi_card(title: str, value: str, tone: str, subtitle: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="metric-label"),
                html.Div(value, className="metric-value"),
                html.Div(subtitle, className="metric-subtitle"),
            ]
        ),
        className=f"metric-card metric-{tone}",
    )


def build_kpi_strip(state: SimulationState) -> list:
    revenue = state.sales.revenue
    gross = revenue - state.sales.cogs
    avg_inventory = state.analytics.inv_value_daysum / max(1, state.day)
    fill_rate = (
        None
        if state.service_totals.orders == 0
        else (state.service_totals.orders - state.service_totals.orders_stockout)
        / state.service_totals.orders
    )
    turns = None if avg_inventory <= 0 else state.sales.cogs / avg_inventory
    gmroi = None if avg_inventory <= 0 else gross / avg_inventory
    after_overhead = gross - state.costs.total
    after_pct = None if revenue <= 0 else after_overhead / revenue

    def fmt_pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.1f}%"

    def fmt_ratio(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"

    return [
        dbc.Row(
            [
                dbc.Col(
                    _kpi_card("Day", f"{state.day}", "neutral", "Simulation clock"),
                    md=6,
                    xl=2,
                ),
                dbc.Col(
                    _kpi_card(
                        "Fill Rate",
                        fmt_pct(fill_rate),
                        "good" if (fill_rate or 0) >= 0.95 else "warn",
                        "Cumulative service",
                    ),
                    md=6,
                    xl=2,
                ),
                dbc.Col(
                    _kpi_card(
                        "After-Overhead GM",
                        fmt_pct(after_pct),
                        "good" if (after_pct or 0) >= 0.15 else "warn",
                        format_money(after_overhead),
                    ),
                    md=6,
                    xl=2,
                ),
                dbc.Col(
                    _kpi_card(
                        "Avg Inventory",
                        format_money(avg_inventory),
                        "neutral",
                        "Daily midpoint cost",
                    ),
                    md=6,
                    xl=2,
                ),
                dbc.Col(
                    _kpi_card("Turns", fmt_ratio(turns), "neutral", "COGS / avg inventory"),
                    md=6,
                    xl=2,
                ),
                dbc.Col(
                    _kpi_card("GMROI", fmt_ratio(gmroi), "neutral", "Gross margin return"),
                    md=6,
                    xl=2,
                ),
            ],
            className="g-3",
        )
    ]


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
            {"field": "receipt_id", "headerName": "Receipt"},
            {"field": "qty", "headerName": "Qty", "type": "numericColumn"},
            {"field": "eta_day", "headerName": "ETA", "type": "numericColumn"},
            {"field": "days_left", "headerName": "Days Left", "type": "numericColumn"},
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className=_grid_theme_class(theme),
        columnSize="sizeToFit",
        dashGridOptions={
            "animateRows": False,
            "rowSelection": {"mode": "singleRow"},
            "suppressRowClickSelection": False,
        },
        style={"height": "420px", "width": "100%"},
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
            {"field": "on_hand", "headerName": "ATS", "type": "numericColumn"},
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
                "cellEditor": "agNumberCellEditor",
            },
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className=_grid_theme_class(theme),
        columnSize="sizeToFit",
        dashGridOptions={"animateRows": False, "stopEditingWhenCellsLoseFocus": True},
        style={"height": "420px", "width": "100%"},
    )


def build_inventory_table(state: SimulationState, theme: str = "light"):
    column_config = {
        "item": ("item", "Item"),
        "usage_rate": ("usage_rate", "Usage"),
        "lead_time": ("lead_time", "Lead Time"),
        "op": ("op", "OP"),
        "lp": ("lp", "LP"),
        "oq": ("oq", "OQ"),
        "pna": ("pna", "PNA"),
        "on_hand": ("on_hand", "On Hand"),
        "on_order": ("on_order", "On Order"),
        "backorder": ("backorder", "Backorder"),
        "soq": ("soq", "SOQ"),
        "safety_allowance": ("safety_allowance", "Safety %"),
        "days_to_op": ("days_to_op", "Days to OP"),
        "daily_usage": ("daily_usage", "Daily Usage"),
    }
    rows = []
    for index, item in enumerate(state.items, start=1):
        rows.append(
            {
                "item": index,
                "usage_rate": round(item.usage_rate, 2),
                "lead_time": round(item.lead_time, 2),
                "op": round(item.op, 2),
                "lp": round(item.lp, 2),
                "oq": round(item.oq, 2),
                "pna": round(item.pna, 2),
                "on_hand": round(item.on_hand, 2),
                "on_order": round(item_on_order(item), 2),
                "backorder": round(item.backorder, 2),
                "soq": round(item.soq, 2),
                "safety_allowance": round(item.safety_allowance * 100.0, 1),
                "days_to_op": round(item.ats_days_frm_op, 2),
                "daily_usage": round(item.daily_ur, 2),
            }
        )
    if not rows:
        return dbc.Alert(
            "No items loaded yet. Add an item or import a sample workbook.", color="secondary"
        )
    selected_columns = visible_columns(state) or tuple(column_config.keys())
    column_defs = []
    for key in selected_columns:
        field, header = column_config[key]
        column_def: dict[str, object] = {"field": field, "headerName": header}
        if key == "item":
            column_def["pinned"] = "left"
            column_def["maxWidth"] = 90
        else:
            column_def["type"] = "numericColumn"
        column_defs.append(column_def)
    return dag.AgGrid(
        id="inventory-table-grid",
        rowData=rows,
        columnDefs=column_defs,
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className=_grid_theme_class(theme),
        columnSize="sizeToFit",
        dashGridOptions={
            "pagination": True,
            "paginationPageSize": min(12, max(1, len(rows))),
            "paginationPageSizeSelector": False,
            "animateRows": False,
        },
        style={"height": "460px", "width": "100%"},
    )


def build_exception_center(state: SimulationState):
    if not state.exception_center:
        return dbc.Alert("No ASQ exceptions recorded.", color="secondary")
    cards = []
    for exception in list(reversed(state.exception_center[-6:])):
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Strong(exception.code),
                                html.Span(f" • Day {exception.day}", className="ms-2 text-muted"),
                            ]
                        ),
                        html.Div(exception.message, className="mt-2"),
                        html.Small(
                            (
                                f"Item {exception.item_index + 1} • "
                                f"OP {exception.op:.2f} • ASQ {exception.asq:.2f}"
                            ),
                            className="text-muted",
                        ),
                    ]
                ),
                className="exception-card",
            )
        )
    return html.Div(cards, className="exception-stack")


def github_footer_card(github_url: str) -> html.Div:
    return html.Div(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div("IMSim", className="footer-title"),
                    html.Small(
                        [
                            "Project source on ",
                            html.A(
                                "GitHub",
                                href=github_url,
                                target="_blank",
                                rel="noopener noreferrer",
                            ),
                        ]
                    ),
                    " ",
                    html.Button(
                        "Hide",
                        id="gh-footer-hide",
                        n_clicks=0,
                        className="imsim-button button-link footer-hide",
                    ),
                ]
            ),
            className="footer-card",
        ),
        id="gh-footer",
    )


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
    if level.index == 2 and state.items:
        item = state.items[0]
        on_order = item_on_order(item)
        reserved = 0.0
        committed = 0.0
        received = 0.0
        return [
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
        ]
    return [
        html.Div(level.formula, className="lesson-formula-chip"),
        html.Ul([html.Li(step) for step in level.tutorial_steps], className="lesson-copy-list"),
    ]


def lesson_objective_children(state: SimulationState) -> list:
    level = active_level(state)
    if level is None:
        return [dbc.Alert("No active lesson.", color="secondary")]
    evaluation = evaluate_active_lesson(state)
    after_overhead = after_overhead_pct(state)
    after_value = "n/a" if after_overhead is None else f"{after_overhead * 100:.1f}%"
    headline = f"{lesson_days_remaining(state)} day(s) remaining"
    if evaluation is not None and evaluation.completed:
        headline = "Lesson window closed"
    rows = list(evaluation.metric_rows if evaluation is not None else ())
    if "fill_rate_min" in level.win_conditions:
        fill = fill_rate(state)
        fill_value = "n/a" if fill is None else f"{fill * 100:.1f}%"
        rows.insert(0, f"Current fill rate: {fill_value}")
    if after_overhead is not None and "after_overhead_min" in level.win_conditions:
        rows.append(f"Current after-overhead GM: {after_value}")
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
        for row in list(evaluation.metric_rows if evaluation is not None else ())[:2]
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
        html.Div("Simulator Mode", className="academy-card-title"),
        html.P(
            "The full IM dashboard with imports, ASQ, and the sandbox reward controls."
            if unlocked
            else "Pass certification to unlock the full simulator and the sandbox reward controls.",
            className="academy-card-copy",
        ),
        dbc.Badge(
            "Unlocked" if unlocked else "Locked",
            color="success" if unlocked else "secondary",
            pill=True,
            class_name="academy-status-badge",
        ),
    ]
