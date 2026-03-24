from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

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


def build_inventory_figure(state: SimulationState, theme: str = "light") -> go.Figure:
    colors = _figure_theme(theme)
    level = active_level(state)
    if level is not None and level.index == 1:
        history = state.history or []
        fig = go.Figure()
        fig.update_layout(
            **_plot_base_layout("On-hand inventory over time", colors),
            xaxis_title="Day",
            yaxis_title="Units",
        )
        if history:
            days = [point.day for point in history]
            on_hand = [point.total_on_hand for point in history]
            backorder = [point.total_backorder for point in history]
            fig.add_scatter(
                x=days,
                y=on_hand,
                mode="lines+markers",
                name="On Hand",
                line=_plot_line(colors["pna"]),
                marker=_plot_marker(colors["pna"], _PLOT_MARKER_SIZE_REM["lesson"]),
                hovertemplate="Day %{x}<br>On hand %{y:.1f}<extra></extra>",
            )
            fig.add_scatter(
                x=days,
                y=backorder,
                mode="lines+markers",
                name="Backorder",
                line=_plot_line(colors["zero"], dash="dash"),
                marker=_plot_marker(colors["zero"], _PLOT_MARKER_SIZE_REM["lesson_backorder"]),
                hovertemplate="Day %{x}<br>Backorder %{y:.1f}<extra></extra>",
            )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color=colors["guide"],
            annotation_text="Zero on hand",
            annotation_font_color=colors["text"],
        )
        fig.update_xaxes(
            tickmode="linear",
            dtick=1,
            color=colors["text"],
            gridcolor=colors["line"],
            linecolor=colors["line"],
            title_font={"color": colors["text"]},
            tickfont={"color": colors["text"]},
        )
        fig.update_yaxes(
            color=colors["text"],
            gridcolor=colors["line"],
            linecolor=colors["line"],
            title_font={"color": colors["text"]},
            tickfont={"color": colors["text"]},
            rangemode="tozero",
        )
        return fig
    if level is not None and level.index == 2:
        history = state.history or []
        fig = go.Figure()
        fig.update_layout(
            **_plot_base_layout("Basic reorder quantities over time", colors),
            xaxis_title="Day",
            yaxis_title="Units",
        )
        if history:
            days = [point.day for point in history]
            fig.add_scatter(
                x=days,
                y=[point.total_on_hand for point in history],
                mode="lines+markers",
                name="On Hand",
                line=_plot_line(colors["pna"]),
                marker=_plot_marker(colors["pna"], _PLOT_MARKER_SIZE_REM["lesson"]),
                hovertemplate="Day %{x}<br>On hand %{y:.1f}<extra></extra>",
            )
            fig.add_scatter(
                x=days,
                y=[point.total_on_order for point in history],
                mode="lines+markers",
                name="On Order",
                line=_plot_line(colors["ats"]),
                marker=_plot_marker(colors["ats"], _PLOT_MARKER_SIZE_REM["lesson"]),
                hovertemplate="Day %{x}<br>On order %{y:.1f}<extra></extra>",
            )
            fig.add_scatter(
                x=days,
                y=[point.total_pna for point in history],
                mode="lines+markers",
                name="PNA",
                line=_plot_line(colors["proposed"]),
                marker=_plot_marker(colors["proposed"], _PLOT_MARKER_SIZE_REM["lesson"]),
                hovertemplate="Day %{x}<br>PNA %{y:.1f}<extra></extra>",
            )
            fig.add_scatter(
                x=days,
                y=[point.total_backorder for point in history],
                mode="lines+markers",
                name="Backorder",
                line=_plot_line(colors["zero"], dash="dash"),
                marker=_plot_marker(colors["zero"], _PLOT_MARKER_SIZE_REM["lesson_backorder"]),
                hovertemplate="Day %{x}<br>Backorder %{y:.1f}<extra></extra>",
            )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color=colors["guide"],
            annotation_text="Zero",
            annotation_font_color=colors["text"],
        )
        fig.update_xaxes(
            tickmode="linear",
            dtick=1,
            color=colors["text"],
            gridcolor=colors["line"],
            linecolor=colors["line"],
            title_font={"color": colors["text"]},
            tickfont={"color": colors["text"]},
        )
        fig.update_yaxes(
            color=colors["text"],
            gridcolor=colors["line"],
            linecolor=colors["line"],
            title_font={"color": colors["text"]},
            tickfont={"color": colors["text"]},
            rangemode="tozero",
        )
        return fig
    if not state.items:
        fig = go.Figure()
        fig.update_layout(
            **_plot_base_layout("Inventory Signal Map", colors, include_margin=False),
            xaxis_title="Item",
            yaxis_title="Days from OP",
        )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color=colors["guide"],
            annotation_text="OP",
            annotation_font_color=colors["text"],
        )
        fig.add_hline(
            y=state.global_settings.r_cycle,
            line_dash="dot",
            line_color=colors["guide"],
            annotation_text="LP",
            annotation_font_color=colors["text"],
        )
        fig.update_xaxes(
            color=colors["text"],
            gridcolor=colors["line"],
            linecolor=colors["line"],
        )
        fig.update_yaxes(
            color=colors["text"],
            gridcolor=colors["line"],
            linecolor=colors["line"],
        )
        return fig

    df = _items_frame(state.items)
    fig = go.Figure()
    fig.update_layout(
        **_plot_base_layout("Inventory Signal Map", colors),
        xaxis_title="Item",
        yaxis_title="Days from OP",
    )
    hover = "Item %{x}<br>%{y:.1f} days<extra>%{fullData.name}</extra>"

    fig.add_scatter(
        x=df["idx"] + 1,
        y=df["pna_days_frm_op"],
        mode="markers",
        name="PNA",
        marker=_plot_marker(colors["pna"], _PLOT_MARKER_SIZE_REM["signal"]),
        hovertemplate=hover,
    )

    mask_proposed = (df["pro_pna_days_frm_op"] != df["pna_days_frm_op"]) & (df["pna"] <= df["lp"])
    if mask_proposed.any():
        dfp = df[mask_proposed]
        fig.add_scatter(
            x=dfp["idx"] + 1,
            y=dfp["pro_pna_days_frm_op"],
            mode="markers",
            name="PNA + SOQ",
            marker=_plot_marker(
                colors["proposed"],
                _PLOT_MARKER_SIZE_REM["signal_proposed"],
                symbol="circle-open",
                line=_plot_marker_outline(colors["proposed"]),
            ),
            hovertemplate=hover,
        )

    fig.add_scatter(
        x=df["idx"] + 1,
        y=df["ats_days_frm_op"],
        mode="markers",
        name="Available to Sell",
        marker=_plot_marker(
            colors["ats"],
            _PLOT_MARKER_SIZE_REM["signal_ats"],
            symbol="x",
        ),
        customdata=df["ats_days_to_stockout"],
        hovertemplate="Item %{x}<br>%{customdata:.1f} days to stockout<extra>ATS</extra>",
    )
    fig.add_scatter(
        x=df["idx"] + 1,
        y=df["no_pna_days_frm_op"],
        mode="markers",
        name="0 PNA",
        marker=_plot_marker(colors["zero"], _PLOT_MARKER_SIZE_REM["signal_zero"]),
        hovertemplate=hover,
    )

    stockouts = df[df["stockout_today"]]
    if not stockouts.empty:
        fig.add_scatter(
            x=stockouts["idx"] + 1,
            y=stockouts["pna_days_frm_op"],
            mode="markers",
            name="Stockout Today",
            marker=_plot_marker(
                colors["zero"],
                _PLOT_MARKER_SIZE_REM["signal_stockout"],
                symbol="circle-open-dot",
                line=_plot_marker_outline(colors["zero"]),
            ),
            hovertemplate=hover,
        )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=colors["guide"],
        annotation_text="OP",
        annotation_font_color=colors["text"],
    )
    fig.add_hline(
        y=state.global_settings.r_cycle,
        line_dash="dot",
        line_color=colors["guide"],
        annotation_text="LP",
        annotation_font_color=colors["text"],
    )
    fig.update_xaxes(
        tickmode="linear",
        dtick=1,
        tick0=1,
        tickformat="d",
        range=[0.5, len(df) + 0.5],
        color=colors["text"],
        gridcolor=colors["line"],
        linecolor=colors["line"],
        title_font={"color": colors["text"]},
        tickfont={"color": colors["text"]},
    )
    fig.update_yaxes(
        color=colors["text"],
        gridcolor=colors["line"],
        linecolor=colors["line"],
        title_font={"color": colors["text"]},
        tickfont={"color": colors["text"]},
        zerolinecolor=colors["guide"],
    )
    return fig


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


def build_po_overview_table(state: SimulationState) -> list:
    rows = []
    for item_index, item in enumerate(state.items):
        for receipt in item.pipeline:
            days_left = max(0, receipt.eta_day - state.day)
            rows.append(
                dbc.Row(
                    [
                        dbc.Col(item_index + 1, width=1),
                        dbc.Col(receipt.receipt_id, width=2),
                        dbc.Col(int(receipt.qty), width=2),
                        dbc.Col(int(receipt.eta_day), width=2),
                        dbc.Col(days_left, width=2),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Button(
                                        "Expedite -1d",
                                        id={"type": "po-expedite", "rid": receipt.receipt_id},
                                        n_clicks=0,
                                        className="imsim-button button-warning button-sm",
                                    ),
                                    html.Button(
                                        "Cancel",
                                        id={"type": "po-cancel", "rid": receipt.receipt_id},
                                        n_clicks=0,
                                        className="imsim-button button-danger button-sm",
                                    ),
                                ],
                                className="inline-button-group",
                            ),
                            width=3,
                        ),
                    ],
                    className="po-row",
                )
            )
    header = dbc.Row(
        [
            dbc.Col(html.Strong("Item"), width=1),
            dbc.Col(html.Strong("Receipt"), width=2),
            dbc.Col(html.Strong("Qty"), width=2),
            dbc.Col(html.Strong("ETA"), width=2),
            dbc.Col(html.Strong("Days Left"), width=2),
            dbc.Col(html.Strong("Actions"), width=3),
        ],
        className="po-header",
    )
    return [header] + rows if rows else [dbc.Alert("No open purchase orders.", color="secondary")]


def build_custom_order_row(index: int, item: InventoryItem):
    return dbc.Row(
        [
            dbc.Col(index + 1, width=1),
            dbc.Col(int(item.on_hand)),
            dbc.Col(int(item_on_order(item))),
            dbc.Col(int(item.backorder)),
            dbc.Col(round(item.usage_rate)),
            dbc.Col(round(item.lead_time)),
            dbc.Col(round(item.op)),
            dbc.Col(round(item.lp)),
            dbc.Col(round(item.oq)),
            dbc.Col(
                dcc.Input(
                    value=int(round(item.soq)),
                    type="number",
                    min=0,
                    id={"type": "order-quantity", "index": index},
                    className="control-input control-input-compact",
                ),
                width=2,
            ),
        ],
        className="custom-order-row",
    )


def build_inventory_table(state: SimulationState):
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
    table_rows = []
    for row in rows:
        table_rows.append(
            {column_config[key][1]: row[column_config[key][0]] for key in selected_columns}
        )
    return html.Div(
        dbc.Table.from_dataframe(
            pd.DataFrame(table_rows),
            striped=True,
            bordered=False,
            hover=True,
            class_name="inventory-table",
        ),
        className="table-scroll-shell",
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
