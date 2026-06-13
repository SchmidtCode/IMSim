from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import Patch

from ..models import InventoryItem, SimulationState
from ..services.planning import effective_review_cycle
from ..services.training import active_layout_variant, active_level, fill_rate


def _items_frame(items: list[InventoryItem]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    return pd.DataFrame([item.to_dict() for item in items]).reset_index(names="idx")


def _figure_theme(theme: str) -> dict[str, str]:
    if theme == "dark":
        return {
            "plot_bg": "rgba(10, 19, 32, 0.72)",
            "surface": "#121f31",
            "text": "#e6eefc",
            "muted": "#9aabc6",
            "line": "rgba(214, 228, 255, 0.14)",
            "guide": "rgba(148, 163, 184, 0.7)",
            "pna": "#2dd4bf",
            "proposed": "#d9904a",
            "ats": "#60a5fa",
            "zero": "#f87171",
        }
    return {
        "plot_bg": "rgba(252,253,249,0.86)",
        "surface": "#fcfdf9",
        "text": "#15231f",
        "muted": "#5b6964",
        "line": "rgba(21, 35, 31, 0.13)",
        "guide": "rgba(21, 35, 31, 0.5)",
        "pna": "#167264",
        "proposed": "#b9621e",
        "ats": "#315f8f",
        "zero": "#b33b3b",
    }


_ROOT_FONT_SIZE_PX = 16


_PLOT_TITLE_SIZE_REM = 1.5


_PLOT_MARGIN_REM = {"l": 1.5, "r": 1.5, "t": 3.5, "b": 1.5}


_PLOT_MARGIN_COMPACT_REM = {"l": 1.5, "r": 1.5, "t": 1.5, "b": 2.4}


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


_LESSON_ONE_GRAPH_HEIGHT = 340


_LESSON_TWO_GRAPH_HEIGHT = 392


_FILL_RATE_GRAPH_HEIGHT = 400


def _rem_to_px(rem: float) -> float:
    return rem * _ROOT_FONT_SIZE_PX


def _plot_base_layout(
    title: str | None,
    colors: dict[str, str],
    *,
    include_margin: bool = True,
    height: int | None = None,
) -> dict[str, object]:
    muted = colors.get("muted", colors["text"])
    layout: dict[str, object] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": colors["plot_bg"],
        "font": {
            "color": colors["text"],
            "family": "IBM Plex Sans, Segoe UI, sans-serif",
            "size": 13,
        },
        "legend_title_text": "",
        "legend": {
            "font": {"color": muted, "size": 12},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        "hoverlabel": {
            "bgcolor": colors["surface"],
            "bordercolor": colors["line"],
            "font": {"color": colors["text"]},
        },
        "xaxis": {
            "gridcolor": colors["line"],
            "linecolor": colors["line"],
            "tickfont": {"color": muted, "size": 11},
            "zerolinecolor": colors["line"],
        },
        "yaxis": {
            "gridcolor": colors["line"],
            "linecolor": colors["line"],
            "tickfont": {"color": muted, "size": 11},
            "zerolinecolor": colors["line"],
        },
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"color": colors["text"], "size": _rem_to_px(_PLOT_TITLE_SIZE_REM)},
        }
    if include_margin:
        margin_source = _PLOT_MARGIN_REM if title else _PLOT_MARGIN_COMPACT_REM
        layout["margin"] = {side: _rem_to_px(size) for side, size in margin_source.items()}
    if height is not None:
        layout["height"] = height
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


def _workspace_graph_height(state: SimulationState) -> int:
    return {
        "intro_pna": 340,
        "workspace_signal": 300,
        "workspace_advanced": 340,
        "workspace_certification": 360,
        "simulator": 460,
    }.get(active_layout_variant(state), 400)


def inventory_graph_height(state: SimulationState) -> int:
    level = active_level(state)
    if level is not None and level.index == 1:
        return _LESSON_ONE_GRAPH_HEIGHT
    if level is not None and level.index == 2:
        return _LESSON_TWO_GRAPH_HEIGHT
    if level is not None and level.index == 3:
        return _FILL_RATE_GRAPH_HEIGHT
    if level is not None and level.index in {12, 13}:
        return 340
    if level is not None and level.index == 17:
        return 360
    return _workspace_graph_height(state)


def inventory_graph_style(state: SimulationState) -> dict[str, str]:
    height = inventory_graph_height(state)
    return {"height": f"{height}px", "minHeight": f"{height}px", "width": "100%"}


def _signal_map_layout_signature(state: SimulationState, rows: pd.DataFrame) -> str:
    effective_cycle = effective_review_cycle(state.global_settings)
    return (
        f"{active_layout_variant(state)}:items:{len(rows)}:"
        f"r_cycle:{state.global_settings.r_cycle}:effective_cycle:{effective_cycle}"
    )


def _exception_map_layout_signature(rows: pd.DataFrame) -> str:
    return f"exceptions:items:{len(rows)}"


_FIGURE_KIND_KEY = "figure_kind"


_FIGURE_THEME_KEY = "theme"


_FIGURE_LAYOUT_SIG_KEY = "layout_signature"


def _figure_meta(
    kind: str,
    theme: str,
    *,
    layout_signature: str = "static",
) -> dict[str, str]:
    return {
        _FIGURE_KIND_KEY: kind,
        _FIGURE_THEME_KEY: theme,
        _FIGURE_LAYOUT_SIG_KEY: layout_signature,
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
        **_plot_base_layout(None, colors, height=_LESSON_ONE_GRAPH_HEIGHT),
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
    op_value = state.items[0].op if state.items else 0.0
    fig = go.Figure(
        data=[
            _empty_line_trace(
                name="On Hand",
                line=_plot_line(colors["pna"]),
                marker=_plot_marker(
                    colors["pna"],
                    _PLOT_MARKER_SIZE_REM["lesson"],
                    line=_plot_marker_outline(colors["surface"]),
                ),
                hovertemplate="On hand %{y:.1f}<extra></extra>",
            ),
            _empty_line_trace(
                name="On Order",
                line=_plot_line(colors["ats"], dash="dot"),
                marker=_plot_marker(
                    colors["ats"],
                    _PLOT_MARKER_SIZE_REM["lesson_backorder"],
                    symbol="square",
                ),
                hovertemplate="On order %{y:.1f}<extra></extra>",
            ),
            _empty_line_trace(
                name="PNA",
                line=_plot_line(colors["proposed"], dash="dash"),
                marker=_plot_marker(
                    colors["proposed"],
                    _PLOT_MARKER_SIZE_REM["lesson_backorder"],
                    symbol="diamond",
                ),
                hovertemplate="PNA %{y:.1f}<extra></extra>",
            ),
            _empty_line_trace(
                name="OP",
                line=_plot_line(colors["guide"], dash="dot"),
                marker=_plot_marker(colors["guide"], _PLOT_MARKER_SIZE_REM["lesson_backorder"]),
                hovertemplate="OP %{y:.1f}<extra></extra>",
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
        **_plot_base_layout(None, colors, height=_LESSON_TWO_GRAPH_HEIGHT),
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
        opacity=0.74,
        selector={"name": "On Order"},
    )
    fig.update_traces(x=days, selector={"name": "PNA"})
    fig.update_traces(
        y=[point.total_pna for point in history],
        opacity=0.78,
        selector={"name": "PNA"},
    )
    fig.update_traces(x=days, selector={"name": "OP"})
    fig.update_traces(y=[op_value for _day in days], selector={"name": "OP"})
    fig.update_traces(x=days, selector={"name": "Backorder"})
    fig.update_traces(
        y=[point.total_backorder for point in history],
        selector={"name": "Backorder"},
    )
    _apply_single_guide(fig, colors, y=0, label="Zero")
    return _finalize_axes(fig, colors, x_linear=True, y_tozero=True)


def _fill_rate_figure(state: SimulationState, theme: str, colors: dict[str, str]) -> go.Figure:
    level = active_level(state)
    target = float((level.win_conditions if level is not None else {}).get("fill_rate_min", 0.0))
    current = fill_rate(state)
    fill_pct = 0.0 if current is None else current * 100.0
    target_pct = target * 100.0
    complete = max(0, state.service_totals.orders - state.service_totals.orders_stockout)
    incomplete = state.service_totals.orders_stockout
    fig = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number",
                value=fill_pct,
                number={"suffix": "%", "font": {"color": colors["text"]}},
                title={"text": "Cumulative Fill Rate", "font": {"color": colors["text"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": colors["text"]},
                    "bar": {"color": colors["pna"]},
                    "bgcolor": colors["plot_bg"],
                    "bordercolor": colors["line"],
                    "steps": [
                        {"range": [0, target_pct], "color": "rgba(248, 113, 113, 0.18)"},
                        {"range": [target_pct, 100], "color": "rgba(45, 212, 191, 0.18)"},
                    ],
                    "threshold": {
                        "line": {"color": colors["zero"], "width": 4},
                        "thickness": 0.8,
                        "value": target_pct,
                    },
                },
                domain={"x": [0, 0.58], "y": [0, 1]},
            ),
            go.Bar(
                x=["Complete", "Incomplete"],
                y=[complete, incomplete],
                marker_color=[colors["pna"], colors["zero"]],
                text=[complete, incomplete],
                textposition="auto",
                hovertemplate="%{x} lines: %{y}<extra></extra>",
                xaxis="x2",
                yaxis="y2",
                showlegend=False,
            ),
        ]
    )
    base_layout = _plot_base_layout(None, colors, height=_FILL_RATE_GRAPH_HEIGHT)
    base_layout["margin"] = {
        "l": _rem_to_px(1.5),
        "r": _rem_to_px(1.5),
        "t": _rem_to_px(2.6),
        "b": _rem_to_px(2.0),
    }
    fig.update_layout(
        **base_layout,
        uirevision=f"lesson-3:{theme}",
        meta=_figure_meta("lesson-3-fill-rate", theme),
        xaxis2={
            "domain": [0.7, 1.0],
            "anchor": "y2",
            "color": colors["text"],
            "gridcolor": colors["line"],
        },
        yaxis2={
            "domain": [0.24, 0.84],
            "anchor": "x2",
            "color": colors["text"],
            "gridcolor": colors["line"],
            "rangemode": "tozero",
            "title": "Lines",
        },
    )
    fig.update_traces(domain={"x": [0.0, 0.6], "y": [0.08, 0.94]}, selector={"type": "indicator"})
    return fig


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
                hovertemplate=(
                    "Item %{x}<br>%{customdata:.1f} days to stockout"
                    "<extra>Available to Sell</extra>"
                ),
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
            None,
            colors,
            include_margin=not rows.empty,
            height=inventory_graph_height(state),
        ),
        xaxis_title="Item",
        yaxis_title="Days from OP",
        hovermode="closest",
        uirevision=f"signal-map:{theme}",
        meta=_figure_meta(
            "signal-map",
            theme,
            layout_signature=_signal_map_layout_signature(state, rows),
        ),
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
        upper_label=(
            f"LP Override ({effective_review_cycle(state.global_settings)}d)"
            if state.global_settings.review_cycle_override_days is not None
            else "LP"
        ),
        upper_y=effective_review_cycle(state.global_settings),
    )
    return _finalize_axes(
        fig,
        colors,
        x_linear=True,
        x_range=[0.5, max(1, len(rows)) + 0.5],
    )


def _exception_signal_figure(
    state: SimulationState,
    theme: str,
    colors: dict[str, str],
) -> go.Figure:
    rows = _items_frame(state.items)
    if rows.empty:
        rows = pd.DataFrame(columns=["idx", "pna", "cp", "surplus_line", "on_hand"])
    item_numbers = (rows["idx"] + 1).tolist() if "idx" in rows else []
    hover = "Item %{x}<br>%{y:.1f} units<extra>%{fullData.name}</extra>"
    fig = go.Figure(
        data=[
            _empty_marker_trace(
                name="PNA",
                marker=_plot_marker(colors["pna"], _PLOT_MARKER_SIZE_REM["signal"]),
                hovertemplate=hover,
            ),
            _empty_marker_trace(
                name="Critical Point",
                marker=_plot_marker(
                    colors["zero"],
                    _PLOT_MARKER_SIZE_REM["signal_zero"],
                    symbol="diamond-open",
                    line=_plot_marker_outline(colors["zero"]),
                ),
                hovertemplate=hover,
            ),
            _empty_marker_trace(
                name="Surplus Threshold",
                marker=_plot_marker(
                    colors["proposed"],
                    _PLOT_MARKER_SIZE_REM["signal_proposed"],
                    symbol="circle-open",
                    line=_plot_marker_outline(colors["proposed"]),
                ),
                hovertemplate=hover,
            ),
            _empty_marker_trace(
                name="On Hand",
                marker=_plot_marker(
                    colors["ats"],
                    _PLOT_MARKER_SIZE_REM["signal_ats"],
                    symbol="x",
                ),
                hovertemplate=hover,
            ),
        ]
    )
    fig.update_layout(
        **_plot_base_layout(
            None,
            colors,
            include_margin=not rows.empty,
            height=inventory_graph_height(state),
        ),
        xaxis_title="Item",
        yaxis_title="Units",
        hovermode="closest",
        uirevision=f"exception-map:{theme}",
        meta=_figure_meta(
            "exception-map",
            theme,
            layout_signature=_exception_map_layout_signature(rows),
        ),
    )
    fig.update_traces(x=item_numbers, selector={"name": "PNA"})
    fig.update_traces(y=rows["pna"].tolist(), selector={"name": "PNA"})
    fig.update_traces(x=item_numbers, selector={"name": "Critical Point"})
    fig.update_traces(y=rows["cp"].tolist(), selector={"name": "Critical Point"})
    fig.update_traces(x=item_numbers, selector={"name": "Surplus Threshold"})
    fig.update_traces(y=rows["surplus_line"].tolist(), selector={"name": "Surplus Threshold"})
    fig.update_traces(x=item_numbers, selector={"name": "On Hand"})
    fig.update_traces(y=rows["on_hand"].tolist(), selector={"name": "On Hand"})
    _apply_single_guide(fig, colors, y=0, label="Zero")
    return _finalize_axes(
        fig,
        colors,
        x_linear=True,
        x_range=[0.5, max(1, len(rows)) + 0.5],
        y_tozero=True,
    )


def build_inventory_figure(state: SimulationState, theme: str = "light") -> go.Figure:
    colors = _figure_theme(theme)
    level = active_level(state)
    if level is not None and level.index == 1:
        return _lesson_one_figure(state, theme, colors)
    if level is not None and level.index == 2:
        return _lesson_two_figure(state, theme, colors)
    if level is not None and level.index == 3:
        return _fill_rate_figure(state, theme, colors)
    if level is not None and level.index == 17:
        return _exception_signal_figure(state, theme, colors)
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
        or current_meta.get(_FIGURE_LAYOUT_SIG_KEY) != target_meta.get(_FIGURE_LAYOUT_SIG_KEY)
        or len((current_figure or {}).get("data") or []) != len(target.data)
    ):
        return target

    patched = Patch()
    for index, trace in enumerate(target.data):
        if getattr(trace, "x", None) is not None:
            patched["data"][index]["x"] = list(trace.x)
        if getattr(trace, "y", None) is not None:
            patched["data"][index]["y"] = list(trace.y)
        if getattr(trace, "value", None) is not None:
            patched["data"][index]["value"] = trace.value
        if getattr(trace, "text", None) is not None:
            patched["data"][index]["text"] = (
                trace.text if isinstance(trace.text, str) else list(trace.text)
            )
        if getattr(trace, "customdata", None) is not None:
            patched["data"][index]["customdata"] = list(trace.customdata)
        if getattr(trace, "hovertemplate", None) is not None:
            patched["data"][index]["hovertemplate"] = trace.hovertemplate
    return patched
