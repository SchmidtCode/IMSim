from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from ..models import default_state
from ..services.simulation import tick_state
from ..services.training import build_level_state, build_simulator_state
from ..ui.components import (
    build_exception_center,
    build_inventory_table,
    build_kpi_strip,
    costs_card_children,
    refresh_inventory_figure,
    sales_card_children,
    service_card_children,
)
from .common import CallbackRegistrarContext


def register_simulation_callbacks(ctx: CallbackRegistrarContext) -> None:
    app = ctx.app

    @app.callback(
        [
            Output("day-display", "children"),
            Output("inventory-graph", "figure"),
            Output("service-card", "children"),
            Output("costs-card", "children"),
            Output("sales-card", "children"),
            Output("kpi-strip", "children"),
            Output("inventory-table-shell", "children"),
            Output("exception-center-shell", "children"),
        ],
        Input("user-data-store", "data"),
        Input("session-revision", "data"),
        Input("theme-store", "data"),
        State("inventory-graph", "figure"),
        prevent_initial_call="initial_duplicate",
    )
    def render_dashboard(client_data, _session_revision, theme, current_figure):
        session_id = (client_data or {}).get("uuid")
        state = ctx.repository.get_or_create(session_id) if session_id else default_state()
        theme_name = ctx.theme_name(theme)
        return (
            f"Day: {state.day}",
            refresh_inventory_figure(state, theme_name, current_figure),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            build_kpi_strip(state),
            build_inventory_table(state, theme_name),
            build_exception_center(state),
        )

    @app.callback(
        [
            Output("session-revision", "data", allow_duplicate=True),
            Output("asq-apply-feedback", "children", allow_duplicate=True),
        ],
        Input("interval-component", "n_intervals"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def update_on_interval(n_intervals, client_data, session_revision):
        if not n_intervals:
            raise PreventUpdate
        session_id, state = ctx.require_session(client_data)
        if not state.is_initialized or not state.items:
            raise PreventUpdate
        summary = tick_state(state)
        if summary.get("lesson_completed"):
            state.is_initialized = False
        ctx.persist_state(session_id, state)
        feedback = dash.no_update
        if summary["asq_changed"]:
            feedback = dbc.Alert(
                f"ASQ month-end applied on {summary['asq_changed']} item(s).",
                color="info",
                duration=4000,
            )
        return ctx.next_session_revision(session_revision), feedback

    @app.callback(
        Output("session-revision", "data", allow_duplicate=True),
        Input("start-button", "n_clicks"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def toggle_simulation(n_clicks, client_data, session_revision):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = ctx.require_session(client_data)
        if (
            state.training.current_view == "main_menu"
            or not state.items
            or ctx.lesson_terminal(state)
        ):
            raise PreventUpdate
        state.is_initialized = not state.is_initialized
        if state.is_initialized and state.training.current_view == "lesson":
            state.training.lesson_status = "running"
        ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision)

    @app.callback(
        [
            Output("session-revision", "data", allow_duplicate=True),
            Output("asq-apply-feedback", "children", allow_duplicate=True),
        ],
        Input("reset-button", "n_clicks"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def reset_simulation(n_clicks, client_data, session_revision):
        if not n_clicks:
            raise PreventUpdate
        session_id = (client_data or {}).get("uuid", "__bootstrap__")
        current = (
            ctx.repository.get_or_create(session_id)
            if session_id != "__bootstrap__"
            else default_state()
        )
        if current.training.current_view == "lesson" and current.training.active_level_id:
            state = build_level_state(current.training.active_level_id, current.training)
        elif current.training.current_view == "simulator":
            state = build_simulator_state(current.training)
        else:
            from ..services.training import reset_progress_state

            state = reset_progress_state()
        ctx.carry_revision(state, current)
        if session_id != "__bootstrap__":
            ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision), dash.no_update

    @app.callback(
        [
            Output("interval-component", "interval"),
            Output("sim-speed-readout", "children"),
        ],
        Input("sim-speed-slider", "value"),
    )
    def set_sim_speed(ticks_per_second):
        if ticks_per_second is None:
            raise PreventUpdate
        ticks_per_second = max(0.5, float(ticks_per_second))
        interval_ms = int(round(1000.0 / ticks_per_second))
        label = (
            "1 tick/sec"
            if abs(ticks_per_second - 1.0) < 1e-9
            else f"{ticks_per_second:.1f} ticks/sec"
        )
        return interval_ms, label

    @app.callback(
        Output("asq-collapse", "is_open"),
        Input("toggle-asq-collapse", "n_clicks"),
        State("asq-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_asq_collapse(n_clicks, is_open):
        if not n_clicks:
            raise PreventUpdate
        return not is_open
