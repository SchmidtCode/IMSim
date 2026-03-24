from __future__ import annotations

import time
import uuid

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import ALL, Input, Output, State, ctx, html
from dash.exceptions import PreventUpdate

from .models import default_state
from .repository import SessionConflictError, SessionRepository
from .services.asq import apply_asq_month_end
from .services.planning import (
    create_inventory_item,
    update_global_settings,
    update_gs_related_values,
)
from .services.simulation import (
    MaintenanceController,
    expedite_or_cancel_receipt,
    place_custom_orders,
    place_purchase_orders,
    tick_state,
)
from .services.training import (
    academy_levels,
    active_level,
    build_level_state,
    build_simulator_state,
    is_action_allowed,
    record_custom_order,
    record_guided_order,
    reset_progress_state,
    simulator_view_allowed,
    visible_panels,
)
from .services.uploads import coerce_uploaded, parse_contents, read_uploaded_table
from .ui.components import (
    academy_level_card_children,
    academy_progress_children,
    academy_result_children,
    build_custom_order_row,
    build_exception_center,
    build_inventory_figure,
    build_inventory_table,
    build_kpi_strip,
    build_po_overview_table,
    costs_card_children,
    lesson_compact_summary_children,
    lesson_locked_children,
    lesson_objective_children,
    lesson_tutorial_children,
    sales_card_children,
    service_card_children,
    simulator_unlock_children,
)


def _triggered_click_count(
    triggered_id: str | dict | None, click_map: dict[str, int | None]
) -> int:
    if not isinstance(triggered_id, str):
        return 0
    return int(click_map.get(triggered_id) or 0)


def _next_ui_refresh(refresh: int | None) -> int:
    return int(refresh or 0) + 1


def register_callbacks(app, repository: SessionRepository, maintenance: MaintenanceController):
    def _require_session(client_data: dict | None):
        session_id = (client_data or {}).get("uuid")
        if not session_id:
            raise PreventUpdate
        return session_id, repository.get_or_create(session_id)

    def _persist_state(session_id: str, state) -> None:
        try:
            repository.save(session_id, state)
        except SessionConflictError as exc:
            raise PreventUpdate from exc

    def _carry_revision(next_state, current_state) -> None:
        if getattr(next_state, "revision", 0) <= 0:
            next_state.revision = current_state.revision

    def _theme_name(theme: str | None) -> str:
        return "dark" if theme == "dark" else "light"

    def _button_class(variant: str, extra: str = "") -> str:
        return " ".join(part for part in ["imsim-button", f"button-{variant}", extra] if part)

    def _start_button_state(
        state,
        *,
        running: bool,
        disabled: bool = False,
        resumable: bool = False,
    ) -> tuple[str, str]:
        mode = "Simulation" if state.training.current_view == "simulator" else "Lesson"
        if disabled:
            label = "Lesson Complete" if state.training.current_view == "lesson" else "Disabled"
            return label, _button_class("secondary", "button-pill")
        if running:
            return f"Pause {mode}", _button_class("warning", "button-pill")
        if resumable:
            return f"Resume {mode}", _button_class("success", "button-pill")
        return f"Start {mode}", _button_class("success", "button-pill")

    def _lesson_terminal(state) -> bool:
        return state.training.current_view == "lesson" and state.training.lesson_status in {
            "passed",
            "failed",
        }

    def _panel_style(enabled: bool) -> dict[str, str]:
        return {} if enabled else {"display": "none"}

    def _current_state(client_data: dict | None):
        session_id = (client_data or {}).get("uuid")
        return repository.get_or_create(session_id) if session_id else default_state()

    def _toggle_enabled(value) -> bool:
        return bool(value)

    def _coerce_number(value, *, integer: bool = False):
        if value in (None, ""):
            return None
        try:
            return int(value) if integer else float(value)
        except (TypeError, ValueError):
            return None

    @app.callback(
        [
            Output("review-cycle-input", "value"),
            Output("r-cost-input", "value"),
            Output("k-cost-input", "value"),
            Output("stockout-penalty-input", "value"),
            Output("expedite-rate-input", "value"),
            Output("gm-input", "value"),
            Output("realization-input", "value"),
            Output("auto-po-enabled", "value"),
            Output("asq-enabled", "value"),
            Output("asq-min-hits", "value"),
            Output("asq-max-diff", "value"),
            Output("asq-period-days", "value"),
            Output("asq-include-transfers", "value"),
        ],
        Input("user-data-store", "data"),
    )
    def sync_parameter_controls(client_data):
        state = _current_state(client_data)
        settings = state.global_settings
        return (
            settings.r_cycle,
            settings.r_cost,
            settings.k_cost * 100.0,
            settings.stockout_penalty,
            settings.expedite_rate * 100.0,
            settings.gm * 100.0,
            settings.realization * 100.0,
            settings.auto_po_enabled if is_action_allowed(state, "auto_po") else False,
            settings.asq.enabled,
            settings.asq.min_hits,
            settings.asq.max_amount_diff,
            settings.asq.period_days,
            settings.asq.include_transfers,
        )

    @app.callback(
        Output("user-data-store", "data"),
        Input("page-load", "data"),
        State("user-data-store", "data"),
    )
    def ensure_uuid_on_load(_page_load, data):
        data = data or {}
        if not data.get("uuid"):
            data["uuid"] = str(uuid.uuid4())
        return data

    @app.callback(
        Output("theme-store", "data"),
        Input("theme-toggle", "n_clicks"),
        Input("academy-theme-toggle", "n_clicks"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_theme(_simulator_clicks, _academy_clicks, current_theme):
        return "dark" if current_theme != "dark" else "light"

    @app.callback(
        [
            Output("app-theme", "className"),
            Output("app-theme", "data-bs-theme"),
            Output("theme-toggle", "children"),
            Output("theme-toggle", "className"),
            Output("academy-theme-toggle", "children"),
            Output("academy-theme-toggle", "className"),
            Output("lesson-intro-modal", "content_class_name"),
        ],
        Input("theme-store", "data"),
    )
    def apply_theme(theme):
        current_theme = "dark" if theme == "dark" else "light"
        next_label = "Light mode" if current_theme == "dark" else "Dark mode"
        toggle_class = _button_class("ghost", "theme-toggle-button button-sm")
        return (
            f"imsim-theme theme-{current_theme}",
            current_theme,
            next_label,
            toggle_class,
            next_label,
            toggle_class,
            (
                "lesson-intro-content theme-dark"
                if current_theme == "dark"
                else "lesson-intro-content"
            ),
        )

    @app.callback(
        [
            Output("day-display", "children", allow_duplicate=True),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("sim-status", "children", allow_duplicate=True),
            Output("interval-component", "disabled", allow_duplicate=True),
            Output("start-button", "children", allow_duplicate=True),
            Output("start-button", "className", allow_duplicate=True),
            Output("start-button", "disabled", allow_duplicate=True),
            Output("reset-button", "children", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        [
            Input("academy-level-1-button", "n_clicks"),
            Input("academy-level-2-button", "n_clicks"),
            Input("academy-level-3-button", "n_clicks"),
            Input("academy-level-4-button", "n_clicks"),
            Input("academy-level-5-button", "n_clicks"),
            Input("academy-level-6-button", "n_clicks"),
            Input("academy-level-7-button", "n_clicks"),
            Input("academy-simulator-button", "n_clicks"),
            Input("return-to-menu-button", "n_clicks"),
            Input("simulator-return-button", "n_clicks"),
            Input("academy-reset-progress-button", "n_clicks"),
        ],
        State("user-data-store", "data"),
        State("theme-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def academy_navigation(
        _level_1,
        _level_2,
        _level_3,
        _level_4,
        _level_5,
        _level_6,
        _level_7,
        _open_simulator,
        _lesson_return,
        _simulator_return,
        _reset_progress,
        client_data,
        theme,
        ui_refresh,
    ):
        trig = ctx.triggered_id
        if trig is None:
            raise PreventUpdate
        click_count = _triggered_click_count(
            trig,
            {
                "academy-level-1-button": _level_1,
                "academy-level-2-button": _level_2,
                "academy-level-3-button": _level_3,
                "academy-level-4-button": _level_4,
                "academy-level-5-button": _level_5,
                "academy-level-6-button": _level_6,
                "academy-level-7-button": _level_7,
                "academy-simulator-button": _open_simulator,
                "return-to-menu-button": _lesson_return,
                "simulator-return-button": _simulator_return,
                "academy-reset-progress-button": _reset_progress,
            },
        )
        # The academy cards are re-rendered while hidden; ignore remounted buttons
        # that appear as triggered inputs with a zero click count.
        if click_count <= 0:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if trig == "academy-reset-progress-button":
            next_state = reset_progress_state()
        elif trig == "academy-simulator-button":
            if not simulator_view_allowed(state.training):
                raise PreventUpdate
            next_state = build_simulator_state(state.training)
        elif trig in {"return-to-menu-button", "simulator-return-button"}:
            state.training.current_view = "main_menu"
            state.training.active_level_id = None
            state.training.lesson_status = "idle"
            state.training.lesson_intro_dismissed = False
            state.is_initialized = False
            next_state = state
        elif (
            isinstance(trig, str) and trig.startswith("academy-level-") and trig.endswith("-button")
        ):
            level_index = int(trig.split("-")[2])
            level = academy_levels()[level_index - 1]
            if level.index > state.training.highest_unlocked_level:
                raise PreventUpdate
            next_state = build_level_state(level.level_id, state.training)
        else:
            raise PreventUpdate
        _carry_revision(next_state, state)
        _persist_state(session_id, next_state)
        label, class_name = _start_button_state(next_state, running=False)
        reset_label = (
            "Reset Sandbox" if next_state.training.current_view == "simulator" else "Restart Lesson"
        )
        return (
            f"Day: {next_state.day}",
            build_inventory_figure(next_state, _theme_name(theme)),
            "Status: Ready",
            True,
            label,
            class_name,
            False,
            reset_label,
            service_card_children(next_state),
            costs_card_children(next_state),
            sales_card_children(next_state),
            _next_ui_refresh(ui_refresh),
        )

    @app.callback(
        [
            Output("academy-menu-shell", "style"),
            Output("lesson-shell", "style"),
            Output("simulator-shell", "style"),
            Output("dashboard-shell", "style"),
            Output("academy-progress-summary", "children"),
            Output("academy-result-banner", "children"),
            Output("lesson-title", "children"),
            Output("lesson-copy", "children"),
            Output("lesson-tutorial", "children"),
            Output("lesson-objectives", "children"),
            Output("lesson-locked", "children"),
            Output("lesson-intro-modal", "is_open"),
            Output("simulator-copy", "children"),
            Output("experience-kicker", "children"),
            Output("experience-title", "children"),
            Output("experience-copy", "children"),
            Output("lesson-compact-summary", "children"),
            Output("lesson-return-wrap", "style"),
            Output("dashboard-shell", "className"),
            Output("sim-status", "children", allow_duplicate=True),
            Output("start-button", "children", allow_duplicate=True),
            Output("start-button", "className", allow_duplicate=True),
            Output("start-button", "disabled", allow_duplicate=True),
            Output("reset-button", "children", allow_duplicate=True),
            Output("po-button", "children"),
            Output("graph-panel-title", "children"),
            Output("service-panel-title", "children"),
            Output("inventory-panel-title", "children"),
            Output("actions-panel", "style"),
            Output("policy-panel", "style"),
            Output("session-panel", "style"),
            Output("graph-panel", "style"),
            Output("service-panel", "style"),
            Output("costs-panel", "style"),
            Output("sales-panel", "style"),
            Output("inventory-panel", "style"),
            Output("exceptions-panel", "style"),
            Output("kpi-strip", "style"),
            Output("guided-po-wrap", "style"),
            Output("custom-order-wrap", "style"),
            Output("po-overview-wrap", "style"),
            Output("add-item-wrap", "style"),
            Output("auto-po-shell", "style"),
            Output("asq-controls-shell", "style"),
            Output("auto-po-enabled", "disabled"),
            Output("academy-level-1-card", "children"),
            Output("academy-level-2-card", "children"),
            Output("academy-level-3-card", "children"),
            Output("academy-level-4-card", "children"),
            Output("academy-level-5-card", "children"),
            Output("academy-level-6-card", "children"),
            Output("academy-level-7-card", "children"),
            Output("academy-simulator-card", "children"),
            Output("academy-simulator-button", "disabled"),
            Output("advanced-sandbox-copy", "children"),
        ],
        Input("user-data-store", "data"),
        Input("day-display", "children"),
        Input("inventory-graph", "figure"),
        prevent_initial_call="initial_duplicate",
    )
    def render_training_shells(client_data, _day_display, _figure):
        state = _current_state(client_data)
        panels = visible_panels(state)
        level = active_level(state)
        is_menu = state.training.current_view == "main_menu"
        is_simulator = state.training.current_view == "simulator"
        lesson_title = level.title if level is not None else "Academy Lesson"
        lesson_copy = (
            level.summary if level is not None else "Start a lesson from the academy menu."
        )
        if is_simulator:
            experience_title = "Simulator control deck"
            experience_copy = (
                "Free play, imports, and the sandbox reward controls are now available."
            )
        elif level is not None:
            experience_title = level.title
            experience_copy = level.summary
        else:
            experience_title = "Lesson control deck"
            experience_copy = "Use the academy menu to start a lesson."
        reset_label = "Reset Sandbox" if is_simulator else "Restart Lesson"
        lesson_terminal = _lesson_terminal(state)
        po_label = (
            "Place SOQ Order"
            if (is_simulator or (level is not None and level.index >= 6))
            else "Place Guided Reorder"
        )
        graph_title = (
            "On-hand lesson trend"
            if level is not None and level.index == 1
            else (
                "Basic reorder signal"
                if level is not None and level.index == 2
                else "Inventory signal map"
            )
        )
        service_panel_title = "Lesson snapshot" if level is not None else "Service"
        start_label, start_class = _start_button_state(
            state,
            running=state.is_initialized,
            disabled=lesson_terminal,
            resumable=(
                not state.is_initialized and state.day > 1 and not is_menu and not lesson_terminal
            ),
        )
        simulator_copy = (
            (
                "The full IM dashboard is unlocked. "
                "Auto purchase orders remain off by default "
                "and live in the sandbox section."
            )
            if state.training.auto_po_reward_unlocked
            else (
                "The full IM dashboard is unlocked. "
                "Auto purchase orders stay hidden until certification "
                "is completed."
            )
        )
        sandbox_copy = (
            "Sandbox reward: auto purchase orders are unlocked here, but they still default to off."
            if state.training.auto_po_reward_unlocked
            else "Auto purchase orders are locked until certification is complete."
        )
        dashboard_class = "dashboard-shell"
        if state.training.current_view == "lesson":
            level_class = (
                f" lesson-dashboard lesson-level-{level.index}" if level is not None else ""
            )
            dashboard_class = f"{dashboard_class}{level_class}"
        elif is_simulator:
            dashboard_class = f"{dashboard_class} simulator-dashboard"
        return (
            _panel_style(is_menu),
            _panel_style(state.training.current_view == "lesson"),
            _panel_style(is_simulator),
            _panel_style(not is_menu),
            academy_progress_children(state),
            academy_result_children(state),
            lesson_title,
            lesson_copy,
            lesson_tutorial_children(state),
            lesson_objective_children(state),
            lesson_locked_children(state),
            state.training.current_view == "lesson" and not state.training.lesson_intro_dismissed,
            simulator_copy,
            "Simulator" if is_simulator else "Lesson",
            experience_title,
            experience_copy,
            lesson_compact_summary_children(state) if level is not None else [],
            _panel_style(state.training.current_view == "lesson"),
            dashboard_class,
            (
                "Status: Running"
                if state.is_initialized
                else (
                    "Status: Lesson complete"
                    if state.training.lesson_status == "passed"
                    else (
                        "Status: Lesson failed"
                        if state.training.lesson_status == "failed"
                        else ("Status: Ready" if not is_menu else "Status: Academy menu")
                    )
                )
            ),
            start_label,
            start_class,
            lesson_terminal,
            reset_label,
            po_label,
            graph_title,
            service_panel_title,
            "Planner grid"
            if is_simulator or (level is not None and level.index >= 6)
            else "Lesson items",
            _panel_style("actions" in panels),
            _panel_style("policy" in panels),
            _panel_style("session" in panels),
            _panel_style("graph" in panels),
            _panel_style("service" in panels),
            _panel_style("costs" in panels),
            _panel_style("sales" in panels),
            _panel_style("inventory" in panels and not (level is not None and level.index == 1)),
            _panel_style("exceptions" in panels),
            _panel_style("kpi" in panels),
            _panel_style(is_action_allowed(state, "guided_po")),
            _panel_style(is_action_allowed(state, "custom_order")),
            _panel_style(is_action_allowed(state, "po_overview")),
            _panel_style(is_action_allowed(state, "add_items")),
            _panel_style(is_simulator and state.training.auto_po_reward_unlocked),
            _panel_style(is_action_allowed(state, "apply_asq")),
            not (is_simulator and state.training.auto_po_reward_unlocked),
            academy_level_card_children(1, state),
            academy_level_card_children(2, state),
            academy_level_card_children(3, state),
            academy_level_card_children(4, state),
            academy_level_card_children(5, state),
            academy_level_card_children(6, state),
            academy_level_card_children(7, state),
            simulator_unlock_children(state),
            not state.training.simulator_unlocked,
            sandbox_copy,
        )

    @app.callback(
        Output("user-data-store", "data", allow_duplicate=True),
        Output("ui-refresh", "data", allow_duplicate=True),
        Input("lesson-intro-dismiss-button", "n_clicks"),
        State("user-data-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def dismiss_lesson_intro(n_clicks, client_data, ui_refresh):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if state.training.current_view != "lesson":
            raise PreventUpdate
        state.training.lesson_intro_dismissed = True
        _persist_state(session_id, state)
        return client_data, _next_ui_refresh(ui_refresh)

    @app.callback(
        [
            Output("day-display", "children"),
            Output("inventory-graph", "figure"),
            Output("service-card", "children"),
            Output("costs-card", "children"),
            Output("sales-card", "children"),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        Input("user-data-store", "data"),
        Input("theme-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def handle_page_load(client_data, theme, ui_refresh):
        session_id = (client_data or {}).get("uuid")
        state = repository.get_or_create(session_id) if session_id else default_state()
        refresh = (
            dash.no_update if ctx.triggered_id == "theme-store" else _next_ui_refresh(ui_refresh)
        )
        return (
            f"Day: {state.day}",
            build_inventory_figure(state, _theme_name(theme)),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            refresh,
        )

    @app.callback(
        Output("kpi-strip", "children"),
        Input("ui-refresh", "data"),
        State("user-data-store", "data"),
    )
    def render_kpi_strip(_ui_refresh, client_data):
        session_id = (client_data or {}).get("uuid")
        state = repository.get_or_create(session_id) if session_id else default_state()
        return build_kpi_strip(state)

    @app.callback(
        Output("inventory-table-shell", "children"),
        Input("ui-refresh", "data"),
        State("user-data-store", "data"),
    )
    def render_inventory_table(_ui_refresh, client_data):
        session_id = (client_data or {}).get("uuid")
        state = repository.get_or_create(session_id) if session_id else default_state()
        return build_inventory_table(state)

    @app.callback(
        Output("exception-center-shell", "children"),
        Input("ui-refresh", "data"),
        State("user-data-store", "data"),
    )
    def render_exception_stack(_ui_refresh, client_data):
        session_id = (client_data or {}).get("uuid")
        state = repository.get_or_create(session_id) if session_id else default_state()
        return build_exception_center(state)

    @app.callback(
        [
            Output("day-display", "children", allow_duplicate=True),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("asq-apply-feedback", "children", allow_duplicate=True),
            Output("sim-status", "children", allow_duplicate=True),
            Output("interval-component", "disabled", allow_duplicate=True),
            Output("start-button", "children", allow_duplicate=True),
            Output("start-button", "className", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        Input("interval-component", "n_intervals"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def update_on_interval(n_intervals, client_data, theme, ui_refresh):
        if not n_intervals:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if not state.is_initialized or not state.items:
            raise PreventUpdate
        summary = tick_state(state)
        _persist_state(session_id, state)
        feedback = dash.no_update
        if summary["asq_changed"]:
            feedback = dbc.Alert(
                f"ASQ month-end applied on {summary['asq_changed']} item(s).",
                color="info",
                duration=4000,
            )
        status = dash.no_update
        interval_disabled = dash.no_update
        button_label = dash.no_update
        button_class = dash.no_update
        if summary.get("lesson_completed"):
            status = (
                "Status: Lesson complete"
                if state.training.lesson_status == "passed"
                else "Status: Lesson failed"
            )
            interval_disabled = True
            button_label, button_class = _start_button_state(
                state,
                running=False,
                disabled=True,
            )
        return (
            f"Day: {state.day}",
            build_inventory_figure(state, _theme_name(theme)),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            feedback,
            status,
            interval_disabled,
            button_label,
            button_class,
            _next_ui_refresh(ui_refresh),
        )

    @app.callback(
        [
            Output("sim-status", "children", allow_duplicate=True),
            Output("interval-component", "disabled", allow_duplicate=True),
            Output("start-button", "children"),
            Output("start-button", "className"),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        Input("start-button", "n_clicks"),
        State("user-data-store", "data"),
        State("interval-component", "disabled"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def toggle_simulation(n_clicks, client_data, is_disabled, ui_refresh):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if state.training.current_view == "main_menu" or not state.items:
            raise PreventUpdate
        if _lesson_terminal(state):
            raise PreventUpdate
        if is_disabled:
            state.is_initialized = True
            state.training.lesson_status = (
                "running"
                if state.training.current_view == "lesson"
                else state.training.lesson_status
            )
            _persist_state(session_id, state)
            label, class_name = _start_button_state(state, running=True)
            return "Status: Running", False, label, class_name, _next_ui_refresh(ui_refresh)
        state.is_initialized = False
        _persist_state(session_id, state)
        label, class_name = _start_button_state(state, running=False, resumable=True)
        return "Status: Paused", True, label, class_name, _next_ui_refresh(ui_refresh)

    @app.callback(
        [
            Output("day-display", "children", allow_duplicate=True),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("sim-status", "children", allow_duplicate=True),
            Output("user-data-store", "data", allow_duplicate=True),
            Output("interval-component", "disabled", allow_duplicate=True),
            Output("start-button", "children", allow_duplicate=True),
            Output("start-button", "className", allow_duplicate=True),
            Output("start-button", "disabled", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("asq-apply-feedback", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        Input("reset-button", "n_clicks"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def reset_simulation(n_clicks, client_data, theme, ui_refresh):
        if not n_clicks:
            raise PreventUpdate
        session_id = (client_data or {}).get("uuid", "__bootstrap__")
        current = (
            repository.get_or_create(session_id)
            if session_id != "__bootstrap__"
            else default_state()
        )
        if current.training.current_view == "lesson" and current.training.active_level_id:
            state = build_level_state(current.training.active_level_id, current.training)
        elif current.training.current_view == "simulator":
            state = build_simulator_state(current.training)
        else:
            state = reset_progress_state()
        _carry_revision(state, current)
        _persist_state(session_id, state)
        store = {"uuid": session_id} if session_id != "__bootstrap__" else client_data or {}
        label, class_name = _start_button_state(state, running=False)
        return (
            f"Day: {state.day}",
            build_inventory_figure(state, _theme_name(theme)),
            "Status: Ready",
            store,
            True,
            label,
            class_name,
            False,
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            dash.no_update,
            _next_ui_refresh(ui_refresh),
        )

    @app.callback(
        [
            Output("add-item-modal", "is_open"),
            Output("add-item-error", "children"),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        [Input("add-item-button", "n_clicks"), Input("submit-item-button", "n_clicks")],
        [
            State("add-item-modal", "is_open"),
            State("usage-rate-input", "value"),
            State("lead-time-input", "value"),
            State("item-cost-input", "value"),
            State("pna-input", "value"),
            State("safety-allowance-input", "value"),
            State("standard-pack-input", "value"),
            State("hits-per-month-input", "value"),
            State("user-data-store", "data"),
            State("theme-store", "data"),
            State("ui-refresh", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_add_item_and_update_graph(
        add_clicks,
        submit_clicks,
        is_open,
        usage_rate,
        lead_time,
        item_cost,
        pna,
        safety_allowance,
        standard_pack,
        hits_per_month,
        client_data,
        theme,
        ui_refresh,
    ):
        if ctx.triggered_id == "add-item-button":
            _session_id, state = _require_session(client_data)
            if not is_action_allowed(state, "add_items"):
                return (
                    False,
                    dbc.Alert("Item editing unlocks in simulator mode.", color="warning"),
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )
            return (
                not is_open,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        if ctx.triggered_id != "submit-item-button":
            raise PreventUpdate
        values = [
            usage_rate,
            lead_time,
            item_cost,
            pna,
            safety_allowance,
            standard_pack,
            hits_per_month,
        ]
        if any(value is None for value in values):
            return (
                is_open,
                dbc.Alert("All fields must be filled out.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        if any(
            value <= 0
            for value in [
                usage_rate,
                lead_time,
                item_cost,
                safety_allowance,
                standard_pack,
                hits_per_month,
            ]
        ):
            return (
                is_open,
                dbc.Alert("All values except PNA must be greater than zero.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        session_id, state = _require_session(client_data)
        if not is_action_allowed(state, "add_items"):
            return (
                False,
                dbc.Alert("Item editing unlocks in simulator mode.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        state.items.append(
            create_inventory_item(
                usage_rate=usage_rate,
                lead_time=lead_time,
                item_cost=item_cost,
                pna=pna or 0,
                safety_allowance=float(safety_allowance) / 100.0,
                standard_pack=standard_pack,
                global_settings=state.global_settings,
                hits_per_month=hits_per_month,
            )
        )
        _persist_state(session_id, state)
        figure = build_inventory_figure(state, _theme_name(theme))
        return (
            False,
            None,
            figure,
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            _next_ui_refresh(ui_refresh),
        )

    @app.callback(
        [
            Output("update-params-conf", "children"),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        Input("update-params-button", "n_clicks"),
        [
            State("review-cycle-input", "value"),
            State("r-cost-input", "value"),
            State("k-cost-input", "value"),
            State("stockout-penalty-input", "value"),
            State("expedite-rate-input", "value"),
            State("gm-input", "value"),
            State("realization-input", "value"),
            State("auto-po-enabled", "value"),
            State("asq-enabled", "value"),
            State("asq-min-hits", "value"),
            State("asq-max-diff", "value"),
            State("asq-period-days", "value"),
            State("asq-include-transfers", "value"),
            State("user-data-store", "data"),
            State("theme-store", "data"),
            State("ui-refresh", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_parameters(
        n_clicks,
        review_cycle,
        r_cost,
        k_cost_pct,
        stockout_penalty,
        expedite_rate_pct,
        gm_pct,
        realization_pct,
        auto_po_enabled,
        asq_enabled,
        asq_min_hits,
        asq_max_diff,
        asq_period_days,
        asq_include_transfers,
        client_data,
        theme,
        ui_refresh,
    ):
        if not n_clicks:
            raise PreventUpdate
        review_cycle = _coerce_number(review_cycle, integer=True)
        r_cost = _coerce_number(r_cost)
        k_cost_pct = _coerce_number(k_cost_pct)
        stockout_penalty = _coerce_number(stockout_penalty)
        expedite_rate_pct = _coerce_number(expedite_rate_pct)
        gm_pct = _coerce_number(gm_pct)
        realization_pct = _coerce_number(realization_pct)
        asq_min_hits = _coerce_number(asq_min_hits, integer=True)
        asq_max_diff = _coerce_number(asq_max_diff)
        asq_period_days = _coerce_number(asq_period_days, integer=True)
        required = (
            review_cycle,
            r_cost,
            k_cost_pct,
            stockout_penalty,
            expedite_rate_pct,
            gm_pct,
            realization_pct,
            asq_min_hits,
            asq_max_diff,
            asq_period_days,
        )
        if any(value is None for value in required):
            return (
                dbc.Alert("Please fill out all parameters.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        if (
            review_cycle <= 0
            or r_cost < 0
            or k_cost_pct < 0
            or stockout_penalty < 0
            or expedite_rate_pct < 0
            or gm_pct < 0
            or gm_pct >= 100
            or realization_pct < 50
            or realization_pct > 100
            or asq_min_hits < 0
            or asq_max_diff < 0
            or asq_period_days <= 0
        ):
            return (
                dbc.Alert("Invalid parameter values.", color="danger"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        session_id, state = _require_session(client_data)
        if not is_action_allowed(state, "update_parameters"):
            return (
                dbc.Alert("Global parameter editing unlocks in later lessons.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        update_global_settings(
            state.global_settings,
            int(review_cycle),
            float(r_cost),
            float(k_cost_pct) / 100.0,
            float(stockout_penalty),
            float(expedite_rate_pct) / 100.0,
            float(gm_pct) / 100.0,
        )
        state.global_settings.realization = float(realization_pct) / 100.0
        state.global_settings.auto_po_enabled = (
            _toggle_enabled(auto_po_enabled) if is_action_allowed(state, "auto_po") else False
        )
        state.global_settings.asq.enabled = (
            _toggle_enabled(asq_enabled) if is_action_allowed(state, "apply_asq") else False
        )
        state.global_settings.asq.min_hits = int(asq_min_hits)
        state.global_settings.asq.max_amount_diff = float(asq_max_diff)
        state.global_settings.asq.period_days = int(asq_period_days)
        state.global_settings.asq.include_transfers = _toggle_enabled(asq_include_transfers)
        for item in state.items:
            update_gs_related_values(item, state.global_settings)
        _persist_state(session_id, state)
        return (
            dbc.Alert("Parameters updated.", color="success", duration=3000),
            build_inventory_figure(state, _theme_name(theme)),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            _next_ui_refresh(ui_refresh),
        )

    @app.callback(
        [
            Output("asq-apply-feedback", "children", allow_duplicate=True),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        Input("apply-asq-button", "n_clicks"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def handle_apply_asq_now(n_clicks, client_data, theme, ui_refresh):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if not is_action_allowed(state, "apply_asq"):
            return (
                dbc.Alert("ASQ unlocks in certification and simulator mode.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        if not state.items:
            return (
                dbc.Alert("No items to adjust.", color="secondary"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        summary = apply_asq_month_end(state)
        _persist_state(session_id, state)
        return (
            dbc.Alert(
                f"ASQ applied on {summary['changed']} item(s).",
                color="info",
                duration=4000,
            ),
            build_inventory_figure(state, _theme_name(theme)),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            _next_ui_refresh(ui_refresh),
        )

    @app.callback(
        [
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        Input("po-button", "n_clicks"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def handle_purchase_order(n_clicks, client_data, theme, ui_refresh):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if not state.items or not is_action_allowed(state, "guided_po"):
            raise PreventUpdate
        summary = place_purchase_orders(state)
        if summary["lines"] > 0:
            record_guided_order(state)
        _persist_state(session_id, state)
        return (
            build_inventory_figure(state, _theme_name(theme)),
            costs_card_children(state),
            service_card_children(state),
            sales_card_children(state),
            _next_ui_refresh(ui_refresh),
        )

    @app.callback(
        [
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("custom-order-items-div", "children"),
            Output("place-custom-order-modal", "is_open"),
            Output("sim-status", "children", allow_duplicate=True),
            Output("interval-component", "disabled", allow_duplicate=True),
            Output("start-button", "children", allow_duplicate=True),
            Output("start-button", "className", allow_duplicate=True),
            Output("start-button", "disabled", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        [
            Input("place-custom-order-button", "n_clicks"),
            Input("cancel-custom-order-button", "n_clicks"),
            Input("place-order-button", "n_clicks"),
        ],
        [
            State({"type": "order-quantity", "index": ALL}, "value"),
            State("user-data-store", "data"),
            State("theme-store", "data"),
            State("ui-refresh", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_custom_order_and_modal_actions(
        place_order_clicks,
        cancel_clicks,
        submit_clicks,
        order_quantities,
        client_data,
        theme,
        ui_refresh,
    ):
        session_id, state = _require_session(client_data)
        rows = [build_custom_order_row(index, item) for index, item in enumerate(state.items)]
        if ctx.triggered_id == "place-custom-order-button":
            if not is_action_allowed(state, "custom_order"):
                raise PreventUpdate
            state.is_initialized = False
            _persist_state(session_id, state)
            label, class_name = _start_button_state(state, running=False, resumable=True)
            return (
                dash.no_update,
                rows,
                True,
                "Status: Paused",
                True,
                label,
                class_name,
                True,
                costs_card_children(state),
                service_card_children(state),
                sales_card_children(state),
                _next_ui_refresh(ui_refresh),
            )
        if ctx.triggered_id == "cancel-custom-order-button":
            label, class_name = _start_button_state(state, running=False, resumable=True)
            return (
                dash.no_update,
                dash.no_update,
                False,
                "Status: Paused",
                True,
                label,
                class_name,
                False,
                costs_card_children(state),
                service_card_children(state),
                sales_card_children(state),
                dash.no_update,
            )
        if ctx.triggered_id != "place-order-button":
            raise PreventUpdate
        if not is_action_allowed(state, "custom_order"):
            raise PreventUpdate
        changed = place_custom_orders(state, list(order_quantities or []))
        if changed:
            record_custom_order(state)
        _persist_state(session_id, state)
        child_rows = rows if state.items else dbc.Alert("No items available.", color="warning")
        label, class_name = _start_button_state(state, running=False, resumable=True)
        return (
            build_inventory_figure(state, _theme_name(theme)) if changed else dash.no_update,
            child_rows,
            False,
            "Status: Paused",
            True,
            label,
            class_name,
            False,
            costs_card_children(state),
            service_card_children(state),
            sales_card_children(state),
            _next_ui_refresh(ui_refresh),
        )

    @app.callback(
        [
            Output("output-item-upload", "children"),
            Output("upload-preview-data", "data"),
            Output("import-uploaded-items", "style"),
        ],
        Input("upload-item", "contents"),
        State("upload-item", "filename"),
        State("upload-item", "last_modified"),
        prevent_initial_call=True,
    )
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if not list_of_contents:
            raise PreventUpdate
        cards = []
        frames = []
        errors = []
        for contents, name, modified in zip(
            list_of_contents,
            list_of_names or [],
            list_of_dates or [],
            strict=False,
        ):
            card = parse_contents(contents, name, modified)
            cards.append(card)
            if isinstance(card, dbc.Alert):
                errors.append(f"{name}: {card.children}")
                continue
            try:
                frames.append(coerce_uploaded(read_uploaded_table(contents, name)))
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if errors and not frames:
            return (
                cards + [dbc.Alert("; ".join(errors), color="warning")],
                None,
                {"display": "none"},
            )
        if not frames:
            return (
                cards + [dbc.Alert("No valid rows found.", color="warning")],
                None,
                {"display": "none"},
            )
        merged = pd.concat(frames, ignore_index=True)
        return cards, merged.to_dict("records"), {"display": "inline-block"}

    @app.callback(
        [
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("upload-feedback", "children"),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        Input("import-uploaded-items", "n_clicks"),
        State("upload-preview-data", "data"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def import_uploaded_items(n_clicks, rows, client_data, theme, ui_refresh):
        if not n_clicks:
            raise PreventUpdate
        if not rows:
            return (
                dash.no_update,
                dbc.Alert("No data to import.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        session_id, state = _require_session(client_data)
        if not is_action_allowed(state, "import_items"):
            return (
                dash.no_update,
                dbc.Alert("Imports unlock in simulator mode.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        df = coerce_uploaded(pd.DataFrame(rows))
        for record in df.to_dict("records"):
            state.items.append(
                create_inventory_item(
                    usage_rate=float(record["usage_rate"]),
                    lead_time=float(record["lead_time_days"]),
                    item_cost=float(record["item_cost"]),
                    pna=float(record["initial_pna"]),
                    safety_allowance=float(record["safety_allowance_pct"]) / 100.0,
                    standard_pack=float(record["standard_pack"]),
                    global_settings=state.global_settings,
                    hits_per_month=float(record["hits_per_month"]),
                )
            )
        _persist_state(session_id, state)
        return (
            build_inventory_figure(state, _theme_name(theme)),
            dbc.Alert(f"Imported {len(df)} items successfully.", color="success"),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            _next_ui_refresh(ui_refresh),
        )

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

    @app.callback(
        [
            Output("po-overview-modal", "is_open"),
            Output("po-overview-table", "children"),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("ui-refresh", "data", allow_duplicate=True),
        ],
        [
            Input("po-overview-button", "n_clicks"),
            Input("po-overview-close", "n_clicks"),
            Input({"type": "po-expedite", "rid": ALL}, "n_clicks"),
            Input({"type": "po-cancel", "rid": ALL}, "n_clicks"),
        ],
        State("user-data-store", "data"),
        State("theme-store", "data"),
        State("ui-refresh", "data"),
        prevent_initial_call=True,
    )
    def po_overview_handler(
        open_clicks,
        close_clicks,
        expedite_clicks,
        cancel_clicks,
        client_data,
        theme,
        ui_refresh,
    ):
        session_id, state = _require_session(client_data)
        trig = ctx.triggered_id
        if trig is None:
            raise PreventUpdate
        is_open = dash.no_update
        if trig == "po-overview-button":
            if not is_action_allowed(state, "po_overview"):
                raise PreventUpdate
            is_open = True
        elif trig == "po-overview-close":
            is_open = False
        elif isinstance(trig, dict) and "rid" in trig:
            action = "expedite" if trig.get("type") == "po-expedite" else "cancel"
            action_name = "expedite_receipt" if action == "expedite" else "cancel_receipt"
            if not is_action_allowed(state, action_name):
                raise PreventUpdate
            expedite_or_cancel_receipt(state, trig["rid"], action)
            _persist_state(session_id, state)
        table = build_po_overview_table(state)
        return (
            is_open,
            table,
            build_inventory_figure(state, _theme_name(theme)),
            costs_card_children(state),
            _next_ui_refresh(ui_refresh)
            if isinstance(trig, dict) and "rid" in trig
            else dash.no_update,
        )

    @app.callback(
        [
            Output("usage-rate-input", "value"),
            Output("lead-time-input", "value"),
            Output("item-cost-input", "value"),
            Output("pna-input", "value"),
            Output("safety-allowance-input", "value"),
            Output("standard-pack-input", "value"),
            Output("hits-per-month-input", "value"),
        ],
        Input("randomize-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def randomize_item_values(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        rng = dash.get_app().server.config.get("IMSIM_RNG")
        if rng is None:
            import numpy as np

            rng = np.random.default_rng()
            dash.get_app().server.config["IMSIM_RNG"] = rng
        usage = int(rng.integers(1, 101))
        lead = max(7, int(abs(rng.normal(30, 30))))
        cost = max(1, int(abs(rng.normal(100, 100))))
        safety_pct = 50 if lead < 60 else max(1, int(round(3000 / lead)))
        pack = int(
            rng.choice(
                [1, 5, 10, 20, 25, 40, 50],
                p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
            )
        )
        pna = int(
            round(usage * (lead / 30.0) + ((usage * (lead / 30.0)) * (safety_pct / 100.0)) + pack)
        )
        hits = max(1, int(rng.poisson(5)))
        return usage, lead, cost, pna, safety_pct, pack, hits

    @app.callback(
        Output("gh-footer-store", "data"),
        Input("gh-footer-hide", "n_clicks"),
        prevent_initial_call=True,
    )
    def hide_footer(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return False

    @app.callback(
        Output("gh-footer", "style"),
        Input("gh-footer-store", "data"),
    )
    def set_footer_visibility(is_visible):
        if is_visible is False:
            return {"display": "none"}
        return {}

    @app.callback(
        [
            Output("maintenance-banner", "children"),
            Output("sim-status", "children", allow_duplicate=True),
            Output("interval-component", "disabled", allow_duplicate=True),
            Output("start-button", "children", allow_duplicate=True),
            Output("start-button", "className", allow_duplicate=True),
            Output("start-button", "disabled", allow_duplicate=True),
        ],
        Input("shutdown-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def maintenance_heartbeat(_n):
        state = maintenance.heartbeat()
        if not state.active:
            return (
                html.Div(),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        remaining = max(0.0, state.at - time.time())
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        banner = dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Strong(state.message),
                            html.Span(f" shutting down in {mins:02d}:{secs:02d}", className="ms-2"),
                        ]
                    ),
                    html.Small(
                        "Sessions are auto-saved and simulation pauses in the final minute.",
                        className="text-muted",
                    ),
                ]
            ),
            className="maintenance-card",
        )
        if remaining <= 60:
            return (
                banner,
                "Status: Paused (maintenance)",
                True,
                "Disabled for Maintenance",
                _button_class("secondary", "button-pill"),
                True,
            )
        return (
            banner,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
