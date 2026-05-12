from __future__ import annotations

from dash import Input, Output, State, html
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..services.training import (
    academy_levels,
    active_level,
    build_level_state,
    build_simulator_state,
    is_action_allowed,
    reset_progress_state,
    simulator_view_allowed,
    visible_panels,
)
from ..ui.components import (
    academy_level_card_children,
    academy_progress_children,
    academy_result_children,
    lesson_compact_summary_children,
    lesson_locked_children,
    lesson_objective_children,
    lesson_tutorial_children,
    simulator_unlock_children,
)
from .common import CallbackRegistrarContext, _triggered_click_count


def register_training_callbacks(ctx: CallbackRegistrarContext) -> None:
    app = ctx.app

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
        Input("session-revision", "data"),
    )
    def sync_parameter_controls(client_data, _session_revision):
        state = ctx.current_state(client_data)
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
            import uuid

            data["uuid"] = str(uuid.uuid4())
        return data

    @app.callback(
        [
            Output("session-revision", "data", allow_duplicate=True),
            Output("asq-apply-feedback", "children", allow_duplicate=True),
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
        State("session-revision", "data"),
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
        session_revision,
    ):
        trig = dash_ctx.triggered_id
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
        if click_count <= 0:
            raise PreventUpdate
        session_id, state = ctx.require_session(client_data)
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
        ctx.carry_revision(next_state, state)
        ctx.persist_state(session_id, next_state)
        return ctx.next_session_revision(session_revision), html.Div()

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
            Output("interval-component", "disabled", allow_duplicate=True),
        ],
        Input("user-data-store", "data"),
        Input("session-revision", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def render_training_shells(client_data, _session_revision):
        state = ctx.current_state(client_data)
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
        lesson_terminal = ctx.lesson_terminal(state)
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
        start_label, start_class = ctx.start_button_state(
            state,
            running=state.is_initialized,
            disabled=lesson_terminal,
            resumable=(
                not state.is_initialized and state.day > 1 and not is_menu and not lesson_terminal
            ),
        )
        simulator_copy = (
            (
                "The full IM dashboard is unlocked. Auto purchase orders remain off by "
                "default and live in the sandbox section."
            )
            if state.training.auto_po_reward_unlocked
            else (
                "The full IM dashboard is unlocked. Auto purchase orders stay hidden "
                "until certification is completed."
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
        status = (
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
        )
        interval_disabled = not state.is_initialized
        return (
            ctx.panel_style(is_menu),
            ctx.panel_style(state.training.current_view == "lesson"),
            ctx.panel_style(is_simulator),
            ctx.panel_style(not is_menu),
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
            ctx.panel_style(state.training.current_view == "lesson"),
            dashboard_class,
            status,
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
            ctx.panel_style("actions" in panels),
            ctx.panel_style("policy" in panels),
            ctx.panel_style("session" in panels),
            ctx.panel_style("graph" in panels),
            ctx.panel_style("service" in panels),
            ctx.panel_style("costs" in panels),
            ctx.panel_style("sales" in panels),
            ctx.panel_style("inventory" in panels and not (level is not None and level.index == 1)),
            ctx.panel_style("exceptions" in panels),
            ctx.panel_style("kpi" in panels),
            ctx.panel_style(is_action_allowed(state, "guided_po")),
            ctx.panel_style(is_action_allowed(state, "custom_order")),
            ctx.panel_style(is_action_allowed(state, "po_overview")),
            ctx.panel_style(is_action_allowed(state, "add_items")),
            ctx.panel_style(is_simulator and state.training.auto_po_reward_unlocked),
            ctx.panel_style(is_action_allowed(state, "apply_asq")),
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
            interval_disabled,
        )

    @app.callback(
        Output("session-revision", "data", allow_duplicate=True),
        Input("lesson-intro-dismiss-button", "n_clicks"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def dismiss_lesson_intro(n_clicks, client_data, session_revision):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = ctx.require_session(client_data)
        if state.training.current_view != "lesson":
            raise PreventUpdate
        state.training.lesson_intro_dismissed = True
        ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision)
