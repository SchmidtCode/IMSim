from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..services.training import (
    academy_levels,
    active_layout_variant,
    active_level,
    build_level_state,
    build_simulator_state,
    cheat_unlock_password_matches,
    is_action_allowed,
    reset_progress_state,
    simulator_view_allowed,
    unlock_all_academy_levels,
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


def dashboard_shell_class_name(state) -> str:
    dashboard_class = "dashboard-shell"
    if state.training.current_view == "lesson":
        level = active_level(state)
        variant = active_layout_variant(state).replace("_", "-")
        level_class = f" lesson-level-{level.index}" if level is not None else ""
        return f"{dashboard_class} lesson-dashboard{level_class} lesson-layout-{variant}"
    if state.training.current_view == "simulator":
        return f"{dashboard_class} simulator-dashboard"
    return dashboard_class


def scroll_reset_view_key(state) -> str:
    if state.training.current_view == "lesson":
        return f"lesson:{dashboard_shell_class_name(state)}"
    if state.training.current_view == "simulator":
        return f"simulator:{dashboard_shell_class_name(state)}"
    return state.training.current_view


def register_training_callbacks(ctx: CallbackRegistrarContext) -> None:
    app = ctx.app
    levels = academy_levels()
    level_button_inputs = [
        Input(f"academy-level-{level.index}-button", "n_clicks") for level in levels
    ]
    level_card_outputs = [
        Output(f"academy-level-{level.index}-card", "children") for level in levels
    ]

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
            Output("review-cycle-override-input", "value"),
            Output("review-cycle-override-indicator", "children"),
            Output("review-cycle-override-indicator", "style"),
        ],
        Input("user-data-store", "data"),
        Input("session-revision", "data"),
    )
    def sync_parameter_controls(client_data, _session_revision):
        state = ctx.current_state(client_data)
        settings = state.global_settings
        override = settings.review_cycle_override_days
        override_active = override is not None
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
            override or settings.r_cycle,
            f"Review Cycle Override Active: {override} days" if override_active else "",
            {} if override_active else {"display": "none"},
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
        Output("session-revision", "data", allow_duplicate=True),
        Input("page-lifecycle-store", "data"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def pause_session_when_page_inactive(lifecycle, client_data, session_revision):
        if not isinstance(lifecycle, dict):
            raise PreventUpdate
        session_id, state = ctx.require_session(client_data)
        is_active = lifecycle.get("active") is not False
        if is_active:
            return ctx.next_session_revision(session_revision)
        if not state.is_initialized:
            raise PreventUpdate
        state.is_initialized = False
        ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision)

    @app.callback(
        [
            Output("session-revision", "data", allow_duplicate=True),
            Output("asq-apply-feedback", "children", allow_duplicate=True),
            Output("view-scroll-store", "data", allow_duplicate=True),
        ],
        [
            *level_button_inputs,
            Input("academy-simulator-button", "n_clicks"),
            Input("return-to-menu-button", "n_clicks"),
            Input("simulator-return-button", "n_clicks"),
            Input("academy-reset-progress-button", "n_clicks"),
        ],
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def academy_navigation(*callback_args):
        *input_values, client_data, session_revision = callback_args
        level_clicks = input_values[: len(levels)]
        (
            open_simulator_clicks,
            lesson_return_clicks,
            simulator_return_clicks,
            reset_progress_clicks,
        ) = input_values[len(levels) :]
        trig = dash_ctx.triggered_id
        if trig is None:
            raise PreventUpdate
        click_count = _triggered_click_count(
            trig,
            {
                **{
                    f"academy-level-{level.index}-button": click_count
                    for level, click_count in zip(levels, level_clicks, strict=True)
                },
                "academy-simulator-button": open_simulator_clicks,
                "return-to-menu-button": lesson_return_clicks,
                "simulator-return-button": simulator_return_clicks,
                "academy-reset-progress-button": reset_progress_clicks,
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
        next_revision = ctx.next_session_revision(session_revision)
        scroll_payload = {
            "view": next_state.training.current_view,
            "level_id": next_state.training.active_level_id,
            "revision": next_revision,
            "view_key": scroll_reset_view_key(next_state),
        }
        return next_revision, html.Div(), scroll_payload

    app.clientside_callback(
        """
        function(scrollTrigger) {
          if (!scrollTrigger) {
            return window.dash_clientside.no_update;
          }

          var targetKey = scrollTrigger.view_key || "";
          var revision = scrollTrigger.revision;
          if (!targetKey || revision === undefined || revision === null) {
            return window.dash_clientside.no_update;
          }

          var token = String(revision) + ":" + targetKey;
          if (window.__imsimLastScrollResetToken === token) {
            return window.dash_clientside.no_update;
          }
          window.__imsimPendingScrollResetToken = token;

          function resetScroll() {
            if (document.activeElement && typeof document.activeElement.blur === "function") {
              document.activeElement.blur();
            }
            window.scrollTo({ top: 0, left: 0, behavior: "auto" });
            if (document.documentElement) {
              document.documentElement.scrollTop = 0;
            }
            if (document.body) {
              document.body.scrollTop = 0;
            }
          }

          function isVisible(element) {
            if (!element) {
              return false;
            }
            var style = window.getComputedStyle(element);
            return style.display !== "none" && style.visibility !== "hidden";
          }

          function currentViewKey() {
            var menu = document.getElementById("academy-menu-shell");
            var dashboard = document.getElementById("dashboard-shell");
            var lesson = document.getElementById("lesson-shell");
            var simulator = document.getElementById("simulator-shell");
            var activeClass = dashboard ? dashboard.className || "" : "";

            if (isVisible(menu)) {
              return "main_menu";
            }
            if (
              isVisible(dashboard) &&
              isVisible(lesson) &&
              activeClass.indexOf("lesson-dashboard") !== -1
            ) {
              return "lesson:" + activeClass;
            }
            if (
              isVisible(dashboard) &&
              isVisible(simulator) &&
              activeClass.indexOf("simulator-dashboard") !== -1
            ) {
              return "simulator:" + activeClass;
            }
            return "";
          }

          var attempts = 0;
          function resetWhenRendered() {
            if (window.__imsimPendingScrollResetToken !== token) {
              return;
            }
            if (currentViewKey() === targetKey) {
              window.__imsimLastScrollResetToken = token;
              resetScroll();
              window.setTimeout(resetScroll, 60);
              window.setTimeout(resetScroll, 180);
              return;
            }
            attempts += 1;
            if (attempts < 60) {
              window.requestAnimationFrame(resetWhenRendered);
            }
          }

          window.requestAnimationFrame(resetWhenRendered);
          return window.dash_clientside.no_update;
        }
        """,
        Output("view-scroll-sink", "data"),
        Input("view-scroll-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        [
            Output("academy-cheat-code-modal", "is_open"),
            Output("academy-cheat-code-feedback", "children"),
            Output("academy-cheat-code-input", "value"),
            Output("session-revision", "data", allow_duplicate=True),
        ],
        [
            Input("academy-cheat-code-button", "n_clicks"),
            Input("academy-cheat-code-cancel", "n_clicks"),
            Input("academy-cheat-code-submit", "n_clicks"),
        ],
        [
            State("academy-cheat-code-input", "value"),
            State("user-data-store", "data"),
            State("session-revision", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_academy_cheat_code(
        open_clicks,
        cancel_clicks,
        submit_clicks,
        password,
        client_data,
        session_revision,
    ):
        trig = dash_ctx.triggered_id
        click_count = _triggered_click_count(
            trig,
            {
                "academy-cheat-code-button": open_clicks,
                "academy-cheat-code-cancel": cancel_clicks,
                "academy-cheat-code-submit": submit_clicks,
            },
        )
        if click_count <= 0:
            raise PreventUpdate
        if trig == "academy-cheat-code-button":
            return True, html.Div(), "", no_update
        if trig == "academy-cheat-code-cancel":
            return False, html.Div(), "", no_update
        if trig != "academy-cheat-code-submit":
            raise PreventUpdate
        if not cheat_unlock_password_matches(password):
            return (
                True,
                dbc.Alert("Nope. The magic words are not magic enough.", color="warning"),
                no_update,
                no_update,
            )

        session_id, state = ctx.require_session(client_data)
        unlock_all_academy_levels(state.training)
        state.is_initialized = False
        ctx.persist_state(session_id, state)
        return (
            False,
            html.Div(),
            "",
            ctx.next_session_revision(session_revision),
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
            Output("review-cycle-override-wrap", "style"),
            Output("auto-po-shell", "style"),
            Output("asq-controls-shell", "style"),
            Output("auto-po-enabled", "disabled"),
            *level_card_outputs,
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
            "Place Suggested Order"
            if (is_simulator or (level is not None and level.index >= 13))
            else "Place Guided Reorder"
        )
        graph_title = (
            "On-hand lesson trend"
            if level is not None and level.index == 1
            else (
                "Basic reorder signal"
                if level is not None and level.index == 2
                else (
                    "Fill-rate service view"
                    if level is not None and level.index == 3
                    else (
                        "Critical-point and surplus map"
                        if level is not None and level.index == 17
                        else "Inventory signal map"
                    )
                )
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
        dashboard_class = dashboard_shell_class_name(state)
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
            ctx.panel_style(is_action_allowed(state, "update_parameters")),
            ctx.panel_style(is_simulator and state.training.auto_po_reward_unlocked),
            ctx.panel_style(is_action_allowed(state, "apply_asq")),
            not (is_simulator and state.training.auto_po_reward_unlocked),
            *[academy_level_card_children(level.index, state) for level in levels],
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

    @app.callback(
        Output("session-revision", "data", allow_duplicate=True),
        Input("lesson-intro-modal", "is_open"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def persist_lesson_intro_close(is_open, client_data, session_revision):
        if is_open:
            raise PreventUpdate
        session_id, state = ctx.require_session(client_data)
        if state.training.current_view != "lesson" or state.training.lesson_intro_dismissed:
            raise PreventUpdate
        state.training.lesson_intro_dismissed = True
        ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision)
