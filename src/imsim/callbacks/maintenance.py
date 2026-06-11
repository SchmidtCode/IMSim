from __future__ import annotations

import time

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from ..ui.components import _grid_theme_class
from .common import CallbackRegistrarContext


def register_maintenance_callbacks(ctx: CallbackRegistrarContext) -> None:
    app = ctx.app

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
            Output("reference-modal", "content_class_name"),
            Output("add-item-modal", "content_class_name"),
            Output("place-custom-order-modal", "content_class_name"),
            Output("po-overview-modal", "content_class_name"),
            Output("custom-order-grid", "className"),
            Output("po-overview-grid", "className"),
        ],
        Input("theme-store", "data"),
    )
    def apply_theme(theme):
        current_theme = "dark" if theme == "dark" else "light"
        next_label = "Light mode" if current_theme == "dark" else "Dark mode"
        toggle_class = ctx.button_class("ghost", "theme-toggle-button button-sm")
        grid_class = _grid_theme_class(current_theme)
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
            (
                "imsim-modal-content theme-dark"
                if current_theme == "dark"
                else "imsim-modal-content"
            ),
            (
                "imsim-modal-content theme-dark"
                if current_theme == "dark"
                else "imsim-modal-content"
            ),
            (
                "imsim-modal-content theme-dark"
                if current_theme == "dark"
                else "imsim-modal-content"
            ),
            (
                "imsim-modal-content theme-dark"
                if current_theme == "dark"
                else "imsim-modal-content"
            ),
            grid_class,
            grid_class,
        )

    @app.callback(
        Output("reference-modal", "is_open"),
        Input("academy-reference-button", "n_clicks"),
        Input("experience-reference-button", "n_clicks"),
        Input("simulator-reference-button", "n_clicks"),
        Input("reference-modal-close", "n_clicks"),
        State("reference-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_reference_modal(
        academy_clicks,
        experience_clicks,
        simulator_clicks,
        close_clicks,
        is_open,
    ):
        triggered = dash.ctx.triggered_id
        if triggered in {
            "academy-reference-button",
            "experience-reference-button",
            "simulator-reference-button",
        } and any([academy_clicks, experience_clicks, simulator_clicks]):
            return True
        if triggered == "reference-modal-close" and close_clicks:
            return False
        return is_open

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
        state = ctx.maintenance.heartbeat()
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
                ctx.button_class("secondary", "button-pill"),
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
