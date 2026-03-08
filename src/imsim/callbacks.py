from __future__ import annotations

import time
import uuid

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import ALL, Input, Output, State, ctx, html
from dash.exceptions import PreventUpdate

from .models import default_state
from .repository import SessionRepository
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
from .services.uploads import coerce_uploaded, parse_contents, read_uploaded_table
from .ui.components import (
    build_custom_order_row,
    build_exception_center,
    build_inventory_figure,
    build_inventory_table,
    build_kpi_strip,
    build_po_overview_table,
    costs_card_children,
    sales_card_children,
    service_card_children,
)


def register_callbacks(app, repository: SessionRepository, maintenance: MaintenanceController):
    def _require_session(client_data: dict | None):
        session_id = (client_data or {}).get("uuid")
        if not session_id:
            raise PreventUpdate
        return session_id, repository.get_or_create(session_id)

    def _theme_name(theme: str | None) -> str:
        return "dark" if theme == "dark" else "light"

    def _button_class(variant: str, extra: str = "") -> str:
        return " ".join(
            part for part in ["imsim-button", f"button-{variant}", extra] if part
        )

    def _start_button_state(*, running: bool, disabled: bool = False) -> tuple[str, str]:
        if disabled:
            return "Disabled for Maintenance", _button_class("secondary", "button-pill")
        if running:
            return "Pause Simulation", _button_class("warning", "button-pill")
        return "Start Simulation", _button_class("success", "button-pill")

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
        session_id = (client_data or {}).get("uuid")
        state = repository.get_or_create(session_id) if session_id else default_state()
        settings = state.global_settings
        return (
            settings.r_cycle,
            settings.r_cost,
            settings.k_cost * 100.0,
            settings.stockout_penalty,
            settings.expedite_rate * 100.0,
            settings.gm * 100.0,
            settings.realization * 100.0,
            settings.auto_po_enabled,
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
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_theme(_n_clicks, current_theme):
        return "dark" if current_theme != "dark" else "light"

    @app.callback(
        [
            Output("app-theme", "className"),
            Output("app-theme", "data-bs-theme"),
            Output("theme-toggle", "children"),
            Output("theme-toggle", "className"),
        ],
        Input("theme-store", "data"),
    )
    def apply_theme(theme):
        current_theme = "dark" if theme == "dark" else "light"
        next_label = "Light mode" if current_theme == "dark" else "Dark mode"
        return (
            f"imsim-theme theme-{current_theme}",
            current_theme,
            next_label,
            _button_class("ghost", "theme-toggle-button button-sm"),
        )

    @app.callback(
        [
            Output("day-display", "children"),
            Output("inventory-graph", "figure"),
            Output("service-card", "children"),
            Output("costs-card", "children"),
            Output("sales-card", "children"),
        ],
        Input("user-data-store", "data"),
        Input("theme-store", "data"),
    )
    def handle_page_load(client_data, theme):
        session_id = (client_data or {}).get("uuid")
        state = repository.get_or_create(session_id) if session_id else default_state()
        return (
            f"Day: {state.day}",
            build_inventory_figure(state, _theme_name(theme)),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
        )

    @app.callback(
        Output("kpi-strip", "children"),
        Input("inventory-graph", "figure"),
        State("user-data-store", "data"),
    )
    def render_kpi_strip(_figure, client_data):
        session_id = (client_data or {}).get("uuid")
        state = repository.get_or_create(session_id) if session_id else default_state()
        return build_kpi_strip(state)

    @app.callback(
        Output("inventory-table-shell", "children"),
        Input("inventory-graph", "figure"),
        State("user-data-store", "data"),
    )
    def render_inventory_table(_figure, client_data):
        session_id = (client_data or {}).get("uuid")
        state = repository.get_or_create(session_id) if session_id else default_state()
        return build_inventory_table(state)

    @app.callback(
        Output("exception-center-shell", "children"),
        Input("inventory-graph", "figure"),
        State("user-data-store", "data"),
    )
    def render_exception_stack(_figure, client_data):
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
        ],
        Input("interval-component", "n_intervals"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def update_on_interval(n_intervals, client_data, theme):
        if not n_intervals:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if not state.is_initialized or not state.items:
            raise PreventUpdate
        summary = tick_state(state)
        repository.save(session_id, state)
        feedback = dash.no_update
        if summary["asq_changed"]:
            feedback = dbc.Alert(
                f"ASQ month-end applied on {summary['asq_changed']} item(s).",
                color="info",
                duration=4000,
            )
        return (
            f"Day: {state.day}",
            build_inventory_figure(state, _theme_name(theme)),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            feedback,
        )

    @app.callback(
        [
            Output("sim-status", "children", allow_duplicate=True),
            Output("interval-component", "disabled", allow_duplicate=True),
            Output("start-button", "children"),
            Output("start-button", "className"),
        ],
        Input("start-button", "n_clicks"),
        State("user-data-store", "data"),
        State("interval-component", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_simulation(n_clicks, client_data, is_disabled):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if is_disabled:
            state.is_initialized = True
            repository.save(session_id, state)
            label, class_name = _start_button_state(running=True)
            return "Status: Running", False, label, class_name
        state.is_initialized = False
        repository.save(session_id, state)
        return "Status: Paused", True, "Resume Simulation", _button_class(
            "success", "button-pill"
        )

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
        ],
        Input("reset-button", "n_clicks"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def reset_simulation(n_clicks, client_data, theme):
        if not n_clicks:
            raise PreventUpdate
        session_id = (client_data or {}).get("uuid", "__bootstrap__")
        state = repository.reset(session_id)
        store = {"uuid": session_id} if session_id != "__bootstrap__" else client_data or {}
        label, class_name = _start_button_state(running=False)
        return (
            f"Day: {state.day}",
            build_inventory_figure(state, _theme_name(theme)),
            "Status: Paused",
            store,
            True,
            label,
            class_name,
            False,
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
            dash.no_update,
        )

    @app.callback(
        [
            Output("add-item-modal", "is_open"),
            Output("add-item-error", "children"),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
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
    ):
        if ctx.triggered_id == "add-item-button":
            return (
                not is_open,
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
            )
        session_id, state = _require_session(client_data)
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
        repository.save(session_id, state)
        figure = build_inventory_figure(state, _theme_name(theme))
        return (
            False,
            None,
            figure,
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
        )

    @app.callback(
        [
            Output("update-params-conf", "children"),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
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
            )
        session_id, state = _require_session(client_data)
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
        state.global_settings.auto_po_enabled = _toggle_enabled(auto_po_enabled)
        state.global_settings.asq.enabled = _toggle_enabled(asq_enabled)
        state.global_settings.asq.min_hits = int(asq_min_hits)
        state.global_settings.asq.max_amount_diff = float(asq_max_diff)
        state.global_settings.asq.period_days = int(asq_period_days)
        state.global_settings.asq.include_transfers = _toggle_enabled(asq_include_transfers)
        for item in state.items:
            update_gs_related_values(item, state.global_settings)
        repository.save(session_id, state)
        return (
            dbc.Alert("Parameters updated.", color="success", duration=3000),
            build_inventory_figure(state, _theme_name(theme)),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
        )

    @app.callback(
        [
            Output("asq-apply-feedback", "children", allow_duplicate=True),
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
        ],
        Input("apply-asq-button", "n_clicks"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def handle_apply_asq_now(n_clicks, client_data, theme):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if not state.items:
            return (
                dbc.Alert("No items to adjust.", color="secondary"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        summary = apply_asq_month_end(state)
        repository.save(session_id, state)
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
        )

    @app.callback(
        [
            Output("inventory-graph", "figure", allow_duplicate=True),
            Output("costs-card", "children", allow_duplicate=True),
            Output("service-card", "children", allow_duplicate=True),
            Output("sales-card", "children", allow_duplicate=True),
        ],
        Input("po-button", "n_clicks"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def handle_purchase_order(n_clicks, client_data, theme):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = _require_session(client_data)
        if not state.items:
            raise PreventUpdate
        place_purchase_orders(state)
        repository.save(session_id, state)
        return (
            build_inventory_figure(state, _theme_name(theme)),
            costs_card_children(state),
            service_card_children(state),
            sales_card_children(state),
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
    ):
        session_id, state = _require_session(client_data)
        rows = [build_custom_order_row(index, item) for index, item in enumerate(state.items)]
        if ctx.triggered_id == "place-custom-order-button":
            state.is_initialized = False
            repository.save(session_id, state)
            return (
                dash.no_update,
                rows,
                True,
                "Status: Paused",
                True,
                "Resume Simulation",
                _button_class("success", "button-pill"),
                True,
                costs_card_children(state),
                service_card_children(state),
                sales_card_children(state),
            )
        if ctx.triggered_id == "cancel-custom-order-button":
            return (
                dash.no_update,
                dash.no_update,
                False,
                "Status: Paused",
                True,
                "Resume Simulation",
                _button_class("success", "button-pill"),
                False,
                costs_card_children(state),
                service_card_children(state),
                sales_card_children(state),
            )
        if ctx.triggered_id != "place-order-button":
            raise PreventUpdate
        changed = place_custom_orders(state, list(order_quantities or []))
        repository.save(session_id, state)
        child_rows = rows if state.items else dbc.Alert("No items available.", color="warning")
        return (
            build_inventory_figure(state, _theme_name(theme)) if changed else dash.no_update,
            child_rows,
            False,
            "Status: Paused",
            True,
            "Resume Simulation",
            _button_class("success", "button-pill"),
            False,
            costs_card_children(state),
            service_card_children(state),
            sales_card_children(state),
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
        ],
        Input("import-uploaded-items", "n_clicks"),
        State("upload-preview-data", "data"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def import_uploaded_items(n_clicks, rows, client_data, theme):
        if not n_clicks:
            raise PreventUpdate
        if not rows:
            return (
                dash.no_update,
                dbc.Alert("No data to import.", color="warning"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        session_id, state = _require_session(client_data)
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
        repository.save(session_id, state)
        return (
            build_inventory_figure(state, _theme_name(theme)),
            dbc.Alert(f"Imported {len(df)} items successfully.", color="success"),
            service_card_children(state),
            costs_card_children(state),
            sales_card_children(state),
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
        ],
        [
            Input("po-overview-button", "n_clicks"),
            Input("po-overview-close", "n_clicks"),
            Input({"type": "po-expedite", "rid": ALL}, "n_clicks"),
            Input({"type": "po-cancel", "rid": ALL}, "n_clicks"),
        ],
        State("user-data-store", "data"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def po_overview_handler(
        open_clicks,
        close_clicks,
        expedite_clicks,
        cancel_clicks,
        client_data,
        theme,
    ):
        session_id, state = _require_session(client_data)
        trig = ctx.triggered_id
        if trig is None:
            raise PreventUpdate
        is_open = dash.no_update
        if trig == "po-overview-button":
            is_open = True
        elif trig == "po-overview-close":
            is_open = False
        elif isinstance(trig, dict) and "rid" in trig:
            action = "expedite" if trig.get("type") == "po-expedite" else "cancel"
            expedite_or_cancel_receipt(state, trig["rid"], action)
            repository.save(session_id, state)
        table = build_po_overview_table(state)
        return (
            is_open,
            table,
            build_inventory_figure(state, _theme_name(theme)),
            costs_card_children(state),
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
