from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, set_props
from dash import ctx as dash_ctx
from dash.exceptions import PreventUpdate

from ..services.asq import apply_asq_month_end
from ..services.planning import (
    create_inventory_item,
    update_global_settings,
    update_gs_related_values,
)
from ..services.simulation import (
    expedite_or_cancel_receipts,
    place_custom_orders,
    place_purchase_orders,
)
from ..services.training import (
    is_action_allowed,
    record_custom_order,
    record_guided_order,
    record_parameter_update,
)
from ..services.uploads import coerce_uploaded, parse_contents, read_uploaded_table
from ..ui.components import (
    build_custom_order_grid,
    build_po_overview_grid,
)
from .common import CallbackRegistrarContext


def register_inventory_callbacks(ctx: CallbackRegistrarContext) -> None:
    app = ctx.app

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
        [
            Output("session-revision", "data", allow_duplicate=True),
            Output("add-item-error", "children"),
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
            State("session-revision", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_add_item(
        _add_clicks,
        _submit_clicks,
        is_open,
        usage_rate,
        lead_time,
        item_cost,
        pna,
        safety_allowance,
        standard_pack,
        hits_per_month,
        client_data,
        session_revision,
    ):
        if dash_ctx.triggered_id == "add-item-button":
            _session_id, state = ctx.require_session(client_data)
            if not is_action_allowed(state, "add_items"):
                return dash.no_update, dbc.Alert(
                    "Item editing unlocks in simulator mode.",
                    color="warning",
                )
            set_props("add-item-modal", {"is_open": not is_open})
            return dash.no_update, dash.no_update
        if dash_ctx.triggered_id != "submit-item-button":
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
            return dash.no_update, dbc.Alert("All fields must be filled out.", color="warning")
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
            return dash.no_update, dbc.Alert(
                "All values except PNA must be greater than zero.",
                color="warning",
            )
        session_id, state = ctx.require_session(client_data)
        if not is_action_allowed(state, "add_items"):
            set_props("add-item-modal", {"is_open": False})
            return dash.no_update, dbc.Alert(
                "Item editing unlocks in simulator mode.",
                color="warning",
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
        ctx.persist_state(session_id, state)
        set_props("add-item-modal", {"is_open": False})
        return ctx.next_session_revision(session_revision), None

    @app.callback(
        [
            Output("session-revision", "data", allow_duplicate=True),
            Output("update-params-conf", "children"),
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
            State("session-revision", "data"),
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
        session_revision,
    ):
        if not n_clicks:
            raise PreventUpdate
        review_cycle = ctx.coerce_number(review_cycle, integer=True)
        r_cost = ctx.coerce_number(r_cost)
        k_cost_pct = ctx.coerce_number(k_cost_pct)
        stockout_penalty = ctx.coerce_number(stockout_penalty)
        expedite_rate_pct = ctx.coerce_number(expedite_rate_pct)
        gm_pct = ctx.coerce_number(gm_pct)
        realization_pct = ctx.coerce_number(realization_pct)
        asq_min_hits = ctx.coerce_number(asq_min_hits, integer=True)
        asq_max_diff = ctx.coerce_number(asq_max_diff)
        asq_period_days = ctx.coerce_number(asq_period_days, integer=True)
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
            return dash.no_update, dbc.Alert("Please fill out all parameters.", color="warning")
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
            return dash.no_update, dbc.Alert("Invalid parameter values.", color="danger")
        session_id, state = ctx.require_session(client_data)
        if not is_action_allowed(state, "update_parameters"):
            return dash.no_update, dbc.Alert(
                "Global parameter editing unlocks in later lessons.",
                color="warning",
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
            ctx.toggle_enabled(auto_po_enabled) if is_action_allowed(state, "auto_po") else False
        )
        state.global_settings.asq.enabled = (
            ctx.toggle_enabled(asq_enabled) if is_action_allowed(state, "apply_asq") else False
        )
        state.global_settings.asq.min_hits = int(asq_min_hits)
        state.global_settings.asq.max_amount_diff = float(asq_max_diff)
        state.global_settings.asq.period_days = int(asq_period_days)
        state.global_settings.asq.include_transfers = ctx.toggle_enabled(asq_include_transfers)
        for item in state.items:
            update_gs_related_values(item, state.global_settings)
        record_parameter_update(state)
        ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision), dbc.Alert(
            "Parameters updated.",
            color="success",
            duration=3000,
        )

    @app.callback(
        [
            Output("session-revision", "data", allow_duplicate=True),
            Output("asq-apply-feedback", "children", allow_duplicate=True),
        ],
        Input("apply-asq-button", "n_clicks"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def handle_apply_asq_now(n_clicks, client_data, session_revision):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = ctx.require_session(client_data)
        if not is_action_allowed(state, "apply_asq"):
            return dash.no_update, dbc.Alert(
                "ASQ unlocks in certification and simulator mode.",
                color="warning",
            )
        if not state.items:
            return dash.no_update, dbc.Alert("No items to adjust.", color="secondary")
        summary = apply_asq_month_end(state)
        ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision), dbc.Alert(
            f"ASQ applied on {summary['changed']} item(s).",
            color="info",
            duration=4000,
        )

    @app.callback(
        Output("session-revision", "data", allow_duplicate=True),
        Input("po-button", "n_clicks"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def handle_purchase_order(n_clicks, client_data, session_revision):
        if not n_clicks:
            raise PreventUpdate
        session_id, state = ctx.require_session(client_data)
        if not state.items or not is_action_allowed(state, "guided_po"):
            raise PreventUpdate
        summary = place_purchase_orders(state)
        if summary["lines"] > 0:
            record_guided_order(state)
        ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision)

    @app.callback(
        [
            Output("custom-order-grid", "rowData"),
            Output("custom-order-grid", "columnDefs"),
            Output("session-revision", "data", allow_duplicate=True),
        ],
        [
            Input("place-custom-order-button", "n_clicks"),
            Input("cancel-custom-order-button", "n_clicks"),
            Input("place-order-button", "n_clicks"),
        ],
        [
            State("custom-order-grid", "rowData"),
            State("user-data-store", "data"),
            State("theme-store", "data"),
            State("session-revision", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_custom_order_modal(
        _open_clicks,
        _cancel_clicks,
        _submit_clicks,
        row_data,
        client_data,
        theme,
        session_revision,
    ):
        session_id, state = ctx.require_session(client_data)
        theme_name = ctx.theme_name(theme)
        if dash_ctx.triggered_id == "place-custom-order-button":
            if not is_action_allowed(state, "custom_order"):
                raise PreventUpdate
            state.is_initialized = False
            ctx.persist_state(session_id, state)
            grid = build_custom_order_grid(state, theme_name)
            if isinstance(grid, dbc.Alert):
                return [], [], ctx.next_session_revision(session_revision)
            set_props("place-custom-order-modal", {"is_open": True})
            return grid.rowData, grid.columnDefs, ctx.next_session_revision(session_revision)
        if dash_ctx.triggered_id == "cancel-custom-order-button":
            set_props("place-custom-order-modal", {"is_open": False})
            return dash.no_update, dash.no_update, dash.no_update
        if dash_ctx.triggered_id != "place-order-button":
            raise PreventUpdate
        if not is_action_allowed(state, "custom_order"):
            raise PreventUpdate
        quantities = [row.get("order_qty") for row in (row_data or [])]
        changed = place_custom_orders(state, list(quantities))
        if changed:
            record_custom_order(state)
        ctx.persist_state(session_id, state)
        grid = build_custom_order_grid(state, theme_name)
        set_props("place-custom-order-modal", {"is_open": False})
        if isinstance(grid, dbc.Alert):
            return [], [], ctx.next_session_revision(session_revision)
        return grid.rowData, grid.columnDefs, ctx.next_session_revision(session_revision)

    @app.callback(
        [
            Output("output-item-upload", "children"),
            Output("upload-preview-data", "data"),
            Output("import-uploaded-items", "style"),
        ],
        Input("upload-item", "contents"),
        State("upload-item", "filename"),
        State("upload-item", "last_modified"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def update_output(list_of_contents, list_of_names, list_of_dates, theme):
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
            card = parse_contents(contents, name, modified, ctx.theme_name(theme))
            cards.append(card)
            if isinstance(card, dbc.Alert):
                errors.append(f"{name}: {card.children}")
                continue
            try:
                frames.append(coerce_uploaded(read_uploaded_table(contents, name)))
            except Exception as exc:  # pragma: no cover - defensive UI guard
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
            Output("session-revision", "data", allow_duplicate=True),
            Output("upload-feedback", "children"),
        ],
        Input("import-uploaded-items", "n_clicks"),
        State("upload-preview-data", "data"),
        State("user-data-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def import_uploaded_items(n_clicks, rows, client_data, session_revision):
        if not n_clicks:
            raise PreventUpdate
        if not rows:
            return dash.no_update, dbc.Alert("No data to import.", color="warning")
        session_id, state = ctx.require_session(client_data)
        if not is_action_allowed(state, "import_items"):
            return dash.no_update, dbc.Alert("Imports unlock in simulator mode.", color="warning")
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
        ctx.persist_state(session_id, state)
        return ctx.next_session_revision(session_revision), dbc.Alert(
            f"Imported {len(df)} items successfully.",
            color="success",
        )

    @app.callback(
        [
            Output("po-overview-grid", "rowData"),
            Output("po-overview-grid", "columnDefs"),
            Output("po-overview-grid", "selectedRows"),
            Output("session-revision", "data", allow_duplicate=True),
        ],
        [
            Input("po-overview-button", "n_clicks"),
            Input("po-overview-close", "n_clicks"),
            Input("po-expedite-button", "n_clicks"),
            Input("po-cancel-button", "n_clicks"),
        ],
        State("po-overview-grid", "selectedRows"),
        State("user-data-store", "data"),
        State("theme-store", "data"),
        State("session-revision", "data"),
        prevent_initial_call=True,
    )
    def po_overview_handler(
        _open_clicks,
        _close_clicks,
        _expedite_clicks,
        _cancel_clicks,
        selected_rows,
        client_data,
        theme,
        session_revision,
    ):
        session_id, state = ctx.require_session(client_data)
        theme_name = ctx.theme_name(theme)
        if dash_ctx.triggered_id == "po-overview-button":
            if not is_action_allowed(state, "po_overview"):
                raise PreventUpdate
            grid = build_po_overview_grid(state, theme_name)
            set_props("po-overview-modal", {"is_open": True})
            if isinstance(grid, dbc.Alert):
                return [], [], [], dash.no_update
            return grid.rowData, grid.columnDefs, [], dash.no_update
        if dash_ctx.triggered_id == "po-overview-close":
            set_props("po-overview-modal", {"is_open": False})
            return dash.no_update, dash.no_update, [], dash.no_update
        if dash_ctx.triggered_id not in {"po-expedite-button", "po-cancel-button"}:
            raise PreventUpdate
        if not selected_rows:
            raise PreventUpdate
        action = "expedite" if dash_ctx.triggered_id == "po-expedite-button" else "cancel"
        action_name = "expedite_receipt" if action == "expedite" else "cancel_receipt"
        if not is_action_allowed(state, action_name):
            raise PreventUpdate
        receipt_ids = [
            str(selected_row.get("receipt_id"))
            for selected_row in selected_rows
            if selected_row.get("receipt_id")
        ]
        if not receipt_ids:
            raise PreventUpdate
        if expedite_or_cancel_receipts(state, receipt_ids, action) <= 0:
            raise PreventUpdate
        ctx.persist_state(session_id, state)
        grid = build_po_overview_grid(state, theme_name)
        if isinstance(grid, dbc.Alert):
            return [], [], [], ctx.next_session_revision(session_revision)
        return grid.rowData, grid.columnDefs, [], ctx.next_session_revision(session_revision)
