from __future__ import annotations

from dash import Patch, html
from dash_ag_grid import AgGrid

from imsim.models import Receipt, TrainingProfile
from imsim.services import simulation as simulation_service
from imsim.services.simulation import (
    expedite_or_cancel_receipt,
    expedite_or_cancel_receipts,
    place_custom_orders,
    place_purchase_orders,
    tick_state,
)
from imsim.services.training import (
    academy_level,
    academy_levels,
    active_layout_variant,
    apply_lesson_evaluation,
    build_level_state,
    build_simulator_state,
    cheat_unlock_password_matches,
    evaluate_active_lesson,
    final_academy_level,
    is_action_allowed,
    record_guided_order,
    reset_progress_state,
    unlock_all_academy_levels,
)
from imsim.ui.components import (
    _plot_base_layout,
    _plot_line,
    _plot_marker,
    _plot_marker_outline,
    build_custom_order_grid,
    build_inventory_figure,
    build_inventory_table,
    build_po_overview_grid,
    custom_order_grid_options,
    inventory_graph_style,
    lesson_compact_summary_children,
    lesson_objective_children,
    lesson_tutorial_children,
    refresh_inventory_figure,
    service_card_children,
)


def test_build_level_state_preserves_progress_and_opens_lesson():
    baseline = reset_progress_state()
    baseline.training.completed_levels = ["level-1"]
    baseline.training.highest_unlocked_level = 2

    state = build_level_state("level-2", baseline.training)

    assert state.training.current_view == "lesson"
    assert state.training.active_level_id == "level-2"
    assert state.training.completed_levels == ["level-1"]
    assert state.training.highest_unlocked_level == 2
    assert len(state.items) == 1
    assert state.global_settings.auto_po_enabled is False


def test_simulator_unlock_and_auto_po_permissions():
    locked_state = reset_progress_state()
    assert is_action_allowed(locked_state, "auto_po") is False

    simulator_state = build_simulator_state(locked_state.training)
    assert is_action_allowed(simulator_state, "auto_po") is False

    simulator_state.training.simulator_unlocked = True
    simulator_state.training.auto_po_reward_unlocked = True
    assert is_action_allowed(simulator_state, "auto_po") is True


def test_tick_state_uses_deterministic_training_demand():
    state = build_level_state("level-1")
    item = state.items[0]
    opening_on_hand = item.on_hand
    daily_usage = item.usage_rate / state.global_settings.day_basis
    state.is_initialized = True

    summary = tick_state(state)

    assert summary["day"] == 2
    assert state.service_today.orders == 1
    assert state.service_today.units_ordered == daily_usage
    assert state.items[0].on_hand == opening_on_hand - daily_usage
    assert [(point.day, point.total_on_hand, point.total_backorder) for point in state.history] == [
        (1, opening_on_hand, 0.0),
        (2, opening_on_hand - daily_usage, 0.0),
    ]


def test_level_one_uses_on_hand_lesson_graph():
    state = build_level_state("level-1")

    figure = build_inventory_figure(state)

    assert figure.layout.title.text is None
    assert figure.layout.height == 340
    assert figure.layout.margin.to_plotly_json() == {"l": 24.0, "r": 24.0, "t": 24.0, "b": 38.4}
    assert [trace.name for trace in figure.data] == ["On Hand", "Backorder"]
    assert figure.data[0].line.width == 3.0
    assert figure.data[0].marker.size == 8.0
    assert figure.data[1].marker.size == 7.0


def test_level_two_uses_simple_quantity_graph():
    state = build_level_state("level-2")

    figure = build_inventory_figure(state)

    assert figure.layout.title.text is None
    assert figure.layout.height == 392
    assert [trace.name for trace in figure.data] == [
        "On Hand",
        "On Order",
        "PNA",
        "OP",
        "Backorder",
    ]
    assert all(trace.line.width == 3.0 for trace in figure.data)
    assert figure.data[-1].marker.size == 7.0
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.uirevision == "lesson-2:light"
    assert figure.layout.meta["figure_kind"] == "lesson-2"
    assert figure.layout.meta["theme"] == "light"
    assert figure.layout.meta["layout_signature"] == "static"


def test_level_three_uses_fill_rate_visual():
    state = build_level_state("level-3")

    figure = build_inventory_figure(state)

    assert figure.layout.height == 400
    assert figure.layout.meta["figure_kind"] == "lesson-3-fill-rate"
    assert [trace.type for trace in figure.data] == ["indicator", "bar"]
    assert "guided_po" in academy_level("level-3").allowed_actions


def test_signal_map_uses_stable_schema_and_uirevision():
    state = build_simulator_state()
    state.items[0].pipeline.append(Receipt(receipt_id="abc123", qty=5, eta_day=5))
    state.items[0].stockout_today = True

    figure = build_inventory_figure(state, theme="dark")

    assert figure.layout.title.text is None
    assert [trace.name for trace in figure.data] == [
        "PNA",
        "PNA + SOQ",
        "Available to Sell",
        "0 PNA",
        "Stockout Today",
    ]
    assert figure.layout.hovermode == "closest"
    assert figure.layout.uirevision == "signal-map:dark"
    assert figure.layout.meta["figure_kind"] == "signal-map"
    assert figure.layout.meta["theme"] == "dark"
    assert figure.layout.meta["layout_signature"].startswith("simulator:items:")


def test_workspace_variants_drive_figure_and_grid_sizes():
    basic_state = build_level_state("level-3")
    intro_pna_state = build_level_state("level-6")
    ranking_state = build_level_state("level-11")
    review_cycle_state = build_level_state("level-12")
    line_point_state = build_level_state("level-13")
    signal_state = build_level_state("level-10")
    exception_state = build_level_state("level-17")
    certification_state = build_level_state("level-18")
    simulator_state = build_simulator_state()

    basic_grid = build_inventory_table(basic_state)
    intro_pna_grid = build_inventory_table(intro_pna_state)
    intro_pna_figure = build_inventory_figure(intro_pna_state)
    ranking_grid = build_inventory_table(ranking_state)
    line_point_grid = build_inventory_table(line_point_state)
    review_cycle_figure = build_inventory_figure(review_cycle_state)
    line_point_figure = build_inventory_figure(line_point_state)
    signal_grid = build_inventory_table(signal_state)
    signal_figure = build_inventory_figure(signal_state)
    exception_figure = build_inventory_figure(exception_state)
    certification_grid = build_inventory_table(certification_state)
    certification_figure = build_inventory_figure(certification_state)
    simulator_grid = build_inventory_table(simulator_state)
    simulator_figure = build_inventory_figure(simulator_state)

    assert isinstance(basic_grid, AgGrid)
    assert basic_grid.style["height"] == "auto"
    assert basic_grid.dashGridOptions["domLayout"] == "autoHeight"
    assert isinstance(intro_pna_grid, AgGrid)
    assert intro_pna_grid.style["height"] == "auto"
    assert intro_pna_grid.dashGridOptions["domLayout"] == "autoHeight"
    assert intro_pna_figure.layout.height == 340
    assert isinstance(ranking_grid, AgGrid)
    assert ranking_grid.style["height"] == "auto"
    assert ranking_grid.dashGridOptions["domLayout"] == "autoHeight"
    assert isinstance(line_point_grid, AgGrid)
    assert line_point_grid.style["height"] == "auto"
    assert line_point_grid.dashGridOptions["domLayout"] == "autoHeight"
    assert [column["field"] for column in line_point_grid.columnDefs] == [
        "item",
        "pna",
        "op",
        "lp",
        "days_to_op",
        "soq",
    ]
    assert review_cycle_figure.layout.height == 340
    assert line_point_figure.layout.height == 340
    assert isinstance(signal_grid, AgGrid)
    assert signal_grid.style["height"] == "auto"
    assert signal_grid.dashGridOptions["domLayout"] == "autoHeight"
    assert signal_figure.layout.height == 300
    assert exception_figure.layout.height == 360
    assert isinstance(certification_grid, AgGrid)
    assert certification_grid.style["height"] == "auto"
    assert certification_grid.dashGridOptions["domLayout"] == "autoHeight"
    assert certification_grid.columnDefs[-1]["field"] == "soq"
    assert certification_grid.columnDefs[-1]["minWidth"] == 96
    assert certification_figure.layout.height == 360
    assert isinstance(simulator_grid, AgGrid)
    assert simulator_grid.style["height"] == "28rem"
    assert simulator_grid.dashGridOptions["pagination"] is True
    assert simulator_grid.columnDefs[-1]["field"] == "soq"
    assert simulator_grid.columnDefs[-1]["minWidth"] == 96
    assert simulator_figure.layout.height == 460
    assert certification_figure.layout.meta["layout_signature"].startswith(
        "workspace_certification:items:"
    )


def test_academy_lesson_inventory_grids_use_compact_height():
    for level_index in range(1, 19):
        state = build_level_state(f"level-{level_index}")
        grid = build_inventory_table(state)

        assert isinstance(grid, AgGrid)
        assert grid.style["height"] == "auto"
        assert grid.dashGridOptions["domLayout"] == "autoHeight"


def test_level_seventeen_uses_exception_boundary_figure():
    state = build_level_state("level-17")

    figure = build_inventory_figure(state)

    assert figure.layout.height == 360
    assert figure.layout.yaxis.title.text == "Units"
    assert figure.layout.uirevision == "exception-map:light"
    assert figure.layout.meta["figure_kind"] == "exception-map"
    assert [trace.name for trace in figure.data] == [
        "PNA",
        "Critical Point",
        "Surplus Threshold",
        "On Hand",
    ]


def test_inventory_graph_style_tracks_figure_height():
    lesson_state = build_level_state("level-2")
    simulator_state = build_simulator_state()

    assert inventory_graph_style(lesson_state) == {
        "height": "392px",
        "minHeight": "392px",
        "width": "100%",
    }
    assert inventory_graph_style(simulator_state) == {
        "height": "460px",
        "minHeight": "460px",
        "width": "100%",
    }


def test_lesson_snapshot_is_collapsed_for_supporting_lessons():
    level_two_children = service_card_children(build_level_state("level-2"))
    level_three_children = service_card_children(build_level_state("level-3"))
    level_six_children = service_card_children(build_level_state("level-6"))
    level_one_children = service_card_children(build_level_state("level-1"))

    assert isinstance(level_two_children[0], html.Details)
    assert getattr(level_two_children[0], "open", None) is None
    assert isinstance(level_three_children[0], html.Details)
    assert getattr(level_three_children[0], "open", None) is None
    assert isinstance(level_six_children[0], html.Details)
    assert getattr(level_six_children[0], "open", None) is None
    assert not isinstance(level_one_children[0], html.Details)


def test_refresh_inventory_figure_returns_patch_when_schema_is_stable():
    state = build_level_state("level-1")
    current_figure = build_inventory_figure(state).to_plotly_json()

    patch = refresh_inventory_figure(state, current_figure=current_figure)

    assert isinstance(patch, Patch)
    operations = patch.to_plotly_json()["operations"]
    assert any(op["location"] == ["data", 0, "x"] for op in operations)
    assert not any(op["location"][0] == "layout" for op in operations)


def test_refresh_inventory_figure_patches_lesson_three_indicator_value():
    state = build_level_state("level-3")
    current_figure = build_inventory_figure(state).to_plotly_json()
    state.service_totals.orders = 4
    state.service_totals.orders_stockout = 1

    patch = refresh_inventory_figure(state, current_figure=current_figure)

    assert isinstance(patch, Patch)
    operations = patch.to_plotly_json()["operations"]
    assert {
        "operation": "Assign",
        "location": ["data", 0, "value"],
        "params": {"value": 75.0},
    } in operations
    assert {
        "operation": "Assign",
        "location": ["data", 1, "y"],
        "params": {"value": [3, 1]},
    } in operations
    assert {
        "operation": "Assign",
        "location": ["data", 1, "text"],
        "params": {"value": ["3", "1"]},
    } in operations


def test_refresh_inventory_figure_rebuilds_when_theme_or_schema_changes():
    lesson_one = build_level_state("level-1")
    current_figure = build_inventory_figure(lesson_one).to_plotly_json()

    themed = refresh_inventory_figure(lesson_one, theme="dark", current_figure=current_figure)
    lesson_two = refresh_inventory_figure(
        build_level_state("level-2"),
        current_figure=current_figure,
    )

    assert not isinstance(themed, Patch)
    assert not isinstance(lesson_two, Patch)
    assert themed.layout.uirevision == "lesson-1:dark"
    assert lesson_two.layout.uirevision == "lesson-2:light"


def test_inventory_table_uses_ag_grid_with_lesson_visible_columns():
    state = build_level_state("level-2")

    grid = build_inventory_table(state)

    assert isinstance(grid, AgGrid)
    assert grid.id == "inventory-table-grid"
    assert [column["field"] for column in grid.columnDefs] == [
        "item",
        "on_hand",
        "on_order",
        "backorder",
        "pna",
        "op",
    ]
    assert grid.columnDefs[0]["pinned"] == "left"
    assert grid.defaultColDef["filter"] is False
    assert grid.dashGridOptions["domLayout"] == "autoHeight"
    assert grid.style["height"] == "auto"


def test_early_ordering_snapshots_are_collapsed_by_default():
    for level_id in ("level-2", "level-3"):
        state = build_level_state(level_id)

        snapshot = service_card_children(state)[0]
        snapshot_json = snapshot.to_plotly_json()

        assert snapshot_json["type"] == "Details"
        assert snapshot_json["props"]["className"] == "lesson-snapshot-disclosure"
        assert "open" not in snapshot_json["props"]


def test_level_seven_uses_full_pna_formula_wording():
    level = academy_level("level-7")

    assert level is not None
    assert (
        level.formula == "PNA = On Hand - Reserved - Committed - Backordered + On Order + Received"
    )


def test_formula_fidelity_lessons_use_day_basis_and_updated_wording():
    lesson_five = academy_level("level-5")
    lesson_eight = academy_level("level-8")
    lesson_nine = academy_level("level-9")
    lesson_ten = academy_level("level-10")
    lesson_sixteen = academy_level("level-16")
    lesson_twelve = academy_level("level-12")
    lesson_seventeen = academy_level("level-17")

    assert lesson_five is not None
    assert lesson_eight is not None
    assert lesson_nine is not None
    assert lesson_ten is not None
    assert lesson_sixteen is not None
    assert lesson_twelve is not None
    assert lesson_seventeen is not None
    assert lesson_five.formula == "Daily usage = monthly usage rate / day basis"
    assert lesson_eight.formula == "OP = monthly usage x lead-time days / day basis"
    assert (
        lesson_nine.formula
        == "Safety stock = monthly usage x lead-time days / day basis x safety allowance"
    )
    assert lesson_ten.formula == "OP = lead-time demand + safety stock"
    assert lesson_sixteen.formula == "SOQ = OQ + shortage below OP, rounded to standard pack"
    assert lesson_twelve.title == "Product Lines and Review Cycle"
    assert lesson_seventeen.formula == (
        "Critical point = monthly usage x lead-time days / day basis; surplus threshold = LP + OQ"
    )


def test_lesson_secondary_notes_render_as_collapsed_help():
    level_sixteen_definition = academy_level("level-16")
    level_eighteen_definition = academy_level("level-18")
    level_sixteen = lesson_tutorial_children(build_level_state("level-16"))
    level_seventeen = lesson_tutorial_children(build_level_state("level-17"))
    level_eighteen = lesson_tutorial_children(build_level_state("level-18"))

    assert level_sixteen_definition is not None
    assert level_eighteen_definition is not None
    assert "SOQ is simplified for training" in level_sixteen_definition.advanced_note
    assert "training workspace" in level_eighteen_definition.csd_mapping_note
    advanced_details = [child for child in level_sixteen if isinstance(child, html.Details)]
    assert [detail.children[0].children[0].children for detail in advanced_details] == [
        "Advanced note",
        "CSD mapping",
    ]
    assert all(getattr(detail, "open", None) is None for detail in advanced_details)
    assert any(
        getattr(child, "children", None) == "Simulator simplified month: 30 days"
        for child in level_seventeen
    )
    assert any(
        isinstance(child, html.Details) and child.children[0].children[0].children == "CSD mapping"
        for child in level_seventeen
    )
    assert any(
        isinstance(child, html.Details) and child.children[0].children[0].children == "CSD mapping"
        for child in level_eighteen
    )


def test_after_overhead_rows_use_level_target_without_duplicate_objective_copy():
    state = build_level_state("level-14")
    state.sales.revenue = 1000.0
    state.sales.cogs = 600.0
    state.costs.total = 100.0

    evaluation = evaluate_active_lesson(state)
    objective_children = lesson_objective_children(state)

    assert evaluation is not None
    assert "After-overhead GM: 30.0% / target -5.0%" in evaluation.metric_rows
    objective_rows = [item.children for item in objective_children[-1].children]
    assert objective_rows.count("After-overhead GM: 30.0% / target -5.0%") == 1
    assert not any("Current after-overhead GM:" in row for row in objective_rows)


def test_compact_summary_keeps_all_goal_rows_for_three_check_lessons():
    state = build_level_state("level-18")

    compact_children = lesson_compact_summary_children(state)
    pill_text = [child.children for child in compact_children[1].children]

    assert len(pill_text) == 4
    assert "Fill rate: n/a / target 97.0%" in pill_text
    assert "After-OH GM: n/a / target 0.0%" in pill_text
    assert "Close backorder: 0" in pill_text


def test_lesson_copy_avoids_official_infor_training_language():
    banned_phrases = (
        "csd-certified",
        "infor-certified",
        "official csd training",
        "cloudsuite distribution course",
        "infor simulator",
    )

    for level in academy_levels():
        content = " ".join(
            [
                level.title,
                level.summary,
                level.formula,
                *level.tutorial_steps,
                *level.locked_features,
                level.advanced_note,
                level.csd_mapping_note,
            ]
        ).casefold()
        for phrase in banned_phrases:
            assert phrase not in content


def test_custom_order_and_po_overview_use_ag_grid():
    state = build_simulator_state()
    custom_order_grid = build_custom_order_grid(state)
    place_purchase_orders(state)
    po_grid = build_po_overview_grid(state)

    assert isinstance(custom_order_grid, AgGrid)
    assert custom_order_grid.id == "custom-order-grid"
    assert custom_order_grid.rowData[0]["order_qty"] >= 0
    order_qty_col = custom_order_grid.columnDefs[-1]
    assert order_qty_col["field"] == "order_qty"
    assert order_qty_col["editable"] is True
    assert order_qty_col["cellEditor"] == "agNumberCellEditor"
    assert order_qty_col["cellClass"] == "custom-order-qty-cell"
    assert custom_order_grid.dashGridOptions == custom_order_grid_options()
    assert custom_order_grid.dashGridOptions["singleClickEdit"] is True
    assert isinstance(po_grid, AgGrid)
    assert po_grid.id == "po-overview-grid"
    assert po_grid.rowData[0]["receipt_id"]
    assert po_grid.columnDefs[1]["headerName"] == "PO Line"
    assert po_grid.dashGridOptions["rowSelection"]["mode"] == "multiRow"
    assert po_grid.dashGridOptions["rowSelection"]["enableClickSelection"] is True
    assert po_grid.dashGridOptions["rowSelection"]["enableSelectionWithoutKeys"] is True
    assert po_grid.dashGridOptions["rowSelection"]["checkboxes"] is True
    assert po_grid.dashGridOptions["rowSelection"]["headerCheckbox"] is True
    assert custom_order_grid.style["height"] == "26rem"
    assert po_grid.style["height"] == "26rem"


def test_custom_orders_accept_user_quantities_and_ignore_bad_entries():
    state = build_simulator_state()
    first_item = state.items[0]
    first_item.standard_pack = 5

    changed = place_custom_orders(state, ["12", "", "not a number"])

    assert changed is True
    assert len(first_item.pipeline) == 1
    assert first_item.pipeline[0].qty == 10


def test_expedite_receipt_can_jump_to_one_week_when_supplier_has_stock(monkeypatch):
    class StubRng:
        def random(self):
            return 0.0

        def integers(self, low, high):
            raise AssertionError("Queue-jump expedites should not use the fallback range.")

    monkeypatch.setattr(simulation_service, "_rng", lambda: StubRng())
    state = build_simulator_state()
    item = state.items[0]
    item.lead_time = 21
    item.pipeline.append(Receipt(receipt_id="exp123", qty=10, eta_day=31))

    changed = expedite_or_cancel_receipt(state, "exp123", "expedite")

    expected_cost = 2.0 * item.item_cost * 10 * state.global_settings.expedite_rate
    assert changed is True
    assert item.pipeline[0].eta_day == state.day + 7
    assert state.costs.expedite == expected_cost
    assert state.costs.total == expected_cost


def test_expedite_receipt_can_still_make_a_smaller_random_improvement(monkeypatch):
    class StubRng:
        def random(self):
            return 0.99

        def integers(self, low, high):
            assert (low, high) == (1, 4)
            return 3

    monkeypatch.setattr(simulation_service, "_rng", lambda: StubRng())
    state = build_simulator_state()
    item = state.items[0]
    item.lead_time = 21
    item.pipeline.append(Receipt(receipt_id="exp234", qty=10, eta_day=13))

    changed = expedite_or_cancel_receipt(state, "exp234", "expedite")

    expected_cost = item.item_cost * 10 * state.global_settings.expedite_rate
    assert changed is True
    assert item.pipeline[0].eta_day == 10
    assert state.costs.expedite == expected_cost
    assert state.costs.total == expected_cost


def test_receipt_actions_handle_multiple_selected_rows():
    state = build_simulator_state()
    state.items[0].pipeline.append(Receipt(receipt_id="keep-me", qty=5, eta_day=9))
    state.items[0].pipeline.append(Receipt(receipt_id="cancel-a", qty=7, eta_day=11))
    state.items[1].pipeline.append(Receipt(receipt_id="cancel-b", qty=9, eta_day=12))

    changed = expedite_or_cancel_receipts(state, ["cancel-a", "cancel-b"], "cancel")

    assert changed == 2
    assert [receipt.receipt_id for receipt in state.items[0].pipeline] == ["keep-me"]
    assert state.items[1].pipeline == []


def test_long_lead_expedite_queue_jump_stays_possible(monkeypatch):
    class StubRng:
        def random(self):
            return 0.01

        def integers(self, low, high):
            raise AssertionError("Queue jump should win before fallback for this test.")

    monkeypatch.setattr(simulation_service, "_rng", lambda: StubRng())
    state = build_simulator_state()
    item = state.items[0]
    item.lead_time = 700
    item.pipeline.append(Receipt(receipt_id="long-hop", qty=4, eta_day=701))

    changed = expedite_or_cancel_receipt(state, "long-hop", "expedite")

    assert changed is True
    assert item.pipeline[0].eta_day == state.day + 7


def test_plot_sizing_helpers_keep_default_plot_dimensions():
    colors = {
        "plot_bg": "#0c1523",
        "surface": "#17273d",
        "text": "#e6eefc",
        "line": "rgba(214, 228, 255, 0.14)",
    }

    layout = _plot_base_layout("Inventory Signal Map", colors)

    assert layout["title"] == {
        "text": "Inventory Signal Map",
        "font": {"color": "#e6eefc", "size": 24.0},
    }
    assert layout["margin"] == {"l": 24.0, "r": 24.0, "t": 56.0, "b": 24.0}
    assert _plot_line("#2dd4bf") == {"width": 3.0, "color": "#2dd4bf"}
    assert _plot_marker("#2dd4bf", 0.5) == {"size": 8.0, "color": "#2dd4bf"}
    assert _plot_marker_outline("#fb923c") == {"width": 2.0, "color": "#fb923c"}


def test_level_one_passes_when_inventory_depletes_and_backorder_appears():
    state = build_level_state("level-1")
    level = academy_level("level-1")
    assert level is not None
    state.day = level.day_window + 1
    state.items[0].on_hand = 0
    state.items[0].backorder = 4

    evaluation = evaluate_active_lesson(state)

    assert evaluation is not None
    assert evaluation.completed is True
    assert evaluation.passed is True


def test_level_one_tick_completes_on_displayed_day_twenty():
    state = build_level_state("level-1")
    state.is_initialized = True

    for _ in range(19):
        summary = tick_state(state)

    assert state.day == 20
    assert summary["lesson_completed"] == 1
    assert state.is_initialized is False
    assert state.training.lesson_status == "passed"


def test_training_tick_forces_auto_po_back_off():
    state = build_level_state("level-2")
    state.is_initialized = True
    state.global_settings.auto_po_enabled = True

    tick_state(state)

    assert state.global_settings.auto_po_enabled is False


def test_level_two_requires_guided_order_to_pass():
    state = build_level_state("level-2")
    level = academy_level("level-2")
    assert level is not None
    state.day = level.day_window + 1
    state.training.guided_orders_below_op = 1
    for item in state.items:
        item.on_hand = item.op

    evaluation = evaluate_active_lesson(state)

    assert evaluation is not None
    assert evaluation.completed is True
    assert evaluation.passed is True


def test_academy_expands_to_eighteen_lessons():
    levels = academy_levels()

    assert len(levels) == 18
    assert levels[2].title == "Customer Promise: Fill Rate"
    assert levels[-1].level_id == "level-18"


def test_lead_time_lesson_requires_on_order_inventory():
    state = build_level_state("level-6")
    level = academy_level("level-6")
    assert level is not None
    state.day = level.day_window + 1
    state.training.guided_orders_placed = 1

    evaluation = evaluate_active_lesson(state)
    assert evaluation is not None
    assert evaluation.completed is True
    assert evaluation.passed is False

    state.items[0].pipeline.append(Receipt(receipt_id="lead", qty=5, eta_day=state.day + 4))
    evaluation = evaluate_active_lesson(state)

    assert evaluation is not None
    assert evaluation.passed is True


def test_line_point_and_exception_lessons_track_new_objectives():
    line_point = build_level_state("level-13")
    line_level = academy_level("level-13")
    assert line_level is not None
    assert line_level.day_window == 40
    assert line_level.win_conditions == {
        "guided_order_below_op_item_min": 1,
        "guided_order_below_lp_item_min": 3,
        "guided_order_below_lp_min": 2,
    }
    line_point.day = line_level.day_window + 1

    evaluation = evaluate_active_lesson(line_point)

    assert evaluation is not None
    assert evaluation.passed is False
    assert (
        "Guided reorders while 1+ item below OP and 3+ total below LP: 0/2"
        in evaluation.metric_rows
    )

    line_point.training.guided_orders_below_lp = 1
    evaluation = evaluate_active_lesson(line_point)

    assert evaluation is not None
    assert evaluation.passed is False

    line_point.training.guided_orders_below_lp = 2
    evaluation = evaluate_active_lesson(line_point)

    assert evaluation is not None
    assert evaluation.passed is True

    exceptions = build_level_state("level-17")
    exception_level = academy_level("level-17")
    assert exception_level is not None
    exceptions.day = exception_level.day_window + 1

    evaluation = evaluate_active_lesson(exceptions)

    assert evaluation is not None
    assert evaluation.passed is True
    assert any("critical point" in row for row in evaluation.metric_rows)
    assert any("surplus threshold" in row for row in evaluation.metric_rows)


def test_line_point_qualified_guided_reorder_requires_op_and_lp_signals():
    state = build_level_state("level-13")

    record_guided_order(state, below_op=True, below_op_count=1, below_lp_count=2)
    assert state.training.guided_orders_below_lp == 0

    record_guided_order(state, below_op=False, below_op_count=0, below_lp_count=3)
    assert state.training.guided_orders_below_lp == 0

    record_guided_order(state, below_op=True, below_op_count=1, below_lp_count=3)
    assert state.training.guided_orders_below_lp == 1


def test_final_lesson_pass_unlocks_simulator_reward():
    state = build_level_state("level-18")
    level = academy_level("level-18")
    assert level is not None
    state.day = level.day_window + 1
    state.service_totals.orders = 100
    state.service_totals.orders_stockout = 0
    state.sales.revenue = 1000.0
    state.sales.cogs = 600.0
    state.costs.total = 100.0
    for item in state.items:
        item.backorder = 0

    evaluation = evaluate_active_lesson(state)
    assert evaluation is not None
    assert evaluation.completed is True
    assert evaluation.passed is True

    apply_lesson_evaluation(state, evaluation)

    assert state.training.simulator_unlocked is True
    assert state.training.auto_po_reward_unlocked is True
    assert "level-18" in state.training.completed_levels
    assert state.training.current_view == "lesson"
    assert state.training.active_level_id == "level-18"


def test_simulator_state_uses_final_academy_lesson_seed():
    simulator_state = build_simulator_state()
    certification = final_academy_level()

    assert certification.level_id == "level-18"
    assert len(simulator_state.items) == len(certification.scenario)
    assert simulator_state.global_settings.asq.enabled == certification.global_settings.asq.enabled


def test_training_profile_migrates_legacy_certification_progress():
    legacy = TrainingProfile.from_dict(
        {
            "training_schema_version": 2,
            "active_level_id": "level-8",
            "highest_unlocked_level": 8,
            "completed_levels": ["level-1", "level-8"],
            "simulator_unlocked": True,
            "auto_po_reward_unlocked": True,
        }
    )

    assert legacy.training_schema_version == 3
    assert legacy.active_level_id == "level-18"
    assert legacy.highest_unlocked_level == 18
    assert legacy.completed_levels == ["level-1", "level-18"]
    assert legacy.simulator_unlocked is True
    assert legacy.auto_po_reward_unlocked is True


def test_cheat_unlock_opens_all_levels_without_completing_lessons():
    profile = TrainingProfile(completed_levels=["level-1"])

    unlock_all_academy_levels(profile)

    assert profile.highest_unlocked_level == len(academy_levels())
    assert profile.completed_levels == ["level-1"]
    assert profile.simulator_unlocked is True
    assert profile.auto_po_reward_unlocked is True
    assert profile.last_result_title == "Academy unlocked"
    assert cheat_unlock_password_matches("  spreadsheets   rule  ") is True
    assert cheat_unlock_password_matches("spreadsheet vibes") is False


def test_active_layout_variant_tracks_bridge_and_certification():
    assert active_layout_variant(build_level_state("level-3")) == "workspace_basic"
    assert active_layout_variant(build_level_state("level-10")) == "workspace_signal"
    assert active_layout_variant(build_level_state("level-11")) == "workspace_basic"
    assert active_layout_variant(build_level_state("level-17")) == "workspace_signal"
    assert active_layout_variant(build_level_state("level-15")) == "workspace_advanced"
    assert active_layout_variant(build_level_state("level-18")) == "workspace_certification"
