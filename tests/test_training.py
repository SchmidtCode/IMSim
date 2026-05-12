from __future__ import annotations

from dash import Patch
from dash_ag_grid import AgGrid

from imsim.models import Receipt
from imsim.services import simulation as simulation_service
from imsim.services.simulation import (
    expedite_or_cancel_receipt,
    expedite_or_cancel_receipts,
    place_purchase_orders,
    tick_state,
)
from imsim.services.training import (
    academy_level,
    apply_lesson_evaluation,
    build_level_state,
    build_simulator_state,
    evaluate_active_lesson,
    is_action_allowed,
    reset_progress_state,
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
    refresh_inventory_figure,
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
    state.is_initialized = True

    summary = tick_state(state)

    assert summary["day"] == 2
    assert state.service_today.orders == 1
    assert state.service_today.units_ordered == item.usage_rate / 30.0
    assert state.items[0].on_hand == opening_on_hand - (item.usage_rate / 30.0)
    assert [(point.day, point.total_on_hand, point.total_backorder) for point in state.history] == [
        (1, opening_on_hand, 0.0),
        (2, opening_on_hand - (item.usage_rate / 30.0), 0.0),
    ]


def test_level_one_uses_on_hand_lesson_graph():
    state = build_level_state("level-1")

    figure = build_inventory_figure(state)

    assert figure.layout.title.text is None
    assert figure.layout.margin.to_plotly_json() == {"l": 24.0, "r": 24.0, "t": 19.2, "b": 24.0}
    assert [trace.name for trace in figure.data] == ["On Hand", "Backorder"]
    assert figure.data[0].line.width == 3.0
    assert figure.data[0].marker.size == 8.0
    assert figure.data[1].marker.size == 7.0


def test_level_two_uses_simple_quantity_graph():
    state = build_level_state("level-2")

    figure = build_inventory_figure(state)

    assert figure.layout.title.text is None
    assert [trace.name for trace in figure.data] == ["On Hand", "On Order", "PNA", "Backorder"]
    assert all(trace.line.width == 3.0 for trace in figure.data)
    assert figure.data[-1].marker.size == 7.0
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.uirevision == "lesson-2:light"
    assert figure.layout.meta["figure_kind"] == "lesson-2"
    assert figure.layout.meta["theme"] == "light"
    assert figure.layout.meta["layout_signature"] == "static"


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
    assert figure.layout.meta["layout_signature"].startswith("items:")


def test_refresh_inventory_figure_returns_patch_when_schema_is_stable():
    state = build_level_state("level-1")
    current_figure = build_inventory_figure(state).to_plotly_json()

    patch = refresh_inventory_figure(state, current_figure=current_figure)

    assert isinstance(patch, Patch)
    operations = patch.to_plotly_json()["operations"]
    assert any(op["location"] == ["data", 0, "x"] for op in operations)
    assert not any(op["location"][0] == "layout" for op in operations)


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
        "daily_usage",
        "on_hand",
        "on_order",
        "backorder",
        "pna",
        "op",
        "soq",
    ]
    assert grid.columnDefs[0]["pinned"] == "left"


def test_custom_order_and_po_overview_use_ag_grid():
    state = build_simulator_state()
    custom_order_grid = build_custom_order_grid(state)
    place_purchase_orders(state)
    po_grid = build_po_overview_grid(state)

    assert isinstance(custom_order_grid, AgGrid)
    assert custom_order_grid.id == "custom-order-grid"
    assert custom_order_grid.rowData[0]["order_qty"] >= 0
    assert isinstance(po_grid, AgGrid)
    assert po_grid.id == "po-overview-grid"
    assert po_grid.rowData[0]["receipt_id"]
    assert po_grid.columnDefs[1]["headerName"] == "PO Line"
    assert po_grid.dashGridOptions["rowSelection"]["mode"] == "multiRow"
    assert po_grid.dashGridOptions["rowSelection"]["enableClickSelection"] is True
    assert po_grid.dashGridOptions["rowSelection"]["enableSelectionWithoutKeys"] is True
    assert po_grid.dashGridOptions["rowSelection"]["checkboxes"] is True
    assert po_grid.dashGridOptions["rowSelection"]["headerCheckbox"] is True


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
    state.training.guided_orders_placed = 1
    for item in state.items:
        item.on_hand = item.op

    evaluation = evaluate_active_lesson(state)

    assert evaluation is not None
    assert evaluation.completed is True
    assert evaluation.passed is True


def test_final_lesson_pass_unlocks_simulator_reward():
    state = build_level_state("level-7")
    level = academy_level("level-7")
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
    assert "level-7" in state.training.completed_levels
    assert state.training.current_view == "lesson"
    assert state.training.active_level_id == "level-7"
