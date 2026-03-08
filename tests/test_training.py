from __future__ import annotations

from imsim.services.simulation import tick_state
from imsim.services.training import (
    academy_level,
    apply_lesson_evaluation,
    build_level_state,
    build_simulator_state,
    evaluate_active_lesson,
    is_action_allowed,
    reset_progress_state,
)
from imsim.ui.components import build_inventory_figure


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

    assert figure.layout.title.text == "On-hand inventory over time"
    assert [trace.name for trace in figure.data] == ["On Hand", "Backorder"]


def test_level_two_uses_simple_quantity_graph():
    state = build_level_state("level-2")

    figure = build_inventory_figure(state)

    assert figure.layout.title.text == "Basic reorder quantities over time"
    assert [trace.name for trace in figure.data] == ["On Hand", "On Order", "PNA", "Backorder"]


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
