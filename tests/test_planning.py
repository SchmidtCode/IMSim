from __future__ import annotations

from imsim.models import GlobalSettings, SimulationState
from imsim.services.asq import apply_asq_month_end
from imsim.services.planning import (
    calculate_soq,
    calculate_surplus_line,
    create_inventory_item,
    critical_point_qty,
    daily_usage_from_monthly,
    lead_time_demand,
    review_cycle_demand,
    round_to_pack,
    round_up_to_pack,
    safety_stock_qty,
)
from imsim.services.simulation import tick_state


def test_pack_rounding():
    assert round_to_pack(13, 5) == 15
    assert round_up_to_pack(11, 5) == 15


def test_day_basis_defaults_to_30():
    assert GlobalSettings().day_basis == 30


def test_day_basis_helpers_use_the_configured_basis():
    settings = GlobalSettings(day_basis=28, r_cycle=14)

    assert daily_usage_from_monthly(56, settings.day_basis) == 2
    assert lead_time_demand(56, 14, settings.day_basis) == 28
    assert review_cycle_demand(56, settings.r_cycle, settings.day_basis) == 28
    assert safety_stock_qty(56, 14, 0.5, settings.day_basis) == 14
    assert critical_point_qty(56, 14, settings.day_basis) == 28


def test_soq_and_pna_are_computed_for_new_item():
    settings = GlobalSettings(r_cycle=14, r_cost=8, k_cost=0.18)
    item = create_inventory_item(
        usage_rate=30,
        lead_time=30,
        item_cost=20,
        pna=5,
        safety_allowance=0.5,
        standard_pack=5,
        global_settings=settings,
        hits_per_month=10,
    )
    assert item.pna == 5
    assert item.op > 0
    assert item.lp > item.op
    assert item.soq >= item.standard_pack


def test_surplus_line_uses_oq_not_eoq():
    assert calculate_surplus_line(40, 12) == 52


def test_soq_branch_logic_is_unchanged():
    settings = GlobalSettings(r_cycle=14, r_cost=8, k_cost=0.18)
    item = create_inventory_item(
        usage_rate=30,
        lead_time=30,
        item_cost=20,
        pna=5,
        safety_allowance=0.5,
        standard_pack=5,
        global_settings=settings,
        hits_per_month=10,
    )
    item.op = 20
    item.lp = 30
    item.oq = 12

    item.pna = 31
    assert calculate_soq(item) == 0

    item.pna = 25
    assert calculate_soq(item) == 15

    item.pna = 18
    assert calculate_soq(item) == 15


def test_asq_month_end_can_raise_op():
    state = SimulationState()
    state.items.append(
        create_inventory_item(
            usage_rate=40,
            lead_time=30,
            item_cost=100,
            pna=10,
            safety_allowance=0.1,
            standard_pack=1,
            global_settings=state.global_settings,
            hits_per_month=10,
        )
    )
    item = state.items[0]
    baseline = item.op
    item.asq_hits_period = 4
    item.asq_usage_period = baseline * 6
    summary = apply_asq_month_end(state)
    assert summary["changed"] == 1
    assert item.op > baseline
    assert item.asq_hits_period == 0


def test_tick_state_updates_inventory_and_cost_buckets(monkeypatch):
    state = SimulationState()
    item = create_inventory_item(
        usage_rate=30,
        lead_time=10,
        item_cost=50,
        pna=10,
        safety_allowance=0.2,
        standard_pack=1,
        global_settings=state.global_settings,
        hits_per_month=10,
    )
    state.items.append(item)
    state.is_initialized = True

    monkeypatch.setattr("imsim.services.simulation.simulate_daily_hits", lambda _hpm: 1)
    monkeypatch.setattr("imsim.services.simulation.simulate_sales", lambda _avg, _pack: 12)

    summary = tick_state(state)

    assert summary["day"] == 2
    assert state.day == 2
    assert state.service_today.orders == 1
    assert state.service_today.units_shipped == 10
    assert state.service_today.units_backordered == 2
    assert state.costs.stockout > 0
    assert state.sales.cogs > 0
