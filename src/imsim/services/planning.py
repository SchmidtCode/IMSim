from __future__ import annotations

import math

from ..models import GlobalSettings, InventoryItem


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        return numerator / denominator if denominator not in (0, 0.0, None) else default
    except Exception:
        return default


def calculate_op(usage_rate: float, lead_time_days: float, safety_allowance: float) -> float:
    monthly_lt = lead_time_days / 30.0
    safety_stock = usage_rate * monthly_lt * safety_allowance
    return usage_rate * monthly_lt + safety_stock


def calculate_lp(usage_rate: float, global_settings: GlobalSettings, op: float) -> float:
    return op + (usage_rate * (global_settings.r_cycle / 30.0))


def calculate_eoq(usage_rate: float, item_cost: float, global_settings: GlobalSettings) -> float:
    annual_demand = usage_rate * 12.0
    annual_holding_cost = max(1e-9, item_cost * global_settings.k_cost)
    return math.sqrt((2.0 * annual_demand * global_settings.r_cost) / annual_holding_cost)


def calculate_oq(eoq: float, usage_rate: float, global_settings: GlobalSettings) -> float:
    review_cycle_qty = usage_rate * (global_settings.r_cycle / 30.0)
    annual_cap = usage_rate * 12.0
    return max(review_cycle_qty, min(eoq, annual_cap))


def round_up_to_pack(qty: float, pack: float) -> float:
    pack = max(1.0, float(pack))
    return math.ceil(max(0.0, qty) / pack) * pack


def round_down_to_pack(qty: float, pack: float) -> float:
    pack = max(1.0, float(pack))
    return math.floor(max(0.0, qty) / pack) * pack


def round_to_pack(qty: float, pack: float) -> float:
    pack = max(1.0, float(pack))
    return round(max(0.0, qty) / pack) * pack


def calculate_soq(item: InventoryItem) -> float:
    if item.pna > item.lp:
        return 0.0
    if item.op < item.pna <= item.lp:
        return round_up_to_pack(item.oq, item.standard_pack)
    return round_up_to_pack(item.oq + max(0.0, item.op - item.pna), item.standard_pack)


def calculate_surplus_line(lp: float, eoq: float) -> float:
    return lp + eoq


def calculate_critical_point(usage_rate: float, lead_time_days: float) -> float:
    return usage_rate * (lead_time_days / 30.0)


def item_on_order(item: InventoryItem) -> float:
    return sum(receipt.qty for receipt in item.pipeline)


def compute_item_pna(item: InventoryItem) -> float:
    return float(item.on_hand) + float(item_on_order(item)) - float(item.backorder)


def update_planning_fields(item: InventoryItem, global_settings: GlobalSettings) -> InventoryItem:
    item.pna = compute_item_pna(item)
    item.daily_ur = safe_div(item.usage_rate, 30.0, 0.0)
    item.surplus_line = calculate_surplus_line(item.lp, item.eoq)
    item.cp = calculate_critical_point(item.usage_rate, item.lead_time)
    item.soq = calculate_soq(item)
    item.proposed_pna = item.pna + item.soq
    item.pna_days = safe_div(item.pna, item.daily_ur, 0.0)
    item.pna_days_frm_op = safe_div(item.pna - item.op, item.daily_ur, 0.0)
    item.pro_pna_days_frm_op = safe_div(item.proposed_pna - item.op, item.daily_ur, 0.0)
    item.no_pna_days_frm_op = safe_div(0.0 - item.op, item.daily_ur, 0.0)
    item.ats_days_to_stockout = safe_div(item.on_hand, item.daily_ur, 0.0)
    item.ats_days_frm_op = safe_div(item.on_hand - item.op, item.daily_ur, 0.0)
    return item


def ensure_item_physical_defaults(
    item: InventoryItem, global_settings: GlobalSettings
) -> InventoryItem:
    if item.on_hand <= 0 and item.pna > 0:
        item.on_hand = round_to_pack(item.pna, item.standard_pack)
    if item.op <= 0:
        item.op = calculate_op(item.usage_rate, item.lead_time, item.safety_allowance)
    if item.lp <= 0:
        item.lp = calculate_lp(item.usage_rate, global_settings, item.op)
    if item.eoq <= 0:
        item.eoq = calculate_eoq(item.usage_rate, item.item_cost, global_settings)
    if item.oq <= 0:
        item.oq = calculate_oq(item.eoq, item.usage_rate, global_settings)
    if item.op_base_raw <= 0:
        item.op_base_raw = item.op
    return update_planning_fields(item, global_settings)


def create_inventory_item(
    usage_rate: float,
    lead_time: float,
    item_cost: float,
    pna: float,
    safety_allowance: float,
    standard_pack: float,
    global_settings: GlobalSettings,
    hits_per_month: float,
) -> InventoryItem:
    item = InventoryItem(
        usage_rate=max(0.0, float(usage_rate)),
        lead_time=max(0.0, float(lead_time)),
        item_cost=max(0.0, float(item_cost)),
        safety_allowance=max(0.0, float(safety_allowance)),
        standard_pack=max(1.0, float(standard_pack)),
        hits_per_month=max(0.01, float(hits_per_month)),
        on_hand=round_to_pack(max(0.0, float(pna)), max(1.0, float(standard_pack))),
    )
    item.daily_ur = safe_div(item.usage_rate, 30.0, 0.0)
    item.op = calculate_op(item.usage_rate, item.lead_time, item.safety_allowance)
    item.lp = calculate_lp(item.usage_rate, global_settings, item.op)
    item.eoq = calculate_eoq(item.usage_rate, item.item_cost, global_settings)
    item.oq = calculate_oq(item.eoq, item.usage_rate, global_settings)
    item.op_base_raw = item.op
    return update_planning_fields(item, global_settings)


def update_global_settings(
    current_settings: GlobalSettings,
    review_cycle: int,
    r_cost: float,
    k_cost: float,
    stockout_penalty: float,
    expedite_rate: float,
    gm: float,
) -> GlobalSettings:
    current_settings.r_cycle = int(review_cycle)
    current_settings.r_cost = float(r_cost)
    current_settings.k_cost = float(k_cost)
    current_settings.stockout_penalty = float(stockout_penalty)
    current_settings.expedite_rate = float(expedite_rate)
    current_settings.gm = float(gm)
    return current_settings


def update_gs_related_values(item: InventoryItem, global_settings: GlobalSettings) -> InventoryItem:
    item.lp = calculate_lp(item.usage_rate, global_settings, item.op)
    item.eoq = calculate_eoq(item.usage_rate, item.item_cost, global_settings)
    item.oq = calculate_oq(item.eoq, item.usage_rate, global_settings)
    return update_planning_fields(item, global_settings)
