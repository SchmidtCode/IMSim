from __future__ import annotations

from ..models import ExceptionRecord, InventoryItem, SimulationState
from .planning import (
    calculate_eoq,
    calculate_lp,
    calculate_op,
    calculate_oq,
    format_money,
    safe_div,
    update_planning_fields,
)


def log_exception(
    state: SimulationState, code: str, message: str, item_index: int, item: InventoryItem
) -> None:
    state.exception_center.append(
        ExceptionRecord(
            day=int(state.day),
            item_index=int(item_index),
            code=code,
            message=message,
            op=float(item.op),
            op_base_raw=float(item.op_base_raw),
            asq=float(item.asq_last_value),
            item_cost=float(item.item_cost),
        )
    )


def apply_asq_to_item(item: InventoryItem, state: SimulationState, item_index: int) -> bool:
    cfg = state.global_settings.asq
    if not cfg.enabled:
        return False
    hits = int(item.asq_hits_period)
    usage = float(item.asq_usage_period)
    item.op_base_raw = calculate_op(item.usage_rate, item.lead_time, item.safety_allowance)
    if hits < cfg.min_hits or hits <= 0:
        item.asq_last_value = 0.0
        return False
    asq = safe_div(usage, hits, 0.0)
    item.asq_last_value = asq
    if asq <= item.op_base_raw:
        return False
    value_increase = (asq - item.op_base_raw) * item.item_cost
    if cfg.max_amount_diff > 0 and value_increase > cfg.max_amount_diff:
        log_exception(
            state,
            "ASQ_MAX_DIFF_EXCEEDED",
            (
                "Skipped ASQ raise: "
                f"+${value_increase:,.2f} exceeds Max $ Diff "
                f"{format_money(cfg.max_amount_diff)}"
            ),
            item_index,
            item,
        )
        return False
    item.op = float(round(asq, 3))
    item.lp = calculate_lp(item.usage_rate, state.global_settings, item.op)
    item.eoq = calculate_eoq(item.usage_rate, item.item_cost, state.global_settings)
    item.oq = calculate_oq(item.eoq, item.usage_rate, state.global_settings)
    update_planning_fields(item, state.global_settings)
    item.asq_last_applied_day = int(state.day)
    return True


def apply_asq_month_end(state: SimulationState) -> dict[str, int]:
    changed = 0
    for index, item in enumerate(state.items):
        if apply_asq_to_item(item, state, index):
            changed += 1
        item.asq_hits_period = 0
        item.asq_usage_period = 0.0
    return {"changed": changed}
