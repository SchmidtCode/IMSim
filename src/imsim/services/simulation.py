from __future__ import annotations

import math
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from ..models import InventoryItem, Receipt, ServiceMetrics, SimulationState
from ..repository import SessionRepository
from .asq import apply_asq_month_end
from .planning import round_to_pack, safe_div, update_planning_fields
from .training import (
    apply_lesson_evaluation,
    demand_mode,
    evaluate_active_lesson,
    is_action_allowed,
)

_thread_local = threading.local()


def _rng():
    if not hasattr(_thread_local, "gen"):
        _thread_local.gen = np.random.default_rng()
    return _thread_local.gen


@dataclass(slots=True)
class DayMetrics:
    orders: int = 0
    orders_stockout: int = 0
    units_ordered: float = 0.0
    units_shipped: float = 0.0
    units_backordered: float = 0.0
    zero_on_hand_hits: int = 0
    holding_add: float = 0.0
    stockout_add: float = 0.0
    revenue_add: float = 0.0
    cogs_add: float = 0.0
    purchases_add: float = 0.0
    inv_value_mid_add: float = 0.0


def simulate_daily_hits(hpm: float) -> int:
    return int(_rng().poisson(max(0.0, float(hpm)) / 30.0))


def simulate_sales(avg_sale_qty: float, standard_pack: float) -> int:
    if avg_sale_qty <= 0:
        return 0
    pack = max(1.0, float(standard_pack))
    lam_packs = float(avg_sale_qty) / pack
    raw_packs = int(_rng().poisson(lam_packs))
    return int(raw_packs * pack)


def _daily_demands(item: InventoryItem, state: SimulationState) -> list[float]:
    mode = demand_mode(state)
    if mode == "deterministic":
        qty = max(0.0, safe_div(item.usage_rate, 30.0, 0.0))
        return [qty] if qty > 0 else []
    hpm = max(0.01, float(item.hits_per_month))
    avg_sale_qty = safe_div(item.usage_rate, hpm, 0.0)
    orders = simulate_daily_hits(hpm)
    demands: list[float] = []
    for _ in range(orders):
        qty = float(simulate_sales(avg_sale_qty, item.standard_pack))
        if qty > 0:
            demands.append(qty)
    return demands


def process_item_day(item: InventoryItem, today: int, state: SimulationState) -> DayMetrics:
    metrics = DayMetrics()
    gs = state.global_settings
    gm = min(max(gs.gm, 0.0), 0.95)
    realization = min(max(gs.realization, 0.5), 1.0)
    price_multiplier = 1.0 / max(1e-9, 1.0 - gm)

    due = [receipt for receipt in item.pipeline if receipt.eta_day <= today]
    if due:
        qty_in = sum(receipt.qty for receipt in due)
        item.on_hand += qty_in
        metrics.purchases_add = qty_in * item.item_cost
        item.pipeline = [receipt for receipt in item.pipeline if receipt.eta_day > today]

    if item.backorder > 0 and item.on_hand > 0:
        settle = min(item.backorder, item.on_hand)
        item.backorder -= settle
        item.on_hand -= settle

    on_hand_before = item.on_hand
    stockout_any = False

    for qty in _daily_demands(item, state):
        metrics.orders += 1
        metrics.units_ordered += qty
        item.asq_hits_period += 1
        item.asq_usage_period += float(qty)

        if item.on_hand <= 0:
            item.backorder += qty
            metrics.orders_stockout += 1
            metrics.zero_on_hand_hits += 1
            metrics.units_backordered += qty
            stockout_any = True
        elif qty <= item.on_hand:
            item.on_hand -= qty
            metrics.units_shipped += qty
        else:
            short = qty - item.on_hand
            metrics.units_shipped += item.on_hand
            metrics.units_backordered += short
            item.backorder += short
            item.on_hand = 0.0
            metrics.orders_stockout += 1
            stockout_any = True

    midpoint_oh = (on_hand_before + item.on_hand) / 2.0
    metrics.holding_add = midpoint_oh * item.item_cost * (gs.k_cost / 365.0)
    metrics.stockout_add = metrics.units_backordered * gs.stockout_penalty
    metrics.inv_value_mid_add = midpoint_oh * item.item_cost
    metrics.cogs_add = metrics.units_shipped * item.item_cost
    metrics.revenue_add = metrics.cogs_add * price_multiplier * realization
    item.stockout_today = stockout_any
    update_planning_fields(item, gs)
    return metrics


def add_metrics(bucket: ServiceMetrics, metrics: DayMetrics) -> None:
    bucket.orders += metrics.orders
    bucket.orders_stockout += metrics.orders_stockout
    bucket.units_ordered += metrics.units_ordered
    bucket.units_shipped += metrics.units_shipped
    bucket.units_backordered += metrics.units_backordered
    bucket.zero_on_hand_hits += metrics.zero_on_hand_hits


def tick_state(state: SimulationState) -> dict[str, int]:
    if not state.is_initialized or not state.items:
        return {"day": state.day, "asq_changed": 0, "lesson_completed": 0}
    state.day += 1
    state.service_today = ServiceMetrics()
    holding_today = 0.0
    stockout_today = 0.0
    revenue_today = 0.0
    cogs_today = 0.0
    purchases_today = 0.0
    inventory_mid_value = 0.0

    for item in state.items:
        metrics = process_item_day(item, state.day, state)
        add_metrics(state.service_today, metrics)
        add_metrics(state.service_totals, metrics)
        holding_today += metrics.holding_add
        stockout_today += metrics.stockout_add
        revenue_today += metrics.revenue_add
        cogs_today += metrics.cogs_add
        purchases_today += metrics.purchases_add
        inventory_mid_value += metrics.inv_value_mid_add

    if state.global_settings.auto_po_enabled and is_action_allowed(state, "auto_po"):
        place_purchase_orders(state)
    else:
        state.global_settings.auto_po_enabled = False

    state.costs.holding += holding_today
    state.costs.stockout += stockout_today
    state.costs.purchases += purchases_today
    state.costs.total = (
        state.costs.ordering + state.costs.holding + state.costs.stockout + state.costs.expedite
    )
    state.sales.revenue += revenue_today
    state.sales.cogs += cogs_today
    state.sales.units_sold += state.service_today.units_shipped
    state.analytics.inv_value_daysum += inventory_mid_value
    state.record_history()

    asq_changed = 0
    if state.day % max(1, state.global_settings.asq.period_days) == 0:
        asq_changed = apply_asq_month_end(state)["changed"]
    lesson_completed = 0
    evaluation = evaluate_active_lesson(state)
    if evaluation is not None and evaluation.completed:
        apply_lesson_evaluation(state, evaluation)
        lesson_completed = 1
    return {"day": state.day, "asq_changed": asq_changed, "lesson_completed": lesson_completed}


def place_purchase_orders(state: SimulationState) -> dict[str, float]:
    lines = 0
    total_qty = 0.0
    receipts: list[dict[str, float | str | int]] = []
    for index, item in enumerate(state.items):
        qty = float(item.soq)
        if qty <= 0:
            continue
        receipt_id = str(uuid.uuid4())[:8]
        eta = int(state.day + math.ceil(max(1.0, item.lead_time)))
        item.pipeline.append(Receipt(receipt_id=receipt_id, qty=qty, eta_day=eta))
        state.costs.ordering += state.global_settings.r_cost
        update_planning_fields(item, state.global_settings)
        lines += 1
        total_qty += qty
        receipts.append({"item_index": index, "rid": receipt_id, "qty": qty, "eta_day": eta})
    state.costs.total = (
        state.costs.ordering + state.costs.holding + state.costs.stockout + state.costs.expedite
    )
    return {"lines": lines, "total_qty": total_qty, "receipts": receipts}


def _custom_order_qty(raw_qty: float | str | None) -> float:
    if raw_qty in (None, ""):
        return 0.0
    try:
        return max(0.0, float(raw_qty))
    except (TypeError, ValueError):
        return 0.0


def place_custom_orders(state: SimulationState, quantities: list[float | str | None]) -> bool:
    changed = False
    for index, raw_qty in enumerate(quantities):
        if index >= len(state.items):
            continue
        qty = _custom_order_qty(raw_qty)
        if qty <= 0:
            continue
        item = state.items[index]
        qty = round_to_pack(qty, item.standard_pack)
        if qty <= 0:
            continue
        item.pipeline.append(
            Receipt(
                receipt_id=str(uuid.uuid4())[:8],
                qty=qty,
                eta_day=int(state.day + math.ceil(max(1.0, item.lead_time))),
            )
        )
        state.costs.ordering += state.global_settings.r_cost
        update_planning_fields(item, state.global_settings)
        changed = True
    if changed:
        state.costs.total = (
            state.costs.ordering + state.costs.holding + state.costs.stockout + state.costs.expedite
        )
    return changed


def _queue_jump_probability(lead_time_days: float) -> float:
    lead_time_days = max(1.0, float(lead_time_days))
    probability = 0.58 * ((30.0 / (lead_time_days + 30.0)) ** 0.55)
    return min(0.6, max(0.08, probability))


def _expedite_outcome(
    state: SimulationState, item: InventoryItem, receipt: Receipt
) -> tuple[int, float]:
    days_until_receipt = max(0, receipt.eta_day - state.day)
    max_reducible_days = max(0, days_until_receipt - 1)
    if max_reducible_days <= 0:
        return 0, 0.0
    rng = _rng()
    jump_target_days = 7
    queue_jump = rng.random() < _queue_jump_probability(item.lead_time)
    if days_until_receipt > jump_target_days and queue_jump:
        return days_until_receipt - jump_target_days, 2.0
    # Otherwise, fall back to a smaller pull-forward so expedites still help
    # when the supplier cannot short-ship from available stock.
    random_cap = max(1, math.ceil(days_until_receipt * 0.25))
    days_reduced = int(rng.integers(1, min(max_reducible_days, random_cap, 5) + 1))
    return days_reduced, 1.0


def expedite_or_cancel_receipts(
    state: SimulationState, receipt_ids: list[str] | tuple[str, ...], action: str
) -> int:
    target_ids = {str(receipt_id) for receipt_id in receipt_ids if receipt_id}
    if not target_ids:
        return 0
    changed = 0
    for item in state.items:
        if action == "cancel":
            remaining_pipeline: list[Receipt] = []
            removed = 0
            for receipt in item.pipeline:
                if receipt.receipt_id in target_ids:
                    removed += 1
                    continue
                remaining_pipeline.append(receipt)
            if removed:
                item.pipeline = remaining_pipeline
                update_planning_fields(item, state.global_settings)
                changed += removed
            continue
        if action != "expedite":
            continue
        for receipt in item.pipeline:
            if receipt.receipt_id not in target_ids:
                continue
            days_reduced, surcharge_multiplier = _expedite_outcome(state, item, receipt)
            if days_reduced <= 0:
                continue
            receipt.eta_day = max(receipt.eta_day - days_reduced, state.day + 1)
            state.costs.expedite += (
                surcharge_multiplier
                * receipt.qty
                * item.item_cost
                * state.global_settings.expedite_rate
            )
            changed += 1
    if changed:
        state.costs.total = (
            state.costs.ordering + state.costs.holding + state.costs.stockout + state.costs.expedite
        )
    return changed


def expedite_or_cancel_receipt(state: SimulationState, receipt_id: str, action: str) -> bool:
    return expedite_or_cancel_receipts(state, [receipt_id], action) > 0


@dataclass(slots=True)
class MaintenanceState:
    active: bool = False
    at: float = 0.0
    message: str = ""
    closing: bool = False


class MaintenanceController:
    def __init__(self, repository: SessionRepository, allow_dev_shutdown: bool):
        self._repository = repository
        self._allow_dev_shutdown = allow_dev_shutdown
        self._state = MaintenanceState()
        self._lock = threading.RLock()
        self._watchdog_started = False

    def schedule_shutdown_in(self, minutes: float, message: str = "Maintenance") -> None:
        with self._lock:
            self._state.active = True
            self._state.at = time.time() + max(0.0, float(minutes)) * 60.0
            self._state.message = message
            self._state.closing = False

    def cancel_shutdown(self) -> None:
        with self._lock:
            self._state = MaintenanceState()

    def snapshot(self) -> MaintenanceState:
        with self._lock:
            return MaintenanceState(
                active=self._state.active,
                at=self._state.at,
                message=self._state.message,
                closing=self._state.closing,
            )

    def gentle_stop_all_sessions(self) -> None:
        self._repository.pause_all()
        self._repository.persist_all()

    def heartbeat(self) -> MaintenanceState:
        state = self.snapshot()
        if not state.active:
            return state
        remaining = state.at - time.time()
        if remaining <= 0.5 and not state.closing:
            self.gentle_stop_all_sessions()
            with self._lock:
                self._state.closing = True
                state = self.snapshot()
            if self._allow_dev_shutdown:
                os.kill(os.getpid(), signal.SIGTERM)
        return state

    def start_watchdog_once(self) -> None:
        if self._watchdog_started:
            return
        thread = threading.Thread(target=self._watchdog, daemon=True)
        thread.start()
        self._watchdog_started = True

    def _watchdog(self) -> None:
        while True:
            time.sleep(1.0)
            self.heartbeat()
