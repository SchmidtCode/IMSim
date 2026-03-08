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

    hpm = max(0.01, float(item.hits_per_month))
    avg_sale_qty = safe_div(item.usage_rate, hpm, 0.0)
    on_hand_before = item.on_hand
    orders = simulate_daily_hits(hpm)
    stockout_any = False

    for _ in range(orders):
        qty = simulate_sales(avg_sale_qty, item.standard_pack)
        if qty <= 0:
            continue
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
        return {"day": state.day, "asq_changed": 0}
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

    if state.global_settings.auto_po_enabled:
        place_purchase_orders(state)

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

    asq_changed = 0
    if state.day % max(1, state.global_settings.asq.period_days) == 0:
        asq_changed = apply_asq_month_end(state)["changed"]
    return {"day": state.day, "asq_changed": asq_changed}


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


def place_custom_orders(state: SimulationState, quantities: list[float | None]) -> bool:
    changed = False
    for index, raw_qty in enumerate(quantities):
        if raw_qty is None or index >= len(state.items):
            continue
        qty = max(0.0, float(raw_qty))
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


def expedite_or_cancel_receipt(state: SimulationState, receipt_id: str, action: str) -> bool:
    changed = False
    for item in state.items:
        for receipt in list(item.pipeline):
            if receipt.receipt_id != receipt_id:
                continue
            if action == "expedite" and receipt.eta_day > state.day + 1:
                receipt.eta_day = max(receipt.eta_day - 1, state.day + 1)
                state.costs.expedite += (
                    receipt.qty * item.item_cost * state.global_settings.expedite_rate
                )
                changed = True
            elif action == "cancel":
                item.pipeline = [rec for rec in item.pipeline if rec.receipt_id != receipt_id]
                update_planning_fields(item, state.global_settings)
                changed = True
            break
    if changed:
        state.costs.total = (
            state.costs.ordering + state.costs.holding + state.costs.stockout + state.costs.expedite
        )
    return changed


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
