"""
app.py — Inventory Management Simulator (Dash) + ASQ OP Adjuster

What’s inside:
- Inventory planning (OP/LP/EOQ/OQ) + physical state (On-Hand / Pipeline / Backorder)
- Demand simulation (hits → orders → shipped/backordered)
- Costing (ordering, holding, stockout, expedite)
- Sales engine (global GM% → revenue; track COGS, units sold)
- Visualization:
    • Current PNA (days from OP)
    • Proposed PNA + SOQ (days from OP)
    • 0 PNA (days from OP)
    • On-Hand (days from OP) — orange "x" dot
- File upload/import for items
- PO placement, custom orders, PO overview (expedite/cancel)
- KPI strip refresh
- Maintenance scheduling API

NEW — ASQ OP Adjuster:
- Tracks line hits & usage during a rolling “month” (configurable period days).
- Computes ASQ = usage / line hits.
- At month-end (or on-demand), if ASQ > raw OP, raises OP to ASQ.
- Applies only when:
    • ASQ Adjuster is enabled.
    • Min line hits met.
    • Max $ Diff (inventory value increase from OP raise) not exceeded.
- Optionally counts transfers (currently the sim does not generate transfers; toggle is provided).

Notes:
- GM input is a global percentage; price per unit = item_cost / (1 - GM).
- “0 PNA” != “0 On-Hand”. PNA = on_hand + pipeline - backorder (net availability).
- On-Hand dot helps visualize imminent stockouts even with incoming POs.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports & Setup
# ──────────────────────────────────────────────────────────────────────────────
import base64
import datetime
import io
import json
import math
import os
import uuid
import time
import requests
import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import RLock, local
from typing import Dict, List, Sequence, TypeAlias, Union, cast
from flask import Flask, request, abort

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import ALL, Input, Output, State, callback, dcc, html, dash_table, ctx
from dash.exceptions import PreventUpdate
from dash_bootstrap_components import Modal, ModalBody, ModalFooter, ModalHeader
from dash_bootstrap_templates import load_figure_template


# ──────────────────────────────────────────────────────────────────────────────
# Infra / Globals
# ──────────────────────────────────────────────────────────────────────────────

# Data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# GitHub repo link (for footer)
GH_URL = os.environ.get("IMSIM_GITHUB_URL", "https://github.com/SchmidtCode/IMSim")
# Footer styles (bottom-right; visible/hidden variants)
GH_FOOTER_STYLE_VISIBLE = {
    "position": "fixed",
    "bottom": "12px",
    "right": "12px",
    "maxWidth": "360px",
    "zIndex": 1050,
    "opacity": 0.9,
}
GH_FOOTER_STYLE_HIDDEN = {**GH_FOOTER_STYLE_VISIBLE, "display": "none"}

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=4)
_store_lock = RLock()
_thread_local = local()

# Plotly Bootstrap template
load_figure_template("darkly")

server = Flask(__name__)
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    server=server,
)
app.title = "IM Sim"

# In-memory server-side data store (persisted to disk periodically)
user_data_store: dict = {}

# Handy aliases
Number: TypeAlias = Union[int, float]
Cell: TypeAlias = Union[bool, int, float, str]
Row: TypeAlias = Dict[str, Cell]

# Graceful shutdown state (polled by all clients)
SHUTDOWN_URL = os.environ.get("SHUTDOWN_URL", "http://127.0.0.1:8050/shutdown")
_shutdown_state = {"active": False, "at": 0.0, "message": "", "closing": False}


def _rng():
    """Thread-local RNG to keep simulation deterministic per-thread."""
    if not hasattr(_thread_local, "gen"):
        _thread_local.gen = np.random.default_rng()
    return _thread_local.gen


def format_money(x: float) -> str:
    """Render currency consistently."""
    return f"${x:,.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# Default structures / factories
# ──────────────────────────────────────────────────────────────────────────────

def new_service_bucket() -> dict:
    """Per-day and cumulative service-level counters."""
    return {
        "orders": 0,
        "orders_stockout": 0,
        "units_ordered": 0.0,
        "units_shipped": 0.0,
        "units_backordered": 0.0,
        "zero_on_hand_hits": 0,
    }


def new_costs_bucket() -> dict:
    """Cost buckets; total is derived as a convenience."""
    return {
        "ordering": 0.0,
        "holding": 0.0,
        "stockout": 0.0,
        "expedite": 0.0,
        "purchases": 0.0,   # cost of received inventory (qty * item_cost)
        "total": 0.0,       # inventory overhead only (ordering+holding+stockout+expedite)
    }


def new_sales_bucket() -> dict:
    """Sales tracking buckets."""
    return {
        "revenue": 0.0,    # ∑ price_per_unit * units_shipped
        "cogs": 0.0,       # ∑ item_cost * units_shipped
        "units_sold": 0.0, # convenience metric
    }


def new_analytics_bucket() -> dict:
    # day-summed inventory value at cost using daily midpoint OH
    return {"inv_value_daysum": 0.0}



def asq_defaults() -> dict:
    """Default ASQ adjuster configuration."""
    return {
        "enabled": True,            # toggle
        "min_hits": 3,              # minimum line hits required to consider ASQ
        "include_transfers": False, # whether transfers count (sim currently has none)
        "max_amount_diff": 2500.0,  # $ cap on increase due to ASQ raise
        "period_days": 30,          # "month end" cadence in days
    }


def get_default_data() -> dict:
    """Initial state for a brand-new session."""
    return {
        "global_settings": {
            "r_cycle": 14,         # Review cycle (days)
            "r_cost": 8.0,         # Ordering cost per item ordered
            "k_cost": 0.18,        # Annual holding rate (decimal)
            "stockout_penalty": 5.0,  # $ per unit short
            "expedite_rate": 0.03,    # fraction of item_cost per unit per day expedited
            "gm": 0.15,               # default GM (15%)
            "realization": 1.0,      # realized price fraction (e.g., promos, leakage)
            "asq": asq_defaults(),    # NEW: ASQ adjuster config
            "auto_po_enabled": False,   # when True, system places orders each tick
        },
        "items": [],
        "day": 1,
        "is_initialized": False,
        "service_today": new_service_bucket(),
        "service_totals": new_service_bucket(),
        "costs": new_costs_bucket(),
        "sales": new_sales_bucket(),
        "exception_center": [],
        "analytics": new_analytics_bucket(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    with _store_lock, open(os.path.join(DATA_DIR, "user_data.json"), "w") as f:
        json.dump(user_data_store, f)


def set_user_data(user_id: str, data: dict):
    """Replace a user's blob in the in-memory store."""
    with _store_lock:
        user_data_store[user_id] = data


def load_data() -> dict:
    path = os.path.join(DATA_DIR, "user_data.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


user_data_store = load_data()


def get_user_data(user_id: str) -> dict:
    with _store_lock:
        if user_data_store.get(user_id) is None:
            user_data_store[user_id] = get_default_data()
        gs = user_data_store[user_id].setdefault("global_settings", {})
        gs.setdefault("asq", asq_defaults())
        gs.setdefault("auto_po_enabled", False)
        user_data_store[user_id].setdefault("analytics", new_analytics_bucket())
        return copy.deepcopy(user_data_store[user_id])
    
# ──────────────────────────────────────────────────────────────────────────────
# Shutdown helpers
# ──────────────────────────────────────────────────────────────────────────────

def schedule_shutdown_in(minutes: float, message: str = "Maintenance"):
    """Schedule a shutdown at now + minutes."""
    with _store_lock:
        _shutdown_state["active"] = True
        _shutdown_state["at"] = time.time() + max(0.0, float(minutes)) * 60.0
        _shutdown_state["message"] = message or "Maintenance"
        _shutdown_state["closing"] = False

def cancel_shutdown():
    """Cancel a scheduled shutdown."""
    with _store_lock:
        _shutdown_state.update({"active": False, "at": 0.0, "message": "", "closing": False})

def gentle_stop_all_sessions():
    """Pause all sims and persist before exit."""
    with _store_lock:
        for uid, blob in user_data_store.items():
            try:
                blob["is_initialized"] = False  # pauses per-user sim
            except Exception:
                pass
        save_data()
    time.sleep(1.0)  # let any in-flight requests finish
    print("[shutdown] All sessions paused and data saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Admin API (no UI) — schedule/cancel via curl
# ──────────────────────────────────────────────────────────────────────────────

from flask import abort, jsonify

ADMIN_TOKEN = os.environ.get("IMSIM_ADMIN_TOKEN")  # optional; if set, required

def _authorized(req) -> bool:
    if not ADMIN_TOKEN:
        return True  # no token required
    # accept either header
    bearer = (req.headers.get("Authorization") or "")
    if bearer.startswith("Bearer "):
        bearer = bearer.split(" ", 1)[1]
    header_token = req.headers.get("X-IMSIM-ADMIN-TOKEN") or bearer
    return header_token == ADMIN_TOKEN

@server.post("/api/admin/schedule_shutdown")
def api_schedule_shutdown():
    if not _authorized(request):
        return abort(401)
    data = request.get_json(silent=True) or {}
    minutes = float(data.get("minutes", 0))
    message = str(data.get("message", "Maintenance"))
    schedule_shutdown_in(minutes, message)
    return jsonify({"ok": True, "active": True, "minutes": minutes, "message": message})

@server.post("/api/admin/cancel_shutdown")
def api_cancel_shutdown():
    if not _authorized(request):
        return abort(401)
    cancel_shutdown()
    return jsonify({"ok": True, "active": False})

@server.get("/api/admin/shutdown_status")
def api_shutdown_status():
    with _store_lock:
        st = dict(_shutdown_state)
    now = time.time()
    remaining = max(0.0, st["at"] - now) if st.get("active") else None
    return jsonify({"active": st.get("active", False),
                    "message": st.get("message", ""),
                    "seconds_remaining": None if remaining is None else int(remaining),
                    "closing": st.get("closing", False)})


# ──────────────────────────────────────────────────────────────────────────────
# Planning math
# ──────────────────────────────────────────────────────────────────────────────

def safe_div(n, d, default=0.0):
    """Division with default on zero/None/exception."""
    try:
        return n / d if d not in (0, 0.0, None) else default
    except Exception:
        return default


def calculate_op(usage_rate: float, lead_time_days: float, safety_allowance: float) -> float:
    """Order point: expected demand over lead-time + safety stock."""
    monthly_lt = lead_time_days / 30.0
    safety_stock = usage_rate * monthly_lt * safety_allowance
    return usage_rate * monthly_lt + safety_stock


def calculate_lp(usage_rate: float, global_settings: dict, op: float) -> float:
    """Line point: demand over review cycle added to OP."""
    review_cycle_days = global_settings["r_cycle"]
    return usage_rate * (review_cycle_days / 30.0) + op


def calculate_eoq(usage_rate: float, item_cost: float, global_settings: dict) -> float:
    """EOQ via standard square-root formula (annualized k-cost)."""
    r_cost = global_settings["r_cost"]
    k_cost = global_settings["k_cost"]
    if item_cost <= 0 or k_cost <= 0 or usage_rate <= 0:
        return 0.0
    return math.sqrt((24.0 * r_cost * usage_rate) / (k_cost * item_cost))


def calculate_oq(eoq: float, usage_rate: float, global_settings: dict) -> float:
    """Smoothed OQ: bounded between ~RC demand and ~12 months of demand."""
    rc_days = global_settings["r_cycle"]
    oq_min = max(0.5 * usage_rate, (rc_days / 30.0) * usage_rate)
    oq_max = 12.0 * usage_rate
    if eoq <= 0:
        return oq_min
    return max(min(eoq, oq_max), oq_min)


def round_up_to_pack(qty: float, pack: float) -> float:
    """Round UP to the next pack (for order suggestions)."""
    if pack is None or pack <= 0:
        return max(0.0, float(qty))
    if qty <= 0:
        return 0.0
    return math.ceil(float(qty) / float(pack)) * float(pack)

def round_down_to_pack(qty: float, pack: float) -> float:
    """Round DOWN to the previous pack (rarely used)."""
    if pack is None or pack <= 0:
        return max(0.0, float(qty))
    if qty <= 0:
        return 0.0
    return math.floor(float(qty) / float(pack)) * float(pack)

def round_to_pack(qty: float, pack: float) -> float:
    """Round to the NEAREST pack (keep for displays if you like)."""
    if pack is None or pack <= 0:
        return max(0.0, float(qty))
    # “Nearest” — but avoid banker’s rounding by adding a tiny epsilon
    q = float(qty) / float(pack)
    return max(0.0, float(pack) * math.floor(q + 0.5 + 1e-12))


def calculate_soq(pna: float, lp: float, op: float, oq: float, standard_pack: float) -> float:
    """
    SOQ policy:
    - If PNA > LP → 0
    - If OP < PNA ≤ LP → OQ
    - If PNA ≤ OP → OQ + (OP - PNA)
    Round UP to packs for actionable order quantities.
    """
    if pna > lp:
        return 0.0
    base = oq if pna > op else (oq + max(0.0, op - pna))
    return round_up_to_pack(base, standard_pack)


def calculate_surplus_line(lp: float, eoq: float) -> float:
    """Surplus threshold line for visualization."""
    return lp + max(0.0, eoq)


def calculate_critical_point(usage_rate: float, lead_time_days: float) -> float:
    """Critical units ~= demand over lead-time (no safety)."""
    return safe_div(usage_rate, 30.0) * max(0.0, lead_time_days)


# ──────────────────────────────────────────────────────────────────────────────
# Inventory helpers
# ──────────────────────────────────────────────────────────────────────────────

def item_on_order(item: dict) -> float:
    """Sum of pipeline quantities."""
    return float(sum(float(r["qty"]) for r in item.get("pipeline", [])))


def compute_item_pna(item: dict) -> float:
    """Projected net available = On-hand + On-order - Backorder."""
    return (
        max(0.0, float(item.get("on_hand", 0.0)))
        + item_on_order(item)
        - max(0.0, float(item.get("backorder", 0.0)))
    )


def update_planning_fields(item: dict, global_settings: dict) -> dict:
    """
    Update all derived fields after any physical/planning change.

    OP-normalized (Days from OP):
      - pna_days_frm_op      : (PNA - OP) converted to days
      - ats_days_frm_op      : (On-Hand - OP) converted to days (ATS = Available to Sell)
      - pro_pna_days_frm_op  : (PNA + SOQ - OP) in days
      - no_pna_days_frm_op   : (0 - OP) in days  (the '0 PNA' guide)

    We keep legacy fields other code may rely on.
    """
    item["pna"] = compute_item_pna(item)

    ur = max(0.0, float(item.get("usage_rate", 0.0)))
    daily_ur = float(item.get("daily_ur", safe_div(ur, 30.0, 0.0)))
    if daily_ur <= 0:
        daily_ur = 1e-9

    on_hand = max(0.0, float(item.get("on_hand", 0.0)))

    # --- OP-normalized days ---
    item["pna_days_frm_op"] = safe_div((item["pna"] - item["op"]), max(1e-9, ur), 0.0) * 30.0
    item["oh_days_frm_op"]  = safe_div((on_hand   - item["op"]), max(1e-9, ur), 0.0) * 30.0
    item["ats_days_frm_op"] = item["oh_days_frm_op"]  # alias for graph/legend rename

    # Ordering math
    item["soq"] = calculate_soq(item["pna"], item["lp"], item["op"], item["oq"], item["standard_pack"])
    item["surplus_line"] = calculate_surplus_line(item["lp"], item["eoq"])
    item["proposed_pna"] = item["pna"] + item["soq"]

    # Proposed/zero anchors in OP-normalized days
    item["pro_pna_days_frm_op"] = safe_div((item["proposed_pna"] - item["op"]), max(1e-9, ur), 0.0) * 30.0
    item["no_pna_days_frm_op"]  = safe_div((0.0 - item["op"]),           max(1e-9, ur), 0.0) * 30.0

    # (Optional: keep absolute-day variants around if you reference them elsewhere)
    item["pna_days"] = safe_div(item["pna"], daily_ur, 0.0)

    return item


def ensure_item_physical_defaults(it: dict, gs: dict) -> dict:
    """
    Ensure physical state exists and compute planning inputs if missing.
    Then update derived fields for consistency.
    """
    # Physical defaults
    if "on_hand" not in it:
        sp = float(it.get("standard_pack", 1.0) or 1.0)
        on_hand_seed = float(it.get("pna", 0.0))
        it["on_hand"] = round_to_pack(on_hand_seed, sp)
    it.setdefault("pipeline", [])
    it.setdefault("backorder", 0.0)
    it.setdefault("stockout_today", False)
    it.setdefault("daily_ur", safe_div(float(it.get("usage_rate", 0.0)), 30.0, 0.0))

    # Ensure planning inputs exist
    if not all(k in it for k in ("op", "lp", "eoq", "oq")):
        ur = float(it.get("usage_rate", 0.0))
        lt = float(it.get("lead_time", it.get("lead_time_days", 0.0)))
        # accept either % or decimal on legacy
        sa = it.get("safety_allowance", it.get("safety_allowance_pct", 0.0))
        sa = float(sa) / 100.0 if "safety_allowance_pct" in it else float(sa)
        ic = float(it.get("item_cost", 0.0))
        it["op"] = calculate_op(ur, lt, sa)
        it["lp"] = calculate_lp(ur, gs, it["op"])
        it["eoq"] = calculate_eoq(ur, ic, gs)
        it["oq"] = calculate_oq(it["eoq"], ur, gs)

    return update_planning_fields(it, gs)


def create_inventory_item(
    usage_rate: float,
    lead_time: float,
    item_cost: float,
    pna: float,
    safety_allowance: float,
    standard_pack: float,
    global_settings: dict,
    hits_per_month: float,
) -> dict:
    """
    Build a new item with planning + physical state, then compute deriveds.
    """
    usage_rate = max(0.0, float(usage_rate))
    lead_time = max(0.0, float(lead_time))
    item_cost = max(0.0, float(item_cost))
    pna = max(0.0, float(pna))
    safety_allowance = max(0.0, float(safety_allowance))
    standard_pack = max(1.0, float(standard_pack))
    hits_per_month = max(0.01, float(hits_per_month))

    daily_ur = safe_div(usage_rate, 30.0, 0.0)

    op = calculate_op(usage_rate, lead_time, safety_allowance)
    lp = calculate_lp(usage_rate, global_settings, op)
    eoq = calculate_eoq(usage_rate, item_cost, global_settings)
    oq = calculate_oq(eoq, usage_rate, global_settings)
    surplus_line = calculate_surplus_line(lp, eoq)
    cp = calculate_critical_point(usage_rate, lead_time)

    # Physical state
    on_hand = round_to_pack(pna, standard_pack)
    pipeline: list[dict] = []
    backorder = 0.0

    item = {
        # static inputs
        "usage_rate": usage_rate,
        "lead_time": lead_time,
        "item_cost": item_cost,
        "safety_allowance": safety_allowance,
        "standard_pack": standard_pack,
        "hits_per_month": hits_per_month,
        "daily_ur": daily_ur,
        # planning
        "op": op,
        "lp": lp,
        "eoq": eoq,
        "oq": oq,
        "surplus_line": surplus_line,
        "cp": cp,
        # physical state
        "on_hand": on_hand,
        "pipeline": pipeline,
        "backorder": backorder,
        "stockout_today": False,
        # ASQ tracking (period counters)
        "asq_hits_period": 0,
        "asq_usage_period": 0.0,
        "asq_last_value": 0.0,
        "asq_last_applied_day": 0,
        "op_base_raw": op,
        # derived placeholders (computed next)
        "pna": 0.0,
        "pna_days": 0.0,
        "pna_days_frm_op": 0.0,
        "soq": 0.0,
        "proposed_pna": 0.0,
        "pro_pna_days_frm_op": 0.0,
        "no_pna_days_frm_op": 0.0,
    }
    return update_planning_fields(item, global_settings)


def update_global_settings(
    current_settings: dict,
    review_cycle: Number,
    r_cost: Number,
    k_cost: Number,
    stockout_penalty: Number,
    expedite_rate: Number,
    gm: Number,
) -> dict:
    """Write-through of global settings with one function for clarity."""
    current_settings["r_cycle"] = review_cycle
    current_settings["r_cost"] = r_cost
    current_settings["k_cost"] = k_cost
    current_settings["stockout_penalty"] = stockout_penalty
    current_settings["expedite_rate"] = expedite_rate
    current_settings["gm"] = gm
    # keep ASQ block as-is (separately updated)
    return current_settings


def update_gs_related_values(item: dict, global_settings: dict) -> dict:
    """When globals change, recompute item planning+deriveds (OP left intact)."""
    item["lp"] = calculate_lp(item["usage_rate"], global_settings, item["op"])
    item["eoq"] = calculate_eoq(item["usage_rate"], item["item_cost"], global_settings)
    item["oq"] = calculate_oq(item["eoq"], item["usage_rate"], global_settings)
    return update_planning_fields(item, global_settings)


# ──────────────────────────────────────────────────────────────────────────────
# ASQ Adjuster — core logic
# ──────────────────────────────────────────────────────────────────────────────

def _log_exception(current: dict, code: str, msg: str, item_index: int, it: dict, today: int):
    current.setdefault("exception_center", []).append(
        {
            "day": int(today),
            "item_index": int(item_index),
            "code": code,
            "message": msg,
            "op": float(it.get("op", 0.0)),
            "op_base_raw": float(it.get("op_base_raw", 0.0)),
            "asq": float(it.get("asq_last_value", 0.0)),
            "item_cost": float(it.get("item_cost", 0.0)),
        }
    )


def _apply_asq_to_item(it: dict, gs: dict, today: int, item_index: int, current: dict) -> bool:
    """
    Apply ASQ OP adjustment to a single item if eligible.
    Returns True if OP changed, else False.
    """
    asq_cfg = dict(gs.get("asq", asq_defaults()))
    if not asq_cfg.get("enabled", True):
        return False

    hits = int(it.get("asq_hits_period", 0))
    usage = float(it.get("asq_usage_period", 0.0))
    min_hits = max(0, int(asq_cfg.get("min_hits", 0)))

    # Compute base/raw OP (month-end baseline)
    base_op = calculate_op(float(it["usage_rate"]), float(it["lead_time"]), float(it["safety_allowance"]))
    it["op_base_raw"] = base_op

    # Not enough signal
    if hits < min_hits or hits <= 0:
        it["asq_last_value"] = 0.0
        return False

    # Average Sale Quantity (units per hit)
    asq = safe_div(usage, hits, 0.0)
    it["asq_last_value"] = asq

    # Only adjust upward if ASQ > raw OP
    if asq <= base_op:
        return False

    # Respect Max $ Diff guardrail
    max_diff = float(asq_cfg.get("max_amount_diff", 0.0))
    value_increase = (asq - base_op) * float(it["item_cost"])
    if max_diff > 0 and value_increase > max_diff:
        _log_exception(
            current,
            "ASQ_MAX_DIFF_EXCEEDED",
            f"Skipped ASQ raise: +${value_increase:,.2f} exceeds Max $ Diff {format_money(max_diff)}",
            item_index,
            it,
            today,
        )
        return False

    # Apply the raise: set OP to ASQ (rounded to 3 decimals for stability)
    it["op"] = float(round(asq, 3))
    # Recompute dependent planning
    it["lp"] = calculate_lp(it["usage_rate"], gs, it["op"])
    it["eoq"] = calculate_eoq(it["usage_rate"], it["item_cost"], gs)
    it["oq"] = calculate_oq(it["eoq"], it["usage_rate"], gs)
    update_planning_fields(it, gs)

    it["asq_last_applied_day"] = int(today)
    return True


def apply_asq_month_end(current: dict) -> dict:
    """
    Apply ASQ OP Adjuster to all items using period counters.
    Resets ASQ counters afterwards. Returns a small summary dict.
    """
    gs = current.get("global_settings", {})
    asq_cfg = dict(gs.get("asq", asq_defaults()))
    today = int(current.get("day", 1))
    changed = 0
    skipped = 0

    for idx, it in enumerate(current.get("items", [])):
        try:
            did = _apply_asq_to_item(it, gs, today, idx, current)
            changed += 1 if did else 0
        except Exception as e:
            skipped += 1
            _log_exception(
                current,
                "ASQ_ERROR",
                f"Exception during ASQ: {e}",
                idx,
                it,
                today,
            )
        finally:
            # Reset ASQ period counters regardless
            it["asq_hits_period"] = 0
            it["asq_usage_period"] = 0.0

    return {"changed": changed, "skipped": skipped, "today": today, "period_days": int(asq_cfg.get("period_days", 30))}


# ──────────────────────────────────────────────────────────────────────────────
# Upload helpers
# ──────────────────────────────────────────────────────────────────────────────

CANONICAL_COLS = [
    "usage_rate",
    "lead_time_days",
    "item_cost",
    "pna",
    "safety_allowance_pct",
    "standard_pack",
    "hits_per_month",
]

HEADER_ALIASES = {
    "usage rate (per month)": "usage_rate",
    "usage rate": "usage_rate",
    "usage_rate": "usage_rate",
    "lead time (days)": "lead_time_days",
    "lead time": "lead_time_days",
    "lead_time_days": "lead_time_days",
    "item cost": "item_cost",
    "item_cost": "item_cost",
    "initial pna": "pna",
    "pna": "pna",
    "safety allowance (%)": "safety_allowance_pct",
    "safety_allowance_pct": "safety_allowance_pct",
    "standard pack": "standard_pack",
    "standard_pack": "standard_pack",
    "hits per month": "hits_per_month",
    "hits_per_month": "hits_per_month",
}


def _extract_base64(contents: str) -> bytes:
    """Decode a base64 upload payload."""
    if not isinstance(contents, str):
        raise ValueError("Invalid upload payload type.")
    _prefix, sep, b64data = contents.partition(",")
    b64data = b64data if sep else contents
    try:
        return base64.b64decode(b64data, validate=False)
    except Exception as e:
        raise ValueError(f"Could not decode uploaded file: {e}")


def coerce_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate/normalize uploaded item dataframe to canonical numeric columns.
    Raises with a helpful error message if mismatched headers or non-numeric data.
    """
    raw_cols = [str(c) for c in df.columns]
    lower_cols = [c.strip().lower() for c in raw_cols]

    mapped: list[str] = []
    unknown: list[str] = []
    sources_by_canon: dict[str, list[str]] = defaultdict(list)

    for raw, lower in zip(raw_cols, lower_cols):
        if lower in HEADER_ALIASES:
            canon = HEADER_ALIASES[lower]
            mapped.append(canon)
            sources_by_canon[canon].append(raw)
        else:
            unknown.append(raw)

    if unknown:
        exp = ", ".join(sorted(set(HEADER_ALIASES.keys())))
        raise ValueError(f"Unrecognized headers: {unknown}. Expected one of: {exp}")

    dups = {k: v for k, v in sources_by_canon.items() if len(v) > 1}
    if dups:
        pretty = "; ".join(f"{k} ⇐ {v}" for k, v in dups.items())
        raise ValueError(f"Duplicate logical columns after header normalization: {pretty}")

    missing = [c for c in CANONICAL_COLS if c not in mapped]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df2 = df.copy()
    df2.columns = mapped
    df2 = df2[CANONICAL_COLS].copy()

    for c in CANONICAL_COLS:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    if df2.isnull().values.any():
        raise ValueError("All required columns must be numeric and non-empty.")

    must_pos = [
        "usage_rate",
        "lead_time_days",
        "item_cost",
        "safety_allowance_pct",
        "standard_pack",
        "hits_per_month",
    ]
    if (df2[must_pos] <= 0).any().any():
        raise ValueError("All values (except PNA) must be > 0.")

    return df2


def parse_contents(contents: str, filename: str | None, date) -> dbc.Card | dbc.Alert:
    """
    Build a pretty preview card for an uploaded file, or an alert if invalid.
    """
    try:
        decoded = _extract_base64(contents)
    except ValueError as e:
        return dbc.Alert(str(e), color="danger")

    try:
        lower = (filename or "").lower()
        if lower.endswith(".csv"):
            raw = pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))
        elif lower.endswith(".xls"):
            raw = pd.read_excel(io.BytesIO(decoded), engine="xlrd")
        elif lower.endswith(".xlsx"):
            raw = pd.read_excel(io.BytesIO(decoded), engine="openpyxl")
        else:
            return dbc.Alert("The input file must be .csv, .xls, or .xlsx.", color="warning")
        df = coerce_uploaded(raw)
    except Exception:
        return dbc.Alert("Problem reading file or invalid columns.", color="warning")

    def _to_native(v) -> Cell:
        """Make numpy scalars json/dash friendly."""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        if isinstance(v, np.generic):
            py = v.item()
            return py if isinstance(py, (bool, int, float, str)) else str(py)
        if isinstance(v, (bool, int, float, str)):
            return v
        return str(v)

    typed_records: List[Row] = []
    for rec in df.to_dict("records"):
        out: Row = {}
        for k, v in rec.items():
            out[str(k)] = _to_native(v)
        typed_records.append(out)

    try:
        ts = float(date)
        if ts > 1e11:  # ms → s
            ts /= 1000.0
        when_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        when_str = ""

    preview_cols = [str(c) for c in df.columns]
    preview_records = typed_records[:250]

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(filename or "Uploaded file", className="card-title"),
                html.H6(when_str, className="card-subtitle"),
                html.Br(),
                dash_table.DataTable(
                    data=cast(Sequence[Dict[str, Cell]], preview_records), # Ignore mypy false positive
                    columns=[{"name": c, "id": c} for c in preview_cols],
                    page_size=15,
                    page_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "rgb(30, 30, 30)", "color": "white"},
                    style_cell={
                        "backgroundColor": "rgb(50, 50, 50)",
                        "color": "white",
                        "border": "1px solid #444",
                        "minWidth": 80,
                        "maxWidth": 240,
                        "whiteSpace": "normal",
                    },
                    style_data={"border": "1px solid #444"},
                ),
                html.Hr(className="my-4"),
            ]
        ),
        className="mb-3",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demand simulation & sales
# ──────────────────────────────────────────────────────────────────────────────

def simulate_daily_hits(hpm: float) -> int:
    """Poisson hits per day (hpm ~ monthly)."""
    return int(_rng().poisson(max(0.0, float(hpm)) / 30.0))


def simulate_sales(avg_sale_qty: float, standard_pack: float) -> int:
    """Order size per hit sampled in pack-space to avoid over-demand from rounding.
    Expected units ≈ avg_sale_qty; result is a whole number of packs."""
    if avg_sale_qty <= 0:
        return 0
    pack = max(1.0, float(standard_pack))
    lam_packs = float(avg_sale_qty) / pack
    raw_packs = int(_rng().poisson(lam_packs))
    return int(raw_packs * pack)


def process_item_day(item: dict, today: int, gs: dict) -> tuple[dict, dict]:
    """
    Run one simulated day for a single item.

    Returns (updated_item, per_item_metrics) where per_item_metrics contains:
      orders, orders_stockout, units_ordered, units_shipped, units_backordered,
      zero_on_hand_hits, holding_add, stockout_add, revenue_add, cogs_add,
      purchases_add  # cost of receipts today (qty * item_cost)
    """
    it = dict(item)
    metrics = {
        "orders": 0,
        "orders_stockout": 0,
        "units_ordered": 0.0,
        "units_shipped": 0.0,
        "units_backordered": 0.0,
        "zero_on_hand_hits": 0,
        "holding_add": 0.0,
        "stockout_add": 0.0,
        "revenue_add": 0.0,
        "cogs_add": 0.0,
        "purchases_add": 0.0,
        "inv_value_mid_add": 0.0,
    }

    # Ensure ASQ period counters exist
    it.setdefault("asq_hits_period", 0)
    it.setdefault("asq_usage_period", 0.0)

    k_cost = float(gs["k_cost"])
    stockout_penalty = float(gs["stockout_penalty"])
    gm = float(gs.get("gm", 0.15))
    gm = min(max(gm, 0.0), 0.95)  # clamp
    realization = float(gs.get("realization", 0.92))
    realization = min(max(realization, 0.5), 1.0)  # clamp
    price_per_unit_multiplier = 1.0 / max(1e-9, (1.0 - gm))  # price = cost / (1-gm)

    # 1) Receive pipeline due today or earlier (book purchases on receipt)
    due = [r for r in it["pipeline"] if int(r["eta_day"]) <= int(today)]
    if due:
        qty_in = sum(float(r["qty"]) for r in due)
        it["on_hand"] = float(it["on_hand"]) + qty_in
        metrics["purchases_add"] = qty_in * float(it["item_cost"])
        it["pipeline"] = [r for r in it["pipeline"] if int(r["eta_day"]) > int(today)]

    # 2) Settle backorders if any on-hand
    if float(it["backorder"]) > 0 and float(it["on_hand"]) > 0:
        settle = min(float(it["backorder"]), float(it["on_hand"]))
        it["backorder"] = float(it["backorder"]) - settle
        it["on_hand"] = float(it["on_hand"]) - settle

    # 3) Demand simulation per hit
    hpm = max(0.01, float(it.get("hits_per_month", 0.0)))
    ur = max(0.0, float(it["usage_rate"]))
    avg_sale_qty = safe_div(ur, hpm, 0.0)

    on_hand_before = float(it["on_hand"])
    orders = simulate_daily_hits(hpm)
    stockout_any = False

    for _ in range(orders):
        q = simulate_sales(avg_sale_qty, it["standard_pack"])
        if q <= 0:
            continue
        metrics["orders"] += 1
        metrics["units_ordered"] += q

        # Track ASQ signal (orders & usage)
        it["asq_hits_period"] = int(it.get("asq_hits_period", 0)) + 1
        it["asq_usage_period"] = float(it.get("asq_usage_period", 0.0)) + float(q)

        if it["on_hand"] <= 0:
            # full stockout
            it["backorder"] = float(it["backorder"]) + q
            metrics["orders_stockout"] += 1
            metrics["zero_on_hand_hits"] += 1
            metrics["units_backordered"] += q
            stockout_any = True
        elif q <= it["on_hand"]:
            # full fill
            it["on_hand"] = float(it["on_hand"]) - q
            metrics["units_shipped"] += q
        else:
            # partial: ship what's available, backorder the rest
            short = q - float(it["on_hand"])
            metrics["units_shipped"] += float(it["on_hand"])
            metrics["units_backordered"] += short
            it["backorder"] = float(it["backorder"]) + short
            it["on_hand"] = 0.0
            metrics["orders_stockout"] += 1
            stockout_any = True

    on_hand_after = float(it["on_hand"])

    # 4) Daily costs
    midpoint_oh = (on_hand_before + on_hand_after) / 2.0
    metrics["holding_add"] = midpoint_oh * float(it["item_cost"]) * (k_cost / 365.0)
    metrics["stockout_add"] = metrics["units_backordered"] * stockout_penalty
    metrics["inv_value_mid_add"] = midpoint_oh * float(it["item_cost"])

    # 5) Daily sales from shipped units
    cogs_add = metrics["units_shipped"] * float(it["item_cost"])
    revenue_add = cogs_add * price_per_unit_multiplier * realization
    metrics["cogs_add"] = cogs_add
    metrics["revenue_add"] = revenue_add

    it["stockout_today"] = stockout_any

    # 6) Refresh derived planning fields
    it = update_planning_fields(it, gs)
    return it, metrics


# ──────────────────────────────────────────────────────────────────────────────
# PO helpers
# ──────────────────────────────────────────────────────────────────────────────

def place_purchase_orders(
    current: dict,
    today: int,
    gs: dict,
    *,
    recompute_total: bool = False,
) -> dict:
    """
    Create POs using each item's current SOQ (which is 0 when PNA > LP).
    Updates ordering cost and refreshes planning fields.

    Returns:
      {"lines": int, "total_qty": float,
       "receipts": [{"item_index": int, "rid": str, "qty": float, "eta_day": int}, ...]}
    """
    items = current.get("items", [])
    if not items:
        return {"lines": 0, "total_qty": 0.0, "receipts": []}
    lines = 0
    total_qty = 0.0
    receipts = []

    for idx, it in enumerate(items):
        qty = float(it.get("soq", 0.0))  # SOQ already encodes OP/LP policy + pack rounding
        if qty <= 0:
            continue

        rid = str(uuid.uuid4())[:8]
        lt  = float(it.get("lead_time", it.get("lead_time_days", 0.0)))
        eta = int(today + math.ceil(max(1.0, lt)))
        it.setdefault("pipeline", []).append({"id": rid, "qty": qty, "eta_day": eta})

        current["costs"]["ordering"] += float(gs["r_cost"])
        update_planning_fields(it, gs)

        lines += 1
        total_qty += qty
        receipts.append({"item_index": idx, "rid": rid, "qty": qty, "eta_day": eta})

    if recompute_total and lines:
        c = current["costs"]
        c["total"] = c["ordering"] + c["holding"] + c["stockout"] + c["expedite"]

    return {"lines": lines, "total_qty": total_qty, "receipts": receipts}



# ──────────────────────────────────────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────────────────────────────────────

def update_graph_based_on_items(items: list[dict], global_settings: dict):
    """Refresh main scatter with OP-normalized 'Days from OP' axis."""
    import plotly.graph_objects as go

    if not items:
        fig = px.scatter(title="Inventory Simulation",
                         labels={"x": "Items", "y": "Days from OP"})
        fig.add_hline(y=0, line_dash="dot", annotation_text="OP")
        fig.add_hline(y=global_settings["r_cycle"], line_dash="dot", annotation_text="LP")
        fig.update_layout(legend_title_text="", legend_tracegroupgap=6)
        return fig

    # Colors
    BLUE   = "#1f77b4"   # PNA (current)
    GREEN  = "#2ca02c"   # PNA + SOQ (proposed)
    ORANGE = "#ff7f0e"   # ATS (Available to Sell)
    RED    = "#d62728"   # 0 PNA guide

    df = pd.DataFrame(items).reset_index(names="idx")

    # Proposed point shown when it actually changes and item is at/under LP policy range
    mask_proposed = (df.get("pro_pna_days_frm_op") != df.get("pna_days_frm_op")) & (df["pna"] <= df["lp"])

    hover_tmpl = "Item = %{x}<br>Days from OP = %{y:.1f}<extra>%{fullData.name}</extra>"

    fig = go.Figure()
    fig.update_layout(title="Inventory Simulation", xaxis_title="Items", yaxis_title="Days from OP")

    # 1) PNA — bigger blue dots (keep this bigger)
    fig.add_scatter(
        x=(df["idx"] + 1),
        y=df["pna_days_frm_op"],
        mode="markers",
        name="PNA",
        marker=dict(symbol="circle", size=14, color=BLUE),  # bigger PNA only
        legendgroup="pna",
        legendrank=1,
        hovertemplate=hover_tmpl,
    )

    # 2) Proposed PNA + SOQ — revert to original size
    if mask_proposed.any():
        dfp = df[mask_proposed]
        fig.add_scatter(
            x=(dfp["idx"] + 1),
            y=dfp["pro_pna_days_frm_op"],
            mode="markers",
            name="PNA + SOQ",
            marker=dict(symbol="circle-open", size=8, line=dict(color=GREEN, width=2)),  # back to normal
            line=dict(color=GREEN),
            legendgroup="proposed",
            legendrank=2,
            hovertemplate=hover_tmpl,
        )

    # 3) ATS (Available to Sell) — revert to original size
    if "ats_days_frm_op" in df.columns:
        fig.add_scatter(
            x=(df["idx"] + 1),
            y=df["ats_days_frm_op"],
            mode="markers",
            name="Available to Sell",
            marker=dict(symbol="x", size=8, color=ORANGE),  # back to normal
            legendgroup="ats",
            legendrank=3,
            hovertemplate=hover_tmpl,
        )

    # 4) 0 PNA guide — keep at normal size
    if "no_pna_days_frm_op" in df.columns:
        fig.add_scatter(
            x=(df["idx"] + 1),
            y=df["no_pna_days_frm_op"],
            mode="markers",
            name="0 PNA",
            marker=dict(symbol="circle", size=8, color=RED),  # normal
            legendgroup="zero",
            legendrank=4,
            hovertemplate=hover_tmpl,
        )

    # Optional ring to call out items that stocked out today (around their PNA dot)
    if "stockout_today" in df.columns:
        idxs = (df.index[df["stockout_today"] == True]).tolist()
        if idxs:
            fig.add_scatter(
                x=(df.loc[idxs, "idx"] + 1),
                y=df.loc[idxs, "pna_days_frm_op"],
                mode="markers",
                name="Stockout Today",
                marker=dict(symbol="circle-open-dot", size=16, line=dict(width=2)),
                legendgroup="alerts",
                legendrank=0,
                hovertemplate=hover_tmpl,
            )

    # Reference lines: OP baseline at 0d, LP line at review-cycle days
    fig.add_hline(y=0, line_dash="dot", annotation_text="OP")
    fig.add_hline(y=global_settings["r_cycle"], line_dash="dot", annotation_text="LP")

    # X axis as integer item #s
    n = len(df)
    fig.update_xaxes(tickmode="linear", dtick=1, tick0=1, tickformat="d", range=[0.5, n + 0.5])

    # Legend polish
    fig.update_layout(legend_title_text="", legend_tracegroupgap=6)

    return fig
 

# ──────────────────────────────────────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────────────────────────────────────

def service_card_children(data: dict) -> list:
    st  = data.get("service_today", new_service_bucket())
    tot = data.get("service_totals", new_service_bucket())
    items = data.get("items", [])

    # ATS = Available to Sell = on_hand
    sum_ats = sum(float(i.get("on_hand", 0)) for i in items)
    sum_oo  = sum(item_on_order(i) for i in items)
    sum_bo  = sum(float(i.get("backorder", 0)) for i in items)

    fill_today = None if st["orders"] == 0 else 100.0 * (st["orders"] - st["orders_stockout"]) / st["orders"]
    fill_total = None if tot["orders"] == 0 else 100.0 * (tot["orders"] - tot["orders_stockout"]) / tot["orders"]
    fmt = lambda x: "—" if x is None else f"{x:.1f}%"

    return [dbc.ListGroup(
        [
            dbc.ListGroupItem(
                f"Today: Orders {st['orders']} • Stockouts {st['orders_stockout']} • Zero ATS {st['zero_on_hand_hits']}"
            ),
            dbc.ListGroupItem(
                f"Fill Rate (Cumulative): {fmt(fill_total)}  (Today: {fmt(fill_today)})"
            ),
            dbc.ListGroupItem(
                dbc.Badge(
                    f"Σ Available to Sell: {int(sum_ats)} • On-Order: {int(sum_oo)} • Backorder: {int(sum_bo)}",
                    pill=True, color="secondary",
                )
            ),
        ],
        flush=True,
    )]


def costs_card_children(data: dict) -> list:
    c = data.get("costs", new_costs_bucket())
    total = c["ordering"] + c["holding"] + c["stockout"] + c["expedite"]
    money = lambda x: format_money(float(x or 0.0))
    return [dbc.ListGroup(
        [
            dbc.ListGroupItem(f"Total: {money(total)}"),
            dbc.ListGroupItem(
                f"Ordering: {money(c['ordering'])} • "
                f"Holding: {money(c['holding'])} • "
                f"Stockout: {money(c['stockout'])} • "
                f"Expedite: {money(c['expedite'])}"
            ),
        ],
        flush=True,
    )]


def sales_card_children(data: dict) -> list:
    s = data.get("sales", new_sales_bucket()); c = data.get("costs", new_costs_bucket())
    revenue = float(s["revenue"]); cogs = float(s["cogs"])
    gross = revenue - cogs
    gm_pct = None if revenue <= 0 else (100.0 * gross / revenue)
    inv_over = float(c["total"])
    after_gm = gross - inv_over
    after_gm_pct = None if revenue <= 0 else (100.0 * after_gm / revenue)
    fmtp = lambda x: "—" if x is None else f"{x:.1f}%"
    money = lambda x: format_money(float(x or 0.0))

    return [dbc.ListGroup(
        [
            dbc.ListGroupItem(f"Sales: {money(revenue)}"),
            dbc.ListGroupItem(f"COGS: {money(cogs)} • GM$: {money(gross)} • GM%: {fmtp(gm_pct)}"),
            dbc.ListGroupItem(f"GM after Inv Costs: {money(after_gm)} • %: {fmtp(after_gm_pct)}"),
        ],
        flush=True,
    )]


# === KPI STRIP HELPERS =========================================================

def _fmt_pct(p: float | None) -> str:
    return "—" if p is None else f"{p*100:.1f}%"

def _color_from_thresholds(value: float | None, good: float, warn: float, reverse: bool = False) -> str:
    """
    Map a numeric value to a Bootstrap color.
    reverse=False  -> higher is better   (e.g., fill rate, margin)
    reverse=True   -> lower  is better   (e.g., bad-cost ratio)
    """
    if value is None:
        return "secondary"
    v = float(value)
    if reverse:
        if v <= good:  # small is good
            return "success"
        if v <= warn:
            return "warning"
        return "danger"
    else:
        if v >= good:  # big is good
            return "success"
        if v >= warn:
            return "warning"
        return "danger"

def _kpi_card(card_id: str, title: str, big_text: str, color: str, tooltip_children, tooltip_style=None) -> html.Div:
    body = dbc.CardBody(
        [
            html.Div(title, className="text-muted small mb-1"),
            html.Div(big_text, className="mb-0 fw-bold",
                     style={"fontSize": "1.6rem", "lineHeight": "1.1"}),
        ],
        className="py-2 px-3",
    )
    card = dbc.Card(
        body,
        id=card_id,
        color=color,
        inverse=True,
        className="text-center shadow-sm h-100",
        style={"minHeight": "88px"},
    )
    tip = dbc.Tooltip(
        tooltip_children,
        target=card_id,
        placement="bottom",
        autohide=True,
        className="text-start",
        # make it wider
        style=tooltip_style or {}
    )
    return html.Div([card, tip])


def build_kpi_strip(data: dict) -> list:
    def _fmt_money(x): return format_money(float(x or 0.0))
    def _fmt_pct(x):   return "—" if x is None else f"{x*100:.1f}%"

    st     = data.get("service_today", new_service_bucket())
    tot    = data.get("service_totals", new_service_bucket())
    costs  = data.get("costs", new_costs_bucket())
    sales  = data.get("sales", new_sales_bucket())
    day    = int(data.get("day", 1))

    months = day / 30.0
    years  = day / 365.0
    day_tip = html.Div([
        html.Div(f"Simulation Day: {day}"),
        html.Div(f"≈ {months:.1f} months (30-day)"),
        html.Div(f"≈ {years:.2f} years (365-day)")
    ])
    day_card = dbc.Col(_kpi_card("kpi-day", "Day", f"{day}d", "info", day_tip), className="d-flex")

    fill_total = None if tot["orders"] == 0 else (tot["orders"] - tot["orders_stockout"]) / tot["orders"]
    fill_today = None if st["orders"]  == 0 else (st["orders"]  - st["orders_stockout"])  / st["orders"]
    service_color = _color_from_thresholds(fill_total, good=0.95, warn=0.90, reverse=False)

    service_tip = html.Div(
        [
            html.Div(html.Strong("Today")),
            html.Div(
                f"Orders: {st['orders']} • Stockouts: {st['orders_stockout']} • Zero ATS Hits: {st['zero_on_hand_hits']}"
            ),
            html.Div(
                f"Units — Ordered: {int(st['units_ordered'])} • Shipped: {int(st['units_shipped'])} • Backordered: {int(st['units_backordered'])}"
            ),
            html.Hr(className="my-1"),
            html.Div(html.Strong("Cumulative")),
            html.Div(
                f"Orders: {tot['orders']} • Stockouts: {tot['orders_stockout']} • Zero ATS Hits: {tot['zero_on_hand_hits']}"
            ),
            html.Div(
                f"Units — Ordered: {int(tot['units_ordered'])} • Shipped: {int(tot['units_shipped'])} • Backordered: {int(tot['units_backordered'])}"
            ),
            html.Hr(className="my-1"),
            html.Div("Fill Rate (Today): " + _fmt_pct(fill_today)),
            html.Div("Fill Rate (Cumulative): " + _fmt_pct(fill_total)),
        ]
    )

    revenue = float(sales.get("revenue", 0.0))
    cogs    = float(sales.get("cogs", 0.0))
    gross   = revenue - cogs
    gm_pct  = None if revenue <= 0 else gross / max(1e-9, revenue)

    # === Avg Inventory at Cost (to-date) ===
    analytics = data.get("analytics", new_analytics_bucket())
    day = int(data.get("day", 1))
    avg_inv_cost = None
    if day > 0:
        inv_daysum = float(analytics.get("inv_value_daysum", 0.0))
        avg_inv_cost = inv_daysum / day if inv_daysum > 0 else None

    # === GMROI (standard) ===
    gm_dollars = gross
    gmroi = None if not avg_inv_cost or avg_inv_cost <= 0 else (gm_dollars / avg_inv_cost)
    gmroi_color = _color_from_thresholds(gmroi, good=2.5, warn=1.5, reverse=False)

    gmroi_tip = html.Div(
        [
            html.Div(html.Strong("GMROI (Gross Profit ÷ Avg Inventory at Cost)")),
            html.Div(f"Gross Profit: {_fmt_money(gm_dollars)}"),
            html.Div(f"Avg Inventory @ Cost: {_fmt_money(avg_inv_cost or 0.0)}"),
            html.Div(f"GMROI: {'—' if gmroi is None else f'{gmroi:.2f}x'}"),
            html.Hr(className="my-1"),
            html.Small("Avg inventory is computed from daily midpoint on-hand × cost."),
            html.Br(),
            html.Small("If you prefer AvgInv ÷ Gross Profit, invert this value."),
        ]
    )

    # === Inventory Turns (annualized) & TEI ===
    turns = None
    if avg_inv_cost and avg_inv_cost > 0 and day > 0:
        turns = (cogs / avg_inv_cost) * (365.0 / day)

    tei = None if (turns is None or gm_pct is None) else (turns * gm_pct)
    tei_color = _color_from_thresholds(tei, good=2.0, warn=1.0, reverse=False)

    tei_tip = html.Div(
        [
            html.Div(html.Strong("Turn & Earn (TEI) = Turns × GM%")),
            html.Div(f"Turns (annualized): {'—' if turns is None else f'{turns:.2f}x'}"),
            html.Div(f"GM%: {_fmt_pct(gm_pct)}"),
            html.Div(f"TEI: {'—' if tei is None else f'{tei:.2f}'}"),
            html.Hr(className="my-1"),
            html.Small("Turns = (COGS ÷ Avg Inventory) × (365 ÷ days run)."),
        ]
    )


    inv_overhead = float(costs.get("ordering", 0.0)) + float(costs.get("holding", 0.0)) \
                 + float(costs.get("stockout", 0.0)) + float(costs.get("expedite", 0.0))
    after_overhead = gross - inv_overhead
    after_overhead_pct = None if revenue <= 0 else after_overhead / max(1e-9, revenue)
    gm_color = _color_from_thresholds(gm_pct, good=0.12, warn=0.08, reverse=False)

    cogs_tip = html.Div([html.Div(f"COGS (to date): {_fmt_money(cogs)}"),
                         html.Small("COGS rises when units ship.")])
    sales_tip = html.Div([html.Div(f"Revenue (to date): {_fmt_money(revenue)}"),
                          html.Div(f"Units Sold: {int(sales.get('units_sold', 0))}")])
    overhead_tip = html.Div(
        [
            html.Div(f"Ordering: {_fmt_money(costs.get('ordering', 0.0))}"),
            html.Div(f"Holding: {_fmt_money(costs.get('holding', 0.0))}"),
            html.Div(f"Stockout: {_fmt_money(costs.get('stockout', 0.0))}"),
            html.Div(f"Expedite: {_fmt_money(costs.get('expedite', 0.0))}"),
            html.Small("Overhead excludes inventory receipts."),
        ]
    )
    gm_tip = html.Div(
        [
            html.Div(f"Gross Margin $: {_fmt_money(gross)}"),
            html.Div(f"GM%: {_fmt_pct(gm_pct)}"),
            html.Hr(className="my-1"),
            html.Div(f"After Overhead $: {_fmt_money(after_overhead)}"),
            html.Div(f"After Overhead %: {_fmt_pct(after_overhead_pct)}"),
        ]
    )

    return [
        day_card,dbc.Col(_kpi_card("kpi-service", "Fill Rate (Cumulative)", _fmt_pct(fill_total), service_color, service_tip, tooltip_style={"maxWidth": "520px", "--bs-tooltip-max-width": "520px"}),className="d-flex"),
        dbc.Col(_kpi_card("kpi-sales",   "Sales",                 _fmt_money(revenue),  "primary",      sales_tip),  className="d-flex"),
        dbc.Col(_kpi_card("kpi-cogs",    "COGS (to date)",        _fmt_money(cogs),     "primary",      cogs_tip),   className="d-flex"),
        dbc.Col(_kpi_card("kpi-allin",   "Inventory Overhead",    _fmt_money(inv_overhead), "primary",  overhead_tip), className="d-flex"),
        dbc.Col(_kpi_card("kpi-margin",  "GM%",                   _fmt_pct(gm_pct),     gm_color,       gm_tip),     className="d-flex"),
        dbc.Col(_kpi_card("kpi-gmroi",   "GMROI",               ("—" if gmroi is None else f"{gmroi:.2f}x"), gmroi_color, gmroi_tip, tooltip_style={"maxWidth":"520px"}), className="d-flex"),
        dbc.Col(_kpi_card("kpi-tei",     "Turn & Earn (TEI)",   ("—" if tei   is None else f"{tei:.2f}"),    tei_color,   tei_tip,   tooltip_style={"maxWidth":"520px"}), className="d-flex"),
    ]


def build_po_overview_table(data: dict) -> list:
    """Rows for the PO overview modal."""
    today = data.get("day", 1)
    rows = []
    any_receipts = False
    for idx, it in enumerate(data.get("items", [])):
        for r in it.get("pipeline", []):
            any_receipts = True
            days_left = max(0, int(r["eta_day"]) - int(today))
            rid = r["id"]
            rows.append(
                dbc.Row(
                    [
                        dbc.Col(html.P(str(idx + 1)), width=1),
                        dbc.Col(html.P(str(rid)), width=2),
                        dbc.Col(html.P(str(int(r["qty"]))), width=2),
                        dbc.Col(html.P(str(int(r["eta_day"]))), width=2),
                        dbc.Col(html.P(str(days_left)), width=2),
                        dbc.Col(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Expedite -1d", id={"type": "po-expedite", "rid": rid},
                                               size="sm", color="warning", className="me-2"),
                                    dbc.Button("Cancel", id={"type": "po-cancel", "rid": rid},
                                               size="sm", color="danger"),
                                ]
                            ),
                            width=3,
                        ),
                    ],
                    className="py-1",
                )
            )
    if not any_receipts:
        rows = [dbc.Alert("No open POs.", color="secondary")]

    header = dbc.Row(
        [
            dbc.Col(html.Strong("Item #"), width=1),
            dbc.Col(html.Strong("Receipt ID"), width=2),
            dbc.Col(html.Strong("Qty"), width=2),
            dbc.Col(html.Strong("ETA Day"), width=2),
            dbc.Col(html.Strong("Days Left"), width=2),
            dbc.Col(html.Strong("Actions"), width=3),
        ],
        className="mb-2",
    )
    return [header] + rows


def build_custom_order_row(index: int, item: dict):
    """Row builder for the custom order modal table."""
    return dbc.Row(
        [
            dbc.Col(html.P(str(index + 1)), width=1),
            dbc.Col(html.P(str(int(item.get("on_hand", 0))))),       # OH
            dbc.Col(html.P(str(int(item_on_order(item))))),          # On-Order
            dbc.Col(html.P(str(int(item.get("backorder", 0))))),     # BO
            dbc.Col(html.P(str(round(item["usage_rate"])))),
            dbc.Col(html.P(str(round(item["lead_time"])))),
            dbc.Col(html.P(str(round(item["op"])))),
            dbc.Col(html.P(str(round(item["lp"])))),
            dbc.Col(html.P(str(round(item["oq"])))),
            dbc.Col(
                dbc.Input(
                    value=int(round(item["soq"])),
                    type="number",
                    min=0,
                    id={"type": "order-quantity", "index": index},
                ),
                width=2,
            ),
        ],
        className="py-1",
    )

# Github Link Card
def github_footer_card() -> html.Div:
    return html.Div(
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Small(
                                [
                                    "To learn more, visit ",
                                    html.A(
                                        "SchmidtCode/IMSim on GitHub",
                                        href=GH_URL,
                                        target="_blank",
                                        rel="noopener noreferrer",
                                        className="link-light",
                                    ),
                                    ".",
                                ],
                                className="text-muted",
                            ),
                            className="pe-2",
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Hide",
                                id="gh-footer-hide",
                                size="sm",
                                color="secondary",
                                outline=True,
                            ),
                            width="auto",
                            className="text-end",
                        ),
                    ],
                    className="g-2 align-items-center",
                )
            ),
            className="shadow-sm border-0 bg-dark",
        ),
        id="gh-footer",
        style=GH_FOOTER_STYLE_VISIBLE,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layout (fixed Add Item + Excel upload inside modal; keeps ASQ + SOQ working)
# ──────────────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        # Top navbar
        dbc.NavbarSimple(
            children=[dbc.NavItem(dbc.NavLink("Inventory Management Simulator", href="#"))],
            brand="IMSim",
            brand_href="#",
            color="primary",
            dark=True,
            className="py-1",
            brand_style={"fontSize": "1rem", "fontWeight": 600},
            style={"paddingTop": "0.25rem", "paddingBottom": "0.25rem"},
        ),

        # Maintenance banner
        html.Div(id="maintenance-banner", className="mt-3"),

        # Main 2-column layout
        dbc.Row(
            [
                # Left column: controls + params + KPIs
                dbc.Col(
                    [
                        # Controls
                        dbc.Card(
                            [
                                dbc.CardHeader("Controls"),
                                dbc.CardBody(
                                    [
                                        dbc.Button("Start/Pause Simulation", id="start-button", n_clicks=0, color="success"),
                                        dbc.Button("Reset Simulation", id="reset-button", n_clicks=0, className="ms-2"),
                                        dbc.Button("Place Purchase Order", id="po-button", n_clicks=0, className="ms-2"),
                                        dbc.Button("Place Custom Order", id="place-custom-order-button", n_clicks=0,
                                                   color="warning", className="ms-2"),
                                        dbc.Button("PO Overview", id="po-overview-button", n_clicks=0,
                                                   color="info", className="ms-2"),
                                        dbc.Button("Add Item", id="add-item-button", n_clicks=0,
                                                   color="secondary", className="ms-2"),
                                        html.Hr(className="my-2 border-secondary opacity-25"),
                                    ],
                                    className="vstack gap-2",
                                ),
                            ],
                            style={"marginTop": "10px"},
                        ),

                        # Parameters (collapsible like the metrics accordion)
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    dbc.Accordion(
                                        [
                                            dbc.AccordionItem(
                                                title="Parameters",
                                                item_id="acc-params",
                                                className="text-light",
                                                children=[
                                                    # --- Core parameters ---
                                                    dbc.InputGroup(
                                                        [dbc.InputGroupText("Review Cycle (Days):"),
                                                        dbc.Input(id="review-cycle-input", type="number", value=14)],
                                                        className="mb-1",
                                                    ),
                                                    dbc.InputGroup(
                                                        [dbc.InputGroupText("R-Cost ($ per item ordered):"),
                                                        dbc.Input(id="r-cost-input", type="number", value=8)],
                                                        className="mb-1",
                                                    ),
                                                    dbc.InputGroup(
                                                        [dbc.InputGroupText("K-Cost (%/yr):"),
                                                        dbc.Input(id="k-cost-input", type="number", value=18.0)],
                                                        className="mb-1",
                                                    ),
                                                    dbc.InputGroup(
                                                        [dbc.InputGroupText("Stockout Penalty ($/unit):"),
                                                        dbc.Input(id="stockout-penalty-input", type="number", value=5.0)],
                                                        className="mb-1",
                                                    ),
                                                    dbc.InputGroup(
                                                        [dbc.InputGroupText("Expedite Rate (% cost/unit/day):"),
                                                        dbc.Input(id="expedite-rate-input", type="number", value=3.0)],
                                                        className="mb-1",
                                                    ),
                                                    dbc.InputGroup(
                                                        [dbc.InputGroupText("Global GM (%):"),
                                                        dbc.Input(id="gm-input", type="number", value=15.0)],
                                                        className="mb-1",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("Realization (% of list):"),
                                                            dbc.Input(
                                                                id="realization-input",
                                                                type="number",
                                                                value=100.0,    # default 100%
                                                                min=50,         # matches clamp in process_item_day (0.5–1.0)
                                                                max=100,
                                                                step=1
                                                            ),
                                                        ],
                                                        className="mb-1",
                                                    ),

                                                    html.Hr(className="my-1 border-secondary opacity-25"),

                                                    # --- Auto-PO toggle (simple replenishment rule) ---
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                dbc.Switch(
                                                                    id="auto-po-enabled",
                                                                    label="Auto Purchase Orders (use SOQ when PNA ≤ LP, don't forget to click Update Parameters)",
                                                                    value=False,
                                                                    className="text-light",
                                                                )
                                                            )
                                                        ],
                                                        className="mb-1",
                                                    ),

                                                    # --- ASQ settings (still collapsible; button toned down to 'secondary') ---
                                                    dbc.Button(
                                                        "ASQ OP Adjuster Settings",
                                                        id="toggle-asq-collapse",
                                                        color="secondary",          # match 'Add Item'
                                                        outline=False,
                                                        size="sm",
                                                        className="w-100 mb-1"
                                                    ),
                                                    dbc.Collapse(
                                                        id="asq-collapse",
                                                        is_open=False,
                                                        className="text-light",
                                                        children=[
                                                            html.H6("ASQ OP Adjuster", className="mb-1"),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dbc.Switch(
                                                                            id="asq-enabled",
                                                                            label="Enable ASQ OP Adjuster",
                                                                            value=True,
                                                                            className="text-light",
                                                                        )
                                                                    )
                                                                ],
                                                                className="mb-1",
                                                            ),
                                                            dbc.InputGroup(
                                                                [dbc.InputGroupText("ASQ Min Line Hits:"),
                                                                dbc.Input(id="asq-min-hits", type="number", value=3, min=0)],
                                                                className="mb-1",
                                                            ),
                                                            dbc.InputGroup(
                                                                [dbc.InputGroupText("ASQ Max $ Diff:"),
                                                                dbc.Input(id="asq-max-diff", type="number", value=2500.0, min=0, step=10)],
                                                                className="mb-1",
                                                            ),
                                                            dbc.InputGroup(
                                                                [dbc.InputGroupText("ASQ Period (Days):"),
                                                                dbc.Input(id="asq-period-days", type="number", value=30, min=1)],
                                                                className="mb-2",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dbc.Switch(
                                                                            id="asq-include-transfers",
                                                                            label="Include Transfers in ASQ",
                                                                            value=False,
                                                                            className="text-light",
                                                                        )
                                                                    )
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                            dbc.Button(
                                                                "Apply Month-End (ASQ) Now",
                                                                id="apply-asq-button",
                                                                n_clicks=0,
                                                                color="warning",
                                                                className="w-100 mb-1",
                                                            ),
                                                            dbc.Row(id="asq-apply-feedback"),
                                                        ],
                                                    ),

                                                    dbc.Row(id="update-params-conf"),
                                                    dbc.Button(
                                                        "Update Parameters",
                                                        id="update-params-button",
                                                        n_clicks=0,
                                                        color="primary",
                                                        className="w-100",
                                                    ),
                                                ],
                                            )
                                        ],
                                        start_collapsed=True,
                                        always_open=False,
                                        id="params-accordion",
                                    ),
                                    className="vstack gap-1",   # tighter vertical spacing
                                )
                            ],
                            style={"marginTop": "10px"},
                        ),

                        # Simulation (status + speed) — not collapsible
                        dbc.Card(
                            [
                                dbc.CardHeader("Simulation"),
                                dbc.CardBody(
                                    [
                                        # Keep for callbacks but hide (Day now lives in top KPI cards)
                                        html.Div(id="day-display", children=f"Day: {1}", style={"display": "none"}),

                                        html.Div(id="sim-status", children="Status: Paused", className="mb-2"),

                                        dbc.Label("Simulation Speed (ms):", className="d-block mb-1"),
                                        dcc.Slider(
                                            id="sim-speed-slider",
                                            min=150, max=2000, value=600,
                                            marks={150: "Fast", 2000: "Slow"},
                                            step=50,
                                        ),
                                    ],
                                    className="vstack gap-2",
                                ),
                            ],
                            style={"marginTop": "10px"},
                        ),

                        # Metrics accordion only (Service • Costs • Sales & Margin)
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dbc.Accordion(
                                            [
                                                dbc.AccordionItem(
                                                    [
                                                        html.Div(id="service-card"),
                                                        html.Hr(className="my-2 border-secondary opacity-25"),
                                                        html.Div(id="costs-card"),
                                                        html.Hr(className="my-2 border-secondary opacity-25"),
                                                        html.Div(id="sales-card"),
                                                    ],
                                                    title="Service • Costs • Sales & Margin",
                                                    item_id="acc-all",
                                                    className="text-light",
                                                )
                                            ],
                                            start_collapsed=True,
                                            always_open=False,
                                            id="left-kpi-accordion",
                                        ),
                                        html.Div(id="store-output", style={"display": "none"}),
                                    ],
                                    className="vstack gap-2",
                                )
                            ],
                            style={"marginTop": "10px"},
                        ),

                        # Local stores
                        dcc.Store(id="user-data-store", storage_type="local", data={}),
                        dcc.Store(id="page-load", data=0),
                        dcc.Store(id="gh-footer-store", storage_type="local", data=True),
                    ],
                    width=3,
                ),
                github_footer_card(),

                # Right column: KPI strip + graph
                dbc.Col(
                    [
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    id="kpi-strip",
                                    className="row-cols-2 row-cols-md-3 row-cols-lg-auto gx-3 gy-2 mb-2 align-items-stretch"
                                ),
                                dcc.Graph(
                                    id="inventory-graph",
                                    style={"height": "68vh"},
                                    figure=px.scatter(title="Inventory Simulation"),
                                ),
                            ]
                        )
                    ],
                    width=9,
                    style={"marginTop": "10px"},
                ),
            ]
        ),

        # Tickers
        dcc.Interval(id="interval-component", interval=600, n_intervals=0, max_intervals=-1, disabled=True),
        dcc.Interval(id="shutdown-poll", interval=1000, n_intervals=0),

        # Add Item modal (manual + upload back inside here)
        Modal(
            [
                ModalHeader("Add New Item"),
                ModalBody(
                    [
                        # Manual add
                        dbc.Card(
                            [
                                dbc.CardHeader("Manually Add Item"),
                                dbc.CardBody(
                                    [
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("Usage Rate (per month):"),
                                             dbc.Input(id="usage-rate-input", type="number", value=20, min=1, step=1)]
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("Lead Time (days):"),
                                             dbc.Input(id="lead-time-input", type="number", value=30, min=1, step=1)]
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("Item Cost ($):"),
                                             dbc.Input(id="item-cost-input", type="number", value=100, min=0.01, step=0.01)]
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("Initial PNA (units):"),
                                             dbc.Input(id="pna-input", type="number", value=50, min=0, step=1)]
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("Safety Allowance (%):"),
                                             dbc.Input(id="safety-allowance-input", type="number", value=50, min=1, step=1)]
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("Standard Pack:"),
                                             dbc.Input(id="standard-pack-input", type="number", value=10, min=1, step=1)]
                                        ),
                                        dbc.InputGroup(
                                            [dbc.InputGroupText("Hits Per Month:"),
                                             dbc.Input(id="hits-per-month-input", type="number", value=5, min=0.01, step=0.01)]
                                        ),
                                        dbc.Row(id="add-item-error"),
                                        dbc.Button("Randomize", id="randomize-button", color="secondary", className="me-md-2"),
                                        dbc.Button("Add item", id="submit-item-button", color="primary"),
                                    ],
                                    className="end",
                                ),
                            ]
                        ),

                        # Upload section (CSV / Excel)
                        dbc.Card(
                            [
                                dbc.CardHeader("Upload Items"),
                                dbc.CardBody(
                                    [
                                        dcc.Upload(
                                            id="upload-item",
                                            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                                            accept=".csv, .xls, .xlsx",
                                            style={
                                                "width": "100%",
                                                "height": "60px",
                                                "lineHeight": "60px",
                                                "borderWidth": "1px",
                                                "borderStyle": "dashed",
                                                "borderRadius": "5px",
                                                "textAlign": "center",
                                                "margin": "10px",
                                            },
                                            multiple=True,
                                        ),
                                        html.Div(id="output-item-upload"),
                                        dbc.Button(
                                            "Import items",
                                            id="import-uploaded-items",
                                            color="primary",
                                            className="mt-2",
                                            style={"display": "none"},  # reveals when preview validates
                                        ),
                                        html.Div(id="upload-feedback"),
                                        dcc.Store(id="upload-preview-data"),
                                    ]
                                ),
                            ],
                            style={"marginTop": "10px"},
                        ),
                    ]
                ),
            ],
            id="add-item-modal",
            is_open=False,
            style={"maxHeight": "calc(95vh)", "overflowY": "auto"},
            size="lg",
        ),

        # Custom Order modal
        Modal(
            [
                ModalHeader("Place Custom Order"),
                ModalBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(html.Strong("Item #"), width=1),
                                dbc.Col(html.Strong("Available to Sell")),
                                dbc.Col(html.Strong("On-Order")),
                                dbc.Col(html.Strong("Backorder")),
                                dbc.Col(html.Strong("Usage")),
                                dbc.Col(html.Strong("Lead Time")),
                                dbc.Col(html.Strong("OP")),
                                dbc.Col(html.Strong("LP")),
                                dbc.Col(html.Strong("OQ")),
                                dbc.Col(html.Strong("Order Qty"), width=2),
                            ]
                        ),
                        html.Div(id="custom-order-items-div"),
                    ]
                ),
                ModalFooter(
                    [
                        dbc.Button("Cancel", id="cancel-custom-order-button", color="secondary"),
                        dbc.Button("Place Order", id="place-order-button", color="primary"),
                    ]
                ),
            ],
            id="place-custom-order-modal",
            is_open=False,
            style={"maxHeight": "calc(95vh)", "overflowY": "auto"},
            size="lg",
        ),

        # PO Overview modal
        Modal(
            [
                ModalHeader("PO Overview"),
                ModalBody([html.Div(id="po-overview-table")]),
                ModalFooter([dbc.Button("Close", id="po-overview-close", color="secondary")]),
            ],
            id="po-overview-modal",
            is_open=False,
            size="lg",
            style={"maxHeight": "calc(95vh)", "overflowY": "auto"},
        ),
    ],
    fluid=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("user-data-store", "data", allow_duplicate=True),
    Input("page-load", "data"),
    State("user-data-store", "data"),
    prevent_initial_call=True,
)
def ensure_uuid_on_load(page_load, data):
    """Ensure client-side store has a uuid when the page first loads."""
    data = data or {}
    if not data.get("uuid"):
        data["uuid"] = str(uuid.uuid4())
    return data


@app.callback(
    Output("user-data-store", "data", allow_duplicate=True),
    Input("user-data-store", "modified_timestamp"),
    State("user-data-store", "data"),
    prevent_initial_call=True,
)
def set_uuid(ts, data):
    """Fallback: set uuid when local store first changes."""
    if ts is None or (data and isinstance(data, dict) and data.get("uuid")):
        raise PreventUpdate
    base = data if isinstance(data, dict) else {}
    if not base.get("uuid"):
        base["uuid"] = str(uuid.uuid4())
        return base
    else:
        raise PreventUpdate


def render_service_costs(data: dict):
    """Helper to render both cards together; kept for compatibility."""
    return service_card_children(data), costs_card_children(data)


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
    prevent_initial_call=True,
)
def update_on_interval(n_intervals, client_data):
    """One simulation tick: advance a day, run items in parallel, update KPIs, and apply ASQ at period end."""
    if not n_intervals or not client_data:
        raise PreventUpdate
    uuid_val = client_data.get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current = get_user_data(uuid_val)
    if not current.get("is_initialized", False) or not current.get("items"):
        raise PreventUpdate

    # Advance day
    today = int(current.get("day", 1)) + 1
    current["day"] = today

    gs = current["global_settings"]
    asq_cfg = dict(gs.get("asq", asq_defaults()))
    period_days = max(1, int(asq_cfg.get("period_days", 30)))

    # Reset today's service
    current["service_today"] = new_service_bucket()
    holding_today = 0.0
    stockout_today_cost = 0.0
    revenue_today = 0.0
    cogs_today = 0.0
    purchases_today = 0.0
    inv_mid_today_value = 0.0

    # Process each item
    futures = [executor.submit(process_item_day, item, today, gs) for item in current["items"]]
    new_items = []
    for fut in futures:
        it, met = fut.result()
        new_items.append(it)

        # Aggregate service
        for k in [
            "orders",
            "orders_stockout",
            "units_ordered",
            "units_shipped",
            "units_backordered",
            "zero_on_hand_hits",
        ]:
            current["service_today"][k] += met[k]
            current["service_totals"][k] += met[k]

        # Aggregate costs & sales
        holding_today += met["holding_add"]
        stockout_today_cost += met["stockout_add"]
        revenue_today += met["revenue_add"]
        cogs_today += met["cogs_add"]
        purchases_today += met.get("purchases_add", 0.0)
        inv_mid_today_value += met.get("inv_value_mid_add", 0.0)

    current["items"] = new_items

    # Auto-PO rule: when enabled, place SOQ orders automatically
    if gs.get("auto_po_enabled", False):
        _ = place_purchase_orders(
            current=current,
            today=today,
            gs=gs,
            recompute_total=False,  # total recomputed below
        )

    # Update costs
    current["costs"]["holding"] += holding_today
    current["costs"]["stockout"] += stockout_today_cost
    current["costs"]["purchases"] += purchases_today
    current["costs"]["total"] = (
        current["costs"]["ordering"]
        + current["costs"]["holding"]
        + current["costs"]["stockout"]
        + current["costs"]["expedite"]
    )

    # Update sales
    current.setdefault("sales", new_sales_bucket())
    current["sales"]["revenue"] += revenue_today
    current["sales"]["cogs"] += cogs_today
    current["sales"]["units_sold"] += current["service_today"]["units_shipped"]

    # Update analytics (for avg inventory at cost)  ← ADD HERE
    current.setdefault("analytics", new_analytics_bucket())
    current["analytics"]["inv_value_daysum"] += inv_mid_today_value

    # Month-end ASQ application (every N days)
    asq_feedback = dash.no_update
    if today % period_days == 0:
        summary = apply_asq_month_end(current)
        asq_feedback = dbc.Alert(
            f"ASQ month-end applied — OP raised on {summary['changed']} item(s).",
            color="info",
            duration=4000,
        )

    # Persist + redraw
    fig = update_graph_based_on_items(current["items"], gs)
    set_user_data(uuid_val, current)
    save_data()

    return (
        f"Day: {today}",
        fig,
        service_card_children(current),
        costs_card_children(current),
        sales_card_children(current),
        asq_feedback,
    )


@app.callback(
    [
        Output("sim-status", "children", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True),
        Output("start-button", "children"),
        Output("start-button", "color"),
    ],
    Input("start-button", "n_clicks"),
    State("user-data-store", "data"),
    State("interval-component", "disabled"),
    prevent_initial_call=True,
)
def toggle_simulation(n_clicks, client_data, is_disabled):
    """Start/pause the simulation clock."""
    if not n_clicks or not client_data:
        raise PreventUpdate
    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current = get_user_data(uuid_val)

    if is_disabled:
        current["is_initialized"] = True
        set_user_data(uuid_val, current)
        save_data()
        return "Status: Running", False, "Pause Simulation", "warning"
    else:
        return "Status: Paused", True, "Resume Simulation", "success"


@app.callback(
    [
        Output("day-display", "children", allow_duplicate=True),
        Output("inventory-graph", "figure", allow_duplicate=True),
        Output("sim-status", "children", allow_duplicate=True),
        Output("user-data-store", "data", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True),
        Output("start-button", "children", allow_duplicate=True),
        Output("start-button", "color", allow_duplicate=True),
        Output("start-button", "disabled", allow_duplicate=True),
        Output("service-card", "children", allow_duplicate=True),
        Output("costs-card", "children", allow_duplicate=True),
        Output("sales-card", "children", allow_duplicate=True),
        Output("asq-apply-feedback", "children", allow_duplicate=True),
    ],
    Input("reset-button", "n_clicks"),
    State("user-data-store", "data"),
    prevent_initial_call=True,
)
def reset_simulation(n_clicks, client_data):
    """Reset the whole simulation/session to defaults (keep same uuid)."""
    if not n_clicks:
        raise PreventUpdate

    default_data = get_default_data()
    uuid_val = (client_data or {}).get("uuid", "__anon__")
    set_user_data(uuid_val, default_data)
    save_data()

    fig = update_graph_based_on_items(default_data["items"], default_data["global_settings"])
    return (
        f"Day: {default_data['day']}",
        fig,
        "Status: Paused",
        default_data,
        True,
        "Start Simulation",
        "success",
        False,
        service_card_children(default_data),
        costs_card_children(default_data),
        sales_card_children(default_data),
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
):
    """Open add-item modal or accept a new item and refresh graph/KPIs."""
    if ctx.triggered_id is None:
        raise PreventUpdate

    if ctx.triggered_id == "add-item-button":
        return (not is_open), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if ctx.triggered_id == "submit-item-button":
        all_inputs = [usage_rate, lead_time, item_cost, pna, safety_allowance, standard_pack, hits_per_month]
        if any(i is None for i in all_inputs):
            return (
                is_open,
                dbc.Alert("All fields must be filled out!", color="warning", duration=4000),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        if any(i <= 0 for i in [usage_rate, lead_time, item_cost, safety_allowance, standard_pack, hits_per_month]):
            return (
                is_open,
                dbc.Alert("All params except PNA must be > 0!", color="warning", duration=4000),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        uuid_val = (client_data or {}).get("uuid")
        if not uuid_val:
            return (
                is_open,
                dbc.Alert("Session not initialized.", color="danger", duration=4000),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        current = get_user_data(uuid_val)
        gs = current.get("global_settings", get_default_data()["global_settings"])

        item = create_inventory_item(
            usage_rate=usage_rate,
            lead_time=lead_time,
            item_cost=item_cost,
            pna=pna,
            safety_allowance=safety_allowance / 100.0,
            standard_pack=standard_pack,
            global_settings=gs,
            hits_per_month=float(hits_per_month),
        )
        current.setdefault("items", []).append(item)
        set_user_data(uuid_val, current)
        save_data()

        fig = update_graph_based_on_items(current["items"], current["global_settings"])
        return False, None, fig, service_card_children(current), costs_card_children(current), sales_card_children(current)

    raise PreventUpdate


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
        # ASQ settings
        State("asq-enabled", "value"),
        State("asq-min-hits", "value"),
        State("asq-max-diff", "value"),
        State("asq-period-days", "value"),
        State("asq-include-transfers", "value"),
        State("user-data-store", "data"),
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
):
    """Apply global parameter changes (including ASQ config) and recompute planning."""
    if not n_clicks:
        raise PreventUpdate
    
    if auto_po_enabled is None:
        auto_po_enabled = False

    if None in (
        review_cycle, r_cost, k_cost_pct, stockout_penalty, expedite_rate_pct,
        gm_pct, realization_pct, asq_min_hits, asq_max_diff, asq_period_days
    ):
        return (
            dbc.Alert("Please fill out all parameters!", color="warning", duration=4000),
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
        )

    if (
        review_cycle <= 0
        or r_cost <= 0
        or k_cost_pct <= 0
        or stockout_penalty < 0
        or expedite_rate_pct < 0
        or gm_pct < 0 or gm_pct >= 100
        or realization_pct < 50 or realization_pct > 100
        or asq_min_hits < 0
        or asq_max_diff < 0
        or asq_period_days <= 0
    ):
        return (
            dbc.Alert("Invalid parameter values.", color="danger", duration=4000),
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
        )

    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current = get_user_data(uuid_val)
    if "global_settings" not in current:
        return (
            dbc.Alert("Global settings not found.", color="danger", duration=4000),
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
        )

    # Standard globals
    current["global_settings"] = update_global_settings(
        current["global_settings"],
        int(review_cycle),
        float(r_cost),
        float(k_cost_pct) / 100.0,         # % → decimal
        float(stockout_penalty),
        float(expedite_rate_pct) / 100.0,  # % → decimal
        float(gm_pct) / 100.0,             # % → decimal
    )

    current["global_settings"]["realization"] = float(realization_pct) / 100.0

    current["global_settings"]["auto_po_enabled"] = bool(auto_po_enabled)

    # ASQ config
    current["global_settings"]["asq"] = {
        "enabled": bool(asq_enabled),
        "min_hits": int(asq_min_hits),
        "include_transfers": bool(asq_include_transfers),
        "max_amount_diff": float(asq_max_diff),
        "period_days": int(asq_period_days),
    }

    # Recompute planning fields for all items (OP unchanged here)
    if current.get("items"):
        current["items"] = [update_gs_related_values(it, current["global_settings"]) for it in current["items"]]

    set_user_data(uuid_val, current)
    save_data()

    fig = update_graph_based_on_items(current["items"], current["global_settings"])
    return (
        dbc.Alert("Parameters updated!", color="success", duration=3000),
        fig,
        service_card_children(current),
        costs_card_children(current),
        sales_card_children(current),
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
    prevent_initial_call=True,
)
def handle_apply_asq_now(n_clicks, client_data):
    """Manually trigger month-end ASQ application."""
    if not n_clicks:
        raise PreventUpdate
    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate
    current = get_user_data(uuid_val)
    if not current.get("items"):
        return dbc.Alert("No items to adjust.", color="secondary", duration=3000), dash.no_update, dash.no_update, dash.no_update, dash.no_update

    summary = apply_asq_month_end(current)
    set_user_data(uuid_val, current)
    save_data()
    fig = update_graph_based_on_items(current["items"], current["global_settings"])
    alert = dbc.Alert(
        f"ASQ applied — OP raised on {summary['changed']} item(s).",
        color="info",
        duration=4000,
    )
    return alert, fig, service_card_children(current), costs_card_children(current), sales_card_children(current)


@app.callback(
    [
        Output("inventory-graph", "figure", allow_duplicate=True),
        Output("costs-card", "children", allow_duplicate=True),
        Output("service-card", "children", allow_duplicate=True),
        Output("sales-card", "children", allow_duplicate=True),
    ],
    Input("po-button", "n_clicks"),
    State("user-data-store", "data"),
    prevent_initial_call=True,
)
def handle_purchase_order(n_clicks, client_data):
    """Manual: user presses the button → place SOQs once."""
    if not n_clicks:
        raise PreventUpdate
    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current = get_user_data(uuid_val)
    if not current.get("items"):
        raise PreventUpdate

    gs = current["global_settings"]
    today = int(current["day"])

    _ = place_purchase_orders(
        current=current,
        today=today,
        gs=gs,
        recompute_total=True,  # reflect ordering cost immediately in the UI
    )

    set_user_data(uuid_val, current)
    save_data()

    fig = update_graph_based_on_items(current["items"], current["global_settings"])
    return fig, costs_card_children(current), service_card_children(current), sales_card_children(current)


@app.callback(
    [
        Output("inventory-graph", "figure", allow_duplicate=True),
        Output("custom-order-items-div", "children"),
        Output("place-custom-order-modal", "is_open"),
        Output("sim-status", "children", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True),
        Output("start-button", "children", allow_duplicate=True),
        Output("start-button", "color", allow_duplicate=True),
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
        State("interval-component", "disabled"),
    ],
    prevent_initial_call=True,
)
def handle_custom_order_and_modal_actions(
    place_order_clicks,
    cancel_clicks,
    submit_clicks,
    order_quantities,
    client_data,
    is_interval_disabled,
):
    """Handle open/close custom order modal and placing custom POs."""
    if ctx.triggered_id is None:
        raise PreventUpdate

    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current = get_user_data(uuid_val)
    items = current.get("items", [])
    rows = [build_custom_order_row(index, it) for index, it in enumerate(items)]

    # Open modal
    if ctx.triggered_id == "place-custom-order-button":
        return (
            dash.no_update,
            rows,
            True,
            "Status: Paused",
            True,
            "Resume Simulation",
            "success",
            True,
            costs_card_children(current),
            service_card_children(current),
            sales_card_children(current),
        )

    # Cancel modal
    if ctx.triggered_id == "cancel-custom-order-button":
        return (
            dash.no_update,
            dash.no_update,
            False,
            "Status: Paused",
            True,
            "Resume Simulation",
            "success",
            False,
            costs_card_children(current),
            service_card_children(current),
            sales_card_children(current),
        )

    # Place custom orders
    if ctx.triggered_id == "place-order-button":
        if not items:
            return (
                dash.no_update,
                dbc.Alert("No items available.", color="warning"),
                True,
                "Status: Paused",
                True,
                "Resume Simulation",
                "success",
                True,
                costs_card_children(current),
                service_card_children(current),
                sales_card_children(current),
            )

        gs = current["global_settings"]
        today = current["day"]
        changed = False

        for index, qty in enumerate(order_quantities or []):
            if qty is None:
                continue
            q = max(0.0, float(qty))
            if q <= 0:
                continue
            it = items[index]
            q = round_to_pack(q, it["standard_pack"])
            if q <= 0:
                continue

            rid = str(uuid.uuid4())[:8]
            eta = int(today + math.ceil(max(1.0, float(it["lead_time"]))))
            it["pipeline"].append({"id": rid, "qty": q, "eta_day": eta})
            current["costs"]["ordering"] += float(gs["r_cost"])
            update_planning_fields(it, gs)
            changed = True

        if changed:
            current["costs"]["total"] = (
                current["costs"]["ordering"]
                + current["costs"]["holding"]
                + current["costs"]["stockout"]
                + current["costs"]["expedite"]
            )

        set_user_data(uuid_val, current)
        save_data()
        fig = update_graph_based_on_items(items, current["global_settings"])
        return (
            fig,
            rows,
            False,
            "Status: Paused",
            True,
            "Resume Simulation",
            "success",
            False,
            costs_card_children(current),
            service_card_children(current),
            sales_card_children(current),
        )

    raise PreventUpdate


# Page-load to seed graph + KPIs
@app.callback(
    [
        Output("day-display", "children"),
        Output("inventory-graph", "figure"),
        Output("page-load", "data"),
        Output("service-card", "children"),
        Output("costs-card", "children"),
        Output("sales-card", "children"),
    ],
    Input("page-load", "data"),
    State("user-data-store", "data"),
)
def handle_page_load(page_load, client_data):
    """Initial render: fetch user data, draw graph, seed KPI cards."""
    if page_load == 0:
        current = (
            get_user_data(client_data["uuid"]) if client_data and "uuid" in client_data
            else get_default_data()
        )
        day_count = current.get("day", 1)
        fig = update_graph_based_on_items(
            current.get("items", []),
            current.get("global_settings", get_default_data()["global_settings"]),
        )
        return (
            f"Day: {day_count}",
            fig,
            1,
            service_card_children(current),
            costs_card_children(current),
            sales_card_children(current),
        )
    else:
        raise PreventUpdate


# Upload flow
@callback(
    [
        Output("output-item-upload", "children"),
        Output("upload-preview-data", "data"),
        Output("import-uploaded-items", "style"),
    ],
    Input("upload-item", "contents"),
    State("upload-item", "filename"),
    State("upload-item", "last_modified"),
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    """Preview uploaded files and enable 'Import items' when valid."""
    if not list_of_contents:
        raise PreventUpdate

    cards = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
    frames = []
    errors = []

    for contents, name in zip(list_of_contents, list_of_names):
        try:
            decoded = _extract_base64(contents)
            lower = (name or "").lower()
            if lower.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))
            elif lower.endswith(".xls"):
                df = pd.read_excel(io.BytesIO(decoded), engine="xlrd")
            elif lower.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(decoded), engine="openpyxl")
            else:
                errors.append(f"{name}: unsupported file type")
                continue
        except Exception as e:
            errors.append(f"{name}: failed to decode/read file ({e})")
            continue

        try:
            df_num = coerce_uploaded(df)
        except Exception as e:
            return (cards + [dbc.Alert(f"'{name}': {e}", color="warning")], None, {"display": "none"})

        frames.append(df_num)

    if errors:
        return (cards + [dbc.Alert("; ".join(errors), color="warning")], None, {"display": "none"})
    if not frames:
        return (cards + [dbc.Alert("No valid rows found.", color="warning")], None, {"display": "none"})

    combined = pd.concat(frames, ignore_index=True)
    ready = html.Div(f"Ready to import {len(combined)} rows.", className="mb-2")
    return cards + [ready], combined.to_dict("records"), {}


@callback(
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
    prevent_initial_call=True,
)
def import_uploaded_items(n_clicks, rows, client_data):
    """Ingest validated rows and create inventory items."""
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

    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        return (
            dash.no_update,
            dbc.Alert("Session not initialized.", color="danger"),
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    current = get_user_data(uuid_val)
    gs = current.get("global_settings", get_default_data()["global_settings"])

    df = pd.DataFrame(rows)
    try:
        df = coerce_uploaded(df)
    except Exception as e:
        return (
            dash.no_update,
            dbc.Alert(f"Import error: {e}", color="danger"),
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    new_items = []
    for row_num, (_, rec) in enumerate(df.iterrows(), start=1):
        try:
            item = create_inventory_item(
                usage_rate=float(rec["usage_rate"]),
                lead_time=float(rec["lead_time_days"]),
                item_cost=float(rec["item_cost"]),
                pna=float(rec["pna"]),
                safety_allowance=float(rec["safety_allowance_pct"]) / 100.0,
                standard_pack=float(rec["standard_pack"]),
                global_settings=gs,
                hits_per_month=float(rec["hits_per_month"]),
            )
            new_items.append(item)
        except Exception as e:
            return (
                dash.no_update,
                dbc.Alert(f"Row {row_num} import error: {e}", color="danger"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

    current.setdefault("items", []).extend(new_items)
    set_user_data(uuid_val, current)
    save_data()

    fig = update_graph_based_on_items(current["items"], gs)
    return (
        fig,
        dbc.Alert(f"Imported {len(new_items)} items successfully.", color="success"),
        service_card_children(current),
        costs_card_children(current),
        sales_card_children(current),
    )


@app.callback(
    Output("interval-component", "interval"),
    Input("sim-speed-slider", "value"),
)
def set_sim_speed(ms):
    """Update the tick interval from the speed slider."""
    if ms is None:
        raise PreventUpdate
    return int(ms)


@app.callback(
    Output("asq-collapse", "is_open"),
    Input("toggle-asq-collapse", "n_clicks"),
    State("asq-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_asq_collapse(n, is_open):
    if not n:
        raise PreventUpdate
    return not is_open


# ──────────────────────────────────────────────────────────────────────────────
# PO Overview actions
# ──────────────────────────────────────────────────────────────────────────────

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
    prevent_initial_call=True,
)
def po_overview_handler(open_clicks, close_clicks, expedite_clicks, cancel_clicks, client_data):
    """Open/close the PO overview and handle expedite/cancel row actions."""
    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate
    current = get_user_data(uuid_val)
    is_open = dash.no_update

    trig = ctx.triggered_id
    if trig is None:
        raise PreventUpdate

    if trig == "po-overview-button":
        is_open = True
    elif trig == "po-overview-close":
        is_open = False
    else:
        # Row action
        if isinstance(trig, dict) and "rid" in trig:
            rid = trig["rid"]
            gs = current["global_settings"]
            today = int(current["day"])

            for it in current.get("items", []):
                pip = it.get("pipeline", [])
                for r in list(pip):
                    if r["id"] == rid:
                        if trig.get("type") == "po-expedite":
                            # Expedite by 1 day if possible (eta >= today+2)
                            if int(r["eta_day"]) > (today + 1):
                                r["eta_day"] = max(r["eta_day"] - 1, today + 1)
                                # Expedite fee = qty * cost * rate * days(=1)
                                fee = float(r["qty"]) * float(it["item_cost"]) * float(gs["expedite_rate"]) * 1.0
                                current["costs"]["expedite"] += fee
                                current["costs"]["total"] = (
                                    current["costs"]["ordering"]
                                    + current["costs"]["holding"]
                                    + current["costs"]["stockout"]
                                    + current["costs"]["expedite"]
                                )
                        elif trig.get("type") == "po-cancel":
                            # Remove receipt; PNA drops
                            it["pipeline"] = [x for x in pip if x["id"] != rid]
                            update_planning_fields(it, gs)
                        break

    set_user_data(uuid_val, current)
    save_data()
    fig = update_graph_based_on_items(current["items"], current["global_settings"])
    table_children = build_po_overview_table(current)
    return is_open, table_children, fig, costs_card_children(current)


# === KPI STRIP RENDER ==========================================================
@app.callback(
    Output("kpi-strip", "children"),
    Input("inventory-graph", "figure"),
    State("user-data-store", "data"),
)
def render_kpi_strip(_fig, client_data):
    """
    Build the KPI row whenever the graph refreshes.
    (All actions that change data already update the graph, so KPIs stay in sync.)
    """
    uuid_val = (client_data or {}).get("uuid")
    current = get_user_data(uuid_val) if uuid_val else get_default_data()
    return build_kpi_strip(current)
# ==============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# Utility endpoints
# ──────────────────────────────────────────────────────────────────────────────

@server.route("/shutdown", methods=["POST"])
def shutdown():
    if os.environ.get("ALLOW_DEV_SHUTDOWN") != "1":
        return abort(404)
    save_data()
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    return "Server shutting down..."


# ──────────────────────────────────────────────────────────────────────────────
# Randomizer (add-item helper)
# ──────────────────────────────────────────────────────────────────────────────

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
)
def randomize_item_values(n):
    """Populate the add-item form with a plausible random item."""
    if not n:
        raise PreventUpdate
    rng = _rng()
    usage = int(rng.integers(1, 101))
    lead = 0
    while lead < 7:
        lead = int(abs(rng.normal(30, 30)))  # keep it at least a week
    cost = max(1, int(abs(rng.normal(100, 100))))
    safety_pct = 50 if lead < 60 else max(1, int(round(3000 / lead)))
    pack = int(rng.choice([1, 5, 10, 20, 25, 40, 50], p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]))
    pna = int(round(usage * (lead / 30.0) + ((usage * (lead / 30.0)) * (safety_pct / 100.0)) + pack))
    hpm = max(1, int(rng.poisson(5)))
    return usage, lead, cost, pna, safety_pct, pack, hpm

# ──────────────────────────────────────────────────────────────────────────────
# GitHub footer (hide/show, persist state)
# ──────────────────────────────────────────────────────────────────────────────
# Hide button -> persist hidden state
@app.callback(
    Output("gh-footer-store", "data"),
    Input("gh-footer-hide", "n_clicks"),
    prevent_initial_call=True,
)
def hide_footer(n):
    if not n:
        raise PreventUpdate
    return False  # store "hidden"

# Store -> apply visibility style
@app.callback(
    Output("gh-footer", "style"),
    Input("gh-footer-store", "data"),
)
def set_footer_visibility(is_visible):
    return GH_FOOTER_STYLE_VISIBLE if (is_visible is not False) else GH_FOOTER_STYLE_HIDDEN


# ──────────────────────────────────────────────────────────────────────────────
# Shutdown scheduler (admin utility)
# ──────────────────────────────────────────────────────────────────────────────

@app.callback(
    [
        Output("maintenance-banner", "children"),
        Output("sim-status", "children", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True),
        Output("start-button", "children", allow_duplicate=True),
        Output("start-button", "color", allow_duplicate=True),
        Output("start-button", "disabled", allow_duplicate=True),
    ],
    Input("shutdown-poll", "n_intervals"),
    prevent_initial_call=True,
)
def maintenance_heartbeat(_n):
    # Snapshot the state (avoid long lock time)
    with _store_lock:
        st = dict(_shutdown_state)

    if not st.get("active"):
        # No maintenance scheduled: clear banner, leave everything else alone
        return html.Div(), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    now = time.time()
    remaining = max(0.0, st["at"] - now)
    mins = int(remaining // 60)
    secs = int(remaining % 60)

    banner = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span("⚠️ ", className="me-1"),
                        html.Strong(st["message"]),
                        html.Span(f" — shutting down in {mins:02d}:{secs:02d}", className="ms-1"),
                    ],
                    className="mb-1"
                ),
                html.Small(
                    "Your session will auto-save and pause. Please finish any edits.",
                    className="text-muted"
                ),
            ],
            role="status",
            className="py-2 px-3",
        ),
        className="shadow-sm border border-danger text-danger bg-transparent"
    )

    # Default: only show banner, don't touch sim until final minute
    sim_status = dash.no_update
    interval_disabled = dash.no_update
    start_text = dash.no_update
    start_color = dash.no_update
    start_disabled = dash.no_update

    # In the final minute: pause/lock controls so users don't lose work mid-tick
    if remaining <= 60:
        sim_status = "Status: Paused (maintenance)"
        interval_disabled = True
        start_text = "Disabled for Maintenance"
        start_color = "secondary"
        start_disabled = True

    # Time to shut down: pause everything, persist, then hit /shutdown
    if remaining <= 0.5 and not st.get("closing"):
        gentle_stop_all_sessions()
        with _store_lock:
            _shutdown_state["closing"] = True
        try:
            # Fire-and-forget; works with your existing @server.route('/shutdown')
            requests.post(SHUTDOWN_URL, timeout=1)
        except Exception:
            # In prod (gunicorn/uwsgi), /shutdown may not exist. Orchestrator should SIGTERM.
            pass

    return banner, sim_status, interval_disabled, start_text, start_color, start_disabled


# ──────────────────────────────────────────────────────────────────────────────
# Background shutdown watchdog (works even with no UI sessions)
# ──────────────────────────────────────────────────────────────────────────────
import threading

_watchdog_started = False

def _shutdown_watchdog():
    while True:
        time.sleep(1.0)
        with _store_lock:
            st = dict(_shutdown_state)
        if not st.get("active"):
            continue
        remaining = st["at"] - time.time()
        if remaining <= 0.5 and not st.get("closing"):
            gentle_stop_all_sessions()
            with _store_lock:
                _shutdown_state["closing"] = True
            try:
                requests.post(SHUTDOWN_URL, timeout=1)
            except Exception:
                pass

def _start_watchdog_once():
    global _watchdog_started
    if _watchdog_started:
        return
    t = threading.Thread(target=_shutdown_watchdog, daemon=True)
    t.start()
    _watchdog_started = True

# Start watchdog in the reloader child (dev) OR whenever debug env isn't on
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("FLASK_DEBUG") not in ("1", "true", "True"):
    _start_watchdog_once()

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # threaded=True lets callbacks run concurrently (e.g., futures above)
    app.run(debug=True, threaded=True)
