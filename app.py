import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html, Input, Output, State, callback, ALL, dash_table
from dash import ctx
import dash_bootstrap_components as dbc
from dash_bootstrap_components import Modal, ModalHeader, ModalBody, ModalFooter
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import math
from concurrent.futures import ThreadPoolExecutor
import uuid
import pandas as pd
import numpy as np
import json
from flask import Flask, request
import os
import base64
import datetime
import io
from threading import local
from threading import RLock
from typing import Dict, List, Union, TypeAlias, cast, Sequence
from collections import defaultdict

executor = ThreadPoolExecutor(max_workers=4)  # Define the number of worker threads
_store_lock = RLock()

load_figure_template("darkly")

server = Flask(__name__)
# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], server=server)

# server-side data store
user_data_store = {}

# Thread-local RNG for NumPy (recommended over global legacy RNG)
_thread_local = local()

Number: TypeAlias = Union[int, float]
Cell: TypeAlias = Union[bool, int, float, str]  # what DataTable cells allow
Row: TypeAlias = Dict[str, Cell]

engine = "openpyxl"


def _rng():
    if not hasattr(_thread_local, "gen"):
        _thread_local.gen = np.random.default_rng()
    return _thread_local.gen


# Default data structure
def get_default_data():
    return {
        "global_settings": {"r_cycle": 14, "r_cost": 8, "k_cost": 0.18},
        "items": [],
        "day": 1,
        "is_initialized": False,
    }


def get_user_data(user_id: str) -> dict:
    with _store_lock:
        if user_data_store.get(user_id) is None:
            user_data_store[user_id] = get_default_data()
        return user_data_store[user_id]


def save_data():
    os.makedirs("data", exist_ok=True)
    with _store_lock, open("data/user_data.json", "w") as f:
        json.dump(user_data_store, f)


def set_user_data(user_id: str, data: dict):
    with _store_lock:
        user_data_store[user_id] = data


# Shutdown and Startup Procedure to save data
def on_shutdown():
    with open("data/user_data.json", "w") as f:
        json.dump(user_data_store, f)


def load_data():
    if os.path.exists("data/user_data.json"):
        with open("data/user_data.json", "r") as f:
            return json.load(f)
    return {}


user_data_store = load_data()


def create_global_settings(r_cycle=14, r_cost=8, k_cost=0.18):
    return {"r_cycle": r_cycle, "r_cost": r_cost, "k_cost": k_cost}


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
    usage_rate = max(0.0, float(usage_rate))
    lead_time = max(0.0, float(lead_time))
    item_cost = max(0.0, float(item_cost))
    pna = max(0.0, float(pna))
    safety_allowance = max(0.0, float(safety_allowance))
    standard_pack = max(1.0, float(standard_pack))
    hits_per_month = max(1.0, float(hits_per_month))

    daily_ur = safe_div(usage_rate, 30.0, 0.0)
    pna_days = safe_div(pna, daily_ur, 0.0)

    op = calculate_op(usage_rate, lead_time, safety_allowance)
    lp = calculate_lp(usage_rate, global_settings, op)
    eoq = calculate_eoq(usage_rate, item_cost, global_settings)
    oq = calculate_oq(eoq, usage_rate, global_settings)
    soq = calculate_soq(pna, lp, op, oq, standard_pack)
    surplus_line = calculate_surplus_line(lp, eoq)
    cp = calculate_critical_point(usage_rate, lead_time)

    proposed_pna = pna + soq

    item = {
        "usage_rate": usage_rate,
        "lead_time": lead_time,
        "item_cost": item_cost,
        "pna": pna,
        "safety_allowance": safety_allowance,
        "standard_pack": standard_pack,
        "daily_ur": daily_ur,
        "pna_days": pna_days,
        "op": op,
        "pna_days_frm_op": safe_div((pna - op), max(1e-9, usage_rate), 0.0) * 30.0,
        "lp": lp,
        "eoq": eoq,
        "oq": oq,
        "soq": soq,
        "surplus_line": surplus_line,
        "cp": cp,
        "proposed_pna": proposed_pna,
        "pro_pna_days_frm_op": safe_div((proposed_pna - op), max(1e-9, usage_rate), 0.0)
        * 30.0,
        "no_pna_days_frm_op": safe_div((0.0 - op), max(1e-9, usage_rate), 0.0) * 30.0,
        "hits_per_month": hits_per_month,
    }
    return item


def safe_div(n, d, default=0.0):
    try:
        return n / d if d not in (0, 0.0, None) else default
    except Exception:
        return default


def calculate_op(
    usage_rate: float, lead_time_days: float, safety_allowance: float
) -> float:
    # usage_rate is per month (per your UI), lead_time is in DAYS
    monthly_lt = lead_time_days / 30.0
    safety_stock = usage_rate * monthly_lt * safety_allowance
    return usage_rate * monthly_lt + safety_stock


def calculate_lp(usage_rate: float, global_settings: dict, op: float) -> float:
    review_cycle_days = global_settings["r_cycle"]
    return usage_rate * (review_cycle_days / 30.0) + op


def calculate_eoq(usage_rate: float, item_cost: float, global_settings: dict) -> float:
    r_cost = global_settings["r_cost"]
    k_cost = global_settings["k_cost"]
    if item_cost <= 0 or k_cost <= 0 or usage_rate <= 0:
        return 0.0
    # Your EOQ variant:
    return math.sqrt((24.0 * r_cost * usage_rate) / (k_cost * item_cost))


def calculate_oq(eoq: float, usage_rate: float, global_settings: dict) -> float:
    rc_days = global_settings["r_cycle"]
    oq_min = max(0.5 * usage_rate, (rc_days / 30.0) * usage_rate)
    oq_max = 12.0 * usage_rate
    if eoq <= 0:
        return oq_min
    return max(min(eoq, oq_max), oq_min)


def round_to_pack(qty: float, pack: float) -> float:
    if pack is None or pack <= 0:
        return max(0.0, float(qty))
    return max(0.0, round(float(qty) / float(pack)) * float(pack))


def calculate_soq(
    pna: float, lp: float, op: float, oq: float, standard_pack: float
) -> float:
    if pna > lp:
        return 0.0
    base = oq if pna > op else (oq + op - pna)
    return round_to_pack(base, standard_pack)


def calculate_surplus_line(lp: float, eoq: float) -> float:
    return lp + max(0.0, eoq)


def calculate_critical_point(usage_rate: float, lead_time_days: float) -> float:
    # If CP should be expressed in "units" at risk during lead time (monthly rate → daily)
    return safe_div(usage_rate, 30.0) * max(0.0, lead_time_days)


def _extract_base64(contents: str) -> bytes:
    """
    Return the decoded bytes from a dcc.Upload 'contents' string.
    Accepts both full data URLs and raw base64 strings.
    """
    if not isinstance(contents, str):
        raise ValueError("Invalid upload payload type.")
    # Always returns a 3-tuple; sep == "" means comma not found
    _prefix, sep, b64data = contents.partition(",")
    b64data = b64data if sep else contents  # handle raw base64 (no prefix)
    try:
        return base64.b64decode(b64data, validate=False)  # tolerate newlines/whitespace
    except Exception as e:
        raise ValueError(f"Could not decode uploaded file: {e}")


def coerce_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize incoming header strings
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

    # Detect duplicate logical columns after aliasing
    dups = {k: v for k, v in sources_by_canon.items() if len(v) > 1}
    if dups:
        pretty = "; ".join(f"{k} ⇐ {v}" for k, v in dups.items())
        raise ValueError(
            f"Duplicate logical columns after header normalization: {pretty}"
        )

    # Ensure all required canonical columns are present
    missing = [c for c in CANONICAL_COLS if c not in mapped]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Rename to canonical names and select only the required seven
    df2 = df.copy()
    df2.columns = mapped  # all columns are now canonical names
    df2 = df2[CANONICAL_COLS].copy()

    # Coerce numerics on the canonical set only
    for c in CANONICAL_COLS:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    if df2.isnull().values.any():
        raise ValueError("All required columns must be numeric and non-empty.")

    # Enforce positivity rules (PNA may be zero)
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

    # Base64 decode
    try:
        decoded = _extract_base64(contents)
    except ValueError as e:
        return dbc.Alert(str(e), color="danger")

    # Read file -> DataFrame
    try:
        lower = (filename or "").lower()
        if lower.endswith(".csv"):
            raw = pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))
        elif lower.endswith(".xls"):
            raw = pd.read_excel(io.BytesIO(decoded), engine="xlrd")
        elif lower.endswith(".xlsx"):
            raw = pd.read_excel(io.BytesIO(decoded), engine="openpyxl")
        else:
            return dbc.Alert(
                "The input file must be .csv, .xls, or .xlsx.", color="warning"
            )

        # Normalize/validate columns (your pipeline)
        df = coerce_uploaded(raw)
    except Exception as e:
        return dbc.Alert(str(e), color="warning")

    # Convert values to JSON-serializable primitives for DataTable
    def _to_native(v) -> Cell:
        # Treat NaNs as empty strings in the preview
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass

        # Ensure numpy scalars become Python scalars (int/float/bool)
        if isinstance(v, np.generic):
            py = v.item()
            if isinstance(py, (bool, int, float, str)):
                return py  # type: ignore[return-value]
            return str(py)

        if isinstance(v, (bool, int, float, str)):
            return v
        return str(v)

    # Ensure keys are strings and values are DataTable-safe cells
    typed_records: List[Row] = []
    for rec in df.to_dict("records"):
        out: Row = {}
        for k, v in rec.items():
            out[str(k)] = _to_native(v)
        typed_records.append(out)

    # Format timestamp (Upload.last_modified is often ms)
    try:
        ts = float(date)
        if ts > 1e11:  # very likely milliseconds
            ts /= 1000.0
        when_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        when_str = ""

    # Cap preview rows for performance on large uploads
    preview_cols = [str(c) for c in df.columns]
    preview_records = typed_records[:250]

    # Build preview card
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(filename or "Uploaded file", className="card-title"),
                html.H6(when_str, className="card-subtitle"),
                html.Br(),
                dash_table.DataTable(
                    data=cast(Sequence[Dict[str, Cell]], preview_records),
                    columns=[{"name": c, "id": c} for c in preview_cols],
                    page_size=15,
                    page_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "rgb(30, 30, 30)",
                        "color": "white",
                    },
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


def initial_graph(store_data=None):
    if not store_data or "items" not in store_data or not store_data["items"]:
        fig = px.scatter(title="Inventory Simulation")
        fig.update_layout(
            xaxis=dict(autorange=True, title="Items"),
            yaxis=dict(autorange=True, title="PNA (Days from OP)"),
        )
    else:
        fig = update_graph_based_on_items(
            store_data["items"], store_data["global_settings"]
        )
    return fig


def simulate_daily_hits(hpm: float) -> int:
    return int(_rng().poisson(max(0.0, float(hpm)) / 30.0))


def simulate_sales(avg_sale_qty: float, standard_pack: float) -> int:
    if avg_sale_qty <= 0:
        return 0
    raw = int(_rng().poisson(avg_sale_qty))
    if raw <= 0:
        return 0
    pack = max(1.0, float(standard_pack))
    return int(math.ceil(raw / pack) * pack)


def process_item(item: dict) -> dict:
    # Treat dict as input, produce a *new* dict (avoid cross-thread side-effects)
    it = dict(item)
    hpm = max(1.0, float(it["hits_per_month"]))  # clamp
    ur = max(0.0, float(it["usage_rate"]))
    avg_sale_qty = safe_div(ur, hpm, 0.0)

    hits = simulate_daily_hits(hpm)
    total = 0
    for _ in range(hits):
        total += simulate_sales(avg_sale_qty, it["standard_pack"])

    it["pna"] = float(it["pna"]) - total
    return update_pna_related_values(it)


def update_pna_related_values(item: dict) -> dict:
    ur = max(0.0, float(item["usage_rate"]))
    op = float(item["op"])
    pna = float(item["pna"])
    item["pna_days_frm_op"] = safe_div((pna - op), ur, 0.0) * 30.0
    item["pna_days"] = safe_div(pna, max(1e-9, item["daily_ur"]), 0.0)
    item["soq"] = calculate_soq(pna, item["lp"], op, item["oq"], item["standard_pack"])
    item["surplus_line"] = calculate_surplus_line(item["lp"], item["eoq"])
    item["proposed_pna"] = pna + item["soq"]
    item["pro_pna_days_frm_op"] = safe_div((item["proposed_pna"] - op), ur, 0.0) * 30.0
    return item


def update_global_settings(current_settings, review_cycle, r_cost, k_cost):
    current_settings["r_cycle"] = review_cycle
    current_settings["r_cost"] = r_cost
    current_settings["k_cost"] = k_cost
    return current_settings


def update_gs_related_values(item, global_settings):
    item["lp"] = calculate_lp(item["usage_rate"], global_settings, item["op"])
    item["eoq"] = calculate_eoq(item["usage_rate"], item["item_cost"], global_settings)
    item["oq"] = calculate_oq(item["eoq"], item["usage_rate"], global_settings)
    item["soq"] = calculate_soq(
        item["pna"], item["lp"], item["op"], item["oq"], item["standard_pack"]
    )
    item["surplus_line"] = calculate_surplus_line(item["lp"], item["eoq"])
    item["proposed_pna"] = item["pna"] + item["soq"]
    ur = max(0.0, float(item["usage_rate"]))
    item["pro_pna_days_frm_op"] = (
        safe_div((item["proposed_pna"] - item["op"]), ur, 0.0) * 30.0
    )
    return item


def update_graph_based_on_items(items: list[dict], global_settings: dict):
    if not items:
        return px.scatter(
            title="Inventory Simulation",
            labels={"x": "Items", "y": "PNA (Days from OP)"},
        )

    # fixed palette
    BLUE = "#1f77b4"  # current
    GREEN = "#2ca02c"  # proposed (outline)
    RED = "#d62728"  # zero PNA

    df = pd.DataFrame(items).reset_index(names="idx")
    mask = (df["pro_pna_days_frm_op"] != df["pna_days_frm_op"]) & (
        df["pna"] <= df["lp"]
    )

    # base: current PNA (filled blue)
    fig = px.scatter(
        df,
        x=df["idx"] + 1,
        y="pna_days_frm_op",
        title="Inventory Simulation",
        labels={"x": "Items", "pna_days_frm_op": "PNA (Days from OP)"},
    )
    fig.update_traces(
        mode="markers",
        name="Current PNA Days from OP",
        marker=dict(symbol="circle", size=8, color=BLUE),
        legendgroup="current",
        legendrank=1,
    )

    # proposed: outlined green
    if mask.any():
        df2 = df[mask]
        fig.add_scatter(
            x=(df2["idx"] + 1),
            y=df2["pro_pna_days_frm_op"],
            mode="markers",
            name="PNA + SOQ Days from OP",
            marker=dict(
                symbol="circle-open",
                size=10,
                color=GREEN,  # for open symbols, this is the outline color
                line=dict(color=GREEN, width=2),
            ),
            legendgroup="proposed",
            legendrank=2,
        )

    # zero PNA: filled red
    fig.add_scatter(
        x=(df["idx"] + 1),
        y=df["no_pna_days_frm_op"],
        mode="markers",
        name="0 PNA",
        marker=dict(symbol="circle", size=8, color=RED),
        legendgroup="zero",
        legendrank=3,
    )

    # guides
    fig.add_hline(y=0, line_dash="dot", annotation_text="OP")
    fig.add_hline(y=global_settings["r_cycle"], line_dash="dot", annotation_text="LP")

    # integers on X
    n = len(df)
    fig.update_xaxes(
        tickmode="linear",
        dtick=1,
        tick0=1,
        tickformat="d",
        range=[0.5, n + 0.5],
    )
    fig.update_layout(yaxis_title="PNA (Days from OP)")

    return fig


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


app.title = "IM Sim"

app.layout = dbc.Container(
    [
        # Navigation Bar
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Inventory Management Simulator", href="#")),
            ],
            brand="CEEUS",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Controls"),
                                dbc.CardBody(
                                    [
                                        dbc.Button(
                                            "Start/Pause Simulation",
                                            id="start-button",
                                            n_clicks=0,
                                            color="success",
                                        ),
                                        dbc.Button(
                                            "Reset Simulation",
                                            id="reset-button",
                                            n_clicks=0,
                                        ),
                                        dbc.Button(
                                            "Place Purchase Order",
                                            id="po-button",
                                            n_clicks=0,
                                        ),
                                        dbc.Button(
                                            "Place Custom Order",
                                            id="place-custom-order-button",
                                            n_clicks=0,
                                            color="warning",
                                        ),
                                        dbc.Button(
                                            "Add Item",
                                            id="add-item-button",
                                            n_clicks=0,
                                            color="info",
                                        ),
                                    ],
                                    className="vstack gap-2",
                                ),
                            ],
                            style={"margin-top": "10px"},
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText(
                                                    "Review Cycle (Days):"
                                                ),
                                                dbc.Input(
                                                    id="review-cycle-input",
                                                    type="number",
                                                    value=14,
                                                ),
                                            ]
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("R-Cost ($):"),
                                                dbc.Input(
                                                    id="r-cost-input",
                                                    type="number",
                                                    value=8,
                                                ),
                                            ]
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("K-Cost (%):"),
                                                dbc.Input(
                                                    id="k-cost-input",
                                                    type="number",
                                                    value=0.18 * 100,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(id="update-params-conf"),
                                        dbc.Button(
                                            "Update Parameters",
                                            id="update-params-button",
                                            n_clicks=0,
                                            color="primary",
                                            className="w-100 mb-2",
                                        ),
                                        dbc.Label(
                                            "Simulation Speed (ms):",
                                            className="d-block mb-2",
                                        ),
                                        dcc.Slider(
                                            id="sim-speed-slider",
                                            min=150,
                                            max=2000,
                                            value=600,
                                            marks={150: "Fast", 2000: "Slow"},
                                            step=50,
                                        ),
                                    ],
                                    className="vstack gap-2",
                                ),
                            ],
                            style={"margin-top": "10px"},
                        ),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="day-display", children=f"Day: {1}"
                                        ),
                                        html.Div(
                                            id="sim-status", children="Status: Paused"
                                        ),
                                        html.Div(id="store-output"),
                                    ],
                                    className="vstack gap-2",
                                )
                            ],
                            style={"margin-top": "10px"},
                        ),
                        dcc.Store(id="user-data-store", storage_type="local", data={}),
                        dcc.Store(id="page-load", data=0),
                        dcc.Store(id="page-load-2", data=0),
                    ],
                    width=2,
                ),
                # Right column for the graph
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Inventory Graph"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="inventory-graph",
                                            style={"height": "80vh"},
                                            figure=initial_graph(),
                                        )
                                    ]
                                ),
                            ]
                        )
                    ],
                    width=10,
                    style={"margin-top": "10px"},
                ),
            ]
        ),
        dcc.Interval(
            id="interval-component",
            interval=600,
            n_intervals=0,
            max_intervals=-1,
            disabled=True,
        ),
        Modal(
            [
                ModalHeader("Add New Item"),
                ModalBody(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Manually Add Item"),
                                dbc.CardBody(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Usage Rate:"),
                                                dbc.Input(
                                                    id="usage-rate-input",
                                                    type="number",
                                                    value=0,
                                                ),
                                            ],
                                            style={"margin-bottom": "10px"},
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Lead Time:"),
                                                dbc.Input(
                                                    id="lead-time-input",
                                                    type="number",
                                                    value=0,
                                                ),
                                            ],
                                            style={"margin-bottom": "10px"},
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Item Cost:"),
                                                dbc.Input(
                                                    id="item-cost-input",
                                                    type="number",
                                                    value=0,
                                                ),
                                            ],
                                            style={"margin-bottom": "10px"},
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Initial PNA:"),
                                                dbc.Input(
                                                    id="pna-input",
                                                    type="number",
                                                    value=0,
                                                ),
                                            ],
                                            style={"margin-bottom": "10px"},
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText(
                                                    "Safety Allowance (%):"
                                                ),
                                                dbc.Input(
                                                    id="safety-allowance-input",
                                                    type="number",
                                                    value=50,
                                                ),
                                            ],
                                            style={"margin-bottom": "10px"},
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Standard Pack:"),
                                                dbc.Input(
                                                    id="standard-pack-input",
                                                    type="number",
                                                    value=0,
                                                ),
                                            ],
                                            style={"margin-bottom": "10px"},
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Hits Per Month:"),
                                                dbc.Input(
                                                    id="hits-per-month-input",
                                                    type="number",
                                                    value=0,
                                                ),
                                            ],
                                            style={"margin-bottom": "10px"},
                                        ),
                                        dbc.Row(id="add-item-error"),
                                        dbc.Button(
                                            "Randomize",
                                            id="randomize-button",
                                            color="secondary",
                                            className="me-md-2",
                                        ),
                                        dbc.Button(
                                            "Add item",
                                            id="submit-item-button",
                                            color="primary",
                                        ),
                                    ],
                                    className="end",
                                ),
                            ]
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Upload Items"),
                                dbc.CardBody(
                                    [
                                        dcc.Upload(
                                            id="upload-item",
                                            children=html.Div(
                                                [
                                                    "Drag and Drop or ",
                                                    html.A("Select Files"),
                                                ]
                                            ),
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
                                            multiple=True,  # Allow multiple files to be uploaded
                                        ),
                                        html.Div(id="output-item-upload"),
                                        dbc.Button(
                                            "Import items",
                                            id="import-uploaded-items",
                                            color="primary",
                                            className="mt-2",
                                            style={"display": "none"},
                                        ),
                                        html.Div(id="upload-feedback"),
                                        dcc.Store(id="upload-preview-data"),
                                    ]
                                ),
                            ],
                            style={"margin-top": "10px"},
                        ),
                    ]
                ),
            ],
            id="add-item-modal",
            is_open=False,  # by default, the modal is not open
            style={"maxHeight": "calc(95vh)", "overflowY": "auto"},
            size="lg",
        ),
        Modal(
            [
                ModalHeader("Place Custom Order"),
                ModalBody(
                    [
                        dbc.Row(
                            [  # This is the header row
                                dbc.Col(html.Strong("Item Index"), width=2),
                                dbc.Col(html.Strong("PNA")),
                                dbc.Col(html.Strong("Usage Rate")),
                                dbc.Col(html.Strong("Lead Time")),
                                dbc.Col(html.Strong("OP")),
                                dbc.Col(html.Strong("LP")),
                                dbc.Col(html.Strong("OQ")),
                                dbc.Col(html.Strong("Order Quantity"), width=2),
                            ]
                        ),
                        html.Div(
                            id="custom-order-items-div"
                        ),  # This is where the items will be populated
                    ]
                ),
                ModalFooter(
                    [
                        dbc.Button(
                            "Cancel", id="cancel-custom-order-button", color="secondary"
                        ),
                        dbc.Button(
                            "Place Order", id="place-order-button", color="primary"
                        ),
                    ]
                ),
            ],
            id="place-custom-order-modal",
            is_open=False,
            style={"maxHeight": "calc(95vh)", "overflowY": "auto"},
            size="lg",
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("user-data-store", "data", allow_duplicate=True),
    Input("page-load", "data"),
    State("user-data-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def ensure_uuid_on_load(page_load, data):
    data = data or {}
    if not data.get("uuid"):
        data["uuid"] = str(uuid.uuid4())
    return data


# Callback to set a new UUID for each session
@app.callback(
    Output("user-data-store", "data"),
    Input(
        "user-data-store", "modified_timestamp"
    ),  # This is an internal property of dcc.Store
    State("user-data-store", "data"),
    prevent_initial_call=True,  # Prevent running on app load
)
def set_uuid(ts, data):
    if ts is None or (data and isinstance(data, dict) and data.get("uuid")):
        raise PreventUpdate
    base = data if isinstance(data, dict) else {}
    if not base.get("uuid"):
        base["uuid"] = str(uuid.uuid4())
        return base
    else:
        raise PreventUpdate


@app.callback(
    [
        Output("day-display", "children", allow_duplicate=True),
        Output("inventory-graph", "figure", allow_duplicate=True),
    ],
    [Input("interval-component", "n_intervals")],
    [State("user-data-store", "data")],
    prevent_initial_call=True,
)
def update_on_interval(n_intervals, client_data):
    if not n_intervals or not client_data:
        raise PreventUpdate

    uuid_val = client_data.get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current_data = get_user_data(uuid_val)

    if not current_data.get("is_initialized", False):
        # If simulation is not initialized, don't proceed
        raise PreventUpdate

    if not current_data or "items" not in current_data or not current_data["items"]:
        # If there are no items, don't proceed
        raise PreventUpdate

    # Update the day count
    day_count = current_data.get("day", 1) + 1
    current_data["day"] = day_count

    # Process each item
    futures = [executor.submit(process_item, item) for item in current_data["items"]]
    current_data["items"] = [future.result() for future in futures]

    # Update the graph based on the processed items
    fig = update_graph_based_on_items(
        current_data["items"], current_data["global_settings"]
    )

    # Save updated data back to user-data-store (this will depend on your implementation of set_user_data)
    set_user_data(uuid_val, current_data)

    # Construct the day display string
    day_display = f"Day: {day_count}"

    return day_display, fig


@app.callback(
    [
        Output("sim-status", "children", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True),
        Output("start-button", "children"),
        Output("start-button", "color"),
    ],
    [Input("start-button", "n_clicks")],
    [State("user-data-store", "data"), State("interval-component", "disabled")],
    prevent_initial_call=True,
)
def toggle_simulation(n_clicks, client_data, is_disabled):
    if not n_clicks or not client_data:
        raise PreventUpdate

    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current_data = get_user_data(uuid_val)

    # Toggle the simulation status based on whether it is currently paused or running.
    if is_disabled:
        # If the simulation is currently disabled/paused, start it.
        current_data["is_initialized"] = True
        set_user_data(uuid_val, current_data)
        save_data()
        return "Status: Running", False, "Pause Simulation", "warning"
    else:
        # If the simulation is running, pause it.
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
    ],
    [Input("reset-button", "n_clicks")],
    [State("user-data-store", "data")],
    prevent_initial_call=True,
)
def reset_simulation(n_clicks, client_data):
    if not n_clicks:
        raise PreventUpdate

    # fresh defaults
    default_data = get_default_data()
    default_data["day"] = 1
    # keep the same session bucket
    uuid_val = (client_data or {}).get("uuid", "__anon__")
    set_user_data(uuid_val, default_data)
    save_data()

    # empty/initial figure
    fig = initial_graph(default_data)

    return (
        f"Day: {default_data['day']}",  # day-display
        fig,  # inventory-graph
        "Status: Paused",  # sim-status
        default_data,  # user-data-store
        True,  # interval disabled (paused)
        "Start Simulation",  # start-button text
        "success",  # start-button color
        False,  # start-button enabled
    )


@app.callback(
    [
        Output("add-item-modal", "is_open"),
        Output("add-item-error", "children"),
        Output("inventory-graph", "figure", allow_duplicate=True),
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
    if ctx.triggered_id is None:
        return is_open, dash.no_update, dash.no_update

    button_id = ctx.triggered_id
    if button_id == "add-item-button":
        # Toggle modal without validating inputs
        return not is_open, dash.no_update, dash.no_update

    if button_id == "submit-item-button":
        # Validate inputs
        all_inputs = [
            usage_rate,
            lead_time,
            item_cost,
            pna,
            safety_allowance,
            standard_pack,
            hits_per_month,
        ]
        if any(i is None for i in all_inputs):
            return (
                is_open,
                dbc.Alert(
                    "All fields must be filled out!", color="warning", duration=4000
                ),
                dash.no_update,
            )

        if any(
            i <= 0
            for i in [
                usage_rate,
                lead_time,
                item_cost,
                safety_allowance,
                standard_pack,
                hits_per_month,
            ]
        ):
            return (
                is_open,
                dbc.Alert(
                    "All item parameters except PNA must be greater than 0!",
                    color="warning",
                    duration=4000,
                ),
                dash.no_update,
            )

        # Add the new item
        uuid_val = (client_data or {}).get("uuid")
        if not uuid_val:
            return (
                is_open,
                dbc.Alert("Session not initialized.", color="danger", duration=4000),
                dash.no_update,
            )
        current_data = get_user_data(uuid_val)
        gs = current_data.get("global_settings", create_global_settings())

        # Build item from modal inputs (safety_allowance is provided as percent in the UI)
        item = create_inventory_item(
            usage_rate=usage_rate,
            lead_time=lead_time,
            item_cost=item_cost,
            pna=pna,
            safety_allowance=safety_allowance / 100.0,
            standard_pack=standard_pack,
            global_settings=gs,
            hits_per_month=max(1.0, hits_per_month),
        )

        current_data.setdefault("items", []).append(item)
        set_user_data(uuid_val, current_data)

        # Update the graph
        fig = update_graph_based_on_items(
            current_data["items"], current_data["global_settings"]
        )

        set_user_data(uuid_val, current_data)
        save_data()

        # Close the modal and clear any error message, then return the updated graph
        return False, None, fig

    raise PreventUpdate


@app.callback(
    [
        Output("update-params-conf", "children"),
        Output("inventory-graph", "figure", allow_duplicate=True),
    ],  # Assuming this is the ID of your graph
    [Input("update-params-button", "n_clicks")],
    [
        State("review-cycle-input", "value"),
        State("r-cost-input", "value"),
        State("k-cost-input", "value"),
        State("user-data-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_parameters(n_clicks, review_cycle, r_cost, k_cost, client_data):
    if not n_clicks:
        raise PreventUpdate

    # Input validation
    if review_cycle is None or r_cost is None or k_cost is None:
        return (
            dbc.Alert(
                "Please fill out all parameters!", color="warning", duration=4000
            ),
            dash.no_update,
        )

    if review_cycle <= 0 or r_cost <= 0 or k_cost <= 0:
        return (
            dbc.Alert(
                "All parameters must be greater than 0!", color="danger", duration=4000
            ),
            dash.no_update,
        )

    # Extract the UUID from client_data and get the current data
    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current_data = get_user_data(uuid_val)
    if "global_settings" not in current_data:
        return (
            dbc.Alert("Global settings not found.", color="danger", duration=4000),
            dash.no_update,
        )

    # Update global settings with new values
    current_data["global_settings"] = update_global_settings(
        current_data["global_settings"],
        review_cycle,
        r_cost,
        k_cost
        / 100,  # Assuming k_cost was provided as a percentage and needs to be converted to a decimal
    )

    # Save the updated global settings back to the user's data store
    set_user_data(uuid_val, current_data)

    # Update all items based on the new global settings
    if "items" in current_data:
        for i, item in enumerate(current_data["items"]):
            current_data["items"][i] = update_gs_related_values(
                item, current_data["global_settings"]
            )

        set_user_data(uuid_val, current_data)
        # Now update the graph based on the updated items
        fig = update_graph_based_on_items(
            current_data["items"], current_data["global_settings"]
        )
    else:
        fig = dash.no_update  # Or set to an empty figure if no items exist

    set_user_data(uuid_val, current_data)
    save_data()

    # Provide user feedback
    return (
        dbc.Alert("Parameters updated successfully!", color="success", duration=4000),
        fig,
    )


@app.callback(
    Output("inventory-graph", "figure", allow_duplicate=True),
    [Input("po-button", "n_clicks")],
    [State("user-data-store", "data")],
    prevent_initial_call=True,
)
def handle_purchase_order(n_clicks, client_data):
    if not n_clicks:
        raise PreventUpdate

    # Extract the UUID from client_data and get the current data
    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        raise PreventUpdate

    current_data = get_user_data(uuid_val)
    if not current_data or "items" not in current_data or not current_data["items"]:
        raise PreventUpdate

    # Increase the PNA of each item by its SOQ and update item details accordingly
    for i, item in enumerate(current_data["items"]):
        item["pna"] += item["soq"]
        current_data["items"][i] = update_pna_related_values(item)

    # Save the updated item data back to the user's data store
    set_user_data(uuid_val, current_data)

    # Now update the graph based on the updated items
    fig = update_graph_based_on_items(
        current_data["items"], current_data["global_settings"]
    )

    set_user_data(uuid_val, current_data)
    save_data()

    return fig


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

    if ctx.triggered_id is None:
        return (
            dash.no_update,
            dash.no_update,
            False,
            dash.no_update,
            is_interval_disabled,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        return (
            dash.no_update,
            dash.no_update,
            False,
            dash.no_update,
            is_interval_disabled,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    current_data = get_user_data(uuid_val)
    items = current_data.get("items", [])
    item_rows = [build_row(index, item) for index, item in enumerate(items)]

    if ctx.triggered_id == "place-custom-order-button":
        # Open modal AND pause sim; set button to "Resume Simulation"
        return (
            dash.no_update,
            item_rows,
            True,
            "Status: Paused",
            True,
            "Resume Simulation",
            "success",
            True,
        )

    elif ctx.triggered_id == "cancel-custom-order-button":
        # Close modal and keep it paused (consistent with your choice)
        return (
            dash.no_update,
            dash.no_update,
            False,
            "Status: Paused",
            True,
            "Resume Simulation",
            "success",
            False,
        )

    elif ctx.triggered_id == "place-order-button":
        if not items:
            return (
                dash.no_update,
                dbc.Alert("No items available for ordering.", color="warning"),
                True,
                "Status: Paused",
                True,
                "Resume Simulation",
                "success",
                True,
            )

        for index, quantity in enumerate(order_quantities):
            if quantity is None or float(quantity) < 0:
                item_rows = update_custom_order_items_div(
                    items, order_quantities, error_index=index
                )
                return (
                    dash.no_update,
                    item_rows,
                    True,
                    "Status: Paused",
                    True,
                    "Resume Simulation",
                    "success",
                    True,
                )

        for index, quantity in enumerate(order_quantities):
            q = float(quantity)
            item = items[index]
            item["pna"] += q
            items[index] = update_pna_related_values(item)

        set_user_data(uuid_val, current_data)
        save_data()
        fig = update_graph_based_on_items(items, current_data["global_settings"])

        # Close the modal and keep paused; set button to "Resume Simulation"
        return (
            fig,
            item_rows,
            False,
            "Status: Paused",
            True,
            "Resume Simulation",
            "success",
            False,
        )

    # If no custom order is placed, do not update anything
    return (
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
    )


def update_custom_order_items_div(items, order_quantities, error_index=None):
    item_rows = []
    for index, item in enumerate(items):
        row = build_row(index, item)
        if index == error_index:
            # Add an error message or styling to the row
            row = html.Div(
                [row, dbc.Alert("Invalid quantity!", color="danger", duration=4000)]
            )
        item_rows.append(row)
    return item_rows


def build_row(index, item):
    # This helper function builds a single row of item data
    return dbc.Row(
        [
            dbc.Col(html.P(str(index + 1)), width=2),
            dbc.Col(html.P(str(round(item["pna"])))),
            dbc.Col(html.P(str(round(item["usage_rate"])))),
            dbc.Col(html.P(str(round(item["lead_time"])))),
            dbc.Col(html.P(str(round(item["op"])))),
            dbc.Col(html.P(str(round(item["lp"])))),
            dbc.Col(html.P(str(round(item["oq"])))),
            dbc.Col(
                dbc.Input(
                    value=round(item["soq"]),
                    id={"type": "order-quantity", "index": index},
                )
            ),
        ]
    )


@app.callback(
    [
        Output("review-cycle-input", "value"),
        Output("r-cost-input", "value"),
        Output("k-cost-input", "value"),
        Output("page-load-2", "data"),
    ],
    [Input("update-params-button", "n_clicks")],
    [State("user-data-store", "data"), State("page-load-2", "data")],
)
def update_input_values_from_data_store(n_clicks, data, page_load):
    data = data or {}
    if page_load == 0:
        uuid_val = data.get("uuid")
        current_data = get_user_data(uuid_val) if uuid_val else get_default_data()
        global_settings = current_data.get("global_settings", {})
        return (
            global_settings.get("r_cycle", 14),
            global_settings.get("r_cost", 8),
            global_settings.get("k_cost", 0.18) * 100,
            1,
        )  # Return defaults if not found
    if not n_clicks:
        raise PreventUpdate
    uuid_val = data.get("uuid")
    current_data = get_user_data(uuid_val) if uuid_val else get_default_data()
    # Extract the global settings
    global_settings = current_data.get("global_settings", {})
    return (
        global_settings.get("r_cycle", 14),
        global_settings.get("r_cost", 8),
        global_settings.get("k_cost", 0.18) * 100,
        dash.no_update,
    )  # Return defaults if not found


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
    [Input("randomize-button", "n_clicks")],
)
def randomize_item_values(n):
    if not n:
        raise PreventUpdate

    rng = _rng()
    usage = int(rng.integers(1, 101))
    lead = 0
    while lead < 7:
        lead = int(abs(rng.normal(30, 30)))
    cost = max(1, int(abs(rng.normal(100, 100))))
    safety_pct = 50 if lead < 60 else max(1, int(round(3000 / lead)))
    pack = int(
        rng.choice([1, 5, 10, 20, 25, 40, 50], p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    )
    pna = int(
        round(
            usage * (lead / 30.0)
            + ((usage * (lead / 30.0)) * (safety_pct / 100.0))
            + pack
        )
    )
    hpm = max(1, int(rng.poisson(5)))

    return usage, lead, cost, pna, safety_pct, pack, hpm


@app.callback(
    [
        Output("day-display", "children"),
        Output("inventory-graph", "figure"),
        Output("page-load", "data"),
    ],  # Assuming 'page-load' is a dcc.Store component
    [Input("page-load", "data")],  # Triggers when 'page-load' data changes
    [State("user-data-store", "data")],
)
def handle_page_load(page_load, client_data):
    if page_load == 0:
        # Assuming get_user_data function fetches the current user's data
        current_data = (
            get_user_data(client_data["uuid"])
            if client_data and "uuid" in client_data
            else {"day": 1}
        )
        day_count = current_data.get("day", 1)
        fig = initial_graph(
            current_data
        )  # Assuming initial_graph is a function that creates the initial graph figure

        # Update 'page-load' to 1 so this initialization doesn't run again
        return f"Day: {day_count}", fig, 1
    else:
        # If 'page-load' is not 0, do not update anything
        raise PreventUpdate


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
    if not list_of_contents:
        raise PreventUpdate

    cards = [
        parse_contents(c, n, d)
        for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
    ]

    frames = []
    errors = []

    for contents, name in zip(list_of_contents, list_of_names):
        try:
            decoded = _extract_base64(contents)  # robust base64 extractor
            lower = (name or "").lower()
            if lower.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))
            elif lower.endswith(".xls"):
                df = pd.read_excel(io.BytesIO(decoded), engine="xlrd")  # ✅ .xls → xlrd
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
            # ✅ always return 3 outputs
            return (
                cards + [dbc.Alert(f"'{name}': {e}", color="warning")],
                None,
                {"display": "none"},
            )

        frames.append(df_num)

    if errors:
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

    combined = pd.concat(frames, ignore_index=True)
    ready = html.Div(f"Ready to import {len(combined)} rows.", className="mb-2")

    # ✅ show the existing button by clearing the style
    return cards + [ready], combined.to_dict("records"), {}


@callback(
    [
        Output("inventory-graph", "figure", allow_duplicate=True),
        Output("upload-feedback", "children"),
    ],
    Input("import-uploaded-items", "n_clicks"),
    State("upload-preview-data", "data"),
    State("user-data-store", "data"),
    prevent_initial_call=True,
)
def import_uploaded_items(n_clicks, rows, client_data):
    if not n_clicks:
        raise PreventUpdate
    if not rows:
        return dash.no_update, dbc.Alert("No data to import.", color="warning")

    uuid_val = (client_data or {}).get("uuid")
    if not uuid_val:
        return dash.no_update, dbc.Alert("Session not initialized.", color="danger")

    current_data = get_user_data(uuid_val)
    gs = current_data.get("global_settings", create_global_settings())

    df = pd.DataFrame(rows)
    try:
        df = coerce_uploaded(df)
    except Exception as e:
        return dash.no_update, dbc.Alert(f"Import error: {e}", color="danger")

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
                hits_per_month=max(1.0, float(rec["hits_per_month"])),
            )
            new_items.append(item)
        except Exception as e:
            return dash.no_update, dbc.Alert(
                f"Row {row_num} import error: {e}", color="danger"
            )

    current_data.setdefault("items", []).extend(new_items)
    set_user_data(uuid_val, current_data)
    fig = update_graph_based_on_items(current_data["items"], gs)
    save_data()

    return fig, dbc.Alert(
        f"Imported {len(new_items)} items successfully.", color="success"
    )


@app.callback(
    Output("interval-component", "interval"),
    Input("sim-speed-slider", "value"),
)
def set_sim_speed(ms):
    if ms is None:
        raise PreventUpdate
    # dcc.Interval expects milliseconds
    return int(ms)


# use curl -X POST http://127.0.0.1:8050/shutdown to save server data
@server.route("/shutdown", methods=["POST"])
def shutdown():
    save_data()
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    return "Server shutting down..."


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
