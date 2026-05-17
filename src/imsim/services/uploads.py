from __future__ import annotations

import base64
import datetime as dt
import io
from typing import Any

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import html

DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024

CANONICAL_COLS = [
    "usage_rate",
    "lead_time_days",
    "item_cost",
    "initial_pna",
    "safety_allowance_pct",
    "standard_pack",
    "hits_per_month",
]

HEADER_ALIASES = {
    "usage rate": "usage_rate",
    "usage_rate": "usage_rate",
    "lead time": "lead_time_days",
    "lead time (days)": "lead_time_days",
    "lead_time": "lead_time_days",
    "lead_time_days": "lead_time_days",
    "item cost": "item_cost",
    "item_cost": "item_cost",
    "initial pna": "initial_pna",
    "initial_pna": "initial_pna",
    "pna": "initial_pna",
    "safety allowance (%)": "safety_allowance_pct",
    "safety allowance": "safety_allowance_pct",
    "safety_allowance_pct": "safety_allowance_pct",
    "standard pack": "standard_pack",
    "standard_pack": "standard_pack",
    "hits per month": "hits_per_month",
    "hits_per_month": "hits_per_month",
}


def extract_base64(contents: str, *, max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> bytes:
    if not isinstance(contents, str):
        raise ValueError("Invalid upload payload type.")
    _, sep, encoded = contents.partition(",")
    encoded = encoded if sep else contents
    try:
        decoded = base64.b64decode(encoded, validate=False)
    except Exception as exc:
        raise ValueError(f"Could not decode uploaded file: {exc}") from exc
    if len(decoded) > max_bytes:
        raise ValueError(
            f"Uploaded file is too large. Maximum allowed size is {max_bytes // (1024 * 1024)} MB."
        )
    return decoded


def coerce_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = [str(col) for col in df.columns]
    lower_cols = [col.strip().lower() for col in raw_cols]
    mapped: list[str] = []
    unknown: list[str] = []
    sources_by_canon: dict[str, list[str]] = {}

    for raw, lower in zip(raw_cols, lower_cols, strict=False):
        if lower in HEADER_ALIASES:
            canon = HEADER_ALIASES[lower]
            mapped.append(canon)
            sources_by_canon.setdefault(canon, []).append(raw)
        else:
            unknown.append(raw)

    if unknown:
        expected = ", ".join(sorted(HEADER_ALIASES.keys()))
        raise ValueError(f"Unrecognized headers: {unknown}. Expected one of: {expected}")

    duplicates = {key: value for key, value in sources_by_canon.items() if len(value) > 1}
    if duplicates:
        pretty = "; ".join(f"{key} <- {value}" for key, value in duplicates.items())
        raise ValueError(f"Duplicate logical columns after header normalization: {pretty}")

    missing = [column for column in CANONICAL_COLS if column not in mapped]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    normalized = df.copy()
    normalized.columns = mapped
    normalized = normalized[CANONICAL_COLS].copy()
    for column in CANONICAL_COLS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if normalized.isnull().values.any():
        raise ValueError("All required columns must be numeric and non-empty.")

    must_be_positive = [
        "usage_rate",
        "lead_time_days",
        "item_cost",
        "safety_allowance_pct",
        "standard_pack",
        "hits_per_month",
    ]
    if (normalized[must_be_positive] <= 0).any().any():
        raise ValueError("All values (except PNA) must be > 0.")

    return normalized


def read_uploaded_table(
    contents: str,
    filename: str | None,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> pd.DataFrame:
    decoded = extract_base64(contents, max_bytes=max_bytes)
    lower = (filename or "").lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))
    if lower.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(decoded), engine="openpyxl")
    raise ValueError("The input file must be .csv or .xlsx.")


def _grid_theme_class(theme: str) -> str:
    return (
        "ag-theme-quartz-dark imsim-ag-grid" if theme == "dark" else "ag-theme-quartz imsim-ag-grid"
    )


def parse_contents(
    contents: str,
    filename: str | None,
    modified_at: float | None,
    theme: str = "light",
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
):
    try:
        raw = read_uploaded_table(contents, filename, max_bytes=max_bytes)
        df = coerce_uploaded(raw)
    except Exception as exc:
        return dbc.Alert(str(exc), color="warning")

    preview_records: list[dict[str, Any]] = []
    for record in df.to_dict("records")[:250]:
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, np.generic):
                cleaned[str(key)] = value.item()
            else:
                cleaned[str(key)] = value
        preview_records.append(cleaned)

    try:
        ts = float(modified_at or 0)
        if ts > 1e11:
            ts /= 1000.0
        when_str = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        when_str = ""

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(filename or "Uploaded file", className="card-title"),
                html.H6(when_str, className="card-subtitle text-muted"),
                html.Br(),
                dag.AgGrid(
                    rowData=preview_records,
                    columnDefs=[{"field": column, "headerName": column} for column in df.columns],
                    defaultColDef={"sortable": True, "filter": True, "resizable": True},
                    className=_grid_theme_class(theme),
                    columnSize="sizeToFit",
                    dashGridOptions={
                        "pagination": True,
                        "paginationPageSize": 10,
                        "paginationPageSizeSelector": False,
                        "animateRows": False,
                    },
                    style={"height": "320px", "width": "100%"},
                ),
            ]
        ),
        className="preview-card",
    )
