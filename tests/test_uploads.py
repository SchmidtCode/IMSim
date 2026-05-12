from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest
from dash_ag_grid import AgGrid

from imsim.services.uploads import coerce_uploaded, parse_contents, read_uploaded_table


def _encode_bytes(payload: bytes, mime: str = "text/csv") -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def test_coerce_uploaded_normalizes_headers():
    frame = pd.DataFrame(
        {
            "Usage Rate": [10],
            "Lead Time (days)": [30],
            "Item Cost": [12],
            "Initial PNA": [3],
            "Safety Allowance (%)": [25],
            "Standard Pack": [5],
            "Hits Per Month": [4],
        }
    )
    normalized = coerce_uploaded(frame)
    assert list(normalized.columns) == [
        "usage_rate",
        "lead_time_days",
        "item_cost",
        "initial_pna",
        "safety_allowance_pct",
        "standard_pack",
        "hits_per_month",
    ]


def test_coerce_uploaded_rejects_duplicate_aliases():
    frame = pd.DataFrame(
        {
            "Usage Rate": [10],
            "usage_rate": [12],
            "Lead Time": [30],
            "Item Cost": [5],
            "Initial PNA": [1],
            "Safety Allowance (%)": [20],
            "Standard Pack": [1],
            "Hits Per Month": [3],
        }
    )
    with pytest.raises(ValueError, match="Duplicate logical columns"):
        coerce_uploaded(frame)


def test_read_uploaded_table_supports_csv_and_xlsx(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "Usage Rate": [10],
            "Lead Time": [30],
            "Item Cost": [5],
            "Initial PNA": [1],
            "Safety Allowance (%)": [20],
            "Standard Pack": [1],
            "Hits Per Month": [3],
        }
    )
    csv_payload = _encode_bytes(frame.to_csv(index=False).encode("utf-8"))
    assert not read_uploaded_table(csv_payload, "sample.csv").empty

    xlsx_path = tmp_path / "sample.xlsx"
    frame.to_excel(xlsx_path, index=False)
    xlsx_payload = _encode_bytes(
        xlsx_path.read_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert not read_uploaded_table(xlsx_payload, "sample.xlsx").empty


def test_read_uploaded_table_rejects_xls():
    with pytest.raises(ValueError, match="csv or .xlsx"):
        read_uploaded_table(_encode_bytes(b"bad"), "legacy.xls")


def test_parse_contents_returns_preview_card():
    frame = pd.DataFrame(
        {
            "Usage Rate": [10],
            "Lead Time": [30],
            "Item Cost": [5],
            "Initial PNA": [1],
            "Safety Allowance (%)": [20],
            "Standard Pack": [1],
            "Hits Per Month": [3],
        }
    )
    payload = _encode_bytes(frame.to_csv(index=False).encode("utf-8"))
    preview = parse_contents(payload, "sample.csv", 0)
    assert getattr(preview, "className", "") == "preview-card"
    grid = preview.children.children[3]
    assert isinstance(grid, AgGrid)
    assert grid.rowData[0]["usage_rate"] == 10
    assert grid.dashGridOptions["paginationPageSize"] == 10
