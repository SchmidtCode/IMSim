# IMSim (Inventory Management Simulator)

A lightweight simulator to experiment with inventory policies. It models **PNA** (Projected Net Available) in **days from OP** (Order Point), with live ticks, item uploads, and quick ordering actions. Built with **Dash 3.2.0**.

**Highlights**
- Live simulation with adjustable tick speed
- **ASQ OP Adjuster** (raise OP to observed ASQ with guardrails)
- **Auto-POs** (place SOQs automatically when PNA ≤ LP)
- PO management: **overview, expedite, cancel**
- Costing & P&L: ordering, holding, stockout, expedite, **COGS & revenue/GM**
- Pack-aware **sales engine** (no rounding inflation; distributors can break packs)
- File upload (CSV/Excel) with header normalization + validation preview
- KPI strip and visualization overlays (PNA, PNA+SOQ, 0 PNA, On-Hand)
- Simple persistence to `data/user_data.json`
- Admin maintenance API (+ graceful shutdown banner & watchdog)
- Dark UI via Bootstrap (Darkly)

---

## Requirements

- **Python**: 3.10 (project pinned via `.python-version`; tested on 3.10)
- **uv**: package & environment manager  
  Install:
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`

---

## Quickstart (uv)

```bash
git clone https://github.com/SchmidtCode/IMSim
cd IMSim

# Ensure the pinned Python from .python-version is available
uv python install

# Install project dependencies from pyproject.toml / uv.lock
uv sync

# Run (no manual venv activation needed)
uv run python app.py
````

App will be at [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

---

## Basic usage

1. **Parameters**: Set Review Cycle (days), R-Cost (\$), K-Cost (%), Stockout Penalty (\$/unit), Expedite Rate (%/day), Global GM (%).
   Optionally enable **Auto Purchase Orders**.
2. **Add Items**: Use **Add Item** (manual form) or **Upload Items** (CSV/XLS/XLSX).
3. **Run**: Click **Start/Pause Simulation**. The **Simulation Speed (ms)** slider updates the tick interval immediately.
4. **Orders**: Use **Place Purchase Order** (SOQs for all items) or **Place Custom Order** for per-item quantities.
5. **PO Overview**: Inspect open receipts; **expedite by 1 day** (adds expedite cost) or **cancel** a receipt.

---

## Upload format

The uploader expects **7 numeric columns** (all values **> 0**, except PNA which may be 0) in this order:

1. `Usage Rate`
2. `Lead Time`
3. `Item Cost`
4. `Initial PNA`
5. `Safety Allowance (%)`
6. `Standard Pack`
7. `Hits Per Month`

**Header aliases** are accepted (e.g., `usage rate`, `lead time (days)`, `safety_allowance_pct`, etc.). The preview validates and shows any issues before import.

---

## Simulation details

* **Hits per day**: Poisson with λ = `hits_per_month / 30`.
* **Sales per hit**: Poisson **in pack-space** (λ = `avg_sale_qty / standard_pack`) and then re-scaled to units to avoid demand inflation from rounding. This matches real behavior where distributors often break packs.
* **PNA vs On-Hand**:
  PNA = On-Hand + On-Order − Backorder. The graph shows:

  * **Current PNA (days from OP)**
  * **PNA + SOQ (days from OP)** for items at/under LP
  * **0 PNA (days from OP)**
  * **On-Hand (days from OP)** — orange “x” marker for “close to stockout” read
* **Price & GM**: Global GM% sets price via `price_per_unit = item_cost / (1 - GM)` and applies a `realization` factor for realized revenue. Sales/COGS update only when units ship.

---

## Planning & ordering

* **OP** (order point) = expected demand over lead time + safety stock
* **LP** (line point) = OP + demand over review cycle
* **EOQ/OQ**: EOQ via classic square-root formula; OQ bounded by RC demand and \~12 months demand
* **SOQ policy**:

  * If PNA > LP ⟶ 0
  * If OP < PNA ≤ LP ⟶ OQ
  * If PNA ≤ OP ⟶ OQ + (OP − PNA)
    SOQs are **rounded up to pack**.
* **Auto-POs**: When enabled, the system places SOQs each tick for items at/under LP.
* **Custom Orders**: Enter any quantity per item; quantities are rounded to the item’s **Standard Pack**.

---

## ASQ OP Adjuster

**Goal:** When observed **Average Sale Quantity (ASQ = usage ÷ line hits)** persistently exceeds the raw OP, raise OP to ASQ with guardrails.

* **Tracked each period** (rolling “month” = configurable **Period (Days)**)
* **Pre-conditions**:

  * Adjuster is enabled
  * **Min Line Hits** met
  * **Max \$ Diff** not exceeded (`(ASQ − rawOP) * item_cost`)
* **Action**: If `ASQ > raw OP`, set OP ← ASQ (recompute LP/OQ/deriveds); counters reset each period.
* **Manual month-end**: “Apply Month-End (ASQ) Now”.
* **Exception Center**: Skipped adjustments (e.g., Max \$ Diff) are logged with context.

---

## Costs, sales & KPIs

* **Costs**: ordering, holding (daily), stockout, expedite; **purchases** (COGS of receipts) are tracked separately.
* **Sales**: revenue, COGS, units sold.
* **KPI strip** shows Day, Fill Rate (cumulative), Sales, COGS, Inventory Overhead, and GM% (with tooltip for after-overhead GM).

---

## PO overview & actions

* **Overview modal** lists open receipts with ETA and days left.
* **Expedite −1 day** (if ETA ≥ today+2): charges `qty * item_cost * expedite_rate * 1 day`.
* **Cancel**: removes the receipt and refreshes derived planning.

---

## Persistence

State is stored to **`data/user_data.json`**. A per-process watchdog/baseline autosave runs during maintenance shutdowns.

---

## Maintenance API (admin)

Optional admin endpoints to schedule/cancel maintenance. If `IMSIM_ADMIN_TOKEN` is set, it is required either as:

* `Authorization: Bearer <token>` or
* `X-IMSIM-ADMIN-TOKEN: <token>`

**Endpoints**

* `POST /api/admin/schedule_shutdown`
  Body: `{"minutes": <float>, "message": "<string>"}`
* `POST /api/admin/cancel_shutdown`
* `GET  /api/admin/shutdown_status`

During the final minute, sessions are paused and the Start button is disabled; at T≈0 the app saves and triggers a graceful shutdown.

**Dev-only:** `POST /shutdown` is available when `ALLOW_DEV_SHUTDOWN=1`. `SHUTDOWN_URL` can override the default local endpoint.

---

## Notes for development

* **Dash**: 3.2.0
* **Change deps**: `uv add <package>` then `uv sync`
* **Run with a specific Python**:
  `uv venv --python 3.10 && uv sync && uv run python app.py`
* **Env (optional)**:

  * `IMSIM_ADMIN_TOKEN=<token>` (protect maintenance API)
  * `ALLOW_DEV_SHUTDOWN=1` (enable local `/shutdown`)
  * `SHUTDOWN_URL=http://127.0.0.1:8050/shutdown`

---

## License

Apache-2.0 — see `LICENSE`.
