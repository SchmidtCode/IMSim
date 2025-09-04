
# IMSim (Inventory Management Simulator)

A lightweight simulator to experiment with inventory policies. It models **PNA** (Projected Net Available) in **days from OP** (Order Point), with live ticks, item uploads, and quick ordering actions. Built with **Dash 3.2.0**.

* Add items manually or import via CSV/Excel
* Start/Pause the simulation; adjust **Simulation Speed (ms)** on the fly
* Place Purchase Orders or custom per-item orders
* Dark UI via Bootstrap (Darkly)

---

## Requirements

* **Python**: 3.10 (project is pinned via `.python-version`; tested on 3.10)
* **uv**: package & environment manager
  Install:

  * macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  * Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`

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
```

App will be at [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

---

## Basic usage

1. **Parameters**: Set Review Cycle (days), R-Cost (\$), K-Cost (%), and **Simulation Speed (ms)**.
2. **Add Items**: Use **Add Item** (manual form) or **Upload Items** (CSV/XLS/XLSX).
3. **Run**: Click **Start/Pause Simulation**. The speed slider updates the tick interval immediately.
4. **Orders**: Use **Place Purchase Order** (adds SOQ) or **Place Custom Order** for per-item quantities.

---

## Upload format

The uploader expects **7 numeric columns** (all values **> 0**) in this order:

1. Usage Rate
2. Lead Time
3. Item Cost
4. Initial PNA
5. Safety Allowance (%)
6. Standard Pack
7. Hits Per Month

---

## Notes for development

* **Dash**: 3.2.0
* **Change deps**: `uv add <package>` (writes to `pyproject.toml` and lockfile), then `uv sync`.
* **Run with a specific Python**: `uv venv --python 3.10 && uv sync && uv run python app.py` (optional; `uv python install` + `uv sync` is usually enough).

---

## License

Apache-2.0 — see `LICENSE`.
