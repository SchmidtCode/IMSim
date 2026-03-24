# IMSim

IMSim is an inventory management training app built with Dash 4. It now starts with an academy-style lesson path that teaches order points, fill rate, PNA, review cycles, and dashboard controls step by step before unlocking the full simulator.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://imsim.pythonanywhere.com/)

## Highlights

- Dash 4 app factory under `src/imsim/` with installable package and WSGI entrypoint
- Training-first UI with academy menu, unlockable lessons, lesson objectives, and a full simulator reward
- Inventory policy engine for OP, LP, EOQ, OQ, SOQ, pack rounding, and auto-PO behavior
- Deterministic early lessons for clean tutorial concepts plus stochastic later lessons for real-world behavior
- Daily demand simulation with stockout, holding, expedite, purchase, revenue, and COGS tracking
- ASQ OP adjuster with min-hit and max-dollar guardrails
- Session persistence through a repository abstraction with PostgreSQL support and file fallback
- CSV and XLSX import flow with header normalization and preview validation
- Docker, CI, tests, and tracked `uv.lock` for reproducible development

## Quickstart

### Local with `uv`

```bash
git clone https://github.com/SchmidtCode/IMSim
cd IMSim
uv python install
uv sync --group dev
uv run imsim
```

The app starts on `http://127.0.0.1:8050/`.

If you want PostgreSQL-backed session storage locally without Docker, set:

```bash
export IMSIM_DATABASE_URL="postgresql+psycopg://imsim:imsim@localhost:5432/imsim"
uv run imsim
```

Legacy entrypoint support is still available:

```bash
uv run python app.py
```

### Docker

Standalone container with file-backed sessions:

```bash
docker build -t imsim .
docker run --rm -p 8050:8050 imsim
```

PostgreSQL-backed stack:

```bash
docker compose up --build
```

## Runtime configuration

| Variable | Purpose |
| --- | --- |
| `IMSIM_DATABASE_URL` | Optional SQLAlchemy database URL for session persistence. If set, the app uses the database instead of filesystem JSON. |
| `IMSIM_DATA_DIR` | Directory for persisted session JSON files. Defaults to `var/sessions/`. |
| `IMSIM_ADMIN_TOKEN` | Optional bearer or `X-IMSIM-ADMIN-TOKEN` value for maintenance endpoints. |
| `ALLOW_DEV_SHUTDOWN` | Enables the `/shutdown` endpoint for local development. |
| `SHUTDOWN_URL` | Retained for compatibility with prior deployments. Internal shutdown no longer self-posts to this URL. |
| `IMSIM_GITHUB_URL` | Footer link override. |
| `IMSIM_HOST` / `IMSIM_PORT` / `IMSIM_DEBUG` | Local app run settings for `python -m imsim`. |

## Upload format

Uploads accept `.csv` and `.xlsx` files with these seven numeric columns:

1. `Usage Rate`
2. `Lead Time`
3. `Item Cost`
4. `Initial PNA`
5. `Safety Allowance (%)`
6. `Standard Pack`
7. `Hits Per Month`

Common header aliases are normalized automatically. The import preview rejects missing columns, duplicate logical columns, non-numeric values, and non-positive inputs other than `Initial PNA`.

## Development

### Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv build
docker run --rm -v "$PWD:/repo" -w /repo ghcr.io/gitleaks/gitleaks:v8.30.1 git . --config .gitleaks.toml --redact --verbose --no-banner
```

CI also runs `gitleaks` on every push and pull request with the repository's full git history.

### Repo layout

```text
src/imsim/        package code, app factory, callbacks, services, and models
assets/           custom CSS and Dash-served frontend assets
examples/         tracked sample input files
tests/            unit and app/API coverage
var/sessions/     runtime session storage (gitignored)
```

### WSGI and deployment

- Package entrypoint: `uv run imsim`
- Module entrypoint: `uv run python -m imsim`
- WSGI target: `imsim.wsgi:server`
- Procfile target: `gunicorn imsim.wsgi:server --workers 4`

## Maintenance API

If `IMSIM_ADMIN_TOKEN` is set, send it as either:

- `Authorization: Bearer <token>`
- `X-IMSIM-ADMIN-TOKEN: <token>`

Endpoints:

- `POST /api/admin/schedule_shutdown`
- `POST /api/admin/cancel_shutdown`
- `GET /api/admin/shutdown_status`

During the final minute, running sessions are paused and the UI is locked. Session data is persisted before shutdown handling.

## Persistence backends

- Default local behavior: file-backed sessions under `var/sessions/` or `IMSIM_DATA_DIR`
- Database behavior: set `IMSIM_DATABASE_URL` to any supported SQLAlchemy URL
- Docker Compose default: PostgreSQL-backed sessions using the bundled `db` service

## Example data

- `examples/sample-items.csv`
- `examples/Example.xlsx`

## License

Apache-2.0. See `LICENSE`.
