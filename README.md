# IMSim

IMSim is an inventory management training app built with Dash 4. It starts with an 18-step academy path that teaches inventory management from first principles: on-hand stock, fill rate, hits, usage, lead time, PNA, order points, safety stock, review cycles, line point, EOQ, SOQ, exceptions, and dashboard controls before unlocking the full simulator.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://imsim.pythonanywhere.com/)

## Highlights

- Dash 4 app factory under `src/imsim/` with installable package and WSGI entrypoint
- Training-first UI with academy menu, 18 unlockable lessons, lesson objectives, and a full simulator reward
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
uv python install 3.14
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

PostgreSQL-backed stack with multi-worker Gunicorn defaults:

```bash
docker compose up --build
```

Compose injects `IMSIM_DATABASE_URL`, so the container uses PostgreSQL-backed sessions and
scales Gunicorn through the shared config in `imsim.gunicorn_config`.

## Runtime configuration

| Variable | Purpose |
| --- | --- |
| `IMSIM_DATABASE_URL` | Optional SQLAlchemy database URL for session persistence. If set, the app uses the database instead of filesystem JSON. |
| `IMSIM_DATA_DIR` | Directory for persisted session JSON files. Defaults to `var/sessions/`. |
| `IMSIM_GUNICORN_WORKERS` | Optional Gunicorn worker count for container/Procfile deploys. Defaults to `4` with a database URL, otherwise `1`. |
| `IMSIM_GUNICORN_THREADS` | Optional Gunicorn thread count. Defaults to `2` with a database URL, otherwise `1`. |
| `IMSIM_GUNICORN_TIMEOUT` | Optional Gunicorn request timeout in seconds. Defaults to `120`. |
| `IMSIM_ADMIN_TOKEN` | Optional bearer or `X-IMSIM-ADMIN-TOKEN` value for maintenance endpoints. |
| `ALLOW_DEV_SHUTDOWN` | Enables the `/shutdown` endpoint for local development. |
| `SHUTDOWN_URL` | Retained for compatibility with prior deployments. Internal shutdown no longer self-posts to this URL. |
| `IMSIM_GITHUB_URL` | Footer link override. |
| `IMSIM_HOST` / `IMSIM_PORT` / `IMSIM_DEBUG` | App run settings for `python -m imsim`. Standard `HOST` / `PORT` env vars are also honored for platform deploys. |

## Free hosting recommendation

For a lightweight public demo, the simplest free setup for IMSim today is:

- Render Free Web Service for the Dash app
- Aiven Free PostgreSQL for session persistence

This split matters because Render's free Postgres offering currently expires 30 days after creation, so it is not a good fit for a reusable demo database.

### Why this works well

- This repo already includes a production `Dockerfile` and Gunicorn entrypoint
- `render.yaml` is included for a one-service Render deploy
- The app now honors platform-provided `PORT`, which removes a common free-hosting issue
- IMSim already accepts either `IMSIM_DATABASE_URL` or `DATABASE_URL`

### Deploy steps

1. Create a free PostgreSQL instance on Aiven.
2. Copy the connection string and append SSL mode if Aiven requires it, for example: ```postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME?sslmode=require```
3. On Render, create a new Blueprint service from this GitHub repo. Render will read `render.yaml`.
4. In the Render dashboard, set `DATABASE_URL` to the Aiven connection string.
5. Deploy and share the generated `onrender.com` URL.

### Free-tier tradeoffs

- Render free web services spin down after 15 minutes idle and take roughly a minute to wake up
- Render free web services use an ephemeral filesystem, so database-backed sessions are the correct setup
- The included `render.yaml` uses `1` Gunicorn worker to stay conservative on free-instance memory

### GitHub-hosted images

The app does not currently depend on a separate image bucket. Frontend assets are served by Dash from `assets/`. If you later want lesson images or marketing screenshots to load from GitHub raw URLs, that can be added as a small config-driven enhancement.

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
- Gunicorn config module: `python:imsim.gunicorn_config`
- Procfile target: `gunicorn --config python:imsim.gunicorn_config imsim.wsgi:server`

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
- Database-backed sessions are the recommended path for higher Gunicorn worker counts

## Container publishing

GitHub Actions now builds and publishes a container image to GHCR on pushes to `main` / `master`
and on version tags. The compose file accepts `IMSIM_IMAGE` if you want to deploy a specific tag:

```bash
export IMSIM_IMAGE=ghcr.io/schmidtcode/imsim:latest
docker compose pull
docker compose up -d
```

## Example data

- `examples/sample-items.csv`
- `examples/Example.xlsx`

## License

Apache-2.0. See `LICENSE`.
