# Reference

## Runtime Configuration

| Variable | Purpose |
| --- | --- |
| `IMSIM_DATABASE_URL` / `DATABASE_URL` | Optional SQLAlchemy database URL for session persistence. If set, the app uses the database instead of filesystem JSON. |
| `IMSIM_DATA_DIR` | Directory for persisted session JSON files. Defaults to `var/sessions/`. |
| `IMSIM_POSTGRES_DB` / `IMSIM_POSTGRES_USER` / `IMSIM_POSTGRES_PASSWORD` | Compose-only variables used by the bundled PostgreSQL service and the default containerized app connection string. |
| `IMSIM_GUNICORN_WORKERS` | Optional Gunicorn worker count for container and Procfile deploys. Defaults to `4` with a database URL, otherwise `1`. |
| `IMSIM_GUNICORN_THREADS` | Optional Gunicorn thread count. Defaults to `2` with a database URL, otherwise `1`. |
| `IMSIM_GUNICORN_TIMEOUT` | Optional Gunicorn request timeout in seconds. Defaults to `120`. |
| `IMSIM_ADMIN_TOKEN` | Optional bearer or `X-IMSIM-ADMIN-TOKEN` value for maintenance endpoints. |
| `IMSIM_CHEAT_UNLOCK_PASSWORD` | Academy unlock phrase. Defaults to `spreadsheets rule` if unset. |
| `ALLOW_DEV_SHUTDOWN` | Enables the `/shutdown` endpoint for local development. |
| `SHUTDOWN_URL` | Retained for compatibility with prior deployments. Internal shutdown no longer self-posts to this URL. |
| `IMSIM_GITHUB_URL` | Footer link override. |
| `IMSIM_HOST` / `IMSIM_PORT` / `IMSIM_DEBUG` | App run settings for `python -m imsim`. Standard `HOST` / `PORT` env vars are also honored for platform deploys. |

Notes:

- The app reads the repo `.env` for local `uv run imsim` and `python app.py` workflows.
- For local `uv run` development, leave `IMSIM_DATABASE_URL` unset unless you intentionally want
  to connect to an external PostgreSQL instance.
- The maintenance schedule and cancel endpoints are disabled unless `IMSIM_ADMIN_TOKEN` is set or
  `ALLOW_DEV_SHUTDOWN=1` is enabled for local development.

## Persistence Backends

- Default local behavior: file-backed sessions under `var/sessions/` or `IMSIM_DATA_DIR`
- Database behavior: set `IMSIM_DATABASE_URL` or `DATABASE_URL` to any supported SQLAlchemy URL
- Docker Compose default: PostgreSQL-backed sessions using the bundled `db` service
- Database-backed sessions are the recommended path for higher Gunicorn worker counts

## Upload Format

Uploads accept `.csv` and `.xlsx` files with these seven numeric columns:

1. `Usage Rate`
2. `Lead Time`
3. `Item Cost`
4. `Initial PNA`
5. `Safety Allowance (%)`
6. `Standard Pack`
7. `Hits Per Month`

Common header aliases are normalized automatically. The import preview rejects missing columns,
duplicate logical columns, non-numeric values, and non-positive inputs other than `Initial PNA`.

## Maintenance API

If `IMSIM_ADMIN_TOKEN` is set, send it as either:

- `Authorization: Bearer <token>`
- `X-IMSIM-ADMIN-TOKEN: <token>`

Endpoints:

- `POST /api/admin/schedule_shutdown`
- `POST /api/admin/cancel_shutdown`
- `GET /api/admin/shutdown_status`

During the final minute, running sessions are paused and the UI is locked. Session data is
persisted before shutdown handling.
