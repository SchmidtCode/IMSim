# Deployment

## Published Container

GitHub Actions publishes IMSim to GitHub Container Registry from `.github/workflows/container.yml`.

Package page:

- [SchmidtCode/IMSim container package](https://github.com/SchmidtCode/IMSim/pkgs/container/imsim)

Published tags include:

- `latest` on the default branch
- `sha-<commit>`
- branch names for `main` and `master`
- version tags such as `v0.2.0`

## Deploy with Docker Compose

For a standalone deployment, use the two files in [`deploy/`](../deploy):

- [`deploy/docker-compose.yml`](../deploy/docker-compose.yml)
- [`deploy/.env.example`](../deploy/.env.example)

That stack points at the published image:

- `ghcr.io/schmidtcode/imsim:latest`

Quick start without cloning:

```bash
mkdir imsim
cd imsim
curl -O https://raw.githubusercontent.com/SchmidtCode/IMSim/main/deploy/docker-compose.yml
curl -O https://raw.githubusercontent.com/SchmidtCode/IMSim/main/deploy/.env.example
cp .env.example .env
docker compose up -d
```

To refresh to the newest published image:

```bash
docker compose pull
docker compose up -d
```

To pin a specific published build, set `IMSIM_IMAGE` before running Compose.

Bash:

```bash
export IMSIM_IMAGE=ghcr.io/schmidtcode/imsim:sha-0e68a59
docker compose pull
docker compose up -d
```

PowerShell:

```powershell
$env:IMSIM_IMAGE = "ghcr.io/schmidtcode/imsim:sha-0e68a59"
docker compose pull
docker compose up -d
```

The Compose stack still provides the bundled PostgreSQL service and injects
`IMSIM_DATABASE_URL` for the containerized app.

The repo-root [docker-compose.yml](../docker-compose.yml) is kept for source checkouts and local
development workflows.

## Homelab Checklist

Before exposing IMSim outside your LAN or sharing it with conference attendees:

- Set `IMSIM_ADMIN_TOKEN` in `.env` so the maintenance API is protected.
- Change the default `IMSIM_POSTGRES_*` values if you do not want the bundled defaults.
- Prefer a pinned `IMSIM_IMAGE` tag over `latest` for demo stability.
- Put the app behind a reverse proxy that terminates TLS.
- Back up the `postgres_data` volume if you care about persistent session state.

## Render Free Demo Setup

For a lightweight public demo, the lowest-friction free setup in this repo is:

- Render Free Web Service for the Dash app
- Aiven Free PostgreSQL for session persistence

This split matters because Render's free Postgres offering expires 30 days after creation, which
makes it a poor fit for a stable demo database.

Why this works well:

- The repo already includes a production `Dockerfile`
- [render.yaml](../render.yaml) is ready for a one-service Render deploy
- The app honors platform-provided `PORT`
- IMSim accepts either `IMSIM_DATABASE_URL` or `DATABASE_URL`

Deploy steps:

1. Create a free PostgreSQL instance on Aiven.
2. Copy the connection string and append SSL mode if required, for example `postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME?sslmode=require`.
3. In Render, create a new Blueprint service from this GitHub repo.
4. Set `DATABASE_URL` in Render to the Aiven connection string.
5. Deploy and share the generated `onrender.com` URL.

Free-tier tradeoffs:

- Render free web services spin down after 15 minutes of idle time
- Wake-up time is usually around a minute
- The filesystem is ephemeral, so database-backed sessions are the right setup
- `render.yaml` uses one Gunicorn worker to stay conservative on free-instance memory

## Container Publishing

The container workflow runs on:

- pushes to `main`
- pushes to `master`
- tags matching `v*`
- manual `workflow_dispatch`

That makes the default README quick start useful for both local evaluation and simple server
deployments.
