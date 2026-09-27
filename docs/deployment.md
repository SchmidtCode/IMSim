# Deployment

## Published Container

GitHub Actions publishes IMSim to GitHub Container Registry from `.github/workflows/container.yml`.

Package page:

- [SchmidtCode/IMSim container package](https://github.com/SchmidtCode/IMSim/pkgs/container/imsim)

Published tags include:

- `latest` on the default branch
- `sha-<commit>`
- branch names for `main` and `master`
- version tags such as `v1.1.0`

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

### PostgreSQL 18 upgrade

The v1.1.0 Compose files use `postgres:18-alpine` and mount `postgres_data18` at
`/var/lib/postgresql`, the data path used by the PostgreSQL 18 image. A fresh deployment creates
this volume automatically.

Existing PostgreSQL 17 data does not upgrade when the image tag changes. Before replacing an
existing stack, make and verify a database backup, restore it into a new PostgreSQL 18 volume,
then confirm the app can read the restored data. Keep the PostgreSQL 17 volume until the new
deployment is verified. Do not attach a PostgreSQL 17 data volume directly to the PostgreSQL 18
container. See the [PostgreSQL upgrade documentation](https://www.postgresql.org/docs/current/upgrading.html)
and [official container notes](https://github.com/docker-library/docs/blob/master/postgres/README.md#pgdata).

For checked-out source work, use the contributor stack under [`deploy/source/`](../deploy/source).

## Homelab Checklist

Before exposing IMSim outside your LAN or sharing it with conference attendees:

- Set `IMSIM_ADMIN_TOKEN` in `.env` so the maintenance API is protected.
- Change the default `IMSIM_POSTGRES_*` values if you do not want the bundled defaults.
- Prefer a pinned `IMSIM_IMAGE` tag over `latest` for demo stability.
- Put the app behind a reverse proxy that terminates TLS.
- Back up the `postgres_data18` volume if you care about persistent session state.

## Container Publishing

The container workflow runs on:

- pushes to `main`
- pushes to `master`
- tags matching `v*`
- manual `workflow_dispatch`

That makes the default README quick start useful for both local evaluation and simple server
deployments.
