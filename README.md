# IMSim

IMSim is a Dash 4 inventory management training app that teaches replenishment concepts step by
step, then unlocks a full simulator for practicing inventory decisions with live feedback.

[![CI](https://github.com/SchmidtCode/IMSim/actions/workflows/ci.yml/badge.svg)](https://github.com/SchmidtCode/IMSim/actions/workflows/ci.yml)
[![GHCR](https://img.shields.io/badge/container-ghcr-blue)](https://github.com/SchmidtCode/IMSim/pkgs/container/imsim)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://imsim.pythonanywhere.com/)

## Highlights

- 18-lesson academy path that unlocks the full simulator
- Inventory policy engine for OP, LP, EOQ, OQ, SOQ, pack rounding, and auto-PO behavior
- Daily demand simulation with stockout, holding, expedite, revenue, and COGS tracking
- PostgreSQL-backed or file-backed session persistence behind a shared repository layer
- CSV and XLSX import flow with header normalization and preview validation
- Reproducible local setup with Docker, `uv`, CI, and a tracked `uv.lock`

## Quick Start

You do not need to clone this repo to run IMSim.

Create a new folder, drop in a `docker-compose.yml` and `.env`, and let Docker pull the published
container from GitHub Container Registry.

```bash
mkdir imsim
cd imsim
curl -O https://raw.githubusercontent.com/SchmidtCode/IMSim/main/deploy/docker-compose.yml
curl -O https://raw.githubusercontent.com/SchmidtCode/IMSim/main/deploy/.env.example
cp .env.example .env
docker compose up -d
```

PowerShell:

```powershell
New-Item -ItemType Directory imsim
Set-Location imsim
Invoke-WebRequest https://raw.githubusercontent.com/SchmidtCode/IMSim/main/deploy/docker-compose.yml -OutFile docker-compose.yml
Invoke-WebRequest https://raw.githubusercontent.com/SchmidtCode/IMSim/main/deploy/.env.example -OutFile .env.example
Copy-Item .env.example .env
docker compose up -d
```

Open `http://127.0.0.1:8050/`.

Useful commands:

```bash
docker compose logs -f app
docker compose down
docker compose down -v
```

The container image is:

```text
ghcr.io/schmidtcode/imsim:latest
```

If you want the Compose stack to build from a checked-out source tree instead of pulling GHCR:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

For a shared homelab deployment, set `IMSIM_ADMIN_TOKEN` in `.env` before you expose the app.
If you want the `uv run` workflow or source checkout instructions, jump to the docs below.

## Docs

- [Getting Started](docs/getting-started.md)
- [Deployment](docs/deployment.md)
- [Development](docs/development.md)
- [Reference](docs/reference.md)

## Example Data

- `examples/sample-items.csv`
- `examples/Example.xlsx`

## License

Apache-2.0. See `LICENSE`.
