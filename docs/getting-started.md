# Getting Started

## Recommended Workflow: Docker Compose

If you just want to run IMSim, you do not need the source repo. The fastest path is a standalone
folder with:

- a `docker-compose.yml`
- a `.env`

The default deployment stack starts:

- `app` from `ghcr.io/schmidtcode/imsim:latest`
- `db` from `postgres:17-alpine` for PostgreSQL-backed session persistence

Prerequisites:

- Docker Desktop or Docker Engine with Compose

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

The app starts on `http://127.0.0.1:8050/`.

Useful commands:

```bash
docker compose ps
docker compose logs -f app
docker compose down
docker compose down -v
```

Notes:

- Compose reads `.env` for variable interpolation and app runtime settings.
- The Compose app service injects its own PostgreSQL connection string for the containerized
  stack.
- Session data is stored in the named `postgres_data` volume unless you remove it with
  `docker compose down -v`.
- For any shared deployment, set `IMSIM_ADMIN_TOKEN` in `.env` before starting the stack.
- `IMSIM_POSTGRES_DB`, `IMSIM_POSTGRES_USER`, and `IMSIM_POSTGRES_PASSWORD` let you override the
  bundled database credentials used by Compose.

## Source Checkout Workflow

Clone the repo only if you want to modify the app, docs, or container setup.

```bash
git clone https://github.com/SchmidtCode/IMSim
cd IMSim
cp .env.example .env
docker compose up -d
```

## Build from Local Source Instead of GHCR

Use this when you are changing Python code, assets, or container config and want the Compose stack
to run your checkout rather than the published image.

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

That override adds the `build:` directive back without changing the default quick-start behavior.

## Run Without Docker for a Fast Edit Loop

If you want the fastest callback or UI iteration loop, run the app directly from the repo with
`uv`.

Prerequisites:

- `uv`

```bash
cp .env.example .env
uv python install 3.14.5
uv sync --group dev
uv run python app.py
```

Alternative entrypoints:

- `uv run imsim`
- `uv run python -m imsim`

If `IMSIM_DATABASE_URL` is unset, local `uv run` uses file-backed sessions under `var/sessions/`.
That is usually the simplest setup for day-to-day UI work.

## Example Data and Uploads

Tracked example inputs live in:

- `examples/sample-items.csv`
- `examples/Example.xlsx`

Uploads accept `.csv` and `.xlsx` files. For the required columns and environment variables, see
[Reference](reference.md).
