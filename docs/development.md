# Development

## Local Development

For the fastest non-container workflow:

```bash
cp deploy/source/.env.example .env
uv python install 3.14.5
uv sync --group dev
uv run imsim
```

If you want development changes to run inside the full Compose stack instead, use:

```bash
docker compose -f deploy/source/docker-compose.yml -f deploy/source/docker-compose.build.yml --project-directory . up -d --build
```

## Common Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv build
docker build .
docker run --rm -v "$PWD:/repo" -w /repo ghcr.io/gitleaks/gitleaks:v8.30.1 git . --config .gitleaks.toml --redact --verbose --no-banner
```

CI also runs `gitleaks` across full git history on every push and pull request.

## Repo Layout

```text
src/imsim/        package code, app factory, callbacks, services, and models
assets/           custom CSS and Dash-served frontend assets
examples/         tracked sample input files
tests/            unit and app/API coverage
var/sessions/     runtime session storage (gitignored)
```

## Entrypoints

- Package entrypoint: `uv run imsim`
- Module entrypoint: `uv run python -m imsim`
- WSGI target: `imsim.wsgi:server`
- Gunicorn config module: `python:imsim.gunicorn_config`

## Performance Tools

The repo includes two performance CLIs:

- `imsim-perf` for repeatable virtual-user load tests
- `imsim-browser-perf` for real Chromium-based browser validation

Examples:

```bash
uv run imsim-perf --scenario simulator --users 4 --tick-rate 6 --ticks 120
uv run imsim-perf --scenario simulator --users 80 --tick-rate 1 --duration 120 --render-profile tick-only
uv run playwright install chromium
uv run imsim-browser-perf --url http://127.0.0.1:8050 --contexts 1 5 10 --duration 300 --json-output tmp/browser-perf.json
```

By default, `imsim-perf` uses a temporary file-session directory so it does not touch real user
sessions. Pass `--session-dir` to keep generated data or `--database-url` to test database-backed
persistence.
