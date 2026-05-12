FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY assets ./assets
COPY examples ./examples
COPY app.py Procfile ./

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV IMSIM_HOST=0.0.0.0
ENV IMSIM_PORT=8050

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/health', timeout=3)"

CMD ["gunicorn", "--config", "python:imsim.gunicorn_config", "imsim.wsgi:server"]
