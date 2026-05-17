FROM python:3.14.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY assets ./assets
COPY examples ./examples

RUN uv sync --frozen --no-dev
RUN groupadd --system imsim && useradd --system --gid imsim --home-dir /app imsim && chown -R imsim:imsim /app

ENV PATH="/app/.venv/bin:$PATH"
ENV IMSIM_HOST=0.0.0.0
ENV IMSIM_PORT=8050

EXPOSE 8050

USER imsim

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/health', timeout=3)"

CMD ["gunicorn", "--config", "python:imsim.gunicorn_config", "imsim.wsgi:server"]
