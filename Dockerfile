FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

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

CMD ["gunicorn", "imsim.wsgi:server", "--bind", "0.0.0.0:8050", "--workers", "2"]
