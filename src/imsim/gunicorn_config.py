from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except TypeError, ValueError:
        return default


def _default_workers() -> int:
    return 4 if os.environ.get("IMSIM_DATABASE_URL") or os.environ.get("DATABASE_URL") else 1


def _default_threads() -> int:
    return 2 if _default_workers() > 1 else 1


bind = (
    f"{os.environ.get('IMSIM_HOST') or os.environ.get('HOST', '127.0.0.1')}:"
    f"{os.environ.get('IMSIM_PORT') or os.environ.get('PORT', '8050')}"
)
worker_class = "gthread"
workers = _env_int("IMSIM_GUNICORN_WORKERS", _default_workers())
threads = _env_int("IMSIM_GUNICORN_THREADS", _default_threads())
timeout = _env_int("IMSIM_GUNICORN_TIMEOUT", 120)
graceful_timeout = _env_int("IMSIM_GUNICORN_GRACEFUL_TIMEOUT", 30)
keepalive = _env_int("IMSIM_GUNICORN_KEEPALIVE", 5)
control_socket_disable = True
