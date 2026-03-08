from __future__ import annotations

import os
import signal
import time

import dash
import dash_bootstrap_components as dbc
from flask import Flask, abort, jsonify, request

from .callbacks import register_callbacks
from .config import IMSimConfig
from .repository import FileSessionRepository
from .services.simulation import MaintenanceController
from .ui.layout import build_layout


def create_app(config: IMSimConfig | None = None) -> dash.Dash:
    config = config or IMSimConfig.from_env()
    server = Flask(__name__)
    repository = FileSessionRepository(config)
    maintenance = MaintenanceController(repository, config.allow_dev_shutdown)
    maintenance.start_watchdog_once()

    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=str(config.assets_dir),
        suppress_callback_exceptions=False,
    )
    app.title = "IMSim"
    app.layout = build_layout(config)

    server.extensions["imsim_config"] = config
    server.extensions["imsim_repository"] = repository
    server.extensions["imsim_maintenance"] = maintenance

    def _authorized(req) -> bool:
        if not config.admin_token:
            return True
        bearer = req.headers.get("Authorization") or ""
        if bearer.startswith("Bearer "):
            bearer = bearer.split(" ", 1)[1]
        header_token = req.headers.get("X-IMSIM-ADMIN-TOKEN") or bearer
        return header_token == config.admin_token

    @server.post("/api/admin/schedule_shutdown")
    def api_schedule_shutdown():
        if not _authorized(request):
            return abort(401)
        data = request.get_json(silent=True) or {}
        minutes = float(data.get("minutes", 0))
        message = str(data.get("message", "Maintenance"))
        maintenance.schedule_shutdown_in(minutes, message)
        return jsonify({"ok": True, "active": True, "minutes": minutes, "message": message})

    @server.post("/api/admin/cancel_shutdown")
    def api_cancel_shutdown():
        if not _authorized(request):
            return abort(401)
        maintenance.cancel_shutdown()
        return jsonify({"ok": True, "active": False})

    @server.get("/api/admin/shutdown_status")
    def api_shutdown_status():
        state = maintenance.snapshot()
        remaining = max(0.0, state.at - time.time()) if state.active else None
        return jsonify(
            {
                "active": state.active,
                "message": state.message,
                "seconds_remaining": None if remaining is None else int(remaining),
                "closing": state.closing,
            }
        )

    @server.post("/shutdown")
    def shutdown():
        if not config.allow_dev_shutdown:
            return abort(404)
        maintenance.gentle_stop_all_sessions()
        func = request.environ.get("werkzeug.server.shutdown")
        if func is not None:
            func()
            return "Server shutting down..."
        os.kill(os.getpid(), signal.SIGTERM)
        return "Server shutting down..."

    register_callbacks(app, repository, maintenance)
    return app
