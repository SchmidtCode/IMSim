from __future__ import annotations

from imsim.models import SimulationState
from imsim.repository import FileSessionRepository


def test_repository_persists_state(test_config):
    repo = FileSessionRepository(test_config)
    state = SimulationState()
    state.day = 42
    repo.save("abc", state)

    loaded = repo.get_or_create("abc")

    assert loaded.day == 42
    assert repo.path_for("abc").exists()


def test_dash_layout_and_admin_status(client):
    assert client.get("/").status_code == 200
    response = client.get("/_dash-layout")
    assert response.status_code == 200
    status = client.get("/api/admin/shutdown_status")
    assert status.status_code == 200
    assert status.get_json()["active"] is False


def test_shutdown_endpoint_disabled_by_default(client):
    response = client.post("/shutdown")
    assert response.status_code == 404


def test_admin_token_is_enforced(tmp_path):
    from pathlib import Path

    from imsim.app import create_app
    from imsim.config import IMSimConfig

    repo_root = Path.cwd()
    app = create_app(
        IMSimConfig(
            repo_root=repo_root,
            assets_dir=repo_root / "assets",
            session_dir=tmp_path / "sessions",
            examples_dir=repo_root / "examples",
            github_url="https://example.com/imsim",
            admin_token="secret",
            allow_dev_shutdown=False,
            shutdown_url=None,
            host="127.0.0.1",
            port=8050,
            debug=False,
        )
    )
    client = app.server.test_client()
    assert client.post("/api/admin/schedule_shutdown", json={"minutes": 5}).status_code == 401
    ok = client.post(
        "/api/admin/schedule_shutdown",
        json={"minutes": 5, "message": "Patch window"},
        headers={"X-IMSIM-ADMIN-TOKEN": "secret"},
    )
    assert ok.status_code == 200
    payload = ok.get_json()
    assert payload["active"] is True
