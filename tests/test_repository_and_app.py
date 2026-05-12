from __future__ import annotations

import sys
from pathlib import Path

import pytest

import imsim.config as config_module
from imsim.app import create_app
from imsim.callbacks import _triggered_click_count
from imsim.config import IMSimConfig
from imsim.models import SimulationState
from imsim.repository import (
    DatabaseSessionRepository,
    FileSessionRepository,
    SessionConflictError,
    create_session_repository,
)
from imsim.services.training import build_level_state


def test_repository_persists_state(test_config):
    repo = FileSessionRepository(test_config)
    state = SimulationState()
    state.day = 42
    repo.save("abc", state)

    loaded = repo.get_or_create("abc")

    assert loaded.day == 42
    assert repo.path_for("abc").exists()


def test_file_repository_refreshes_from_disk_between_instances(test_config):
    repo_one = FileSessionRepository(test_config)
    repo_two = FileSessionRepository(test_config)

    initial = repo_one.get_or_create("shared-session")
    assert initial.day == 1

    updated = repo_two.get_or_create("shared-session")
    updated.day = 19
    repo_two.save("shared-session", updated)

    refreshed = repo_one.get_or_create("shared-session")

    assert refreshed.day == 19


def test_file_repository_pause_all_scans_disk(test_config):
    repo_one = FileSessionRepository(test_config)
    repo_two = FileSessionRepository(test_config)

    alpha = repo_one.get_or_create("alpha")
    alpha.is_initialized = True
    repo_one.save("alpha", alpha)

    beta = repo_two.get_or_create("beta")
    beta.is_initialized = True
    repo_two.save("beta", beta)

    repo_one.pause_all()

    assert repo_two.get_or_create("alpha").is_initialized is False
    assert repo_two.get_or_create("beta").is_initialized is False


def test_file_repository_rejects_stale_save(test_config):
    repo_one = FileSessionRepository(test_config)
    repo_two = FileSessionRepository(test_config)

    state_one = repo_one.get_or_create("shared")
    state_two = repo_two.get_or_create("shared")

    state_one.day = 9
    repo_one.save("shared", state_one)

    state_two.day = 11
    with pytest.raises(SessionConflictError):
        repo_two.save("shared", state_two)


def test_database_repository_persists_state(tmp_path):
    repo_root = Path.cwd()
    config = IMSimConfig(
        repo_root=repo_root,
        assets_dir=repo_root / "assets",
        session_dir=tmp_path / "sessions",
        examples_dir=repo_root / "examples",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'imsim.db'}",
        github_url="https://example.com/imsim",
        admin_token=None,
        allow_dev_shutdown=False,
        shutdown_url=None,
        host="127.0.0.1",
        port=8050,
        debug=False,
    )
    repo = DatabaseSessionRepository(config)
    state = SimulationState()
    state.day = 17
    repo.save("db-session", state)

    loaded = repo.get_or_create("db-session")

    assert loaded.day == 17


def test_database_repository_rejects_stale_save(tmp_path):
    repo_root = Path.cwd()
    config = IMSimConfig(
        repo_root=repo_root,
        assets_dir=repo_root / "assets",
        session_dir=tmp_path / "sessions",
        examples_dir=repo_root / "examples",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'imsim.db'}",
        github_url="https://example.com/imsim",
        admin_token=None,
        allow_dev_shutdown=False,
        shutdown_url=None,
        host="127.0.0.1",
        port=8050,
        debug=False,
    )
    repo_one = DatabaseSessionRepository(config)
    repo_two = DatabaseSessionRepository(config)

    state_one = repo_one.get_or_create("db-shared")
    state_two = repo_two.get_or_create("db-shared")

    state_one.day = 5
    repo_one.save("db-shared", state_one)

    state_two.day = 8
    with pytest.raises(SessionConflictError):
        repo_two.save("db-shared", state_two)


def test_database_repository_accepts_replacement_state_with_carried_revision(tmp_path):
    repo_root = Path.cwd()
    config = IMSimConfig(
        repo_root=repo_root,
        assets_dir=repo_root / "assets",
        session_dir=tmp_path / "sessions",
        examples_dir=repo_root / "examples",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'imsim.db'}",
        github_url="https://example.com/imsim",
        admin_token=None,
        allow_dev_shutdown=False,
        shutdown_url=None,
        host="127.0.0.1",
        port=8050,
        debug=False,
    )
    repo = DatabaseSessionRepository(config)
    current = repo.get_or_create("db-level")

    next_state = build_level_state("level-1", current.training)
    next_state.revision = current.revision
    repo.save("db-level", next_state)

    loaded = repo.get_or_create("db-level")

    assert loaded.training.current_view == "lesson"
    assert loaded.training.active_level_id == "level-1"


def test_repository_factory_prefers_database(tmp_path):
    repo_root = Path.cwd()
    config = IMSimConfig(
        repo_root=repo_root,
        assets_dir=repo_root / "assets",
        session_dir=tmp_path / "sessions",
        examples_dir=repo_root / "examples",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'imsim.db'}",
        github_url="https://example.com/imsim",
        admin_token=None,
        allow_dev_shutdown=False,
        shutdown_url=None,
        host="127.0.0.1",
        port=8050,
        debug=False,
    )

    assert isinstance(create_session_repository(config), DatabaseSessionRepository)


def test_dash_layout_and_admin_status(client):
    assert client.get("/").status_code == 200
    response = client.get("/_dash-layout")
    assert response.status_code == 200
    payload = response.get_data(as_text=True)
    assert "IMSim Academy" in payload
    assert "academy-simulator-button" in payload
    assert "academy-level-18-button" in payload
    assert '"disabled":true' in payload
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json() == {"status": "ok"}
    status = client.get("/api/admin/shutdown_status")
    assert status.status_code == 200
    assert status.get_json()["active"] is False


def test_pause_session_endpoint_stops_active_session(dash_app, client):
    repository = dash_app.server.extensions["imsim_repository"]
    state = build_level_state("level-1")
    state.is_initialized = True
    repository.save("pause-me", state)

    response = client.post("/api/session/pause", json={"uuid": "pause-me"})

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "paused": True}
    loaded = repository.get_or_create("pause-me")
    assert loaded.is_initialized is False
    assert loaded.training.current_view == "lesson"


def test_dash_component_suites_are_warmed(dash_app):
    registered = dash_app.registered_paths

    assert "dash" in registered
    assert "dash_ag_grid" in registered
    assert "dash_bootstrap_components" in registered


def test_shutdown_endpoint_disabled_by_default(client):
    response = client.post("/shutdown")
    assert response.status_code == 404


def test_triggered_click_count_ignores_recreated_buttons():
    click_map = {
        "academy-level-1-button": 0,
        "academy-reset-progress-button": None,
        "return-to-menu-button": 2,
    }

    assert _triggered_click_count("academy-level-1-button", click_map) == 0
    assert _triggered_click_count("academy-reset-progress-button", click_map) == 0
    assert _triggered_click_count("return-to-menu-button", click_map) == 2
    assert _triggered_click_count(None, click_map) == 0


def test_admin_token_is_enforced(tmp_path):
    repo_root = Path.cwd()
    app = create_app(
        IMSimConfig(
            repo_root=repo_root,
            assets_dir=repo_root / "assets",
            session_dir=tmp_path / "sessions",
            examples_dir=repo_root / "examples",
            database_url=None,
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


def test_config_from_env_uses_checkout_root(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    package_dir = repo_root / "src" / "imsim"
    package_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'imsim'\n", encoding="utf-8")
    (repo_root / "assets").mkdir()
    (repo_root / "examples").mkdir()
    fake_file = package_dir / "config.py"
    fake_file.write_text("# test fixture\n", encoding="utf-8")

    monkeypatch.setattr(config_module, "__file__", str(fake_file))
    monkeypatch.delenv("IMSIM_DATA_DIR", raising=False)
    monkeypatch.delenv("IMSIM_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    config = IMSimConfig.from_env()

    assert config.repo_root == repo_root
    assert config.assets_dir == repo_root / "assets"
    assert config.examples_dir == repo_root / "examples"
    assert config.session_dir == repo_root / "var" / "sessions"
    assert config.database_url is None


def test_config_from_env_uses_installed_distribution_root(monkeypatch, tmp_path):
    site_packages = (
        tmp_path
        / "venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    package_dir = site_packages / "imsim"
    package_dir.mkdir(parents=True)
    (site_packages / "assets").mkdir()
    (site_packages / "examples").mkdir()
    fake_file = package_dir / "config.py"
    fake_file.write_text("# installed package fixture\n", encoding="utf-8")

    monkeypatch.setattr(config_module, "__file__", str(fake_file))
    monkeypatch.delenv("IMSIM_DATA_DIR", raising=False)
    monkeypatch.delenv("IMSIM_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    config = IMSimConfig.from_env()

    assert config.repo_root == site_packages
    assert config.assets_dir == site_packages / "assets"
    assert config.examples_dir == site_packages / "examples"
    assert config.session_dir == site_packages / "var" / "sessions"
    assert config.database_url is None


def test_config_from_env_reads_database_url(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    package_dir = repo_root / "src" / "imsim"
    package_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'imsim'\n", encoding="utf-8")
    (repo_root / "assets").mkdir()
    (repo_root / "examples").mkdir()
    fake_file = package_dir / "config.py"
    fake_file.write_text("# test fixture\n", encoding="utf-8")

    monkeypatch.setattr(config_module, "__file__", str(fake_file))
    monkeypatch.setenv("IMSIM_DATABASE_URL", "sqlite+pysqlite:///imsim.db")

    config = IMSimConfig.from_env()

    assert config.database_url == "sqlite+pysqlite:///imsim.db"


def test_config_from_env_reads_platform_host_and_port_fallbacks(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    package_dir = repo_root / "src" / "imsim"
    package_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'imsim'\n", encoding="utf-8")
    (repo_root / "assets").mkdir()
    (repo_root / "examples").mkdir()
    fake_file = package_dir / "config.py"
    fake_file.write_text("# test fixture\n", encoding="utf-8")

    monkeypatch.setattr(config_module, "__file__", str(fake_file))
    monkeypatch.delenv("IMSIM_HOST", raising=False)
    monkeypatch.delenv("IMSIM_PORT", raising=False)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "10000")

    config = IMSimConfig.from_env()

    assert config.host == "0.0.0.0"
    assert config.port == 10000
