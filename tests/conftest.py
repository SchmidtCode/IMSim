from __future__ import annotations

from pathlib import Path

import pytest

from imsim.app import create_app
from imsim.config import IMSimConfig


@pytest.fixture
def test_config(tmp_path: Path) -> IMSimConfig:
    repo_root = Path.cwd()
    return IMSimConfig(
        repo_root=repo_root,
        assets_dir=repo_root / "assets",
        session_dir=tmp_path / "sessions",
        examples_dir=repo_root / "examples",
        github_url="https://example.com/imsim",
        admin_token=None,
        allow_dev_shutdown=False,
        shutdown_url=None,
        host="127.0.0.1",
        port=8050,
        debug=False,
    )


@pytest.fixture
def dash_app(test_config: IMSimConfig):
    return create_app(test_config)


@pytest.fixture
def client(dash_app):
    return dash_app.server.test_client()
