from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_repo_root(config_file: Path) -> Path:
    package_dir = config_file.resolve().parent
    source_root = package_dir.parents[1]
    if (source_root / "pyproject.toml").exists():
        return source_root
    return package_dir.parent


def _resolve_runtime_dir(repo_root: Path, package_dir: Path, name: str) -> Path:
    primary = repo_root / name
    if primary.exists():
        return primary
    packaged = package_dir / name
    if packaged.exists():
        return packaged
    return primary


@dataclass(slots=True)
class IMSimConfig:
    repo_root: Path
    assets_dir: Path
    session_dir: Path
    examples_dir: Path
    github_url: str
    admin_token: str | None
    allow_dev_shutdown: bool
    shutdown_url: str | None
    host: str
    port: int
    debug: bool

    @classmethod
    def from_env(cls) -> IMSimConfig:
        config_file = Path(__file__).resolve()
        package_dir = config_file.parent
        repo_root = _resolve_repo_root(config_file)
        assets_dir = _resolve_runtime_dir(repo_root, package_dir, "assets")
        examples_dir = _resolve_runtime_dir(repo_root, package_dir, "examples")
        session_dir = Path(
            os.environ.get("IMSIM_DATA_DIR") or (repo_root / "var" / "sessions")
        ).expanduser()
        return cls(
            repo_root=repo_root,
            assets_dir=assets_dir,
            session_dir=session_dir,
            examples_dir=examples_dir,
            github_url=os.environ.get("IMSIM_GITHUB_URL", "https://github.com/SchmidtCode/IMSim"),
            admin_token=os.environ.get("IMSIM_ADMIN_TOKEN"),
            allow_dev_shutdown=os.environ.get("ALLOW_DEV_SHUTDOWN") == "1",
            shutdown_url=os.environ.get("SHUTDOWN_URL"),
            host=os.environ.get("IMSIM_HOST", "127.0.0.1"),
            port=int(os.environ.get("IMSIM_PORT", "8050")),
            debug=os.environ.get("IMSIM_DEBUG", "0").lower() in {"1", "true", "yes"},
        )
