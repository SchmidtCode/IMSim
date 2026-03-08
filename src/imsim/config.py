from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
        repo_root = Path(__file__).resolve().parents[2]
        session_dir = Path(os.environ.get("IMSIM_DATA_DIR") or (repo_root / "var" / "sessions"))
        return cls(
            repo_root=repo_root,
            assets_dir=repo_root / "assets",
            session_dir=session_dir,
            examples_dir=repo_root / "examples",
            github_url=os.environ.get("IMSIM_GITHUB_URL", "https://github.com/SchmidtCode/IMSim"),
            admin_token=os.environ.get("IMSIM_ADMIN_TOKEN"),
            allow_dev_shutdown=os.environ.get("ALLOW_DEV_SHUTDOWN") == "1",
            shutdown_url=os.environ.get("SHUTDOWN_URL"),
            host=os.environ.get("IMSIM_HOST", "127.0.0.1"),
            port=int(os.environ.get("IMSIM_PORT", "8050")),
            debug=os.environ.get("IMSIM_DEBUG", "0").lower() in {"1", "true", "yes"},
        )
