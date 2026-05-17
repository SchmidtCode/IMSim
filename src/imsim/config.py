from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _strip_wrapping_quotes(value.strip())


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
    database_url: str | None
    github_url: str
    admin_token: str | None
    allow_dev_shutdown: bool
    shutdown_url: str | None
    host: str
    port: int
    debug: bool
    cheat_unlock_password: str = "spreadsheets rule"
    max_upload_bytes: int = 5 * 1024 * 1024

    @classmethod
    def from_env(cls) -> IMSimConfig:
        config_file = Path(__file__).resolve()
        package_dir = config_file.parent
        repo_root = _resolve_repo_root(config_file)
        _load_dotenv(repo_root / ".env")
        assets_dir = _resolve_runtime_dir(repo_root, package_dir, "assets")
        examples_dir = _resolve_runtime_dir(repo_root, package_dir, "examples")
        session_dir = Path(
            os.environ.get("IMSIM_DATA_DIR") or (repo_root / "var" / "sessions")
        ).expanduser()
        database_url = os.environ.get("IMSIM_DATABASE_URL") or os.environ.get("DATABASE_URL")
        return cls(
            repo_root=repo_root,
            assets_dir=assets_dir,
            session_dir=session_dir,
            examples_dir=examples_dir,
            database_url=database_url,
            github_url=os.environ.get("IMSIM_GITHUB_URL", "https://github.com/SchmidtCode/IMSim"),
            admin_token=os.environ.get("IMSIM_ADMIN_TOKEN"),
            allow_dev_shutdown=os.environ.get("ALLOW_DEV_SHUTDOWN") == "1",
            shutdown_url=os.environ.get("SHUTDOWN_URL"),
            host=os.environ.get("IMSIM_HOST") or os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("IMSIM_PORT") or os.environ.get("PORT", "8050")),
            debug=os.environ.get("IMSIM_DEBUG", "0").lower() in {"1", "true", "yes"},
            cheat_unlock_password=os.environ.get(
                "IMSIM_CHEAT_UNLOCK_PASSWORD", "spreadsheets rule"
            ),
            max_upload_bytes=max(
                1024,
                int(os.environ.get("IMSIM_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
            ),
        )
