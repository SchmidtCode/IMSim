from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from sqlalchemy import JSON, DateTime, Integer, String, create_engine, inspect, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import IMSimConfig
from .models import SimulationState, default_state


class SessionConflictError(RuntimeError):
    pass


class InvalidSessionIdError(ValueError):
    pass


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_WINDOWS_REPLACE_ATTEMPTS = 5
_WINDOWS_REPLACE_RETRY_DELAY_SECONDS = 0.05


def normalize_session_id(session_id: str) -> str:
    normalized = str(session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(normalized):
        raise InvalidSessionIdError("Session ID must be 1-128 chars of letters, numbers, _ or -.")
    return normalized


class SessionRepository(Protocol):
    def get_or_create(self, session_id: str) -> SimulationState: ...

    def save(self, session_id: str, state: SimulationState) -> None: ...

    def reset(self, session_id: str) -> SimulationState: ...

    def pause_all(self) -> None: ...

    def persist_all(self) -> None: ...


class Base(DeclarativeBase):
    pass


class SessionStateRecord(Base):
    __tablename__ = "session_states"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FileSessionRepository:
    def __init__(self, config: IMSimConfig):
        self._config = config
        self._lock = RLock()
        self._session_dir = Path(config.session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self._session_dir / f"{normalize_session_id(session_id)}.json"

    def get_or_create(self, session_id: str) -> SimulationState:
        session_id = normalize_session_id(session_id)
        with self._lock:
            path = self.path_for(session_id)
            state = self._load_path(path)
            if state is None:
                state = default_state()
                state.revision = 1
                self._write_path(path, state)
            return state.clone()

    def save(self, session_id: str, state: SimulationState) -> None:
        session_id = normalize_session_id(session_id)
        with self._lock:
            path = self.path_for(session_id)
            current = self._load_path(path)
            if current is None:
                if state.revision > 0:
                    raise SessionConflictError(f"Session {session_id} no longer exists.")
                snapshot = state.clone()
                snapshot.revision = 1
                self._write_path(path, snapshot)
                state.revision = snapshot.revision
                return
            if current.revision != state.revision:
                raise SessionConflictError(
                    f"Session {session_id} changed from revision "
                    f"{state.revision} to {current.revision}."
                )
            snapshot = state.clone()
            snapshot.revision = current.revision + 1
            self._write_path(path, snapshot)
            state.revision = snapshot.revision

    def reset(self, session_id: str) -> SimulationState:
        session_id = normalize_session_id(session_id)
        state = default_state()
        self.save(session_id, state)
        return state.clone()

    def pause_all(self) -> None:
        with self._lock:
            for path in self._session_dir.glob("*.json"):
                state = self._load_path(path)
                if state is None:
                    continue
                state.is_initialized = False
                state.revision += 1
                self._write_path(path, state)

    def persist_all(self) -> None:
        return None

    def _load_path(self, path: Path) -> SimulationState | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return SimulationState.from_dict(json.load(handle))

    def _write_path(self, path: Path, state: SimulationState) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(state.to_dict(), handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            self._replace_path(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _replace_path(self, source: Path, target: Path) -> None:
        for attempt in range(_WINDOWS_REPLACE_ATTEMPTS):
            try:
                os.replace(source, target)
                return
            except PermissionError:
                if attempt == _WINDOWS_REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_WINDOWS_REPLACE_RETRY_DELAY_SECONDS)


class DatabaseSessionRepository:
    def __init__(self, config: IMSimConfig):
        if not config.database_url:
            raise ValueError("DatabaseSessionRepository requires IMSIM_DATABASE_URL.")
        connect_args: dict[str, Any] = {}
        if config.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        self._engine = create_engine(
            config.database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self._initialize_schema()
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> SimulationState:
        session_id = normalize_session_id(session_id)
        with self._lock, self._session_factory() as session:
            record = session.get(SessionStateRecord, session_id)
            if record is None:
                state = default_state()
                state.revision = 1
                session.add(
                    SessionStateRecord(
                        session_id=session_id,
                        state_json=state.to_dict(),
                        revision=state.revision,
                        updated_at=datetime.now(UTC),
                    )
                )
                try:
                    session.commit()
                    return state.clone()
                except IntegrityError:
                    session.rollback()
                    record = session.get(SessionStateRecord, session_id)
                    if record is None:
                        raise
            return self._state_from_record(record)

    def save(self, session_id: str, state: SimulationState) -> None:
        session_id = normalize_session_id(session_id)
        with self._lock, self._session_factory() as session:
            if state.revision <= 0:
                snapshot = state.clone()
                snapshot.revision = 1
                session.add(
                    SessionStateRecord(
                        session_id=session_id,
                        state_json=snapshot.to_dict(),
                        revision=snapshot.revision,
                        updated_at=datetime.now(UTC),
                    )
                )
                try:
                    session.commit()
                except IntegrityError as exc:
                    session.rollback()
                    raise SessionConflictError(
                        f"Session {session_id} was created concurrently."
                    ) from exc
                state.revision = snapshot.revision
                return
            snapshot = state.clone()
            snapshot.revision = state.revision + 1
            result = session.execute(
                update(SessionStateRecord)
                .where(
                    SessionStateRecord.session_id == session_id,
                    SessionStateRecord.revision == state.revision,
                )
                .values(
                    state_json=snapshot.to_dict(),
                    revision=snapshot.revision,
                    updated_at=datetime.now(UTC),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                raise SessionConflictError(
                    f"Session {session_id} changed from revision {state.revision}."
                )
            session.commit()
            state.revision = snapshot.revision

    def reset(self, session_id: str) -> SimulationState:
        session_id = normalize_session_id(session_id)
        state = default_state()
        self.save(session_id, state)
        return state.clone()

    def pause_all(self) -> None:
        with self._lock, self._session_factory() as session:
            records = session.scalars(select(SessionStateRecord)).all()
            for record in records:
                state = self._state_from_record(record)
                state.is_initialized = False
                state.revision += 1
                record.state_json = state.to_dict()
                record.revision = state.revision
                record.updated_at = datetime.now(UTC)
            session.commit()

    def persist_all(self) -> None:
        return None

    def _initialize_schema(self) -> None:
        lock_id = 1847349125
        with self._engine.begin() as connection:
            if self._engine.dialect.name == "postgresql":
                connection.execute(text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": lock_id})
            try:
                Base.metadata.create_all(connection)
                self._ensure_revision_column(connection)
            finally:
                if self._engine.dialect.name == "postgresql":
                    connection.execute(
                        text("SELECT pg_advisory_unlock(:lock_id)"),
                        {"lock_id": lock_id},
                    )

    def _ensure_revision_column(self, connection) -> None:
        inspector = inspect(connection)
        if "session_states" not in inspector.get_table_names():
            return
        columns = {column["name"] for column in inspector.get_columns("session_states")}
        if "revision" in columns:
            return
        connection.execute(
            text("ALTER TABLE session_states ADD COLUMN revision INTEGER NOT NULL DEFAULT 1")
        )

    def _state_from_record(self, record: SessionStateRecord) -> SimulationState:
        state = SimulationState.from_dict(record.state_json)
        state.revision = max(1, int(record.revision))
        return state


def create_session_repository(config: IMSimConfig) -> SessionRepository:
    if config.database_url:
        return DatabaseSessionRepository(config)
    return FileSessionRepository(config)
