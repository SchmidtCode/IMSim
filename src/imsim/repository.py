from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from sqlalchemy import JSON, DateTime, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import IMSimConfig
from .models import SimulationState, default_state


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
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FileSessionRepository:
    def __init__(self, config: IMSimConfig):
        self._config = config
        self._lock = RLock()
        self._cache: dict[str, SimulationState] = {}
        self._session_dir = Path(config.session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self._session_dir / f"{session_id}.json"

    def get_or_create(self, session_id: str) -> SimulationState:
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id].clone()
            path = self.path_for(session_id)
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    state = SimulationState.from_dict(json.load(handle))
            else:
                state = default_state()
                self._write_path(path, state)
            self._cache[session_id] = state
            return state.clone()

    def save(self, session_id: str, state: SimulationState) -> None:
        with self._lock:
            snapshot = state.clone()
            self._cache[session_id] = snapshot
            self._write_path(self.path_for(session_id), snapshot)

    def reset(self, session_id: str) -> SimulationState:
        state = default_state()
        self.save(session_id, state)
        return state.clone()

    def pause_all(self) -> None:
        with self._lock:
            for session_id, state in list(self._cache.items()):
                state.is_initialized = False
                self._write_path(self.path_for(session_id), state)

    def persist_all(self) -> None:
        with self._lock:
            for session_id, state in list(self._cache.items()):
                self._write_path(self.path_for(session_id), state)

    def _write_path(self, path: Path, state: SimulationState) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2)


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
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> SimulationState:
        with self._lock, self._session_factory() as session:
            record = session.get(SessionStateRecord, session_id)
            if record is None:
                state = default_state()
                self._upsert(session, session_id, state)
                session.commit()
                return state.clone()
            return SimulationState.from_dict(record.state_json)

    def save(self, session_id: str, state: SimulationState) -> None:
        with self._lock, self._session_factory() as session:
            self._upsert(session, session_id, state)
            session.commit()

    def reset(self, session_id: str) -> SimulationState:
        state = default_state()
        self.save(session_id, state)
        return state.clone()

    def pause_all(self) -> None:
        with self._lock, self._session_factory() as session:
            records = session.scalars(select(SessionStateRecord)).all()
            for record in records:
                state = SimulationState.from_dict(record.state_json)
                state.is_initialized = False
                record.state_json = state.to_dict()
                record.updated_at = datetime.now(UTC)
            session.commit()

    def persist_all(self) -> None:
        return None

    def _upsert(self, session: Session, session_id: str, state: SimulationState) -> None:
        payload = state.to_dict()
        record = session.get(SessionStateRecord, session_id)
        if record is None:
            session.add(
                SessionStateRecord(
                    session_id=session_id,
                    state_json=payload,
                    updated_at=datetime.now(UTC),
                )
            )
            return
        record.state_json = payload
        record.updated_at = datetime.now(UTC)


def create_session_repository(config: IMSimConfig) -> SessionRepository:
    if config.database_url:
        return DatabaseSessionRepository(config)
    return FileSessionRepository(config)
