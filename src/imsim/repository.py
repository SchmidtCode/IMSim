from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from .config import IMSimConfig
from .models import SimulationState, default_state


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
