from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dash
from dash.exceptions import PreventUpdate

from ..models import default_state
from ..repository import InvalidSessionIdError, SessionConflictError, SessionRepository
from ..services.simulation import MaintenanceController


def _triggered_click_count(
    triggered_id: str | dict | None, click_map: dict[str, int | None]
) -> int:
    if not isinstance(triggered_id, str):
        return 0
    return int(click_map.get(triggered_id) or 0)


@dataclass(slots=True)
class CallbackRegistrarContext:
    app: dash.Dash
    repository: SessionRepository
    maintenance: MaintenanceController

    def require_session(self, client_data: dict | None):
        session_id = (client_data or {}).get("uuid")
        if not session_id:
            raise PreventUpdate
        try:
            return session_id, self.repository.get_or_create(session_id)
        except InvalidSessionIdError as exc:
            raise PreventUpdate from exc

    def persist_state(self, session_id: str, state) -> None:
        try:
            self.repository.save(session_id, state)
        except SessionConflictError as exc:
            raise PreventUpdate from exc

    def carry_revision(self, next_state, current_state) -> None:
        if getattr(next_state, "revision", 0) <= 0:
            next_state.revision = current_state.revision

    def theme_name(self, theme: str | None) -> str:
        return "dark" if theme == "dark" else "light"

    def button_class(self, variant: str, extra: str = "") -> str:
        return " ".join(part for part in ["imsim-button", f"button-{variant}", extra] if part)

    def start_button_state(
        self,
        state,
        *,
        running: bool,
        disabled: bool = False,
        resumable: bool = False,
    ) -> tuple[str, str]:
        mode = "Simulation" if state.training.current_view == "simulator" else "Lesson"
        if disabled:
            label = "Lesson Complete" if state.training.current_view == "lesson" else "Disabled"
            return label, self.button_class("secondary", "button-pill")
        if running:
            return f"Pause {mode}", self.button_class("warning", "button-pill")
        if resumable:
            return f"Resume {mode}", self.button_class("success", "button-pill")
        return f"Start {mode}", self.button_class("success", "button-pill")

    def lesson_terminal(self, state) -> bool:
        return state.training.current_view == "lesson" and state.training.lesson_status in {
            "passed",
            "failed",
        }

    def panel_style(self, enabled: bool) -> dict[str, str]:
        return {} if enabled else {"display": "none"}

    def current_state(self, client_data: dict | None):
        session_id = (client_data or {}).get("uuid")
        if not session_id:
            return default_state()
        try:
            return self.repository.get_or_create(session_id)
        except InvalidSessionIdError:
            return default_state()

    def toggle_enabled(self, value: Any) -> bool:
        return bool(value)

    def coerce_number(self, value, *, integer: bool = False):
        if value in (None, ""):
            return None
        try:
            return int(value) if integer else float(value)
        except TypeError, ValueError:
            return None

    def next_session_revision(self, revision: int | None) -> int:
        return int(revision or 0) + 1
