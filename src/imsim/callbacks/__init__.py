from __future__ import annotations

from .common import CallbackRegistrarContext, _triggered_click_count
from .inventory import register_inventory_callbacks
from .maintenance import register_maintenance_callbacks
from .simulation import register_simulation_callbacks
from .training import register_training_callbacks

__all__ = ["_triggered_click_count", "register_callbacks"]


def register_callbacks(app, repository, maintenance):
    ctx = CallbackRegistrarContext(app=app, repository=repository, maintenance=maintenance)
    register_training_callbacks(ctx)
    register_simulation_callbacks(ctx)
    register_inventory_callbacks(ctx)
    register_maintenance_callbacks(ctx)
