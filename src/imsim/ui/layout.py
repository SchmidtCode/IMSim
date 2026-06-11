from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from ..config import IMSimConfig
from ..models import default_state
from .components import github_footer_card
from .layout_academy import (
    academy_menu_shell,
    lesson_shell,
    reference_modal,
)
from .layout_dashboard import dashboard_shell, simulator_shell
from .layout_modals import add_item_modal, custom_order_modal, po_overview_modal


def _stores_and_intervals(config: IMSimConfig) -> list:
    return [
        dcc.Store(id="user-data-store", storage_type="local", data={}),
        dcc.Store(id="page-load", data=0),
        dcc.Store(id="gh-footer-store", storage_type="local", data=True),
        dcc.Store(id="upload-preview-data"),
        dcc.Store(id="theme-store", storage_type="local", data="light"),
        dcc.Store(id="session-revision", data=0),
        dcc.Store(id="dashboard-tick", data=0),
        dcc.Store(id="view-scroll-store"),
        dcc.Store(id="view-scroll-sink"),
        dcc.Store(id="page-lifecycle-store", data={"active": True, "reason": "initial"}),
        dcc.Interval(id="interval-component", interval=1000, disabled=True),
        dcc.Interval(
            id="shutdown-poll",
            interval=1000,
            n_intervals=0,
            disabled=not (config.admin_token or config.allow_dev_shutdown),
        ),
        html.Div(id="maintenance-banner"),
    ]


def build_layout(config: IMSimConfig):
    initial_state = default_state()
    return html.Div(
        dbc.Container(
            [
                *_stores_and_intervals(config),
                academy_menu_shell(initial_state),
                lesson_shell(),
                reference_modal(),
                simulator_shell(),
                dashboard_shell(),
                add_item_modal(),
                custom_order_modal(),
                po_overview_modal(),
                github_footer_card(config.github_url),
            ],
            fluid=True,
            className="imsim-shell py-4",
        ),
        id="app-theme",
        className="imsim-theme theme-light",
        **{"data-bs-theme": "light"},
    )
