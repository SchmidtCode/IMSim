from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from ..models import SimulationState
from ..services.training import academy_levels
from .academy_components import (
    academy_level_card_children,
    reference_modal_children,
    simulator_unlock_children,
)
from .primitives import action_button, hero_row, shell_card


def _academy_card_col(children: object) -> dbc.Col:
    return dbc.Col(
        shell_card(
            children,
            class_name="academy-card h-100",
        ),
        lg=4,
        md=6,
        className="d-flex",
    )


def _academy_level_col(level_index: int, initial_state: SimulationState) -> dbc.Col:
    return _academy_card_col(
        html.Div(
            academy_level_card_children(level_index, initial_state),
            id=f"academy-level-{level_index}-card",
            className="academy-card-content",
        )
    )


def _simulator_col(initial_state: SimulationState) -> dbc.Col:
    return _academy_card_col(
        html.Div(
            [
                html.Div(
                    simulator_unlock_children(initial_state),
                    id="academy-simulator-card",
                    className="academy-card-status-block",
                ),
                action_button(
                    "Open Simulator" if initial_state.training.simulator_unlocked else "Locked",
                    "academy-simulator-button",
                    "primary",
                    disabled=not initial_state.training.simulator_unlocked,
                    class_name="button-block mt-3",
                ),
            ],
            className="academy-card-content",
        )
    )


def _lesson_detail_col(label: str, component_id: str, *, lg: int) -> dbc.Col:
    return dbc.Col(
        shell_card(
            [
                html.Div(label, className="panel-label"),
                html.Div(id=component_id),
            ],
            class_name="lesson-detail-card h-100",
        ),
        lg=lg,
    )


def academy_menu_shell(initial_state: SimulationState) -> html.Div:
    return html.Div(
        [
            shell_card(
                [
                    hero_row(
                        kicker="Inventory Management Training",
                        title="IMSim Academy",
                        copy=(
                            "Practice replenishment decisions through guided lessons, "
                            "then unlock the simulator when the operating model clicks."
                        ),
                        actions=[
                            action_button(
                                "Reset Progress",
                                "academy-reset-progress-button",
                                "secondary",
                                class_name="button-pill",
                            ),
                            action_button(
                                "Unlock All",
                                "academy-cheat-code-button",
                                "secondary",
                                class_name="button-pill",
                            ),
                            action_button(
                                "Dark mode",
                                "academy-theme-toggle",
                                "ghost",
                                class_name="theme-toggle-button button-sm",
                            ),
                            action_button(
                                "Reference",
                                "academy-reference-button",
                                "ghost",
                                class_name="button-pill",
                            ),
                        ],
                        left_lg=6,
                        actions_lg=6,
                    ),
                    html.Div(id="academy-result-banner"),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    id="academy-progress-summary", className="academy-progress"
                                ),
                                md=6,
                                lg=8,
                            ),
                            _simulator_col(initial_state),
                        ],
                        className="g-3 align-items-center academy-overview",
                    ),
                    dbc.Row(
                        [
                            _academy_level_col(level.index, initial_state)
                            for level in academy_levels()
                        ],
                        className="g-3 mt-2",
                    ),
                ],
                class_name="academy-shell mb-4",
            )
        ],
        id="academy-menu-shell",
    )


def lesson_shell() -> html.Div:
    return html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Lesson",
                                        id="lesson-intro-kicker",
                                        className="hero-kicker",
                                    ),
                                    html.H2(
                                        "Academy Lesson",
                                        id="lesson-title",
                                        className="lesson-intro-title",
                                    ),
                                    html.P(
                                        "Start a lesson from the academy menu.",
                                        id="lesson-copy",
                                        className="lesson-intro-copy",
                                    ),
                                ]
                            ),
                            html.Button(
                                id="lesson-intro-dismiss-button",
                                className="btn-close",
                                **{"aria-label": "Enter lesson"},
                            ),
                        ],
                        close_button=False,
                        class_name="lesson-intro-header",
                    ),
                    dbc.ModalBody(
                        [
                            dbc.Row(
                                [
                                    _lesson_detail_col(
                                        "Tutorial",
                                        "lesson-tutorial",
                                        lg=8,
                                    ),
                                    _lesson_detail_col(
                                        "Objectives",
                                        "lesson-objectives",
                                        lg=4,
                                    ),
                                ],
                                className="g-3 lesson-intro-grid",
                            ),
                            html.Div(
                                id="lesson-locked",
                                className="lesson-locked-cache",
                            ),
                        ],
                        class_name="lesson-intro-body",
                    ),
                ],
                id="lesson-intro-modal",
                labelledby="lesson-title",
                is_open=False,
                centered=True,
                scrollable=True,
                size="xl",
                class_name="lesson-intro-modal",
                dialog_class_name="lesson-intro-dialog",
                content_class_name="lesson-intro-content",
                backdrop_class_name="lesson-intro-backdrop",
            )
        ],
        id="lesson-shell",
        style={"display": "none"},
    )


def reference_modal() -> dbc.Modal:
    return dbc.Modal(
        dbc.ModalBody(
            [
                html.H2("Reference", id="reference-title", className="visually-hidden"),
                html.Button(
                    id="reference-modal-close",
                    className="btn-close reference-close",
                    **{"aria-label": "Close reference"},
                ),
                *reference_modal_children(),
            ]
        ),
        id="reference-modal",
        is_open=False,
        centered=True,
        scrollable=True,
        size="xl",
        labelledby="reference-title",
        content_class_name="imsim-modal-content",
    )
