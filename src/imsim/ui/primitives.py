from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html


def class_names(*parts: str | None) -> str:
    return " ".join(part for part in parts if part)


def optional_id(component_id: str | None) -> dict[str, str]:
    return {"id": component_id} if component_id else {}


def action_button(
    label: str,
    component_id: str,
    variant: str,
    *,
    disabled: bool = False,
    class_name: str = "",
    title: str | None = None,
    aria_label: str | None = None,
) -> html.Button:
    return html.Button(
        label,
        id=component_id,
        n_clicks=0,
        className=class_names("imsim-button", f"button-{variant}", class_name),
        disabled=disabled,
        title=title,
        **({"aria-label": aria_label} if aria_label else {}),
    )


def shell_card(
    children: Any,
    *,
    class_name: str = "",
    body_class_name: str = "",
    body_id: str | None = None,
    **card_props: Any,
) -> dbc.Card:
    body_props = {}
    if body_class_name:
        body_props["className"] = body_class_name
    if body_id:
        body_props["id"] = body_id

    return dbc.Card(
        dbc.CardBody(children, **body_props),
        className=class_names("shell-card", class_name),
        **card_props,
    )


def hero_row(
    *,
    kicker: str,
    title: str,
    copy: str,
    actions: Any,
    kicker_id: str | None = None,
    title_id: str | None = None,
    copy_id: str | None = None,
    extra_left: list[Any] | None = None,
    left_lg: int = 8,
    actions_lg: int = 4,
) -> dbc.Row:
    left_children = [
        html.Div(kicker, className="hero-kicker", **optional_id(kicker_id)),
        html.H1(title, className="hero-title", **optional_id(title_id)),
        html.P(copy, className="hero-copy", **optional_id(copy_id)),
        *(extra_left or []),
    ]
    return dbc.Row(
        [
            dbc.Col(left_children, lg=left_lg),
            dbc.Col(
                html.Div(actions, className="hero-actions hero-actions-compact"),
                lg=actions_lg,
                className="hero-actions-wrap",
            ),
        ],
        className="g-3 align-items-center",
    )


def hero_card(*, class_name: str, **hero_props: Any) -> dbc.Card:
    return shell_card(hero_row(**hero_props), class_name=class_name)


def modal_actions(children: Any, *, class_name: str = "") -> dbc.ModalFooter:
    return dbc.ModalFooter(children, className=class_names("modal-actions", class_name))


def work_modal(
    title: Any,
    component_id: str,
    body: Any,
    *,
    footer: Any | None = None,
    body_class_name: str = "",
    size: str | None = None,
    centered: bool = False,
    scrollable: bool = False,
    content_class_name: str = "imsim-modal-content",
    **modal_props: Any,
) -> dbc.Modal:
    body_props = {"class_name": body_class_name} if body_class_name else {}
    children = [dbc.ModalHeader(title), dbc.ModalBody(body, **body_props)]
    if footer is not None:
        children.append(footer)

    return dbc.Modal(
        children,
        id=component_id,
        is_open=False,
        size=size,
        centered=centered,
        scrollable=scrollable,
        content_class_name=content_class_name,
        **modal_props,
    )


def number_field(
    label: str,
    component_id: str,
    *,
    value: float | int | None = None,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    step: float | int | None = None,
    class_name: str = "",
) -> html.Div:
    return html.Div(
        [
            html.Label(label, htmlFor=component_id, className="control-label"),
            dbc.Input(
                id=component_id,
                type="number",
                value=value,
                min=min_value,
                max=max_value,
                step=step,
                className="control-input",
                inputMode="decimal",
            ),
        ],
        className=class_names("control-field", class_name),
    )


def toggle_field(
    label: str,
    component_id: str,
    *,
    enabled: bool,
    class_name: str = "",
) -> html.Div:
    return html.Div(
        [
            html.Span(label, className="imsim-toggle-copy"),
            dbc.Switch(
                id=component_id,
                value=enabled,
                label="",
                class_name="imsim-toggle-switch",
                input_class_name="imsim-toggle-input",
            ),
        ],
        className=class_names("toggle-field", class_name),
    )


def review_cycle_override_control() -> html.Div:
    return html.Div(
        [
            number_field(
                "RC Override",
                "review-cycle-override-input",
                value=14,
                min_value=1,
                step=1,
                class_name="mb-1",
            ),
            html.Div(
                "Use this for the current buy only. It resets after an order is placed.",
                className="helper-copy mb-2",
            ),
            html.Div(
                id="review-cycle-override-indicator",
                className="review-override-indicator",
            ),
            html.Div(
                id="review-cycle-override-feedback",
                className="mb-2",
            ),
        ],
        className="review-override-control mt-3",
        id="review-cycle-override-wrap",
    )
