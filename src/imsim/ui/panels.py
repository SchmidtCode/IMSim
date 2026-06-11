from __future__ import annotations

from collections.abc import Callable

import dash_bootstrap_components as dbc
from dash import html

from ..models import InventoryItem, SimulationState
from ..services.planning import format_money, item_on_order
from ..services.training import active_layout_variant, active_level, lesson_days_remaining


def _lesson_snapshot_table(
    label: str, headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="lesson-snapshot-label"),
            html.Table(
                [
                    html.Thead(html.Tr([html.Th(header) for header in headers])),
                    html.Tbody([html.Tr([html.Td(value) for value in row]) for row in rows]),
                ],
                className="lesson-service-table",
            ),
        ],
        className="lesson-snapshot-block",
    )


def _lesson_snapshot_disclosure(
    body: html.Div,
    *,
    label: str = "Show snapshot details",
    hint: str = "Status and goal details",
    open_by_default: bool = False,
) -> html.Details:
    props: dict[str, object] = {"className": "lesson-snapshot-disclosure"}
    if open_by_default:
        props["open"] = True
    return html.Details(
        [
            html.Summary(
                [
                    html.Span(label),
                    html.Span(hint, className="lesson-disclosure-hint"),
                ],
                className="lesson-disclosure-summary",
            ),
            html.Div(body, className="lesson-disclosure-body"),
        ],
        **props,
    )


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _lesson_item_snapshot_columns(level_index: int) -> tuple[str, ...]:
    return {
        1: ("item", "on_hand", "daily_usage", "backorder"),
        2: ("item", "on_hand", "on_order", "backorder", "pna"),
        3: ("item", "on_hand", "daily_usage", "backorder"),
        4: ("item", "usage_rate", "hits_per_month", "on_hand"),
        5: ("item", "usage_rate", "daily_usage", "on_hand"),
        6: ("item", "on_hand", "on_order", "lead_time", "pna"),
        7: ("item", "on_hand", "on_order", "backorder", "pna"),
        8: ("item", "on_hand", "usage_rate", "op", "days_to_op"),
        9: ("item", "safety_allowance", "op", "on_hand"),
        10: ("item", "pna", "op", "safety_allowance", "soq"),
        11: ("item", "hits_per_month", "usage_rate", "pna"),
        12: ("item", "pna", "op", "lp", "soq"),
        13: ("item", "pna", "op", "lp", "soq"),
        14: ("item", "item_cost", "pna", "soq"),
        15: ("item", "eoq", "oq", "pna", "op"),
        16: ("item", "pna", "oq", "standard_pack", "soq"),
        17: ("item", "pna", "cp", "surplus_line"),
        18: ("item", "pna", "op", "lp", "soq", "on_order"),
        19: ("item", "pna", "op", "lp", "soq"),
    }.get(level_index, ("item", "on_hand", "on_order", "backorder"))


_LESSON_ITEM_SNAPSHOT_LABELS = {
    "item": "Item",
    "on_hand": "On Hand",
    "daily_usage": "Daily Usage",
    "usage_rate": "Usage",
    "lead_time": "Lead Time",
    "hits_per_month": "Hits",
    "item_cost": "Cost",
    "on_order": "On Order",
    "backorder": "Backorder",
    "pna": "PNA",
    "op": "OP",
    "lp": "LP",
    "eoq": "EOQ",
    "oq": "OQ",
    "soq": "SOQ",
    "standard_pack": "Pack",
    "safety_allowance": "Safety %",
    "cp": "CP",
    "surplus_line": "Surplus threshold",
    "days_to_op": "Days to OP",
}

_LESSON_ITEM_SNAPSHOT_FORMATTERS: dict[str, Callable[[int, InventoryItem], str]] = {
    "item": lambda index, _item: str(index),
    "on_hand": lambda _index, item: f"{item.on_hand:.1f}",
    "daily_usage": lambda _index, item: f"{item.daily_ur:.1f}",
    "usage_rate": lambda _index, item: f"{item.usage_rate:.1f}",
    "lead_time": lambda _index, item: f"{item.lead_time:.1f}",
    "hits_per_month": lambda _index, item: f"{item.hits_per_month:.1f}",
    "item_cost": lambda _index, item: format_money(item.item_cost),
    "on_order": lambda _index, item: f"{item_on_order(item):.1f}",
    "backorder": lambda _index, item: f"{item.backorder:.1f}",
    "pna": lambda _index, item: f"{item.pna:.1f}",
    "op": lambda _index, item: f"{item.op:.1f}",
    "lp": lambda _index, item: f"{item.lp:.1f}",
    "eoq": lambda _index, item: f"{item.eoq:.1f}",
    "oq": lambda _index, item: f"{item.oq:.1f}",
    "soq": lambda _index, item: f"{item.soq:.1f}",
    "standard_pack": lambda _index, item: f"{item.standard_pack:.1f}",
    "safety_allowance": lambda _index, item: f"{item.safety_allowance * 100.0:.1f}%",
    "cp": lambda _index, item: f"{item.cp:.1f}",
    "surplus_line": lambda _index, item: f"{item.surplus_line:.1f}",
    "days_to_op": lambda _index, item: f"{item.ats_days_frm_op:.1f}",
}


def _lesson_item_snapshot_value(column: str, index: int, item: InventoryItem) -> str:
    try:
        return _LESSON_ITEM_SNAPSHOT_FORMATTERS[column](index, item)
    except KeyError as exc:
        raise ValueError(f"Unsupported lesson snapshot column: {column}") from exc


def _lesson_item_snapshot_block(level_index: int, items: list[InventoryItem]) -> html.Div:
    columns = _lesson_item_snapshot_columns(level_index)
    return _lesson_snapshot_table(
        "Inventory",
        tuple(_LESSON_ITEM_SNAPSHOT_LABELS[column] for column in columns),
        tuple(
            tuple(_lesson_item_snapshot_value(column, index, item) for column in columns)
            for index, item in enumerate(items, start=1)
        ),
    )


def _lesson_position_snapshot_block(state: SimulationState) -> html.Div:
    variant = active_layout_variant(state)
    on_order = sum(item_on_order(item) for item in state.items)
    backorder = sum(item.backorder for item in state.items)
    if variant == "workspace_basic":
        at_or_below_op = sum(1 for item in state.items if item.on_hand <= item.op)
        rows = (
            (
                f"{sum(item.on_hand for item in state.items):.1f}",
                f"{on_order:.1f}",
                f"{backorder:.1f}",
                f"{at_or_below_op}/{len(state.items)}",
            ),
        )
        return _lesson_snapshot_table(
            "Position",
            ("Available", "On Order", "Backorder", "At/Below OP"),
            rows,
        )
    at_or_below_op = sum(1 for item in state.items if item.pna <= item.op)
    rows = (
        (
            f"{sum(item.pna for item in state.items):.1f}",
            f"{on_order:.1f}",
            f"{backorder:.1f}",
            f"{at_or_below_op}/{len(state.items)}",
        ),
    )
    return _lesson_snapshot_table(
        "Position",
        ("PNA", "On Order", "Backorder", "At/Below OP"),
        rows,
    )


def service_card_children(state: SimulationState) -> list:
    level = active_level(state)
    today = state.service_today
    totals = state.service_totals
    ats = sum(item.on_hand for item in state.items)
    on_order = sum(item_on_order(item) for item in state.items)
    backorder = sum(item.backorder for item in state.items)
    fill_today = (
        None if today.orders == 0 else 100.0 * (today.orders - today.orders_stockout) / today.orders
    )
    fill_total = (
        None
        if totals.orders == 0
        else 100.0 * (totals.orders - totals.orders_stockout) / totals.orders
    )
    if level is not None and level.index == 1 and state.items:
        item = state.items[0]
        status = "Depleted" if item.on_hand <= 0 else "Available"
        return [
            html.Div(
                [
                    _lesson_snapshot_table(
                        "Service",
                        ("On Hand", "Daily Usage", "Backorder", "Status"),
                        (
                            (
                                f"{item.on_hand:.1f} units",
                                f"{item.daily_ur:.1f} units/day",
                                f"{item.backorder:.1f} units",
                                status,
                            ),
                        ),
                    ),
                    _lesson_snapshot_table(
                        "Inventory",
                        ("Item", "On Hand", "Daily Usage", "Backorder"),
                        (
                            (
                                "1",
                                f"{item.on_hand:.0f}",
                                f"{item.daily_ur:.0f}",
                                f"{item.backorder:.0f}",
                            ),
                        ),
                    ),
                ],
                className="lesson-snapshot-stack",
            )
        ]
    if level is not None and level.index == 2 and state.items:
        item = state.items[0]
        pna_at_or_above_op = sum(1 for candidate in state.items if candidate.pna >= candidate.op)
        reorder_target = int(level.win_conditions.get("guided_order_below_op_min", 1))
        snapshot_body = html.Div(
            [
                _lesson_snapshot_table(
                    "PNA Position",
                    ("On Hand", "On Order", "Backorder", "PNA", "OP"),
                    (
                        (
                            f"{item.on_hand:.1f}",
                            f"{item_on_order(item):.1f}",
                            f"{item.backorder:.1f}",
                            f"{item.pna:.1f}",
                            f"{item.op:.1f}",
                        ),
                    ),
                ),
                _lesson_snapshot_table(
                    "Goal Check",
                    ("Close PNA >= OP", "Below-OP Reorders", "Days Left"),
                    (
                        (
                            f"{pna_at_or_above_op}/{len(state.items)}",
                            f"{state.training.guided_orders_below_op}/{reorder_target}",
                            str(lesson_days_remaining(state)),
                        ),
                    ),
                ),
            ],
            className="lesson-snapshot-grid",
        )
        return [
            _lesson_snapshot_disclosure(
                snapshot_body,
                hint="PNA and goal status",
            )
        ]
    if level is not None and level.index == 3 and state.items:
        snapshot_body = html.Div(
            [
                html.Div(
                    [
                        _lesson_snapshot_table(
                            "Service",
                            ("Orders", "Stockouts", "Today Fill", "Total Fill", "Zero Available"),
                            (
                                (
                                    str(today.orders),
                                    str(today.orders_stockout),
                                    _format_pct(fill_today),
                                    _format_pct(fill_total),
                                    str(today.zero_on_hand_hits),
                                ),
                            ),
                        ),
                        _lesson_position_snapshot_block(state),
                    ],
                    className="lesson-snapshot-grid",
                ),
                _lesson_item_snapshot_block(level.index, state.items),
            ],
            className="lesson-snapshot-stack",
        )
        return [
            _lesson_snapshot_disclosure(
                snapshot_body,
                hint="Service and order-point status",
            )
        ]
    if level is not None and state.items:
        top_blocks = [
            _lesson_snapshot_table(
                "Service",
                ("Orders", "Stockouts", "Today Fill", "Total Fill", "Zero Available"),
                (
                    (
                        str(today.orders),
                        str(today.orders_stockout),
                        _format_pct(fill_today),
                        _format_pct(fill_total),
                        str(today.zero_on_hand_hits),
                    ),
                ),
            ),
            _lesson_position_snapshot_block(state),
        ]
        return [
            _lesson_snapshot_disclosure(
                html.Div(
                    [
                        html.Div(top_blocks, className="lesson-snapshot-grid"),
                        _lesson_item_snapshot_block(level.index, state.items),
                    ],
                    className="lesson-snapshot-stack",
                ),
                hint="Service, position, and item status",
            )
        ]

    return [
        dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    f"Today: Orders {today.orders} • Stockouts {today.orders_stockout} • "
                    f"Zero available hits "
                    f"{today.zero_on_hand_hits}"
                ),
                dbc.ListGroupItem(
                    f"Fill Rate (cumulative): {_format_pct(fill_total)}  •  "
                    f"Today: {_format_pct(fill_today)}"
                ),
                dbc.ListGroupItem(
                    dbc.Badge(
                        f"Available {int(ats)} • On-order {int(on_order)} • Backorder "
                        f"{int(backorder)}",
                        color="secondary",
                        pill=True,
                    )
                ),
            ],
            flush=True,
        )
    ]


def costs_card_children(state: SimulationState) -> list:
    costs = state.costs
    total = costs.ordering + costs.holding + costs.stockout + costs.expedite
    return [
        dbc.ListGroup(
            [
                dbc.ListGroupItem(f"Inventory Overhead: {format_money(total)}"),
                dbc.ListGroupItem(
                    " • ".join(
                        [
                            f"Ordering {format_money(costs.ordering)}",
                            f"Holding {format_money(costs.holding)}",
                            f"Stockout {format_money(costs.stockout)}",
                            f"Expedite {format_money(costs.expedite)}",
                        ]
                    )
                ),
                dbc.ListGroupItem(f"Receipts Booked: {format_money(costs.purchases)}"),
            ],
            flush=True,
        )
    ]


def sales_card_children(state: SimulationState) -> list:
    sales = state.sales
    costs = state.costs
    gross = sales.revenue - sales.cogs
    gross_pct = None if sales.revenue <= 0 else 100.0 * gross / sales.revenue
    after_overhead = gross - costs.total
    after_pct = None if sales.revenue <= 0 else 100.0 * after_overhead / sales.revenue

    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f}%"

    return [
        dbc.ListGroup(
            [
                dbc.ListGroupItem(f"Sales: {format_money(sales.revenue)}"),
                dbc.ListGroupItem(
                    f"COGS {format_money(sales.cogs)} • "
                    f"GM$ {format_money(gross)} • GM% {fmt(gross_pct)}"
                ),
                dbc.ListGroupItem(
                    f"After overhead {format_money(after_overhead)} • {fmt(after_pct)}"
                ),
            ],
            flush=True,
        )
    ]


def _kpi_card(title: str, value: str, tone: str, subtitle: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="metric-label"),
                html.Div(value, className="metric-value"),
                html.Div(subtitle, className="metric-subtitle"),
            ]
        ),
        className=f"metric-card metric-{tone}",
    )


def build_kpi_strip(state: SimulationState) -> list:
    revenue = state.sales.revenue
    gross = revenue - state.sales.cogs
    avg_inventory = state.analytics.inv_value_daysum / max(1, state.day)
    fill_rate = (
        None
        if state.service_totals.orders == 0
        else (state.service_totals.orders - state.service_totals.orders_stockout)
        / state.service_totals.orders
    )
    turns = None if avg_inventory <= 0 else state.sales.cogs / avg_inventory
    gmroi = None if avg_inventory <= 0 else gross / avg_inventory
    after_overhead = gross - state.costs.total
    after_pct = None if revenue <= 0 else after_overhead / revenue

    def fmt_pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.1f}%"

    def fmt_ratio(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"

    kpis = (
        ("Day", f"{state.day}", "neutral", "Simulation clock"),
        (
            "Fill Rate",
            fmt_pct(fill_rate),
            "good" if (fill_rate or 0) >= 0.95 else "warn",
            "Cumulative service",
        ),
        (
            "After-Overhead GM",
            fmt_pct(after_pct),
            "good" if (after_pct or 0) >= 0.15 else "warn",
            format_money(after_overhead),
        ),
        ("Avg Inventory", format_money(avg_inventory), "neutral", "Daily midpoint cost"),
        ("Turns", fmt_ratio(turns), "neutral", "COGS / avg inventory"),
        ("GMROI", fmt_ratio(gmroi), "neutral", "Gross margin return"),
    )
    return [
        dbc.Row(
            [dbc.Col(_kpi_card(*kpi), md=6, xl=2) for kpi in kpis],
            className="g-3",
        )
    ]


def build_exception_center(state: SimulationState):
    if not state.exception_center:
        return dbc.Alert("No ASQ exceptions recorded.", color="secondary")
    cards = []
    for exception in list(reversed(state.exception_center[-6:])):
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Strong(exception.code),
                                html.Span(f" • Day {exception.day}", className="ms-2 text-muted"),
                            ]
                        ),
                        html.Div(exception.message, className="mt-2"),
                        html.Small(
                            (
                                f"Item {exception.item_index + 1} • "
                                f"OP {exception.op:.2f} • ASQ {exception.asq:.2f}"
                            ),
                            className="text-muted",
                        ),
                    ]
                ),
                className="exception-card",
            )
        )
    return html.Div(cards, className="exception-stack")


def github_footer_card(github_url: str) -> html.Div:
    return html.Div(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div("IMSim", className="footer-title"),
                    html.Small(
                        [
                            "Project source on ",
                            html.A(
                                "GitHub",
                                href=github_url,
                                target="_blank",
                                rel="noopener noreferrer",
                            ),
                        ]
                    ),
                    html.Small(
                        "Independent open-source inventory management training project.",
                        className="d-block mt-2",
                    ),
                    html.Small(
                        ("Not affiliated with, endorsed by, sponsored by, or supported by Infor."),
                        className="d-block",
                    ),
                    " ",
                    html.Button(
                        "Hide",
                        id="gh-footer-hide",
                        n_clicks=0,
                        className="imsim-button button-link footer-hide",
                    ),
                ]
            ),
            className="footer-card",
        ),
        id="gh-footer",
    )
