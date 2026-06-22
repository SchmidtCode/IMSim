from __future__ import annotations

import dash
from dash_ag_grid import AgGrid

import imsim.ui.components as ui_components
from imsim.callbacks.simulation import _inventory_table_update, _lesson_tick_session_revision
from imsim.callbacks.training import (
    _dashboard_layout_revision_update,
    dashboard_shell_class_name,
)
from imsim.services.training import build_level_state, build_simulator_state


def _walk_components(component):
    yield component
    children = getattr(component, "children", None)
    if children is None or isinstance(children, str):
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_components(child)
        return
    yield from _walk_components(children)


def _output_pairs(spec):
    outputs = spec["output"]
    if not isinstance(outputs, list):
        outputs = [outputs]
    return {(output.component_id, output.component_property) for output in outputs}


def _input_pairs(spec):
    return {(input_spec["id"], input_spec["property"]) for input_spec in spec["inputs"]}


def _find_callback(dash_app, required_outputs):
    required = set(required_outputs)
    for spec in dash_app.callback_map.values():
        if required.issubset(_output_pairs(spec)):
            return spec
    raise AssertionError(f"Callback with outputs {sorted(required)} not found")


def test_components_facade_exports_existing_ui_surface():
    expected = {
        "_grid_theme_class",
        "_plot_base_layout",
        "_plot_line",
        "_plot_marker",
        "_plot_marker_outline",
        "academy_level_card_children",
        "build_custom_order_grid",
        "build_exception_center",
        "build_inventory_figure",
        "build_inventory_table",
        "build_kpi_strip",
        "build_po_overview_grid",
        "github_footer_card",
        "inventory_graph_style",
        "refresh_inventory_figure",
        "service_card_children",
    }
    assert expected <= set(ui_components.__all__)
    assert all(hasattr(ui_components, name) for name in expected)


def test_layout_keeps_callback_target_ids(dash_app):
    component_ids = {
        getattr(component, "id", None)
        for component in _walk_components(dash_app.layout)
        if getattr(component, "id", None)
    }
    assert {
        "academy-menu-shell",
        "lesson-shell",
        "simulator-shell",
        "dashboard-shell",
        "academy-cheat-code-button",
        "reference-modal",
        "add-item-modal",
        "place-custom-order-modal",
        "po-overview-modal",
        "inventory-graph",
        "inventory-table-shell",
        "custom-order-grid",
        "po-overview-grid",
        "dashboard-layout-revision",
    } <= component_ids


def test_dashboard_render_waits_for_dashboard_layout_revision(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("inventory-graph", "figure"),
            ("inventory-graph", "style"),
            ("kpi-strip", "children"),
            ("inventory-table-shell", "children"),
            ("exception-center-shell", "children"),
        ],
    )
    assert _input_pairs(spec) == {
        ("user-data-store", "data"),
        ("dashboard-layout-revision", "data"),
        ("dashboard-tick", "data"),
        ("theme-store", "data"),
    }


def test_running_lesson_tick_does_not_rebuild_training_shell():
    class RevisionContext:
        def next_session_revision(self, revision):
            return int(revision or 0) + 1

    ctx = RevisionContext()

    assert _lesson_tick_session_revision({"lesson_completed": 0}, 7, ctx) is dash.no_update
    assert _lesson_tick_session_revision({"lesson_completed": 1}, 7, ctx) == 8


def test_lesson_dashboard_tick_does_not_touch_inventory_grid():
    state = build_level_state("level-3")
    simulator_state = build_simulator_state()

    initial_table = _inventory_table_update(state, "light", "dashboard-layout-revision")
    lesson_tick = _inventory_table_update(state, "light", "dashboard-tick")
    simulator_tick = _inventory_table_update(simulator_state, "light", "dashboard-tick")

    assert isinstance(initial_table, AgGrid)
    assert lesson_tick is dash.no_update
    assert isinstance(simulator_tick, AgGrid)


def test_interval_tick_updates_terminal_lesson_controls_immediately(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("day-display", "children"),
            ("sim-status", "children"),
            ("start-button", "children"),
            ("start-button", "className"),
            ("start-button", "disabled"),
            ("lesson-compact-summary", "children"),
            ("interval-component", "disabled"),
        ],
    )
    assert _input_pairs(spec) == {("interval-component", "n_intervals")}


def test_training_shell_render_listens_to_session_revision(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("academy-menu-shell", "style"),
            ("lesson-shell", "style"),
            ("dashboard-shell", "className"),
            ("interval-component", "disabled"),
            ("dashboard-layout-revision", "data"),
        ],
    )
    assert _input_pairs(spec) == {
        ("user-data-store", "data"),
        ("session-revision", "data"),
    }


def test_dashboard_layout_revision_ignores_start_only_changes():
    class RevisionContext:
        def next_session_revision(self, revision):
            return int(revision or 0) + 1

    ctx = RevisionContext()
    state = build_level_state("level-3")
    initial_revision = _dashboard_layout_revision_update(state, 0, ctx)

    state.is_initialized = True
    state.training.lesson_status = "running"

    assert _dashboard_layout_revision_update(state, initial_revision, ctx) is dash.no_update

    state.day = 2

    changed_revision = _dashboard_layout_revision_update(state, initial_revision, ctx)
    assert changed_revision["revision"] == initial_revision["revision"] + 1


def test_page_lifecycle_changes_refresh_session_state(dash_app):
    assert any(
        ("page-lifecycle-store", "data") in _input_pairs(spec)
        and ("session-revision", "data") in _output_pairs(spec)
        for spec in dash_app.callback_map.values()
    )


def test_academy_navigation_wires_final_lesson_button(dash_app):
    assert any(
        ("academy-level-19-button", "n_clicks") in _input_pairs(spec)
        for spec in dash_app.callback_map.values()
    )


def test_academy_navigation_emits_scroll_reset_trigger(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("session-revision", "data"),
            ("asq-apply-feedback", "children"),
            ("view-scroll-store", "data"),
        ],
    )
    assert ("academy-simulator-button", "n_clicks") in _input_pairs(spec)
    assert ("return-to-menu-button", "n_clicks") in _input_pairs(spec)


def test_scroll_reset_only_listens_for_navigation_tokens(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("view-scroll-sink", "data"),
        ],
    )
    assert _input_pairs(spec) == {("view-scroll-store", "data")}


def test_dashboard_shell_class_names_follow_lesson_variants():
    assert "lesson-layout-workspace-basic" in dashboard_shell_class_name(
        build_level_state("level-3")
    )
    assert "lesson-layout-workspace-signal" in dashboard_shell_class_name(
        build_level_state("level-10")
    )
    assert "lesson-layout-workspace-advanced" in dashboard_shell_class_name(
        build_level_state("level-15")
    )
    assert "lesson-layout-workspace-certification" in dashboard_shell_class_name(
        build_level_state("level-19")
    )


def test_theme_callback_updates_control_modal_content_classes(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("lesson-intro-modal", "content_class_name"),
            ("reference-modal", "content_class_name"),
            ("add-item-modal", "content_class_name"),
            ("place-custom-order-modal", "content_class_name"),
            ("po-overview-modal", "content_class_name"),
        ],
    )
    assert _input_pairs(spec) == {("theme-store", "data")}


def test_reference_modal_toggle_is_wired(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("reference-modal", "is_open"),
        ],
    )
    assert _input_pairs(spec) == {
        ("academy-reference-button", "n_clicks"),
        ("experience-reference-button", "n_clicks"),
        ("simulator-reference-button", "n_clicks"),
        ("reference-modal-close", "n_clicks"),
    }


def test_unlock_all_button_updates_progress(dash_app):
    matches = [
        spec
        for spec in dash_app.callback_map.values()
        if ("session-revision", "data") in _output_pairs(spec)
        and _input_pairs(spec) == {("academy-cheat-code-button", "n_clicks")}
    ]
    assert matches


def test_randomize_button_populates_manual_item_fields(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("usage-rate-input", "value"),
            ("lead-time-input", "value"),
            ("item-cost-input", "value"),
            ("pna-input", "value"),
            ("safety-allowance-input", "value"),
            ("standard-pack-input", "value"),
            ("hits-per-month-input", "value"),
        ],
    )
    assert _input_pairs(spec) == {("randomize-button", "n_clicks")}


def test_state_changes_emit_session_revision(dash_app):
    cases = [
        [
            ("dashboard-tick", "data"),
            ("session-revision", "data"),
            ("asq-apply-feedback", "children"),
        ],
        [
            ("session-revision", "data"),
            ("add-item-error", "children"),
        ],
        [("session-revision", "data"), ("update-params-conf", "children")],
        [
            ("review-cycle-override-feedback", "children"),
            ("session-revision", "data"),
        ],
        [("session-revision", "data"), ("upload-feedback", "children")],
        [
            ("custom-order-grid", "rowData"),
            ("custom-order-grid", "columnDefs"),
            ("session-revision", "data"),
        ],
        [
            ("po-overview-grid", "rowData"),
            ("po-overview-grid", "columnDefs"),
            ("po-overview-grid", "selectedRows"),
            ("session-revision", "data"),
        ],
    ]

    for required_outputs in cases:
        spec = _find_callback(dash_app, required_outputs)
        assert ("session-revision", "data") in _output_pairs(spec)
