from __future__ import annotations

from imsim.callbacks.training import dashboard_shell_class_name
from imsim.services.training import build_level_state


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


def test_dashboard_render_listens_to_session_revision_and_theme(dash_app):
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
        ("session-revision", "data"),
        ("dashboard-tick", "data"),
        ("theme-store", "data"),
    }


def test_training_shell_render_listens_to_session_revision(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("academy-menu-shell", "style"),
            ("lesson-shell", "style"),
            ("dashboard-shell", "className"),
            ("interval-component", "disabled"),
        ],
    )
    assert _input_pairs(spec) == {
        ("user-data-store", "data"),
        ("session-revision", "data"),
    }


def test_page_lifecycle_changes_refresh_session_state(dash_app):
    assert any(
        ("page-lifecycle-store", "data") in _input_pairs(spec)
        and ("session-revision", "data") in _output_pairs(spec)
        for spec in dash_app.callback_map.values()
    )


def test_academy_navigation_wires_final_lesson_button(dash_app):
    assert any(
        ("academy-level-18-button", "n_clicks") in _input_pairs(spec)
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


def test_rendered_lesson_view_emits_scroll_reset(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("view-scroll-sink", "clear_data"),
        ],
    )
    assert _input_pairs(spec) == {
        ("dashboard-shell", "className"),
        ("dashboard-shell", "style"),
        ("lesson-shell", "style"),
        ("simulator-shell", "style"),
    }


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
        build_level_state("level-18")
    )


def test_theme_callback_updates_control_modal_content_classes(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("lesson-intro-modal", "content_class_name"),
            ("academy-cheat-code-modal", "content_class_name"),
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


def test_academy_cheat_code_modal_updates_progress(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("academy-cheat-code-modal", "is_open"),
            ("academy-cheat-code-feedback", "children"),
            ("session-revision", "data"),
        ],
    )
    assert _input_pairs(spec) == {
        ("academy-cheat-code-button", "n_clicks"),
        ("academy-cheat-code-cancel", "n_clicks"),
        ("academy-cheat-code-submit", "n_clicks"),
    }


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
