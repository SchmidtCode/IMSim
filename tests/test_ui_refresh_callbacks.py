from __future__ import annotations


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


def test_theme_callback_updates_control_modal_content_classes(dash_app):
    spec = _find_callback(
        dash_app,
        [
            ("lesson-intro-modal", "content_class_name"),
            ("add-item-modal", "content_class_name"),
            ("place-custom-order-modal", "content_class_name"),
            ("po-overview-modal", "content_class_name"),
        ],
    )
    assert _input_pairs(spec) == {("theme-store", "data")}


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
