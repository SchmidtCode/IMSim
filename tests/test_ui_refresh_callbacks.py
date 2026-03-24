from __future__ import annotations


def _output_pairs(spec):
    outputs = spec["output"]
    if not isinstance(outputs, list):
        outputs = [outputs]
    return {(output.component_id, output.component_property) for output in outputs}


def _input_pairs(spec):
    return {
        (input_spec["id"], input_spec["property"])
        for input_spec in spec["inputs"]
    }


def _find_callback(dash_app, required_outputs):
    required = set(required_outputs)
    for spec in dash_app.callback_map.values():
        if required.issubset(_output_pairs(spec)):
            return spec
    raise AssertionError(f"Callback with outputs {sorted(required)} not found")


def test_refresh_driven_panels_listen_to_ui_refresh(dash_app):
    for component_id in (
        "kpi-strip",
        "inventory-table-shell",
        "exception-center-shell",
    ):
        spec = _find_callback(dash_app, [(component_id, "children")])
        assert _input_pairs(spec) == {("ui-refresh", "data")}


def test_state_changes_emit_ui_refresh(dash_app):
    cases = [
        [
            ("day-display", "children"),
            ("inventory-graph", "figure"),
            ("service-card", "children"),
            ("costs-card", "children"),
            ("sales-card", "children"),
        ],
        [
            ("day-display", "children"),
            ("asq-apply-feedback", "children"),
            ("interval-component", "disabled"),
        ],
        [
            ("user-data-store", "data"),
            ("asq-apply-feedback", "children"),
        ],
        [
            ("add-item-modal", "is_open"),
            ("add-item-error", "children"),
        ],
        [("update-params-conf", "children")],
        [("upload-feedback", "children")],
        [
            ("custom-order-items-div", "children"),
            ("place-custom-order-modal", "is_open"),
        ],
        [
            ("po-overview-modal", "is_open"),
            ("po-overview-table", "children"),
        ],
    ]

    for required_outputs in cases:
        spec = _find_callback(dash_app, required_outputs)
        assert ("ui-refresh", "data") in _output_pairs(spec)
