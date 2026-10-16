from __future__ import annotations

import re
from threading import Thread

import pytest
from werkzeug.serving import make_server


@pytest.fixture
def browser_page(request, dash_app, monkeypatch):
    if not request.config.getoption("--run-browser"):
        pytest.skip("Pass --run-browser after installing Playwright Chromium.")

    from playwright.sync_api import sync_playwright

    monkeypatch.setenv("IMSIM_CHEAT_UNLOCK_PASSWORD", "")
    server = make_server("127.0.0.1", 0, dash_app.server, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "console",
                lambda message: (
                    errors.append(message.text)
                    if message.type == "error" and "ID not found in layout" in message.text
                    else None
                ),
            )
            page.on(
                "response",
                lambda response: (
                    errors.append(f"HTTP {response.status}: {response.url}")
                    if response.url.startswith(url) and response.status >= 500
                    else None
                ),
            )
            try:
                yield page, url, errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_initial_layout_has_no_missing_callback_targets(browser_page):
    page, url, errors = browser_page
    page.goto(url, wait_until="networkidle")
    page.locator("#academy-level-1-button").wait_for(state="visible")

    assert page.locator("#inventory-table-grid").count() == 0
    assert errors == []


def _assert_grid_readable(page):
    from playwright.sync_api import expect

    grid = page.locator("#inventory-table-grid")
    grid.scroll_into_view_if_needed()
    expect(grid).to_have_css("opacity", "1")
    grid.locator(".ag-center-cols-container .ag-row").first.wait_for(state="visible")
    sizes = grid.evaluate(
        """grid => ({
            rowHeight: grid.querySelector('.ag-center-cols-container .ag-row')
                .getBoundingClientRect().height,
            headerHeight: grid.querySelector('.ag-header').getBoundingClientRect().height
        })"""
    )
    assert sizes["rowHeight"] >= 28, sizes
    assert sizes["headerHeight"] >= 28, sizes


@pytest.mark.parametrize("width", [390, 1440])
def test_planner_grid_has_readable_rows_and_headers(browser_page, width):
    page, url, errors = browser_page
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(url, wait_until="networkidle")
    page.locator("#academy-cheat-code-button").click()
    page.locator("#academy-simulator-button").click()
    _assert_grid_readable(page)

    from playwright.sync_api import expect

    page.locator("#theme-toggle").click()
    expect(page.locator("#inventory-table-grid")).to_have_class(re.compile("ag-theme-quartz-dark"))
    _assert_grid_readable(page)

    initial_day = page.locator("#day-display").inner_text()
    page.locator("#start-button").click()
    expect(page.locator("#day-display")).not_to_have_text(initial_day, timeout=10000)
    page.locator("#start-button").click()
    expect(page.locator("#start-button")).to_have_text("Resume Simulation")

    page.locator("#simulator-return-button").click()
    expect(page.locator("#academy-menu-shell")).to_be_visible()
    page.locator("#academy-level-2-button").click()
    page.locator("#lesson-intro-dismiss-button").click()
    _assert_grid_readable(page)
    assert errors == []


@pytest.mark.parametrize("width,height", [(390, 844), (820, 844), (1440, 844), (1440, 1000)])
def test_github_card_stays_in_viewport_when_scrolling(browser_page, width, height):
    page, url, errors = browser_page
    page.set_viewport_size({"width": width, "height": height})
    page.goto(url, wait_until="networkidle")
    card = page.locator("#gh-footer .footer-card")
    card.wait_for(state="visible")
    before = card.bounding_box()
    assert before["x"] >= 0, before
    assert before["x"] + before["width"] <= width, before
    assert 0 < height - before["y"] - before["height"] <= 32, before
    page.evaluate("window.scrollTo(0, 500)")
    after = card.bounding_box()
    assert abs(after["y"] - before["y"]) <= 1, (before, after)
    card.get_by_role("link", name="GitHub").click(trial=True)
    page.locator("#gh-footer-hide").click()
    from playwright.sync_api import expect

    expect(page.locator("#gh-footer .footer-card")).not_to_be_visible()
    assert errors == []


@pytest.mark.parametrize("width", [390, 820])
def test_lesson_two_cards_have_space_when_stacked(browser_page, width):
    page, url, errors = browser_page
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(url, wait_until="networkidle")
    page.locator("#gh-footer-hide").click()
    page.locator("#academy-cheat-code-button").click()
    page.locator("#academy-level-2-button").click()
    page.locator("#lesson-intro-dismiss-button").click()
    _assert_grid_readable(page)
    from playwright.sync_api import expect

    page.locator("#theme-toggle").click()
    expect(page.locator("#inventory-table-grid")).to_have_class(re.compile("ag-theme-quartz-dark"))
    _assert_grid_readable(page)
    cards = {
        name: page.locator(f"#{name}-panel > .card").bounding_box()
        for name in ["actions", "session", "graph", "inventory", "service"]
    }
    gaps = {
        f"{above}-{below}": cards[below]["y"] - cards[above]["y"] - cards[above]["height"]
        for above, below in [
            ("actions", "session"),
            ("graph", "inventory"),
            ("inventory", "service"),
        ]
    }
    assert all(gap >= 8 for gap in gaps.values()), gaps
    assert errors == []


@pytest.mark.parametrize("width", [390, 570, 590, 820, 1100, 1199])
def test_lessons_use_available_width_and_keep_snapshot_below_inventory(browser_page, width):
    from playwright.sync_api import expect

    page, url, errors = browser_page
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(url, wait_until="networkidle")
    page.locator("#gh-footer-hide").click()
    page.locator("#academy-cheat-code-button").click()
    for level in range(1, 20):
        page.locator(f"#academy-level-{level}-button").click()
        page.locator("#lesson-intro-dismiss-button").click()
        expect(page.locator("#dashboard-shell")).to_have_class(
            re.compile(rf"\blesson-level-{level}\b")
        )
        cards = page.locator(".dashboard-panel > .card").evaluate_all(
            """elements => Object.fromEntries(elements
                .filter(e => e.getBoundingClientRect().height > 0)
                .map(e => { const r = e.getBoundingClientRect();
                    return [e.parentElement.id,
                        {x: r.x, y: r.y, width: r.width, bottom: r.bottom}]; }))"""
        )
        context = (level, width, cards)
        main_width = page.locator(".dashboard-main-column").bounding_box()["width"]
        for name in ["inventory-panel", "service-panel"]:
            if name in cards:
                assert abs(cards[name]["width"] - main_width) <= 2, context
        if "inventory-panel" in cards:
            assert cards["service-panel"]["y"] - cards["inventory-panel"]["bottom"] >= 8, context
        if "costs-panel" in cards and "sales-panel" in cards:
            costs, sales = cards["costs-panel"], cards["sales-panel"]
            if width >= 768:
                assert abs(costs["y"] - sales["y"]) <= 1, context
                assert sales["x"] - costs["x"] - costs["width"] >= 8, context
            else:
                assert sales["y"] - costs["bottom"] >= 8, context
        ordered = sorted(cards.values(), key=lambda card: (card["y"], card["x"]))
        for above, below in zip(ordered, ordered[1:], strict=False):
            if abs(above["y"] - below["y"]) > 1:
                assert below["y"] - above["bottom"] >= 8, context
        page.locator("#return-to-menu-button").click()
        expect(page.locator("#academy-menu-shell")).to_be_visible()
    assert errors == []


@pytest.mark.parametrize("width", [576, 820])
def test_expanded_lesson_snapshot_headers_fit(browser_page, width):
    from playwright.sync_api import expect

    page, url, errors = browser_page
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(url, wait_until="networkidle")
    page.locator("#gh-footer-hide").click()
    page.locator("#academy-cheat-code-button").click()
    page.locator("#academy-level-14-button").click()
    page.locator("#lesson-intro-dismiss-button").click()
    page.locator("#lesson-snapshot-disclosure summary").click()
    expect(page.locator("#lesson-snapshot-disclosure")).to_have_attribute("open", "")
    overflowing = page.locator("#service-panel th").evaluate_all(
        """headers => headers.filter(h => h.scrollWidth > h.clientWidth + 1)
            .map(h => h.textContent)"""
    )
    assert overflowing == []
    assert errors == []


def test_hidden_github_card_can_be_restored_after_reload(browser_page):
    from playwright.sync_api import expect

    page, url, errors = browser_page
    page.goto(url, wait_until="networkidle")
    page.locator("#gh-footer-hide").click()
    expect(page.locator("#gh-footer .footer-card")).not_to_be_visible()
    page.reload(wait_until="networkidle")
    expect(page.locator("#gh-footer .footer-card")).not_to_be_visible()
    page.get_by_role("button", name="Show GitHub card").click(timeout=3000)
    expect(page.locator("#gh-footer .footer-card")).to_be_visible()
    page.reload(wait_until="networkidle")
    expect(page.locator("#gh-footer .footer-card")).to_be_visible()
    assert errors == []


@pytest.mark.parametrize("width", [590, 768, 800, 990, 1440])
def test_simulator_service_badge_stays_inside_card(browser_page, width):
    page, url, errors = browser_page
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(url, wait_until="networkidle")
    page.locator("#gh-footer-hide").click()
    page.locator("#academy-cheat-code-button").click()
    page.locator("#academy-simulator-button").click()
    # Dash may replace service-card children after the simulator becomes visible.
    # Query and measure the current nodes in one browser turn, without a stale handle.
    bounds = page.wait_for_function(
        """() => {
            const card = document.querySelector('#service-panel > .card');
            const badge = card?.querySelector('.badge');
            if (!badge || !badge.checkVisibility()) return false;
            return {
                badge: badge.getBoundingClientRect().toJSON(),
                card: card.getBoundingClientRect().toJSON(),
                overflow: badge.scrollWidth - badge.clientWidth
            };
        }"""
    ).json_value()
    assert bounds["badge"]["right"] <= bounds["card"]["right"], bounds
    assert bounds["overflow"] <= 1, bounds
    assert errors == []


@pytest.mark.parametrize("width", [390, 570, 575, 590])
def test_academy_cards_and_actions_fit_their_containers(browser_page, width):
    page, url, errors = browser_page
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(url, wait_until="networkidle")
    overflowing = page.locator(
        "#academy-menu-shell .metric-card, "
        "#academy-menu-shell .academy-card .imsim-button, "
        "#academy-menu-shell .academy-card-formula"
    ).evaluate_all(
        """elements => elements.filter(e => {
            const bounds = e.getBoundingClientRect();
            const parent = e.parentElement.getBoundingClientRect();
            return bounds.right > parent.right + 1 || bounds.left < parent.left - 1;
        }).map(e => e.textContent)"""
    )
    assert overflowing == []
    assert errors == []
