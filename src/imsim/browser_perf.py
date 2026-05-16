from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BrowserPerfOptions:
    url: str
    contexts: tuple[int, ...]
    duration_seconds: float
    headless: bool
    password: str
    json_output: Path | None
    progress: bool
    progress_interval_seconds: float


@dataclass(frozen=True, slots=True)
class BrowserContextResult:
    context_count: int
    duration_seconds: float
    pages_started: int
    pages_failed: int
    total_day_updates: int
    mean_day_updates: float
    max_update_gap_ms: float
    p95_update_gap_ms: float
    mean_frame_gap_ms: float
    p95_frame_gap_ms: float
    console_errors: int
    page_errors: int
    http_errors: int
    max_used_js_heap_mb: float | None
    sample_errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BrowserPerfResult:
    options: BrowserPerfOptions
    runs: tuple[BrowserContextResult, ...]


def run_browser_performance(options: BrowserPerfOptions) -> BrowserPerfResult:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Playwright is not installed. Run `uv sync --group dev` first.") from exc

    runs: list[BrowserContextResult] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=options.headless)
        try:
            for context_count in options.contexts:
                _progress(options, f"starting batch with {context_count} browser context(s)")
                runs.append(
                    _run_context_batch(
                        browser,
                        options,
                        context_count,
                        PlaywrightTimeoutError,
                    )
                )
                _progress(options, f"finished batch with {context_count} browser context(s)")
        finally:
            browser.close()
    return BrowserPerfResult(options=options, runs=tuple(runs))


def _run_context_batch(browser, options, context_count: int, timeout_error) -> BrowserContextResult:
    contexts = []
    pages = []
    setup_errors: list[str] = []
    page_error_counts: dict[int, int] = {}
    console_error_counts: dict[int, int] = {}
    http_error_counts: dict[int, int] = {}
    console_samples: list[str] = []

    for index in range(context_count):
        _progress(options, f"opening context {index + 1}/{context_count}")
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()
        contexts.append(context)
        pages.append(page)
        page_error_counts[index] = 0
        console_error_counts[index] = 0
        http_error_counts[index] = 0
        page.on("pageerror", _count_page_error(index, page_error_counts, console_samples))
        page.on("console", _count_console_error(index, console_error_counts, console_samples))
        page.on("response", _count_http_error(index, http_error_counts, console_samples))

    ready_pages = []
    for index, page in enumerate(pages):
        try:
            _progress(options, f"preparing simulator page {index + 1}/{context_count}")
            _prepare_simulator_page(page, options, timeout_error)
            ready_pages.append(page)
        except Exception as exc:
            setup_errors.append(f"page {index}: {type(exc).__name__}: {exc}")

    started_pages = 0
    for page in ready_pages:
        try:
            if _start_simulation_with_retries(page, timeout_error):
                started_pages += 1
        except Exception as exc:
            setup_errors.append(f"start: {type(exc).__name__}: {exc}")

    _sample_for_duration(options, context_count, started_pages)

    telemetry = []
    for page in pages:
        try:
            telemetry.append(page.evaluate("() => window.__imsimPerf || {}"))
        except Exception as exc:
            setup_errors.append(f"telemetry: {type(exc).__name__}: {exc}")

    for context in contexts:
        context.close()

    return _summarize_browser_batch(
        context_count=context_count,
        duration_seconds=options.duration_seconds,
        pages_started=started_pages,
        setup_errors=setup_errors,
        telemetry=telemetry,
        console_error_counts=console_error_counts,
        page_error_counts=page_error_counts,
        http_error_counts=http_error_counts,
        console_samples=console_samples,
    )


def _progress(options: BrowserPerfOptions, message: str) -> None:
    if not options.progress:
        return
    print(f"[imsim-browser-perf] {message}", file=sys.stderr, flush=True)


def _sample_for_duration(
    options: BrowserPerfOptions,
    context_count: int,
    started_pages: int,
) -> None:
    started = time.monotonic()
    deadline = started + options.duration_seconds
    _progress(
        options,
        (
            f"sampling {started_pages}/{context_count} started page(s) for "
            f"{options.duration_seconds:.0f}s"
        ),
    )
    next_progress = started + max(1.0, options.progress_interval_seconds)
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        if now >= next_progress:
            remaining = max(0.0, deadline - now)
            _progress(options, f"{remaining:.0f}s remaining in current batch")
            next_progress = now + max(1.0, options.progress_interval_seconds)
        time.sleep(min(1.0, max(0.0, deadline - now)))


def _count_page_error(index: int, counts: dict[int, int], samples: list[str]):
    def handler(error) -> None:
        counts[index] += 1
        if len(samples) < 10:
            detail = getattr(error, "stack", None) or getattr(error, "message", None) or str(error)
            samples.append(f"page {index} error: {detail}")

    return handler


def _count_console_error(index: int, counts: dict[int, int], samples: list[str]):
    def handler(message) -> None:
        if message.type not in {"error", "warning"}:
            return
        counts[index] += 1
        if len(samples) < 10:
            samples.append(f"page {index} console {message.type}: {message.text}")

    return handler


def _count_http_error(index: int, counts: dict[int, int], samples: list[str]):
    def handler(response) -> None:
        if response.status < 500:
            return
        counts[index] += 1
        if len(samples) >= 10:
            return
        detail = ""
        try:
            text = response.text()
            detail = f" body={text[:500].replace(chr(10), ' ')}"
        except Exception:
            detail = ""
        samples.append(f"page {index} http {response.status}: {response.url}{detail}")

    return handler


def _prepare_simulator_page(page, options: BrowserPerfOptions, timeout_error) -> None:
    page.goto(options.url, wait_until="networkidle", timeout=30000)
    page.locator("#academy-cheat-code-button").click(force=True, timeout=10000)
    page.locator("#academy-cheat-code-input").fill(options.password, timeout=10000)
    page.locator("#academy-cheat-code-submit").click(timeout=10000)
    _wait_for_modal_to_close(page, timeout_error)
    page.wait_for_function(
        "() => document.querySelector('#academy-simulator-button')"
        " && !document.querySelector('#academy-simulator-button').disabled",
        timeout=10000,
    )
    _open_simulator_with_retries(page, timeout_error)
    page.wait_for_selector("#day-display", timeout=15000)
    _install_browser_telemetry(page)

    _enable_auto_po_best_effort(page)


def _enable_auto_po_best_effort(page) -> None:
    try:
        auto_po = page.locator("#auto-po-enabled")
        if auto_po.count() > 0 and not auto_po.is_checked():
            try:
                auto_po.check(force=True, timeout=5000)
            except Exception:
                page.evaluate(
                    """
                    () => {
                      const input = document.querySelector("#auto-po-enabled");
                      if (input && !input.checked) {
                        input.click();
                      }
                    }
                    """
                )
        page.locator("#update-params-button").click(timeout=5000)
    except Exception:
        return


def _wait_for_modal_to_close(page, timeout_error) -> None:
    try:
        page.locator(".modal.show").wait_for(state="hidden", timeout=10000)
        return
    except timeout_error:
        pass
    try:
        page.keyboard.press("Escape")
        page.locator(".modal.show").wait_for(state="hidden", timeout=3000)
    except timeout_error:
        pass


def _open_simulator_with_retries(page, timeout_error) -> None:
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            button = page.locator("#academy-simulator-button")
            button.scroll_into_view_if_needed(timeout=5000)
            button.click(timeout=5000)
        except Exception as exc:
            last_error = exc
            try:
                page.evaluate(
                    """
                    () => {
                      const button = document.querySelector("#academy-simulator-button");
                      if (button) {
                        button.click();
                      }
                    }
                    """
                )
            except Exception as eval_exc:
                last_error = eval_exc
        try:
            page.wait_for_function(
                """
                () => {
                  const dashboard = document.querySelector("#dashboard-shell");
                  if (!dashboard) {
                    return false;
                  }
                  const style = window.getComputedStyle(dashboard);
                  return style.display !== "none" && dashboard.offsetParent !== null;
                }
                """,
                timeout=10000,
            )
            return
        except timeout_error as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    page.wait_for_selector("#dashboard-shell", state="visible", timeout=15000)


def _start_simulation_with_retries(page, timeout_error) -> bool:
    if page.locator("#start-button").count() <= 0:
        return False
    last_error: Exception | None = None
    for _attempt in range(3):
        _wait_for_modal_to_close(page, timeout_error)
        try:
            button = page.locator("#start-button")
            button.scroll_into_view_if_needed(timeout=5000)
            button.click(timeout=5000)
        except Exception as exc:
            last_error = exc
            try:
                page.evaluate(
                    """
                    () => {
                      const button = document.querySelector("#start-button");
                      if (button) {
                        button.click();
                      }
                    }
                    """
                )
            except Exception as eval_exc:
                last_error = eval_exc
        try:
            page.wait_for_function(
                """
                () => {
                  const status = document.querySelector("#sim-status")?.textContent || "";
                  return status.includes("Running");
                }
                """,
                timeout=5000,
            )
            return True
        except timeout_error as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return False


def _install_browser_telemetry(page) -> None:
    page.evaluate(
        """
        () => {
          const target = document.querySelector("#day-display");
          const perf = {
            startedAt: performance.now(),
            dayText: target ? target.textContent : "",
            dayUpdates: [],
            frameGaps: [],
            memorySamples: [],
          };
          window.__imsimPerf = perf;
          let lastDayAt = performance.now();
          let lastFrameAt = performance.now();
          if (target) {
            const observer = new MutationObserver(() => {
              const now = performance.now();
              const text = target.textContent || "";
              if (text !== perf.dayText) {
                perf.dayUpdates.push({ at: now, gap: now - lastDayAt, text });
                perf.dayText = text;
                lastDayAt = now;
              }
            });
            observer.observe(target, { childList: true, subtree: true, characterData: true });
          }
          function frame(now) {
            perf.frameGaps.push(now - lastFrameAt);
            if (perf.frameGaps.length > 1200) {
              perf.frameGaps.shift();
            }
            lastFrameAt = now;
            if (performance.memory) {
              perf.memorySamples.push(performance.memory.usedJSHeapSize);
              if (perf.memorySamples.length > 120) {
                perf.memorySamples.shift();
              }
            }
            window.requestAnimationFrame(frame);
          }
          window.requestAnimationFrame(frame);
        }
        """
    )


def _summarize_browser_batch(
    *,
    context_count: int,
    duration_seconds: float,
    pages_started: int,
    setup_errors: list[str],
    telemetry: list[dict[str, Any]],
    console_error_counts: dict[int, int],
    page_error_counts: dict[int, int],
    http_error_counts: dict[int, int],
    console_samples: list[str],
) -> BrowserContextResult:
    update_counts = [len(page.get("dayUpdates") or []) for page in telemetry]
    update_gaps = [
        float(update.get("gap") or 0.0)
        for page in telemetry
        for update in (page.get("dayUpdates") or [])
    ]
    frame_gaps = [
        float(gap) for page in telemetry for gap in (page.get("frameGaps") or []) if float(gap) > 0
    ]
    memory_samples = [
        float(sample) for page in telemetry for sample in (page.get("memorySamples") or [])
    ]
    sample_errors = tuple([*setup_errors, *console_samples][:10])
    return BrowserContextResult(
        context_count=context_count,
        duration_seconds=duration_seconds,
        pages_started=pages_started,
        pages_failed=max(0, context_count - pages_started),
        total_day_updates=sum(update_counts),
        mean_day_updates=statistics.fmean(update_counts) if update_counts else 0.0,
        max_update_gap_ms=max(update_gaps) if update_gaps else 0.0,
        p95_update_gap_ms=_percentile(update_gaps, 95),
        mean_frame_gap_ms=statistics.fmean(frame_gaps) if frame_gaps else 0.0,
        p95_frame_gap_ms=_percentile(frame_gaps, 95),
        console_errors=sum(console_error_counts.values()),
        page_errors=sum(page_error_counts.values()),
        http_errors=sum(http_error_counts.values()),
        max_used_js_heap_mb=(max(memory_samples) / (1024.0 * 1024.0) if memory_samples else None),
        sample_errors=sample_errors,
    )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def result_to_dict(result: BrowserPerfResult) -> dict[str, Any]:
    return {
        "options": {
            **asdict(result.options),
            "contexts": list(result.options.contexts),
            "json_output": None
            if result.options.json_output is None
            else str(result.options.json_output),
        },
        "runs": [asdict(run) for run in result.runs],
    }


def format_result(result: BrowserPerfResult) -> str:
    lines = [
        "IMSim browser performance run",
        f"URL: {result.options.url}",
        f"Duration: {result.options.duration_seconds:.1f}s",
        f"Headless: {result.options.headless}",
    ]
    for run in result.runs:
        memory = "n/a" if run.max_used_js_heap_mb is None else f"{run.max_used_js_heap_mb:.2f} MiB"
        lines.extend(
            [
                "",
                f"Contexts: {run.context_count}",
                f"Pages started/failed: {run.pages_started}/{run.pages_failed}",
                f"Day updates: total {run.total_day_updates}, mean {run.mean_day_updates:.1f}",
                (
                    "Update gaps: "
                    f"p95 {run.p95_update_gap_ms:.1f}ms, max {run.max_update_gap_ms:.1f}ms"
                ),
                (
                    "Frame gaps: "
                    f"mean {run.mean_frame_gap_ms:.1f}ms, p95 {run.p95_frame_gap_ms:.1f}ms"
                ),
                f"Console errors/warnings: {run.console_errors}",
                f"Page errors: {run.page_errors}",
                f"HTTP 5xx errors: {run.http_errors}",
                f"Max JS heap: {memory}",
            ]
        )
        if run.sample_errors:
            lines.append("Sample errors: " + " | ".join(run.sample_errors[:3]))
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small real-browser IMSim simulator responsiveness check."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8050")
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Browser context batch sizes to run sequentially.",
    )
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--headed", action="store_true", help="Show the browser window.")
    parser.add_argument(
        "--password",
        default=os.environ.get("IMSIM_CHEAT_UNLOCK_PASSWORD", "spreadsheets rule"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between progress messages while sampling.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    options = BrowserPerfOptions(
        url=args.url.rstrip("/"),
        contexts=tuple(max(1, count) for count in args.contexts),
        duration_seconds=max(1.0, float(args.duration)),
        headless=not args.headed,
        password=str(args.password),
        json_output=args.json_output,
        progress=not args.quiet,
        progress_interval_seconds=max(1.0, float(args.progress_interval)),
    )
    result = run_browser_performance(options)
    if options.json_output:
        output_path = options.json_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result_to_dict(result), indent=2), encoding="utf-8")
    print(format_result(result))


if __name__ == "__main__":
    main()
