# IMSim Agent Instructions

IMSim is a Plotly Dash inventory management training app. Preserve the existing feature set unless a task explicitly asks for product changes.

## Project Commands

- Run the app: `uv run imsim`
- Run tests: `uv run pytest`
- Run lint: `uv run ruff check .`
- Format Python: `uv run ruff format .`

If `uv` cannot write to the default cache in a sandboxed environment, set `UV_CACHE_DIR=tmp/uv-cache`.

## UI Implementation Notes

- Primary UI files are `src/imsim/ui/layout.py`, `src/imsim/ui/components.py`, and `assets/imsim.css`.
- Dash Bootstrap Components are already used. Prefer existing helpers and class names over introducing a new UI framework.
- Plotly visual defaults live in `src/imsim/ui/components.py`; table and grid styling lives mostly in `assets/imsim.css`.
- Keep CSS token-driven. Add or adjust custom properties in `:root` before sprinkling one-off values.

## Design Direction

Use `DESIGN.md` as the source of truth for redesign work. The target is a production-ready operations training tool: precise, calm, practical, and crafted. Avoid generic AI dashboard styling such as excessive card grids, soft gradient blobs, oversized rounded containers, vague helper copy, and decorative visual noise.

When changing UI, verify at desktop and mobile widths. Important actions should be discoverable from the running app without reading the README.
