# Unreleased

## Fixed

- Fix the missing `inventory-table-grid` callback target reported when the Academy first
  loads. Dash now validates callbacks against a layout that includes the dynamic planner grid.
- Keep planner, custom-order, and PO overview rows readable when their grids first mount
  inside hidden panels. Explicit row and header heights prevent zero-height measurements,
  including at phone widths and after switching themes.
- Restore the GitHub project card to the bottom-right of the viewport while scrolling,
  including mobile and shorter desktop windows. Its link and Hide control remain interactive.
  A Show GitHub card button restores the card after hiding it, including across reloads.
- Separate stacked Lesson 2 cards below the desktop breakpoint.
- Fix narrow inventory and snapshot cards, overlapping card edges, and snapshot ordering
  in lesson layouts below 1200 pixels, including phone widths. Lesson snapshots now
  follow inventory, and expanded snapshot headings wrap without overlapping.
- Place Costs and Sales side by side at 768-1199 pixels in lessons that show them; stack them
  at smaller widths. Full desktop lesson layouts retain their existing arrangement.
- Keep Academy summary cards, lesson buttons, and formula labels inside their containers
  at phone widths instead of clipping their right edges.
- Allow the simulator Service card's availability pill to wrap inside its column at
  intermediate widths.

## Development

- Add Chromium browser regression coverage for callback validation, grid visibility,
  theme changes, simulation controls, the floating GitHub card, and responsive lessons.
- Run browser regressions in CI. Local setup and commands are documented in
  [the development guide](../development.md#browser-regression-tests).

These changes address the two UI issues listed in the [v1.1.1 release notes](v1.1.1.md).
