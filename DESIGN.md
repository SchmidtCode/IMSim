# IMSim Design Brief

## Product Frame

IMSim is an inventory management academy and simulator for people who need to learn replenishment decisions by doing. The UI should feel like a serious operations workspace that can still teach: clear, responsive, readable, and confident.

The app is primarily Plotly Dash, Dash Bootstrap Components, Plotly charts, and AG Grid. Redesign work should improve the current app rather than replacing the stack.

## Visual Thesis

Make IMSim feel like a crafted operations console, not a generated dashboard template.

The product should read as:

- precise rather than decorative
- calm rather than flashy
- instructional without being wordy
- dense enough for operations work, but not cramped
- polished enough for public reviewers in the first minute

## Target Screens

Review and polish these surfaces together:

- Academy landing and lesson cards
- Simulator dashboard
- Settings and import flows
- Modals for add items, custom orders, PO overview, reference, and academy override
- Public first-run and simulator unlock/discovery path
- README screenshots and GIF capture state

## Core UX Goals

- Public reviewers should understand the value quickly: guided academy first, simulator unlock path second, full dashboard payoff third.
- Important actions should be easy to find without reading documentation.
- The simulator should feel discoverable even while locked behind academy progression.
- Empty states, alerts, helper text, and labels should use the same voice and density.
- Desktop should support scanning and comparison; mobile should stack cleanly without awkward text wrapping.

## Layout Principles

- Use page sections and workspace regions before adding cards.
- Avoid cards inside cards. If a repeated item needs framing, keep it light and purposeful.
- Use consistent spacing: small gaps for related controls, larger gaps between functional regions.
- Make controls line up on a stable grid. Avoid one-off widths unless the content demands it.
- Keep chart and table regions visually connected to their controls.
- Make modals feel like focused workspaces, not small pages embedded in dialogs.

## Component Principles

- Buttons should have a clear hierarchy: primary action, secondary action, quiet utility.
- Group related controls with labels that can be scanned quickly.
- Use compact, consistent control heights.
- Use icons only when they clarify a repeated or familiar action; do not use icons as decoration.
- Alerts and empty states should be useful and brief.
- Tables should prioritize legibility, alignment, row density, and hover/read state.
- Charts should use a restrained palette and readable margins; avoid default-looking Plotly output.

## Color And Type

Current tokens live in `assets/imsim.css`. Keep the palette grounded and operational. Avoid letting the app become dominated by one hue family, especially beige, dark blue, teal, or purple.

Use accent color to communicate action, state, or signal. Do not use color as background decoration without function.

Typography should favor clarity. Large display type belongs only in the first-view academy or simulator entry areas; panels, cards, controls, and tables should use compact headings.

## Dash-Specific Guidance

- Prefer shared CSS classes over inline style dictionaries for repeated visual behavior.
- Keep Dash component structure understandable; do not add wrappers just for styling if CSS can target the existing structure.
- For Plotly figures, update the shared layout helpers before styling individual charts.
- For AG Grid, adjust theme variables and grid options consistently.
- Preserve callback IDs unless a feature change requires otherwise.

## Open Design Workflow

Use Open Design as the exploration and critique layer, then implement the chosen direction in Dash.

Suggested loop:

1. Capture screenshots of academy, simulator, settings/import, and mobile widths.
2. Ask Open Design/Codex for 2 or 3 design directions using this file as the brief.
3. Choose one direction and convert it into tokens, CSS rules, layout adjustments, and Plotly/table styling.
4. Run the app locally and verify desktop and mobile screenshots.
5. Update README screenshots/GIFs after the UI stabilizes.

## Acceptance Checklist

- Spacing feels consistent across academy, simulator, settings, and import views.
- No obvious nested-card clutter or card-heavy repetition.
- Primary actions and simulator discovery are visible without documentation.
- Public demo path makes simulator value clear within the first few minutes.
- Desktop and mobile layouts avoid text overlap and awkward wrapping.
- Charts and tables feel integrated with the surrounding UI.
- README screenshots/GIFs show the polished UI.
