# Open Design Workflow

This repo can use Open Design as a local design exploration layer for the Dash UI while keeping IMSim's implementation in Python, Dash, Plotly, AG Grid, and CSS.

## Current No-Admin Setup

For this workspace, Open Design was set up without a system Node install:

- Portable Node: `tmp/tools/node-v24.16.0-win-x64`
- Open Design checkout: `tmp/open-design`
- Open Design project-location root: `tmp/od-projects`
- Corepack cache: `tmp/corepack`
- Open Design logs: `tmp/open-design-tools-dev.*.log`
- IMSim logs: `tmp/imsim.*.log`

These paths are under `tmp/` and are not intended to be committed.

## Start Open Design

From the IMSim repo root:

```powershell
$env:PATH = "C:\GitHub\IMSim\tmp\tools\node-v24.16.0-win-x64;" + $env:PATH
$env:COREPACK_HOME = "C:\GitHub\IMSim\tmp\corepack"
Set-Location C:\GitHub\IMSim\tmp\open-design
corepack.cmd pnpm tools-dev
```

Open the web URL printed by the command. In the initial setup it was:

```text
http://127.0.0.1:61784
```

The port can change between runs.

## Project Location Gotcha

Do not add `C:\GitHub\IMSim` itself as an Open Design project location while this no-admin setup is in use. Open Design stores its daemon data in `C:\GitHub\IMSim\tmp\open-design\.od`, and it rejects project-location roots that overlap daemon data.

Use this folder instead:

```text
C:\GitHub\IMSim\tmp\od-projects
```

That folder is only for Open Design's own generated project workspaces. IMSim's real source still lives at `C:\GitHub\IMSim`; point prompts at the live app URL and the repo files rather than making the repo root the Open Design storage root.

## Start IMSim

In a second terminal:

```powershell
Set-Location C:\GitHub\IMSim
uv run imsim
```

Then open:

```text
http://127.0.0.1:8050
```

## First Design Prompt

Use this with Open Design/Codex after IMSim is running:

```text
Review the live IMSim Dash app at http://127.0.0.1:8050 using DESIGN.md as the product and visual brief.

Focus on academy, simulator, settings/import, and mobile widths. Produce 2-3 concrete visual directions for making the app feel like a crafted operations training console rather than a generic AI dashboard.

Do not replace the Dash stack. Recommend changes that can be implemented in assets/imsim.css, src/imsim/ui/layout.py, and src/imsim/ui/components.py. Prioritize spacing, hierarchy, card reduction, simulator discovery, button grouping, chart/table density, and README screenshot readiness.
```

After choosing a direction, ask Codex to implement one focused pass at a time and verify the running app at desktop and mobile widths.
