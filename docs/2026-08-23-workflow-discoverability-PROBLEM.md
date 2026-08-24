# Problem statement: the deliverable is a workflow, but the install only surfaces nodes

**Date:** 2026-08-23
**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha` (published to the Comfy Registry as
`comfyui-old-time-radio`, publisher `fluxus`)
**Status:** open design question -- no code written for the contested part

---

## 1. The operator's complaint, in his words

> "wow i have all these nodes -- where the friggin workflow"
> "the nodes live in the backend, that's not a good user experience, what do we do"

He is right, and it is the whole product question. OTR is not a parts bin. A user does
not want 34 `OTR_` nodes to wire together; they want **one graph they queue and walk
away from**, which then writes, performs, masters and renders a radio-drama episode.

## 2. What actually happens today (verified, not assumed)

Verified on a clean ComfyUI Desktop v0.33.3 install on a second machine (RTX 4060),
installing from the registry by name:

* The pack installs and **all 34 nodes register** -- console prints
  `[OldTimeRadio] OK - All 34 nodes loaded successfully`, and `/object_info` lists 34
  `OTR_` classes.
* The workflow **does** ship and **is** served. ComfyUI v0.33.3 scans a pack for any of
  `["example_workflows", "example", "examples", "workflow", "workflows"]`
  (`custom_node_manager.py:94`); our `workflows/` qualifies. The live endpoint returns
  `{"comfyui-old-time-radio": ["otr_canonical", "otr_story_only"]}` and the pack folder
  is statically mounted at `/api/workflow_templates/comfyui-old-time-radio/`.
* So the graph is reachable at: **Workflow -> Browse Templates -> scroll the left
  sidebar to EXTENSIONS -> click `comfyui-old-time-radio` -> pick `otr_canonical`.**

**That is five clicks behind a menu the user has no reason to open, with no signal
anywhere that an episode graph exists at all.** Nothing in the node menu, nothing on
the canvas, nothing in the UI says "start here". The pack looks like a parts bin
because on first contact it behaves like one.

## 3. What has already been done (shipped, uncontested, not up for debate)

These were cheap, safe and are already published -- do NOT spend rounds re-deciding them:

1. **Startup banner pointer** (`__init__.py`, after the load banner): prints
   `[OldTimeRadio] Load the show:  Workflow > Browse Templates > EXTENSIONS >
   comfyui-old-time-radio > otr_canonical`.
2. **README** now leads with that same path in a callout near the top, and install
   step 4 names the Browse Templates route first (file-drag demoted to a fallback).
3. **Template thumbnail** (`workflows/otr_canonical.jpg`, shipped in `2.0.0-alpha.5`)
   so the template card is a real preview instead of a placeholder gradient.
4. **Registry card art** -- an animated GIF of the pipeline, so the listing reads as a
   product rather than a library.

These remove *"I did not know it existed"*. **They do not reduce the five clicks.**

## 4. The actual question for the panel

**What is the right way for a ComfyUI custom-node pack whose deliverable is a WORKFLOW
(not a node library) to put that workflow in front of a first-time user -- and is a
frontend extension worth the maintenance and risk?**

The candidate under consideration:

> Set `WEB_DIRECTORY` and ship a small `web/otr.js` that calls
> `app.registerExtension(...)` to add a toolbar/menu command -- **"Load Old Time Radio
> Episode"** -- which fetches
> `/api/workflow_templates/comfyui-old-time-radio/otr_canonical.json` and hands it to
> `app.loadGraphData()`. One click from a cold install to a runnable episode graph.

**Constraints and non-negotiables (from this repo's operating rules):**

* `workflows/otr_canonical.json` is the **single source of truth** for the graph. Any
  solution that creates a second copy of the graph (in git or in the bundle) is
  rejected -- copies drift. Note the pack must also NOT add an `example_workflows/`
  folder alongside `workflows/`: ComfyUI globs all five accepted names and extends one
  list, so a pack with both gets its templates **double-listed** and its static mounts
  collide (verified in `custom_node_manager.py`).
* The pack is offline-first. No telemetry, no network calls, no CDN assets.
* Failure must be loud and contained: a broken frontend extension must never prevent
  the 34 nodes from registering or break the canvas for an unrelated workflow.
* The `__init__.py` per-node isolated-import design (each node in its own try/except)
  is deliberate and stays.

**Specific things to pressure-test:**

1. **Is a frontend extension the right call at all**, or is it over-engineering for a
   problem that documentation plus the template thumbnail already mostly solves? Argue
   the "ship nothing more" case honestly if it is the strong one.
2. **API stability.** `app.registerExtension`, `loadGraphData`, and the toolbar/menu
   registration surface are frontend APIs that have churned across ComfyUI versions.
   What is the least version-fragile way to add a command in current ComfyUI, and what
   breaks on upgrade? What is the graceful-degradation story when the API moves?
3. **Where should the affordance live** -- toolbar button, the Workflow menu, a sidebar
   tab, a canvas context-menu entry, or a node-menu category header? Which is most
   discoverable to someone who just installed and does not know what OTR is, and which
   is most likely to annoy an experienced user?
4. **Is auto-loading on first run ever acceptable?** Current position: no -- hijacking
   someone's canvas uninvited is worse than five clicks. Challenge that if there is a
   defensible middle (e.g. a one-time dismissible toast that offers to load it).
5. **Prior art.** Which existing ComfyUI packs solve "my pack IS a workflow" well, and
   what did they actually do? Name real packs and mechanisms, not hypotheticals.
6. **What does this cost forever?** Shipping a `web/` directory means owning frontend
   code against a moving target, on a project whose author does not primarily write JS.
   Is the one-click win worth that ongoing tax, and is there a lower-maintenance shape
   that gets 80% of the benefit?

## 5. Evidence and where to read it

* `__init__.py` -- node registration, the load banner, the new pointer line, and the
  existing HTTP route registrations (`/otr/latest_ledger`, the render routes) which
  prove the pack already extends the server safely with fail-closed guards.
* `workflows/otr_canonical.json` -- the graph in question (~74 KB, the canonical
  source of truth per `CLAUDE.md` section 0).
* `README.md` -- current first-run instructions.
* `CLAUDE.md` sections 0, 7 and 7A -- source-of-truth rule, git policy, and the
  registry publishing contract (including why `example_workflows/` must not be added).
* ComfyUI's own `custom_node_manager.py:94` -- the accepted template folder names.
