# Problem statement: Old-Time Radio templates are hard to find in ComfyUI

Written 2026-10-29 at the LA ComfyUI build night, on the 4060 (MRKT).

## The problem

The operator remembers finding the Old-Time Radio workflows by searching ComfyUI's
Templates window. Tonight, on this laptop, searching does not find them. They can only
be reached by scrolling the Templates window's left-hand list to **EXTENSIONS → Old-Time
Radio**. A person who does not already know to look there will conclude the pack ships
no templates.

The operator's hypothesis: the templates are searchable in the **Comfy Desktop app**
and not in the **plain web UI**.

## What was measured tonight

- **Setup:** ComfyUI 0.43.1, frontend `comfyui_frontend_package` 1.57.4,
  `comfyui_workflow_templates` 0.11.98, Python 3.13.12.
- **How it was opened:** the plain web UI in Chrome at `http://127.0.0.1:8188`. ComfyUI was
  started by `D:\demo_night\1_start_comfyui.cmd` with `--disable-all-custom-nodes` and a
  whitelist of four packs (OldTimeRadio, LogBlaster, AnimateDiff-Evolved, LTXVideo), and
  without `--enable-manager`. This is not how the Desktop app launches ComfyUI.
- **The server side works:** `GET /api/workflow_templates` returns 30 workflows under the key
  `ComfyUI-OldTimeRadio` (`otr_8gb_still`, `otr_canonical`, ...).
- **The browse list works:** the Templates window's left column, read from the page, ends
  `... Node Basics, Partner Nodes, EXTENSIONS, Old-Time Radio`. The label comes from
  `DisplayName = "Old-Time Radio"` in `pyproject.toml`.
- **Search does not work:**
  - With **All** selected, "old-time Radio" and "comfyui-old-time-radio" both return only
    built-in templates (SeedVR2, MiniMax H3, FLUX.2 Klein, ...).
  - With **App** selected, "comfyui-old-time-radio" shows "0 of 13 templates".
  - No OTR workflow appears in any search result.
- **No registry search:**
  - Uninstalled packs never appear. The Templates window lists only installed extensions.
  - The Extensions menu item (C menu → Extensions) only toggles frontend extensions; it
    does not search or install packs.
  - The Manager is not reachable in this launch, because ComfyUI was started without
    `--enable-manager`, although `comfyui-manager` 4.2.2 is installed.

## Settled for the web UI: search finds the filename, not the pack name

Checked in two ways on 2026-10-29: by reading the frontend source, and by searching the live
web UI here (All selected, filters cleared). Desktop has not been checked.

- **Searching by filename works.**
  - `otr_canonical` gives "Showing 1 of 679 templates" (otr_canonical).
  - `otr_` gives "Showing 30 of 679 templates", which is all 30 workflows.
- **Searching by pack name does not.** `old-time radio` returns only built-in templates
  (SeedVR2, MiniMax H3, FLUX.2 Klein, ...).
- **Why, in the source.** The frontend's build manifest matches upstream `ComfyUI_frontend`
  at `8d7bb487c2740a3a771d65faff19fcfc30147fab`.
  - An extension workflow's `name`, `title` and `description` are all its filename. The pack
    name is kept only in `sourceModule` and `searchableText`
    (`src/platform/workflow/templates/repositories/workflowTemplatesStore.ts` L257-272).
  - Search indexes `title`, `description`, `tags`, `models` and `name`, and leaves out both
    fields that hold the pack name (`packages/shared-frontend-utils/src/templateSearch.ts`
    L22-28).
- **The App tab's "0 of 13" is a separate filter.** That tab requires `isApp: true`, which
  extension entries never have (`src/platform/workflow/templates/utils/templateDisplay.ts`
  L20-30), so its zero proves nothing about search.

## Hypotheses still open

1. **Settled for the web UI.** Extension workflows are found by searching their filename;
   searching the pack's name does not find them. The Desktop app is not yet checked.
2. **The Desktop app ships a different frontend, or a different search.** The Desktop app
   bundles its own frontend build. If its version or search code differs, the operator's
   memory is of that build.
3. **Search used to work and a frontend update dropped it.** Earlier frontends may have
   indexed extension templates. Then "we used to find it" is a regression on Comfy-Org's
   side, not ours.
4. **The demo launch flags change what the window offers.** This is unlikely, since the
   EXTENSIONS entry is present, but `--disable-all-custom-nodes` and the missing
   `--enable-manager` are both non-standard.

## How to settle it

- [ ] In the **Comfy Desktop app** on this laptop, normally launched, open Templates with
      All selected and filters cleared. Search `old-time radio`, `otr_canonical` and
      `otr_8gb_still`, and record the frontend version from Settings → About.
- [ ] In the **web UI** with a normal launch (no whitelist, `--enable-manager`), run the
      same three searches.
- [x] Read the template-search code in `ComfyUI_frontend` at 1.57.4 (done 2026-10-29; see
      above).
- [ ] If search used to index extension templates, find the frontend version that stopped,
      and decide whether to report it to Comfy-Org.

## What done looks like

A one-line answer to "can a user find Old-Time Radio's templates by searching inside
ComfyUI?", for both the Desktop app and the web UI, with versions.

- **Web UI 1.57.4:** yes by filename (`otr_`), no by pack name.
- **Either way,** the README can tell people to search `otr_` or to browse to Templates →
  EXTENSIONS → Old-Time Radio.
- **On Comfy-Org's side:** leaving `searchableText` out of the search index looks like a
  frontend gap worth reporting to them. Ours is to report, not to fix.

Out of scope: discovery without installing the pack. The Templates window only lists
installed packs. Appearing for uninstalled users would mean getting into Comfy-Org's own
`workflow_templates` library, which is a separate decision.
