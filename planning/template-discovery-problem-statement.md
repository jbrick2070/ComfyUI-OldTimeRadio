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

## Hypotheses, to test in this order

1. **Search covers built-in templates only.** In frontend 1.57.4, extension templates can be
   browsed but are not indexed by search. This fits everything above, and would mean the
   Desktop app behaves the same way.
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

- [ ] In the **Comfy Desktop app** on this laptop, normally launched, open Templates and
      search "old-time radio". Record the frontend version from Settings → About.
- [ ] In the **web UI** with a normal launch (no whitelist, `--enable-manager`), run the
      same search.
- [ ] Read the template-search code in `ComfyUI_frontend` at 1.57.4 and see which template
      sets it indexes.
- [ ] If search used to index extension templates, find the frontend version that stopped,
      and decide whether to report it to Comfy-Org.

## What done looks like

A one-line answer to "can a user find Old-Time Radio's templates by searching inside
ComfyUI?", for both the Desktop app and the web UI, with versions. If the answer is no,
the README and the stage line should send people to Templates → EXTENSIONS → Old-Time
Radio. If it is a Comfy-Org regression, the report should go to them.

Out of scope: discovery without installing the pack. The Templates window only lists
installed packs. Appearing for uninstalled users would mean getting into Comfy-Org's own
`workflow_templates` library, which is a separate decision.
