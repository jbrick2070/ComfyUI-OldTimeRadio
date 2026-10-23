# Docs, README and the app form, against ComfyUI's own practice

Written 2026-10-23 from the pack as it stands at 3dd8516c, the Comfy Registry
specification, the custom-node help-page and i18n pages at docs.comfy.org,
and the frontend's App Mode source (comfyui_frontend 1.55.18 is what this
machine runs; the builder components are in `src/components/builder/`). Every
claim below was read from one of those, never remembered.

## What a stranger sees, in order

1. **The registry card** (registry.comfy.org, and Node Manager's list):
   `DisplayName`, `description`, `Icon`, `Banner`, the version list. The
   README is NOT rendered there -- only on GitHub.
2. **The GitHub README**, reached by "View Repository".
3. **The canvas**: node titles ("1 - Story Writer"), the Start here note.
4. **The App Mode form** (`extra.linearData`): one flat list of rows. Each
   row shows `widget.label || widget.name` and, under it, ONE truncated line
   of `config.description`. Nothing else: the `tooltip` from INPUT_TYPES is
   not drawn in App Mode (`AppModeWidgetList.vue`, `WidgetDescription.vue`
   -- the description prop or an i18n placeholder, `truncate`).
5. **The node help panel** (the "?" on a node): with no `docs/` folder it
   shows `DESCRIPTION` and the per-input tooltips. With
   `WEB_DIRECTORY/docs/<NodeClass>.md` (and `<NodeClass>/<locale>.md`) it
   shows that page instead. The frontend here fetches exactly those paths.
6. **The guides**: every `apple/*.md` not named in `.comfyignore` ships in
   the zip and is linked from the README's guide table.

## Findings, each with the practice it fails and the fix

### A. The registry card (pyproject.toml -- a release trigger, so one batch at the next bump)

| Finding | The practice | Fix |
|---|---|---|
| `DisplayName = "ComfyUI-OldTimeRadio"` | The spec says a name should not carry "ComfyUI"; `DisplayName` is the user-friendly name. The template category already says "Old-Time Radio". | `DisplayName = "Old-Time Radio"`. The id `comfyui-old-time-radio` cannot change (a new listing). |
| `Icon` is 800x400 | The spec: Icon is square, at most 400x400; `Banner` is 21:9. The pyproject comment still says "max 800x400", an older rule. | A 400x400 icon: `assets/otr_icon_400.png`, drawn by `tools/make_square_icon.py` from the card's own parts -- the title in its typeface, the magenta rule, the signal wave, the grid floor -- so it reads at the registry's row size (a centre crop of the card cut the title in half, and a letterbox read as a placeholder). The hand-finished square gallery art (`assets/otr_gallery_thumb_master.webp`) is the alternative, busier at 48 px. `Banner = assets/otr_banner.jpg` (1672x717 -- already 21:9); fix the comment. The exact edit is `pyproject_next_bump.diff` beside this plan. |
| `banner_url` empty on the live listing | Icon and Banner are "displayed on the ComfyUI Registry and ComfyUI-Manager". | As above. |
| `description` opens "Press Queue once" | ComfyUI's button is **Run**; the README already says Run. | "Press Run once". |
| `classifiers` name no OS or GPU | The spec: OS and GPU classifiers "help users find nodes for their systems". | Add `Operating System :: Microsoft :: Windows`, `:: MacOS`, `:: POSIX :: Linux`, `Environment :: GPU :: NVIDIA CUDA`. |
| `[project.urls]` has no `Documentation` | The spec's example lists `Documentation` and `"Bug Tracker"`. | `Documentation = <README on GitHub>`; keep `Issues`. |

Not changed: the version string. 2.3.36 and 2.3.37 were published today and
the API still resolves `latest_version` to 2.3.35 (Flagged), so the next bump
is the operator's call, as always.

### B. The App Mode form (config/app_mode.json -> every generated workflow)

| Finding | The practice | Fix |
|---|---|---|
| 26 of 40 rows have no description, and the tooltip is not drawn in App Mode | The one line under a row is the only help a form user gets. | A note (30 characters or fewer -- the row truncates) on every row. **Done below.** |
| Rows carry the operator's order, no sections | The frontend has no grouping; the role prefixes in `labels` ("Announcer - video") already do that job. | Keep. |
| My Story text rows open at the default height | `InputWidgetConfig.height` persists a textarea height. | Proposed: `height` on the four My Story rows and the photos row, via one more optional column in `form`. Needs build_variants and its tests. |

### C. The node help panel (js/docs/)

| Finding | The practice | Fix |
|---|---|---|
| No `js/docs/` -- the panel shows DESCRIPTION plus tooltips, some of them 1,000-1,800 characters (the writer's `episode_language`, `source_bank`, both writer models) | Per-node markdown under `WEB_DIRECTORY/docs/<NodeClass>.md`, locale variants beside it. | A generator (`scripts/build_node_docs.py`) writes one page per node from the live INPUT_TYPES: what the node is for, which guide to read, a table of inputs (label, choices, default, first sentence of the tooltip), the full tooltips below, the outputs. Hand-written intros and guide links live in `config/node_docs.json`; a test regenerates and diffs, as the dependency doc's test does. **Done below.** |
| Display names carry build tags: "LPL Script Writer (v2.0)", "MasterAudioMux (terminal mux-LAST, -c:a copy)", "Video Render Batch (A-S7.5 in-process render)", "Workflow Validator (opt-in, S14.2)", "CaptionBurn (SDH open captions on the silent video; default-OFF)" | ComfyUI's own nodes use plain names; the tags are commit history. Saved workflows key on the class name, so a rename breaks nothing. `tests/test_naming_conventions.py` pins the current strings. | Proposed, for the operator: "Script Writer", "Master Audio Mux", "Video Render Batch", "Workflow Validator", "Caption Burn"; categories unified under `OldTimeRadio/<stage>` (today a mix of `OldTimeRadio`, `OldTimeRadio/v2`, `OldTimeRadio/v2/audio`, `OldTimeRadio/diagnostics`). |
| Node names and tooltips are English only, while the pack writes episodes in eight languages and the Studio page speaks nine | `locales/<lang>/nodeDefs.json` translates display names, input names, tooltips and choice labels. | Proposed, later: generate `nodeDefs.json` skeletons from the same generator and translate the display names and labels first (short strings), the tooltips never -- the help pages carry those, with `<NodeClass>/<locale>.md` when a language is worth it. |

### D. The README (GitHub only)

| Finding | The practice | Fix |
|---|---|---|
| The hero -- card art, badges, the episode still and "Hear a complete episode" -- sits under the five install steps and three paragraphs of prose | Title, one line on what it makes, the proof (still + link), then install. | Move the hero block under the H1. **Done below.** |
| 944 lines; "Install and run it" (90) and "Make an episode" (187) both install the pack, and "The widgets worth touching", "When something goes wrong", "The engines" each have a twin in a shipped guide | A README states, a guide explains; one home per topic. No paragraph is duplicated verbatim today, so this is overlap of topic, not copy. | Proposed, for the operator: cut "Make an episode" to its machine table plus a pointer to apple/RUN.md, "API keys" to a pointer to apple/CLOUD.md, "A workflow pre-set for your machine" to its table plus apple/MACHINES.md, "When something goes wrong" to the five most common lines plus apple/RUN.md. Roughly a 500-line README. His prose; his call. |

### E. The shipped guides (apple/ and .comfyignore)

| Finding | The practice | Fix |
|---|---|---|
| These ship in the zip and are not user guides: CONTEST_DEMO_SCRIPT, CONTEST_DEMO_STORYBOARD, CONTEST_EXPLAINER_PROMPT, CONTEST_SUBMISSION_DRAFT, RESEARCH_PROMPT_TTS_KILLER, SPEC_pack_owned_vram_residency, RIGHTS_DECISION_LEMMY_VOICE, FLUX3_KLEIN_LICENSE_ATTESTATION and model-license-comfy-org--gemma-4 (their siblings are already excluded) | `.comfyignore` names what is internal; the README's guide table is the shipped set. | Exclude them. **Done below.** CONTEST_RADIO_PLAY_STUDIO.md stays: it describes the app that ships in `contest/`. ARCHITECTURE.md and SHAKESPEARE_CORPUS_SPEC.yaml stay: one is a reader's map, the other a schema. |

## Build order

1. **Now, one push:** the form notes (B), the node help pages and their test (C), the ignore lines (E), the README hero (D). Mechanical, one right answer each. Then Codex, briefed to refute, on the pushed diff.
2. **At the next version bump, with the operator:** the pyproject batch (A), as one commit, since the file fires a publish.
3. **Proposed, his call:** display names and categories; the README trim; row heights; `nodeDefs.json` for the eight languages.
