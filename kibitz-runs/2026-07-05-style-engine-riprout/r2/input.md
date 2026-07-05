# OTR style engine consolidation — 100% rip-out plan v2 (post-r1 kibitz)

Operator directive (2026-07-05, verbatim intent): the story should have ONE
internal engine driving the plot/style. Too many things are influencing it
today. This plan is a full rip-out, not a migration: **no fallback paths, no
back-compat shims, no trace of the retired system left in runtime code, no
negative/back-compat tests.** Companion doc:
`docs/2026-07-05-style-dropdown-blast-radius/ANALYSIS.md` (root-cause map of
the current 4 disconnected style-slug surfaces).

This is v2 -- r1 kibitz (Codex, gpt-5.5 high reasoning; Antigravity did not
respond and was dropped for r1) already changed this plan materially: cut a
stale ~90-cue MusicGen authoring requirement, and surfaced a real shipped
widget (`story_scaffold`) the original draft missed entirely. r1 judgment:
`kibitz-runs/2026-07-05-style-engine-riprout/r1/final.md`.

## 0. Hard constraints (non-negotiable per operator)

- One engine only: `nodes/_otr_style_catalog.py` (`STYLE_CATALOG` /
  `build_story_contract` / `select_style`) becomes the SOLE source of both
  the tone/style label AND the climax shape + sound world. Today these are
  two independent draws feeding the same prompt -- that duplication is
  itself the bug being killed.
- No fallback. No "if the new path fails, revert to the old list" shim.
- No trace. Delete the retired modules/constants outright — not deprecate,
  not comment out, not rename with a `_legacy` suffix.
- No negative tests. Delete tests whose sole purpose was pinning the OLD
  system's behavior; write only positive pins on the new single-engine
  behavior.

## 1. Target end-state

- `style_custom` (free-text) is NO LONGER a silent bypass. r1 kibitz
  grounding confirmed it currently wins outright in `_resolve_inputs`
  (`OTR_LedgerScriptWriter.py:1345-1353`) while `build_story_contract()`
  still runs its own independent catalog draw (lines 3340-3344) — free
  text tone + an unrelated catalog grammar landing on the same episode.
  DECISION REQUIRED before build: either (a) `style_custom` becomes a
  label override that still produces one `StoryContract` (custom tone
  string, catalog-default grammar/climax for that draw), or (b) it is
  retired from this cleanbreak entirely.
- `style` combo — repopulated from `_otr_style_catalog.all_slugs()` /
  labels (100 real entries, replacing the old 10) plus exactly one sentinel
  ("let the story decide" or equivalent). Picking a specific catalog entry
  pins that slug; picking the sentinel (or leaving default) runs
  `build_story_contract()` ONCE and its `.label` becomes the tone string,
  its `.grammar`/`.story_engine` becomes the climax/sound-world injection —
  same call, same contract, single draw, both consumers. NOTE: today's
  `_otr_style_catalog.build_story_contract()` always calls `select_style()`
  internally (`_otr_style_catalog.py:754-762`) — there is no `forced_slug`
  path yet. The catalog needs a `from_slug`/`forced_slug` entry point so an
  explicit combo pick can build a `StoryContract` from a pinned slug instead
  of the hash draw; sentinel keeps calling the deterministic draw.
- `resolved["style"]` derivation collapses from three branches
  (`style_custom` > `style_combo` > `llm_auto`/two-pass-LLM-invent) to two
  (`style_custom` > catalog engine).
- Ledger field canonicalization: `meta.gen_params_initial` currently stamps
  four sibling fields (`style`, `style_combo`, `style_custom`,
  `style_source` — confirmed live at `OTR_LedgerScriptWriter.py:5505-5508`)
  and the freeze validator enforces snake_case shape on `style`
  (`_otr_ledger_freeze.py`). Pick ONE canonical field for the catalog SLUG
  (snake_case, freeze-validated) and a separate field for the human-
  readable LABEL if one is still needed.

## 1a. `story_scaffold` widget — a real, already-shipped third control

r1 kibitz grounding surfaced a widget this plan initially missed entirely:
`story_scaffold` (`OTR_LedgerScriptWriter.py:2244-2260`, combo
`["auto","on","off"]`, added 2026-06-24, appended at the end of `optional`
per the BUG-LOCAL-097 positional convention). `_apply_story_scaffold_env`
(line 1710) mutates `OTR_ENABLE_STYLE_GRAMMAR` straight from this widget.
Its own tooltip says `off` = "a story drawn straight from the news seed...
no style catalog, no climax-shape grammar, no grounding gate -- the
writer's own take." That is a real, user-facing SECOND STORY MODE, not an
abstract env-var escape hatch. DECISION REQUIRED before build:
  (a) keep `story_scaffold` as an intentional, documented creative option
      (a symmetric "scaffold off" mode is a legitimate distinct story
      style, not a silent fallback), or
  (b) delete it along with the rest of the retired duality, making the
      catalog engine mandatory on every render.
Whichever is chosen must also cover the widget's positional slot in the
workflow JSON.

## 2. Delete outright (zero trace)

- `nodes/_otr_style_picker.py` — the whole file (2-pass LLM inventor:
  Pass 1 candidate invention, Pass 2 chooser, `StyleGenerationFailedError`,
  `StylePick`, `pick_style`).
- `OTR_LedgerScriptWriter.py`: `_STYLE_CHOICES`, `_STYLE_PICKER_SEED_POOL`,
  `_LLM_STYLE_FALLBACK`, the `pick_style(...)` call site (~line 2995) and
  its surrounding RNG plumbing (`_resolve_style_rng_seed`, `picker_rng`) if
  nothing else calls them, `meta["style_pick"]` stamp, the stale NOTE
  comment block (~lines 806-815) that narrates the now-doubly-dead
  `_generate_style_via_llm` ancestor.
- Any `style_pending` / `llm_auto` branch in `_resolve_inputs`.
- Grep sweep (must return zero hits before declaring done) across BOTH
  `nodes/` AND `tests/` (r1 kibitz grounding found the picker referenced in
  seven test files, not the one or two originally assumed):
  `_otr_style_picker`, `pick_style`, `StylePick`, `StyleGenerationFailedError`,
  `_STYLE_PICKER_SEED_POOL`, `_LLM_STYLE_FALLBACK`, `style_pick`.
  Confirmed test-side referencers to fold into this sweep:
  `tests/test_otr_style_picker.py`, `tests/test_pick_style_routing.py`,
  `tests/test_helper_paired_signatures.py`, `tests/test_audio_byte_identical.py`,
  `tests/test_story_pack_stage1.py`, `tests/test_writer_paired_wiring.py`,
  `tests/test_meta_slot_transitions.py`.

## 3. MusicGen palette — CUT (r1 finding: the premise was stale)

`compose_music_prompt()` (`nodes/_otr_music_prompt.py:76-99`) builds every
cue prompt from `meta` brief fields (`story_brief_terms`, `music_mood_terms`,
keyword-mined `script_brief`) — it does not read `STYLE_PALETTE` at all. A
grep of `nodes/*.py` for `STYLE_PALETTE`/`_otr_style_palette` returns only
the palette's own file. `_otr_style_palette.py` is dead relative to runtime,
kept alive only by its own `tests/test_style_palette_drift.py`.

Revised scope: NO cue-authoring work. Decide whether `_otr_style_palette.py`
+ `test_style_palette_drift.py` are deleted outright as part of "no trace
of the retired system" — verify-at-build: grep for any dynamic/string-keyed
access to `STYLE_PALETTE` before deleting.

## 4. Workflow JSON (positional widget)

- `workflows/otr_scifi_16gb_full.json`: the `style` combo's allowed-choices
  list can change freely (dropdown metadata, not a positional widget_value),
  but `widgets_values[8]`'s frozen DEFAULT must still resolve correctly
  under the new engine. If the sentinel string changes, `widgets_values[8]`
  must be updated in the SAME change (CLAUDE.md section 0).
- If section 1a resolves to removing `story_scaffold`, its positional slot
  (appended per BUG-LOCAL-097 -- a later widgets_values index than [8]) must
  also be audited/removed in the same change.
- Re-validate after edit: `OTR_WorkflowValidator` + JSON round-trip +
  widget-count-vs-INPUT_TYPES audit + link referential integrity.

## 5. Ledger / meta schema

- `meta.story_contract` already exists and is the correct single record —
  keep as-is. `_otr_ledger_consistency.py`'s existing
  `MatrixRow("style", "contract.slug", "ledger.meta.story_contract.slug")`
  is unaffected by this rip (confirmed).
- `meta.style_pick` (old picker stamp) is deleted along with the picker.
  Audit `_otr_ledger_consistency.py` and any other reader of
  `meta.style_pick` before deleting.
- Already-rendered episodes on disk keep their historical
  `meta.style_pick` / old-slug `meta.style` values untouched — this rip
  targets runtime code, not archived output.

## 6. (superseded by 1a — kept as a pointer only)

## 7. Sequencing — one atomic cleanbreak sprint

No staged dual-system interim state (this repo's standing cleanbreak rule).
Order within the single sprint:

1. Resolve section 1 (`style_custom`) and section 1a (`story_scaffold`)
   decisions FIRST — operator/panel judgment calls, not mechanical work.
2. Add `_otr_style_catalog`'s `forced_slug`/`from_slug` entry point so an
   explicit combo pick builds a `StoryContract` without the hash draw.
3. Rewire the widget combo + `_resolve_inputs` + the single
   `build_story_contract()` call site to be the one and only style/climax
   source; canonicalize the ledger slug/label fields.
4. Delete the dead modules/constants (section 2, `nodes/` + `tests/`) —
   confirm the grep sweep is clean. Decide + execute the
   `_otr_style_palette.py` fate (section 3).
5. Rewrite tests — positive pins only.
6. Re-validate + re-freeze the workflow JSON (section 4).
7. Full regression suite + Bug Bible, per CLAUDE.md, after the whole
   chunk — not after each sub-step.
8. Commit AND push to `v2.0-alpha` in the same session (CLAUDE.md section 7).

## 8. Risk / blast radius

- Positional-widget drift risk on the workflow JSON (BUG-LOCAL-097 class),
  now covering TWO widgets (`style` and, if retired, `story_scaffold`).
- C7 determinism: confirm the single-draw contract stays
  cast_seed-keyed/reproducible exactly like today's `select_style`.
- Sweep `docs/`, `kibitz-runs/`, dashboards for mentions of the deleted
  symbols — archival text only.
- Sections 1 and 1a must be resolved before build starts, not discovered
  mid-sprint.

## 9. Explicit asks for this round (r2 -- coding plan / implementability)

1. Give the CONCRETE function signature for the new
   `_otr_style_catalog.forced_slug`/`from_slug` entry point and how
   `build_story_contract()`'s call sites (widget path vs sentinel path)
   should branch to it vs the hash draw.
2. Is `resolved["style"]` (a plain string used elsewhere for logging /
   prompts) still the right shape once the "canonical slug field" decision
   in section 1 lands, or does downstream code need a richer type?
3. Concretely: what is the exact `_resolve_inputs` branch structure once
   `style_custom` / `style` combo / catalog engine collapse to two paths --
   write the pseudocode, not just the prose description in section 1.
4. Any other code-level implementability gap in sections 1-7.

---

COMFYUI CUSTOM-NODE PROFILE (append to each round prompt)

When the target repo is a ComfyUI custom-node pack, also verify the
domain invariants below. Cite the real node file/class for every claim; if you
cannot see the code, write "verify: <what>" rather than asserting it. These are
ComfyUI-specific failure modes the general rounds do not weight.

1. NODE-CLASS CONTRACT
   - Every exported node class is registered in NODE_CLASS_MAPPINGS (and given a
     label in NODE_DISPLAY_NAME_MAPPINGS). A class that is defined but not mapped
     is dead -- it never loads. Flag any node in the plan that is not wired into
     the mappings.
   - INPUT_TYPES is a @classmethod returning a dict with "required" (and optional
     "optional"/"hidden") keys. Each input is (TYPE, {options}) where TYPE is a
     real type string ("IMAGE", "LATENT", "MODEL", "CONDITIONING", "STRING",
     "INT", "FLOAT", ...) or a list-of-choices for a dropdown.
   - RETURN_TYPES is a tuple (note the trailing comma for a single output),
     length-matched to what FUNCTION actually returns; RETURN_NAMES if present
     must be the same length. CATEGORY and FUNCTION must be set.
   - Widget order is POSITIONAL: appending an optional input is safe; inserting
     one mid-list silently shifts every saved widget value in existing graphs.

2. TENSOR LAYOUT / SHAPE CONVENTIONS
   - ComfyUI IMAGE tensors are float32 in [0,1], shape [B, H, W, C] (channels
     LAST), C usually 3. MASK is [B, H, W]. LATENT is a dict {"samples": tensor}
     with the model's own channel layout (commonly [B, C, H, W]). Flag any node
     that assumes channels-first for IMAGE or forgets the batch dim.
   - Check dtype/device handling: tensors may arrive on cuda or cpu; a node must
     not hard-assume one. Verify .to(device)/.cpu() moves are correct and that
     outputs match the declared RETURN_TYPES layout.

3. VRAM / MODEL MANAGEMENT
   - Heavy models should load through comfy.model_management (residency, offload,
     and eviction are managed there) rather than being pinned in module globals.
     Flag any plan that holds a model resident across runs without an eviction or
     free path, or that bypasses model_management's load/offload.
   - Verify large allocations are freed (or handed to model_management) so a
     long ComfyUI session does not leak VRAM across queued prompts.

4. IS_CHANGED / CACHING CORRECTNESS
   - ComfyUI caches a node's output and re-runs only when inputs change. If a
     node depends on external state (a file on disk, a clock, RNG, a network
     fetch), it must implement IS_CHANGED to return a value that varies when that
     state changes -- otherwise it serves stale cached output. Flag any node with
     hidden external inputs and no IS_CHANGED, and any IS_CHANGED that is more
     conservative/looser than the node's real dependencies.

5. IMPORT ISOLATION (no heavy imports at module top)
   - The module top level is imported at ComfyUI startup. Heavy or optional
     dependencies (torch extras, model libraries, CUDA ext) imported at top level
     slow every boot and hard-crash startup if missing. Move them inside the
     node method (lazy import) so an unrelated missing dep cannot take down the
     whole node pack. Flag top-level imports of optional/heavy packages.
   - Side effects at import time (downloading weights, opening files, mutating
     global state) are a defect -- they run for every user on every boot.
