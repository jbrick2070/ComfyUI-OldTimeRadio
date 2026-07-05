# OTR style engine consolidation — 100% rip-out plan (draft for kibitz)

Operator directive (2026-07-05, verbatim intent): the story should have ONE
internal engine driving the plot/style. Too many things are influencing it
today. This plan is a full rip-out, not a migration: **no fallback paths, no
back-compat shims, no trace of the retired system left in runtime code, no
negative/back-compat tests.** Companion doc:
`docs/2026-07-05-style-dropdown-blast-radius/ANALYSIS.md` (root-cause map of
the current 4 disconnected style-slug surfaces).

Status: DRAFT, unbuilt. Going through kibitz (Codex + Antigravity local
panel, Cowork Claude as anchor+judge) before any code is touched.

## 0. Hard constraints (non-negotiable per operator)

- One engine only: `nodes/_otr_style_catalog.py` (`STYLE_CATALOG` /
  `build_story_contract` / `select_style`) becomes the SOLE source of both
  the tone/style label AND the climax shape + sound world. Today these are
  two independent draws feeding the same prompt (see ANALYSIS.md) — that
  duplication is itself the bug being killed.
- No fallback. No "if the new path fails, revert to the old list" shim.
  No env kill-switch that reverts to a DIFFERENT engine (see open question
  in section 8 on whether `OTR_ENABLE_STYLE_GRAMMAR` itself must go).
- No trace. Delete the retired modules/constants outright — not deprecate,
  not comment out, not rename with a `_legacy` suffix.
- No negative tests. Delete tests whose sole purpose was pinning the OLD
  system's behavior; write only positive pins on the new single-engine
  behavior. Do not add tests that assert "the old thing is gone" as a
  permanent regression guard — that is not what this repo's test suite is
  for, and it's dead weight the moment the rip lands.

## 1. Target end-state

- `style_custom` (free-text, verbatim, highest precedence) — UNCHANGED,
  already the correct escape hatch.
- `style` combo — repopulated from `_otr_style_catalog.all_slugs()` /
  labels (100 real entries, replacing the old 10) plus exactly one sentinel
  ("let the story decide" or equivalent). Picking a specific catalog entry
  pins that slug; picking the sentinel (or leaving default) runs
  `build_story_contract()` ONCE and its `.label` becomes the tone string,
  its `.grammar`/`.story_engine` becomes the climax/sound-world injection —
  same call, same contract, single draw, both consumers.
- `resolved["style"]` derivation collapses from three branches
  (`style_custom` > `style_combo` > `llm_auto`/two-pass-LLM-invent) to two
  (`style_custom` > catalog engine, where "catalog engine" covers both the
  explicit-pick and the auto/sentinel case).

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
- Grep sweep (must return zero hits before declaring done):
  `_otr_style_picker`, `pick_style`, `StylePick`, `StyleGenerationFailedError`,
  `_STYLE_PICKER_SEED_POOL`, `_LLM_STYLE_FALLBACK`, `style_pick`.

## 3. MusicGen palette — the real cost of "no fallback"

`_otr_style_palette.py` `STYLE_PALETTE` currently covers only the old 10
slugs. With one engine now capable of returning any of 100 slugs as the
canonical style, EVERY one of the 100 needs a real opening/closing/
interstitial cue triple — a fallback/default cue would violate "no
fallback." This is content-authoring work (100 x 3 cue prompts), not a
mechanical change, and is very likely the single largest line item in this
sprint. Flag explicitly for the kibitz panel to size.

`tests/test_style_palette_drift.py` repinned so `KNOWN_STYLE_SLUGS` /
`STYLE_PALETTE.keys()` == `set(_otr_style_catalog.all_slugs())`. Delete the
old 10-slug fixture and its test bodies outright, not skip/xfail.

## 4. Workflow JSON (positional widget)

- `workflows/otr_scifi_16gb_full.json`: the `style` combo's allowed-choices
  list can change freely (it's dropdown metadata, not a positional
  widget_value), but `widgets_values[8]`'s frozen DEFAULT must still
  resolve correctly under the new engine. Confirm whether the sentinel
  string itself is kept identical ("let the story decide") or renamed —
  if renamed, `widgets_values[8]` must be updated in the SAME change (per
  this repo's CLAUDE.md section 0: JSON change lands with the code change,
  never a follow-up).
- Re-validate after edit: `OTR_WorkflowValidator` + JSON round-trip +
  widget-count-vs-INPUT_TYPES audit + link referential integrity.

## 5. Ledger / meta schema

- `meta.story_contract` already exists and is the correct single record —
  keep as-is.
- `meta.style_pick` (old picker stamp) is deleted along with the picker.
  Audit `_otr_ledger_consistency.py` and any other reader of
  `meta.style_pick` before deleting.
- Already-rendered episodes on disk keep their historical
  `meta.style_pick` / old-slug `meta.style` values untouched — this rip
  targets runtime code, not archived output. BUG_LOG.md's historical
  entries (BUG-LOCAL-216, -240, -270, etc.) stay as archival record; they
  describe past incidents, not live code paths.

## 6. Config lever — open question, not yet decided

`_otr_config.style_grammar_enabled()` (env `OTR_ENABLE_STYLE_GRAMMAR`)
currently lets an operator turn the catalog engine OFF, reverting the
climax-shape selection to the old default-only behavior
(`irreversible_choice` always). Once there is only ONE engine, is an "off"
switch itself a forbidden fallback? Two readings:
  (a) "No fallback" means no fallback to a DIFFERENT selection mechanism —
      the kill-switch merely disables the grammar injection feature
      entirely (a simpler, degraded, but still-single-engine state), which
      is arguably fine and worth keeping for triage.
  (b) "No fallback" is absolute — the lever itself is the thing being
      ripped out, since its OFF state is exactly the two-tier
      (old-default/new-engine) duality the operator wants gone.
This is a genuine fork for the kibitz panel + operator judgment, not
something to guess at silently.

## 7. Sequencing — one atomic cleanbreak sprint

Per this repo's standing cleanbreak rule ("no runtime gates inside
cleanbreak sprints; each cleanbreak sprint is the LAST one"): no staged
dual-system interim state. Order within the single sprint:

1. Author the ~90 missing MusicGen cue triples (content work; gates
   everything downstream — nothing else lands without it, per "no
   fallback").
2. Rewire the widget combo + `_resolve_inputs` + the single
   `build_story_contract()` call site to be the one and only style/climax
   source.
3. Delete the dead modules/constants (section 2) — confirm the grep sweep
   is clean.
4. Rewrite tests (section 3 + section 4's guardrail tests) — positive
   pins only.
5. Re-validate + re-freeze the workflow JSON (section 4).
6. Full regression suite + Bug Bible, per CLAUDE.md, after the whole
   chunk — not after each sub-step.
7. Commit AND push to `v2.0-alpha` in the same session (CLAUDE.md section 7).

## 8. Risk / blast radius (carried from ANALYSIS.md + new)

- Positional-widget drift risk on the workflow JSON (BUG-LOCAL-097 class).
- Content-authoring risk: 90 new cue triples is real creative writing, not
  boilerplate — easy to under-scope, and "no fallback" means it can't ship
  partial.
- C7 determinism: confirm the single-draw contract stays
  cast_seed-keyed/reproducible exactly like today's `select_style`.
- Sweep `docs/`, `kibitz-runs/`, dashboards for mentions of the deleted
  symbols — archival text only, not a blocker, but worth a pass so a
  future reader isn't pointed at dead code.
- Section 6's open question must be resolved before build starts, not
  discovered mid-sprint.

## 9. Explicit asks for the kibitz panel

1. Is collapsing to ONE `build_story_contract()` call (tone + climax +
   sound-world from the same draw) actually sound, or does tying the
   user's explicit hard-pick (a specific catalog slug) to the SAME
   grammar-injection path risk a regression the current split design was
   accidentally protecting against?
2. Section 6: does the `OTR_ENABLE_STYLE_GRAMMAR` lever survive?
3. Is authoring 90 cue triples in-scope for this sprint, or does "100%
   rip-out, no fallback" force it to be (i.e., can MusicGen ship with a
   narrower catalog subset instead of the full 100, to avoid blocking the
   code rip on unrelated content work) — pressure-test whether that would
   itself be a smuggled-back fallback.
4. Anything else that reads `meta.style`, `meta.style_pick`, or the old
   10-slug shape that this plan hasn't found yet.

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
