# Image-Gen Preflight

Run this checklist whenever an **image model** is added or materially changed.
Format and acceptance protocol follow `SOURCE_BANK_PREFLIGHT.md` (the house
pattern): every hard item receives `PASS`, `FAIL`, or an explicitly allowed
`N/A`, plus evidence - a file and line, test name, validator output, or receipt
path. Save an `ID | status | evidence` matrix; the final receipt names that
matrix and its SHA-256. Any hard `FAIL` stops the engine. Machine enforcement
lives in `tests/test_image_gen_preflight_matrix.py`; this document narrates
those checks and is never a substitute for running them.

**Scope - what this file is, and what `VIDEO_LANE_PREFLIGHT.md` already owns.**
`still_flat`, `still_word`, `still_pan` and `still_motion` are **video** lanes:
they sit on `OTR_VideoDirector`'s per-role *video* dropdowns and the video
preflight gates them (operator, 2026-08-21: *"still technically is a video lane
option"*). This file gates the **other** dropdown - the per-role *image* model
that actually mints the picture those lanes hold. It has its own registry
(`nodes/_otr_image_engines`), its own menu, its own cache key and its own
licence posture, and none of that is video.

**The one sentence the whole file serves** (operator, 2026-08-21): *"All video
dropdowns should obey the image gen dropdowns - unless of course viz, there is
no image gen for visualization video models."* The video half of that invariant
is gated at `VIDEO_LANE_PREFLIGHT.md` G3.6; the image half is Gate IG3 below.

---

## Gate IG1 - Declarations are explicit, never a caller's fallback

- **IG1.1 `engine_version` is DECLARED on the adapter.** The still cache key is
  `(role, object_id, prompt_hash, seed, engine_id, engine_version)`, and the
  dispatcher reads the version as `getattr(engine, "engine_version", "1")`. An
  engine that never declares it gets `"1"` from that fallback and can never
  invalidate its own cached stills - the operator would have to change the
  prompt or the seed to get a different picture. Worst on a PARTNER row, where
  the provider can move the model behind a stable id and nothing in this repo
  changes at all.
  *Origin: 2026-08-21 census - seven of eleven registered engines (the six
  `_CloudImageBase` rows plus `google_image`) declared no version. Declaring
  `"1"` explicitly is byte-identical to the fallback they were already getting,
  so the fix moved no cache key; it removed the silence. Note the mechanism is
  real and in use: `z_image_turbo` and `lumina_image` both sit at `"2"`, and the
  z-image bump to `"2"` on 2026-08-21 is what invalidated the gridded v1 stills.*
- **IG1.2 `commercial_clean` is declared as a real bool.** Absent reads as False
  at some call sites and as "unknown" at others, and the honest answer differs
  per row: `flux_gen1` is False (Flux.1-dev is BFL non-commercial) while the
  Apache-2.0 locals are True. A missing declaration lets a non-commercial engine
  pass as clean by omission.
- **IG1.3 `required_inputs` declares `text_prompt`** - the reduced
  prompt -> image contract every still composer writes against.

## Gate IG2 - The registry IS the menu

- **IG2.1 Every registered engine is selectable exactly once.** C4 (2026-06-29)
  removed the validated-subset gate: every registered image engine is
  SELECTABLE and validation is the operator's manual process, never a code gate.
  So the dropdown must be the registry plus the `+ Add Custom Model` sentinel -
  nothing dropped (an engine that exists but cannot be chosen is dead code the
  operator is never told about) and nothing duplicated.
- **IG2.2 Every engine serves every role the menu offers it for.** The three
  slots are not per-role filtered: one unfiltered list is built and used for the
  announcer, music and character slots alike. An engine declaring only two roles
  would still be selectable in the third and would fail at render - after the
  episode had already been written and voiced.
- **IG2.3 One `CAPABILITIES` row per registered engine and vice versa.**
  *Not re-asserted by this file's matrix on purpose - one invariant, one owner:
  it is enforced in `tests/test_capability_profiles.py` and
  `tests/test_cloud_image_adapters.py`.*

## Gate IG3 - The operator's pick is what renders

- **IG3.1 Each role mints on the model picked FOR THAT ROLE.** Three different
  engines in the three slots must come back as three different engines, each
  from its own slot, with `fallback_used` false - no shared default, no silent
  substitution, no leakage of the character pick into the named slots. The
  image-side twin of video G3.6.
- **IG3.2 A named slot that is PRESENT but blank fails LOUD.** Never quietly
  borrow the character model.
  *Origin: the 2026-07-03 no-fallback rip. A silent substitution renders a whole
  episode on a model the operator did not choose, and never mentions it.*
- **IG3.3 The video lane consumes what this engine minted.** Gated from the
  video side by `VIDEO_LANE_PREFLIGHT.md` G3.6 (`accepts_still` declared
  explicitly) and swept by
  `tests/test_still_spine_engine_coverage.py::test_no_video_engine_is_silently_exempt_from_the_image_dropdown`.
  Audited clean 2026-08-21: 27 video lanes obey, the 4 `viz_*` lanes are
  declared exempt, 0 silent.

## Gate IG4 - A refusal is not a render

**NOT YET ENFORCED - no engine declares a refusal classifier. Narrated here
because it is measured, reproducible, and the next image engine will meet it.**

- **IG4.1 An engine whose provider can refuse must CLASSIFY the refusal.**
  Exit status is not evidence. Measured on Ideogram 4, 2026-08-21: a refused
  prompt returns ComfyUI `status: SUCCESS`, a valid non-black PNG, at the exact
  requested dimensions. The only signal is in the pixels - a flat pale card,
  `min > 50 AND std < 15`, where a real render of the same card sits at
  `min ~= 0-1, std ~= 27-33`. File size separates the two just as cleanly here
  (~230-500 KB refused vs 1.2-2.7 MB rendered) but that is a per-model
  coincidence, not a contract.
  *Bible 12.125 (model refusal arriving as a successful render).*
- **IG4.2 A refusal finding names the variable it isolated.** The same probe
  first read as "6/6 refused, the model will not take this content", and that
  reading was wrong: holding the card lines fixed and changing only the PROMPT
  SHAPE (prose composition vs the JSON card schema) moved the result to
  rendered. Whenever an engine is declared unusable, the receipt must say which
  single variable was moved - shape, canvas, aspect field, seed - because
  "refuses Shakespeare" and "refuses this prompt shape" are different findings
  with different fixes.

## Gate IG5 - Prompt-shape ownership

**NOT YET ENFORCED - narrated; there is no declared prompt-shape attribute.**

- **IG5.1 The still composer stays model-agnostic.**
  `compose_still_word_prompt` is contractually model-agnostic - it composes only
  the PROMPT STRING, and whichever image engine the role's slot selects mints
  it. An engine that needs a different prompt SHAPE owns that adaptation on its
  own side of the boundary; it must never push a shape requirement back into the
  shared composer, because the composer serves every engine in the dropdown at
  once.
- **IG5.2 A hygiene negative floor, if the engine has one, is DECLARED.**
  Today none is: `z_image_turbo` keeps `_HYGIENE_NEGATIVE` as a module-private
  constant and `lumina_image` has no floor at all. Any telemetry that wants to
  report the floor must read a declared attribute, never match on engine name.
  *Origin: item H, 2026-08-17 - `negative_source_label` once reported
  `engine_hygiene`, which was accurate for `z_image_turbo` and false for
  `lumina_image` purely by coincidence.*

## Gate IG6 - Live proof

- **IG6.1 One real mint on the engine, through `workflows/otr_canonical.json`,
  reaching `otr/obs/`.** A bench or probe render measures; only the canonical
  path qualifies (`CLAUDE.md` section 0A). Canvas probed, `engine_id` and
  `engine_version` present in the ledger row, `portrait_content_hash` stable
  across a repeat render at the same key.
- **IG6.2 VRAM: OOM or no OOM.** Operator directive, 2026-08-21: *"don't chase
  numbers, please fail OOM only."* Record the peak in the receipt because it is
  free to record; do not gate on a margin.

## Receipt

`IMAGE_GEN_PREFLIGHT receipt: <engine> | <date> | matrix sha256 <...> |
suite run <test output path> | live receipt <path> | verdict PASS/FAIL`

## The family

Siblings live beside this file; each is created when its subsystem is next
touched, backed by its own enforcement code, and never as an empty paper
checklist. `VIDEO_LANE_PREFLIGHT.md` holds the master list.
