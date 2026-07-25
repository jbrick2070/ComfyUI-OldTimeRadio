# R1 -- the LEAN architecture for per-model still operations

**Operator directive 2026-07-25 (overnight, verbatim intent): "run a new R1 so
we get a good lean, clean architecture."** This is a HIGH-LEVEL ARC round, not
a defect sweep. Written by CODER WINDOW A at HEAD `8403ab58` + the landed S1b
chunk. Claude is a grounded panelist and the sole judge.

Prior rounds: r1-r5 (converged, 2026-07-25 @ `84328aa1`), r3 ordering panel,
r4 defect sweep (`docs/2026-07-25-still-plans-r4-judgment.md`). Spec of record
`docs/2026-07-25-still-plans-locked-build-spec.md`.

## Why this round exists

The plan is ACCUMULATING CORRECTIONS rather than converging. Tally on one
architecture: r3 found three must-fixes in the S0b chunk's own spec plus four
omitted env-read sites; r4 codex found ten more, one of which was a
MISDECLARED row in already-landed code; and this window found two the panel
missed. agy called the structure converged; codex did not. That pattern -- a
structure that keeps needing exceptions -- is the question this round asks.

**The specific smell.** `still_plan_helpers.VALID_STYLE_TAIL_POLICIES` is a
CLOSED enum of two tokens, `full` and `minimal_clean`. Production has a THIRD
behaviour: `build_radio_host_prompt`'s `ltx_radio_mouth` branch
(`nodes/otr_meta_brief_image_prompt.py:394-401`) returns EARLY with
`"%s, warm dramatic lighting"`, skipping both `finish_visual_prompt(...,
era_profile="still")` and the `image_grade_tail` append -- deliberately, per
the 2026-07-02 operator look direction. A closed schema that cannot express a
shipped path is a schema fitted to a model of the system rather than to the
system.

## MEASURED reality (driven, not read)

All numbers below come from driving the LIVE registry and the REAL producer at
HEAD, not from reading code. Probes: `tmp/_kbA_s1b_dump.py`,
`tmp/_kbA_sp_parity.py`, `tmp/_kbA_s0b_polyver.py`.

1. **31 registered engines produce THREE shapes.** Scene spine x26, the
   `mesh_stage` fork, `viz_*` zero -- plus ONE aspect knob. 27 of 31 produce
   an IDENTICAL 4-target fingerprint. The operator's own read ("this was
   over-engineered") is correct on the evidence.
2. **The plan table adds no differentiation.** 31 engines -> 6 distinct
   whole-plan signatures; 19 engines share ONE signature. Post-S1b the
   framing text is the producer's own six constants, so per-engine variation
   is still zero. **S5 exists to create differentiation that does not yet
   exist anywhere -- it is NEW AUTHORING, not a migration.**
3. **FIVE modules independently re-derive WHICH ENGINE IS EFFECTIVE**, from
   live env, at five different moments: `otr_video_director`,
   `otr_image_gen_dispatcher`, `otr_meta_brief_image_prompt`,
   `otr_shot_lock:919-933`, `render_driver`. And
   `validate_and_repair_still_spine` (`otr_video_render_batch.py:322`) runs
   BEFORE `apply_engine_override` (`render_driver.py:2751`), so with a force
   map set the spine is validated against the PICKED engine and rendered with
   the FORCED one. It survived because the validator is skipped entirely under
   `OTR_TEST_MODE` with no target receipt.
4. **SEVEN mechanisms decide "this engine requires a still."**
   `_still_spine_requires_scene` has four fall-throughs (id list, family,
   `required_inputs`, `provider_side and accepts_still`); then the LTX-I2V gate
   (`render_driver.py:1801-1817`, default ON), the IA2V portrait gate
   (`:1709-1721`), and the HuMo portrait requirement keyed on `family ==
   "audio_driven_face"` inside the validator. Three of the seven are invisible
   to the helper that looks authoritative.
5. **Layer 2 is eight named constants; layer 3 is pack-owned.**
   `STILL_FRAMING_OPEN` / `_SCENE_BEAT` / `_SCENE_CHARACTER`,
   `PORTRAIT_GEOMETRY` / `WIDE_PORTRAIT_GEOMETRY` /
   `TALKING_PORTRAIT_GEOMETRY`, `MESH_FODDER_POS_SCAFFOLD`,
   `BACKGROUND_PLATE_GEOMETRY`. The chunk-A1 split makes geometry Python-owned
   engine-safety and LOOK pack-owned. Portrait geometry is chosen at RUNTIME by
   `_style_anchor_for_aspect(aspect, talking, style)`.
6. **`provider_side` is not an attribute.** `_is_cloud_video_engine`
   (`render_driver.py:1274-1295`) is a three-part rule: `cloud_` id prefix OR
   the attribute OR `node_key.startswith("cloud_")`. `cloud_kling_avatar` is
   caught by the id prefix only.
7. **`wants_talking_prompt()` reads live env.** It calls
   `_recipe_config(self._recipe())`, and `_recipe()` re-reads
   `OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` / the UNET name on EVERY call, by
   documented design. Any `required="when_engine_talking"` row evaluated
   through the hook escapes whatever freeze wraps it.
8. **`policy_version` literals: 41 sites carrying a literal `2`** across 17
   files -- 35 in tests, 6 in production. Four other files
   (`_otr_casting.py`, `_otr_voice_bank.py`, `cast_lock.py`,
   `test_voice_bank.py`) mention `policy_version` for CASTING/VOICE policies
   and must not be touched.

## What is already LANDED (do not reopen)

- `33c4d8cf` S0a characterization fixture (31 engines x 8 configurations).
- `e60185a0` S0a-b isolation property amendment.
- `a98b1d5d` S1 schema + 31 declarations + post-registration audit.
- **S1b (this window)** -- all 57 rows' `framing_geometry` replaced with the
  producer's real GEOMETRY constants; the `ltx_audio_in` bookend row corrected
  from `scene_character`/scene to `portrait`/portrait; `_HUMO_STILL_PLAN` split
  into portrait and wide plans; new fence
  `tests/test_still_plan_layer2_parity.py` (text equals the producer constant
  per engine/row, no pack-owned LOOK in layer 2, no plan object shared across
  differing shipped aspects, no empty portrait row).
- **S0b is NOT landed** (filed BLOCKED at `c8db4c92`). Everything about the
  routing freeze is still open, and that is where the corrections keep landing.

## THE FORK this round must decide

**Option A -- finish the declarative table as specced.** A 7-field closed
`StillPlanRow` per engine, 31 plans, wired into seven consumers atomically at
S2, with a frozen `routing_state` v3 underneath.

**Option B -- freeze routing, then DELETE the table.** The measured variation
is 3 shapes + 1 aspect knob. Option B says: land the routing freeze (which is
the actual bug fix -- 5 re-derivations and a validator running before the
override), express requiredness as ONE capability-keyed function over the
frozen state, and keep the six geometry constants where they already live
instead of copying them into 31 declarations. Per-engine prompt customisation
(the operator's S5 directive) becomes a SEPARATE, thin per-engine text hook --
not a 7-field schema.

**Option C -- something neither. Propose it.**

## Questions for the panel

1. **Is the 31-plan table earning its keep?** It currently encodes 3 shapes and
   1 knob in 31 x 4-6 rows x 7 fields. What does it buy that a capability-keyed
   function over frozen routing does not? Be concrete about what breaks if it
   is deleted.
2. **Is S5 achievable on top of the table, or does the table make S5 harder?**
   S5 must give ~27 engines genuinely differentiated prompt text. Does a closed
   7-field row help an author, or does it push authoring into a schema that
   already cannot express one shipped style-tail behaviour?
3. **What is the minimum change that closes the REAL defect** (picked vs
   effective engine, validated-before-override)? If that is much smaller than
   the plan, say so plainly -- the operator's directive is lean and clean.
4. **The style_tail enum:** third token, or is style-tail simply not the plan's
   business? Argue from the shipped `ltx_radio_mouth` behaviour.
5. **Sequencing risk.** S2 wires seven consumers atomically. If the table
   shrinks or dies, S2 shrinks with it. Is there an ordering that lands the
   routing freeze and its live proof BEFORE any consumer cutover, so the bug
   fix ships even if the table question stays open?
6. **What should be DELETED?** Name code that should go away rather than be
   rewired -- the id list at `render_driver.py:635-637`, the tri-state basis in
   the dispatcher, `_effective_*_for_role` duplicates, stale prose at
   `eng_humo.py:497`. Deletion is the operator's stated preference.

## Hard constraints on any answer

- **THE LAW:** an audit may improve a story, never fail one for length,
  language, style, visual vocabulary or quality. Structural failures stay
  fail-closed.
- **NO FALLBACKS:** no substitute asset, no scene still as mesh fodder, no
  text-only or dark-floor degradation, no silent resize.
- **HuMo does not flip.** Four HuMo engines; `humo` / `humo_1.7B` are portrait,
  both `_169` are already wide, and the ComfyUI dropdown shows that split to
  the operator as a visible product contract. The only delta is FOUR hosts-off
  bookend role-cells redirecting to the wide `ltx_audio_in`.
- **`viz_camera` / `viz_green` / `viz_mxc_cpu` / `viz_mxc_mandala` need NO
  images.** An all-procedural episode invokes no image model at all.
- Any node/widget/link change edits `workflows/otr_canonical.json` in the SAME
  commit, append-only at the end of `widgets_values`, and regenerates the 11
  committed variants + 4 paired `.env.json` master hashes.
- Every chunk: focused tests + full Windows suite + Bug Bible +
  AST/BOM/zero-byte/UTF-8 + canonical hash, then commit AND push to
  `v2.0-alpha` with `HEAD == origin`.
- Root-cause fixes only. No shims. Do not remove an LLM pass or a field owner
  without giving every ledger field it wrote a new single owner.
