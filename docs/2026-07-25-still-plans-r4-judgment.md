# Still plans -- r4 judgment + AMENDED plan of record

**Written 2026-07-25 by CODER WINDOW A. Panel: codex `gpt-5.6-sol`
(reasoning high) + agy `Gemini 3.6 Flash (High)` -- both pins VERIFIED from
`kibitz-runs/2026-07-25-still-plans-s0b-order/r4/codex_model_selected.txt`,
`codex_reasoning_selected.txt`, `agy_model_selected.txt`. Driver `claude` so
the third seat spent no Claude pool. Claude is the grounded panelist and the
sole judge.**

Input reviewed: `docs/2026-07-25-still-plans-r4-corrected-plan.md` @ `562f9c85`.
This file AMENDS it and is the plan of record for the remaining chunks.

## Convergence verdict

- **agy: CONVERGED.** `yes-with-fixes` with exactly three must-fix items, and
  all three were ALREADY on the list -- S0b correction (a) `resolve_row_aspect`
  portrait fallback, correction (b) the prepass missing `_enforce_radio_is_host`,
  and this window's own geometry-vs-LOOK finding. Under the operator's rule
  ("a round that returns no, because of items already on the list, is
  CONVERGED") agy is converged. It also independently CUT any "all HuMo ->
  wide" flattening, which is the third confirmation of the S2 framing.
- **codex: NOT CONVERGED.** Ten must-fix items, several of them genuinely new
  and grounded. r4 therefore surfaced new material.

**Consequence:** S1b's content IS converged (both seats agree on it, with the
one split resolved below on evidence), so S1b proceeds. S0b-core and
everything after it are re-gated by one more r4 pass against this amended
document. No design round is reopened.

## Claims I GROUNDED and ACCEPTED

Each was checked against the real Windows files at HEAD, not taken on the
panel's word.

1. **codex #2 -- the `ltx_audio_in` bookend row is MISDECLARED. CONFIRMED.**
   Production mints that object with `"kind": "portrait"` at
   `nodes/otr_meta_brief_image_prompt.py:1782-1790` (`object_id` from
   `_ltx_radio_face_object_id`, `source: "ltx_radio_face"`), built by
   `build_radio_host_prompt(meta, "wide", radio_host_style="ltx_radio_mouth")`.
   S1 declares it `kind="scene_character"` / `target_class="scene"`
   (`eng_ltx_av.py:1148-1154`). The row must become `kind="portrait"`,
   `cardinality="per_bookend_role"`, `target_class="portrait"`,
   `aspect="wide"`, geometry `WIDE_PORTRAIT_GEOMETRY`.
2. **codex #5 -- `provider_side` is NOT an attribute lookup. CONFIRMED.**
   `_is_cloud_video_engine` (`render_driver.py:1274-1295`) is a THREE-part
   rule: the `cloud_` id prefix, OR `getattr(engine, "provider_side", False)`,
   OR `node_key.startswith("cloud_")`. `cloud_kling_avatar` is recognised by
   the id prefix. An `engine_facts` builder that used the bare attribute would
   classify it local and redirect a cloud avatar to local LTX.
3. **codex #3 -- `when_engine_talking` is NOT frozen by the plan. CONFIRMED.**
   `LtxAudioInEngine.wants_talking_prompt()` (`eng_ltx_av.py:390-400`) calls
   `_recipe_config(self._recipe())`, and `_recipe()` (`:402-432`) reads
   `OTR_LTX_AV_SHARP`, `OTR_LTX_AV_RECIPE` and `self._unet_name()` LIVE --
   its own docstring says "Read fresh every call". So a `required=
   "when_engine_talking"` row evaluated through the hook re-reads the
   environment after the freeze. One shared `row_is_active(...)` evaluator over
   captured state is required; no consumer may call the live hook.
4. **codex #4 -- deferring the mismatch gate leaves a live window. CONFIRMED
   in principle.** Between a landed S0b-core and a later S0c, upstream consumes
   frozen `ltx_resolved` while `eng_ltx_av` still builds from live env.
   RESOLUTION -- see "Judge call on the operator's split" below.
5. **codex #9 / agy should-fix #3 -- the `_use_i2v` contradiction. CONFIRMED**
   (already an OPEN BUGS row): `eng_ltx_video.py:559-572` logs and degrades to
   text-to-video on a missing init image while `render_driver.py:1801-1817`
   raises `RenderError` "NO FALLBACK to text-only rendering" on the same state.
   Both seats independently want the adapter to raise.
6. **codex #7 -- existing parity coverage cannot protect PROMPT parity.
   CONFIRMED by inspection**: `tests/test_still_plan_parity.py` freezes object
   id / kind / dimensions, not composed prompt text, `prompt_field_source`,
   `visual_style` or hashes. S1b's field diff proves the right constant is in
   the row; it cannot prove S2 wired it into the right LAYER. A post-S2
   composed-prompt comparison is owed.
7. **codex #1 -- the source-of-record documents CONFLICT. CONFIRMED.**
   `docs/GO_FORWARD_PLAN.md` still tells a builder to restore the composed
   inventory strings verbatim, and still carries the pre-correction chunk
   order in its coder-queue section, while the locked spec's section 11 knows
   nothing about S1b / S5. This document plus the GO_FORWARD refresh are the
   fix.

## The one PANEL SPLIT, and how the evidence decides it

**The `ltx_audio_in` bookend row.** codex #2 says change the row to a wide
`portrait` and use `WIDE_PORTRAIT_GEOMETRY`, explicitly "do not substitute
`STILL_FRAMING_SCENE_BEAT` or defer this structural correction to S5". agy
should-fix #2 says the opposite -- assign `STILL_FRAMING_SCENE_BEAT` as an
honest baseline and defer, because hard-coding one string "freezes a single
branch of a 3-way runtime switch (`radio_object` / `console_face` /
`ltx_radio_mouth`)".

**codex is right and agy is MISREAD. Discarded, on this evidence:** the
three-way switch is a parameter of `build_radio_host_prompt`, not a runtime
branch at this call site. The `_LTX_RADIO_FACE_ROLES` loop
(`otr_meta_brief_image_prompt.py:1768-1790`) passes
`radio_host_style="ltx_radio_mouth"` as a LITERAL. No environment, ledger or
style value can send that site down the `radio_object` or `console_face`
branch, so there is nothing to freeze away. And the `ltx_radio_mouth` branch
(`:394-401`) calls `_style_anchor_for_aspect(aspect, style=_style)` with
`aspect="wide"`, which returns `WIDE_PORTRAIT_GEOMETRY` -- codex's prescribed
constant is the one production actually composes.

## NEW must-fix that NEITHER panelist found (mine, grounded)

**The `style_tail_policy` closed enum cannot express the `ltx_radio_mouth`
path.** `build_radio_host_prompt`'s `ltx_radio_mouth` branch RETURNS EARLY at
`otr_meta_brief_image_prompt.py:401` with `"%s, warm dramatic lighting"`,
skipping BOTH `finish_visual_prompt(..., era_profile="still")` and the
`image_grade_tail` append -- deliberately, per the 2026-07-02 operator look
direction (the brief palette plus the grade tail rendered the talking-radio
bookend dark, blue and murky). So that path's style tail is neither `full` nor
`minimal_clean`, yet the S1 row declares `style_tail_policy="full"`.

`VALID_STYLE_TAIL_POLICIES` is a CLOSED enum and the spec says adding a token
is an operator decision, never a coder's. Therefore:

- **S1b does NOT touch `style_tail_policy`** -- S1b's scope is the
  `framing_geometry` text, and a parity chunk may not silently restate a style
  policy.
- **S2 must NOT treat the plan as the style-tail authority for the
  `ltx_radio_face` path.** That path keeps its own early return, and the
  mismatch is recorded here with a named owner rather than left unowned.
- **OPERATOR DECISION FLAGGED (not a blocker):** either add a third
  `style_tail_policy` token for "canonical warm, no era tail, no grade tail",
  or ratify that the `ltx_radio_face` path is exempt from the plan's style-tail
  authority. Default if the operator does not rule: the exemption, because it
  changes no behaviour.

## Judge call on the operator's S0b-core / S0c split

The operator's kickoff settled the split (S0b-core with `ltx_resolved` FROZEN;
only the `assert_usable` mismatch ASSERTION defers to S0c) and marked it
DO-NOT-RE-ASK. codex #4 now shows that split leaves a real window. Both are
satisfied without overriding the operator:

> **S0c stays a named sub-chunk of the S0b work and lands in the SAME PUSH
> BURST as S0b-core.** No commit that ships frozen `ltx_resolved` to consumers
> may be pushed without the mismatch gate in the same burst.

This is the operator's own standing directive for S0b, quoted in
`docs/S0b_KIBITZ_NEEDED.md`: "keep the whole S0b sequence in one push burst
before flipping DONE." The split survives as a review boundary; the window
codex found closes. codex's "cut S0c entirely" is therefore NOT adopted -- the
defect it names is closed by the burst rule.

## Amended chunk order

`S1b -> S0b-core (+S0c in the same push burst) -> S2 -> S3 -> S5 -> S4`

## ACCEPTED into S1b (this chunk)

1. Every row's `framing_geometry` becomes the producer's layer-2 GEOMETRY
   constant for that row's kind -- `STILL_FRAMING_OPEN`,
   `STILL_FRAMING_SCENE_BEAT`, `STILL_FRAMING_SCENE_CHARACTER`,
   `MESH_FODDER_POS_SCAFFOLD`, `BACKGROUND_PLATE_GEOMETRY`, and for `portrait`
   one of `PORTRAIT_GEOMETRY` / `WIDE_PORTRAIT_GEOMETRY` /
   `TALKING_PORTRAIT_GEOMETRY`. GEOMETRY only -- never a `*_LOOK_DEFAULT`.
2. The `ltx_audio_in` bookend row is corrected to `kind="portrait"`,
   `cardinality="per_bookend_role"`, `target_class="portrait"`,
   `aspect="wide"`, `framing_geometry=WIDE_PORTRAIT_GEOMETRY` (codex #2).
3. `_HUMO_STILL_PLAN` is SPLIT: a portrait plan for `humo` / `humo_1.7B` and a
   wide plan for `humo_1.7B_169` / `humo_14B_169`. One plan object cannot serve
   two shipped aspects once the portrait row carries aspect-specific text --
   and the spec's own "no inheritance or shared defaults" rule already forbade
   it. Machine-enforced by a new invariant, so it cannot regrow.
4. Adapter files carry the geometry as LITERAL TEXT. They do NOT import
   `otr_meta_brief_image_prompt` or `_otr_story_brief_helpers` to obtain it
   (codex should-fix #2) -- those are heavy, non-leaf modules and
   `still_plan_helpers` is cold-import clean by contract.
5. New fence `tests/test_still_plan_layer2_parity.py` pins four invariants:
   text equals the producer constant per engine/row; no row carries pack-owned
   LOOK vocabulary; no plan object is shared across differing shipped aspects;
   no `portrait` row is empty. This is agy should-fix #1 and codex should-fix #3
   made executable, and it is a DRIFT test -- it never judges prose, so it
   cannot collide with THE LAW.

## ACCEPTED into S0b-core (carried forward, in addition to the r3 three)

- `engine_facts` is `{engine_id, family, provider_side, render_aspect}`, and
  `provider_side` is computed by the EXISTING three-part cloud rule, never a
  bare `getattr`. Add a regression proving picked AND forced
  `cloud_kling_avatar` stay provider-side (codex #5).
- `resolve_row_aspect` REJECTS a missing/None aspect on an `inherit_engine`
  row instead of returning portrait (r3 (a); both seats).
- The prepass freezes each role's FINAL effective engine including
  `_enforce_radio_is_host` (r3 (b); both seats).
- ONE shared `row_is_active(...)` activation evaluator over CAPTURED state; the
  captured talking result joins the closed routing state (inside
  `ltx_resolved`). No consumer calls `wants_talking_prompt()` live (codex #3).
- The routing-state schema can represent a malformed-force-map receipt so
  unset and malformed cannot hash identically through `IS_CHANGED`
  (codex #6).
- The `policy_version` literal inventory is DERIVED, not tabled. Mechanically
  measured at HEAD (`tmp/_kbA_s0b_polyver.py`): 17 files mention
  `policy_version`, 41 carry a literal `2` -- **35 in tests, 6 in production**
  (`render_driver.py:2516`, `otr_image_director.py:375`,
  `otr_image_gen_dispatcher.py:542`, `otr_shot_lock.py:1306`,
  `otr_video_director.py:353` comment + `:354`). `test_hybrid_voice_fit` has
  ZERO (the doc claimed one) and `test_still_plan_parity` adds five. The
  remaining `policy_version` mentions in `_otr_casting.py`, `_otr_voice_bank.py`,
  `cast_lock.py`, `test_voice_bank.py` are CASTING/VOICE policies and must NOT
  be touched.
- `eng_ltx_video` raises on a missing/stale init image when the frozen I2V flag
  is enabled, so the adapter and the driver stop disagreeing (codex #9 + agy
  should-fix #3). Static finding -- no PBUG/Bible row without live admission.

## ACCEPTED into S2 / S4 / S5

- **S2** stays the FOUR ROLE-CELLS and nothing else; both seats confirm, agy
  explicitly cuts any HuMo flattening. Add the post-S2 composed-prompt parity
  comparison (codex #7): final prompt text, `prompt_field_source`,
  `visual_style`, subject preservation and style-tail placement per
  kind/configuration.
- **S4** gains a THIRD fresh-boot leg: forced portrait HuMo bookend with
  `OTR_ENABLE_HUMO_HOSTS=1`, verifying the effective engine STAYS HuMo and the
  still stays 832x1216 (codex #10). Two legs exercised only the redirect.
- **S5** needs a per-engine table before it is implementable: for every
  non-`viz_*` engine and row -- mandatory geometry, permitted engine-specific
  addition, prohibited subject/style ownership, expected
  cardinality/requiredness, and an exact composed-output assertion
  (codex #8). **ADOPTED CUT (codex cut #1):** "every non-visualizer must have a
  unique signature" is NOT the acceptance metric. Forced uniqueness invites
  cosmetic prompt drift; independent ownership plus a mechanically justified
  difference is the goal. Punctuation or engine-name injection does not count
  as customization.

## Not adopted, and why

- **codex cut #2 ("cut S0c entirely")** -- the window it names is closed by the
  same-push-burst rule above, which is the operator's own directive. The
  review boundary is worth keeping.
- **agy should-fix #2 (`STILL_FRAMING_SCENE_BEAT` on the bookend row)** --
  misread; see the split section.
- **codex should-fix #1 (`+ Add Custom Model` v3 semantics)** -- real, but it
  is a custom-model-registration question that reaches beyond this build's
  scope. Recorded for the planner window, not folded into S0b-core.
