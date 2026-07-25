# ARCHITECTURE ROUND -- per-beat still ownership, and ratify-or-improve the Option-C cut

**Repo:** ComfyUI-OldTimeRadio, branch `v2.0-alpha`, HEAD `22e65a07`
(HEAD == origin). Suite `6444 passed / 27 skipped / 1 xfailed`; Bug Bible
`17 passed`; canonical workflow SHA-256 prefix `5377914B`.

**This is an ARCHITECTURE round, not a coding plan.** Do not produce a
file-by-file patch list. Produce the right SHAPE, with the reasoning that
justifies it, grounded in the code you can read in this repo.

**Read the real files. Every claim you make must cite `path:line`.** A claim
you cannot anchor will be discarded by the judge, not weighed.

---

## 1. What this build is, in one paragraph

OTR renders old-time-radio episodes: a story writer produces a ledger, an
IMAGE phase mints stills, and a VIDEO phase animates them through one of ~31
registered motion engines (`nodes/_otr_video_engines/`). Different engines
need different stills: an image-to-video engine's still IS its init frame, a
text-to-video engine's still is optional, a HuMo audio-driven-face engine
needs a portrait, a mesh lane needs clay-blob fodder, a visualizer lane needs
nothing. The "still plans" build exists because that knowledge was smeared
across five capability maps and seven ad-hoc requiredness mechanisms.

## 2. What has already landed (do not re-litigate; ground against it)

- `S0a` / `S0a-b` / `S1` / `S1b`: a closed seven-field `StillPlanRow` schema
  (`nodes/_otr_shared/still_plan_helpers.py`) plus 31 per-adapter `still_plan`
  declarations across 12 adapter modules, validated by a POST-REGISTRATION
  audit (`tests/test_still_plan_audit.py`) and a layer-2 parity fence
  (`tests/test_still_plan_layer2_parity.py`).
- **Nothing consumes the plan in production yet.** The module docstring says
  so explicitly. The wiring chunk (`S2`) never ran.
- Measurement that motivated the prior R1: driving the live registry over all
  31 engines yields 14 shared plan objects but only SIX distinct signatures
  and SIX distinct structures. 19 engines share ONE signature. The declared
  prose adds ZERO per-engine differentiation today.

## 3. What the prior R1 decided (2026-07-25, same panel, `--driver claude`)

Both seats independently said **CUT** the 31-plan table. The judge took
codex's **Option C** over agy's Option B:

> Freeze the effective routing state, give each ADAPTER a COMPACT capability
> descriptor (`still_mode = scene|mesh|none`, narrow activation flags,
> aspect), run ONE pure materializer over it, and keep a SEPARATE per-engine
> layer-2 prompt hook. Delete `StillPlanRow`, its closed enums, and the 31
> copied declarations. `style_tail_policy` leaves the structural contract
> entirely.

Rationale on the fork: agy's single central `engine_requires_still(routing_state)`
recreates the central-authority shape this build exists to kill, and the
operator's directive requires per-adapter ownership.

Judgment of record: `docs/2026-07-25-still-plans-r1-lean-judgment.md`.

## 4. THE OPERATOR'S NEW INPUT -- this is why you are being asked again

The operator did NOT ratify. He gave two instructions:

1. **"kibitz codex and then confirm w/ agy the best architecture."** Option C
   is a CANDIDATE, not a settled decision. Improve it, replace it, or ratify
   it -- but argue from the code.
2. **"we want per-beat stills."** This is now a REQUIREMENT, not a
   preference, and it lands directly on the open LTX question below.

### The operator's settled doctrine, stated verbatim (2026-07-25, mid-round)

He clarified twice. This is NOT a preference -- he says it was decided long
ago and is being restated because the plan drifted off it:

> "for video, if it needs a still, it's ALWAYS a still per beat"
> "needs as many stills as to cover the video for that beat"

Read those together. Two parts, and the second is the one no current code or
plan expresses:

- **(A) Requiredness is per-adapter; CARDINALITY IS NOT AN OPEN QUESTION.**
  If a video path needs stills at all, they are per beat. Always. The
  `cardinality` enum in `still_plan_helpers.py` (`per_beat` / `per_subject` /
  `per_recurring_subject` / `per_bookend_role`) encodes a choice the operator
  says was never open. Live counts across the adapters: 43 `per_beat`, 14
  `per_subject`, 1 `per_bookend_role`, 1 `per_recurring_subject`. **Either
  those 16 non-`per_beat` declarations are wrong, or the doctrine has an
  exception nobody wrote down. Settle it explicitly; do not paper over it.**
  My own read is that the portrait/cast rows may be a DIFFERENT AXIS (a
  portrait is per subject by nature, and is an INPUT to a beat's render
  rather than the beat's own still) rather than a contradiction -- but that
  is precisely what I want the panel to decide.
- **(B) COVERAGE, not multiplicity. This is the new requirement.** A beat
  needs **as many stills as it takes to COVER that beat's video.** One still
  per beat is NOT the contract; enough stills to cover the beat's rendered
  duration is. Today the schema's finest granularity is `per_beat`, i.e. ONE.
  Nothing expresses N-stills-covering-one-beat.

  This is load-bearing because a beat's render is ALREADY length-constrained
  and can be shorter than the beat: `eng_wan_ti2v._floor_length`,
  `video.max_render_frames` (`nodes/_otr_shared/capability_profiles.py`,
  `nodes/_otr_video_engines/motion_common.py`, `render_driver.py`), and
  per-shot `target_frame_count` (`schemas.py:302`). The 8-GB WAN tier pins a
  render ceiling of 17 frames against a 177-frame default. **A 177-frame beat
  rendered in 17-frame units is a COVERAGE problem, and the still plan is
  where coverage has to be decided.** Say how many stills such a beat needs,
  WHO computes that number, and WHEN -- it depends on the engine's ceiling,
  which is exactly the routing state this build wants to freeze.

## 5. The tension you must resolve -- freeze versus per-beat

The prior R1's Option C rests on FREEZING the effective routing state so the
still spine can be validated against the engine that will actually render.
The operator now requires per-beat variation. Those pull against each other.
Resolving that tension IS this round's job.

**Grounded facts. I verified each of these at HEAD; cite them, do not
re-derive them, and CORRECT me if you find one wrong:**

1. `nodes/_otr_video_engines/eng_ltx_av.py:402-405` -- `_recipe()` docstring,
   verbatim: *"Resolve the active recipe. Read fresh every call (an operator
   flips daily<->hero per beat by swapping `OTR_LTX_AV_UNET` /
   `OTR_LTX_AV_RECIPE`)."* So per-beat recipe switching is an ADVERTISED,
   DOCUMENTED capability today -- implemented as an ambient environment
   re-read, not as a shot-owned field.
2. `eng_ltx_av.py:392` (`wants_talking_prompt`) calls
   `_recipe_config(self._recipe())`, so ANY activation flag keyed on
   "is this engine talking" re-reads the environment at evaluation time and
   escapes any capture taken earlier.
3. `eng_ltx_av.py:407-412` -- `_recipe()` RAISES `EngineUnusable` when the
   retired `OTR_LTX_AV_SHARP` is present. An `IS_CHANGED` / capture surface
   that ignores that variable can cache across a hard-error state change.
4. `nodes/otr_video_render_batch.py:322` calls
   `_rd.validate_and_repair_still_spine(ledger)` -- the still spine is
   validated HERE.
5. `nodes/_otr_video_engines/render_driver.py:2783` `apply_engine_override`
   applies `OTR_FORCE_ENGINE_MAP`, and on a parse error it logs
   `"OTR_FORCE_ENGINE_MAP IGNORED (parse)"` and returns the ledger UNCHANGED
   -- a fall-back, against the project's fail-closed law.
6. `render_driver.py:1413` `_enforce_radio_is_host` mutates `shot["engine_id"]`
   IN PLACE and is called FIRST inside `build_request_from_shot`
   (`render_driver.py:1510`) -- i.e. PER SHOT, at request-build time, AFTER
   the spine validation at (4). This is the second engine mutation, and the
   one the prior plan's "freeze" did not cover.
7. Therefore the live defect: **with a force map or a radio-host redirect in
   play, the spine is validated against the PICKED engine and rendered with a
   DIFFERENT one.**
8. `still_plan_helpers.py:177-189` `resolve_row_aspect` SILENTLY RETURNS
   `portrait` when the engine aspect is absent, and the descriptor the prior
   plan specified (`{engine_id, family, provider_side}`) carries no aspect
   field -- so every `inherit_engine` row would resolve portrait, including
   the two WIDE `_169` HuMo engines and `cloud_kling_avatar`.
9. `render_driver.py:1274-1295` `_is_cloud_video_engine` is a THREE-part rule
   (id prefix OR attribute OR node_key prefix). `cloud_kling_avatar` has no
   `provider_side` attribute and is caught by the id prefix alone, so a naive
   `getattr` in a facts builder classifies it LOCAL.
10. `eng_ltx_av.py:345` -- LTX-AV declares `render_aspect = "wide"` with a
    comment recording the 2026-06-17 operator catch: without it the director
    minted a portrait still that the wide render centre-cropped, "lopping the
    subject's head off". Aspect correctness is a scar, not a nicety.

## 6. The questions -- answer these, in this order

**Q1. Is Option C the right shape, given per-beat is now a requirement?**
Ratify, amend, or replace. If you replace it, say what breaks in Option C
that your alternative fixes. Judge-relevant: an answer of "yes, as-is" must
still explain how it satisfies per-beat variation.

**Q2a. What computes STILL COVERAGE for a beat, and where does the answer
live?** Given the operator's doctrine (as many stills as cover the beat's
video), the count is a FUNCTION of the beat's duration and the effective
engine's render ceiling -- neither of which the adapter knows in isolation.
Name the owner, the seam, and the failure mode when coverage cannot be met
(fail closed, or degrade -- and if you say degrade, reconcile that with the
project's no-fallback law). Also say whether `cardinality` survives at all as
a schema concept once coverage is computed rather than declared.

**Q2b. HOW IS EACH CLIP'S STILL OBTAINED? (operator's own question, verbatim,
and he wants your answer on it specifically.)**

> "if a beat needs 5 clips of video, the still at the end of each clip will be
> used to fill the still for the next -- or? Or we reuse the one still, or gen
> a new one."

So for a beat that renders as N clips, weigh at least these three, and say
which is right, under what conditions, and who decides:

  - **(1) CHAIN.** Extract the LAST FRAME of clip k and use it as the init
    still for clip k+1. Maximum continuity; risk is generational drift /
    colour creep across a long beat, and it serializes clips that could
    otherwise render in parallel.
  - **(2) REUSE.** One minted still initialises every clip. No drift, no
    serialization; risk is that each clip restarts from the same pose, which
    reads as a stutter or a loop rather than continuous motion.
  - **(3) REGENERATE.** Mint a fresh still per clip from the beat's prompt.
    Most variety; weakest continuity unless the stills are conditioned on
    each other somehow.

  **Do not answer this from first principles alone -- the machinery may
  already partly exist and I want you to check it before proposing anything:**
  - `nodes/_otr_video_engines/eng_google_veo_video.py:129-131` defines
    `_last_frame_ref(request)` reading a `last_frame` asset off the request.
    So at least one adapter already has a last-frame concept.
  - `nodes/_otr_video_engines/schemas.py:296-299` -- `ExecutionGroup` carries
    `depends_on: list[str]` and `produces_base_for: list[str]`.
  - `nodes/_otr_shared/resolver.py:18-28` documents these as a
    PROVIDER/CONSUMER DAG: *"`depends_on` is the only edge source of truth for
    ordering; `produces_base_for` is provider-side bookkeeping (a provider
    lists the consumers it feeds)"*, validated by `validate_execution_groups`
    at lock.

  **That is a base-frame-producer -> consumer edge that already exists,
  is already schema'd, and is already validated for cycles and ordering.**
  Tell me whether clip chaining is the thing that machinery was built for, and
  whether coverage clips should BE execution groups rather than a new
  parallel concept. If reusing it is wrong, say why.

  **SCOPE, operator verbatim: "and that goes for ALL video models where we
  have more than 1 video clip per beat."** So this is NOT an LTX or a WAN
  special case. Every registered engine that can ever render more than one
  clip for a single beat is in scope, and the contract must be general.

  That leaves ONE genuinely open question, and it is the crux of this round:
  **is the STRATEGY uniform or per-adapter?** The REQUIREMENT is universal --
  he has said so. But:
  - **Uniform reading:** one rule for every engine (e.g. always chain), so
    there is exactly one code path, one thing to test, and no per-adapter
    divergence to drift. Simplest, and it fits the "lean and mean" direction.
  - **Per-adapter reading:** the requirement is universal, the strategy is
    owned by the adapter, because the still MEANS different things to
    different engines -- a continuity anchor to an i2v scene engine, an
    IDENTITY anchor to an audio-driven face (chaining a drifting last frame
    into a HuMo beat would let the character's face change mid-beat, which
    re-opens the 2026-05-01 / 2026-06-30 "generic human host" defect class),
    and nothing at all to the four `viz_*` lanes.

  **Argue BOTH and pick one.** If you pick per-adapter, name the exact field
  and its closed token set, and say what stops it drifting into 31 bespoke
  behaviours. If you pick uniform, say what happens to the HuMo identity case.
  Do not split the difference without a rule that decides every engine.

**Q2c. What owns per-beat still VARIATION?** Concretely: what is the unit of
capture, where does it live, and who writes it? Candidate shapes to weigh
(add your own if better):
  - (i) shot-owned fields on the ledger's `video.shots[]` rows, captured once
    per beat at plan time;
  - (ii) an episode-level frozen routing state plus an explicit per-beat
    override channel;
  - (iii) per-beat re-capture -- freeze at BEAT granularity rather than
    episode granularity, so "frozen" and "per-beat" stop being in conflict;
  - (iv) keep ambient env re-reads and drop the freeze (state plainly what
    that costs).

**Q3. Where does the ENGINE MUTATION boundary belong?** Both mutations (fact
5 and fact 6) must be settled before still validation. Does the redirect move
earlier, does validation move later, or does a third pass own "final
effective engine per shot"? Name the seam.

**Q4. What is the minimum contract for `+ Add Custom Model`?**
`nodes/otr_video_director.py:443-481` permits an unknown custom engine id.
Neither a closed table nor a registry-keyed function can know its still
requirements. Fail closed, or something better?

**Q5. What is the ordering of record?** The prior R1 concluded the routing
freeze was always the real bug fix and should ship FIRST and ALONE. Does
per-beat change that? If the freeze must now know about per-beat, it may no
longer be independently shippable -- say so if you believe it.

## 7. Invariants any answer must respect (rejecting one is an automatic fail)

- **THE LAW:** an audit may improve a story; it may NEVER fail one for
  length, language, style, visual vocabulary, or quality. Only the whole-word
  safety authority and structural/provider/rights failures are terminal.
- **Fail closed, no shims, no fallbacks.** Root-cause fixes only. A malformed
  config is an error, not a warning-and-continue.
- **Per-adapter ownership.** The operator's directive is that each video path
  owns its own still operations. A central function keyed on engine id is the
  disease, not the cure.
- **Geometry vs LOOK.** Layer-2 framing geometry is Python-owned engine
  safety; LOOK (`VisualStyle.portrait_look` / `portrait_look_talking` /
  `plate_look`) is pack-owned. A plan may contribute geometry and may NEVER
  decide style. The eight `*_GEOMETRY` / `STILL_FRAMING_*` constants live in
  `nodes/otr_meta_brief_image_prompt.py` and
  `nodes/_otr_story_brief_helpers.py` -- NOT in `render_driver.py`, and there
  are EIGHT, not six.
- **ComfyUI node contract.** Adapter imports are wrapped in
  `try/except: pass` (`nodes/_otr_video_engines/__init__.py`), so anything
  that raises at class-body or decorator time SILENTLY DELETES the engine
  from the menu. Validation must stay a post-registration audit.
- Any node / widget / link / schema change edits `workflows/otr_canonical.json`
  in the SAME change. Unwired code is dead code.
- UTF-8, no BOM, ASCII where practical, SFW.

## 8. What to return

A VERDICT line, then MUST-FIX, then SHOULD-FIX, then CUT (what to delete from
the candidate). For each item: the claim, the `path:line` anchor, and the
consequence if ignored. Be specific about where you DISAGREE with the prior
R1 -- agreement that merely restates it is worth less than a grounded
objection.

---

## 10. LATE OPERATOR AMENDMENT (arrived after the brief was frozen)

Operator, verbatim:

> "yes good for Veo, we need similar for all HuMos, WAN 8GB, LTXes and the
> other cloud engines."

Context: this was his reply to the observation that
`nodes/_otr_video_engines/eng_google_veo_video.py:129-131` already defines a
`last_frame` asset ref. He wants the equivalent per-clip still capability to
reach the HuMo family, `wan_8gb`, the LTX engines, and the remaining cloud
engines -- i.e. this is a GENERAL contract, not a Veo feature.

**Two things to check rather than assume, because I am not certain of either:**

1. **What does Veo's `last_frame` actually MEAN?** Read
   `eng_google_veo_video.py:269-296`. If it is a provider-API input describing
   the frame the generated clip should END on, that is FIRST-FRAME/LAST-FRAME
   INTERPOLATION, which is NOT the same thing as chaining clip k's output into
   clip k+1. Say plainly which it is, because the whole "extend it to the other
   engines" instruction means something different in each case, and the
   operator may be generalizing from a feature that does not do what the name
   suggests.
2. **Do the named engines even render more than one clip per beat today?**
   Check how a beat's frames are actually produced -- in particular whether
   any engine already fills a beat by a means OTHER than rendering multiple
   clips (look hard at `eng_wan_ti2v.py` around `_floor_length` and the render
   tail). If a beat is currently ONE clip, then "more than 1 clip per beat" is
   a NEW capability being requested, not an existing one to standardise, and
   the honest plan must say so and cost it.

Answer both explicitly. If the existing mechanism is not what it appears to
be, say so directly -- the operator would rather be corrected now than build
on a wrong premise.
