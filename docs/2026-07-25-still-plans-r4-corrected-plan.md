# Per-model still plans -- CORRECTED PLAN for r4 convergence

**Written 2026-07-25 by CODER WINDOW A at HEAD `17ad2d57` (`v2.0-alpha`,
HEAD == origin, canonical `5377914B`).**
Spec of record: `docs/2026-07-25-still-plans-locked-build-spec.md` (locked
@ `84328aa1`). Grounding inventory: `docs/STILL_PLAN_SEED_INVENTORY.md`.
S0b site inventory: `docs/S0b_HANDOFF.md` + `docs/S0b_KIBITZ_NEEDED.md`.

## What this round IS

**r4 = a DEFECT SWEEP at current HEAD, not a redesign.** The r1->r5 arc
already converged and codex explicitly cut the design round. S0a, S0a-b and
S1 are LANDED. This document is the plan as CORRECTED by the 2026-07-25 r3
panel plus this window's own grounding, and r4's only job is:

> Is there a NEW must-fix in the plan below -- something that would break a
> build or ship a silent behaviour change -- that is not already on the list?

A round that answers "no, because of items already listed" is CONVERGED.
Do NOT re-argue the architecture, the chunk order, or Path A vs Path B
(both r3 panelists rejected Path B; the operator ratified Path A-lite).

## Chunk order (settled)

`S1b -> S0b-core -> S2 -> S3 -> S0c -> S5 -> S4`

## Landed already

- `33c4d8cf` S0a -- characterization fixture, 31 engines x 8 configurations.
- `e60185a0` S0a-b -- isolation property amendment.
- `c8db4c92` S0b -- **NOT LANDED**, filed BLOCKED (no half-land of a
  cross-module atomic refactor).
- `a98b1d5d` S1 -- `nodes/_otr_shared/still_plan_helpers.py`, 31 per-engine
  `still_plan` attributes across 16 adapters, `tests/test_still_plan_audit.py`.
  **Nothing reads the plan yet.**

## S1b -- restore the layer-2 framing text (PURE PARITY, blocked on nothing)

### Why it exists

Spec section 5 makes `framing_geometry` the literal layer-2 prompt TEXT
("authored TEXT -- the ONE free field"). S1's 31 declarations are
PARAPHRASES, not transplants. Wiring S1 as it stands silently degrades every
prompt in the system. Measured at HEAD by driving the live registry
(`tmp/_kbA_s1b_dump.py`): 31 engines, 6 distinct whole-plan signatures,
19 engines sharing one.

### The authoritative source is the CODE, not the inventory doc

This is this window's correction to the r3 plan. The seed inventory records
the COMPOSED output strings; the producer holds the layer-2 constants as
named literals. S1b transplants the CONSTANTS. Ground truth at HEAD:

`nodes/_otr_story_brief_helpers.py` (layer-3 framing hints):

- `STILL_FRAMING_OPEN` (:513) -> kind `scene_open`
- `STILL_FRAMING_SCENE_BEAT` (:518-520) -> kind `scene_beat`
- `STILL_FRAMING_SCENE_CHARACTER` (:530-533) -> kind `scene_character`

`nodes/otr_meta_brief_image_prompt.py` (geometry contracts):

- `PORTRAIT_GEOMETRY` (:105-107)
- `WIDE_PORTRAIT_GEOMETRY` (:108-110)
- `TALKING_PORTRAIT_GEOMETRY` (:111-115)
- `MESH_FODDER_POS_SCAFFOLD` (:617-624) -- the clay-blob clause
- `BACKGROUND_PLATE_GEOMETRY` (:635-638)

### NEW FINDING 1 -- geometry vs LOOK. Do not transplant the composed string.

`otr_meta_brief_image_prompt.py:96-104` records the chunk-A1 GEOMETRY-vs-LOOK
split: the `*_GEOMETRY` constants are ENGINE-SAFETY framing contracts and
"NEVER move into style packs"; the LOOK segment (costume / environment /
lighting vocabulary) is PACK-OWNED (`VisualStyle.portrait_look` /
`portrait_look_talking` / `plate_look`). `_style_anchor_for_aspect` (:152-169)
returns `"%s, %s" % (geometry, look)`.

The seed inventory's `portrait` line reads "... (never crop the top of the
head), period-accurate costume and environment, dramatic film lighting" --
that trailing clause is `PORTRAIT_LOOK_DEFAULT`, i.e. PACK-OWNED LOOK, and it
survives in Python ONLY as the sci_fi_radio extraction fixture. Spec section 4
says a plan "may only contribute layer 2 ... it may never decide style."

**Therefore: transplanting the inventory's composed strings verbatim would
move pack-owned style vocabulary into the plan and hard-code the sci_fi_radio
pack's look into all 31 engines.** S1b transplants the GEOMETRY constants
only. Same for `scene_background_plate`: `BACKGROUND_PLATE_GEOMETRY`, NOT
`BACKGROUND_PLATE_POS_SCAFFOLD` (which appends `PLATE_LOOK_DEFAULT`
"period-accurate set").

**r4 question 1:** is that the right call, or does the plan legitimately own
the composed string? If the plan owns only geometry, the inventory doc's
"restore every clause VERBATIM" line needs the same correction.

### NEW FINDING 2 -- `portrait` has THREE geometries, chosen at runtime

`_style_anchor_for_aspect(aspect, talking, style)`:

- `talking` -> `TALKING_PORTRAIT_GEOMETRY` (face-forward close-up bust)
- `wide` -> `WIDE_PORTRAIT_GEOMETRY` (head-and-shoulders medium; the
  2026-06-17 operator catch -- a three-quarter body shot decapitates in a
  short landscape frame)
- else -> `PORTRAIT_GEOMETRY` (three-quarter; the 2026-06-10 KEEPER)

But at HEAD **all 27 portrait rows declare `aspect="inherit_engine"`**, and
`framing_geometry` is ONE static string per row. A single static string
cannot express a switch that resolves differently per engine. So a naive
"paste the portrait clause into every portrait row" transplant would ship
`PORTRAIT_GEOMETRY` to ~20 WIDE engines and re-introduce exactly the
decapitation defect `WIDE_PORTRAIT_GEOMETRY` was authored to fix.

Proposed S1b resolution (mechanical, per adapter file, NOT design): each
adapter's `render_aspect` is a static class attribute in the SAME file as its
plan, so the switch is resolvable at authoring time.

- `render_aspect == "portrait"` (`humo`, `humo_1.7B`) -> `PORTRAIT_GEOMETRY`
- `render_aspect == "wide"` (all others) -> `WIDE_PORTRAIT_GEOMETRY`
- row with `required="when_engine_talking"` (`ltx_audio_in`, the ONLY engine
  declaring `wants_talking_prompt`) -> `TALKING_PORTRAIT_GEOMETRY`

The `aspect="inherit_engine"` FIELD is unchanged (it drives DIMENSIONS via
`resolve_row_aspect`); only the authored TEXT is pinned per file.

**r4 question 2:** does pinning the text per file while leaving
`aspect="inherit_engine"` create a divergence hazard -- a later engine whose
`render_aspect` changes would silently keep the wrong text? Should the audit
gain a check that the row's text matches the engine's shipped aspect?

### Known losses S1b must fix (all confirmed at HEAD)

| kind | S1 paraphrase at HEAD | restore to |
|---|---|---|
| `portrait` (22 rows) | `""` EMPTY STRING | the aspect-correct geometry above |
| `portrait` (5 face rows) | "Face-forward portrait framing; head and upper body of the named subject centered in the frame." | as above |
| `portrait` (ltx_audio_in talking) | "Face-forward portrait framing for a character beat; ..." | `TALKING_PORTRAIT_GEOMETRY` |
| `scene_open` (26 rows) | "Wide establishing shot; the scene an audience is entering." | `STILL_FRAMING_OPEN` |
| `scene_beat` (26 rows) | "Wide continuity framing for the beat, matching the scene_open geometry." | `STILL_FRAMING_SCENE_BEAT` |
| `scene_character` (26 rows) | "Wide framing that keeps the named character legible in the scene." | `STILL_FRAMING_SCENE_CHARACTER` |
| `mesh_fodder` | "Clean mesh fodder: subject centered, neutral studio backdrop, no letterboxing; ONE isolated subject only." | `MESH_FODDER_POS_SCAFFOLD` (the FULL clay-blob clause) |
| `scene_background_plate` | "Wide background plate for the beat; the stage the meshed subject will inhabit." | `BACKGROUND_PLATE_GEOMETRY` |

`mesh_fodder`'s `style_tail_policy` is ALREADY `minimal_clean` at HEAD --
verified, no change. Every other row is `full`.

### The one row with no inventory clause

`ltx_audio_in` declares a `scene_character` / `per_bookend_role` /
`when_engine_talking` row whose paraphrase is "Wide radio-face framing for
the announcer/music bookend; period broadcast host at the microphone." The
seed inventory has NO per-KIND clause for a bookend radio face -- the
producer composes that path from `get_open_subject` +
`_style_anchor_for_aspect` + `announcer_subject_face` /
`radio_host_style` (`otr_meta_brief_image_prompt.py:375-394`), and the
FACELESS-vs-face decision is a live `radio_host_style` branch
(`radio_object` / `console_face` / `ltx_radio_mouth`).

**r4 question 3 (the one I most want broken):** is this row transplantable at
all in a PARITY chunk, or is it inherently an S5 authoring decision? Pinning
one static string here risks freezing one branch of a three-way runtime
switch. Candidate answers: (a) give it `STILL_FRAMING_SCENE_BEAT` as the
honest kind-level parity value and let S5 author the radio-face text;
(b) leave the paraphrase and mark the row explicitly S5-owned;
(c) something the panel sees that I do not.

### S1b acceptance

- Field-by-field diff of every restored clause against the named constants,
  printed before commit.
- The S0a fixture (`tests/test_still_plan_parity.py`) stays GREEN --
  nothing reads the plan at S1b, so a parity delta would mean a real defect.
- Full Windows suite + Bug Bible + AST/BOM/zero-byte/UTF-8 + canonical hash
  `5377914B` unchanged (no node/widget/link touched).

## S0b-core -- the routing freeze (atomic), WITH the three r3 corrections

Scope of record: `docs/S0b_KIBITZ_NEEDED.md` sections 1-11 and
`docs/S0b_HANDOFF.md`. Corrections that MUST be applied before it is built:

**(a) The closed `engine_facts` descriptor has no aspect field.**
Spec:230 fixes `{engine_id, family, provider_side}`, but
`resolve_row_aspect` (`still_plan_helpers.py:177-189`) reads
`engine_render_aspect` / `render_aspect` and **SILENTLY RETURNS PORTRAIT
when absent**. Every `inherit_engine` row would therefore resolve PORTRAIT --
including `cloud_kling_avatar` and both wide `_169` HuMos. Fix: add a
canonical `render_aspect` field to the descriptor and REJECT a missing value
instead of falling back. (agy misread this as "key-name insensitivity
confirmed" -- true but irrelevant; the field is ABSENT. codex is right.)

**(b) The frozen-routing prepass as specified does not close its own defect.**
`apply_engine_override` (`render_driver.py:2784`) applies ONLY
`OTR_FORCE_ENGINE_MAP`. The radio-host redirect is a SEPARATE mutation at
`render_driver.py:1413-1513`. A prepass that runs only `apply_engine_override`
before `validate_and_repair_still_spine` leaves the reproduced defect open.
Fix: the prepass must freeze each role's FINAL EFFECTIVE engine, redirect
included.

**(c) The `policy_version=2` literal inventory is stale.** ~35 sites, not 31:
`test_hybrid_voice_fit` has NONE (the doc lists 1) and `test_still_plan_parity`
adds five. Derive the list MECHANICALLY, never from the doc's table.

**Four env-read sites the S0b inventory omits** (r3, grounded):

- `eng_ltx_video.py:541-564` -- `OTR_ENABLE_LTX_I2V`
- `render_driver.py:1176-1203` -- `OTR_ENABLE_HUMO_HOSTS`
- `otr_meta_brief_image_prompt.py:297-300` -- `OTR_ENABLE_HUMO_HOSTS`
- `eng_ltx_av.py:352-353, 403-432` -- recipe / UNET re-read outside
  `assert_usable`

**SCOPE (judge call on a panel split):** adopt agy's S0b-core / S0c relief,
BUT keep `ltx_resolved` FROZEN inside S0b-core -- that answers codex's
objection that deferring it desynchronizes `when_engine_talking`. ONLY the
`eng_ltx_av.assert_usable` mismatch ASSERTION defers to S0c.

## S2 -- the cutover. FOUR ROLE-CELLS, nothing else.

OPERATOR EYEBALL RESOLVED 2026-07-25. There are FOUR HuMo engines. Only
`humo` and `humo_1.7B` ship `render_aspect="portrait"`; `humo_1.7B_169` and
`humo_14B_169` are ALREADY wide, and the ComfyUI dropdown shows the operator
"(portrait)" / "(16:9)" -- a VISIBLE PRODUCT CONTRACT that is never flattened.

**Nothing about HuMo flips.** The S2 delta is exactly four role-cells: the two
PORTRAIT HuMo picks x {announcer, music}, under the hosts-off DEFAULT, where
`_enforce_radio_is_host` redirects the beat to the WIDE `ltx_audio_in` that
actually renders it. With `OTR_ENABLE_HUMO_HOSTS=1` a portrait HuMo KEEPS its
portrait still. A blanket "HuMo -> wide" is a REGRESSION.

The "via the `_169` siblings' render_aspect" framing in
`docs/S2_EYEBALL_REQUEST.md` is WRONG ON MECHANISM and is corrected in the
same chunk, along with the S0a fixture's `special_cases` rows.

## S3 / S0c / S5 / S4

- **S3** -- shim + stale-prose deletion (incl. `eng_humo.py:497`, which still
  describes a degrade chain ripped 2026-07-02 while `:502` sets
  `fallback_engine = None`).
- **S0c** -- the `eng_ltx_av.assert_usable` recipe/UNET mismatch gate.
- **S5** -- the operator's actual directive: "each video path has its own
  customized still operations". NOT met today (6 signatures, 19 engines
  sharing one, zero per-engine prompt differentiation). EVERY engine except
  the four `viz_*` needs its own real prompt text -- including the four
  `still_*` engines (where the still IS the whole picture, not an init frame)
  and the `mesh_stage` 3D option. **S5 CHANGES PROMPTS: its own acceptance,
  after the wiring, never inside a parity chunk.**
- **S4** -- two fresh-boot live legs (default route + forced HuMo bookend).

## Standing constraints on every chunk

- THE LAW: an audit may improve a story, never fail one for length, language,
  style, visual vocabulary or quality. Structural failures stay fail-closed.
- NO FALLBACKS: no substitute asset, no scene still as mesh fodder, no
  text-only or dark-floor degradation, no silent resize.
- Any node/widget/link change edits `workflows/otr_canonical.json` in the SAME
  commit, APPEND-ONLY at the end of `widgets_values`, and regenerates the 11
  committed variants + the 4 paired `.env.json` master hashes.
- Focused tests + FULL Windows suite + Bug Bible + AST/BOM/zero-byte/UTF-8 +
  canonical hash, then commit AND push to `v2.0-alpha`, `HEAD == origin`.
- Root-cause fixes only. No shims.

## Known open defect carried into this build (not a blocker)

`eng_ltx_video._use_i2v` contradicts fail-closed: with I2V enabled and the
init image missing it LOGS and degrades to text-to-video
(`eng_ltx_video.py:559-572`), while `render_driver.py:1801-1817` RAISES
`RenderError` "NO FALLBACK to text-only rendering" on that same state. Two
contradictory policies; whichever fires first wins. STATIC finding at HEAD --
needs a live reproduction before it becomes a PBUG row.

**r4 question 4:** does S0b-core's `when_ltx_i2v_enabled` freeze make this
contradiction WORSE, better, or unchanged? If the frozen flag is read by both
sites the contradiction becomes deterministic rather than order-dependent --
is that an improvement or does it just pick a winner silently?
