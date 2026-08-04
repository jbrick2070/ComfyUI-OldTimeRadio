# OTR Go-Forward Plan

**Newest update: 2026-08-03 (evening) -- THE ADAPTATION LANES WERE PERFORMING
FICTION ABOUT REAL BOOKS. THEY NOW PERFORM THE BOOKS.**

Branch `v2.0-alpha`. Operator ruling, hard: **`public_domain` and `shakespeare`
are FIDELITY lanes -- true to source, inventing nothing.** Supporting rulings:
"I'm open to multiple characters" (when a scene needs more speakers than the
cast widget asks for, THE SCENE WINS); "we don't need cast hints and other
wired variables for those"; "do its best to summarize the story into X words
per the user's selection"; and the web may be used to find more public-domain
sources.

## WHAT WAS WRONG (all verified against the real files)

* **The source text on disk was not the source.** The Wells fixture held 145
  words containing NO H.G. Wells -- invented modern prose wrapped in genuine
  `*** START/END OF THE PROJECT GUTENBERG EBOOK ***` markers, under which the
  bank stamped a Wells attribution. **A false attribution, and 20 episodes on
  disk were generated from it.** All fourteen Shakespeare fixtures were 93-125
  word collages (1,603 words for the entire selectable library), with lines
  reassigned between characters.
* **No pass that writes spoken words ever saw the source.** `_otr_compose_exchange`
  has ZERO source-text references; the interpreter compresses to a 2-3 sentence
  brief and every creative pass works from that, while the pack prompts order the
  model to "carry" words it is never shown. Hence Verona over Arden and
  "Arkham, Massachusetts" over Wells' Richmond.
* **Casting truncated the source's people by list order**, dropping Orlando from
  "Rosalind tests Orlando"; and the packs *instruct* the model to "fold or drop
  minor figures", so a code-only fix would have been silently undone.
* **`visual_style_policy: "derive_from_source"` is declared on every scene,
  schema-required, and read by nothing** -- the planned consumer module was never
  merged. That is why a Folger comedy rendered `archival_documentary`.

## WHAT LANDED (evening)

* **`scripts/otr_fetch_public_domain.py`** -- crawls once, vendors locally, keeps
  the render path offline. Gutenberg by id, Folger by play/act/scene. Provenance
  sidecar per unit (URL, timestamp, bytes, words, SHA-256 of the LF-normalized
  body, parsed speakers). Fails loud: refuses a body under 2,000 words as a stub,
  and refuses a scene whose cast it cannot parse.
* **Real Wells** (`d6c0bb7e`): authentic Chapter III, "The Time Traveller
  Returns". Proven through the bank's own fetch path -- Richmond, Filby, the
  Medical Man, the Psychologist PRESENT; Arkham, "pocket watch" ABSENT.
* **Real Folger, all fourteen scenes**: 24,243 words reaching the bank.
* **Two speech-prefix parsers fixed.** Folger marks speeches two ways and neither
  uses a colon (verse: name alone on its line; prose: `TOBY  Come thy ways`).
  The repo's `_speaker_from_line` REQUIRED a colon, so on real text it returned
  nothing and `payload_from_scene` fell back to the curated `cast_hints` -- the
  mis-ordered list that dropped Orlando. AYL 3.2 now reports ORLANDO, CORIN,
  TOUCHSTONE, ROSALIND, CELIA, JAQUES. Every gap the reviews predicted is
  confirmed real: LUCE, LEONATO, QUINCE, FABIAN, the NURSE.

## THE DESIGN, AFTER TWO KIBITZ ROUNDS + FABLE + SONNET

Hardened plan: `kibitz-runs/2026-08-03-adaptation-fidelity/r2/final.md`.
The keystone correction, which is NOT yet built: **compile source speech, do not
generate it.** A ledger row that merely POINTS at a source segment proves
structure, not meaning -- `PRODUCTION_SPRINT_LESSONS.md` lesson 11 already
documents that exact failure class. Source-owned text must be materialized
deterministically from an authenticated segmented artifact and verified against
it. "Summarize into X words" then means SELECTING WHICH REAL SEGMENTS FIT THE
BUDGET, not paraphrasing -- which also removes the VRAM hazard, since no model
sits in the source-speech path.

Settled by arithmetic: an episode cannot exceed **1,520 words** (19 voiced beats
at act_count 7, `BEAT_WORD_HARD_MAX` 80), so full-scene performance is
impossible without redesigning beat topology. Build target is the 300-word unit.

## NEXT, IN ORDER

1. **The segmented source artifact** (schema, spans, hashes, `body[start:end] ==
   segment.text`, omission receipts) and the pass-to-field ownership table --
   nothing else codes until that table exists.
2. **Cast from the selected cut.** Real scenes carry 3-12 speakers against a
   6-character ceiling (`_otr_casting.py` 1-6, `OutlineRequest` rejects >6), so
   which speakers appear must follow from the cut that fits the word budget.
   Coupled hard to the capacity guard: at act_count 1 there are exactly THREE
   voiced beats, so a 4-person cast is a mathematically guaranteed
   `CastVoiceCoverageError` -- the failure that killed `scifi_news` in the
   six-bank run. `compute_episode_budget` must also receive the TRUE locked cast.
3. **Loosen the count-match invariant** (`OTR_LedgerScriptWriter.py:4061-4067`
   hard-raises on any locked != requested) and change the pack text that tells
   the model to drop figures.
4. **Extend `_otr_provenance.py`** -- do not add a second attribution owner --
   and bind its output to the verified body hash.
5. **Schema migration** to retire `cast_hints` / `visual_style_policy`; both are
   still required by the validators and by `public_domain_manifest_schema.json`,
   so manifests and tests migrate in the same change.

## OPERATOR DECISIONS STILL OPEN

* The **20 public_domain episodes** built from the fabricated fixture --
  regenerate, relabel, or leave with the defect recorded? They must not count as
  adaptation evidence either way.
* The **superseded fabricated fixture** is still on disk, unreferenced, as the
  evidence. Deleting it is the operator's call.
* **PROD_BUG_LOG entries** for both live failures (repo policy requires naming
  the live artifacts before any Bug Bible promotion).

## KNOWN AND NOT FIXED

`canonicalize_shakespeare_text` truncates at 12,000 chars and the interpreter
sees only the first 5,000, so a 3,445-word scene reaches the brief as ~880 words,
silently. Belongs with the artifact work, where each beat is fed its own segment
rather than a blind prefix.

---

**Previous update: 2026-08-03 (afternoon) -- ORIGINAL MEANS "MAKE A RANDOM RADIO
DRAMA", AND THE LEDGER NOW TELLS THE TRUTH ABOUT IT.**

Branch `v2.0-alpha`, HEAD `de6b2ce2+`, pushed. Suite 8312 / Bible green.

## WHAT LANDED TODAY (afternoon window)

* **The premise catalog no longer runs on the original bank** (operator
  definition: "make a random radio drama" -- it may INVENT a premise and tag
  it, never CHOOSE from a list). Bank-level `defaults.story_scaffold: "off"` in
  banks.json, validated in `_otr_story_routing`, FOLDED into
  `_style_grammar_on` at the contract site so every scaffold branch takes the
  documented byte-identical off path. **Live-proven on
  `signal_lost_fogbound_rails`**: no `meta.style`, `story_style_status =
  story_scaffold_off`, `visual_plan.style` empty, `canon.sound_palette` empty,
  and cast bios story-native (a Timekeeper and a Station Master, not a
  hard-hatted miner).
* **Shape 4 + family-balanced markers** in `_canonicalize_transport_line`
  (whole-line wrappers, transport-gated), live-proven on `tempests_chart`.
* **Credits scroll doubled** (`_SCROLL_PPS` 60 -> 120, ~49s -> ~28s), operator
  confirmed the speed on a real episode.
* **The four-agent identity forensics** (cast chain / scaffold consumption /
  visual style / brief order) -- reports in the 2026-08-03 session; diagram
  delivered. Sweep report CORRECTED: seeds were NOT pinned (OS-entropy
  receipts); pinning requires env at SERVER BOOT.

## THE FIVE-BANK 30-WORD BEAT TEST (2026-08-03 afternoon; logs `tmp/_beat_*.log`)

5 legs, bank PINNED per leg, cheap viz engine, after the scaffold-gate fold:

| bank | result | credits style surface | reading |
|---|---|---|---|
| original (earlier, Fogbound Rails) | PASS | `story_scaffold_off` status | the fix, live |
| media_archive | PASS | `pirate_radio_resistance_drama` | MEDIA pool mechanically -- **content-FALSE**: the story is a film-reel standoff seeded by a real Library of Congress item on 'Midnight' (1939). Operator caught it on screen. Second specimen of the content-blind-draw class. |
| public_domain | PASS | `lamplit_road_and_threshold` | adaptation STAGING slug -- defensible by design |
| shakespeare | PASS | `lamplit_road_and_threshold` | same (4-pool collision is unremarkable) |
| scifi_news | PASS | `story_scaffold_off` status | **ANSWERS the dispatched-lane question**: dispatched lanes hardcode scaffold off, so their credits show the status token and never a premise -- and always have |
| scifi_news_pro | **FAIL** | -- | writer, THIRD distinct markup class: bare prose stage directions as unlabelled lines (`BAD_LINE_SHAPE` -> skeleton cascade). NOT shape-4 -- that fix held. Exactly the class the repair increments 1-5 target. |

Consequences: (1) the derived-tag + display-repoint increment fixes the credits
line for ALL SIX banks uniformly -- it is the highest-leverage item below;
(2) OPEN OPERATOR QUESTION: does media_archive want the catalog premise at all,
or the same scaffold-off treatment as original? Same two-generators-fighting
shape, but the rule so far was stated only for original; (3) the raw-text
fable2 lane still dies on markup classes increment 0 cannot reach.

## THE STYLE / IDENTITY DECISION WORK (next CODER window, one campaign)

Grounded by the forensics; every line has a file:line in the session traces:

1. **"Invent one and tag it"**: add a derived style/genre field to
   `run_story_brief_reflection` (`_otr_story_brief.py:446` -- proven
   content-loyal on both specimens), stamp beside `story_brief`, repoint the
   treatment `Style:` line (`video_engine.py:1762`) and the HUD
   (`video_engine.py:1336` -> `_build_left` `:1466`) at it.
2. **Rename `meta.style` -> `meta.story_scaffold`** (operator: too many metas;
   the field is neither scifi nor a description). Consumers move in one
   change: writer stamps, credits `_story_style_receipt`, `visual_plan.style`,
   `video_engine.py:1336`, tests.
3. **Ghost-name reconciliation fork**: pitch cast never reaches `lock_cast`
   (names are a pure pool draw; `source_character_names` deliberately None for
   invention lanes). Decide: scrub briefs after cast lock, or propagate pitch
   names. Evidence: Evelyn/Leonard as offscreen lore; Fogbound Rails bio still
   opens "Lizzie Gray".
4. **Dead fields found**: `ending_template` computed but zero LineRequest call
   sites pass it; `seed_policy.style_seed_env` validated but unconsumed;
   `dramatic_state` derived PRE-dialogue goes stale in the treatment.
5. **`meta` is a 120-key drawer** -- the cleanup the operator keeps asking for.
   Scope as its own rip with the ledger law (every field one owner).

## STILL OPEN (carried from the morning block below)

Writer scaffolding repair increments 1-5 (r3-corrected spec in
`docs/2026-08-03-script-parse-repair-CODE-READY.md`); the reuse detector to the
panel; section 0A carve-out ruling before M2 numbers move caps; Wan 2.2 I2V
checkpoint download + `wan_i2v` re-run; the `OTR_CastLock` freeze cascade
(`wan_ti2v`); TTS parenthetical stripping (documented as BUG-07.12, never
wired); credits shows `visual_style` never the scaffold (rides item 1 above).

---

**Previous update: 2026-08-03 -- M1 AND M2 ARE ANSWERED, AND BOTH ANSWERS ARE
"THE ASSUMPTION WAS WRONG".**

Branch `v2.0-alpha`, HEAD `350ab0f0`, pushed. Suite 8286 / Bug Bible green.

## THE SWEEP IS DONE -- 16 EPISODES, AND THE VIDEO LAYER NEVER FAILED

`docs/2026-08-03-SWEEP-30word-all-local-engines.md`. **13/17 legs passed, plus
the two HuMo legs before it, so 15 of 19 local engines published a real episode
in one night.** Every passing leg reports COVERS -- video meets or exceeds its
audio -- which is the no-mirror invariant holding across thirteen consecutive
episodes on the canonical path. Both randomizers ran live with seeds pinned at
4242 and the shipped recipes untouched, which is the live proof they have owed
since 2026-07-31.

**Not one of the four failures was an engine.** All four are upstream, and they
are four different causes, so `viz_mxc_cpu` and `ltx_audio_in` are UNPROVEN
rather than broken.

## THE NEXT CONCRETE ACTION

1. **Fix the markup parser -- the one real code defect the night found.** The
   writer emitted `**SCENE 5**`-style markdown headings and the parser read each
   as a speaker (`UNKNOWN_SPEAKER: **SCENE 5`, then `**MUSIC`, `**CODA`),
   exhausting all four ladder attempts and killing the `ltx_audio_in` leg. Then
   re-run that leg.
2. **Re-run `viz_mxc_cpu`** -- its writer failure was a different cause (an
   invented cast member, `DR. MOURKIOTI`) and non-deterministic.
3. **Download `wan2.2-i2v.safetensors`**, then re-run `wan_i2v`. Its failure was
   the fail-closed contract working: the checkpoint is simply absent.
4. **Investigate the `OTR_CastLock` freeze cascade** that stamped
   `freeze_verdict='needs_full_rerun'` and killed `wan_ti2v`.
5. **Do NOT build the HuMo pre-roll fix.** M1 killed its premise -- see below.
6. **The reuse detector goes to the panel before it is touched again.** Two
   strikes are already spent on it (three false-positive classes in one night).
7. **The section 0A carve-out ruling is still owed** before any M2 number moves
   a cap, tier or profile.

**Writer and cast failures cost 3 of 17 legs (18%). That is the highest-value
thing to harden for unattended runs -- higher than anything in the video layer,
which did not fail once.**

## WHAT M1 AND M2 SETTLED

**M1 -- the lip-sync premise fails, twice over.** `BUG_BIBLE.yaml` entry
BUG-07.13 claims audio leads the lips by a CONSTANT 100-200 ms, every clip,
every episode. Its stated CAUSE, a 3-6 frame leading freeze, is absent from
two-thirds of 27 production clips (median 0). Its SYMPTOM does not appear
either: of 20 measured segments, **zero** land in +100..+200 ms at any
confidence gate, and the sign is predominantly OPPOSITE -- about -30 to -60 ms,
one frame with the video slightly ahead. **The prescribed pre-roll would move
sync the wrong way.** Rewrite BUG-07.13 rather than implement it.
Docs: `docs/2026-08-02-MEASUREMENT-humo-static-onset.md` and
`docs/2026-08-02-MEASUREMENT-M1-humo-lipsync-offset.md`.

**M2 -- peak VRAM FALLS as frames rise, and the orientations match.** 16 cells,
both orientations, cold and warm, server restart before every cold cell: 49 ->
97 frames costs about a GIGABYTE LESS peak, in all four series, while render
time stays linear. Orientation deltas are 374/47/1/65 MB at 49/65/81/97 against
a 290 MB repeatability -- no consistent difference. **97 is a QUALITY bound, as
`eng_humo.py:106` always said; it is not buying memory safety.**
`docs/2026-08-02-MEASUREMENT-M2-humo-vram-ladder.md`.

**READ THE CORRECTIONS BLOCK IN THAT DOC BEFORE QUOTING IT.** Four claims were
withdrawn under review: it is a RENDER-WINDOW peak (the probe starts after
`prepare()` loads the handles), the coverage-splitting recommendation does not
follow (production reuses handles once per BEAT; the ladder used fresh sessions),
the "1 in 331,000" statistic assumed independence a fixed ascending order does
not provide, and **the whole ceiling-breach framing is withdrawn** -- ComfyUI
stages 16,531 MB against a 16,303 MB card, so a peak near capacity is a dynamic
loader working, not a near miss. Operator: recipes have been stable, no OOM.

## OPERATOR DIRECTIVES FROM THIS WINDOW (hard)

* **The recipes are not on the table.** "We spent a lot of time perfecting the
  recipes to look good and we can't lose that." No VRAM, speed or cap finding
  justifies a recipe change; measurement runs the SHIPPED recipe unchanged. This
  specifically forbids reading "peak falls as frames rise" as a reason to raise
  the 97 trained-length cap, and makes the deferred no-LoRA HuMo control a
  recipe change rather than a control.
* **Per-segment rendering is BY DESIGN** -- "each audio clip takes its own
  journey, to keep VRAM low." Never classify an assembled beat as one render.
* **The ending credits roll should be halved**, ~49 s -> ~25 s. Check whether the
  duration is a scroll rate, a per-entry dwell or a fixed clip length, and keep
  the text legible at 2x. (The 35-48% share measured on sweep legs is a 30-word
  test artifact -- "30-word episodes are an anomaly anyway" -- do not quote it as
  a production problem.)

## STILL OPEN

* The reuse detector cannot separate a deliberately quiet shot from a duplicated
  frame; it is ADVISORY in `otr_w45_campaign.py` until that is solved. The
  engine-layer and composite guards (`MirrorExtensionForbidden`,
  `ClipUnderrunsItsBeat`) are terminal and unaffected.
* `docs/2026-08-02-IDEA-hardware-compatibility-matrix.md` -- captured, not
  scoped. Includes the Mac research: Metal has no `Float8_e4m3fn`, ComfyUI+MPS
  video is impractical (82 min for a 2-second clip), Draw Things and MLX are
  ~100x faster and DO support LTX-2.3 with joint audio, and the `viz_*`/`still_*`
  lanes need no GPU at all.
* `humo_1.7B` and `ltx_8gb` are marked CUDA-only with no fp8, no fp4 and no
  stated reason. Unexamined, not proven.
* M2's raw rows sit in swept `tmp/` with no pinned digest or config manifest.

---

**Previous update: 2026-08-02 -- THE MIRROR IS DEAD, THE HUMO CAP FOLLOWS THE
MODEL, AND THE NEXT ACTION IS GPU VALIDATION.**

Branch `v2.0-alpha`, HEAD `392a86f7`, pushed, suite **8289 passed / 0 failures**.

## THE NEXT CONCRETE ACTION (for the `/rc` window with the GPU)

Validate today's spec changes on the real GPU, **locals first**, before any
campaign. Three jobs, in this order:

1. **M1 -- classify the lip-sync error.** `BUG_BIBLE.yaml:2343` says HuMo audio
   leads the lips by 100-200 ms with the face static for the first 3-6 frames.
   **The prescribed fix cannot be built until this is classified**, because
   pre-roll + equal trim is algebraically a NO-OP if the lag is constant rather
   than onset-only. Render clear frontal lines with sharp plosive onsets, mux a
   zero-based CFR-25 / 16 kHz diagnostic, read the offset in EARLY, MIDDLE and
   LATE speech windows (validate the sign on a deliberately shifted control
   first). Early-only -> pre-roll fix. Constant -> advance the 25 Hz conditioning
   features instead. Growing -> a rate/timestamp bug, not a pad. Run a matched
   no-LoRA control: Kijai reports the lightx2v distill is not fully HuMo-
   compatible, so the defect may be ours rather than HuMo's.
2. **M2 -- the paired HuMo ladder**, `49 -> 65 -> 81 -> 97` in BOTH orientations,
   everything else identical, cold and warm recorded separately. HuMo now carries
   a `VramPeakProbe` (it was the only heavy lane without one), so this finally
   produces data. **Prediction: the two orientations match at every rung** --
   equal pixels, equal token grid. If they do not, the defect is tiling or kernel
   selection, not the model. Stop at the first rung breaching 14.5 GiB cold.
3. **The 30-45 word randomizer, one episode per LOCAL engine.** Cloud cannot run
   (no keys, offline-first) and is qualified statically only.

**M3 is already answered -- do not spend GPU hours on it.**
`ltx23_16gb_audio_in` has rendered **497 frames at 832x480, peak 12,999 MB**, and
79 production samples show peak is FLAT against frame count (the 400-499 bucket
has the LOWEST mean). See `docs/2026-08-02-MEASUREMENT-ltx-av-vram-vs-frames.md`.

## WHAT CHANGED TODAY -- the invariants a GPU run will exercise

* **No mirrors, no ping-pong, no boomerang.** `extend_frames_to_target` DELETED;
  `fit_frames_to_target` lost `allow_mirror`; `OTR_LTX_LOOP_VIA_REVERSE` is inert
  for every value. A short render is TERMINAL. Coverage planning splits instead.
  Credits untouched (black background, called neither path).
* **The HuMo 14B cap follows the CHECKPOINT, not the orientation** -- both 14B
  routes share 97 (the trained length); the 1.7B pair declares its own 33-177 and
  stays uncapped. `humo_14B_169` drops from TEN cuts to five on a 17.68 s beat.
  **97 is a reasoned bound, not a measured ceiling** -- that is what M2 settles.
* **One-segment coverage plans now execute as coverage.** The router asked
  `is_multi_clip`; a one-segment plan owing a tail trim rendered its surplus and
  kept it. Same fix applied to `BeatSession`'s discriminator and to the
  per-segment AUDIO window.
* **VRAM admission boundary** before `prepare()`, free VRAM read once with no
  hoist correction. It enforces NOTHING today and says so: `QUALIFIED_COST_ROWS`
  is empty, because `wan_ti2v`'s row is present but disqualified and would refuse
  every ordinary beat.
* **Per-segment video seed** (segment 0 unchanged, so archives re-render
  identically).
* **`ENGINE_MATRIX.md` is generated** with effective canvas, multi-clip maths and
  an EVIDENCE column. It found five adapters citing docs that are not in the repo.

## KNOWN OPEN -- do not rediscover these

* The admission guard covers coverage-executed beats only; the single-clip path
  returns via `render_shot()` first, and `ltx_audio_in` is not in
  `PLANNING_CAP_ENGINES` -- so the hottest-peaking engine is unguarded.
* `FRAME_COST_MODEL` is keyed by engine NAME while recipe/quant/LoRA/reserve are
  env-configurable; a measured row needs a calibration IDENTITY.
* Four adapters still cite missing receipts (`ltx_audio_in`, `mesh_stage`,
  `viz_green`, `viz_mxc_mandala`).
* The lip-sync onset fix is SPECIFIED but unbuilt, blocked on M1.
* Cap authority is not yet collapsed to one (`video.max_render_frames` should be
  sole; env twins must be absent-or-equal).
* `otr_w45_campaign.py` runs SIX engines while claiming all local ones, and its
  acceptance would not reject a mirror. Fix before trusting a campaign result.

## RESET BEFORE ANY HEADLESS RUN

Per `CLAUDE.md` section 4: kill selectively by CommandLine (never a blanket
python kill -- it severs the MCP tools), confirm port 8000 is not listening and
`nvidia-smi` is back to ~1.5 GB, then boot fresh.

---

**Previous update: 2026-07-31 (CODER closeout) -- THE TWO RANDOMIZERS ARE LANDED
AND PUSHED, AND THE LANE AUTHORITY LEFT THE WRITER.**

Operator directive: `source_bank` and `visual_style` are **TWO SEPARATE
randomizers that can be turned on or off individually**. Both ship.

Each dropdown carries its OWN roll sentinel as choice 0 -- `roll (any eligible
bank)` and `roll (any style)`. Picking the sentinel rolls that surface; picking a
concrete row pins it. Roll both, one, or neither; neither roll can enable the
other. Because the sentinel is a UI COMMAND prepended to an existing combo,
there is **no new widget, no positional `widgets_values` shift (BUG-LOCAL-097 is
not engaged), and ZERO canonical-JSON diff** -- a graph persists the selected
VALUE, never the choice list, and the shipped defaults are still `scifi_news` /
`sci_fi_radio`. `workflows/otr_canonical.json` is therefore correctly untouched;
the guardrail test proves it.

Mechanics (`nodes/_otr_rolls.py`, pure, ZERO LLM calls): eligibility ->
sorted-by-id pool -> seeded draw -> receipt. Bank eligibility is TWO filters and
only two -- `bank.runnable` and the lane's declared request compatibility (no
rights filter: operator ruling 2026-07-12 stands, and `banks.json` is untouched).
Style eligibility is EVERY registered style, by design not omission: a style has
no execution lane to be missing. Pools come from the LIVE registry, so an
activated client bank is an ordinary peer. Separate seeds -- `OTR_BANK_SEED` and
`OTR_VISUAL_STYLE_SEED` (NOT `OTR_STYLE_SEED`, which is narrative arc shape) --
so either roll replays alone; a malformed override RAISES. Receipts land at
`meta.bank_roll` / `meta.style_roll` and are ABSENT (not null) on a manual pick.
Sentinel + a pinned `source_ref` is refused loudly: a pinned source belongs to
one bank.

Supporting rip, same commit: the pipeline->runner table left the writer for
`nodes/_otr_lane_specs.py` -- `_RUNNER_BY_PIPELINE`, `_LEGACY_INLINE_PIPELINES`,
`_resolve_lane_runner`, `_run_fable2_lane` and `_run_scifi_codex_lane` are GONE,
not aliased (a test asserts the writer holds no second table). It stores NAMES
resolved lazily, so runner modules still stay out of ComfyUI startup. It also
carries each lane's request-compatibility policy through two entry points with
deliberately different contracts: `assert_supported` (writer gate, re-raises the
lane's NATIVE error unwrapped) and `is_roll_compatible` (roll filter, bool,
catches ONLY declared errors so a broken runner propagates instead of silently
shrinking the pool). Today exactly one lane is constrained --
`scifi_news_circuit`'s 30..900 band, hoisted out of the runner as
`assert_supported_target_words` sharing `WordSteerV4` as the one source of truth.
That is a CAPABILITY statement, not a length verdict; we still never chase word
count.

**Known hazard, written down on purpose:** the rolls resolve once per `run()`
entry, and there is no refine re-entry at HEAD (the refine machinery is gone;
`refine_target_grade` is an inert widget). Whoever rebuilds a loop that re-enters
`run()` MUST carry these receipts back in and short-circuit, or every pass
re-rolls and the ledger records a bank the episode never used. Both kibitz
panelists found exactly that bug in the r2 draft.

**CORRECTION to the 2026-07-30 closeout below:** it lists "`scifi_news_pro_multipass`
still has its separate 3,600-character dossier cap; complete-source Pro support
remains an explicit follow-up". That is STALE -- commit `33e6a276` ("Read
complete Pro sources in bounded windows", +573 runner / +739 test lines) closed
it. `_DIGEST_CHAR_CAP` is now the WINDOW SIZE of an overlapping window set whose
coverage of the selected body is PROVEN (`_validate_dossier_windows`, and tests
`test_digest_windows_cover_complete_source`,
`test_partial_window_coverage_fails_before_any_model_call`,
`test_old_prefix_counterfactual_loses_tail`), not a source clip. No rip is owed.
Live GPU requalification IS still owed.

**CURRENT STEP (operator directive 2026-07-31, revised same day):
GET WAN 8-GB READY *FIRST*, THEN the 30-word randomizer sweep.**

**JUDGMENT OF RECORD (2026-07-31 @ `aff09bde`, Codex as final judge):
`docs/2026-07-31-wan-8gb-adversarial-review/report.md`. READ THAT FIRST** -- it
adjudicates this window's research against the actual repo and corrects it in
four places. Headline rulings:

- **A broken estimator does NOT prove Wan is impossible on 8 GB.** The
  2026-07-23 failure this window cited was **177 frames on a 16 GB RTX 5080**,
  not a 17-frame test on physical 8 GB. The arithmetic holds as arithmetic about
  the ESTIMATOR; it has never been observed on real 8 GB hardware.
- **The proximate blocker is the WRITER, not the video estimator.** The
  configured writer is refused at ~8.13 GiB against the profile's declared 6.8 GB
  ceiling, so the canonical 8 GB profile cannot reach Wan at all today.
- **Stock GGUF misses AIMDO Dynamic VRAM but STILL has legacy partial
  loading/offload** -- this window overstated it. The official ComfyUI path is an
  **FP16 UNet + scaled-FP8 encoder**, not fp8 safetensors throughout.
- **The canvas-authority bug is real but does NOT cause today's refusal.**
- **Qualification must cover EVERY heavy pipeline phase with fail-closed cleanup
  receipts** -- the render-only 4-cell sweep this window proposed would qualify a
  tier that still dies in the writer.
- **DO NOT lower the guard, DO NOT promote 14B, DO NOT call the tier qualified.**
- **FastWan 5B** (`FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`) is the
  leading potentially shippable answer to the 5B's missing step-distillation.
  Turbo-GGUF is research-only (CC BY-NC-SA) with a non-standard sampling
  contract -- do not ship it.

Supporting research, now read SECOND and with the corrections above applied:
`docs/2026-07-31-low-vram-video-research.md`, then the parameter analysis.
Three independent passes (this window, a live web-research pass, and the
operator's ChatGPT pass) converged: **our VRAM estimator is the wrong SHAPE.**
`overhead + per_frame*frames` is a CO-RESIDENT model; the real low-VRAM technique
is STAGED, so peak is `max(text_encode, image_encode, sample, vae_decode) +
reserve`, not a sum. Proof that addition is wrong: ComfyUI's official Wan 2.2
workflow is ~18 GB of model files and its own docs say it "fits well on 8GB vram
with the ComfyUI native offloading".

**And the finding that outranks everything else: our GGUF opts OUT of Dynamic
VRAM.** `ComfyUI-GGUF` defines `GGUFModelPatcher(ModelPatcher)`, NOT
`ModelPatcherDynamic` (Comfy-Org/ComfyUI#13953, 18 May 2026), so our
`Q5_K_M` UNET + encoder run on the legacy 2025 path while the official 8 GB
workflow that "fits well" uses fp8 scaled SAFETENSORS -- which do get the dynamic
patcher. Also: `--lowvram` is now INERT when Dynamic VRAM is on.

KEEP Wan 2.2 TI2V-5B -- Wan 2.5/2.6/2.7 are not open weights (verified against
the Wan-AI HF org; the "Wan 2.7 local" articles are SEO fiction), and LTX-2.3 is
22B + a Gemma-3-12B encoder (~20 GB of weights, 12 GB community floor, and
ComfyUI rates it 5.5 vs Wan's ~1.38 on activation cost). The problem was never
the model.

Revised order: (1) cache text embeddings -- `WanVideoTextEncodeCached` already
exists with `use_disk_cache=True` by default, and OTR's bounded repetitive prompt
set is the ideal case; it removes 3.9 GiB (the umt5 encoder is LARGER than the
UNET) plus minutes per leg; (2) A/B fp8 safetensors vs our GGUF -- verified
mechanism, unmeasured consequence, possibly the biggest structural win;
(3) replace the estimator with the max-over-stages shape; (4) declare
`render_canvas` (still unfixed, still 3.07x); (5) then measure.

**Nobody anywhere has published a real peak-VRAM number for Wan 2.2 TI2V-5B on an
8 GB card. We are the best-positioned party to measure and publish it.**

**The older ceiling-first analysis is
`docs/2026-07-31-wan-8gb-parameter-analysis.md` -- its canvas and t5_device
findings still stand; its framing of the cost model as merely miscalibrated does
not.**
The headline: with the shipped cost model
(`FRAME_COST_MODEL["wan_ti2v"] = (7000.0, 185.0)`, margin 0.85) an 8 GB card
needs **9,442 MB free** to render even the 17-frame motion floor at 832x480 --
more than the card HAS. The overhead term alone is 8,235 MB. The tier raises
`MotionBudgetError` before rendering a frame, which is exactly the recorded
2026-07-23 failure.

Three real defects, in order:

1. **`wan_ti2v` declares NO `render_canvas`**, so a plain canonical run falls
   through to `render_driver.py:2494`'s `1472x832` default -- **3.07x the
   intended pixels**. `ltx_8gb` fixed this class in B5 by declaring
   `render_canvas = (512, 288)` statically. **Recommend declaring 768x432**: it
   is exactly 16:9 and /16-clean, and 17% cheaper than 832x480 (which is 26:15
   and pillarboxes).
2. **No `t5_device` knob exists.** `umt5-xxl-encoder-Q5_K_M` is 3.861 GiB --
   LARGER than the 3.549 GiB UNET; all three required weights total 8.722 GiB.
   ltx MEASURED that this is the decisive lever (t5 on GPU -> 16.0-16.1 GB peak,
   i.e. an 8 GB box does not render). Porting `t5_device: "cpu"` is the single
   highest-value change available.
3. **The cost model is ONE 2026-06 data point** (10,277 MB @ 17 frames @
   1472x832) taken on a 16 GB card with machine-wide NVML. It charges 7,000 MB
   overhead for a path whose peak resident weights are ~4,980 MB; the ~2 GB gap
   has no provenance. Must be re-fit from a real sweep.

Order: (1) declare the canvas + pin the profile with a test; (2) add the
`t5_device` knob defaulting to `cpu`; both are OFFLINE code. Then (3) the 4-cell
clamped sweep (`OTR_HEADLESS_RESERVE_VRAM_GB=8`) copying ltx's B6 shape --
judge on the SPREAD, not the minimum -- and (4) re-fit the cost model and set the
tier ceiling from it (17 stays until the data says 33).

Sampler settings need no change: `steps=30` is a TIME knob and will not make the
tier fit; `cfg=5.0` / `shift=5.0` are correct for a non-distilled 5B (do NOT copy
ltx's `cfg=1.0`, that is a distilled-model artifact).

**THEN, once WAN renders on 8 GB: RE-TEST EVERY LOCAL VIDEO MODEL AT 30 WORDS
WITH THE RANDOMIZERS ON.** RENDER window. It doubles as the live proof both
randomizers still owe, and 30 words is the cheapest leg that exercises the whole
chain.

**PIN THE SEEDS -- this is the difference between a sweep and a mess.** If both
rolls run free, the bank AND the visual style change per leg, and no difference
between two legs can be attributed to the video engine: that is the channel-
isolation failure `PRODUCTION_SPRINT_LESSONS` s12 exists to prevent. Set
`OTR_BANK_SEED` and `OTR_VISUAL_STYLE_SEED` to fixed values for the sweep. The
roll still runs end to end -- sentinel resolved, pool built from the live
registry, draw executed, `meta.bank_roll` / `meta.style_roll` stamped -- but
every leg draws the SAME bank and style, so the ENGINE is the only variable.

Order of operations:

1. **One unseeded pilot leg first**, to prove the entropy path and the replay
   contract: assert `draw(receipt.eligible_order, receipt.seed) == receipt.selected`
   for both surfaces, and confirm both receipts are present with
   `seed_source = "OS entropy"`.
2. **Then the seeded sweep** across every local video engine at 30 words.
   RE-PIN THE ROSTER AT HEAD from the engine registry -- do NOT trust a
   remembered count (past campaigns were variously described as 18 and 19
   engines, and the roster has changed since).
3. **One leg with a fidelity bank pinned** (`shakespeare` or `public_domain`,
   picked directly, not rolled) to close the OTHER outstanding live proof:
   no LEMMY in the locked cast and
   `meta.cast_contract.lemmy_policy == "source_fidelity_exclusion"`.

Per-leg acceptance is the standing bar: `RESULT SUCCESS` + `obs_publish OK` +
the asset on disk at `otr/episodes/<ep>/`, final under `otr/obs/`. Word count
NEVER gates a leg. Reset the box per CLAUDE.md s4 before every headless run
(selective CIM kill by CommandLine, never a blanket python kill).

Record per leg: engine, resolved bank + style (from the receipts, not the
widget), both seeds, render time, peak VRAM, and the asset path. That table is
the deliverable.

**Deferred by this directive, not cancelled:** the WAN 8-GB block (already
re-grounded as code-complete / proof-incomplete, blocked on one operator
decision -- see OPEN BUGS) and the SFX engine lane
(`docs/2026-07-31-sfx-engine-lane-SPEC.md`, contracts settled, blocked on a
checkpoint that does not exist on this host). Neither is a keyboard item.

---

**2026-07-31 (CODER closeout) -- TWO OPERATOR DIRECTIVES FROM THE
320-WORD TEST ARE LANDED AND PUSHED.** Commit `e577f9ef`; HEAD == origin on
`v2.0-alpha`.

(1) **Source-faithful banks never roll the Lemmy cameo.** `public_domain` and
`shakespeare` -- plus their bake-off `_v2`/`_v3` variants, normalized through
`base_source_bank_id` -- force `lemmy_hit=False` inside
`assemble_pre_locked_rows`, AHEAD of both the OS-entropy ~11% roll and the
`force_lemmy` branch, so the exclusion overrides the operator-facing
`lemmy_cameo` widget as well as the roll. LEMMY is also filtered out of any
source-supplied character list. The writer threads
`_source_bank_row.source_bank_id` into `lock_cast` and stamps
`cast_contract.lemmy_policy`. Bark-voice replay is unaffected:
`replay_voice_assignment` already replays the frozen `lemmy_hit`.

(2) **P3 stops reserving output capacity.** The fixed
`_RADIO_SCORE_CONTEXT_CAP_TOKENS = 8192` / `_RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS
= 1829` reservation is RIPPED -- that is chasing a word count by another name.
RadioScoreDraft now runs on `ProviderCapacityMessages` with
`max_new_tokens=None` and reports `output_budget_mode=provider_capacity`.
Finiteness comes from the structural graph bounds (scenes<=3, shots<=2,
beats<=4), never a prose length gate. OpenRouter `finish_reason=length` now
raises the typed `PromptContextOverflowError(phase=output_limit)` -- after
reasoning-tag stripping -- carrying the partial completion and token count, so a
truncated artifact is re-rollable capacity signal rather than JSON-repair input.

Receipts: focused **181 passed**; full Windows suite **7966 passed / 130 skipped
/ 1 xfailed**; Bug Bible **17 passed / 24 skipped / 3 xfailed**; UTF-8 / no-BOM /
nonzero / AST green on all eight touched files. No workflow JSON, node, widget,
link or schema was touched, so canonical `otr_canonical.json` is unchanged BY
DESIGN (pure code + tests). No GPU/headless run, no new PBUG (offline root fix).

**CURRENT STEP:** this chunk is closed. Both directives are suite-proven only --
neither has run LIVE. The natural next proofs are a `shakespeare` or
`public_domain` leg (confirm no LEMMY in the locked cast and
`cast_contract.lemmy_policy == "source_fidelity_exclusion"`) and a long-form
sci-fi leg through the now-unreserved P3. Preserve the three pre-existing
modified `tmp/*.ps1` files and the untracked `config/profiles/otr_*.json` set.

**Newest update: 2026-07-30 (CODER closeout) -- COMPLETE-SOURCE SCI-FI
AUTHORING AND CANDIDATE-LOCAL RECOVERY ARE LANDED AND PUSHED.** The operator's
controlling contract is now implemented: the RSS/article is a factual
springboard, while the accepted canonical ledger is the audible downstream
authority. Fiction may replace an abandoned draft completely. Facts represented
as facts and the factual coda still require validated evidence. Recoverable
JSON/schema/content/safety/output-limit defects retire a candidate, not the
episode; fresh complete model-authored candidates continue until acceptance or
operator cancellation. No canned story, patch-in-place ledger, or fixed outer
candidate ceiling was added.

Code commit `3bc3d8a0` preserves the complete selected RSS/article body, chooses
the richest usable RSS row and richest available body, records exact
route/index/count/byte/hash provenance in the actual episode ledger, and removes
the local 12,000-character article clip. P0 covers long RSS text with overlapping
windows, validates locally, rebases exact coordinates, merges deterministically,
and validates against complete A0. P0/P1/P2/P3/P5 use fresh-candidate recovery;
rejected prose is excluded from prompts and journals; P5 validates raw and
canonicalized text before acceptance; the production ledger is assembled and
stamped once from the winner.

Receipts: focused **242 passed**; seven production mutations fired red and were
restored; full Windows suite **7936 passed / 130 skipped / 1 xfailed**; read-only
Bug Bible **17 passed / 24 skipped / 3 xfailed**; variants **11 / 0 failures**;
UTF-8/no-BOM/nonzero/AST/diff hygiene green. Canonical workflow SHA-256 remains
`9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`.
The four-round live panel record and final plan are under
`docs/2026-07-30-story-never-fails/`; actual panel spend was **$1.3127**.

No GPU/headless campaign, Window B/degrade implementation, workflow/frozen
artifact rewrite, or survival-guide edit was performed. This was an offline root
fix, so no new PBUG was admitted and older live PBUGs are not claimed
requalified. `scifi_news_pro_multipass` still has its separate 3,600-character
dossier cap; complete-source Pro support and live GPU requalification remain
explicit follow-ups. Preserve unrelated dirty/untracked work and the three
pre-existing modified `tmp/*.ps1` files.

**CURRENT STEP:** the implementation is closed. The plan of record is
`docs/2026-07-30-story-never-fails/FINAL_PLAN.md`. A later explicitly scoped
window may do live GPU requalification and the separate Pro whole-source
follow-up; do not infer authority here to start Window B/degrade.

**Newest update: 2026-07-30 (CODER closeout) -- ITEM 5'S LIMITED RSS
`full_text` COORDINATE MIGRATION IS LANDED AND PUSHED.** The operator directed
that the story engine produce the story and fill the ledger as best it can
without failing; this was the approval gate for the recommended future-only
root fix. Code commit `331f46ea` adds block-aware extraction only at
`content[0].value -> rss_full -> full_text`, so real RSS paragraph/list/break
boundaries no longer fuse into damaged evidence. Inline markup still joins
without invented spaces (`H<sub>2</sub>O`, `anti-<em>microbial</em>`), literal
entity spellings such as `&nbsp;` remain untouched, and resulting whitespace
is collapsed.

The migration may change new live `source_digest` values and downstream P0
coordinates, previews, reranking, inline-RSS choice, and content-floor
decisions for clients sharing the fetcher. That is intentional and applies
only to future fetches. Frozen ledgers and snapshots were not migrated or
re-pinned. The similar summary extraction, derived `seed_text`,
`_normalize_span_source_text`, URL scraping route, and canonical workflow were
not changed.

Receipts: focused seam gates **94 passed**; the production-call mutation made
the firing test fail, then restoration returned
`nodes/story_orchestrator.py` byte-identically to SHA-256
`2D076104E80278CC3F9969342EE6D24D9BDE8DC9D940F63EC1CB580FBB8E84F6`.
Full suite **7898 passed / 130 skipped / 1 xfailed**; Bug Bible **17 passed /
24 skipped / 3 xfailed**; `build_variants --check` **11 variants / 0
failures**. Canonical workflow SHA-256 remains
`9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`.
No GPU campaign, headless render, Window B/degrade work, survival-guide write,
or frozen-artifact rewrite was performed.

**CURRENT STEP:** Item 5 is closed. Stop this coder window here; do not infer
authority to begin Window B/degrade or a GPU campaign from this closeout.
Preserve the survival-guide repo's unrelated dirty work and the three
pre-existing modified `tmp/*.ps1` files.

**Updated:** 2026-07-30 (remote Cowork, CODER window) -- **WRITER REPAIR:
SECTION 2 IS MEASURED, AND WINDOW A IS THREE CHUNKS IN.** HEAD == origin
`41683fc9`; suite **7865 passed / 130 skipped / 1 xfailed**; Bible 17;
`build_variants --check` 11 variants / 0 failures; canonical `9872624A`
byte-identical (no node, widget, link or schema touched). Landed: `fb400526`
**A-1** (the output-limit raise carries the completion and the arithmetic it was
raised about), `f781234c` **A-3** (the narrow `&nbsp;` decode, with both
production fixtures and a digest-stability pin), `41683fc9` **A-4** (a capacity
failure carries a PHASE, and an `output_limit` phase spends the ladder budget it
was always advertised -- PBUG-20260729-02). **A-2 is folded into A-1** (its
premise was a miscount; see below). **NEXT = A-5.** Section 2's finding is below
under CURRENT STEP; per-chunk detail is in `docs/HANDOFF_LOG.md`, not here.

**Superseded header (2026-07-28 06:15, RENDER/QA window) -- **THE
CREDITS CARD IS LIVE-PROVEN AT 1920x1080 AND A CLOUD ENGINE WAS FOUND SITTING
IN THE "LOCAL ONLY" ROSTER.** HEAD == origin `72282083`; canonical
`9872624A` unchanged (pinned by hash at campaign launch). No production code
was changed by this window -- the work was the GPU lane harness (`tmp/`) and
the kibitz r1-r4 arc, so the suite/Bible numbers below stand from `1959fb49`.

**LIVE PROOF EARNED TONIGHT (first ever for the five encoder chunks):** a full
canonical leg on `still_flat` published green --
`[OTR_CreditsRoll] appended 52.0s console (hero='THE FROST ON THE GLASS')` then
`obs_publish OK`, 14,637,297 bytes in `otr/obs/`, `engine_histogram
{"still_flat": 7}`, VRAM peak 1712 MB. The credits col1 ladder (`24f4251a` +
`1959fb49`) has now rendered live at the production canvas. `OTR_CreditsRoll`
is CONFIRMED on the executed canonical path (it is node 95 and it both failed
and succeeded live).

**GPU LANE CAMPAIGN RUNNING** as of 06:06:46 -- master
`tmp/gpu_lane_all_models_20260728_060646`, 18 local engines over 4 lanes,
frozen controls (`z_image_turbo` + `original` + `sci_fi_radio`, 45 words),
harness pinned by SHA-256 in `harness_pins.txt`. Expect ~6 h. Verdicts land in
each lane's `lane_summary.json` and the master `campaign_summary.json`.
Read `tmp/_kbA_gpu_campaign.done` for PASS/FAIL. **These cases are engine
COVERAGE and are NOT an 8 GB qualification.**

**Prior header (still true for the code):** ONE STREAMING CLIP ENCODER INSTEAD
OF THREE, BOTH PROOF HALVES ENFORCED, THE GEOMETRY DOWN TO ONE DERIVATION, AND
THE END CARD NO LONGER OVERFLOWS THE CANVAS THE BUILD SHIPS. At `1959fb49`:
suite **7464 passed / 27 skipped / 1 xfailed**; Bible 17; `build_variants
--check` 11 variants / 0 failures.

**OPERATOR RULING 2026-07-28 -- THE STILL FLOOR IS ALLOWED, BUT ONLY WHERE THE
MATH IS IMPOSSIBLE.** Operator, verbatim in substance: "if mathematically there
is no way to get moving video, we could use a still in place. But ideally we
want to divide the beat into enough different clips so that we can have video
for the whole beat."

The gate is ARITHMETIC, never convenience and never engine failure. A still
floor is permitted for a beat if and ONLY if `coverage_plan.partition_beat`
can produce NO legal plan, which is exactly three cases:
  1. target frames < the engine's `min_frames` (no legal clip exists at all);
  2. the cover would need more segments than the planner's ceiling;
  3. a `discrete_frames` menu engine with no exact cover for the target.
In every other case the beat is DIVIDED into enough legal clips to carry video
for its whole duration. That is the preference, stated by the operator.

**THIS RULING DOES NOT COVER THE SIX 2026-07-28 FAILURES.** HuMo's 184-frame
beat has a legal two-clip partition; wan's two-segment split is legal
arithmetic. Those are MISSING CAPABILITY (no per-segment audio slicer, no
`session_identity()`), not impossible math, and they get FIXED, not stilled.
Any future window that reaches for the still floor to close them is reading
this ruling backwards -- a fallback that fires on "the engine refused" is the
silent degradation this build exists to remove, and the existing `NO FALLBACK`
refusals plus the image-phase row's "no text-only or dark-floor degradation"
stay in force everywhere the math is satisfiable.

**OPERATOR RULING 2026-07-28 -- EVERY AUDIO-IN BEAT GETS A STILL WITH A MOUTH.**
Operator: "all audio in video should have a still with a mouth so the audio-in
works as expected."

Any beat routed to an engine whose `required_inputs` contain `audio_ref` --
`humo`, `humo_1.7B`, `humo_1.7B_169`, `humo_14B_169`, `ltx_audio_in` -- MUST
receive a still that carries a visible face/mouth. The audio-in lane animates a
mouth against speech; a still without one gives it nothing to drive. This is a
STILL-CONTRACT requirement, not a nice-to-have, and it is the operator's answer
to the Fable seat's proposal that the show keep the camera off the speaker
entirely. **That proposal is CONSIDERED AND OVERRULED** -- lip sync stays in
the show, so the HuMo lane stays load-bearing and the per-segment audio slicer
is NOT descoped.

**THE LIPS MAY BE A PERSON *OR* A RADIO** (operator refinement, same day). The
audio-in engine needs a mouth-shaped thing to move; it does not need a human
one. Fable seat, asked again under this refinement, REVISED its own defect
verdict and produced the house rule:

- **THE SET SPEAKS BY DEFAULT; A FACE MUST BE OVERHEARD.** The cabinet is the
  speaker for the announcer (who IS the station), for the music bookends, and
  for characters while they are performing. A human mouth appears only when a
  character stops broadcasting and says something private -- the confession,
  the last line. **One face per episode at most**, and only for a line the
  engine can hold in a single take.
- **POINT THE ENGINE AT THE TUNING EYE.** Late-1930s console, dark walnut,
  three-quarter, room black. The dominant feature is the magic-eye tube -- the
  green phosphor iris that opens and closes with signal strength -- with the
  dial glass a warm OUT-OF-FOCUS band below and the grille cloth falling into
  shadow. The eye's fan closes on sibilants; the dial lamp breathes; the grille
  micro-trembles on plosives. **The wood never moves.** Rationale: these models
  want a bright mouth-shaped region on a dark ground, and the tuning eye is an
  iris whose entire job in 1938 was already to move with audio.
- **THE STILL MUST DENY THE ENGINE ANYTHING IT CAN VISIBLY BREAK.** No legible
  text, no sharp slat pattern, no straight edge in focus. Rigid geometry and
  typography punish drift far harder than flesh does -- "a dial that re-spells
  its own numbers every render is the most legible error this show could
  produce." Let light carry the motion. **A generated still showing legible
  dial numbers or brand type is a BAD STILL: regenerate, do not hope.**
  Watch one failure mode in test: these engines invent motion, and wood grain
  that swims continuously is worse than a face that cuts six times.
- **CONSEQUENCE FOR `humo_14B_169`:** its 49-frame ceiling STOPS being a defect
  for the set-subject. Two renders of dark wood and a glow butt together like a
  soft dissolve, and residual flicker reads as signal behaviour. It gets its
  long beats back ON THE CABINET ONLY. For human faces the original verdict
  stands: breath-length lines, cuts on breaths, one face an episode. The face
  ration and the frame ceiling are now THE SAME RULE.

Two consequences, both now load-bearing:

1. **The talking-still decision must be STABLE.** `wants_talking_prompt()`
   (`eng_ltx_av.py:390-400`) returns `_recipe_config(self._recipe())["two_stage"]`,
   and `_recipe()` re-reads `OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` / the UNET
   name from the environment on EVERY call by design. `route_freeze.routing_env_snapshot()`
   captures only `OTR_FORCE_ENGINE_MAP` and `OTR_ENABLE_HUMO_HOSTS`, so the
   recipe axis escapes the freeze. Concrete divergence: the director stamps
   `policy["talking"][role]=True` under one UNET and MetaBrief mints a
   face-forward still; the operator swaps `OTR_LTX_AV_UNET` before the render
   leg; ShotLock's guard passes because neither watched var moved; the render
   driver now answers False. Under THIS ruling that is no longer a latent
   hardening -- a flipped answer means an audio-in beat gets a still with no
   mouth, which is the exact failure the ruling forbids.
2. **The portrait index must not be empty for a talking beat.** The 2026-07-28
   campaign logged "talking-head shot b003 char_id='c03' has NO portrait-index
   entry -- HuMo will fail closed LOUD (NO FALLBACKS)". Under this ruling that
   warning is a contract violation, not noise.

**NEW OPEN ROWS FOUND BY THE 2026-07-28 RENDER/QA WINDOW** (detail in
`kibitz-runs/2026-07-28-gpu-lane-all-models/r{1,2,3,4}/final.md`):

1. **`word_razzle` IS A CLOUD ENGINE AND EVERY NAME-PREFIX FILTER IN THE TREE
   CALLS IT LOCAL.** `nodes/_otr_video_engines/eng_cloud_video.py:946`
   `CloudWordRazzleEngine`, `node_key="cloud_pixverse_i2v"`, needs
   `OTR_COMFY_API_KEY`. Any harness that classifies locality with
   `startswith("cloud_")/("google_")` puts a PAID provider in a local-only
   run -- the 2026-07-28 campaign had it as case 10/10 of its floor lane and
   logged it under "19 LOCAL". The build already owns the right answer:
   `render_driver._is_cloud_video_engine` (`:1599`) tests `provider_side` and
   `node_key`. **The true local roster is 18, not 19.** Promote that helper to
   a public registry function so nobody derives locality from spelling again.
2. **THE HEADLESS LAUNCHER SETS NO IMAGE-ENGINE FLAGS.**
   `scripts/_otr_soak_server_launch.cmd` sets `OTR_ENABLE_HUMO/LTX/WAN` per
   lane and nothing for images, so `flux2_klein` and `lumina_image` are
   UNUSABLE in any headless/soak run even though their weights are on disk
   (`flux-2-klein-4b-Q4_K_M.gguf`, 2484 MB; `lumina_2_model_bf16.safetensors`,
   4978 MB) and the operator's User-scope flags are set. Cause: a process
   inherits its parent's environment block, and the campaign's tree predates
   those variables. Live cost: two cases died at 552 s each in
   `OTR_ImageGenDispatcher` with `missing_model`, proving nothing about the
   video engine under test. Same mechanism means `OTR_COMFY_API_KEY` is also
   absent in the server tree.
3. **`mesh_stage` CAN NEVER PUBLISH A WHOLE-EPISODE CASE, BY DESIGN.**
   `otr_credits_roll.plan_backdrop` (`:1152-1161`, dated 2026-07-03) excludes
   DIRECTORY clips and raises `CreditsDataError` when nothing loopable
   remains. An episode rendered entirely by `mesh_stage` has no file clip, so
   the terminal node refuses it. NOT a regression from the encoder chunks --
   `git blame` puts it at `0606d1cd8`/`f00a8e8ef`. Score mesh coverage-only;
   do not read the red row as an engine defect.
4. **`poll_history` TRUNCATES ITS ERROR AT 500 CHARACTERS**
   (`scripts/otr_api.py:748`). Any harness matching on an exception MESSAGE
   will silently never match -- the node id and `node_type` survive, the
   message body does not. Parse structured history if the reason matters.

Landed: `27a4f97c` the second encoder + the widened gate's COLOUR half,
`afeb5b84` cheap_families' four `still_*` count proofs + the gate's COUNT half,
`6aad4fe5` the THIRD copy of the encoder deleted (not hardened a third time)
plus a gate against a fourth, `b1f2ee86` the odd-canvas stride defect closed at
the batch encoder, `1959fb49` the credits-card col1 ladder. Per-chunk detail is
in `docs/HANDOFF_LOG.md`, not here.

**A ROW FILED AS LATENT WAS LIVE ON THE DEFAULT PATH, and the lesson is the
one this file keeps writing down.** The credits-card overflow row said
"reachable only if something renders the card at 480p -- the shipped render
tests use 720p and 1080p". Derived from the PRODUCERS instead: `roll()` sizes
the card from `_probe_video(video_path)` -- the FINISHED EPISODE's own
dimensions -- the canonical workflow's VideoDirector ships **832x480**, and the
ltx_8gb tier renders **512x288**. Both overflowed; PIL clipped in silence; and
`render_static_base` captured the column's returned `y` and never used it, so
nothing logged it either. **A reachability claim in a bug row is a claim like
any other -- derive it from the producers before trusting it.**

**STANDING POLICY FOR THE END CARD (ruled by Fable 2026-07-28 under CLAUDE.md
section 9; do not re-litigate, extend).** THE CARD IS A VIEW OF THE DURABLE
LEDGER, NOT THE LEDGER. A record may never elide; a view may elide WITH NOTICE.
It may show less than it knows; it may never claim more than it shows. The
column gives up the cheapest honest thing first: the optional recipe note
(unmarked -- a gloss's absence asserts nothing), then inter-block WHITESPACE
(unmarked -- whitespace is not a claim), then ledger ROWS, fine print first and
always MARKED. **Type is never shrunk**: a receipt in unreadable type is a
receipt-shaped object claiming credit for a disclosure that never happened --
worse than a missing row, because unlike a clip it is a lie the policy tells on
purpose. **It never raises.** Missing TRUTH stays structural and still raises
`CreditsDataError`; insufficient GLASS degrades with marks. Do not soften that
guard on the strength of this one.

**HOW THAT DECISION WAS MADE, and what the panel was worth.** Fable ruled;
**agy (rung 2, $0) DISSENTED** and was overruled with the reason recorded --
it conflated a missing RECEIPT (missing truth -> raise) with insufficient GLASS
(presentational). Two of agy's MECHANICAL findings survived grounding and both
shipped: the unused `y`, and that compacting whitespace recovers enough to save
the canonical canvas without spending a single ledger row. **The dissent was
wrong and the mechanics were right, which is the usual shape -- ground every
claim separately rather than accepting or rejecting a review whole.**

**THE ARC'S ONE LESSON, said plainly: THE SAME DEFECT KEPT BEING IN A COPY.**
An encoder was duplicated three times, and each copy carried a dead frame-count
parameter, a declared size that disagreed with the bytes it piped, and no
shape/dtype check. Two of the three copies were found only because a gate was
widened; the third was found by a fan-out. Deleting the copy is what actually
closed it -- `otr_scene_aware_scopes` now calls the shared encoder, and
`_RAWVIDEO_STDIN_ENCODERS` pins the six remaining rawvideo-stdin encoders with
a reason each so a fourth fails by name.

**AND THE GEOMETRY DEFECT PASSED EVERY PROOF THIS ARC ADDED.** The batch
encoder declared `even_dim(w)` while piping the array's real odd rows;
measured, a `(5,63,47,3)` batch wrote a 46x62 clip of skewed pixels, exit 0,
and the frame-count proof AGREED -- five in, five out. A count proof
structurally cannot see a stride error. Worse, `test_ffmpeg_silent_cmd_contract`
REQUIRED the rounding ("odd width -> even"), so the suite was actively
defending the defect. **A latent row that the tests assert as the contract is
not latent, it is protected.**

**THE GATE IS STRUCTURAL NOW, AND THAT IS THE DURABLE PART.** It used to grep
two literal substrings, so a third spelling was invisible and
`encoders - provers == set()` was vacuously true over four LIVE engines. It
derives the roster instead: a command that WRITES video must name a video codec
(`-c:v` / `-vcodec` / `-codec:v`) and a command that only reads never does, so
the flag is chased from every subprocess spawn AND every argv BUILDER (
cheap_families builds its command and hands it to a generic runner). The
subprocess alias comes from the module's own imports. Where a proof legitimately
lives inside the encoder it is pinned per entry point and VERIFIED against the
credited function's source. **Known limit, recorded in the file: a codec flag
assembled at runtime (an f-string, `"-c:%s" % stream`, or the `-c:0` spelling)
is invisible to it. Nothing in the tree does that; an encoder that ever needs
to must be pinned by hand.**

**THE COLOUR AND COUNT HALVES SHIPPED AS TWO CHUNKS ON PURPOSE.** The colour
gate could go green while the count gate was still red on `cheap_families`. A
gate is allowed to arrive a chunk later; it is never allowed to be NARROWED so
it arrives clean. Both halves now bill the same roster by the same rule.

**THE FAN-OUT PAID FOR ITSELF SIX TIMES, AND FIVE OF THOSE WERE MY OWN NEW
CODE.** The first draft of the sweep tested `"sp" in name`, which is false for
`"subprocess"` -- the inventory came back EMPTY and both gates passed
vacuously, which is precisely the failure being closed. A refusal raised
mid-stream left ffmpeg ALIVE holding the output file open. `stderr` was a PIPE
read only after the whole stream was written, which deadlocks the moment ffmpeg
emits more than one OS buffer -- and that state raises nothing, so the child
would never have been reaped. `_has_proof` matched substrings, so `wan_shared`,
which DEFINES both proof helpers, was excusing itself on its own `def` lines.
And the ORDERING test one function over was still blind to these four engines:
moving the proof BEFORE the encode -- the exact defect it is named for -- stayed
green. Two independent lenses found that last one.

**A LATENT BOX-DEPENDENT FAILURE WAS MEASURED AND FIXED ON THE WAY.** The
encoder chose `h264_nvenc` whenever the box had it, and NVENC refuses a canvas
below 145x49. Measured here: 144x48 refused, 146x50 accepted, libx264 accepted
every size from 96x64 up. A small-canvas beat therefore died on a machine WITH
a GPU and succeeded on one without. It is codec SELECTION, not a fallback --
both encoders emit the same contract and the caller proves it either way.

**MUTATION: 23/24 real mutants caught, 6/6 controls survived.** TWO were
RECLASSIFIED as controls with reasons rather than chased (spelling the declared
size `(w, h)` is provably identical once the equality is proven two lines
above; dropping `Popen` from the spawn set changes nothing while every encoder
entry point is also a returner). **ONE SURVIVOR IS RECORDED, NOT HIDDEN:**
deleting the self-proving membership assertion is catchable only by a meta-test
of that assertion, which is not written. Harness: `tmp/_kbA_se_mutate.py`.

---

**Superseded header (2026-07-28, CODER window) -- ALL THREE
REMOTE-SAFE ROWS ARE DONE, AND THE FAN-OUT RAN BEFORE EVERY PUSH THIS TIME --
WHICH IS THE ONLY REASON TWO OF THE THREE SHIPPED CORRECT.** HEAD == origin
`48e3c6fb`; suite **7429 passed / 27 skipped / 1 xfailed**; Bible 17;
`build_variants --check` 11 variants / 0 failures; canonical `9872624A`
byte-identical across all three commits.

Landed: `bcaab4db` the `by_engine` roll-up, `24f4251a` the credits-card
`video_suffix` + the `_row()` clamp, `48e3c6fb` the encoder frame count.
Per-chunk detail is in `docs/HANDOFF_LOG.md`, not here.

**THE FAN-OUT EARNED ITS KEEP TWICE, ON THINGS MUTATION STRUCTURALLY CANNOT
SEE.** Row 2's fixed two-line recipe note overran the footer by 27px at
1280x720 -- the size this repo's own render tests already use -- because col1
flows downward with no backstop and PIL clips the overflow silently; no
mutation of the code reveals that the LAYOUT no longer fits. Row 3 turned a
zero-frame batch from `return (path, 0)` into a raise describing a failed
multi-segment ASSEMBLY -- true words about the wrong event. Both were found
BEFORE the push and are fixed in the same commit. Mutation was load-bearing
too and on its own ground: 38/38 real mutants caught, 13/13 controls survived
across the three chunks, and it killed three decorative assertions of mine.

**THE ENCODER FRAME-COUNT DECISION WAS A FALSE CHOICE.** The row framed it as
"pay a decode per clip, or leave the count self-declared". `nb_frames` is the
MUXER'S OWN count and rides the SAME stream read `ffprobe_clip_fields` already
performs on every emitted clip -- the identical argument that put width/height
in that query at chunk 6. A header can disagree with the picture data on a
CONCATENATED file, which is exactly why `assemble_beat_segments` decodes and
must keep decoding; a single-render clip was never concatenated. The decode is
now the FALLBACK, for a container that records no count. Measured on this box:
header 29-45ms flat from 50 to 18000 frames, decode 35-168ms and scaling,
against real beat renders of 744-842 SECONDS. Cost was never the obstacle.

**AND THE FAN-OUT FOUND A SECOND ENCODER NOBODY HAD FILED.** The four `viz_*`
engines do not write clips through `encode_frames_to_silent_mp4` at all -- they
go through `nodes/_otr_shared/scope_draw.py`, which has NO ffprobe call of any
kind, and they are structurally invisible to the M7 roster gate. That is the
`cheap_families` shape of 2026-07-27 repeating one module over. See OPEN BUGS;
it is the highest-value row this window did not take.

---

**Superseded header (2026-07-27 20:24, CODER window) -- THE RANKED
BUG QUEUE IS DONE: SIX OF SEVEN ROWS SHIPPED, A2 HELD, AND A POST-PUSH PANEL
FOUND THAT ONE OF THE SIX HAD BRICKED TWO SHIPPED WORKFLOWS.** HEAD == origin
`40780b82`; suite **7384 passed / 27 skipped / 1 xfailed**; Bible 17; canonical
`9872624A` byte-identical across all eight commits.

Landed: `ebec0f1f` A1, `ba24af29` A6, `c9b89769` A4, `57caf43d` B4,
`de50786e` A5-lite, `58e288af` the frame_count M7 sweep, `40780b82` the QA
fan-out corrections. Per-chunk detail is in `docs/HANDOFF_LOG.md`, not here.

**B5 DID NOT GATE A1 OR A6, AND THE OPERATOR HAD ALREADY RULED.** The triage
said the profile retain/retire question gated the top three. It gates only A2:
the ceiling has a live NON-profile channel (`llm_vram_ceiling_gb` is a widget in
`otr_canonical.json`, which is exactly the channel the retirement direction
KEEPS), and the GGUF artifact table belongs to the loader, not to any profile.
A2 is the one row whose whole subject is `apply_profile_to_workflow`, so it is
HELD rather than built on a channel scheduled for deletion.

**A1 NEEDED NO NEW ESTIMATOR -- IT WAS A PURE HOIST, and the "obvious" fix
would have refused the shipped default.** `check_vram_fit` already prices a
`gguf_native` row from its pinned on-disk artifact plus KV, and already answers
correctly at both ceilings (gemma GGUF estimates 14.6 GB: WARN at 14.5, FAIL at
6.8). The defect was placement only: the ceiling sat below BOTH cache-hit
returns and below the GGUF dispatch, so it could only ever gate a fresh
transformers load. A hard `estimate > ceiling` comparison, which looks like the
right fix, refuses today's canonical default. The 1.5x FAIL ratio is untouched.

**THE POST-PUSH PANEL FOUND WHAT MUTATION STRUCTURALLY COULD NOT.** Every chunk
ran a mutation round before its push (32/32 real mutants caught, 10/10 controls
survived) and they were load-bearing -- one caught an ordering test of my own
that was decorative. But no mutation of the CODE can reveal that a shipped
JSON ARTIFACT selects a quant the code just made illegal. A6 refused unpinned
GGUF artifacts; `config/profiles/otr_mac_mps.json` and `otr_nv40_12gb.json`
both select `Q6_K`, which has no pin, and their generated variant workflows
carried it in `widgets_values` with no in-workflow remedy. **Run the fan-out
BEFORE the push. This file already said so, and this window did it after.**

**Q6_K IS NOW UNUSABLE UNTIL SOMEONE PINS IT** -- it has no size, no sha and no
file on this box, so the loader refuses it by name. Both draft profiles moved to
the pinned `Q4_K_M`, which is also the only one that fits their declared 10.0 /
10.5 GB ceilings (Q6_K is ~9.1 GiB of weights plus a 2.8 GiB KV cache). The
refusal surfaced an older sizing defect rather than creating one.

---

**Superseded header (2026-07-27, CODER window) -- THE OPEN BUGS ARE
TRIAGED AND RANKED, AND THE ONE NOBODY HAD FILED SHIPPED THE SAME SESSION.**
HEAD == origin `54b3626b`; suite **7356 passed / 27 skipped / 1 xfailed**;
Bible 17; canonical `9872624A` byte-identical.

**THE PANEL DISAGREED WITH ME MORE THAN THE TWO SEATS DISAGREED WITH EACH
OTHER.** Operator-directed triage of the OPEN BUGS list: kibitz r1 with codex
`gpt-5.6-sol` high (seat verified for this run -- that pin has drifted to 5.5 on
past arcs) plus agy `Gemini 3.6 Flash (High)`, then a Fable consult under
CLAUDE.md section 9's reality exception. Of five anchor rows the panel corrected
three, cut one as already-covered, and added one absent from this document
entirely. Every panel claim was grounded against the real Windows files before
anything was acted on. Record: `docs/2026-07-27-open-bug-triage.md`.

**WHAT CHANGED IN THE LIST, not just in the code:** A1's fix shape was
INCOMPLETE (a cache hit never enters preflight, and `reuse_key()` excludes the
ceiling); A2's causal chain was WRONG (the override is at submission, and the
applier already flattens `llm` -- only the printed echo is stale); A3 is CUT
(three existing tests cover it -- it was filed after reading the code and not
the tests); A6 is NEW and is the highest-value row (`Q4_K_M` has neither an
expected size nor a SHA, so a truncated download of the quant the shipped 8 GB
profile selects passes readiness).

**FABLE RESOLVED BOTH SPLITS THE MECHANICAL SEATS LEFT OPEN, AND KILLED ONE
FINDING OUTRIGHT.** A5 is cut as a live bug but keeps codex's location at a
fraction of his scope (one `dtype == uint8` assert). B4 `ShotRow` is a CODER
fix, not an operator ruling -- ShotLock stamps six fields a model declaring
`extra="forbid"` does not have, so the "live safety net" other docs cite cannot
validate a single shipped episode. agy's heavy-import finding is NOT a violation
as this build defines the gate, and is not to be filed.

**AND FABLE FOUND TWO DEFECTS NOBODY HAD FILED, BOTH IN THE LAST NODE OF THE
GRAPH** -- `OTR_MasterAudioMux`, where everything raises AFTER the whole episode
has rendered. `OTR_MAX_CREDITS_TAIL_S` was an unguarded `float(os.environ.get(
...))`: the `PBUG-20260723-02` shape, at the opposite end of the pipeline from
where this build usually pays for it -- a malformed value killed a finished
episode over a knob that only widens a sanity ceiling. And the duration gate
failed OPEN when ffprobe was absent while the receipt still printed `OK`. Both
landed at `54b3626b`: the knob is IGNORED and NAMED, the skipped gate reports
`UNPROVEN` and says it was SKIPPED, not passed.

**A DEFECT IN THE BUG LIST ITSELF: every line cite checked had moved.** The
defects are mostly still real; their coordinates are not. Re-pin a row's cite
when you touch it.

---

**Superseded header (2026-07-27, CODER window) -- LANE 1 AND LANE 2 ARE
DONE. EVERY VIDEO RECIPE IN THE BUILD NOW BINDS FROM CODE, AND A MEASUREMENT
CLIP SAYS WHICH CELL PRODUCED IT.** HEAD == origin `8424f369`; suite **7346
passed / 27 skipped / 1 xfailed**; Bible 17; canonical `9872624A`
byte-identical.

**LANE 2 CLOSED THE RECEIPT'S OTHER HALF.** B6 made a sweep artifact
distinguishable from production; it left the four cells of the 2026-07-27 sweep
indistinguishable from EACH OTHER, all stamping one generic
`+prequalification`, so the winner was selected from a table kept outside the
ledger that outlives the run. A cell now names its departures --
`..._v2+prequalification[tiled_vae=off]` -- on all three adapters
(`71e231ec` ltx_8gb + the shared format, `8424f369` both WAN).

Rules the implementation holds, and the next window must not undo: RESOLVED
values never env presence (re-exporting a knob at its frozen value is not a
departure); only knobs that actually BOUND the render (tile geometry is
reported only when tiled decode ran); a production receipt byte-identical to
B6's, on a path that never reaches the resolver at all; per-entry bounds
(prose becomes a digest) but NEVER a truncated list, because a silent cap
destroys exactly the distinguishability the chunk exists for.

**THE FORMAT HAS ONE HOME** -- `nodes/_otr_video_engines/recipe_departures.py`,
pure, imported by both lanes. `eng_ltx_8gb` reaching into `wan_recipe` would be
the cross-lane coupling this build keeps paying for, and one consumer chain
reads both, so two implementations would grow two dialects in the ledger.

**THE FAN-OUT FOUND A LATENT LIE AND IT IS NOW STRUCTURALLY IMPOSSIBLE.**
`_build_graph` lets a per-shot `negative_prompt` win -- correct in production,
and why B6 called the negative a demotion rather than a removal. But the receipt
is SESSION-scoped (element [1] of `session_identity`, read before the weights
land and before every segment), so it can only report what the RECIPE resolved.
A sweep varying the negative would have rendered one conditioning and stamped a
receipt naming another. Under the consent act that displacement is now TERMINAL;
production is untouched. Making the receipt request-aware instead was rejected:
it would differ between the two stamp sites and refuse every multi-segment sweep
beat on identity drift.

**THE MUTATION ROUNDS BEAT THE PANELS AGAIN, AND THE PATTERN IS NOW A RULE.**
Three lenses cleared the change; mutation then found four more defects,
including a `pytest.raises(KeyError)` that passed with its named guard deleted
(the line below raises the same type incidentally) and a digest test that
`"#" + text[:8]` satisfied on every assertion. **A test that verifies a thing it
also constructs, and an exception type asserted without its message, are this
build's two most reliable blind spots.** Record:
`docs/2026-07-27-lane2-prequalification-receipt-qa-findings.md`.

---

**Superseded header (2026-07-27, CODER window) -- LANE 1 IS DONE. BOTH
WAN ADAPTERS NOW BIND THEIR RECIPE FROM CODE, AND A WAN CLIP FINALLY STAMPS A
RECEIPT.** HEAD == origin `3acc7fed`; suite **7291 passed / 27 skipped / 1
xfailed**; Bible 17; canonical `9872624A` byte-identical (no node, widget, link
or schema touched -- LANE 1 CLOSES an env channel).

`PBUG-20260723-02` is now closed on the WAN lane, one tier over from B6.
`eng_wan_ti2v` read sampler, scheduler, steps, cfg, shift, negative, the
tiled-VAE flag and four tile-geometry vars from the environment on every leg;
`eng_wan_i2v` read six of its own INLINE in `_build_graph` with bare
`int()`/`float()`, no range check and no named refusal. Both are frozen:
`WAN_TI2V_RECIPE` (`71753cb4`) and `WAN_I2V_RECIPE` (`3acc7fed`), with the
mechanism shared in `nodes/_otr_video_engines/wan_recipe.py` and the DATA per
adapter. Record: `docs/2026-07-27-lane1-wan-recipe-freeze-qa-findings.md`.

**THE RECEIPT HOLE IS CLOSED TOO.** A WAN clip stamped `recipe: None`, so there
was not even a wrong receipt to catch a drift with. `render_clip` now threads
`recipe_receipt()` through `_clip_from_raw` into
`stamp_durable(meta.render_engines)`, and a measurement run marks its own
artifacts with `+prequalification`.

**PER-ADAPTER CONSENT VARS, NOT ONE SWITCH** (`OTR_WAN_TI2V_PREQUALIFICATION` /
`OTR_WAN_I2V_PREQUALIFICATION`). One shared switch would open the other tier and
stamp `+prequalification` on a clip that had rendered with its frozen recipe --
a receipt that lies in the safer direction still lies.

**WHAT LANE 1 DELIBERATELY DID NOT FREEZE, and it is not the same list as ltx's:**
`OTR_WAN_TI2V_MAX_FRAMES` is a CEILING **and a live shipped channel** --
`otr_8gb_wan.json` sets both `launch.env` and `video.max_render_frames`, so
folding it in would have retired the 8 GB tier's launch contract. Weight names
and their loader-class selectors stay live TOGETHER (the class is inferred from
the basename; freezing one and not the other gives one fact two owners).
`wan_i2v` keeps `uni_pc`, NOT the portable floor's `euler` -- the freeze
preserves behaviour, it does not add policy.

**THE MUTATION ROUND CAUGHT WHAT THREE QA LENSES MISSED, and the lesson is
portable: 4 of 10 real mutants SURVIVED the first i2v pass.** A renamed consent
constant was undetectable because the tests set the CONSTANT rather than the
literal an operator types; `recipe` and `vram_peak_mb` could be dropped from
`render_clip` because both were only ever checked on a HAND-BUILT raw (the test
constructed the thing it was verifying); and `shift` had no production-leg test
at all, so its consent-act test AGREED with the mutant. All fixed on BOTH
adapters -- 30/30 real mutants caught across two rounds, 4 CONTROLs survived.

**ONE REAL BUG FIXED ON THE WAY (pre-push fan-out, lens C):** `eng_wan_i2v`
started an NVML probe, measured the render-window peak, logged it, and
DISCARDED it -- NEWBUG-1's 2026-07-20 fix landed for `wan_ti2v` and never
reached the sibling, so every `wan_i2v` clip reported `vram_peak_mb: None` and
`render_shot` silently fell back to an instantaneous post-render read that can
under-report, which then rolls up into the episode figure and the credits card.

---

**Superseded header (2026-07-27, RENDER window) -- PREQUALIFICATION IS
DONE. THE `ltx_8gb` RECIPE IS MEASURED AND FROZEN AS v2.** HEAD == origin
`dcdcccde`; suite **7226 passed / 27 skipped / 1 xfailed**; Bible 17; canonical
`9872624A` byte-identical (no node, widget, link or schema touched).

**THE 8 GB TIER RENDERED AT ITS OWN CANVAS FOR THE FIRST TIME.** Four full
canonical legs at 512x288, each `RESULT SUCCESS` + `obs_publish OK` with the
asset verified on disk. That also proves B5 end to end: the canonical JSON's
VideoDirector says 832x480 and the engine still rendered 512x288, because the
canvas is a static declaration. A declaration is not displaced by where it is
pointed.

| cell | t5_device | tiled_vae | shots | min MB | max MB | SPREAD | wall |
|---|---|---|---:|---:|---:|---:|---:|
| A | cpu | off | 11 | 8662 | 10859 | 2197 | 842s |
| B | default | off | 12 | 11163 | 16127 | 4964 | 744s |
| **C** | **cpu** | **ON** | 10 | **8241** | **8278** | **37** | 824s |
| D | default | ON | 11 | 11062 | 16086 | 5024 | 765s |

**THE DECISIVE COLUMN IS THE SPREAD, NOT THE MINIMUM.** With tiled decode ON
the peak is flat across every clip length the sweep produced (17 to 161
frames); OFF it climbs with length. An 8 GB tier needs a ceiling a long beat
cannot grow through. It costs no wall clock. `t5_device` stays `cpu` and now
has a number behind it: on GPU the peak sits at 16.0-16.1 GB of a 16.3 GB card.
Landed `1fe7dc8c` (v2 + `LTX8_RECIPE`, v1 kept unmutated so `_v1` receipts stay
interpretable) and `dcdcccde` (consent-act knobs fail CLOSED).

**HONEST LIMIT, DO NOT LET A LATER WINDOW FORGET IT:** `VramPeakProbe` samples
MACHINE-WIDE NVML and the sweep ran UNCLAMPED, so those absolutes are not a
proof of 8 GB fit -- they support the RANKING, which is what selects a recipe.
**A clamped confirmation of the winner is still owed** (see OPEN BUGS).

**OPERATOR DIRECTION 2026-07-27, TWO CHANGES:**
1. **Order is now WAN 8-GB -> Randomizer A -> `dynamic_story` -> SFX ->
   LEAN-MEAN FRONT -> TAIL.** Lean-mean moves LAST on its own logic: it is a
   deletion campaign whose value IS its file-and-line kill inventory, so it
   should re-ground once against the final tree rather than rip first and
   re-ground after two blocks land on freshly-rewired code. SFX carries its
   Timeline Cue Ledger C0/C1 gate + the R4.1 refit as a precondition.
   **AMENDED 2026-07-29 (operator): lean-mean is not merely last, it is OFF
   THIS PLAN.** Both blocks moved to the Lean-mean campaign section of `ROADMAP.md`. This plan's order
   is now: LOCAL-ENGINE OBS WIRING -> WAN 8-GB -> Randomizer A ->
   `dynamic_story` -> SFX -> re-observe the parked story bugs. Anything after
   that is ROADMAP work, and lean-mean is the first item there.
2. **PROFILES ARE BEING RETIRED.** Operator: there is no tier -- whoever runs
   the workflow picks the creative/technical LLM, and the 8gb/16gb variants
   will be the SAME canonical JSON saved with different dropdowns, with no
   auto profile selection. Treat `config/profiles/*.json` as legacy; do not
   build new behaviour on the profile channel.

---

**Superseded header (2026-07-27, CODER A session 5b) -- B3, B4, B5 AND
B6 ARE IN. THE TIER CEILING PLANS, THE PING-PONG IS GONE, THE 8 GB TIER RENDERS
AT THE CANVAS IT DECLARES, AND ITS RECIPE IS FROZEN IN CODE. Suite 7213 passed
/ 27 skipped / 1 xfailed; Bible 17; canonical `9872624A` byte-identical.**

**B6 FROZE THE RECIPE AND THE PANEL FOUND THE HOLE IN THE FREEZE ITSELF.** The
first draft demoted the sampling knobs and left `OTR_LTX_8GB_NEGATIVE` -- a
render input, read straight from `os.environ` on every leg -- plus four tiled
decode-geometry vars, still binding from the server's boot. Two independent
lenses found it separately. Worse, the draft stamped the SAME recipe receipt on
a prequalification sweep as on production, so a measurement artifact and a
published one were indistinguishable in `meta.render_engines`. All closed, and
the fix for the geometry then created its own defect -- a second range-check
implementation that failed OPEN where every sibling knob fails closed -- caught
by the post-fix lens and collapsed into one `_config_number`. Record:
`docs/2026-07-27-b6-qa-findings.md`.

**B5 CLOSED O1, AND THE PANEL SENT THE FIRST DESIGN BACK.** The draft read the
profile's 512x288 off `video.canonical_canvas`; it was green and
mutation-proven and it was WRONG. `docs/2026-07-26-o1-canvas-arc-judgment.md` --
one of the three authorities this step names -- lists that channel as the one
DEAD channel of five and requires the engine to declare its canvas STATICALLY,
"not an env var, not a ledger read". The later 8gb judgment's B5 paragraph says
the opposite and never reconciles the two. The deciding evidence was concrete:
`tmp/_run_canonical_engine_matrix_20260723.py` routes `ltx_8gb` onto the
CANONICAL 832x480 workflow through profile `role_overrides` and copies no
canvas, so a ledger-reading design would pillarbox or REFUSE a live QA campaign
that still owes a requalification leg. **A declaration cannot be displaced by
where it is pointed.** Record: `docs/2026-07-27-b5-qa-findings.md`.

**B3** made `max_render_frames` narrow the contract the coverage planner
partitions against -- for `ltx_8gb` and nothing else. It is NOT a general
planning cap: WAN reads 17, renders short, then ping-pongs to the beat length,
so narrowing WAN before `partition_beat()` would turn every WAN beat into a pile
of 17-frame renders and undo `PBUG-20260723-02`. The derivation is an allowlist
of one, a `coverage_contract` receipt rides the shot beside the plan, and the
render boundary re-derives it and refuses on any difference. **B4** deleted the
`ltx_8gb` ping-pong, deleted `_ltx8_frame_length` with it, moved the ladder onto
the engine's own `frame_contract`, and replaced the pad with two invariants.

**THE PRE-CODE PANEL REFUSED THE JUDGMENT'S OWN B4 RECIPE, AND IT WAS RIGHT.**
Deleting the pad with only a cap refusal in its place would have sent short
clips to `otr_silent_composite`, which hard-loops them AND suppresses its own
underrun warning once loop-fill activates -- a logged mirror traded for a
SILENT jump-cut repeat, on the majority path. What shipped instead: an off-grid
ask renders the next legal rung UP and trims the surplus in REAL frames (100
renders 105, keeps 100), plus a pre-render refusal when no legal rung reaches
the ask and a post-render `len(frames) != length` refusal. Records:
`docs/2026-07-27-b3-qa-findings.md`, `docs/2026-07-27-b4-qa-findings.md`.

**THE CONSTRAINT FROM B3 IS NOW LIFTED.** B3 shipped with "do not pin an LTX
ceiling before B4 lands" because the ping-pong laundered the plan-vs-adapter
frame-count disagreement. B4 landed. A profile may now pin
`video.max_render_frames` on an `ltx_8gb` tier and the disagreement is terminal
at both ends.

**NEXT = B6, AND IT HAS AN OPEN QUESTION -- see CURRENT STEP.**

---

**Superseded header (2026-07-26, session 4) -- B1b-0 AND B1b; THE LOADERS ARE
HOISTED.** HEAD `d708408d`; suite 7097; canonical `9872624A`.

**THE HOIST IS DONE AND THE 4 GiB FLOOR MOVED WITH IT.** `Ltx8gbEngine.prepare()`
now runs a loader-only mini-graph and hands the checkpoint to every segment
through `external_results`; `_build_graph` omits the definitions of ids the
caller supplied and keeps every wire. CHECKPOINT ONLY -- the T5 stays
per-segment by decision. The floor is a shared helper called BEFORE
`super().prepare()` takes the lease.

**THE QA PANEL KILLED THE PREVIOUS SESSION'S OWN ACCEPTANCE CRITERION.** The
B1b-0 net declared two assertions would FLIP under the hoist. Two independent
seats proved neither could: `_build_graph` stays conditional, and every test in
that file hands `render_clip` a hand-built prepared dict with no
`external_results`, so they all stay on the unsupplied branch. **Nothing in the
net would have gone red against a hoist that silently did nothing.** Corrected
before the hoist: every one of those tests is a CONTROL, exactly ONE assertion
flipped, and the 1-load proof was written WITH the hoist. Records:
`docs/2026-07-26-b1b0-qa-findings.md`, `docs/2026-07-26-b1b-hoist-qa-findings.md`.

**NEXT = B3 + B4.** See CURRENT STEP.

---

**Superseded header (2026-07-26, session 3) -- THREE KIBITZ ARCS JUDGED AND
EIGHT GREEN CHUNKS PUSHED at `095be05b`; suite 7071.**

**THE DISCIPLINE THAT PAID:** the operator required a fan-out BEFORE each fix as
well as after. Before the single-clip fix it killed MY OWN proposal (a wholesale
delegation would have dropped the 4 GiB checkpoint floor); before the `*_DIR`
fix it produced the evidence that scoped the change (the Wan suites use `*_DIR`
as their mock seam) and refuted the register-the-folder design (ComfyUI ships no
unregister). After each fix it found real defects in already-green code --
including one of my own tests that was DECORATIVE: it claimed to detect a branch
swap and would have passed under one.

**THE HEADLINE: O1 (the canvas) WAS NEVER THE ONLY 7d BLOCKER.**
`session_identity` appeared in exactly ONE file (`beat_session.py`) and NO
adapter declared it, so `BeatSession.open()` refused EVERY multi-segment beat
for all 31 engines -- before the weights land, no fallback. A 169- or 237-frame
beat was rejected before the render canvas was ever consulted. That refusal is
now lifted for `ltx_8gb` (`582dfbd8`).

**Landed this session:** `8caf3516` B1a (`run_graph` learns `external_results`
+ `on_result`), `55c8a811` B2a (`resolve_session_config` -- one frozen
resolution, and the receipt stops describing a weight the loader never loads),
`582dfbd8` B2b (`ltx_8gb` declares `session_identity`). All three
mutation-proven WITH controls. Judgments: `docs/2026-07-26-o1-canvas-arc-judgment.md`
and `docs/2026-07-26-8gb-1080p-arc-judgment.md` -- **read both before coding.**

**CANVAS DECIDED: 512x288.** Four independent sources. It is the only exact-16:9
/32-clean rung besides 1024x576, and Fable's viewer call settles the choice
between them: softness is a STATE viewers habituate to, a motion reset is an
EVENT they never do. 832x480 is 26:15 and would pillarbox to 1872x1080. The
tier file was right all along.

**ACCEPTANCE IS 237, NOT 169 -- and that CUTS O4 entirely.** The canonical
`OTR_EpisodeAssembler` already ships `opening_duration_sec=10.0` /
`crossfade_ms=500`, so it produces `round((10-0.5)*25) = 237`. At a 65 cap:
`[65,65,65,49]` -> 241 chained -> trim 4 -> 237, every segment `8n+1` (verified).
No profile-schema or widget-mapping work is needed for the opening beat.

**NEXT = B1b**, then B3+B4, B5+B6, prequalification, 7d. See CURRENT STEP.

---

**Superseded header (2026-07-26 earlier) -- CHUNKS 5, 6 AND 7a ARE COMPLETE.**
Eleven green chunks pushed that session: QA4 (`b0e383f5`), the beat session
(`4fa992e6`) + its QA (`451309de`), the ffprobe helpers (`3a76c47a`), the
per-segment init image by object id (`a888c423`), QA6 over those three
(`a818b5d1`), the terminal-frame extractor (`4d5795b1`), the per-segment render
loop + transactional assembly (`5845e635`), QA7 over the loop (`a05b5ac6`),
**the 31-engine frame-contract sweep with the opt-in deleted (`e90dedf1`), and
the two live-path defects its QA panel found (`42db9af9`)**. HEAD == origin
`42db9af9`; suite **6891 passed / 27 skipped / 1 xfailed**; canonical
`5377914B` (byte-identical -- no chunk in 1-7 touches it).

**NEXT = chunk 7b.** Chunk 7a landed the per-engine frame contracts and DELETED
the opt-in entirely (operator ruling: everything gets an equal term). See
CURRENT STEP -- the "first adapter opt-in" step no longer exists, and
`docs/ENGINE_MATRIX.md` is now the generated per-model requirements record.

**OPERATOR RESCOPE 2026-07-24 -- ITS CUTS STAND, ITS ORDER DOES NOT.** The
45-word scene matrix, the 54-case visual-style sweep and the WHOLE quick-wins
block are CUT -- the operator wants coding, not matrices, and will triage bugs
as a batch later. ENGINE_MATRIX survives the cut as a W6 sub-step, not a
standalone chunk. ~~The order is now WAN 8-GB contract -> LEAN-MEAN FRONT ->
Randomizer A -> `dynamic_story` -> LEAN-MEAN TAIL -> SFX -> re-observe the
parked story bugs.~~ **THAT ORDER IS SUPERSEDED TWICE OVER. The live order is:
LOCAL-ENGINE OBS WIRING -> WAN 8-GB -> Randomizer A -> `dynamic_story` -> SFX
-> re-observe the parked story bugs. LEAN-MEAN IS NOT SECOND AND IS NOT ON THIS
PLAN AT ALL** -- operator direction 2026-07-29 moved FRONT and TAIL to
the Lean-mean campaign section of `ROADMAP.md`. This paragraph's "supersedes the older queue everywhere in this file"
clause was true on 07-24 and is the single most misread line in the document;
it does not outrank a later operator direction.

Two story-shaped defects are PARKED, not closed (see OPEN BUGS).

**And every remaining big block must be RE-GROUNDED by a kibitz arc before it
executes -- r3+r4 by default, dropping to r2 anywhere the coding plan itself
proves stale. See STANDING RE-GROUND GATE. These plans are two weeks old and
the tree moved under all of them.** (The full `r2 -> r3 -> r4` pin on both
lean-mean blocks travelled with them to `ROADMAP.md`.)

This file contains only go-forward work, open bugs, and standing operator
contracts. Completed work is NEVER re-described here -- it moves to
`docs/HANDOFF_LOG.md` (history) and `docs/PROD_BUG_LOG.md` (bugs) the session
it ships. Doctrine lives in `docs/PRODUCTION_SPRINT_LESSONS.md`.

## CURRENT VERIFIED HANDOFF -- 2026-07-26

Nothing in this file is an instruction to reset, stash, delete, or overwrite
user changes.

- Branch: `v2.0-alpha`; HEAD and origin are `a05b5ac6`. Multi-clip coverage
  chunks 1a/1b/1c/2/3/3b/4/**5/6a/6b/6c/6d** are all landed and pushed;
  per-chunk detail lives in `docs/HANDOFF_LOG.md`, not here. The worktree is CLEAN of task-owned changes
  -- what remains is `tmp/` scratch (including another window's modified
  `tmp/_chain_720.ps1`, `tmp/_rearm_gate.ps1`, `tmp/_status_bake.ps1` --
  PRESERVE), untracked campaign receipts, `config/profiles/otr_sbcov_1..6.json`
  (intentionally untracked coverage-campaign scratch) and untracked
  `docs/_bakeoff_*.log.err` + `docs/otr-*.pdf` from an earlier window.
- LANDED overnight 2026-07-26 (suite 6634 -> ... -> **6723 passed / 27 skipped
  / 1 xfailed**; Bible 17; AST/BOM/zero-byte/UTF-8/ASCII clean; canonical
  byte-identical; each pushed, HEAD == origin): `b0e383f5` QA4, `4fa992e6`
  **chunk 5**, `451309de` its QA, `3a76c47a` **6a**, `a888c423` **6b**,
  `a818b5d1` **QA6**, `4d5795b1` **6c-1** (terminal frame), `5845e635`
  **6c/6d** (the loop + transactional assembly), `a05b5ac6` **QA7**.
- Verification: full Windows OTR suite `6723 passed, 27 skipped, 1 xfailed`;
  Bug Bible `17 passed, 24 skipped, 3 xfailed`.
- Canonical workflow byte-identical at SHA-256
  `5377914B14911B7362D2516BAD3008BB6EF6ACB87C6E13C77C3D4C0D9D8A8C39`.
- Prior root fix at `f150213f`: `nodes/_otr_video_engines/render_driver.py` requires
  an authoritative scene-target manifest only for scene/mesh-consuming shots;
  visualizer-only `viz_mxc_cpu`, `viz_mxc_mandala`, and `viz_camera` lanes may
  execute without one. Regression coverage:
  `tests/test_ledger_cleanup_contracts.py`.
- Live media proof: isolated `media_archive@120w` passed with `RESULT SUCCESS`,
  `obs_publish OK`, and non-zero episode/OBS assets. In the monitored run
  `tmp/six_bank_sweep_20260723_205002_331`, `original`, `public_domain`,
  `shakespeare`, and `scifi_news_pro` passed at 120 words. `scifi_news` failed
  closed on provider/context capacity and produced no publish artifact. The
  `scifi_news_pro@120w` pass does not clear its known `requested_output=2800`
  versus provider cap `512` blocker.
- WAN is already canonically qualified and remains closed. LTX remains
  untouched/unqualified until its explicit cases run.
- Overnight monitoring automation is active in the Codex app as
  `otr-overnight-qualification-monitor`. It must continue from the live logs,
  preserve canonical assets, and report terminal receipts or reproduced bugs.
- LANDED @ `314dd481` (2026-07-24; suite 6182 passed / 27 skipped / 1
  xfailed; Bible 17; AST/BOM/zero-byte/canonical-hash gates passed; pushed,
  HEAD == origin): word-fit ceilings /
  candidate ownership retired (length = non-gating telemetry on all six
  routes); provider-capacity whole-artifact output contracts with preserved
  list-subclass markers; `scifi_news` P1/P2/P3/P5 + `scifi_news_pro`
  pitch/treatment/news/script/casting migrated to provider-capacity output (no
  target-derived cap, no +25% missing-END branch); `scifi_news_pro` markup
  acceptance now structural delimiter/order/roster only; placeholder G13 fully
  retired; campaign receipt truth hardened (no PASS without canonical
  `RESULT SUCCESS`); the repair-first plan (explicit P0 slice identity, bounded
  tagged repair context, one direct alternate owner, original post-validator
  reuse, journaled owner/backend/rung/nonce/disposition).

### Immediate next actions

1. Preserve the completed run artifacts and record its 4/5 120-word receipt
   result; do not rerun the known provider-capacity failure as a workaround.
2. Open a coder window on the WAN 8-GB low-VRAM launch contract. It is the
   first item of the rescoped order and needs no GPU to write.
3. For any reproduced failure, fix the owning producer/receipt boundary,
   re-run focused tests, the full Windows suite, and Bug Bible, then commit and
   push the green code chunk to `v2.0-alpha` and verify `HEAD == origin`.
4. Never add fallback assets, truncation, silent resizing, arbitrary provider
   caps, or prose-quality rejection.

## MODEL & CREDIT BUDGET (operator, 2026-07-24 -- read this EVERY window)

Every window states, in its first reply, which rung of this ladder it is on
and why. Pick the cheapest tool that can win; escalate only when the cheaper
rung cannot decide.

**Reset state 2026-07-24: Claude weekly credits FRESH; Codex credits FRESH
(reset taken today). Both pools reset weekly -- front-load heavy coder windows
and the big Codex spends early in the credit week; late-week, drop to the $0
rungs instead of grinding a paid pool dry.**

| Rung | Model / tool | Cost | Use for | Never for |
|---:|---|---|---|---|
| 1 | Local Qwen on the 4060 (`10.55.0.2:1234`, LM Studio/ACPX): `qwen3-coder-30b-a3b-instruct` now; `Qwen2.5-Coder-14B Q4_K_M` as the fast tier once installed | $0 | Read-only FIRST-PASS triage of failures, logs, diffs before any credit spend | Final diagnosis, patches, tests, live qualification (Codex/Claude own those); NEVER loaded on the 5080 (ComfyUI renders only) |
| 2 | agy / Antigravity, `KIBITZ_AGY_MODEL="Gemini 3.6 Flash (High)"` (operator 2026-07-24: 3.6 > 3.5; DISPLAY name exactly -- a wrong id silently kills agy and the arc runs codex-only; check antigravity.log per round) | $0 | Default grounded reviewer for ALL mechanical review; second panelist on every kibitz | -- |
| 3 | Codex CLI `gpt-5.6-sol` (high) | weekly credits | The second opinion of record: two-strikes law (mandatory 3rd-attempt panel), sec-16 + r5 extensibility confirm, pre-execution grounding of big blocks, live-failure kibitz, HANDOFF_CODEX grind delegation | Mechanical review agy can do alone. Verify `codex_model_selected.txt` every arc (stale skill cache once drifted to gpt-5.5 mid-arc unnoticed) |
| 4 | Claude (Cowork, this) | weekly credits | The actual work: planner + coder windows, anchor/judge on every panel, live-run drive | Babysitting renders (the Codex-app overnight monitor owns that); single-small-item windows (batch per the Window packing rules) |
| 5 | Cloud roundtable (OpenRouter) | real $ | Genuine R1 ideas passes only; <$20 autonomy rule applies | Mechanical/grounding review (that is rungs 2-3) |
| 6 | Fable | scarce | Single final gate on a lean-mean epoch commit only (section-9 reality exception) -- and lean-mean is now ROADMAP work, so this rung has NO GO_FORWARD claimant | Anything else |

Production (in-pipeline, all $0/local, offline-first): writers = Mistral-Nemo
(ctx cap 16384) + `gemma-4-12b` (saved runtime-qualified local default);
stills/video-init = `z_image_turbo` (Qwen-Image engine is REMOVED -- keep
Qwen3/Qwen2.5 LLM support and Z-Image's `CLIPLoader(type="qwen_image")`
encoder, unrelated). Cloud writers (Sonnet-4.5 etc.) stay opt-in bake-off
arms, never the default.

Per-window model mapping:

- RENDER / qualification windows: local production models + the Codex-app
  monitor; Claude only to launch and wrap.
- CODER windows: Claude codes; rung-1 Qwen triages every failure first; Codex
  only via the two-strikes law.
- PLANNER window: Claude; the sec-16 + r5 kibitz (codex + agy) is THIS WEEK's
  scheduled Codex spend while both pools are fresh -- it is the operator
  bottleneck on the critical path.
- CODER E extensibility (21-31 d): spans multiple credit weeks -- plan wave
  boundaries at the weekly resets; mid-build Codex only via two-strikes.

## THE LAW (operator, 2026-07-22 -- supersedes anything that disagrees)

> **AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE FOR LENGTH, LANGUAGE,
> STYLE, VISUAL VOCABULARY, OR QUALITY.**

The sole terminal spoken-prose policy is the shared whole-word safety
authority: profanity, explicit guns/knives/weapons, and explicit
sexual/nudity content. Smoking and benign substrings such as `begun` pass.
Structural JSON/schema/IDs/roster/source-proof/rights/graph/markup/nonempty/
provider-integrity failures remain fail-closed because they protect a usable
ledger rather than judge prose. Across all six banks, requested word length,
actual word count, drift, one-breath estimates, visual/world vocabulary,
noun/POS heuristics, casing/title/honorific style, craft, and quality are
guidance or telemetry only -- they may never reject, reroll, retire, replace,
or block an episode. Same-story LLM cleanup is allowed.

## HISTORICAL WRITER PLAN -- SUPERSEDED 2026-07-30

The former Section 0 operator gate, Window A ordering, and Window B/degrade
proposal below are retained as history. They are no longer current instructions.
The operator resolved the gate in favor of fresh complete model-authored
candidates before ledger assembly, without a canned or degraded ledger floor.
The current plan of record is
`docs/2026-07-30-story-never-fails/FINAL_PLAN.md`; code landed in `3bc3d8a0`.

### A-7 IS HALF LANDED, AND THE OTHER HALF IS BLOCKED BY A DIRTY REPO (2026-07-30)

`3bbf9757` landed the OTR half: `docs/PRODUCTION_SPRINT_LESSONS.md` lessons 34
and 35 carry dated A-4 amendments. **Correction to the plan's premise:** lesson
35 already argued the right doctrine ("only deterministic impossibility may fail
loud") -- the code simply did not implement it until A-4, so the amendment names
the phase vocabulary and points at it rather than rewriting the lesson. Lesson
34's "measurable delivery exhaustion raises a typed error" needed the qualifier
that exhaustion is the END of the ladder, not the first thing that goes wrong
on it.

**The Bible half is APPLIED BUT NOT COMMITTED, on purpose.** Entries 11.50 and
12.68 in `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\BUG_BIBLE.yaml`
now carry the capacity-phase rule (12.68 gets the rule plus a `verify` clause
including the fixture-collision trap; 11.50 gets one clause saying its pre-call
refusal is correctly terminal and must not be generalized). `tests\bug_bible_regression.py`
is green at 17 passed / 24 skipped / 3 xfailed.

It is not committed because **that repo has 1,226 uncommitted lines on `main`
that are not mine**: `BUG_BIBLE.yaml` +679, `tests/bug_bible_regression.py`
+546, `README.md` +8. My ~40 lines sit inside the same file, so a pathspec
commit cannot separate them and committing would sweep in someone else's
in-flight work. Whoever owns that work should commit it; my lines ride along and
are described here so they are not a surprise in the diff. **Do not `git
checkout` or reset that file to "clean up" -- that destroys both.**

Also recorded so nobody re-derives it: `yaml.safe_load` on `BUG_BIBLE.yaml`
fails at line 834 (an unquoted JSON snippet in an entry) and fails IDENTICALLY
at HEAD, so it is pre-existing and not a parse regression. The authoritative
gate is `tests\bug_bible_regression.py`, not a naive safe_load.

### A-5 AND A-6, GROUNDED AT HEAD 2026-07-30 (read this before coding either)

Every anchor below was re-grepped at `41683fc9`; the plan's line numbers had
all drifted, as its own Section 8 warns.

**A-5 -- canonicalise spoken text at acceptance. RECOMMENDED ORDER: AFTER A-6.**
The four identity consumers today all read the SAME `script` object, so the
plan's "make them read one object" is already true in the narrow sense; the
real change is WHICH TEXT the identity is taken over. Verified sites, all in
`nodes/_otr_scifi_codex.py`:

- `_assemble_ledger` (`def` at `:2187`) writes the ledger row's
  `"text": src.text` AND `expected[lid] = src.text` from one loop (`:2207-2213`)
  -- so today they cannot disagree.
- `line_text_sha256` + `accepted_lines` are stamped from `expected` at
  `:2257-2258`.
- `_CodexTailFinalizer._proof` (`:1932-1939`) re-derives those hashes and
  raises `CodexPreTailAuditError` if the in-memory ledger has moved.
- `_script_digest` (`def :1924`) hashes `script.model_dump()`; the call is at
  `:2521`, beside `stamp_receipt(..., accepted_artifacts={"final_script":
  script})` at `:2513-2517`.

So A-5's work is: build a COPIED artifact whose spoken rows carry
`clean_spoken_text(...)` output, hand THAT to `_assemble_ledger` and to both
receipt sites, and record the generation. **The grandfather rule is not
optional** -- frozen ledgers keep their raw-text hash and are never re-pinned;
only ledgers produced after that commit use the cleaned-text hash, and the
receipt says which. `clean_spoken_text` (`nodes/_otr_script_prep.py:21`) also
strips a leading speaker label, which is a SEPARATE P5 rejection: do not
conflate them.

**A-6 FIRST, and the plan's premise for it is WRONG in one word.** The plan
says `_otr_ledger_cleanup.py` "SILENTLY sets `skip=True` on a voiced row that
cleans to empty". It is not silent: the real site (`:254-269`) sets `skip`,
zeroes the text through `set_line_text_metrics` (the atomic owner -- a direct
`row["text"]` assignment is forbidden by
`tests/test_text_metric_ownership.py`), sets `tts_skip_reason`, and APPENDS
`{"action": "marked_explicit_skip_no_spoken_surface"}` to the cleanup receipt.
Its own comment defends the choice: an explicit skip is completion, and a
silent hole is what broke slicing and captions downstream.

What is still wrong is the OUTCOME, not the silence: a voiced line disappears
from the episode instead of being re-authored, which is Invariant 7. So A-6
must turn it into a P5 **finding** at `_spoken_text_finding`
(`nodes/_otr_scifi_codex.py:2046`) that sends the row back to be re-authored --
never an assert, which would itself be a veto -- and it must not simply delete
the cleanup branch, because that branch is holding a real downstream contract
until the re-author path exists.

**Why A-6 before A-5:** A-5 canonicalises with the same cleaner, so a row that
cleans to empty becomes A-5's problem the moment it lands. With A-6 in first,
that row is already a finding that goes back for re-authoring, and A-5 cannot
create a silent empty by construction.

### SECTION 2 IS MEASURED (2026-07-30) -- THE ORDER HOLDS, THREE PREMISES DO NOT

Method: the 50 campaign-day `tmp\otr_headless_*.log` files, every P0 rejection
read verbatim and classified, then the code paths re-read. The census
reproduces exactly: **15 P0, 9 P5, 1 P3** deaths (the plan's 8 P5 excludes one
of the three `PromptContextOverflowError` runaways).

**The deterministic rung would NOT have saved those 15, and the reason is
structural: `_span_ok` (`nodes/_otr_scifi_codex.py`, directly above
`_span_mismatch`) ALREADY does the coordinate repair** -- when a quote does not
match its declared slice it runs `source.find(quote)` and snaps `start`/`end`
inside the validator. So "a literal quote with wrong coordinates" can never
produce this error, and **0 of 15 deaths are that case.** Every one is a quote
that is not a byte-exact substring of its cited field. `47c554fa`'s own comment
("arithmetic, not authorship") describes a case that was already handled
upstream; the rung's only live mechanism on these legs is PRUNING the
unsupported rows.

Classification of the 15 (verbatim evidence in HANDOFF_LOG):

- **12** -- the quote is real article prose but paraphrased, from the wrong
  region, or is the model's own claim text. Only pruning can help.
- **2** -- the quote is character-identical to its source EXCEPT the feed
  carried `&nbsp;`. **A-3 fixes exactly these** (`otr_headless_65212`,
  `otr_headless_65452`), and both are now regression fixtures.
- **1** -- `&nbsp;` in the failing window plus a diverging quote.

**Prune-survivability is PLAUSIBLE, NOT PROVEN.** In 10 of the 15 the first
failing row is F02 or later and `_validate_fact_index` returns on the FIRST
error, so at least one earlier fact validated and a pruned index would satisfy
`facts` `min_length=1`. It cannot be replayed offline: the logs carry only a
truncated `raw head` of the artifact and no source payload.

**The plan's "28 decode-message legs" is a miscount.** 28 is the count across
the whole 17-day `tmp/` history (222 logs). On the campaign it is **8 legs** --
4 at P0, 4 at P3 -- and the conclusion still holds (`otr_headless_49672` and
`otr_headless_65401` decode-failed and did not die of it), so A-2 has almost
nothing left to classify and folds into A-1's receipt rather than its own pass.

**THE RUN IS ALREADY EARNING ITS KEEP. Three bugs in the first five legs, none
of which the 7,700-test suite could see:**

- `beat l001 needs 2 clips on humo (185 frames, cap 177)` -- killed all four
  HuMo legs. Fixed by **WIRE-W4e** (a per-line voice WAV is sliced per segment
  too, not just the master), which let ShotLock's audio-driven multi-clip
  refusal be lifted.
- `P5 failed: ... primary_ladder_exhausted; last error -> l001: spoken text is
  production markup` -- killed `ltx_8gb` before any video engine ran. The
  writer's ONE typed-repair shot was being spent one defect at a time. Fixed:
  the P5 validator now reports every offending line, and a compile refusal
  carries the raw markup findings with it.
- a UTF-8 BOM in `tests/test_wire_w7_mouth_ownership.py` -- invisible to the
  interpreter, caught only by the AST-scan guard.

  See HANDOFF_LOG 2026-07-29 (WIRE-W4e + P5 REPAIR COMPLETENESS) for all three.

**RE-RUN OWED** once `tmp\_w45_run\DONE.txt` exists (two campaigns cannot share
the box -- each leg kills other OTR ComfyUI servers at start):

    tmp\_w45_campaign.ps1 -Words 45 -Only humo,humo_1.7B,humo_1.7B_169,humo_14B_169,ltx_8gb

**CLEANUP OWED at the end:** `Remove-Item config\profiles\otr_w45_*.json` --
nineteen scratch profiles in a tracked set; they must never be committed.

## HISTORICAL DESIGN INPUT -- WRITER NEVER VETOES (LANDED/SUPERSEDED 2026-07-30)

**The ruling, in the operator's words:** "the writer should not be allowed to
kill the run, it just needs to fix the ledger" -- and, restated: "the writer
should never veto, the writers should keep on passing in a loop to agents to
clean up the ledger." This historical design block led to the later ruling and
implementation in `docs/2026-07-30-story-never-fails/FINAL_PLAN.md`: invalid
pre-ledger candidates are abandoned and freshly authored, rather than degrading
or patching an assembled ledger.

**Why it is now the top of the forward queue.** The live 45-word campaign spent
its first ten legs proving the writer is the run's dominant failure mode, not
the video engines -- roughly four legs in ten died before a renderer was ever
reached, each one after minutes of GPU time. Three different mechanisms, ONE
shape:

- `ltx_8gb` -- P5 markup: the retry ladder genuinely exhausted, because the
  validator surfaced one defect at a time (ROOT-FIXED `3b49d3f8`).
- `ltx_audio_in` -- P5 runaway: `PromptContextOverflowError` is a plain
  `RuntimeError`, which `structured_call` does not catch, so it fired on
  attempt 1 of 3 and SKIPPED the remaining two rungs. The ladder advertised a
  budget it never spent. 24 minutes, one leg (PBUG-20260729-02, OPEN).
- `mesh_stage`, `still_flat` -- P0: `repair context is 16796 bytes, over the
  hard limit 14336`, so the repair could not even be ATTEMPTED. Terminal before
  it started.

In every one, the writer treated "I could not produce a clean artifact on this
attempt" as "this episode cannot exist." There is no degrade path anywhere in
the pass topology; every pass is a veto.

**What the ruling asks for, as a contract:**

1. A writer pass failure DEGRADES to a workable ledger. It never terminates the
   episode. Fail-loud stays -- the receipt must say exactly what was wrong and
   what was done about it -- but loud is not the same as fatal.
2. Passes keep going in a LOOP, handing the artifact to cleanup agents, until
   the ledger is workable. A bounded ladder that gives up is the shape being
   replaced; the loop is the shape being asked for.
3. PBUG-20260729-02's candidate 2 -- let a runaway advance the ladder instead
   of being terminal -- is now the RULED DIRECTION, not an open design
   question. Its recorded trap still applies: the typed-repair factory would be
   handed the ~14,700-token truncated output as `failed_output`, so the repair
   prompt must be bounded before that path opens. The P0 hard-limit failures
   above are the same trap from the other side and belong in the same design.
4. **THE LAW IS UNTOUCHED.** Requested word length, actual word count and drift
   remain telemetry that may never reject, reroll, retire, replace or block an
   episode. "Never veto" widens what the writer must survive; it does not
   license a word-count gate, and `output_budget_mode: "provider_capacity"`
   stays the decision it already is.

**Do NOT start this while the campaign is live.** It is writer-topology surgery
across `_otr_structured_call.py`, `_otr_scifi_codex.py` and
`OTR_LedgerScriptWriter.py`, and the engines are the thing being proven right
now.

### COVERAGE IS THE ANSWER, NOT A LENGTH CAP (operator ruling, 2026-07-29)

**In the operator's words:** *"there's no max length. We just... I'm not putting
length requirements in. Whatever words that we get, we just need to make sure
there's enough video and enough stills to cover every beat."*

This is THE LAW stated from the render side, and it settles a question the
"writer never vetoes" plan had open.

**WHAT IT CUTS.** The plan's C2 -- a `max_length` on
`ScriptTextDraftLineV4.text` -- is **CUT, not deferred.** It was the plan's own
declared weakest seam and both panel agents flagged it: Codex said the corpus
criterion did not prove the distinction from a word gate, Antigravity asked for
a docstring insisting it was only an infrastructure ceiling. A cap that needs
a paragraph explaining why it is not a length requirement is a length
requirement. The P5 runaway must be answered by the retry/degrade topology
(C3/C4), never by refusing to let the writer write.

**WHAT IT REQUIRES INSTEAD.** However many words arrive, the render side owes
enough clips and enough stills to cover them:

- A line too long for one clip means MORE CLIPS. That is what `partition_beat`
  and the coverage plan are for, and it is a normal outcome, not a defect.
- No render-side gate may refuse a beat for being long. The single-take clause
  in `mouth_policy` was already corrected from terminal to warning on exactly
  this reasoning; the one-face-per-episode cap is the remaining refusal of this
  shape and must ROUTE rather than refuse.
- "Enough stills" is the same contract on the still spine: every beat that owes
  a still gets one, whatever the line length.

**THE TEST OF ANY FUTURE GATE:** if the remedy it names is a routing decision,
the build owes the routing, not the refusal. Refusing is only honest when the
alternative is shipping a defect.

**LANDED 2026-07-29 (CODER window, HEAD == origin `a14ecdfa`; suite
7551 passed / 27 skipped / 1 xfailed; Bible 17; `build_variants --check` 11
variants / 0 failures; canonical `9872624A` byte-identical across all three --
no node, widget, link or schema touched by any of them):**

- **`5efd2baf` WIRE-W1** -- the partitioner takes the FEWEST legal clips. It
  ran two walks over the segment count (every count for an EXACT cover, then
  every count again for a TRIMMED one), so an exact cover at a HIGH count beat
  a trimmed cover at a LOW one. Now one walk: exact, then a permitted trim,
  then advance. Measured over 798,510 (contract, target) pairs across 2,538
  contract shapes -- **46,949 plans changed, EVERY ONE a segment-count
  reduction, zero refusals introduced, zero count increases.** `184 ->
  [153, 33]` trim 2, and 185-240 all two segments, as codex pinned.
- **`a218b1f7` WIRE-W2** -- a cast-time image gap DECLARES itself.
  `DeferredImageGapError(RenderError)` in the new dependency-leaf
  `nodes/_otr_video_engines/render_errors.py`; `RenderError` moved with it and
  is re-exported from `render_driver`. Five cast-time sites raise it (one was a
  bare `ValueError` -- that is why r3's `:1985` was not in the RenderError
  list); the three post-image still-spine gaps and both wrong-aspect failures
  stay terminal. **Both fail-open swallows in ShotLock are gone.**
- **`a14ecdfa` WIRE-W6** -- the end card rides the body video's frozen final
  frame; `plan_backdrop` is DELETED. Terminal/presentation boundary per r4/A7.
- **`3e89d6b2` WIRE-W3a** -- `wan_i2v` gets its beat session. `session_identity()`
  AND the UNET-only hoist in ONE commit, because codex's r3 warning is real:
  the identity alone silences `SessionIdentityUnavailable` and the segment
  graph still runs `UNETLoader` every segment. **Measured: a 3-segment beat
  loads the UNET once and the CLIP/VAE three times each** -- the auxiliaries
  reloading is the NARROWED contract ("primary diffusion patcher once per
  beat; auxiliaries may reload"), not a shortfall, because `free_after_use`
  keeping umt5 and the 14B out of co-residency is the only thing between this
  lane and an OOM at 14,499 MB. Receipt mechanism shared in `wan_shared`, data
  per adapter. Suite 7561; mutation 7/7.

**WIRE-W3b IS LANDED.** The session half mirrors `eng_wan_i2v` (identity,
UNET-only `prepare`, `external_results` in `_build_graph` + `render_clip`,
`teardown` dropping the handles first). Three things are this adapter's own:

- **THE PING-PONG IS NARROWED, NOT RIPPED.** `eng_wan_ti2v` renders a
  VRAM-bounded short clip and mirror-extends it to the beat target -- on
  purpose, and it is the shipped 8 GB tier contract (`PBUG-20260723-02`), so it
  still does exactly that on a SINGLE-CLIP beat. On a coverage-planned segment
  it is forbidden: `_planned_length` renders the segment's own length whole, or
  refuses BY NAME before anything is staged (an off-ladder length means the
  stamped plan and the `frame_contract` disagree; a tier ceiling below the
  planned length means a ceiling and a plan contradict, which they CAN because
  WAN is deliberately out of `PLANNING_CAP_ENGINES`). The discriminator is
  `prepared["session_ctx"]["multi_clip"]` -- the only honest one available,
  because a planned segment's REQUEST is shaped exactly like a single-clip
  beat's.
- **The pipeline invariant came with it** (`eng_ltx_8gb`'s B4 invariant #1): a
  decode that returns a different count than the graph asked for now RAISES.
  The pad used to absorb an under-delivery for ANY reason, so a short decode
  was indistinguishable from a render that did what it was told -- and
  `test_wan_recipe_freeze`'s own fake decoder had been exercising the pad on
  every render since it was written.
- **`native_frame_count` + `extension_mode` ride every WAN receipt** through
  the shared `wan_shared._clip_from_raw` into the manifest. `frame_count`
  cannot answer the question: a padded clip carries the same number as a real
  one, which is exactly the evidence a pad forges. `wan_i2v` stamps them too
  (`"none"`, always) -- an ABSENT field is indistinguishable from an unanswered
  one, and a family where one adapter answers and its sibling does not gives
  the W5 grader no lane it can trust.
- **The hoist cost is MEASURED and added back to the budget** (r3's third
  bullet, and it is not cosmetic -- without it the session half BREAKS the lane
  it fixes). The cost model's `overhead` is "the resident model + fixed
  buffers"; hoisting the UNET moves those GB out of *free* before
  `_floor_length` reads it, so the same weights get charged twice and
  `MotionBudgetError` refuses renders that fit. `prepare` reads free VRAM
  either side of the loader graph and hands the delta to every segment -- one
  number per beat, so two segments of one beat cannot pick different lengths.

**ONE TEST WAS CORRECTED, not silenced.**
`test_ltx_8gb_session_identity.py`'s CONTROL asserted that `wan_ti2v` alone had
no session identity, which made it a control over exactly one engine --
`wan_i2v` gained one at WIRE-W3a and nothing in that file noticed. It now
asserts the whole SET against a named list carrying the chunk that added each
entry, so it fails in both directions.

**WIRE-W4a IS LANDED** -- the HuMo beat session. All four HuMo lanes declared
no `session_identity()`, so `BeatSession` refused every multi-segment beat from
them, and HuMo beats are dialogue, so they are long, so they are exactly the
beats the partitioner splits.

- **The hoist is WIDER here than on the WAN lanes, and that is a property of
  the family.** WAN renders with `free_after_use=True` so umt5 and the
  diffusion UNET are never co-resident; hoisting its CLIP would delete the one
  mitigation it has. HuMo renders FULLY RESIDENT by contract (BUG-265), so
  every loader is held for the whole render anyway -- hoisting changes how many
  times they are READ, not how much is held. `prepare` takes the UNET, the
  LoRA, umt5, the VAE and whisper; measured, a 3-segment beat loads each once.
- **The reclaim is the other half, and neither half works alone.** The LOUD
  `reclaim_idle_models` at the end of `render_clip` exists "so the resident
  stack drops back down before the NEXT SOAK BEAT starts". Run between two
  segments of ONE beat it would `detach(unpatch_all=True)` the very handles
  `prepare` hoisted -- the load count would still read 1 while the weights
  bounced to CPU and back every segment. Drop it WITHOUT the hoist and segment
  2 builds new umt5 and whisper handles beside segment 1's. So it is skipped
  between segments and run once at `teardown`; the cross-beat promise is
  unchanged.
- `loadaudio`/`audioenc` stay per segment deliberately -- they are the one
  thing that genuinely differs per segment on a lip-synced lane, and **W4b
  needs them there to have anywhere to land.**
- `_lora_is_skipped` is now THE one reading of "this tier runs LoRA-free". The
  session and the graph have to agree: hoisting a `lora` node the graph never
  defines wires a handle nothing reads; skipping one the graph DOES define lets
  it reload every segment.
- `prepare` builds its loader mini-graph BY FILTERING THE REAL GRAPH rather
  than re-spelling the loader inputs, and refuses NAMED when the two disagree
  about what this lane loads, or when a hoisted loader produces no handle.

**WIRE-W4b IS LANDED** -- a lip-synced segment is driven by its OWN audio.
Every segment of a multi-clip beat used to be handed the WHOLE beat's slice, so
a 3-segment HuMo beat rendered three clips all lip-syncing to the same waveform
FROM THE TOP: an assembled beat that says the opening of the line three times.
Nothing caught it, because every clip carried the right frame count and the
right init image -- only the sound was wrong, and no gate listens.

- The arithmetic is `coverage_plan.segment_render_window(plan, index, fps)` --
  pure and CPU-tested. `render_driver` adds the beat's own `start_s` and
  nothing else.
- **It is the RENDER window, not the visible one, and the difference is exactly
  `drop_head`.** A chained successor renders one frame EARLIER than it
  contributes, because its first frame duplicates its predecessor's terminal
  frame and is dropped at assembly. Give it the visible window and every
  chained segment's mouth runs a frame ahead of its own audio for the whole
  clip -- small enough to ship, and wrong on every segment but the first.
- `trim_tail` does NOT shorten the window -- the adapter renders those frames,
  so they need audio like any others. **W4b filled them from the master; W4c
  below corrects that to SILENCE.** The window length was right; the source
  was not.
- A single-clip beat -- and a one-segment plan -- asks for a byte-identical
  slice, which is what production renders today.
- Measured mutation CONTROL, recorded rather than chased: the negative-offset
  clamp in `segment_render_window` is unreachable, because
  `validate_coverage_plan` already refuses a first segment carrying a
  `drop_head`. It stays because the alternative is a negative ffmpeg seek.

**WIRE-W4c IS LANDED** -- and it CORRECTS W4b against r4/A4, which I had not
re-read closely enough when I built it. The ratified contract is *"conditioning
WAV duration EQUALS `render_frames`; copy only the `visible_frames` source
interval and APPEND SILENCE for `trim_tail_frames` -- never speech from the
next segment."* W4b took the whole window straight off the master.

- **That looked harmless and is not.** The trimmed frames are discarded at
  assembly, so nobody sees them -- but the AUDIO ENCODER SEES THE WHOLE
  WAVEFORM before a single frame is sampled, so the next beat's speech sitting
  in the tail conditions the frames that DO survive. On the pinned 184-frame
  case that is 2 frames of the next line leaning on a 31-frame take.
- `segment_render_window` now returns `SegmentAudioWindow(offset_s, copy_s,
  pad_s)`; `total_s` still equals `render_frames`, which is the generation
  length and is unchanged. `_slice_master_audio` grew `pad_tail_s` and builds
  `-af apad` plus an OUTPUT `-t` of the total -- the pair is the contract,
  because `apad` alone never terminates and a bare output `-t` would just
  re-cut the source. It also fixes the other end for free: a window running
  past the END of the master pads to length instead of emitting a short WAV.
- **The pad is IN the cache key and `SLICER_VERSION` moves 2 -> 3.** Two
  segments can copy the identical source interval and owe different silence, so
  a key that ignored the pad would serve the first one's WAV to the second; and
  every WAV already on disk describes the OLD contract for the same
  `(master, start, dur)`.
- The slicer now honours `OTR_FFMPEG`. It used the bare literal while
  `otr_credits_roll` already honoured the config, so on a box where ffmpeg is
  configured but not on PATH the credits rendered and the slice silently
  returned `""` -- which reads downstream as "this beat has no voice line".

**WIRE-W4d IS LANDED** -- r3: *"Prebuild and validate all segment requests and
audio slices BEFORE entering BeatSession; only terminal-image chaining stays in
the render loop."* Every segment request is now built before the session opens.

- **The builder is neither cheap nor pure**: it resolves stills off the ledger
  and SHELLS OUT TO FFMPEG to cut each segment's conditioning WAV. Run inside
  the session, that filesystem and subprocess work happened while the
  cross-process GPU lease was held and a 14B UNET sat resident -- once per
  segment, between renders. The lease is the scarcest thing this build owns and
  every heavy render on the box blocks its full 120 s timeout behind it.
- It is also where a bad request SHOULD surface: a builder that raises on
  segment 2 used to do it after two renders and a 6 GiB load.
- **The chain's terminal-frame substitution stays in the loop** and that is not
  an oversight: segment N's init image is segment N-1's last RENDERED frame,
  which does not exist until N-1 has run.

**STILL UNBUILT from that paragraph:** the durable slice RECEIPT (source PCM
hash, segment index, start sample, sample count, rate/channels, output PCM
hash) under the canonical episode directory rather than tmp. It is telemetry
for the W5 grader, not correctness, so it is filed rather than jammed in here.

**WIRE-W7 IS LANDED** -- the operator's three mouth rulings finally have an
OWNER. r3 MUST-FIX 11 named this as the gap nobody held: *"The plan has the
rulings and no owner for them."*

- `nodes/_otr_video_engines/mouth_policy.py` is the authority: pure, and
  IMPORTS NOTHING (a test asserts the import list is exactly `__future__`), so
  ShotLock can ask it at plan time and the W5 grader can ask it later without
  either becoming a cycle.
- **Decided from the FROZEN ROUTE, never from prose** -- the operator's own
  wording. The policy takes `engine_id`, `family`, `role`,
  `is_character_face` and no text at all; a test pins the signature, which is
  the strongest form of that promise.
- **THE SCHEMA IS NOT EXTENDED.** `still_plan_helpers` carries CLOSED enums and
  says adding a token is an operator decision, never a coder's. There is no
  `bears_a_mouth` field and W7 does not add one.
- ShotLock owns both halves: the per-beat gate in the cast-time preflight
  (after the route-freeze and the radio-is-host redirect, so the ruling is
  judged against the engine that will really render), and the episode
  cardinality after the coverage plans are stamped -- because the single-take
  clause is a question about the STAMPED plan.
- **MEASURED, not missed:** the one live route this closes is a cloud
  `audio_conditioned_video` lane (`cloud_seedance_2`, `cloud_wan_i2v_audio`)
  aimed at a character beat. Those declare no roles, so an operator can pick
  one; `_is_character_face_beat` says False, so the beat gets the ambient mix
  and a SCENE still -- an audio-in engine animating an image with no lips. It
  now refuses with the remedy in the message. `cloud_kling_avatar` is the
  CONTROL: same empty `roles`, same beat, answers HUMAN, because the refusal is
  about the family's relationship to a face and not about being a cloud lane.

**WIRE-W5 IS LANDED -- THE WIRING BLOCK IS COMPLETE.**
`nodes/_otr_video_engines/acceptance.py` is the pure grader and
`scripts/grade_episode.py` is r4/A6's "durable repository script": a grader
nobody can run is the same failure mode as an unowned ruling.

    python scripts/grade_episode.py --ledger <ledger.json> --manifest <manifest.json>
    0 = the episode delivered the route it froze; 1 = findings; 2 = unreadable

- **Per shot, both halves of A6.** `shots[].engine_id ==
  roles_effective[role]`, then every DELIVERED manifest row's `engine_id`
  against that same frozen value -- never against the shot row, because a
  rewritten row would agree with its own rewrite. A role missing from the
  frozen map is a finding, not a pass.
- **Histograms are CUT, and there is an experiment rather than an assertion:**
  a test swaps two shots' engines, shows `engine_histogram` is byte-identical
  either way, and shows the per-shot grader reports both.
- **The multi-clip honesty check is what WIRE-W3b's receipts were for.** A beat
  the plan splits may not deliver `extension_mode="ping_pong"`, and a
  `"none"` claim must have `native_frame_count == frame_count`. **Silence is
  not a pass** -- a multi-clip row with no receipt at all is reported, because
  that is exactly what a lane padding without saying so looks like. A
  SINGLE-clip beat may pad all it likes: that is the shipped 8 GB WAN tier.
- **Three refusals, each with a test:** it imports nothing but `__future__`
  (so it CANNOT query live routing state -- the director froze at plan time and
  grading against later environment state is a clock-domain mismatch); it never
  reads `engine_histogram`; and it grades SOURCE receipts, never a composited
  frame (kibitz r1: `test_credits_roll_spec.py:446-470` scrolls text over a
  deliberately constant backdrop, so "did the frame change" goes green on a
  frozen background because the overlay moved).
- `build_clip_manifest` now carries `native_frame_count` / `extension_mode`
  onto every row, the same passthrough shape as `recipe` -- a grader reading a
  field nobody stamps is a grader that always passes, and a test pins the stamp.

**FILED, NOT BUILT:** grading OBS PUBLICATION and the canonical artifacts (A6
says grade them separately). That is a filesystem question about the `otr/obs/`
contract, and it belongs with the 45-word run rather than ahead of it.

## NEXT: THE 45-WORD RUN OVER ALL 18 LOCAL VIDEO/STILL ENGINES

The operator's stated first priority, and **the only thing that proves any of
the wiring block.** Everything above is suite-and-contract green and NOTHING in
it is live-proven. The run is what turns "the code says it will" into "the box
did".

**OPERATOR RULING 2026-07-29 -- THE MOTION FLOOR, AND THE CREDITS EXCEPTION.**

Operator, on the beats: *"For video models, there needs to be video for every
beat. ... if the minimum is, like, four seconds, then we should have video for
four seconds."* And, separately and explicitly, on the credits: *"The credits
is an exception. I'm fine with a still for the credits, or ping-ponging. In
fact, for credits I'm fine with just a black background."*

**THE CREDITS QUESTION IS CLOSED. No eyeball is owed and no work is queued.**
WIRE-W6's held final frame stands as shipped -- same darkened drama imagery as
the looped clip it replaced, motionless, and it is what removes the manifest
read that made an all-`mesh_stage` episode unpublishable. The 2026-06-17 look
contract's "never credits-over-black" is RELAXED by this ruling -- black is
acceptable under the console. Do not spend a chunk making the credits backdrop
move; the operator has said twice that he does not care about it.

**THE BEAT RULE WAS ALREADY THE SHIPPED BEHAVIOUR, and it is now PINNED.**
For any contract permitting a tail trim, `partition_beat` renders the SMALLEST
LEGAL LENGTH at or above the beat target and trims the surplus -- so a beat
shorter than an engine's minimum gets real rendered frames, never a held
image. Audited across the live registry: **all 31 registered engines cover a
1-second beat with real video**, no engine declares `allow_tail_trim=False`,
and `google_veo` renders 100 frames (4.0 s) and trims 75 -- which is the
operator's sentence, executed, since 2026-07-25.
`tests/test_motion_floor_roster.py` is the roster gate that fails BY NAME if a
future adapter declares a minimum without the trim and reopens the still floor.

**WHAT THE PANEL FOUND THAT IS WORTH KEEPING** (kibitz r1,
`kibitz-runs/2026-07-29-motion-floor/r1/final.md`; codex `gpt-5.6-sol` high +
agy, Claude judge):

- **The first-ever live green episode was SEVEN DEAD-FLAT STILLS.** The proof
  cited at the top of this file logs `engine_histogram {"still_flat": 7}`, and
  `cheap_families.py:330` documents `still_flat` as *"A DEAD-FLAT still: the
  selected image held STATIC"*. It passed every gate we own, because a still
  engine emits N identical frames and satisfies every frame-count and coverage
  check. **Not a defect to fix -- `still_flat` is a declared still route doing
  its job -- but nobody should cite that leg as proof the VIDEO lanes work.**
  The 45-word run over all 18 local engines is what proves those.
- **A whole-frame motion check would PASS a frozen backdrop.**
  `tests/test_credits_roll_spec.py:446-470` proves col-3 text scrolls over a
  deliberately constant backdrop, so "did the frame change" goes green on a
  frozen background because the overlay moves. Relevant to WIRE-W5: grade
  source components BEFORE overlays, never the composited frame.
- **agy argued to KEEP the 52-second held frame**; recorded because a panel
  arguing for the defect is worth knowing about. The operator has since made
  the point moot in the other direction.

**ONE NEW OPEN ROW, filed not built (see OPEN BUGS):** the fewest-segments
preference can accept a disproportionate trim on a discrete menu whose largest
entry dwarfs its smallest. A bound was written, MEASURED and REVERTED -- it
cost more than it saved on real ladders. No shipped contract can produce the
pathological case.

**A TRAP THAT COST A RED HEAD, worth knowing before the next chunk:** the B7
forbidden sweep (`tests/test_b7_forbidden_sweep.py`) diffs **tracked** files
only, so a brand-new test file passes the gate while untracked and fails the
commit AFTER it lands. It flags `alias` as a runtime identifier. Fixed in
`a14ecdfa`; run the full suite once more after the first commit of any new
test file.

**THIS IS NOT LEAN-MEAN, AND LEAN-MEAN IS NOT ON THIS DOCUMENT.** Operator
direction 2026-07-29 moved LEAN-MEAN FRONT and TAIL to the Lean-mean campaign
section of `ROADMAP.md`, where they sit behind SFX and product expansion. A
window that opens on lean-mean has read a stale list -- see the struck-through
2026-07-24 rescope, and the voided CODER D / CODER G rows. **This plan's order
is: LOCAL-ENGINE OBS WIRING -> WAN 8-GB -> Randomizer A -> `dynamic_story` ->
SFX -> re-observe the parked story bugs. There is no lean-mean line in it.**

**WHY THIS STEP EXISTS.** The 45-word engine-coverage campaign ran to
completion over all 18 LOCAL engines: **11 publish to `otr/obs/`, 6 NO_RENDER,
and `mesh_stage` renders 7/7 and publishes nothing.** The six are NOT a stills
problem -- 5 of 6 are MULTI-SEGMENT COVERAGE (wan x2 at node 92, no
`session_identity()`; humo x3 at node 90, beat > cap with no per-segment audio
slicer) and 1 is a preflight string match (`ltx_video`:
`_is_deferred_image_gap`'s four needles at `nodes/otr_shot_lock.py:947-955`
miss the LTX-I2V wording, so ShotLock re-raises and node 91 never runs).

**BUILD FROM `kibitz-runs/2026-07-28-local-engine-obs-wiring/r3/final.md` AS
AMENDED BY `r4/final.md`.** codex's VERIFY-AT-BUILD checklist (`r4/codex.md`,
last section) is the adopted per-chunk gate. Order:

```text
WIRE-W1  partition   DONE 5efd2baf
WIRE-W2  typed gap   DONE a218b1f7
WIRE-W6  end card    DONE a14ecdfa
WIRE-W3a wan_i2v     DONE 3e89d6b2  (session: identity + UNET-only hoist)
WIRE-W3b wan_ti2v    <-- NEXT. Same session shape, PLUS the ping-pong work
WIRE-W4  HuMo        session BEFORE slicer; SUPPRESS eng_humo.py:525-531
                     per-segment reclaim or the hoist is evicted; conditioning
                     WAV = render_frames with silence padding for trim_tail
WIRE-W7  mouth-still ShotLock is the sole cardinality owner; ZERO OR ONE human
                     face, never inferred from prose
WIRE-W5  grader      per-shot frozen-route comparison; histograms are CUT
```

**THE `WIRE-` PREFIX IS LOAD-BEARING.** LEAN-MEAN FRONT also has a W0..W8
chain and LEAN-MEAN TAIL has a W8; a bare "W1" is ambiguous across three
blocks and has already misrouted one window. In this block the chunks are
named in `r3/final.md` as W1..W7 -- say `WIRE-W1` out loud, always.

**THREE OPERATOR RULINGS BIND THIS BLOCK** (recorded verbatim above): a still
floor is legal ONLY where the partition math is impossible, never where an
engine refused; every audio-in beat gets a still with a mouth; the lips may be
a person OR A RADIO.

**A2 stays HELD. 7d stays PARKED. THE LAW holds. Preserve other windows' dirty
paths.** The 13 CLOUD engines stay parked pending explicit spend approval.

### Still true from the prior step (2026-07-28, the second encoder)

**7d IS STILL PARKED** until the operator is at the desk -- his call, recorded
in `docs/TRAVEL_RELAY_PROTOCOL.md`. Do not start it from a remote window.

**TWO OPERATOR DECISIONS ARE STILL OPEN, both untouched by this window:** the
profile retire-now vs retire-later scope (which gates A2 and nothing else), and
the LTX per-beat recipe capability.

**THE REMOTE-SAFE QUEUE IS DOWN TO ONE FILED ROW, and it is a DESIGN job, not
a bug fix:** a small-canvas variant of the credits card. At 512x288 -- the
ltx_8gb tier -- col1 is still 65px past its footer even fully abridged (it was
131px), and 640x360 is 12px over. Those canvases are LOGGED at ERROR now
instead of clipped in silence, and the log says what it needs: at 288 lines the
three-column console is a polite fiction, and the answer is a card designed for
a small canvas, not more ladder. NOT a blocker for anything.

**STILL OWED ON THE GPU LANE, and none of it is remote:** the clamped
confirmation of ltx recipe v2, then a WAN prequalification sweep (the mechanism
is built and named on both WAN tiers; no WAN sweep has ever run), then 7d.
**These do NOT gate the wiring block above -- WIRE-W1..WIRE-W7 are code, not
GPU time.** The block order after the wiring block: WAN 8-GB -> Randomizer A ->
`dynamic_story` -> SFX -> re-observe the parked story bugs. **Lean-mean is not
in it.** Operator direction 2026-07-29 moved LEAN-MEAN FRONT and TAIL off this
plan entirely, to the Lean-mean campaign section of `ROADMAP.md`, where they run after the randomizer
and SFX.

**CARRY FORWARD, and this window is the sixth consecutive proof:** run the
mutation round even after a QA fan-out clears the change, run the FAN-OUT
BEFORE THE PUSH, and treat as presumed decorative until proven otherwise -- a
test that verifies a thing it also CONSTRUCTS; an exception type asserted
without its message; an assertion inside a bare `except`; an assertion against
the CONSTANT the code uses rather than the documented literal. **Three
additions this window earned, all of them found on my own new code:**

1. **A STRUCTURAL SWEEP CAN BE SATISFIED BY A DEFINITION.** `_has_proof`
   matched its markers as substrings, and `wan_shared.py` DEFINES both proof
   helpers -- so `def ffprobe_counted_frames(` satisfied the check on its own,
   and the one module able to regress its real comparison was the one module
   neither gate could see it in. Match CALLS (AST), not text.
2. **A SWEEP THAT FINDS NOTHING PASSES EVERY GATE BUILT ON IT.** The first
   draft's subprocess-alias test was simply wrong, the roster came back empty,
   and both gates went green over an empty set. Every roster gate now asserts
   that named engines ARE BILLED, not merely that nobody failed.
3. **FIXING A BLIND TEST DOES NOT FIX ITS NEIGHBOUR.** The roster gate was
   rewritten to see the second encoder; the ORDERING test in the same file
   still regexed the one old spelling and stayed green with the proof moved
   before the encode. When you widen one sweep, sweep the file for its
   siblings.

**RE-PIN THE CITE WHEN YOU TOUCH A ROW.** Confirmed again: every cite this
window touched had moved.

## SUPERSEDED -- the second-encoder step (2026-07-28, BOTH HALVES DONE)

The step was: widen the M7 roster gate to find a clip WRITER rather than a
known call spelling (expecting it to go RED), give
`nodes/_otr_shared/scope_draw.py::encode_silent_mp4` the two proofs the first
encoder has, then close `cheap_families`' four `still_*` count proofs. All of
it shipped -- `27a4f97c` and `afeb5b84` -- and is tombstoned in OPEN BUGS. The
gate went red on exactly the four `viz_*` engines, as predicted, and green by
fix rather than by narrowing. See the header for what the fan-out found on top.

---

## SUPERSEDED -- the three remote-safe rows (2026-07-28, ALL THREE LANDED)

The ranked lane was: (1) the `by_engine` roll-up keeping only the first clip's
receipt per engine, (2) the credits-card `video_suffix` write with zero readers
plus the `_row()` clamp rider, ordered AFTER (1) so the card could not draw
confidently-wrong per-engine data, and (3) the encoder frame-count decision.
All three shipped -- `bcaab4db`, `24f4251a`, `48e3c6fb` -- and are tombstoned
in OPEN BUGS. The third was not the choice the row framed: see the header.

---

## SUPERSEDED -- the ranked-queue step (2026-07-27, all but A2 now landed)

**LANE 1 AND LANE 2 ARE DONE.** **7d IS STILL PARKED** until the operator is at
the desk -- his call, recorded in `docs/TRAVEL_RELAY_PROTOCOL.md`. Do not start
it from a remote window.

A codex `gpt-5.6-sol` high + agy `Gemini 3.6 Flash (High)` panel, then a Fable
consult, went through the OPEN BUGS list. **The panel corrected three of the
five anchor rows, cut one entirely, and added one nobody had filed.** One fix
landed the same session (`54b3626b`, the mux). Read the triage doc before
touching any row -- several fix SHAPES in the OPEN BUGS entries above are now
known to be wrong, and those entries say so inline.

**B5 FIRST: it is a dependency, not a peer.** Whether the profile family is
retained or retired changes the value AND the acceptance target of A1, A2 and
A6. Get that ruling before coding any of the three.

Then, in order:

1. **A1 -- the GGUF policy ceiling.** ONE policy-admission calculation before
   BOTH cache reuse and load. A resident model returns at
   `_otr_model_loader.py:982-992` without entering preflight, and
   `GGUFLoadConfig.reuse_key()` (`_otr_gguf_backend.py:435-439`) excludes the
   ceiling -- a preflight-only fix misses the cache-hit path entirely. Test
   permissive-cache -> stricter-request at the same load identity.
2. **A6 -- the Q4 artifact has neither an expected size nor a SHA**
   (`_otr_gguf_backend.py:56-60` and `:226-233` both give `None` for
   `Q4_K_M`). The checks are conditional, so a truncated download passes
   readiness on the very quant the shipped 8 GB profile selects. Pin both;
   reject a non-zero short file.
3. **A2 -- generate the applied-overrides echo FROM the applier's flattened
   map.** The override happens at submission
   (`scripts/otr_canonical_api_run.py:157` -> `apply_profile_to_workflow`), not
   from the validator's env export, and `nodes/_otr_workflow_apply.py:492-540`
   ALREADY flattens `llm`. Only the printed echo (`scripts/otr_api.py:816-825`)
   is stale -- adding `llm` by hand leaves the next drift intact.
4. **A4 -- make the LTX adapter refuse a missing or stale init image**, and
   replace the fallback assertions in `tests/test_video_motion.py:340-344`.
   CONFIRMED reachable; not a misread.
5. **B4 -- complete `ShotRow`.** No longer an operator ruling: ShotLock stamps
   `role`, `char_id`, `start_s`/`dur_s`, `coverage_plan` and
   `coverage_contract`, none of which exist on a model declaring
   `extra="forbid"` -- so `ShotRow(**real_row)` raises on every real ledger and
   the "live safety net" other docs cite cannot validate one shipped episode.
   The repo's own `observability` / `requires_mesh_portrait` precedent settles
   the shape. No product question is left.
6. **A5-lite -- one `dtype == uint8` assert at the encoder boundary.** Cut as a
   LIVE bug (every producer pipes exact-size uint8 and ffmpeg raises on a short
   write); the residual is a future float32 caller getting a clean receipt over
   4x the bytes.
7. **The `frame_count` asymmetry** (new OPEN BUG below). Copy the four
   siblings' M7 probe line into `eng_humo` and `eng_ltx_av`.

**CUT -- do not re-derive:** A3, the `provider_side` redirect regression
(covered by `test_video_render_driver_perbeat_audio.py:319-325`,
`test_video_platform_aseam.py:903-920` and `test_still_plan_parity.py:114-116`);
agy's heavy-import finding (the enforced gate
`test_capability_profiles.py:481-503` excludes the audio lane BY DESIGN and says
so in its own docstring); B1 the WAN knob rename (default: leave); B2 the
style-tail enum (default: ratify the exemption).

**RE-PIN THE CITE WHEN YOU TOUCH A ROW.** Every line cite checked during the
triage had moved: `_is_cloud_video_engine` is `render_driver.py:1599` not
`1274-1295`; the "NO FALLBACK to text-only" refusal is `:2148` not `1801-1817`;
`_use_i2v` is `eng_ltx_video.py:583` not `559-572`. The defects are mostly still
real; their coordinates are not.

**CARRY FORWARD:** run the mutation round even after a QA fan-out has cleared
the change -- it has now found real defects the lenses missed on three
consecutive chunks -- and treat a test that verifies a thing it also CONSTRUCTS,
or an exception type asserted without its message, as presumed decorative until
proven otherwise.

---

## SUPERSEDED -- the pre-triage OPERATOR'S PICK step (2026-07-27)

**LANE 1 AND LANE 2 ARE DONE** (see the header). **7d IS STILL PARKED** until
the operator is at the desk -- his call, recorded in
`docs/TRAVEL_RELAY_PROTOCOL.md`. Do not start it from a remote window.

Every video adapter now binds its recipe from code and marks its measurement
artifacts specifically. What is left in the remote-safe queue is smaller and
each item wants a decision rather than a coder's judgement:

1. **`schemas.py`'s `ShotRow` -- wire it or demote it IN WRITING.** A closed
   model (`extra="forbid"`) that no boundary enforces, missing `beat_id`,
   `role`, `char_id`, `start_s`, `dur_s`, `coverage_plan` and
   `coverage_contract`. Not a live break -- nothing validates a real ledger
   through it -- but other docs in this tree cite it as a live safety net, and
   any future `model_validate` at any boundary would hard-fail every real shot.
   Suite-provable either way; the DECISION (complete it vs. demote it) is the
   whole chunk.
2. **The credits-card display gap.** The recipe reaches the durable ledger but
   `_draw_models` never reads `video_suffix` -- one write, zero readers. Belongs
   to whoever owns the credits card, and note the LANE 2 rider below before
   wiring it.
3. **The `by_engine` roll-up** (new OPEN BUG below) -- a one-line decision about
   whether a lossy summary beside a lossless one is worth changing.

**THE HIGHER-VALUE WORK IS THE OPERATOR'S SEQUENCE**, and none of it is remote:
the clamped confirmation of ltx recipe v2, a WAN prequalification sweep (the
mechanism is now built and named on both WAN tiers, and no WAN sweep has ever
run -- both v1 dicts are today's shipped defaults, stated honestly as such),
then 7d.

**IF A NEXT WINDOW TAKES A CODER LANE ANYWAY, carry these two forward:** run the
mutation round even after a QA fan-out has cleared the change -- it has now
found real defects the lenses missed on three consecutive chunks -- and treat a
test that verifies a thing it also CONSTRUCTS, or an exception type asserted
without its message, as presumed decorative until proven otherwise.

---

## SUPERSEDED -- LANE 2 (DONE @ `71e231ec` and `8424f369`)

**LANE 1 IS DONE** (see the header). **7d IS STILL PARKED** until the operator
is back at the desk -- his call, recorded in `docs/TRAVEL_RELAY_PROTOCOL.md`.
Do not start it from a remote window.

**LANE 2, the next default remote lane: name the DEPARTURES in the
prequalification receipt.** Suite-provable end to end, no GPU, no operator
judgment. It is already an OPEN BUG (found 2026-07-27 by the kibitz codex seat,
verified): `recipe_receipt()` returns a single generic `+prequalification`
suffix, so a winning sweep artifact cannot prove WHICH knob values produced it
-- the ledger says a sweep ran, not which cell. Shape: name the departures from
the frozen recipe, e.g. `..._v2+prequalification[tiled_vae=off]`. No workflow
schema work; the receipt is a string on the manifest row.

**IT IS CHEAPER NOW THAN WHEN IT WAS WRITTEN, and that is the reason to take it
next.** It was deferred out of `dcdcccde` because it touches `session_identity`
and several call sites. LANE 1 has since put the receipt behind ONE shared
`wan_recipe.recipe_receipt(frozen, consent_env)`, and the same shape exists in
`eng_ltx_8gb.recipe_receipt()`. Computing the departure list needs the resolved
values and the frozen dict in one place, which is exactly what
`_resolve_render_config` already returns on both WAN adapters.

**THREE THINGS LANE 1 ESTABLISHED THAT LANE 2 MUST NOT UNDO:**

- **A knob that cannot bind is IGNORED, never FATAL.** Outside the consent act
  the demoted vars are NAMED and never PARSED. A departure list must be computed
  from RESOLVED values under the consent act only -- computing it on a
  production leg would re-introduce parsing of knobs that cannot bind.
- **A run under the consent act MARKS ITS OWN ARTIFACTS**, and the mark rides
  into `stamp_durable(meta.render_engines)`. Making the mark richer must not
  make it absent on any path.
- **PER-ADAPTER CONSENT, per-adapter receipt.** Three adapters now carry this
  mechanism (`ltx_8gb`, `wan_ti2v`, `wan_i2v`). If LANE 2 hoists anything, hoist
  the MECHANISM and leave the DATA where it is.

**AND THE TEST TRAP THAT COST THE MOST TIME IN LANE 1 -- it will recur:** a test
that sets the imported CONSTANT rather than the literal an operator types can
never notice the adapter reading a var nobody sets; and a receipt checked only
on a HAND-BUILT raw stays green when `render_clip` stops putting it there. Both
survived three QA lenses and were caught only by mutation. Drive the real
`render_clip` (`tests/test_wan_recipe_freeze.py` has an ffmpeg-free harness for
it) and assert the DOCUMENTED literal.

**OTHER REMOTE-SAFE LANES, if LANE 2 is not the operator's pick:** the
`schemas.py` `ShotRow` decision (wire it or demote it in writing -- it is a
closed model no boundary enforces, and other docs cite it as a live safety net),
and the credits-card display gap (the recipe reaches the durable ledger but
`_draw_models` never reads `video_suffix`). Both are in OPEN BUGS with their
shapes already spelled out.

---

## SUPERSEDED -- LANE 1 (DONE @ `71753cb4` and `3acc7fed`)

**PREQUALIFICATION IS DONE** (see the header). **7d IS PARKED** until the
operator is back at the desk -- his call, recorded in
`docs/TRAVEL_RELAY_PROTOCOL.md`: it is the next real milestone and it wants his
eyes on it. Do not start it from a remote window.

**LANE 1, the default remote lane: freeze the WAN recipes, mirroring B6.**
Suite-provable end to end, no GPU, no operator judgment. `eng_wan_ti2v` reads
loader class, tiled-VAE class, all three weight NAMES, sampler, scheduler,
steps, cfg, shift, negative and four VAE-tile vars straight from the
environment; `eng_wan_i2v` reads six INLINE in `_build_graph` with bare
`int()`/`float()` -- no range check, no named refusal. Neither emits a `recipe`
receipt at all, so a WAN clip stamps `recipe: None` and there is not even a
wrong receipt to catch the drift with.

**B6 + v2 ARE THE SHIPPED REFERENCE.** Read `docs/2026-07-27-b6-qa-findings.md`
and `nodes/_otr_video_engines/eng_ltx_8gb.py`, and copy four things that were
each earned by a panel finding: (1) the recipe is a versioned dict plus a
single `LTX8_RECIPE`-style active binding -- NEVER edit a versioned dict in
place, or existing receipts stop being interpretable; (2) the version lives IN
the receipt string so it reaches the durable ledger and moves `session_identity`
for free; (3) outside the consent act a demoted knob is NAMED, NEVER PARSED, so
a stale malformed value cannot kill a leg it has no effect on; (4) inside the
consent act every knob fails CLOSED by name -- `_config_number` and the
`_TRUTHY`/`_FALSY`/`_T5_DEVICES` refusals, one rule with one implementation.

**AND THE TEST TRAP THAT COST THE MOST TIME, which WAN will hit too:** when you
flip a frozen default, every test whose override happened to AGREE with the new
value goes decorative in silence. Six did on the v2 flip. Each override must
OPPOSE the frozen value, and the test should assert what it opposes.

**HOW THE B6 FORK WAS RESOLVED: option (a), freeze what ships today.** The
judgment orders "mechanics first, MEASURE second, freeze third" and no
measurement had happened, so there was no measured selection to freeze. Option
(a) was taken because it is behaviour-preserving on any box that set nothing
and fully reversible -- prequalification measures and produces v2. **The code
says so in its own words rather than implying a measurement it does not have:**
`LTX8_RECIPE_V1`'s comment states plainly that these are today's shipped
defaults, each with a recorded reason (the T5 offloads to CPU because
`t5xxl_fp16` alone is ~9 GB -- load-bearing, not an optimisation; tiled decode
is OFF because core `VAEDecode` handles the 8 GB peak at the smoke canvas).
`docs/2026-07-26-8gb-1080p-arc-judgment.md:188` says "MEASURED"; it was NOT
rewritten to match what shipped, because a judgment is a record of what was
decided, not a living doc -- the departure is recorded here and in the code.

**THE "PREQUALIFICATION SIGNAL" QUESTION IS ANSWERED: an explicit env var,
`OTR_LTX_8GB_PREQUALIFICATION`, truthy `{1,true,yes,on}`.** Deliberately NOT
"the absence of an episode ledger" or any other ambient condition: a signal you
can arrive at by accident is one a production leg can arrive at by accident.
Present-but-falsy (`0`, empty, `no`) is a production leg.

Two rules B6 established that the next window must not undo:

- **A knob that cannot bind is IGNORED, never FATAL.** Outside the consent act
  the demoted vars are named in a warning and NEVER PARSED. The first draft
  parsed then discarded, which meant a stale `OTR_LTX_8GB_STEPS=not-a-number`
  in a long-booted server's environment would kill a leg over a value with no
  effect on it -- `PBUG-20260723-02` wearing the opposite mask.
- **A run under the consent act MARKS ITS OWN ARTIFACTS.** A prequalification
  clip stamps `..._v1+prequalification`, because `recipe` rides the manifest
  into `stamp_durable(meta.render_engines)` and a sweep artifact must never be
  mistaken for a production one in the record that outlives the run.

**WHAT PREQUALIFICATION ACTUALLY DOES NOW:** boot with
`OTR_LTX_8GB_PREQUALIFICATION=1` and the knobs bind again, range-checked and
fail-closed exactly as before, with every honoured override logged. Measure T5
device on/off and tiled decode on/off at 512x288, then freeze the winner as
`LTX8_RECIPE_V1` v2 -- bump the version IN the `RECIPE_LTX8_I2V` string, which
moves the session identity for free. Record: `docs/2026-07-27-b6-qa-findings.md`.

**A SECOND FACT THAT BEARS ON THE ORDERING (B5 panel, verified):**
`render_single` and both HTTP entry points never reach the canvas seam -- they
use the older ledger-free `build_request` and default to
`OTR_VIDEO_RENDER_CANVAS` (832x480). **That means the 7d-preflight that "proved
the GPU" ran at 832x480, not at the production canvas.** The production canvas
for `ltx_8gb` has still never been exercised live, so prequalification is the
first time it will be.

---

## SUPERSEDED -- B5 (DONE @ `a0141cdd`)

**HEAD == origin `5929e19a` at entry.** Suite 7134 / Bible 17 / canonical `9872624A`.
Authorities, read ALL THREE first: `docs/2026-07-26-8gb-1080p-arc-judgment.md`
(the architecture), `docs/2026-07-26-o1-canvas-arc-judgment.md` (the canvas seam)
and `docs/2026-07-26-dir-override-arc-judgment.md`.

**DONE:** B1a `8caf3516`, B2a `55c8a811`, B2b `582dfbd8`, the post-code QA fixes
`ea1652f9` / `f33c5e15` / `fdeee600`, QA-4 `823b9929`, the `*_DIR` tripwire
`095be05b`, B1b-0 `b214481b`, B1b `d708408d` (the hoist + the 4 GiB floor),
**B3 `b23fc035`** (the LTX-only effective contract + the WAN topology
regression) and **B4 `5929e19a`** (the ping-pong deleted, the ladder moved onto
the contract, two invariants in its place).

**B5 (NEXT) -- the canvas seam, fail-closed.** Derive and validate `(w, h)` from
`ledger.video.canonical_canvas` after route locking, thread it through every
segment request, and SUPPRESS the `OTR_VIDEO_LANDSCAPE_CANVAS` overwrite when a
stamp is present. Reject unless positive, /32, exactly 16:9, and 25 fps.
**Validate BEFORE `BeatSession` opens** -- today `render_driver.py:2902-2905`
opens the session (which prepares, i.e. LOADS) and per-segment `assert_usable`
only runs later at `:2760-2765`, so after B1b that means loading a 6.34 GiB
checkpoint before rejecting a bad canvas. **`wan_shared._dims` is NOT to be
touched** -- editing a shared default to satisfy an LTX gate is the cross-lane
damage this block exists to avoid, and the fallback is unreachable there anyway
because a missing stamp is already terminal.

**B6 -- freeze the measured recipe in CODE.** The profile schema accepts only
`device_policy` / `dtype_policy` / `max_render_frames`, so T5 device, tiled VAE
and sampling have NO end-to-end channel, and per `PBUG-20260723-02` the env vars
cannot bind on a production leg either. Freeze the measured selection into a
versioned `ltx_8gb` recipe in code and demote the env vars to
prequalification-only, logging a WARNING whenever an override is honoured there.
The generic `_get_engine_setting` accessor is CUT (it would preserve hidden
production env channels).

**THEN, in order:** prequalify 512x288 (fresh boot per cell, canonical path
only, `fraction = 8192.0 / detected_total_mib` set BEFORE CUDA init, probe
started BEFORE `BeatSession`; label it PREQUALIFICATION), then **7d** -- the
canonical 237-frame opening beat, `[65,65,65,49]`, trim 4, `RESULT SUCCESS` +
`obs_publish OK` + the asset on disk.

**A PROFILE MAY NOW PIN AN LTX CEILING.** B3 landed with that blocked because
the ping-pong laundered the plan-vs-adapter disagreement; B4 removed it. Pinning
`video.max_render_frames` on `config/profiles/otr_8gb_ltx.json` is what makes B3
reachable on a live leg at all -- it is currently production-inert. Do it as
part of the prequalification step, not before, and remember
`docs/ENGINE_MATRIX.md` reports the DECLARED contract only (see OPEN BUGS).

---

## SUPERSEDED -- B3 + B4 (DONE @ `b23fc035` and `5929e19a`)

`max_render_frames` is NOT a planning cap: WAN reads 17, renders short, then
PING-PONGS to the beat length, so applying it before `partition_beat()` would
turn every WAN beat into a pile of 17-frame renders. The effective contract was
scoped strictly to `engine_id == "ltx_8gb"` and the WAN regression shipped in the
same commit. Ripping ping-pong is LANE-SPECIFIC -- load-bearing for WAN, a
correctness hole for LTX.

---

## SUPERSEDED -- B1b (DONE @ `d708408d`)

**HEAD == origin `095be05b` at entry.** Suite 7071 / Bible 17 / canonical `9872624A`.
Authorities, read ALL THREE first: `docs/2026-07-26-8gb-1080p-arc-judgment.md`
(the architecture), `docs/2026-07-26-o1-canvas-arc-judgment.md` (the canvas
seam + the five channels) and `docs/2026-07-26-dir-override-arc-judgment.md`
(which env channels can and cannot reach a loader). They supersede the 7b
judgment's O1/O4 framing.

**DONE:** B1a `8caf3516`, B2a `55c8a811`, B2b `582dfbd8`, the POST-CODE QA fixes
`ea1652f9` / `f33c5e15` / `fdeee600`, QA-4 `823b9929` (the single-clip path), and
`095be05b` (the `*_DIR` deprecation tripwire).

**THE QA FAN-OUT DID NOT RUN BEFORE THOSE FIRST THREE PUSHES. It should have.**
The operator caught the omission; running it afterwards found FIVE code defects
and six test defects in code that was green, mutation-proven with controls, and
already on origin -- the sixth time a panel has done that in this project. All
five are now fixed (`docs/2026-07-26-b1b2-qa-findings.md` has the full list and
status). Two could break production: a path guard that falsely REFUSED correct
configs (`abspath` folds neither case nor junctions, and this box runs through a
junction), and a raising baseline identity read that STRANDED the GPU lease for
the life of the server (when `__enter__` raises, `__exit__` never runs).
**Run the fan-out BEFORE the push, not after.**

**THE IDENTITY LIE IS NOW CLOSED ON BOTH PATHS.** `823b9929` routed
`_ckpt_path` / `_t5_path` through `_loader_token_path`, the ONE authority, so the
single-clip gate (`assert_usable`) and the multi-segment gate
(`session_identity` -> `resolve_session_config`) can no longer disagree about
which file is the checkpoint. The pre-code panel killed my own proposal first: a
wholesale `assert_usable = resolve_session_config()` rewrite would have silently
dropped the 4 GiB integrity floor, which had zero coverage. `095be05b` then
closed the same lie one level up -- a `*_DIR` override that the LOADER cannot
see is now terminal, because ComfyUI resolves the graph's bare basename through
`folder_paths` and `*_DIR` never touched that channel. Scoped to `ltx_8gb`: the
Wan suites still use `*_DIR` as their no-ComfyUI mock seam, so `wan_shared` took
an additive split only. Both mutation-proven with controls that name controls as
their targets.

**B1b (NEXT) -- the weights are still not shared.** `BeatSession` now OPENS a
multi-segment session, but `Ltx8gbEngine.load()` only resolves node CLASSES and
the graph still carries its own `ckpt`/`clip` nodes, so every segment
re-executes `CheckpointLoaderSimple` + `CLIPLoader`. Hoist them: `prepare()`
runs them ONCE off the frozen config into `prepared["external_results"]`,
`_build_graph` OMITS those three ids and keeps its wires, `render_clip` passes
`external_results` through. **Transaction in the same slice** --
`motion_common.prepare()` releases only the GPU lease today, so a
ckpt-loads-then-T5-fails path strands a patcher; use `on_result` to register
each handle as it lands and unwind in REVERSE. Remove per-segment patcher
discovery from `render_clip`. Prove exactly ONE checkpoint load, ONE T5 load,
ONE model-sampling construction per multi-segment beat.

**THEN, in order:**

- **B3 + B4** -- the LTX-only effective contract, then delete ping-pong.
  **`max_render_frames` is NOT a planning cap**: WAN reads 17, renders short,
  then PING-PONGS to the beat length, so applying it before `partition_beat()`
  would turn every WAN beat into a pile of 17-frame renders. Scope the effective
  contract strictly to `engine_id == "ltx_8gb"`; a WAN regression proving
  `max_render_frames=17` does not move its coverage-plan topology ships in the
  same commit. B4 only AFTER B3: `render_driver.py:2982` already hard-asserts
  `got == segment.render_frames`, and ping-pong is what currently hides a
  non-`8n+1` segment. Ripping ping-pong is LANE-SPECIFIC -- load-bearing for WAN.
- **B5 + B6** -- canvas seam fail-closed (positive, /32, exactly 16:9, 25 fps),
  validated BEFORE `BeatSession` opens (today `render_driver.py:2902-2905`
  loads first and `assert_usable` runs at `:2760-2765`). Do NOT touch
  `wan_shared._dims`. B6: the profile schema accepts only `device_policy` /
  `dtype_policy` / `max_render_frames`, so T5 device / tiled VAE / sampling have
  no channel -- freeze the MEASURED selection into a versioned recipe in CODE,
  env demoted to prequalification-only.
- **Prequalify** 512x288 on the 16 GB box: fresh boot per cell, canonical path
  only, `fraction = 8192.0 / detected_total_mib` set before CUDA init, probe
  started BEFORE `BeatSession`. Label it PREQUALIFICATION.
- **7d** -- the canonical 237-frame opening beat.

**OPEN, deferred by operator (2026-07-26): the Bug Bible update.**
`PBUG-20260723-02` declares itself bible-worthy but has no `BUG_BIBLE.yaml`
entry, and that one rule -- a contract declared only in a process-launch
environment cannot bind work submitted to an already-running server -- has now
explained C1, C1b, the canvas, and the 8 GB levers. Do it when the build lands.

---

## SUPERSEDED -- 7b BLOCKERS: THREE LANDED; the canvas framing (see above)

**Updated 2026-07-27 (overnight, remote Cowork), HEAD `8f41af27` == origin.**
Suite **6983 passed / 27 skipped / 1 xfailed**; Bible 17; link validator 0
violations; canonical `9872624A` (moved by C1, intentionally).
**Read `docs/2026-07-27-7b-blockers-arc-judgment.md` FIRST** -- full r1-r4 arc,
8 agent calls, and the authority for this step.

**LANDED:** `7f4644a1` C1 (node 87's `max_render_frames` descriptor -- the
profile-ceiling channel was dead at its first hop), `ac609d25` C2 (the
plan-vs-output proof's fail-OPEN predicate), `8f41af27` C1b (the same dead
widget in ALL ELEVEN variant workflows). All mutation-proven, C2/C1b with
controls.

**THE FINDING THAT MATTERED MOST:** `variants/otr_8gb_wan.json` carried an
orphan widget value of **17**, not the harmless `0` -- matching
`config/profiles/otr_8gb_wan.json:56`, the ONLY shipped profile pinning
`max_render_frames`. The WAN 8GB ceiling had been configured and silently
ignored since it shipped. Found because the wiring script REFUSED an
unexpected value rather than assuming one.

**GPU IS PROVEN.** The server path is a JUNCTION to this repo (identical
SHA-256, same HEAD) -- every "live proof" in this build rested on that and it
was unverified until now. First live render of this architecture PASSED:
`ltx_8gb`, 25 frames, 20.8s, `frame_count=25` exactly, VRAM 3004 MB. That is
`7d-preflight`, NOT qualification.

**NEXT, AND IT IS A HARD 7d BLOCKER -- O1, THE CANVAS.** Both seats
independently. `build_request_from_shot` overwrites the canvas to `1472x832`
for every non-face engine (`render_driver.py:2268-2273`), with deliberate
per-engine branches after it for `ltx_video` and `ltx_av` but **none for
`ltx_8gb`**; `OTR_VideoRenderBatch` passes no canvas
(`otr_video_render_batch.py:372-373`) and `build_request` hard-codes 25 fps.
So `otr_8gb_ltx`'s 512x288 render canvas is displaced by 1472x832 on the tier
that exists precisely because 8GB cannot afford the big canvas. Deliberately
NOT fixed overnight: the two seats prescribe different remedies, the
surrounding comments document per-engine canvases that exist for real quality
reasons (BUG-LOCAL-412, "LTX-2B re-noises into mush at 1472x832"), and it is a
hot path every engine traverses. Likely shape: an `ltx_8gb` branch consuming
the already-stamped `ledger.video.canonical_canvas`
(`otr_shot_lock.py:1537-1541`); no new widget is needed.

**THREE MORE OPEN, all verified (detail in the judgment):** O2 a THIRD
validation bypass -- exported `run_episode` renders without
`resolve_final_shot_engines`/`assert_coverage_plans` and the soak calls it
directly; O3 `run_graph` cannot accept preloaded results, so 7c's loader
removal has a required 6-step order; O4 the 169-frame seam needs
`opening_duration_sec`/`crossfade_ms` accepted by the profile schema
(`render.frame_budget` is INERT in episode mode -- it is not the mechanism).

**ORDER:** O1 canvas -> C3 (per-ENGINE policy registry, alias-resolved, equal
to `registry.all_engine_names()`; typed `UnresolvableEngineError` vs
`InvalidEngineConfigError` at all three swallow sites) -> C4a receipt as a
SIBLING of `coverage_plan`, one builder for BOTH paths -> C5a offline
preflight tests -> O2 -> C6 -> O4 -> **7c** -> C4b -> **7d**.

---

## SUPERSEDED -- 7b FORK SETTLED; FOUR BLOCKERS ARE NEXT

**Updated 2026-07-27 (overnight, remote Cowork), HEAD `07a84627` == origin.**
Suite **6925 passed / 27 skipped / 1 xfailed**; Bible 17; canonical `5377914B`
byte-identical. Read `docs/2026-07-27-multiclip-7b-fork-judgment.md` FIRST --
it is the authority for this step and it supersedes the A-vs-B framing in
`docs/2026-07-26-chunk-7b-window-prompt.md` and
`docs/2026-07-27-next-window-prompt-nogpu.md`.

**THE FORK IS DECIDED. Neither A nor B.** Option A (refuse on
env-vs-declaration) is CUT: it breaks six documented operator-knob tests, cannot
reach `OTR_ACTIVE_PROFILE`, and enumerates its inputs forever. Option B is
DEMOTED from "the fix" to "an optimisation", because `render_driver.py:2952-2958`
**already makes the divergence terminal on the multi-segment path** by comparing
the rendered OUTPUT to the plan -- one predicate that catches all fifteen env
vars, the profile ceiling, the boomerang and the provider clamps without
enumerating any of them. What is missing is that same proof on the
SINGLE-segment path, which is the only path production runs today.

**LANDED THIS WINDOW (both green, pushed, mutation-proven):**
`499541b6` 7b-1 -- a malformed env value may no longer take `eng_ltx_av`'s
IMPORT down (a `ValueError` at module scope meant the adapter never registered
and `frame_contract_for` answered `SINGLE_ONLY` for it, so one typo silently
removed an engine). `07a84627` 7b-6 -- the boomerang tripwire: `ltx_video`
declares 169 and returns 193 by default, deferred to 7c ON PURPOSE with tests
that say so and tell the 7c author to delete them rather than relax them.

**DO NOT START THE RESOLVER YET. FOUR BLOCKERS, ALL VERIFIED AGAINST SOURCE**
(detail + line numbers in the judgment, section 7b):

1. **B1 -- `max_render_frames` is NOT WIRED in the canonical workflow.** Node 87
   has no input descriptor for it; only an unbound trailing `0` in
   `widgets_values`. The entire profile-ceiling channel Option B rests on is
   dead in `otr_canonical.json` today. Fix it IN that file, same commit.
2. **B2 -- ComfyUI serves a STALE plan across a frame-cap env change.**
   `OTR_ShotLock.IS_CHANGED` fingerprints only the two ROUTING env vars.
   `route_freeze.py:46-48` already warns about exactly this.
3. **B3 -- both plan boundaries SWALLOW what 7b intends to make terminal**
   (`otr_shot_lock.py:1150-1155`, `render_driver.py:3430-3438`). Chunk 1a's
   lesson, with the two catches already located.
4. **B4 -- `frame_count` is `round(duration*fps)` for 13 of 31 engines**, so the
   output proof is decorative for provider lanes. `ffprobe_counted_frames`
   already exists in `wan_shared.py`; this is wiring, not new capability.

**ORDER: B1 -> B4 (+ the `if got` fail-open, same predicate) -> the resolver
(with B3 fixed in the same change) -> the stamp (+ B2) -> the single-segment
proof -> the boundary comparison.** The single-segment proof must come AFTER
the resolver: on an 8GB box with `OTR_WAN_TI2V_MAX_FRAMES=49` it would refuse a
177-frame beat with no remedy available until the plan can say 49.

**TWO DRIVER CLAIMS WERE REFUTED BY THE PANEL AND BOTH REFUTATIONS VERIFIED.**
Live VRAM does NOT silently shorten a render (S4 killed that 2026-07-10;
`compute_real_frame_budget` RAISES `MotionBudgetError`), and the single path
ALREADY asks for `plan.segments[0].render_frames` via `segment_render_frames`,
so the trim_tail coupling the r3 plan was built around does not exist.

**PROCESS -- CHECK THIS EVERY ARC.** The r2 codex seat silently ran `gpt-5.5`:
kibitz's `CODEX_MODEL_PREFERENCE` tuple was stale against a catalog that already
had `gpt-5.6-sol`. Root-caused in `kibitz/scripts/kibitz.py` and pinned via
`KIBITZ_CODEX_MODEL`; r3 confirms `gpt-5.6-sol`, and the r3 seat found things
the r2 seat did not. **`kibitz/` is UNTRACKED here, so that fix is in no commit
and will not survive a fresh clone -- it belongs upstream in the skill.**

---

## SUPERSEDED -- MULTI-CLIP COVERAGE: 1-6 + **7a** DONE, **7b NEXT**

**Updated 2026-07-26 (remote Cowork), HEAD `42db9af9`.** Chunks 1a/1b/1c/2/3/
3b/4/**5/6a/6b/6c/6d/7a** are LANDED, GREEN and PUSHED, plus NINE adversarial QA
rounds. Suite 6454 -> **6891 passed / 27 skipped / 1 xfailed**; canonical
byte-identical `5377914B` across every commit.

**7a IS DONE AND IT CHANGED THE PLAN'S SHAPE. READ THIS BEFORE 7b.**

The plan below said chunk 7 begins by opting `ltx_8gb` in. **There is no longer
anything to opt in to.** The operator's ruling (2026-07-26, verbatim): *"this
architecture should work with all video and still models. There's no gate with
opt in or opt out. If there is, we need to remove that. Everything gets an
equal term... I don't like any hidden opt-ins. It either works or it fails."*

So `supports_multi_clip` is DELETED from `FrameContract`, from `join_mode_for`
and from `validate_coverage_plan`. All 31 registered engines carry a static
`FrameContract` (`docs/ENGINE_MATRIX.md`, generated, with a `--check` drift
gate in the suite). Multi-clip is universal; the only thing still EARNED per
engine is the CHAIN, via `continuity=strict_first_frame`.

**Landed in `e90dedf1` + `42db9af9`:**

| | |
|---|---|
| all 31 contracts | min/max/quantum/discrete_frames/native_fps/allow_tail_trim/continuity |
| `discrete_durations` -> `discrete_frames` | the field is FRAMES; the old name invited a seconds substitution no validator can catch |
| `+ native_fps` | so the rate those frames are counted at is stated, not implied |
| `supports_multi_clip(engine)` -> `can_split(engine)` | derived arithmetic ("has a ceiling"), not a stored opinion |
| `docs/ENGINE_MATRIX.md` | aspect, resolution contract, seconds per clip, prompt contract, still requirements, continuity, provider-side |
| `tests/test_engine_contract_roster.py` | a registered engine with no contract fails BY NAME |
| `tests/test_multiclip_goes_live.py` | the multi-segment path driven with REAL engines, no stubs |

**Two QA panels found six real defects in already-green, mutation-proven code.**
Four in the declarations (no multi-clip escape for over-cap beats; Veo declared
at the provider's 24 fps when clips are counted at the canvas's 25;
`humo_14B_169` inheriting a 177 ceiling when its real cap is 49; cloud lanes
claiming `quantum=1` when only whole seconds are reachable). Two when multi-clip
went live (`jump_segment_still_path` demanded a still for every segment >= 1,
killing all four chain engines and every HuMo beat past its cap; audio-driven
lanes would have shipped garbled lip-sync because nothing slices audio per
segment). **The stubs passed all six.** Keep the panel before every push.

**CHUNK 7 IS NOW 7b / 7c / 7d:**

1. **7b -- the env-vs-contract refusal.** Three ceilings are still
   env-overridable at RENDER time while the contract asserts a literal:
   `OTR_LTX_MAX_FRAMES`, `OTR_LTX_8GB_MAX_FRAMES`, `OTR_LTX_AV_MAX_FRAMES`.
   A contract that moves with the environment is a partition the image phase
   could not have planned for. The engine must REFUSE when they disagree, not
   quietly re-plan. (The declarations are already literals and a test pins that
   they do not read the `*_DEFAULT` constants -- what is missing is the runtime
   refusal.)
2. **7c -- rip the fallbacks.** `extend_frames_to_target` ping-pong
   (`eng_wan_ti2v.py:521-533`, `eng_ltx_8gb.py:426-437`), composite loop-fill
   (`otr_silent_composite._should_loop_fill`), held-last-frame. **Add to the
   list, found by 7a's audit:** the provider-side clamps --
   `_CloudVideoBase._duration_seconds` ends `max(min_s, min(max_s, secs))`,
   `word_razzle` does `8 if secs > 5 else 5`, and Veo's `_duration_s` discards
   the requested length outright at 1080p/4k. Same defect as the ping-pong,
   provider side. **Also 7c:** `trim_tail` is computed on single-segment plans
   and never applied, because `render_beat_coverage` early-returns to the
   historical path. That is PRE-EXISTING drift the composite absorbs (wan_i2v
   already quantized 50 to 53 and shipped 53) -- wiring the trim and removing
   the absorption belong together.
3. **7c also owes:** the adapter-side half of chunk 5 (r4's shape) -- each
   segment graph takes the prepared handles as LITERALS and omits its loader
   nodes; and `ltx_video`'s boomerang loop, which returns `2N-1` frames by
   default and is unchecked against its 169 ceiling.
4. **7d -- the live slice.** A 169-frame beat (`161 + (9-1)`; 169 mod 8 == 1 is
   why that number) -- >= 2 forward-only clips, ONE heavy load, no ping-pong --
   plus a 162-frame CPU tail-trim case. Acceptance is `RESULT SUCCESS` +
   `obs_publish OK` + the asset on disk, confirmed with Test-Path. **Needs a
   selective box reset per CLAUDE.md section 4: kill by CommandLine via CIM,
   never a blanket python kill -- that severs the MCP pythons and, in a remote
   window, the bridge you are watching through.**

**NOTHING HAS STILL RENDERED THROUGH IT.** 7a drove multi-clip end to end
through the planner and the request builder with real engines, and that is what
caught the two live-path defects. But no GPU leg has run. 7d is where it first
does.

**THE SEAMS CHUNK 7 BUILDS ON (landed, do not re-invent):**

| seam | where | contract |
|---|---|---|
| `BeatSession` | `beat_session.py` | ONE prepare/load per beat, ONE teardown in the outer `finally`; refuses a multi-segment beat whose adapter cannot name its handles |
| `SegmentSlot` | `beat_session.py` | session + index + the beat the caller claims; segments must be CONTIGUOUS and forward |
| `_render_one(..., segment=slot)` | `render_driver.py` | reuses the session's handles, does NOT tear down |
| `build_request_from_shot(..., segment_index=N)` | `render_driver.py` | N>0 swaps in that segment's own still AND its own length; a FODDER lane keeps its fodder |
| `segment_render_frames` | `render_driver.py` | the segment's `render_frames` off the stamped plan -- 6c does NOT need to adjust `target_frame_count` itself |
| `jump_segment_still_path` | `render_driver.py` | resolves it BY OBJECT ID off the spine receipt -- **never `_still_index`** |
| `ffprobe_counted_frames` | `wan_shared.py` | the decoded count, for the assembly boundary only (it decodes) |
| `render_beat_coverage` | `render_driver.py` | THE LOOP. `run_episode` calls it for every beat; one session, per-segment requests, terminal transaction inside, assembly after. A no-plan or single-clip beat takes the historical path |
| `extract_terminal_frame` | `wrapper_bridge.py` | a clip's LAST frame, decode-all not tail-seek; proves the file landed |
| `assemble_beat_segments` | `wan_shared.py` | concat + PROVE (one shape, exact decoded count, silent-clip contract); deletes the output if any check fails |

| chunk | commit | what landed |
|---|---|---|
| 1a | `933a78ba` | `_otr_shared/route_freeze.py` = THE route authority; four mirrors collapsed; malformed force map TERMINAL everywhere |
| 1b | `9006b76d` | the freeze at node 87 + forwarding + ShotLock consumption + `IS_CHANGED`; **the DECAPITATION fix** |
| 1c | `49944fb1` | render-time equality: verify, never repair; legacy branch NAMED |
| 2 | `ffc14693` | `frame_contract.py` + the roster audit (swallowed-import blindspot) |
| 3 | `bfacec2b` | `coverage_plan.py` -- the exact-sum partitioner (pure core) |
| QA1 | `6dc39f1f` | 3 partitioner math defects + 2 swallowed fail-closed sites + the unproven `talking` half |
| QA2 | `0bc863f4` | 2 MORE swallowed fail-closed sites + the dormant 3D picked-vs-effective trap |
| 3b | `00339e32` | the `CoveragePlan` rides the ledger, validated at BOTH wire boundaries |
| 4 | `583b3ea3` | the jump-still image-phase consumer: ShotLock mints per-segment requests, the dispatcher merges them into `objects` + `required_scene_targets`, the spine proves every one |
| QA3 | `4faabe0e` | ONE predicate decides whether a lane owes segment stills; the minter validates its plan; the `OTR_TEST_MODE` bypass can no longer wave a jump shot through |
| QA4 | `b0e383f5` | the LEGACY route path validates the coverage plan AFTER the route is final (it was checking against the picked engine, then letting the force map swap it); a `still_*` lane can never opt in to multi-clip |
| 5 | `4fa992e6` | **the beat session**: one load per beat, one teardown in the outer `finally`, a named identity (engine + recipe + weights) re-proved before every segment |
| QA5 | `451309de` | the GPU lease releases even when an engine's `unload()` raises (**live pre-existing leak** -- a stranded lease hung the NEXT episode for 120s); segments must be contiguous; a session with no `beat_id` latches the first beat claimed |
| 6a | `3a76c47a` | `ffprobe_clip_fields` learns `width`/`height`; new `ffprobe_counted_frames` (`-count_frames`) for the assembly boundary |
| 6b | `a888c423` | a jump segment resolves its init image **BY OBJECT ID** off the spine receipt -- the chunk-4 carry-forward, with a differential test showing `_still_index` returning the wrong image |
| 6c-1 | `4d5795b1` | `extract_terminal_frame` -- what a CHAIN successor begins on, proven to be the LAST frame and proven to exist; `otr_engine_tmp_path` generalises the in-tree allocator |
| 6c/6d | `5845e635` | `render_beat_coverage`: ONE session per beat, per-segment requests, the terminal transaction INSIDE the loop, then a transactional assembly that proves the beat is the length the plan promised. Also corrected `segment_render_frames`, which short-circuited index 0 to the BEAT's length -- segment 0 of a two-segment beat would have rendered the whole beat |
| QA7 | `a05b5ac6` | Sonnet + agy over 6c/6d: **the chain terminal frame was written to a top-level `request["init_image"]` that NO production code reads** (every adapter reads `asset_refs["init_image"]`), so a chained successor would have silently rendered from its original still -- and the test stub agreed with the bug because the test's own builder used the same wrong key; the concat moved INSIDE the transaction; a short segment is now named at the segment; the beat reports its PEAK VRAM not its last segment's; `max(1, keep)` became a refusal; the assembly checks fps and pixel format, not just canvas |
| QA6 | `a818b5d1` | Sonnet + agy over QA4/6a/6b: a segment request now carries its own **LENGTH** off the plan (it had the right picture and the whole beat's duration), a **fodder lane keeps its fodder** (the segment still is its background plate, and clobbering it is the clay blob through a second door), a pathless duplicate receipt entry no longer hides the proven one, a bad `segment_index` fails closed NAMED, and the still-lane guardrail no longer skips an unbuildable engine in silence |

**CHUNK 4'S DURABLE LESSON -- TWO POLICIES OVER ONE STATE IS THE DEFECT, NOT
THE SYMPTOM.** The QA panel found the merge inferring "no scene object and no
required target means this lane consumes no still" and skipping, while the
spine demanded every STAMPED request back regardless. The inference did not
avoid the failure -- it moved it to the render boundary and made the message a
lie. The fix was neither side: it was to ask the question ONCE, at the mint,
using `render_driver._still_spine_requires_scene` -- the spine's own predicate
-- so the disagreement is unconstructible rather than merely caught. **When two
places must agree about one fact, make them the same call, not two calls that
happen to match today.**

Also settled and worth not relitigating: segment stills deliberately do NOT
wear a `scene_*` kind (both beat-keyed lookups take the LAST matching scene
row, so a scene-kinded segment still would shadow the beat's own image and
segment 0 would render from the LAST segment's still); and a cloned bookend
segment deliberately drops off the fixed 4242 seed onto the request-hash seed,
which stays reproducible -- what it loses is the shared canonical LOOK across
its own segments, which is what cutting means.

**THE QA ROUNDS ARE THE STORY OF THIS SESSION AND THE LESSON IS PORTABLE.**
A six-lens Sonnet fan-out plus an agy pass found NINE defects in code that was
already green, already reviewed by a 4-round kibitz arc, and already pushed:
- **FOUR swallowed fail-closed sites.** Chunk 1a made a malformed
  `OTR_FORCE_ENGINE_MAP` terminal; four pre-existing broad `except Exception`
  blocks silently absorbed it, each one individually defeating the entire
  chunk. **When you make something newly terminal, grep every caller for a
  broad catch in the SAME change** -- the suite will not tell you.
- **THREE partitioner math defects**, all found by brute-force differential
  testing against an independent reference, none by reading: a tail-trim search
  capped at one quantum (832 coverable beats refused), an unmemoized recursion
  that HUNG instead of refusing, and a `join_mode_for` that claimed SINGLE for
  targets no single render can cover. **A pure algorithm deserves a
  differential sweep, not a code review.** The standing sweep now lives in
  `tests/test_multiclip_coverage_plan.py`.
- **Mutation testing found an unproven fix.** Reverting `talking` to the picked
  engine left the WHOLE suite green -- the decapitation fix's twin had shipped
  with zero coverage. **A green suite is not proof a fix is proven.**
- **Two "exhaustive" sweep tests were theatre**: 112 of 128 targets asserted
  nothing, and corrupting the chain arithmetic left both passing.

**THE DURABLE ARCHITECTURAL LESSON:** node ids are NOT execution order. There
is no `89 -> 90` edge in `otr_canonical.json` -- MetaBrief (89) and ShotLock
(90) are INDEPENDENT branches reconverging only at 91, so a freeze at ShotLock
can never inform the image phase. Node 87 is the unique common ancestor.
**Verify a claimed node ORDER against the link list, never the ids.**

## SUPERSEDED -- the chunk 1-3 detail (kept for the arc record)

**Updated 2026-07-25 (afternoon), HEAD `bfacec2b`.** r4 CONVERGED (both seats
yes-with-fixes, `48e02241`), a six-way grounded Sonnet fan-out ran before code
by operator direction, and **SIX green chunks shipped and pushed**. Suite
6454 -> **6769 passed / 27 skipped / 1 xfailed**; Bible 17; canonical
byte-identical `5377914B` across every one (no node/widget/input/link change
anywhere in chunks 1-3).

| chunk | commit | what landed |
|---|---|---|
| 1a | `933a78ba` | `nodes/_otr_shared/route_freeze.py` = the ONE route authority; FOUR mirrors collapsed onto it; malformed force map now TERMINAL at every reader |
| 1b | `9006b76d` | the freeze at node 87 + ImageDirector forwarding + ShotLock consumption + `IS_CHANGED`; **the DECAPITATION fix** |
| 1c | `49944fb1` | render-time equality: verify, never repair; legacy branch NAMED and logged |
| 2 | `ffc14693` | `frame_contract.py` (`FrameContract` + continuity) + the roster audit for the swallowed-import blindspot |
| 3 | `bfacec2b` | `coverage_plan.py` -- the exact-sum partitioner (pure core) |

**THE r3/r4 PLAN WAS WRONG ABOUT WHERE THE FREEZE GOES, and the correction is
the durable lesson:** node ids are NOT execution order. There is no `89 -> 90`
edge in `otr_canonical.json` -- MetaBrief (89) and ShotLock (90) are
INDEPENDENT branches that reconverge only at 91, so a freeze at ShotLock can
never inform the image phase. Node 87 (VideoDirector) is the unique common
ancestor. **Verify a claimed node ORDER against the link list, never the ids.**

**THE DECAPITATION BUG IS FIXED (1b) and it was LIVE under the DEFAULT
environment** -- not latent. A portrait HuMo picked for `announcer_visual` with
`OTR_ENABLE_HUMO_HOSTS` unset redirects to the WIDE `ltx_audio_in`, but
`aspects` was derived from the PICKED portrait engine, so a portrait still was
minted and the wide render centre-cropped it. `eng_ltx_av.py:345-347` had
recorded that exact outcome verbatim. Pinned by
`test_redirected_bookend_gets_a_WIDE_still_not_a_decapitated_portrait`.

**NEXT -- chunk 5, then 6-7 in the r3 order:**

1. ~~**3b**~~ -- DONE @ `00339e32`.
2. ~~**4**~~ -- DONE @ `583b3ea3` + QA `4faabe0e`. Still behaviour-inert by
   construction (every adapter is `SINGLE_ONLY`, so nothing jump cuts yet) and
   pinned as such.
   **CARRY-FORWARD INTO CHUNK 6, HARD (found by the chunk-4 QA panel, judged
   out of chunk-4 scope because no per-segment render loop exists yet): the
   per-segment request builder MUST resolve `init_image` BY OBJECT ID off
   `shot["jump_still_requests"]`, never through `render_driver._still_index`.**
   `_still_index` filters to `kind.startswith("scene_")`, and segment stills
   deliberately do not wear a scene kind, so a per-segment loop that reuses the
   existing lookup would hand EVERY segment segment-0's still -- silently
   re-creating the held-frame degradation chunks 3-4 exist to remove, with the
   correct stills sitting unused on disk.
3. ~~**5**~~ -- DONE @ `4fa992e6` + QA `451309de`. The driver-side half of the
   r4 shape is landed (session owns the handles; `run_graph` did NOT gain a
   prepared-handles parameter). **The ADAPTER-side half is still owed and
   belongs to chunk 7**: "each segment graph takes the handles as literals,
   omitting its loader nodes" is a per-adapter change, and `ltx_8gb` is the
   first adapter that will need it.
4. **6:** ~~6a ffprobe~~ DONE @ `3a76c47a`; ~~6b per-segment init image~~ DONE
   @ `a888c423`. **6c (NEXT)** = the per-segment render loop + the terminal
   transaction INSIDE it (segment N+1 needs segment N's terminal frame
   synchronously, so it cannot wait for the post-episode pass). **6d** =
   transactional assembly, verified with the 6a helpers.
5. **7:** the FIRST adapter opt-in + the `ltx_8gb` LIVE slice at a 169-frame
   beat + a 162-frame CPU tail-trim case. **Needs a selective box reset per
   CLAUDE.md section 4** -- kill by CommandLine via CIM, never a blanket python
   kill. Three things chunk 7 must carry, all grounded this session:
   (a) `eng_ltx_8gb.frame_contract()` must declare a STATIC ladder -- the
   contract is pure by contract, and `_resolve_render_config()["max_frames"]`
   reads the environment, so the declaration cannot be derived from it;
   (b) the adapter must declare `session_identity()` (engine + recipe + weight
   names) or `BeatSession` will REFUSE to reuse handles across its segments --
   that refusal is the design, not a bug;
   (c) **the CLIP-FILL ping-pong at `eng_ltx_8gb.py:426-437` MUST NOT run for a
   coverage-planned segment.** The plan already sized the render to a legal
   length; extending it re-introduces exactly the boomerang this build removes.
   The same applies to `eng_wan_ti2v.py:521-533` when Wan opts in.
6. **8 (later):** the pause map (RANKS legal cut points, never chooses them);
   then further adapters; audio lanes LAST.

**A REAL ARITHMETIC LIMIT found building chunk 3, carry it forward:** chaining
`8n+1` segments always assembles to `8m+1` visible frames, so a beat whose
target is not congruent to 1 mod 8 has NO exact cover on that ladder and the
partitioner REFUSES rather than drift. Those beats need `allow_tail_trim` --
which is why it lives in the adapter's declaration, not in the assembler. The
169-frame acceptance case works precisely because 169 mod 8 == 1.

## SUPERSEDED -- the r4 gate (now closed; kept for the arc record)

**Operator requirement of record (2026-07-25):** *"we need as much video to
capture the beat... we need enough clips per the beat for MOVING video."*
Chain (last frame -> next clip's first) PREFERRED; jump cut acceptable; reuse
only if loop-closed; `still_*` lanes are one still; **audio lanes cut at
phrase boundaries, never arbitrary.** Per-adapter: its own PROMPTS + frame
numbers. Shared: ONE splitter, ONE assembler (operator's own division).

**Arc status: r1, r2, r3 JUDGED and pushed. r4 convergence is OWED before any
code on this block.** Judgments of record:
`docs/2026-07-25-multiclip-coverage-r1-judgment.md`, `-r2-judgment.md`,
`-r3-judgment.md`. Runs under `kibitz-runs/2026-07-25-multiclip-coverage*/`.

**LANDED AND GREEN this session (both pushed, HEAD == origin):**
- `57f4983a` **route lock** -- `resolve_final_shot_engines` applies the force
  map AND the radio-host redirect in ONE idempotent pass BEFORE
  `validate_and_repair_still_spine`; malformed `OTR_FORCE_ENGINE_MAP` now
  FAILS CLOSED (was: log `IGNORED (parse)` and render the unforced plan).
- `a1d810f1` **lip-sync no-mirror** -- `fit_frames_to_target(...,
  allow_mirror=False)` + `MirrorExtensionForbidden`; HuMo (`audio_driven_face`)
  can no longer mirror a short capped render. Trimming stays legal. Operator:
  *"no render backwards, that doesn't work."*

**THE BUILD ORDER (r3-judged, 8 chunks). Chunk 1 is the biggest single win:**
1. **Hoist the route freeze into `OTRShotLock.lock`** (after policy
   validation, before `build_execution_plan`, `otr_shot_lock.py:1091-1142`) +
   `IS_CHANGED` over every captured env var + render-time
   `resolve_final_shot_engines` becomes an EQUALITY ASSERTION. Retires the
   MetaBrief/dispatcher effective-engine MIRRORS. Independently shippable.
2. Declaration surface (`FrameContract` = min/max/quantum/discrete/
   allow_tail_trim + continuity token on the `VideoEngine` Protocol,
   `registry.py:51-98`) + roster audit at the BOTTOM of
   `_otr_video_engines/__init__.py` after all guarded imports. All adapters
   `single_only`.
3. Partitioner + `CoveragePlan`, durably stamped, validated at BOTH boundaries
   (do NOT make legacy `ShotRow` authoritative -- judged).
4. Jump-still image-phase consumer (ShotLock patches requests -> dispatcher
   merges into `objects` + `required_scene_targets` -> spine validates every
   jump segment). **Without this a jump cut has no still at all.**
5. Beat-session lifecycle: reusable MODEL/CLIP/VAE handles, teardown in one
   outer `finally`, assert LOADER-call count (not `prepare` count).
6. Terminal transaction INSIDE the render loop + transactional assembly + a
   new ffprobe helper with `-count_frames`.
7. **`ltx_8gb` live slice at a 169-frame beat** (`161 + (9-1)`): >= 2
   forward-only clips, one heavy load, no ping-pong, `RESULT SUCCESS` +
   `obs_publish OK` + asset on disk. Plus a 162-frame CPU tail-trim case.
8. Later: the pause map (RANKS legal cut points, never chooses them); then
   further adapters; audio lanes last.

**Named test files (r3-judged):** `tests/test_multiclip_coverage_plan.py`,
`tests/test_ltx_8gb_multiclip.py`,
`tests/test_multiclip_transactional_assembly.py`; extend
`tests/test_workflow_json_wiring_invariants.py` and
`tests/test_capability_profiles.py:384`. **KEEP `tests/test_clip_fill.py`** --
the mirror helper stays legal for `still_*`/decorative lanes; add a pin that
the `ltx_8gb` coverage path never calls it.

**Canonical JSON:** no chunk in 1-7 should touch it. Confirm at r4.

### The still-plans block -- SUPERSEDED, not deleted

The 31-plan table cut (both R1 seats) still stands as analysis, and S0a /
S0a-b / S1 / S1b remain landed and green. But the coverage block now owns the
same seams (effective engine, still requiredness, per-engine prompts), so the
still-plans chunk order (`S0b-core -> S2 -> S3 -> S5 -> S4`) is PARKED and
must NOT be resumed as-is. Records:
`docs/2026-07-25-still-plans-r1-lean-judgment.md` +
`docs/2026-07-25-per-beat-stills-r1-judgment.md` (which carries every operator
ruling verbatim). A later window folds the surviving descriptor work into the
coverage block rather than running it standalone.

## SUPERSEDED -- STILL PLANS: the R1 CUT THE TABLE (history; see above)

**The R1 arc round the operator authorised on 2026-07-25 has been RUN and
JUDGED. Both seats independently said CUT the 31-plan table.** Judgment of
record: `docs/2026-07-25-still-plans-r1-lean-judgment.md`. Nothing was torn
down -- the tree is GREEN at `5dd74f93` with S1b landed.

**THE ANSWER (judge call: codex's Option C over agy's Option B).** Frozen
effective routing + a COMPACT per-adapter capability descriptor
(`still_mode = scene|mesh|none` plus narrow activation flags and aspect) + ONE
pure materializer + a SEPARATE per-engine layer-2 prompt hook. The seven-field
`StillPlanRow`, its closed enums and the 31 copied declarations are CUT.
`style_tail_policy` leaves the structural contract entirely -- tail selection
stays in the prompt composer. Option C beat Option B because agy's single
central `engine_requires_still()` recreates the very central-authority shape
this build exists to kill, and because the operator's own directive ("each
video path has its own customized still operations") requires per-adapter
ownership.

**THE OLD ORDER IS DEAD.** `S0b-core -> S2 -> S3 -> S5 -> S4` and the
seven-consumer atomic cutover are superseded. **New order:**

1. **OPERATOR RATIFIES THE CUT** -- it makes landed green code a teardown
   target, so it is a ratification, not a coder call. Also rule on the LTX
   per-beat recipe question below.
2. **ONE consolidated build spec** (Option C descriptor + materializer +
   prompt hook + explicit teardown list + every accepted r4/r4b/R1
   correction). Both r4 passes and both R1 seats asked for this. Mark the
   locked spec, the corrected plan and both judgments history-only.
3. **The ROUTING FREEZE, first and alone,** with the forced-route live proof.
   It is the ACTUAL bug fix (`otr_video_render_batch.py:322` validates the
   spine before `render_driver.py:2784` applies the override) and it ships
   independent of the table question.
4. Then descriptor + materializer, then the teardown, then the prompt hook.

**OPERATOR DECISION NEEDED -- the LTX per-beat recipe capability.** Freezing
`ltx_resolved` is NOT behaviour-preserving. `eng_ltx_av.py:402-405` documents
the current contract verbatim: "Read fresh every call (an operator flips
daily<->hero per beat by swapping `OTR_LTX_AV_UNET` / `OTR_LTX_AV_RECIPE`)."
The freeze would silently make the recipe episode-scoped. Either (a) accept
episode-scoped and DELETE the contrary docstring contract, or (b) keep per-beat
switching via an explicit SHOT-OWNED field instead of ambient env. Default if
unruled: (a) -- a frozen state whose recipe changes mid-episode is not frozen
-- but it removes an advertised capability, so it is the operator's call.

**Doctrine lesson from this arc, for the log:** the routing freeze was always
the bug fix and should have gone FIRST. The inherited order put the table's
characterization (S0a) and declaration (S1/S1b) ahead of it, so two chunks
landed against a structure the arc then cut. S1b still earned its keep -- see
the judgment's honest accounting -- but the ordering was wrong.

~~**S1b**~~ -- **DONE @ `69328cec`** (2026-07-25; suite 6444 passed / 27
skipped / 1 xfailed; Bible 17; AST/BOM/zero-byte/UTF-8/ASCII clean on 13 files;
canonical byte-identical `5377914B`; pushed, HEAD == origin). 57 rows across 12
adapters now carry the producer's real layer-2 GEOMETRY constants instead of
S1's paraphrases. Detail in `docs/HANDOFF_LOG.md`.

**CORRECTION to this file's own earlier instruction (do NOT re-derive the old
one).** S1b did NOT "restore every clause VERBATIM from the inventory". The
seed inventory records COMPOSED output strings, and the producer splits
GEOMETRY (Python-owned engine-safety framing) from LOOK (pack-owned:
`VisualStyle.portrait_look` / `portrait_look_talking` / `plate_look`) at chunk
A1 (`otr_meta_brief_image_prompt.py:96-104`). Restoring the composed strings
would have hard-coded the `sci_fi_radio` pack's look into all 31 engines,
against spec section 4 ("a plan may only contribute layer 2 ... it may never
decide style"). **The transplant source is the eight named `*_GEOMETRY` /
`STILL_FRAMING_*` constants, never the inventory's composed text.** Both agy
and this window's grounding reached that independently.

**S0b-core (corrected).** Land the routing freeze atomically. THREE
corrections to `docs/S0b_KIBITZ_NEEDED.md` before it is built:
  1. The closed `engine_facts` descriptor `{engine_id, family, provider_side}`
     (spec:230) has NO aspect field, but `resolve_row_aspect`
     (`still_plan_helpers.py:177-189`) needs `engine_render_aspect` /
     `render_aspect` and SILENTLY RETURNS PORTRAIT when absent -- so every
     `inherit_engine` row would resolve portrait, including `cloud_kling_avatar`
     and both wide `_169` HuMos. Add a canonical `render_aspect` field and
     reject missing values instead of falling back.
  2. The frozen-routing prepass as specified does NOT close the defect it is
     named for. `apply_engine_override` (`render_driver.py:2784`) applies only
     `OTR_FORCE_ENGINE_MAP`; the radio-host redirect is a SEPARATE mutation at
     `:1413-1513`. The prepass must freeze each role's FINAL effective engine,
     redirect included, before `validate_and_repair_still_spine`.
  3. The test-literal inventory is stale: ~35 `policy_version=2` sites, not 31
     (`test_hybrid_voice_fit` has none; `test_still_plan_parity` adds five).
     Derive the list mechanically.
  SCOPE (judge call on a panel split): adopt agy's S0b-core / S0c relief, but
  keep `ltx_resolved` FROZEN inside S0b-core -- that answers codex's objection
  that deferring it desynchronizes `when_engine_talking`. Only the
  `eng_ltx_av.assert_usable` mismatch ASSERTION defers to S0c.

**S2 (cutover).** OPERATOR EYEBALL RESOLVED 2026-07-25 -- and it is far
narrower than three docs claimed. There are FOUR HuMo engines; only `humo` and
`humo_1.7B` ship `render_aspect="portrait"`, and `humo_1.7B_169` /
`humo_14B_169` are ALREADY wide (the ComfyUI dropdown labels this to the
operator as "(portrait)" / "(16:9)" -- a visible product contract). Nothing
about HuMo "flips". The S2 delta is FOUR ROLE-CELLS: two portrait HuMo picks x
announcer/music, under the hosts-off DEFAULT, where `_enforce_radio_is_host`
redirects the beat to the WIDE `ltx_audio_in` that actually renders it. With
`OTR_ENABLE_HUMO_HOSTS=1` a portrait HuMo keeps its portrait still. Confirmed
three ways: operator, codex, and agy independently. The old "via the `_169`
siblings' render_aspect" framing in `docs/S2_EYEBALL_REQUEST.md` is WRONG on
mechanism and must be corrected along with the S0a fixture's special_cases
rows.

**S3** shim + stale-prose deletion. **S0c** the ltx_av mismatch gate.

**S5 (NEW, the operator's actual directive).** Operator 2026-07-25: "ensure
that each video path has its own customized still operations." It is NOT met
today. Driving the live registry over all 31 engines yields 14 shared plan
objects but only SIX distinct signatures and SIX distinct structures -- meaning
the framing prose adds ZERO per-engine differentiation, and 19 engines
(`wan_ti2v`, `google_*`, `still_*`, `word_razzle`, `cloud_*`, `ltx_8gb`) share
one identical signature whose portrait row is empty. S5 diverges the restored
clauses per engine so an i2v engine whose still IS the init frame, a t2v engine
whose still is optional, and a Ken Burns pan stop receiving identical
instructions. S5 CHANGES PROMPTS: it needs its own acceptance and must land
after the wiring, never inside a parity chunk.

**S4** two fresh-boot live legs (default route + forced HuMo bookend).

Gate: r4 convergence at CURRENT HEAD on the corrected plan before code. Both
r3 panelists explicitly rejected Path B (S2-first against live env); do not
revive it.

**THE LIVE ORDER IS THE 2026-07-27 OPERATOR DIRECTION ABOVE (this file, the
"OPERATOR DIRECTION 2026-07-27" block), NOT the 2024-07-24 list that used to
sit here.** The 07-24 ordering put LEAN-MEAN FRONT second; the operator moved
lean-mean to LAST on 07-27 because it is a deletion campaign whose value IS
its file-and-line kill inventory, so it must re-ground once against the FINAL
tree instead of ripping first and re-grounding after two blocks land on
freshly-rewired code. The stale list was left un-struck and a 2026-07-28
window read it and reported the wrong order to the operator. Corrected here so
the next window cannot repeat it.

Live order:

0. **LOCAL-ENGINE OBS WIRING** (`WIRE-W1..WIRE-W7`) -- THE CURRENT STEP. Arc
   r1-r4 CLOSED; spec `kibitz-runs/2026-07-28-local-engine-obs-wiring/r3/final.md`
   as amended by `r4/final.md`.
1. ~~**WAN 8-GB low-VRAM launch contract**~~ -- DONE @ `f914f0a4`
   (`PBUG-20260723-02`). The live 8-GB requalification leg is still owed and
   belongs to a render window.
2. **Randomizer A.**
3. **`dynamic_story`** -- WIRING ONLY; rev-5's design stays FINAL, do not
   rerun the design panels.
4. **SFX** -- carries the Timeline Cue Ledger C0/C1 gate + the R4.1 refit as a
   precondition.
5. **Re-observe the parked story bugs** -- after SFX, see whether they still
   occur at that HEAD (see OPEN BUGS).

**AND THAT IS THE END OF GO_FORWARD.** Operator direction 2026-07-29: LEAN-MEAN
FRONT and LEAN-MEAN TAIL are no longer items 5 and 6 of this list -- they are
**off this plan**, in the Lean-mean campaign section of `ROADMAP.md`, which is where a window goes
after the list above is exhausted. The packing table's CODER D and CODER G rows
are struck through and their gates voided to match.

Standing constraints, unchanged by the rescope: keep the RTX 5080 free for
ComfyUI; the 4060 Qwen endpoint is a read-only QA reviewer, not a production
ComfyUI slot; six-bank requalification (canonical `RESULT SUCCESS`,
`obs_publish OK`, exact episode/OBS assets, and the archival final's parent
equal to the ledger-owned episode root -- PBUG-20260720-05 acceptance) is
still owed whenever a render window next opens, and was NOT cut.

## OPERATOR CAMPAIGN QUEUE -- 2026-07-23 (PAUSED)

The overnight media qualification was aborted after the WAN lane and the LTX
visual-style sweep stalled at case 6/54. No new GPU run is authorized while
confirmed bugs are being closed. Failure inventory / staging record:
`docs/2026-07-23-video-failure-inventory.md`.

Bug-first order before resuming:

1. Requalify receipt truth against the captured six-bank stdout and confirm
   the old false PASS is now a terminal FAIL (fix LANDED @ `314dd481`;
   needs live confirmation only).
2. Make the image phase own every required scene-still, mesh-fodder, and
   opening-still target, with a complete target/path receipt before video
   dispatch; no text-only or dark-floor degradation for a missing required
   still. (`f150213f` fixed the no-still visualizer spine handoff; the
   scene/mesh-consuming ownership contract is the remaining piece.)
3. Make the WAN 8-GB profile carry its actual 832x480/17-frame low-VRAM
   launch contract instead of falling back to the 177-frame default.
4. Then provider-capacity and SciFi News markup-repair residuals.

Remaining media qualification (CUT DOWN by the operator rescope 2026-07-24 --
the 45-word model-coverage matrix and the 54-case visual-style sweep are
DELETED, not deferred; reviving either is a new operator decision):

1. Six 120-word canonical runs in bank order `media_archive`, `original`,
   `public_domain`, `shakespeare`, `scifi_news`, `scifi_news_pro`:
   `google/gemma-4-12b-it` both writer slots, `viz_mxc_cpu` /
   `viz_mxc_mandala` / `viz_camera` video slots, `z_image_turbo` all three
   image slots. (4/5 of the 120w receipts are already banked from
   `tmp/six_bank_sweep_20260723_205002_331`; `scifi_news` is the open FAIL.)
   This is the ONLY surviving matrix.

The coordinator keeps one canonical API prompt active at a time, reloads
`workflows/otr_canonical.json` for every case, and records each prompt and
receipt under `tmp/`.

## MEASURED -- the 8 GiB-clamped video bench (2026-07-31, `8bd82efb`)

Nine cells ran live: arms A (`wan_ti2v` 832x480, 30 steps, recipe v1),
B-partial (same graph, fp8 safetensors text encoder) and D (`ltx_8gb` 512x288,
8 steps, `LTX8_RECIPE_V2`), each at 17 / 49 / 81 frames, seed 42,
`--reserve-vram 8`, selective box reset between every leg. Receipts, per-cell
server logs and all nine assets live at
`C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\_bench_4arm\`
(`video_arm_bench_results.json` is the machine record;
`video_arm_bench_table.md` is the printed table).

**READ THE NUMBERS THE RIGHT WAY.** The table prints ABSOLUTE machine-wide NVML
peaks; the grading bar is on the DELTA. Every cell measures its own desktop
baseline (2181.6 - 2221.9 MiB across the nine) and records
`peak_delta_mib = peak - that cell's own baseline`. Subtracting one shared
baseline from the printed absolutes is close but wrong -- use the
`peak_delta_mib` field. The bar is
`GREENLIGHT_PEAK_DELTA_MIB = 8192 - DISPLAY_ALLOWANCE_MIB = 7168 MiB`, i.e.
8 GiB minus a 1 GiB display allowance.

1. **`wan_ti2v` at 832x480 FITS the clamp.** Arm A peak_delta
   6568.2 / 6563.1 / 6563.1 MiB at 17 / 49 / 81 frames -- PASS on all three,
   roughly 600 MiB under the 7168 bar. This discharges the "no live
   8-GB-clamped WAN render receipt" obligation, and ONLY that: it is a
   PREQUALIFICATION on a 16 GB card told to reserve 8 GiB, not a render on a
   physical 8 GB card. The harness says so itself
   (`run_video_arm_bakeoff.py:1247`): results are never worded as
   "8 GB qualified".

2. **Frame count is very nearly free, and the estimator says otherwise.** A's
   delta across 17 -> 81 frames moves -5.1 MiB. Negative, i.e. inside
   run-to-run noise, so the measured marginal cost of a frame is
   indistinguishable from zero. `FRAME_COST_MODEL["wan_ti2v"] = (7000.0, 185.0)`
   at `_FRAME_COST_REF_PIXELS = 1472*832` scales to 60.3 MB/frame at 832x480 --
   a predicted +3,861 MB over that same span. The OVERHEAD term is roughly right
   (7000 MB predicted against a ~6565 MiB measured intercept); the PER-FRAME
   term is the entire error. D is not zero but is also tiny: +128.0 MiB over 64
   frames, about 2.0 MiB/frame. **Do not re-fit from these nine points** -- see
   the open items below.

3. **Arm B is dead: the fp8 text encoder costs MORE, not less.** B-partial
   peak_delta 7907.1 / 7811.1 / 7715.3 MiB -- FAIL on all three, over the bar by
   547 - 739 MiB, and about 1.25 GiB WORSE than arm A on the same graph, canvas
   and step count. The hypothesis that a scaled-fp8 encoder buys headroom is
   REFUTED by measurement on this box. Do not re-run it hoping for a different
   answer; if it is ever revisited, the receipt above is the number to beat.

4. **`ltx_8gb` recipe v2 rendered clamped AT ITS PRODUCTION CANVAS.** Arm D
   peak_delta 6691.1 / 6755.3 / 6819.1 MiB, PASS on all three, 15.4 - 20.4 s
   wall. 512x288 is the shipped `render_canvas`, and it had never rendered live
   before this bench.

5. **THE LIKE-FOR-LIKE LEG SETTLES IT: D's speed is the recipe, not the
   engine.** Arm A re-run at D's canvas, everything else unchanged (30 steps,
   seed 42, `--reserve-vram 8`), three more PASS cells, assets decode-validated:
   `output/otr/episodes/_bench_4arm/diagnostic_512x288/`.

   | frames | A @ 832x480 | A @ 512x288 | D @ 512x288 (8 steps) |
   |---:|---|---|---|
   | 17 | 6568.2 MiB / 76.2 s / 1.15 s/it | 6524.6 MiB / 40.4 s / 0.474 s/it | 6691.1 MiB / 20.4 s / 1.74 s/it |
   | 49 | 6563.1 MiB / 145.7 s / 3.02 s/it | 6578.5 MiB / 60.3 s / 1.07 s/it | 6755.3 MiB / 15.4 s / 1.19 s/it |
   | 81 | 6563.1 MiB / 221.5 s / 5.06 s/it | 6486.9 MiB / 81.1 s / 1.42 s/it | 6819.1 MiB / 20.4 s / 1.20 s/it |

   At the SAME canvas, Wan is FASTER per iteration than LTX at 17 and 49 frames
   (0.474 vs 1.74; 1.07 vs 1.19) and 18% slower at 81 (1.42 vs 1.20). The
   wall-clock ratio at 81 frames is 81.1 / 20.4 = 3.98x against a step ratio of
   30 / 8 = 3.75x. **The entire "10x" in the headline table was 3.75x steps
   times 2.71x pixels.** Nothing about `ltx_8gb` is intrinsically fast, and
   nothing about `wan_ti2v` is intrinsically slow.

   It also settles the cheaper engine at a fixed canvas: at 512x288 **Wan uses
   LESS VRAM than LTX** -- 166 / 177 / 332 MiB less at 17 / 49 / 81.

6. **PIXELS ARE AS FREE AS FRAMES, so the estimator is wrong in BOTH scaling
   terms.** A's delta at 512x288 minus its delta at 832x480 is
   -43.6 / +15.4 / -76.2 MiB -- it straddles zero, so 2.71x fewer pixels bought
   nothing. The estimator scales `per_frame` by the pixel ratio (60.33 MB/frame
   at 832x480, 22.27 at 512x288), which at 81 frames predicts the smaller canvas
   should need **3,083 MB less**. Across this whole range the measured cost is
   the resident model and essentially nothing else.

**All twelve assets were validated by DECODE, not by container header.**
`ffprobe -count_frames` reports exactly 17 / 49 / 81 read frames per arm, at
832x480 (A, B-partial), 512x288 (D) and 512x288 (the A diagnostic leg), 24/1 fps.
Nothing here is an empty or truncated file.

### 7. ARM C EXISTS NOW, AND IT IS THE RESULT WORTH READING (2026-08-01)

A FOUR-arm campaign, all twelve cells in ONE run against ONE campaign baseline
and ONE manifest, at `1a49fdb0`. Every `s/it` below is log-parsed
(`s_per_it_source = "log"` on all twelve), so the seconds columns are
comparable within the caveat of item 5.

| arm | canvas | steps | 17f delta | 49f delta | 81f delta | **worst** | 81f wall | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A -- Wan Q5_K_M incumbent | 832x480 | 30 | 6563.1 | 6531.1 | 6563.1 | **6563.1** | 171.2 s | PASS |
| B-partial -- safetensors encoder | 832x480 | 30 | 7715.1 | 7715.1 | 7907.1 | **7907.1** | 156.1 s | **FAIL** |
| C -- FastWan DMD LoRA | 832x480 | **3** | 6563.1 | 6531.1 | 6563.1 | **6563.1** | **62.1 s** | PASS |
| D -- ltx_8gb v2 | 512x288 | 8 | 6467.1 | 6723.0 | 6819.1 | **6819.1** | 13.9 s | PASS |

**A and C are VRAM-IDENTICAL at every rung -- 6563.1 / 6531.1 / 6563.1, to the
decimal.** Same canvas, same base GGUF, same encoder, same VAE. The only
difference is the FastWan distillation LoRA and its 3-step restart recipe. So
the LoRA's peak-VRAM cost is exactly zero, and C is **2.76x faster end-to-end at
81 frames** (62.1 s vs 171.2 s), 2.40x at 49, 1.73x at 17.

Per-step, C is 1.16 / 1.58 / 2.37 s/it against A's 1.02 / 2.29 / 3.85. At 81
frames **C is faster per step as well as running a tenth as many**, because
cfg 1.0 runs ONE forward per step where A's cfg 5.0 runs two. The LoRA's real
compute cost shows only at 17 frames (1.16 vs 1.02, ~14%), where the patched
matmuls are not yet amortised over enough sequence.

The LoRA is NOT free -- its cost is in HOST RAM, not VRAM: C's sysram delta is
14.7-15.2 GB against A's 10.5-11.1 GB, roughly +4 GB for the patch data and the
dequant staging. Well inside the 24576 MiB ceiling, and worth knowing before
anyone runs this beside a resident writer LLM.

**B-partial got WORSE with a complete ladder and is decisively out.** It failed
every rung, and its 81-frame cell is 7907.1 MiB -- 739 MiB over the bar, and 192
MiB above its own 17/49 figure. Swapping the GGUF encoder for
`umt5_xxl_fp8_e4m3fn_scaled.safetensors` costs ~1.2-1.3 GB of peak. **The GGUF
encoder is load-bearing for this tier**, which is the one thing the encoder-only
arm existed to isolate.

**What this does NOT say.** Arm C's advantage is measured in VRAM and seconds
ONLY. A 3-step distilled render versus a 30-step render is a QUALITY question
this bench deliberately does not answer -- the automated visual discriminator
was cut unanimously, and schedule correctness is telemetry, not a picture. Arm
C's recipe fidelity IS proven (the sampler logs
`transition=restart(predict_x0->renoise_fresh) timesteps=1000,757,522,0` every
run); whether the output looks good enough to ship is an operator eyeball.

**Never cite the seconds column as an engine ranking.** Item 5 is why: two arms
at different canvases and different step counts are two different questions, and
the only honest engine comparison in this tree is A vs D at 512x288.

**Still open, and NOT closed by this bench:**

- No render on a physical 8 GB card. `--reserve-vram` sets
  `comfy.model_management.EXTRA_RESERVED_VRAM`, a Python integer that allocates
  nothing; it changes what ComfyUI BELIEVES it has, and that belief is what was
  tested.
- `FRAME_COST_MODEL` is measurably wrong in BOTH its frame term and its pixel
  scaling, but must not be re-fit off twelve points with no per-stage
  breakdown. The clamp is also invisible to the estimator by construction --
  `EXTRA_RESERVED_VRAM` is an integer while `free_vram_mb()` is a
  `torch.cuda.mem_get_info` driver read -- so the predictor structurally cannot
  observe the experiment's independent variable. Re-fitting is its own task with
  its own design, not a coefficient tweak.
- Per-stage VRAM measurement was DECLINED (operator ruling O7); whole-window
  peak is what this build has, and no per-stage primitive exists on the video
  path. The stage probe in the harness is ADVISORY for exactly that reason.

**NEXT, in order, when this reopens:**

1. ~~The like-for-like leg~~ -- **DONE 2026-07-31, item 5 above.**
1b. ~~Arm C (a step-distilled 5B)~~ -- **DONE 2026-08-01, item 7 above.** Built,
   passing, VRAM-identical to the incumbent at 2.76x the speed.
   **The live question is now a QUALITY eyeball, not a measurement:** is a
   3-step distilled render good enough to ship against the 30-step incumbent?
   This bench cannot answer that by design. Two candidates survive for the
   under-8 GB tier -- A (incumbent, known-good output) and C (same VRAM, 2.76x
   faster, output unreviewed) -- with D a third at a smaller canvas and an
   unresolved licence. Pick between A and C by looking at the renders.
2. **A physical 8 GB card.** Everything above is a clamp on a 16 GB card. Until
   this exists, "8 GB qualified" stays unsaid.
3. **Re-fit `FRAME_COST_MODEL` -- its own task, its own design.** Not a
   coefficient tweak: the estimator cannot see the clamp, and items 2 and 6 say
   both of its scaling terms are near-zero in reality, so decide what it should
   be MEASURING before deciding what the numbers should be.
4. **Adding a model is data entry ONLY when it reuses an existing recipe
   shape.** One `ArmSpec` row plus one video-only graph file gets a candidate a
   render and a VRAM number *if* it samples the way an arm already in the table
   samples -- swap `unet_name`, keep `KSampler{steps, cfg, euler, simple}`. That
   is how arms A / B-partial / B relate to each other, and all three ran clean
   on the first attempt.
   **It does NOT hold for a candidate that brings a new sampling contract.**
   Falsified 2026-07-31 on arm C (FastWan, 3-step DMD): expressing its schedule
   needs `KSamplerSelect` + `ManualSigmas` + `SamplerCustom` in place of
   `KSampler`, and the change set is a new graph file, a NEW required-class
   tuple (`_WAN_CLASSES` pins `KSampler` and omits all three), an `ArmSpec` row,
   a graph SHA pin, an `ARM_LICENCE` row, the arm-absence tests inverted, and a
   recipe contract in `offline_preflight` so the sigma literal cannot drift
   silently. Budget that as engineering, not data entry.
   **And a candidate is not admitted by having a file on disk.** The FastWan
   GGUF acquired 2026-07-31 has byte-perfect tensor-key parity with the
   incumbent and still fails to load -- 32 tensors are rank-2 where ComfyUI
   declares rank-3, and the norm vectors are the wrong length. Load-probe a new
   substrate through `gguf_sd_loader` -> `load_diffusion_model_state_dict`
   BEFORE writing a row for it; it is the cheapest falsifier there is and it
   costs no GPU.

Do NOT: re-run arm B, promote 14B, lower the greenlight bar, or start per-stage
measurement. The first is refuted, the next two are standing operator rulings,
and the last was declined.

### 8. `wan_ti2v` IS PRODUCTION-PROVEN, AND ITS LIVE VRAM IS NOT ITS BENCH VRAM (2026-08-01)

Asked whether the incumbent had ever rendered a live episode. **My first probe
said no, and my first probe was wrong.** It matched the string `wan_ti2v` in
filenames, then read `.meta.voice_cast_decision.*.engine` -- an AUDIO field,
which is why it reported `indextts2` and `z_image_turbo`. The video field is
`meta.render_engines`. Scanning all **1474** episode ledgers under
`output/otr/episodes` for a literal `"wan_ti2v"` VALUE:

| episode | wan_ti2v clips | all clips | ledger `vram_peak_mb` |
|---|---:|---:|---:|
| breathing_between_verses 2026-06-21 | 5 | 19 | 3734 |
| the_warming_knife 2026-06-21 | 5 | 19 | 4046 |
| illuminating_doubt 2026-06-25 | 19 | 19 | 3467 |
| bells_beneath_sardis 2026-06-25 | 19 | 19 | 3464 |
| breath_of_aethelgard 2026-06-30 | 7 | 7 | 3372 |
| unwrapped_secrets 2026-06-30 | 6 | 6 | 3435 |
| the_damp_grave_of_julian_vane 2026-07-23 | 7 | 7 | **9811** |

**68 clips delivered across 7 episodes, 2026-06-21 to 2026-07-23**, in all three
video roles (`music_visual`, `announcer_visual`, `character_video`). The most
recent episode's final asset decodes: 1920x1080, 25/1 fps, 118.80 s,
47,932,471 bytes. There is also an older single-clip receipt,
`output/otr/episodes/_shared/state/node_single_wan_ti2v.json` (2026-06-17):
`engine wan_ti2v`, 33 frames, 57.8 s, `vram_used_mb 8193` -- its scratch mp4 has
since been swept, but the receipt records `exists: true, size: 194278` at write
time. **The incumbent is proven live. Treat any claim that it is not as
refuted.**

**The live peak is NOT the bench peak, and that is the finding that matters.**
The 2026-07-23 ledger reports per-clip `vram_peak_mb` of 8251 / 9747 / 9778 /
9778 / 9810 / 9811 / 9811 against the bench's 6563.1 MiB for the same engine.
The six June episodes report 3372-4046 MB in the same field. `render_canvas` is
`null` in every per-clip row, so **the spread is measured and UNEXPLAINED** --
do not attribute it to canvas, to frame count, or to a measurement-scope change
without a probe that shows which. What it does establish: the bench's
`peak_delta_mib` is a property of the BENCH GRAPH, and the shipped adapter at
production settings has been observed 3.2 GB above it. No under-8-GB claim for
the live path survives this row.

**Arm C has ZERO episode receipts.** It exists as a bench arm only. So the two
candidates are not symmetric: C wins every measured axis (item 7) and A owns all
of the production history. Whatever ships, C earns its receipts before A loses
its menu row -- a swap in the other order trades a proven engine for an
unproven one on the strength of a bench that measures no quality axis.

### 9. ARM D'S BENCH GRAPH IS FAITHFUL TO THE SHIPPED RECIPE (2026-08-01)

Operator observation: arm D's render "looks like a fuzzy mess -- either it's bad
or you got the recipe wrong." Checked: **the recipe is not wrong.** Every
sampling-relevant value in `scripts/bench_graphs/arm_d_ltx_8gb.json` matches
`LTX8_RECIPE_V2` in `nodes/_otr_video_engines/eng_ltx_8gb.py` exactly -- steps 8,
cfg 1.0, `max_shift` 2.05, `base_shift` 0.95, `terminal` 0.1, sampler `euler`,
T5 on `cpu`, tiled VAE 512 / 64 / 16 / 8, checkpoint
`ltxv-2b-0.9.8-distilled.safetensors`.

So the softness is the CANVAS, not a drift: D renders **512x288 = 147,456 px**
against A/C's **832x480 = 399,360 px**, i.e. **36.9%** of the pixels. Viewed at
the same size, it is a 2.7x upscale. That was a deliberate campaign choice (D is
the only cross-family arm and 512x288 is its shipped tier), and it is exactly
why item 7 says never to read the seconds column across canvases.

One open discrepancy found while checking, NOT a fuzziness cause and NOT yet
run down: node 15 `LTXVConditioning` sets `frame_rate 25.0` while node 11
`CreateVideo` writes `fps 24.0`. 24/1 is the campaign-wide container constant
for cross-arm comparability; whether the shipped `ltx_8gb` adapter conditions at
25 and writes at 24 the same way is unverified. If it does, live LTX output is
running ~4% slow. Owed: read the adapter's own fps plumbing.

## OPEN BUGS / DEFECTS (live, not yet closed)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not.
That split is why the two eyeball-era entries below are PARKED rather than
listed as live.

- **NEW 2026-07-30: `full_text` reaches the span coordinate system carrying
  HTML BLOCK JOINS WITH NO SEPARATOR, and on the live evidence this is the
  DOMINANT P0 failure cause.** Measured in the campaign logs:
  `'...Field of Martian PolygonsNASA/JPL-'`, `'...and the School ofEngine'`,
  `'...what you're doing.Let's s'`, `'...(AMR).The resea'`. The RSS adapter
  strips tags without inserting whitespace, so two elements fuse into one
  token. `_normalize_span_source_text` collapses whitespace RUNS but cannot
  insert a space that was never there, so the model quotes the sentences a
  reader sees and they are not byte-exact in the stored text -- which is
  exactly the "non-literal source span" rejection that killed 12 of the 15 P0
  legs. **Deliberately NOT fixed by A-3:** A-3 is the narrow `&nbsp;` decode
  the plan ratified, and inserting separators is a WIDER change to the
  coordinate system `source_digest` pins -- an operator decision, and it
  belongs in the source adapter rather than the codex normalizer. Owed: which
  adapter builds `full_text`, whether a separator can be inserted at admission
  without breaking any accepted ledger, and a fixture from these four strings.
- **NEW 2026-07-30: the deterministic P0 rung PRUNES SILENTLY, which violates
  the plan's own Invariant 3.** `repair_literal_source_metadata` drops an
  unsupported span, then its evidence row, then the fact -- and emits no
  receipt. An accepted P0 index simply has fewer facts than the model wrote,
  and nothing says which were dropped or why. Under "fail loud, not fatal" the
  degrade is the right direction and the silence is not.
- **NEW 2026-07-30: the deterministic P0 rung is ALL-OR-NOTHING across an
  artifact, and can poison its own good work.** It is handed `a0_payload` (all
  seven keys) while `_validate_fact_index` restricts spans to
  `allowed_source_fields` (the projection). A quote rehomed into a field the
  projection omitted makes `post_validator` reject the WHOLE repaired
  artifact -- "cites source field ... outside the supplied P0 evidence" -- so
  one unlucky rehome discards every correct prune in the same pass. Either give
  the repairer the allowlist or prune per row.
- **NEW 2026-07-30 (recorded, no action owed yet): nothing measures whether a
  pruned P0 index is ACCEPTED.** No live leg has ever run with the
  deterministic rung reachable (it became reachable at `47c554fa`, after the
  campaign stopped), and the rejection logs carry only a truncated `raw head`
  plus no source payload, so the question cannot be answered offline. A-1's
  instrumentation is what makes the next campaign able to answer it.

- **NEW 2026-07-29 (LATENT, no shipped contract can reach it): the
  fewest-segments partitioner can accept a disproportionate trim on a WIDE
  discrete menu.** WIRE-W1 makes `partition_beat` take the lowest segment count
  that covers, including via a permitted tail trim. On a ladder that is always
  the right trade. On a DISCRETE menu whose largest entry dwarfs its smallest
  it need not be: covering 1019 frames from a `(10, 999)` menu, two segments
  give `[999, 999]` and discard 979 frames -- half the work -- where three give
  `[999, 10, 10]` exactly. **A bound was written and MEASURED and REVERTED, and
  the measurement is the point:** rejecting a trim of a whole smallest-clip
  turned `[12, 12]` into `[12, 4, 4]` on a `min=4 max=12 quantum=8` ladder --
  a third render and a third model load to recover four frames -- across 4,885
  cases in the sweep grid. **Not reachable today:** the widest shipped menus
  are Veo's `(100, 150, 200)` and Pixverse's `(125, 200)`, whose worst real
  trim is 25 frames. Revisit only if an adapter declares a menu with an extreme
  ratio; the reasoning is recorded in `coverage_plan.partition_beat` so the
  next reader does not re-derive the bound and re-ship the regression. Found by
  the pre-push fan-out (lens A), 2026-07-29.
- **NEW 2026-07-29: the B7 forbidden sweep cannot see an UNTRACKED file, so a
  new test file passes the gate and fails one commit later.**
  `tests/test_b7_forbidden_sweep.py` builds its input from
  `git diff s29-clean-slate-gate -- *.py`, which covers tracked files only. A
  new test file added and gated in the same session is green; the moment it is
  committed it enters the diff, and a forbidden runtime identifier in it turns
  HEAD red with nothing else changed. Cost one red HEAD this session (`alias`
  as a loop variable, the CW-6 marker). **Not fixed, because the fix is a
  judgment call:** sweeping the working tree instead of the diff would widen
  the gate to every untouched file in the repo. Cheap mitigation until then --
  re-run the full suite once after the FIRST commit of any new test file.

- **THE 8 GB PROFILE FAMILY CANNOT RUN ITS OWN WRITER** (found 2026-07-27,
  RENDER window; LIVE-REPRODUCED TWICE on two different banks, then confirmed
  by a two-strikes kibitz panel -- codex `gpt-5.6-sol` high and agy
  independently reached the same diagnosis). `config/profiles/otr_8gb_ltx.json`
  pairs a 12B GGUF writer (`gemma-4-12b-it-Q4_K_M`, 6.63 GB of weights) with
  `llm.gguf_n_ctx: 2048` under a declared `vram_ceiling_gb: 6.8`. The
  pipeline's own smallest prompt needs **2064 input tokens** and P0 reserves
  2800 output (`_P0_BASE_OUTPUT_TOKENS`), so the leg dies in
  `OTR_LedgerScriptWriter` before any render. Live preflight, verbatim:
  `Needed=8.13 GB (weights=6.63, kv=1.40 @ n_ctx=2048)`.
  **ctx is the SYMPTOM; the writer MODEL is the cause** -- 4096 puts it near
  9.4-9.5 GB, OOM on the very card the tier exists for. Every 2048-ctx profile
  (`otr_8gb_ltx`, `otr_8gb_wan`, `8gb_lite`, `cpu_floor`, `otr_amd8_rocm`,
  `otr_cloud_lanes`) is `status=draft` and every one pairs 2048 with the 12B;
  the only `status=shipping` profile is `16gb_full` (4096 + Mistral-Nemo).
  **NOT a one-line profile edit:** the GGUF registry ships exactly two rows
  (`unsloth/gemma-4-12b-it-GGUF`, `unsloth/Qwen3-8B-GGUF`);
  `google/gemma-2-2b-it` is in the TRANSFORMERS catalog, a different lane.
  agy proposed it and was wrong on that point -- recorded so nobody re-derives
  it. **Largely mooted by the operator's profile retirement** (see the header):
  with no profile passed, the canonical JSON's own `gguf_n_ctx=4096` / Q8_0
  binds and the leg runs -- which is exactly how all four sweep cells ran. Fix
  the profiles or finish retiring them; do not leave both.
- **A2 -- HELD 2026-07-27 pending the profile retire-now vs retire-later
  scope. The profile's `llm` section silently overrides the canonical JSON, and
  the applied-overrides echo HIDES it** (found 2026-07-27, RENDER window;
  grounded). Held because its entire subject is `apply_profile_to_workflow` and
  the printed echo -- a channel the operator has directed be retired, so
  building on it now may be work on something scheduled for deletion. The fix
  SHAPE below is correct and ready when the scope is settled.
  `_otr_workflow_validator.py:377` exports
  `os.environ["OTR_ACTIVE_PROFILE"]`, and the profile's `llm.*` values then win
  over the widgets the operator set in `otr_canonical.json` -- which ships
  `creative`/`technical` = `google/gemma-4-12b-it`, `gguf_n_ctx=4096`,
  `gguf_quant=Q8_0`, `llm_vram_ceiling_gb=14.5`. `scripts/otr_api.py:817`
  flattens only `role_overrides` / `slot_overrides` / `features` + two
  `seed_policy` keys for the printed summary, so the run reports "16 overrides"
  while ALSO having replaced the entire LLM configuration. The operator's
  mental model (the workflow is the authority) is the correct one and the JSON
  is already set up for it; the profile channel contradicts it invisibly.
  **CAUSAL CHAIN CORRECTED** (triage 2026-07-27, codex; grounded): the override
  does NOT come from the validator's `OTR_ACTIVE_PROFILE` export -- it happens
  at submission, `scripts/otr_canonical_api_run.py:157` ->
  `apply_profile_to_workflow`. And the real applier
  (`nodes/_otr_workflow_apply.py:492-540`) ALREADY flattens `llm`; only the
  printed echo (`scripts/otr_api.py:816-825`) is stale. **Fix: generate the echo
  FROM the applier's flattened map.** Adding `llm` to the echo by hand leaves
  the next drift intact.
- ~~**The 6.8 GB profile ceiling is DECORATIVE on the GGUF path**~~ --
  **CLOSED @ `ebec0f1f`** (A1). ONE policy-admission calculation now runs for
  every LOCAL lane before every cache read and every load
  (`_assert_policy_admits_vram`); remote lanes stay exempt by placement. It
  needed no new estimator and no ratio change -- see the header for why the
  "obvious" hard comparison would have refused the shipped default. Original
  row below, kept only for its cites. `_otr_model_loader.py`
  dispatches and caches GGUF before the generic `check_vram_fit` block, and
  `_otr_gguf_backend.py` checks PHYSICAL free VRAM instead of
  `policy.vram_ceiling_gb`. On a 16 GiB box an 8.13 GB writer therefore passes
  an "8 GB tier" boot -- which is why this surfaced as a context overflow
  rather than a ceiling refusal. ~~Enforce the policy ceiling inside GGUF
  preflight~~ -- **that fix shape is INCOMPLETE** (triage 2026-07-27, codex
  `gpt-5.6-sol`; grounded). A resident model returns at
  `_otr_model_loader.py:982-992` WITHOUT entering preflight at all, and
  `GGUFLoadConfig.reuse_key()` (`_otr_gguf_backend.py:435-439`) excludes the
  ceiling -- so a permissive-policy load satisfies a stricter-policy request by
  cache hit and a preflight-only fix misses it. **Correct shape: ONE
  policy-admission calculation before BOTH cache reuse and loading**, tested for
  permissive-cache -> stricter-request at the same load identity, plus the case
  where physical free VRAM exceeds the ceiling.
- ~~**A CLAMPED confirmation of recipe v2 is owed**~~ (RENDER window,
  2026-07-27) -- **DISCHARGED 2026-07-31 by the four-arm bench @ `8bd82efb`;
  see MEASURED above.** The original sweep ran unclamped because the
  profile-free writer is gemma-4-12b at Q8_0 (~13 GB), which cannot coexist
  with an 8 GiB reservation; the bench sidesteps the writer entirely by
  submitting direct-node graphs, so the clamp and the recipe could finally be
  observed together. Arm D ran `LTX8_RECIPE_V2` at the shipped 512x288 canvas
  under `--reserve-vram 8` and PASSED at 17 / 49 / 81 frames (peak_delta
  6691.1 / 6755.3 / 6819.1 MiB against a 7168 MiB bar). `VramPeakProbe` is
  still machine-wide, which is exactly why the bench grades on
  `peak_delta_mib` against each cell's own baseline rather than on the
  absolute. The RANKING never needed re-proving and was not re-run.
- ~~**All prequalification cells share ONE receipt**~~ -- **CLOSED by LANE 2**
  @ `71e231ec` + `8424f369`, on all three adapters. Record:
  `docs/2026-07-27-lane2-prequalification-receipt-qa-findings.md`.
- ~~**The `by_engine` roll-up keeps only the FIRST clip's receipt per
  engine**~~ -- **CLOSED @ `bcaab4db`.** The roll-up is PER FIELD now: an
  identity field (recipe / quant / use_lora / render_canvas / family) is
  reported only when every clip on that engine agrees, otherwise it is `None`
  and its NAME is listed in a new `varied`, so `None` means "no single value"
  and `varied` distinguishes "stamped nothing" from "stamped several".
  `vram_peak_mb` became the WORST clip's value (a measurement has a correct
  aggregate; a NaN peak is skipped because it compares False against
  everything and would silently discard a real measurement). Both credits
  readers moved in the same commit. **The reason it was untested rather than
  accepted, and the trap one level up:** the shipped fixture defined exactly
  one clip per engine -- and the first draft of the NEW tests had several
  clips but only ever one engine, so cross-engine isolation was still
  untested. The pre-push fan-out caught that; mutation could not.

- ~~**A THIRD copy of the scope encoder, with every defect the second one just
  had, plus the deadlock**~~ -- **CLOSED @ `6aad4fe5`, by DELETION.**
  `otr_scene_aware_scopes.render_scopes` calls
  `_otr_shared.scope_draw.encode_silent_mp4` and its private copy (and its
  private `_has_nvenc`) are gone; `_find_ffmpeg` stays, still used for ffprobe
  resolution. `_RAWVIDEO_STDIN_ENCODERS` in
  `tests/test_video_scope_draw_encoder.py` pins the six remaining
  rawvideo-stdin encoders with a reason each, and names this module in its own
  assertion, so a FOURTH copy fails by name. Mutation proved the delegation
  earns its keep: passing the node's dimensions SWAPPED, and declaring one more
  frame than the generator yields, are both refused now and were both silently
  accepted before. No behaviour change on the live path -- `out_w`/`out_h` are
  cast once and drive the plan, every frame canvas and the encode call; `_gen()`
  yields exactly `total` by construction; a zero-length manifest is already
  refused upstream; a sub-floor canvas selects libx264 rather than being
  refused. Original row below, kept for its cites.
  `nodes/otr_scene_aware_scopes.py::_encode_silent_mp4` (def at `:361`, body
  `:362-388`) is a near-copy of the scope_draw encoder: `total` is accepted and
  NEVER READ (no counter, no comparison before `return out_path` at `:387`);
  the rawvideo `-s` is built from the caller's `w`/`h` at `:368` and no frame's
  shape is ever consulted; the write loop at `:380` calls `.tobytes()` on
  whatever the generator yields with no shape or dtype check; `use_nvenc =
  _has_nvenc(fb)` at `:365` has no 145x49 minimum-size floor; and `stderr` is a
  PIPE read only at `:383`, AFTER `proc.stdin.close()` at `:382`, so an ffmpeg
  error burst larger than one OS buffer deadlocks the render with the child
  alive and the output file held open. **It is a live registered node**
  (`OTR_SceneAwareScopes`, `__init__.py:265`) but its artifact is a
  whole-episode compositing OVERLAY consumed as a bare path string by
  `OTR_PostUpscaleProcgenBlend` -- it never becomes a CanonicalClip and never
  reaches the per-beat ledger, which is why the roster gate does not and should
  not bill it. So this is a REAL defect in live code and NOT a clip-contract
  hole. `nodes/video_engine.py::_encode_mp4` shares some of the smell but
  already uses a temp-file stderr sink and never stamps a `frame_count` at all;
  `nodes/_otr_shared/encode_sink.py` is imported only by
  `scripts/profile_scope_render.py` and is not a live writer. Found by the
  pre-push fan-out (lens F), 2026-07-28.
- **KNOWN LIMIT of the widened roster gate, recorded so it is not rediscovered
  as a surprise:** the codec flag is matched as a STRING CONSTANT, so a flag
  assembled at runtime (an f-string, `"-c:%s" % stream`) or the stream-index
  spelling `-c:0` is invisible to the sweep. Nothing in the tree does that
  today; an encoder that ever needs to must be pinned in `_ENTRY_POINT_PROOFS`
  by hand, which the inventory test makes a visible decision. Separately, ONE
  mutant survives the round by construction: deleting the self-proving
  membership assertion is catchable only by a meta-test of that assertion.
- ~~**A SECOND clip encoder exists, it proves nothing, and the M7 roster gate
  cannot see it**~~ -- **CLOSED @ `27a4f97c`.** The four `viz_*` engines prove
  the colour/stream contract and stamp a frame count read off the FILE; the
  encoder itself stopped ignoring its `total` parameter, derives the declared
  size from the first frame (one derivation, so the stride cannot skew),
  refuses a frame that changes shape or dtype mid-stream, reaps ffmpeg on every
  refusal path and writes its stderr to a temp file rather than a pipe. The
  gate identifies a clip WRITER structurally. A latent box-dependent failure
  was fixed on the way: nvenc was selected whenever present and refuses a
  canvas below 145x49 (MEASURED here: 144x48 refused, 146x50 accepted).
  Original row below, kept for its cites. `nodes/_otr_shared/scope_draw.py::encode_silent_mp4`
  is a hand-rolled `Popen` wrapper returning only `out_path`, with NO ffprobe
  call anywhere -- no colour/stream contract, no frame count. Four LIVE engines
  write their clips through it (`eng_visualizer.py` viz_green,
  `eng_viz_camera.py`, `eng_viz_mandala.py`, `eng_viz_rainbow.py`), each
  returning its pre-computed loop bound as `frame_count`. The M7 gate
  (`tests/test_terminal_frame.py`) builds its `encoders` set by grepping for
  the literal substrings `encode_frames_to_silent_mp4(` and `run_ffmpeg(`;
  neither string appears in any of those four files, so they never enter the
  set and `encoders - provers == set()` can never flag them. **This is exactly
  the `cheap_families` shape of 2026-07-27 repeating one module over** -- the
  gate that was widened to catch a wrong filename now misses a wrong CALL
  SPELLING. Three of the four are the video slots the surviving six-bank 120w
  matrix uses, so this is the live path. Fix shape: make the gate identify a
  clip WRITER structurally rather than by known call spelling, expect it to go
  red, then give the second encoder both proofs. Found by the pre-push fan-out
  (lens F), 2026-07-28.

- ~~**`cheap_families`' four `still_*` engines self-declare their frame
  count**~~ -- **CLOSED @ `afeb5b84`.** `render_clip` reads the count back off
  the file (`proven_frame_count`, the muxer's own count off the SAME stream
  read `ffprobe_clip_fields` already performs one line above -- no decode) and
  `_floor_clip` stamps THAT. The M7 gate gained a matching COUNT half, billing
  the same roster by the same rule as the colour half, and it credits
  `ffprobe_counted_frames` too -- `wan_shared` proves its assembled beats by
  DECODING, the stronger proof, and requiring only the cheap header read would
  have failed it for proving more. **The trap the fan-out caught here:** the
  gate matched its proof markers as SUBSTRINGS, and `wan_shared.py` defines
  both helpers, so `def ffprobe_counted_frames(` satisfied the check on its own
  -- the one module that could regress its real `counted != expect_frames`
  comparison was the one module neither gate could notice it in. Proof is
  matched as an AST CALL now.
- ~~**A6 -- the Q4 artifact has neither an expected size nor a SHA**~~ --
  **CLOSED @ `ba24af29`**, corrected @ `40780b82`. Q8_0 and Q4_K_M are pinned
  by MEASUREMENT (three independent copies agreed byte for byte; the Q8_0
  measurement reproduced its existing pin, which corroborates the set); an
  unpinned artifact is REFUSED by name; the gemma registry row and the
  env-fallback path now read the one table instead of discarding its shas.
  **`Q6_K` remains unpinned and is therefore UNUSABLE until someone pins it
  from a box that carries the file** -- see the header.
- **`CanonicalClip.frame_count` -- "the integer timing authority" -- has two
  derivations** -- **HALF CLOSED @ `58e288af` + `40780b82`.** Every module that
  writes a clip now ffprobes it: the sweep found FOUR adapters missing the M7
  proof, not the two the row named (`eng_ltx_video` was listed as already
  probing and did not, on either recipe path; `eng_still_parallax` was absent
  from the row), plus `cheap_families` behind all four still_* engines. A
  roster gate in `tests/test_terminal_frame.py` fails by name for any module
  that writes a clip without proving it.
  **THE COUNT IS NOW CLOSED TOO @ `48e3c6fb`, and without paying a decode.**
  `encode_frames_to_silent_mp4` returns `proven_frame_count(...)`: the muxer's
  own `nb_frames`, which rides the SAME stream read `ffprobe_clip_fields`
  already performs on every clip, with the decode kept as the FALLBACK for a
  container that records no count, and a NAMED refusal on any disagreement in
  either direction. A zero-frame batch is refused by name at the encoder (it
  used to return `frame_count=0` over a container with no video stream).
  Measured: header 29-45ms flat, decode 35-168ms and scaling, against real
  beat renders of 744-842 SECONDS. **What this proves and what it does not:**
  it proves the muxer wrote what it was piped, which is the right question for
  a clip written by ONE ffmpeg pass; it does NOT prove decodability, which is
  why `assemble_beat_segments` still decode-counts every ASSEMBLED beat and
  must keep doing so. **Still self-declared elsewhere:** the four `viz_*`
  engines and the four `still_*` engines -- two separate rows above.
- ~~**The encoder boundary does not assert `dtype == uint8`**~~ -- **CLOSED @
  `de50786e`** (A5-lite). One assert at `wrapper_bridge.encode_frames_to_silent_mp4`,
  naming the byte multiplier and the converter to use.
- ~~**A FATAL env knob at the terminal node** and **the duration gate fails open
  while the receipt prints OK**~~ -- **CLOSED @ `54b3626b`** (found 2026-07-27,
  Fable consult; both in `nodes/otr_master_audio_mux.py`, the LAST node of the
  graph, where a raise costs the whole rendered episode).
  `OTR_MAX_CREDITS_TAIL_S` was an unguarded `float(os.environ.get(...))` -- the
  `PBUG-20260723-02` shape at the opposite end of the pipeline; now IGNORED and
  NAMED via `_credits_tail_ceiling()`. And `_probe_float` returning `-1.0` when
  ffprobe is absent skipped the only video-longer-than-audio guard while the
  report still appended `... OK`; the receipt now says `UNPROVEN` and names the
  gate as SKIPPED. Not made fatal -- it is the final sanity ceiling, not the
  primary correctness guard, and refusing would lose a finished episode on a
  box that merely lacks ffprobe.
- ~~**The `provider_side` redirect regression**~~ -- **CUT 2026-07-27, do not
  re-derive.** Already covered by
  `test_video_render_driver_perbeat_audio.py:319-325` (the redirect preserves
  `cloud_kling_avatar`), `test_video_platform_aseam.py:903-920` (picked route)
  and `test_still_plan_parity.py:114-116` (forced route). Filed originally after
  checking the CODE and not the TESTS.
- **EVERY LINE CITE IN THIS SECTION IS SUSPECT.** Each one checked during the
  2026-07-27 triage had moved: `_is_cloud_video_engine` is
  `render_driver.py:1599` not `1274-1295`; the "NO FALLBACK to text-only"
  refusal is `:2148` not `1801-1817`; `_use_i2v` is `eng_ltx_video.py:583` not
  `559-572`. The defects are mostly still real; their coordinates are not.
  **Re-pin a row's cite when you touch it.**
- ~~**RIDER on the credits-card display gap: `_row()` right-aligns with no
  clamp**~~ -- **CLOSED @ `24f4251a`** with the card wiring itself. `_row()`
  clamps against the space left of the label, and the ANNOTATION gives way
  before the engine id; every cut is marked with `...` rather than silently
  shortened. Measured pre-fix: `vx = -754` on a 120-character engine id.

- **NEW 2026-07-28: the credits card needs a SMALL-CANVAS VARIANT, and the
  ladder is not it.** At 512x288 (the ltx_8gb tier) col1 is 65px past its
  footer even with every ledger row this policy may drop already dropped; at
  640x360 it is 12px over. Both are now drawn anyway (a terminal node never
  destroys a finished episode) and LOGGED at ERROR naming the canvas -- the old
  behaviour was drawn, clipped by PIL, silent. At 288 lines the three-column
  console is already a polite fiction: col3's scrolling transcript is as
  unreadable as anything col1 clips. This is a DESIGN job -- a card laid out
  for a small canvas -- not more ladder heroics. Filed 2026-07-28 with the
  ladder that made the shipped 832x480 canvas fit.
- ~~**col1 overflows the footer at 854x480 on its REQUIRED content alone**~~ --
  **CLOSED @ `1959fb49`, and it was NOT latent.** `roll()` sizes the card from
  the FINISHED VIDEO, the canonical workflow ships 832x480, and that canvas was
  overflowing on every episode. The col1 ladder now spends the optional note,
  then inter-block WHITESPACE, then ledger ROWS (fine print first, always
  MARKED, SEED and COMMIT never dropped) -- and at 832x480 the whitespace rung
  alone clears the footer with 6px spare and the FULL ledger intact, no row
  dropped and nothing logged. Type is never shrunk; the card never raises. See
  the header for the standing policy and who ruled it. Original row below,
  kept for its cites. PRE-EXISTING -- it was already 20px past `h - 56*sx` before
  the recipe note existed, and PIL clips the overflow silently, so the tail of
  the [PRODUCTION LEDGER] grid is drawn off-canvas. `24f4251a` made the card
  measure what it can afford and spend the note allowance DOWN to zero, so the
  note cannot deepen this; it does not rescue it, and deliberately does not
  pretend to. Reachable only if something renders the card at 480p -- the
  shipped render tests use 720p and 1080p. Fix belongs to whoever next opens
  the card's geometry: either shrink the required blocks at small canvases or
  refuse the canvas. Found by the pre-push fan-out (lens D), 2026-07-28.

- ~~**An ODD canvas dimension makes the encoder's declared stride disagree with
  the bytes it pipes**~~ -- **CLOSED @ `b1f2ee86`.** `ffmpeg_silent_mp4_cmd`
  declares the REAL width/height (the `-s` describes the INPUT byte stream, so
  it is not ours to round), and `encode_frames_to_silent_mp4` REFUSES an odd
  canvas by name -- yuv420p subsamples chroma 2x2 and cannot represent an odd
  dimension, so there is no correct clip to write and rounding at the encoder
  would only move the same mistake one level down. `even_dim` stays on the
  three builders that SCALE or PAD to a target (still motion, still static, the
  lavfi floor), where ffmpeg is being told what to PRODUCE; both halves are
  asserted so they cannot later be collapsed into one. **The reason this sat
  filed as latent: the suite was DEFENDING it** --
  `test_ffmpeg_silent_cmd_contract` required `"832x480" in joined and "833x480"
  not in joined`, commented "odd width -> even". A latent row the tests assert
  as the contract is not latent, it is protected. Still true and NOT fixed
  here: neither `WanInitImageMixin._dims()` nor the `Canvas` schema validates
  evenness, so an odd canvas is caught at the encoder rather than where it was
  chosen. Original row below, kept for its cites.
- **NEW 2026-07-28 (LATENT, pre-existing): an ODD canvas dimension makes the
  encoder's declared stride disagree with the bytes it pipes.**
  `ffmpeg_silent_mp4_cmd` declares rawvideo `-s even_dim(w)xeven_dim(h)` while
  `encode_frames_to_silent_mp4` pipes `frames.tobytes()` at the array's REAL
  odd H/W, so ffmpeg slices the byte stream on the wrong boundaries. Measured:
  `(5,63,47,3)` encodes "successfully" to a 46x62 clip with skewed content and
  the new count proof PASSES it (5 frames either way); `(27,63,47,3)` and
  `(10,17,17,3)` now RAISE -- correctly refusing, but blaming a timing drift
  when the real cause is the stride. No live producer builds an odd canvas
  (every value in the tree -- 832x480, 512x288, 1472x832 -- is even) and
  neither `WanInitImageMixin._dims()` nor the `Canvas` schema validates
  evenness. Fix belongs with the geometry, not the count proof: make the
  declared size and the piped bytes ONE derivation. Found by the pre-push
  fan-out (lens G), 2026-07-28.

- **NOTED 2026-07-28, not a defect: two `scripts/` bake-off runners now abort
  a whole sweep on a count mismatch.** `scripts/run_ltx_av_q_bakeoff.py:453`
  and `scripts/run_humo_bakeoff.py:660` call the encoder inside per-leg loops
  with no try/except and DISCARD its return value (both set
  `result["frame_count"]` from `int(frames.shape[0])` independently). A
  disagreement that was previously invisible there is now fatal to the run.
  That is the correct direction -- a lying count is not a leg worth
  finishing -- but a sweep operator should know it before an overnight run.

- **The `ltx_8gb` render-length ceiling has TWO owners that only agree by
  coincidence** (found 2026-07-27, B6 panel, two lenses independently; grounded
  against the real files). The coverage PLANNER reads
  `config/profiles/otr_8gb_ltx.json` `video.max_render_frames` -- the channel
  B3 built, and `ltx_8gb` is the sole member of `PLANNING_CAP_ENGINES`. The
  ADAPTER's own pre-render refusal reads `OTR_LTX_8GB_MAX_FRAMES`. Today both
  land on 161 (profile unpinned, env unset), so nothing breaks. But
  `workflows/variants/otr_8gb_ltx.env.json` ships `OTR_LTX_8GB_MAX_FRAMES=97`
  and NOTHING currently reads that file. The day a launcher honours it without
  also pinning the profile, the planner emits a 98-161 frame segment and the
  adapter refuses it MID-EPISODE -- after the stills are minted and, on a
  multi-segment beat, after the 6.34 GiB checkpoint is hoisted. **Deliberately
  NOT fixed in B6:** pinning the profile to 97 changes how a 237-frame beat
  partitions, which is a production planning decision on the eve of 7d, not a
  cleanup. The preset now carries a `_ceiling_note` saying do not export it
  alone. Compare WAN, which B3 wired correctly: `otr_8gb_wan.json` sets BOTH
  `launch.env.OTR_WAN_TI2V_MAX_FRAMES` and `video.max_render_frames`.
- ~~**The recipe reaches the ledger but never the credits CARD**~~ --
  **CLOSED @ `24f4251a`.** `_draw_models` draws it as an indented micro-type
  note UNDER its engine row (the right-aligned suffix slot cannot hold ~90
  characters), and the COLUMN measures what it can afford: it flows itself
  onto a scratch canvas at the largest note allowance and steps down until it
  ends above the footer. **A fixed two-line allowance overran the footer by
  27px at 1280x720 -- the size this repo's own render tests use -- and the
  pre-push fan-out is the only reason that did not ship.** Rows that carry no
  recipe render pixel-identically to before.
- ~~**The WAN adapters have the whole pre-B6 defect, unfrozen**~~ -- **CLOSED by
  LANE 1** @ `71753cb4` + `3acc7fed`. Both recipes are frozen in code behind
  per-adapter consent acts and both adapters stamp a receipt. Detail in
  `docs/2026-07-27-lane1-wan-recipe-freeze-qa-findings.md` and HANDOFF_LOG.
- **OPERATOR CALL FLAGGED (LANE 1, not taken by the coder): rename the
  un-namespaced `OTR_WAN_*` knobs?** `eng_wan_i2v`'s six frozen knobs are
  `OTR_WAN_STEPS` / `_CFG` / `_SHIFT` / `_SAMPLER` / `_SCHEDULER` / `_NEGATIVE`
  -- no `I2V` namespace, unlike every sibling. LANE 1 left them alone on
  purpose: renaming an operator-facing knob is an operator's call, and the
  freeze already removed the power that made the missing namespace dangerous
  (they cannot bind a production leg at all; they are consent-act-only now).
  Default if unruled: leave them, because the risk they carried is gone and a
  rename would silently break any operator muscle memory for a sweep.
- **`eng_wan_i2v` threw away a VRAM peak it measured** -- FIXED @ `3acc7fed`,
  recorded because the CLASS matters: NEWBUG-1's 2026-07-20 fix landed on
  `wan_ti2v` and never reached its sibling, and nothing caught it for a week
  because no test drove `wan_i2v.render_clip`'s RETURN. When a receipt fix lands
  on one adapter, grep the siblings in the same change.

- **The route lock is ONE NODE TOO LATE for the image phase** (found
  2026-07-25, r3, both seats, node order confirmed against the canonical JSON:
  `87 VideoDirector -> 88 ImageDirector -> 89 MetaBrief -> 90 ShotLock ->
  91 ImageGenDispatcher -> 92 VideoRenderBatch`). `resolve_final_shot_engines`
  runs at node 92, but stills are minted at 91 and image PROMPTS at 89. The
  landed fix closed the spine-validation gap; the image phase still relies on
  its own MIRROR (`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`,
  whose docstring says it "mirrors the image dispatcher's effective-engine
  seam"). **Chunk 1 of the coverage block is the fix.** Note node 89 precedes
  node 90, so hoisting to ShotLock still does not put MetaBrief downstream of
  the authority -- that needs a VideoDirector-time freeze and is NOT in scope.
- **THREE silent coverage mechanisms exist, not one** (found 2026-07-25, r1,
  codex). **UPDATED 2026-07-27 (B4):** mechanism 1, the engine
  mirror/ping-pong (`wrapper_bridge.extend_frames_to_target`), is GONE from
  `eng_ltx_8gb` -- pinned behaviourally by a test that detonates the helper and
  renders successfully. It REMAINS in `eng_wan_ti2v`, deliberately and
  permanently: WAN renders a short native clip on purpose and fills the beat
  with it, which is the shipped 8GB tier contract `PBUG-20260723-02` protects.
  Still open: composite loop-fill (`otr_silent_composite._should_loop_fill`,
  which also SUPPRESSES its own underrun warning once it activates) and
  held-last-frame. For `ltx_8gb` the composite path is now de facto
  unreachable -- the adapter returns exactly the requested count or raises --
  but not structurally impossible: `encode_frames_to_silent_mp4` reports the
  size of the array it piped into ffmpeg rather than re-probing what ffmpeg
  wrote, so an encode-side drop could still under-report. PRE-EXISTING; close
  it when the assembly boundary is next opened.

- **THE 7d-PREFLIGHT THAT "PROVED THE GPU" RAN AT THE WRONG CANVAS** (found
  2026-07-27, B5 panel; verified, NEW -- and it corrects a claim this file
  made). `render_single` and both HTTP entry points use the older ledger-free
  `build_request`, which never reaches the canvas seam and defaults to
  `OTR_VIDEO_RENDER_CANVAS` (832x480). So the "GPU IS PROVEN" leg
  (`ltx_8gb`, 25 frames, 3004 MB) exercised 832x480, not the production
  canvas. ~~**The production canvas for `ltx_8gb` has never rendered live.**~~
  -- **NO LONGER TRUE as of 2026-07-31 @ `8bd82efb`.** Bench arm D rendered
  `ltx_8gb` at 512x288 three times (17 / 49 / 81 frames) under
  `--reserve-vram 8`, all PASS, assets decode-validated. That was a DIRECT-NODE
  graph, so it proves the CANVAS and the RECIPE, not the seam -- everything
  below still stands unchanged. `render_single` parity is
  explicitly deferred by the O1 judgment; what must NOT happen is another
  "proof" through that harness being read as a production proof.

- **The ShotLock WRITE-side canvas validation is still owed** (O1 judgment
  item 1; NEW). `otr_shot_lock.py` stamps `video.canonical_canvas` unvalidated
  from a possibly-empty policy. B5 made this non-load-bearing for the render
  (the engine declares its own canvas now), so it is no longer urgent -- the
  drift guard in `tests/test_ltx_8gb_canonical_canvas.py` covers the
  disagreement that matters. Close it when the general canvas resolver lands.

- ~~**`schemas.py`'s `ShotRow` is a closed model that no boundary enforces**~~
  -- **CLOSED @ `57caf43d`** (B4), completed rather than demoted. **The row's
  own field list was wrong in both directions**: it named a `beat_id` no
  producer stamps (`source_line_ids` carries the beat) and missed
  `jump_still_requests` and `motion_clause`, which are stamped. The shipped
  list was derived MECHANICALLY from the producers. Three fields default to
  `None` rather than an empty container because their ABSENCE is load-bearing
  (`coverage_plan`, `coverage_contract`, `motion_clause`).

- **`docs/ENGINE_MATRIX.md` reports the DECLARED contract only** (found
  2026-07-27, B3 post-code panel; NEW). Correct today and consistent with its
  own stated design (every number read from the live registry). But the moment
  a profile pins an `ltx_8gb` ceiling, the matrix keeps printing `9-161 step 8`
  for a tier whose real window is narrower, and the `--check` drift gate cannot
  notice because it diffs the registry, which the effective contract never
  touches. Owed at the prequalification step, not before.
- **`ltx_av` underruns long beats** (found 2026-07-25, r2, codex; confirmed).
  It caps at `_LTX_AV_MAX_FRAMES` (`eng_ltx_av.py:58`, default 497,
  env-overridable) and clamps at `:950-953`. It is NOT "renders to target
  natively" as three earlier docs claimed.
- **Ping-pong on a capped HuMo beat played lip sync BACKWARDS** -- FIXED in
  code @ `a1d810f1`, but the finding is STATIC (no live artifact), so it is
  NOT a PBUG row. A capped-14B leg would reproduce it. Kept here so the live
  proof is not forgotten.
- **`_should_loop_fill` names the permanent fix and it is now being built**
  (`otr_silent_composite.py:244-266`): *"The real fix is phrase-chunking
  (render the beat's correct duration so it never underruns) -- tracked as a
  follow-up."* The coverage block IS that follow-up.

- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0
  after two attempts on non-literal fact source spans; provider/model
  convergence, extends BUG-11.35. NOT a word/length gate. Blocks the last 120w
  receipt and the `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed
  in tree, reverify still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs
  provider cap `512`; the whole-artifact retry contracts LANDED @ `314dd481`
  are the base; the residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do
  not raise the minimum word target as a capacity workaround.
- **WAN 8-GB low-VRAM launch contract** -- **RE-GROUNDED AT HEAD 2026-07-31: this
  block is CODE-COMPLETE and PROOF-INCOMPLETE. It is not a coding item, and the
  one thing blocking it is an OPERATOR DECISION, not a keyboard.** Read this
  before opening a coder window on it -- the old one-line entry named no files,
  no seams and no acceptance criteria, and two nearby lines contradicted it
  (`:965` "WAN is already canonically qualified and remains closed" and the
  "needs no GPU to write" instruction, both stale).

  **What is already BUILT and WIRED end to end** (verified hop by hop at HEAD):
  `otr_8gb_wan.json` `video.max_render_frames=17` -> `capability_profiles`
  optional-key validator -> `_otr_workflow_apply.py:532` flatten ->
  `workflows/variants/otr_8gb_wan.json` node-87 widget slot 14 = 17 ->
  `otr_video_director.py:423` policy stamp -> `otr_shot_lock.py:1722`
  `ledger.video.max_render_frames` -> `render_driver.py:3328` per-adapter policy
  -> `motion_common.profile_max_render_frames()` -> `eng_wan_ti2v._floor_length`
  hard cap (`:730`) and `_planned_length` refusal (`:785`), with
  `render_driver.py:3845` refusing on drift. Landed `f914f0a4` (2026-07-24),
  dead node-87 widget repaired `7f4644a1` + `8f41af27`, WAN deliberately excluded
  from `frame_contract.PLANNING_CAP_ENGINES` by `b23fc035`, recipe frozen
  `71753cb4`/`8424f369`, whole-beat single-UNET-load `439ce8c7`. Regression net:
  `tests/test_remaining_video_contracts.py:16-194` (nine hop-by-hop tests) plus
  `tests/test_multiclip_effective_contract.py:216,234`.

  **THE ONE OPERATOR DECISION (this is the actual blocker).** The ceiling reaches
  a leg ONLY through a variant workflow or a hand-set widget: `otr_canonical.json`
  node 87 ships `max_render_frames=0`, so a plain canonical WAN run is UNPINNED
  and inherits `_TI2V_MAX_FRAMES = 177` -- exactly the 2026-07-23 failure shape.
  The obvious patch (pin 17 in the canonical) is WRONG: the canonical serves every
  tier, and 17 is the 8-GB tier's number, so pinning it would cap LTX/HuMo 16-GB
  legs too. The channel that carries 17 today is `config/profiles/*.json`, which
  `:749-753` puts on the RETIREMENT list -- so writing new behaviour onto it is
  explicitly forbidden. **Decision needed: after profile retirement, who owns a
  tier's native render ceiling?** The shape that fits the per-adapter-ownership
  doctrine is that `eng_wan_ti2v` DECLARES its own tier ceiling (a capability-row
  field), the widget becomes an operator OVERRIDE with 0 meaning "use the
  adapter's own contract", and the profile channel simply stops mattering. That is
  a real design change with a live-behaviour blast radius on any card with
  headroom (the VRAM predictor currently gets to ask for more than 17 and often
  can), so it is NOT being written on assumption. Ratify the shape first.

  **Also open, all PROOF obligations rather than build work:** (1) `:224` the
  18-engine GPU campaign is engine COVERAGE, NOT an 8-GB qualification;
  ~~(2) a CLAMPED confirmation is owed~~ and ~~(3) nothing in the tree is a live
  8-GB-clamped WAN render receipt~~ -- **BOTH DISCHARGED 2026-07-31 by the
  four-arm bench @ `8bd82efb`; see MEASURED above.** Arm A rendered
  `wan_ti2v` at 832x480, 17 / 49 / 81 frames, under `--reserve-vram 8`, and
  PASSED all three at peak_delta 6568.2 / 6563.1 / 6563.1 MiB against a
  7168 MiB bar; arm D did the same for `ltx_8gb` at 512x288. **Read the scope
  exactly:** that is a PREQUALIFICATION on a 16 GB card told to reserve 8 GiB.
  A render on a PHYSICAL 8 GB card is still owed and is the only thing that
  would make "fits 8 GB on real hardware" true without qualification; (4)
  `:2843-2858` every 8-GB profile INCLUDING `otr_8gb_wan`
  cannot run its own writer (12B GGUF, 8.13 GB needed under a declared 6.8 GB
  ceiling) -- largely mooted by profile retirement, but "fix the profiles or
  finish retiring them; do not leave both" still stands.

  **One untested edge, cheap to close whenever this reopens:** WAN is out of
  `PLANNING_CAP_ENGINES`, so a tier ceiling and a multi-clip plan CAN contradict
  by design, and `_planned_length` hard-refuses mid-episode when they do -- but no
  test asserts a 17-frame tier survives a multi-segment beat. `:216`/`:234` in
  `test_multiclip_effective_contract.py` pin the topology, not that outcome.

  **Sibling defect, unchanged, ltx side:** `eng_ltx_8gb` reads
  `OTR_LTX_8GB_MAX_FRAMES` and NOT the profile ceiling, `otr_8gb_ltx.json` leaves
  `video.max_render_frames` absent, and `otr_8gb_ltx.env.json` ships `97` that
  nothing loads -- the two-owner split already written up at `:3155-3171`.
- **Image-phase still ownership** -- bug-first item 2 above.
- ~~**`eng_ltx_video._use_i2v` contradicts fail-closed**~~ -- **CLOSED @
  `c9b89769`** (A4). The adapter now REFUSES instead of degrading, holding the
  same policy `render_driver.py:2146-2150` already held. The adapter's check is
  the stronger one: the driver asks only whether the still INDEX holds a
  non-empty path, so a STALE path passed it and degraded silently, and requests
  built through the ledger-free `build_request` never met it at all.
  `OTR_ENABLE_LTX_I2V=0` stays the named opt-out for a deliberate text render.
- **`style_tail_policy` closed enum cannot express a SHIPPED path** (found
  2026-07-25 by this window, missed by both r4 panelists; grounded).
  `VALID_STYLE_TAIL_POLICIES` has two tokens, `full` and `minimal_clean`, but
  `build_radio_host_prompt`'s `ltx_radio_mouth` branch
  (`otr_meta_brief_image_prompt.py:394-401`) RETURNS EARLY with
  `"%s, warm dramatic lighting"`, skipping BOTH
  `finish_visual_prompt(..., era_profile="still")` and the `image_grade_tail`
  append -- deliberately, per the 2026-07-02 operator look direction (the brief
  palette plus the grade tail rendered the talking-radio bookend dark, blue and
  murky). The `ltx_audio_in` bookend row nonetheless declares
  `style_tail_policy="full"`. **OPERATOR DECISION FLAGGED** (adding an enum
  token is explicitly an operator call, never a coder's): either add a third
  token for "canonical warm, no era tail, no grade tail", or ratify that the
  `ltx_radio_face` path is EXEMPT from the plan's style-tail authority.
  Default if unruled: the exemption, because it changes no behaviour. S1b did
  NOT touch `style_tail_policy`. S2 must not treat the plan as the style-tail
  authority for that path.
- **`wants_talking_prompt()` escapes any routing freeze** (r4 codex, grounded).
  It calls `_recipe_config(self._recipe())` and `_recipe()`
  (`eng_ltx_av.py:402-432`) re-reads `OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` /
  the UNET name on EVERY call by documented design ("Read fresh every call").
  So a `required="when_engine_talking"` row evaluated through the hook re-reads
  the environment after capture. S0b-core needs ONE shared `row_is_active(...)`
  evaluator over captured state, with the talking result inside `ltx_resolved`.
- **`provider_side` is a THREE-part rule, not an attribute** (r4 codex,
  grounded). `_is_cloud_video_engine` (`render_driver.py:1274-1295`) accepts a
  `cloud_` id prefix OR the attribute OR `node_key.startswith("cloud_")`.
  `cloud_kling_avatar` has no `provider_side` attribute and is caught by the id
  prefix alone, so an `engine_facts` builder using a bare `getattr` would
  classify it local and let the radio-host redirect send a cloud avatar to
  local LTX. Needs a regression on picked AND forced `cloud_kling_avatar`.
- **Four env-read sites missing from the S0b inventory** (r3 panel, grounded):
  `eng_ltx_video.py:541-564` (`OTR_ENABLE_LTX_I2V`), `render_driver.py:1176-1203`
  and `otr_meta_brief_image_prompt.py:297-300` (`OTR_ENABLE_HUMO_HOSTS`), and
  `eng_ltx_av.py:352-353,403-432` (recipe/UNET re-read outside `assert_usable`).

**PARKED -- unverified at HEAD, re-observe AFTER SFX (operator 2026-07-24).**
Both were eyeball observations against a story engine that has since had its
LLM vetoes ripped, THE LAW imposed (2026-07-22), six banks renamed onto new
packs, word-fit ceilings retired, the repair-first plan landed, and a ledger
cleanup pass added. Neither has a reproduction at current HEAD, and under the
standing rule a finding with no reproduction is not a row. Do NOT schedule
coder time against either. They are settled by the operator eyeballing a real
render leg after SFX: still there -> re-admit as a FRESH dated row with that
leg as evidence; gone -> the LAW-era work already fixed it, tombstone it.

- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`)
  -- PARKED. Episodes START a story instead of admitting you into one; the
  announcer takes debate turns instead of framing. Operator eyeball
  2026-07-11. If it survives re-observation the fix is still seam + score
  contract + fail-closed validator, never Python authorship.
- **Name-splice defect #2** -- PARKED. v4-campaign Phase 0 record in
  HANDOFF_LOG; its timebox predates THE LAW.

- **PBUG-20260710-07** -- root fix shipped; stays ROOT-OPEN in the log until
  ratified at the next operator fan-out (green codex leg `c1f3891f` is the
  retire candidate).
- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema
  `.v4` literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

## Coder queue (operator order 2026-07-27, lean-mean removed 2026-07-29)

One coder window at a time; every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

```text
LOCAL-ENGINE OBS WIRING (WIRE-W1..WIRE-W7)
       (arc r1-r4 CLOSED and converged at HEAD -- no re-ground owed;
        spec = kibitz-runs/2026-07-28-local-engine-obs-wiring/r3/final.md
        as amended by r4/final.md. THIS IS THE CURRENT STEP.)
  -> WAN 8-GB low-VRAM launch contract  (no re-ground needed: a live
                                         2026-07-23 defect, not an old plan)
  -> [r3+r4] Randomizer A
  -> [r3+r4] dynamic_story           (wiring only -- rev-5 DESIGN stays FINAL)
  -> [R4.1 refit = its re-ground] SFX campaign (after Timeline Cue Ledger C0/C1)
  -> re-observe the PARKED story bugs; batch-triage whatever is left
  -> THEN, and only then, ROADMAP.md -- OFF THIS PLAN
       (its order: SFX -> product expansion -> LEAN-MEAN -> RunPod -> release)
```

**LEAN-MEAN FRONT AND TAIL ARE NOT IN THIS QUEUE.** Operator direction
2026-07-29 moved both to the Lean-mean campaign section of `ROADMAP.md`. That is the whole point of
their position: a deletion campaign's value IS its file-and-line kill
inventory, so it re-grounds ONCE against the final tree instead of ripping
first and re-grounding after every block lands on freshly-rewired code. **Do
not re-add a lean-mean line to this fence.**

The bracket is the STANDING RE-GROUND GATE below. Every one of these plans was
written against a tree that no longer exists. Default entry is r3 (wiring);
drop to r2 if the coding plan itself is wrong; if in doubt, start at r2. No
block executes without an r4 convergence at current HEAD.

CUT by the operator 2026-07-24 and NOT to be re-derived by a later window: the
45-word scene matrix, the 54-case visual-style sweep, and the entire
quick-wins block. Image-phase still ownership and the six-bank requalification
were not cut -- they stay in OPEN BUGS / the campaign queue and get picked up
whenever a render window opens.

### Quick-wins block -- CUT 2026-07-24 (operator)

The whole block is gone. The operator's call, verbatim in intent: "we will
triage more bugs later" -- the block was a schedule, and ripping a schedule
does not rip the underlying defects. Everything in it that was a real bug
still lives in OPEN BUGS above; everything in it that was a nice-to-have is
simply not being built. Do NOT re-derive this table from git history.

ONE item survived the cut, folded into LEAN-MEAN W6 as a sub-step rather than
kept as a standalone chunk. **It travelled to the Lean-mean campaign section of
`ROADMAP.md` with that block on 2026-07-29; the spec is reproduced there. Kept
here only so the record of what the 07-24 cut spared stays complete:**

- **`docs/ENGINE_MATRIX.md`** -- emit from the three live CAPABILITIES
  registries per the existing generator pattern (`build_variants.py`
  ~:276-338): write during `--all` / explicit emit; `--check` regenerates in
  memory and FAILS on drift without writing. Columns + stable ordering; link
  from README. The lean-mean doc (`:301-304`) only needs W6's README policy
  line to link it, so this is an ordering preference the operator set on
  2026-07-10 -- NOT a hard technical dependency. W6 executes without it; the
  README link is what suffers. Estimate 0.5-1 d.

Also recorded so a later window does not re-open them: quick-win 6
(`scifi_news_pro` C5 consumers) was already CLOSED IN CODE under
PBUG-20260720-04. The `scifi_news` live reverify (PBUGs 20260712-22/23/24/25)
is not lost either -- it moved into the `scifi_news` P0 convergence row in
OPEN BUGS, which is what actually blocks it.

### STANDING RE-GROUND GATE -- r3/r4 before ANY remaining block (operator 2026-07-24)

Every remaining big block was planned on a tree that no longer exists. Since
those docs were written the LLM vetoes were ripped, THE LAW landed, six banks
were renamed onto new packs, word-fit ceilings were retired, the whole
extensibility build shipped (seven waves, a new routing authority, a new
network seam, a new ledger-cleanup pass in the writer tail), and the suite grew
past 6,400. A plan's line cites, seam names and file inventories are the FIRST
things to rot, and every one of these blocks is a rip or a rewire that acts on
exactly those.

**THE GATE, in the operator's words: run an r3-r4 for all remaining blocks; if
issues turn up go back to r2; if in doubt, restart at r2.** Concretely:

- **Default entry point is `r3` (wiring).** These plans already have an r1
  (arc) and an r2 (coding plan) on record, so the cheap re-ground is the wiring
  round run against CURRENT code, followed by `r4` (convergence). Use the local
  panel (`/kibitz`: codex `gpt-5.6-sol` high + agy) -- it crawls the REAL repo,
  which is the whole point here.
- **Drop to `r2` when r3 finds the CODING PLAN wrong, not just the line
  numbers.** Stale cites are an r3 fix. A seam that no longer exists, an
  authority that moved, a step whose precondition another build already
  satisfied or destroyed -- that invalidates the coding plan itself, and
  patching an r2 from inside an r3 produces a plan nobody reviewed.
- **If in doubt, start at r2.** A wasted r2 costs one panel round; executing a
  stale coding plan costs a day of rips against the wrong file list, and the
  rips are the hard kind to unwind.
- **No block executes without an r4 convergence at current HEAD.** Record the
  run under `kibitz-runs/<date>-<block>-r<N>/` and cite it in the block entry
  below when it lands, so the next window can see how fresh the grounding is.
- **The lean-mean full-`r2 -> r3 -> r4` operator pin MOVED with the block** --
  it now lives in the Lean-mean campaign section of `ROADMAP.md`, not here. Nothing on THIS plan is
  pinned below the r3 default.
- **Credit note:** this is rung 2-3 work (agy is $0; codex is weekly credits).
  Roughly four panel rounds across the remaining GO_FORWARD blocks (r3+r4 for
  randomizer and dynamic_story; the wiring block's arc is already spent and
  SFX's R4.1 refit IS its re-ground) -- front-load it early in a credit week,
  run each block's arc when that block opens rather than all at once, and never
  let a stale `codex_model_selected.txt` silently drop an arc to codex-only or
  to the wrong model. The two lean-mean arcs (three rounds each) are ROADMAP
  spend now, not GO_FORWARD spend.

### Big blocks (in operator order, 2026-07-27)

**LEAN-MEAN IS NOT IN THIS LIST, AND MUST NOT BE RE-ADDED.** Operator
direction 2026-07-29: lean-mean comes OFF GO_FORWARD entirely and lives in
the Lean-mean campaign section of `ROADMAP.md` -- FRONT and TAIL both, with their chunk chains, the W2
migration-first mandate, the ENGINE_MATRIX W6 sub-step and the full
`r2 -> r3 -> r4` operator pin carried over intact. It runs AFTER the randomizer
and SFX, off this plan. A window that wants to rip dead code is on the wrong
document.

1. **Randomizer Rolls Design A** --
   `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`. NO LONGER GATED --
   extensibility landed, and its `_otr_lane_specs` authority was ABSORBED by
   that build, so this shrinks to `_otr_bank_roll` + eligibility. **RE-GROUND:
   r3 + r4 REQUIRED.** Note what the doc's own filename admits -- it is an r2
   coding plan that NEVER got an r3 or r4, so this is the arc completing, not
   repeating. Its r3 brief must carry two known deltas: the absorbed
   `_otr_lane_specs` authority, and that the bank list is now a LIVE registry
   read (`list_bank_ids()` can return a CLIENT bank; eligibility must treat one
   as an ordinary peer) rather than a six-row literal. 1-2 d + 1 GPU day.
2. **`dynamic_story` visual direction** -- rev-5 FINAL; roster-agnostic;
   re-derive IDs at build. After the randomizer. **RE-GROUND: r3 + r4
   REQUIRED, and the "do not rerun panels" rule still holds -- these are not in
   conflict.** That rule protects the DESIGN (the r1 arc: what the feature
   should be, already settled over five revisions). r3 asks a different
   question -- does this design still WIRE to the code that exists today -- and
   the roster, the routing authority and the writer tail have all moved since
   rev-5. Re-litigating the design is forbidden; re-grounding the wiring is
   mandatory. 5-9 coder-days + 2-4 GPU days.
3. **SFX campaign** (after the Timeline Cue Ledger C0/C1 gate) -- **RE-GROUND:
   the R4.1 refit already IS this gate.** The generated-SFX R4 candidate stays
   local/ignored evidence until it is re-grounded into a tracked current-HEAD
   R4.1 plan; treat that refit as the r3/r4 pass for this block rather than
   scheduling a second one. Sequencing + scope contract live in `ROADMAP.md`
   (no second SFX queue, no library fallback).

Open judgment question (render-window, not coder-slot): the LOCAL
mistral/gemma writer matrix -- the Sonnet arm of the creative-writer question
is answered (record: `docs/2026-07-17-model-bakeoff-scoreboard.md`); the local
roster comparison never ran.

## Window packing (credit discipline -- one line starts any window)

Starting any window costs the same boot context, so BATCH chunks per window
and never open one for a single small item. Every window starts by pasting
its one-line kickoff -- the `otr-handoff` skill reads this file + git and
states the current step. No manual context handoff, ever. This planner window
keeps GO_FORWARD + HANDOFF_LOG current; coder windows never write plans
(window-roles rule).

| Window | Scope | Model rung (see MODEL & CREDIT BUDGET) | Gate | Size |
|---|---|---|---|---|
| RENDER | finish the six-bank 120w wrap ONLY (the 45w matrix and 54-case sweep are CUT); fillers: cpu-tier smoke + nv50 re-soak | local production + Codex-app monitor | opens whenever the operator wants a live leg | GPU days |
| **CODER W "local-engine OBS wiring" -- THIS IS THE OPEN SLOT** | Close the 6 NO_RENDER local engines so all 18 land a 45-word episode in `otr/obs/`. Arc r1-r4 CLOSED. Spec = `kibitz-runs/2026-07-28-local-engine-obs-wiring/r3/final.md` AS AMENDED BY `r4/final.md`; per-chunk gate = codex's VERIFY-AT-BUILD checklist in `r4/codex.md`. Order **WIRE-W1 -> WIRE-W2 -> WIRE-W6 -> WIRE-W3 -> WIRE-W4 -> WIRE-W7 -> WIRE-W5** (see CURRENT STEP for each). **These W numbers are NOT lean-mean's W0..W8 -- different block, different spec file; prefix them WIRE- in every message.** | Claude codes + judges; kibitz = codex `gpt-5.6-sol` high + agy | arc already converged at HEAD -- no further re-ground owed; code now | multi-day |
| CODER A "multi-clip coverage" | WAN 8-GB `f914f0a4`; still-plans S0a/S0a-b/S1/S1b landed then SUPERSEDED. r1/r2/r3/r4 arc JUDGED and CONVERGED. **Chunks 1-7a COMPLETE plus nine QA rounds; then the 8GB block: B1a, B2a, B2b, QA-4, the `*_DIR` tripwire, B1b-0, B1b, and now B3 + B4 + B5 + B6. All LANDED GREEN and PUSHED (suite 7213).** THE CODER-WINDOW BLOCK IS COMPLETE. NEXT = **prequalify 512x288** -- a GPU step, so a RENDER window owns it, not this one (boot with `OTR_LTX_8GB_PREQUALIFICATION=1`, measure T5 device and tiled decode, freeze the winner as recipe v2) -- then **7d** -- the canonical 237-frame opening beat, which is where a GPU first renders through this machine at its production canvas. Seams tabulated in CURRENT STEP -- do not re-invent them. Pause map and audio lanes come LAST. Plans of record: `docs/2026-07-26-8gb-1080p-arc-judgment.md` (the architecture), `docs/2026-07-25-multiclip-coverage-r{1,2,3}-judgment.md`; operator rulings verbatim in `docs/2026-07-25-per-beat-stills-r1-judgment.md`. | Claude codes + judges; Sonnet fan-out + agy for QA rounds (cheap, $0 for agy, and between them they have found real defects in already-green code five times); kibitz = codex `gpt-5.6-sol` high + agy | chunk 7 needs a selective box reset per CLAUDE.md section 4 | multi-day |
| ~~CODER B~~ | quick-wins harness window -- **DISSOLVED** by the 2026-07-24 rescope (its whole scope was quick-wins) | -- | -- | -- |
| ~~CODER C~~ | quick-wins foundations window -- **DISSOLVED** by the 2026-07-24 rescope; ENGINE_MATRIX moved into the lean-mean W6 sub-step, which is now in `ROADMAP.md` | -- | -- | -- |
| ~~CODER D~~ | lean-mean front -- **REMOVED FROM THIS PLAN 2026-07-29 (operator).** The block is not cancelled; it MOVED to the Lean-mean campaign section of `ROADMAP.md` with its chunk chain, its W2 migration-first mandate and its full `r2 -> r3 -> r4` pin. It runs after the randomizer and SFX, off GO_FORWARD. **Do not re-add this row.** | -- | -- | -- |
| PLANNER | extensibility hardening + `docs/EXTENDING_OTR.md` DONE 2026-07-24; NEXT = Bug Bible operator fan-out + the `check_compatibility` fork; plan upkeep | rungs 2-4 | parallel with any coder window | docs |
| ~~CODER E~~ | independent client-authored source banks v1 -- **ALL SEVEN WAVES DONE @ `30358ad1`**; slot RETIRED, do not reopen (deferred power-user tiers are a NEW block, not this one) | -- | -- | -- |
| CODER F | **r3 + r4 arc per block FIRST**, then Randomizer A -> `dynamic_story`. For `dynamic_story` the arc is WIRING ONLY -- rev-5's design stays FINAL, do not rerun the design panels. | Claude codes + judges; kibitz = codex + agy | after the WIRING block and WAN 8-GB (the old "after D" gate is VOID -- D is off this plan entirely); NO code before r4 converges at HEAD | ~6-11 d |
| ~~CODER G~~ | lean-mean tail -- **REMOVED FROM THIS PLAN 2026-07-29 (operator)**, same move as D: the Lean-mean campaign section of `ROADMAP.md`, after the randomizer and SFX. Its SW-1 writer re-survey still runs against the THEN-current writer, which is exactly why it is off a plan whose blocks keep editing that writer. **Do not re-add this row.** | -- | -- | -- |

### If the window is a REMOTE / cloud Cowork session -- READ THIS FIRST

Learned the hard way 2026-07-26. A Cowork session running IN THE CLOUD is not
the same box as the repo, and two of CLAUDE.md's assumptions do not hold:

- **Read/Write/Edit hit the CONTAINER, not the Windows files.** CLAUDE.md
  section 1 ("the file tools are your primary editor") is written for an
  on-computer session. In a remote window every read, edit and write goes
  through Desktop Commander against
  `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, and so
  does git, the venv python and pytest. Everything else in CLAUDE.md holds.
- **There is a LAGGING Linux snapshot at `/mnt/user-data/uploads/`.** Never
  read the repo through it, and say so explicitly to every subagent: a prior
  session's agents reported phantom corruption from that mount. Every quote in
  a review must come from a Desktop Commander read of the Windows path.
- **The bridge can drop mid-edit.** It did, once, for a few minutes. If the
  `mcp__remote-devices__*` tools vanish, STOP -- do not retry in a loop and do
  not leave a half-applied edit. Report what is on disk, uncommitted, and wait.
  Nothing was lost when this happened because the last green chunk was already
  pushed, which is the actual argument for pushing every chunk.
- **The 60s MCP call ceiling.** The full suite takes ~110s, so launch it
  detached (`Start-Process -FilePath 'powershell' -ArgumentList ...
  -WindowStyle Hidden`) writing to a `tmp/` log + a `.done` marker, then poll.
  PowerShell `*>` redirection writes UTF-16, so read results with
  `Select-String`, not by eyeballing the raw file.

Kickoff lines (paste as the FIRST message of the new window).

**NEVER boot a window by LETTER alone.** The letters are a scope index, not a
running order, and the Gate column outranks them -- a window that reads its
letter and starts coding will run whatever block sat there when the letter was
assigned. **Boot by CURRENT STEP, always:**

> resume the OTR build -- you are a CODER window. Read GO_FORWARD "CURRENT
> STEP" and execute THAT block only, in its stated order, one green pushed
> chunk at a time. If CURRENT STEP names a different block than a letter row
> in "Window packing", CURRENT STEP WINS -- say so and proceed. State your
> MODEL & CREDIT BUDGET rung first.

## Parallel lane -- no coder slot required

- **Bug Bible operator fan-out** -- 9+ closed candidates + the
  duplicate-legacy_id cleanup waiting on one fan-out session.
- **Render-window fillers:** cpu-tier smoke (needs the google image lane or
  stills) + nv50 re-soak -- the two open portability remainders; release QA
  validation time, not coding.
- **SFX R4.1 re-ground** (0.5-1 docs day): re-ground the local generated-SFX
  R4 candidate into a tracked current-HEAD R4.1 plan. Sequencing + scope
  contract live in `ROADMAP.md` (Timeline Cue Ledger C0/C1 gate first; no
  second SFX queue, no library fallback).
- **Operator-promotable option:** SFX C0 (per-line WAV stems + transcript
  drift report) is independently shippable per ROADMAP but stays parked
  unless explicitly promoted.

## Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk (its quick-win-9 home was cut 2026-07-24); never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval
queue is `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented
fixture creates a row.

## Validation and handoff law

- Current whole-tree receipt (2026-07-31, the A@512x288 like-for-like leg + the
  `parse_timing` precision fix): full Windows suite
  `8135 passed / 130 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; `scripts/build_variants.py --check`
  11 variants / 0 failures; canonical
  `9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`
  byte-identical (no node, widget, link or schema touched); AST/BOM/zero-byte/
  UTF-8/ASCII clean; HEAD == origin verified after the push. Three new GPU cells
  under `_bench_4arm/diagnostic_512x288/`, all PASS, all decode-validated.
- Prior receipt (2026-07-31 @ `f3bc01cc`, the diagnostic canvas
  override + the measured 8 GiB-clamped bench written into the record): full
  Windows suite `8133 passed / 130 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; `scripts/build_variants.py --check`
  11 variants / 0 failures; canonical
  `9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`
  byte-identical across both commits (no node, widget, link or schema touched --
  a bench script, its tests, and four docs); AST/BOM/zero-byte/UTF-8/ASCII clean
  on every touched file, verified after the push with HEAD == origin. Measured
  with the tree exactly as found: another window's three modified `tmp/*.ps1`
  (the `bake420e` leg) and the untracked `docs/*.log.err` + `docs/*.pdf`
  scratch were preserved and NOT committed.
- Prior receipt (2026-07-28 @ `1959fb49`, the encoder arc + the
  credits-card ladder): full Windows suite
  `7464 passed / 27 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; `scripts/build_variants.py --check`
  11 variants / 0 failures; canonical `9872624A` byte-identical across all
  five commits; AST/BOM/zero-byte/UTF-8 clean. Same pre-existing-section-sign
  note as below.
- Prior receipt (2026-07-28 @ `b1f2ee86`, the encoder arc: second
  encoder, still_* counts, the third copy deleted, the odd-canvas stride): full
  Windows suite `7453 passed / 27 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; `scripts/build_variants.py --check`
  11 variants / 0 failures; canonical
  `9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`
  byte-identical across all four commits (no node, widget, link or schema
  touched); AST/BOM/zero-byte/UTF-8 clean on every touched file. **Note on
  ASCII:** `otr_scene_aware_scopes.py` and its test carry a pre-existing
  literal section sign; the non-ASCII inventory is byte-identical to HEAD, so
  nothing new was introduced and they were not rewritten for it.
- Prior receipt (2026-07-28 @ `afeb5b84`, the second encoder +
  the still_* count proofs): full Windows suite
  `7449 passed / 27 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; `scripts/build_variants.py --check`
  11 variants / 0 failures; canonical
  `9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`
  byte-identical across both commits (no node, widget, link or schema
  touched); AST/BOM/zero-byte/UTF-8/ASCII clean on every touched file.
  Measured with the tree exactly as found -- another window's three modified
  `tmp/*.ps1` and its six untracked `config/profiles/otr_sbcov_*.json` were
  preserved throughout and NO variants were generated from them.
- Prior receipt (2026-07-28 @ `48e3c6fb`, the three remote-safe
  rows): full Windows suite `7429 passed / 27 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; `scripts/build_variants.py --check`
  11 variants / 0 failures; canonical
  `9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`
  byte-identical (no node, widget, link or schema touched by any of the
  three); AST/BOM/zero-byte/UTF-8 clean on every touched file. Measured with
  the tree exactly as found -- another window's six untracked
  `config/profiles/otr_sbcov_*.json` were left in place and NO variants were
  generated from them (`--check` regenerates in memory and writes nothing), so
  the count reproduces on a clean clone.
- Prior receipt (2026-07-27 @ `40780b82`, the ranked bug queue +
  its QA fan-out corrections): full Windows suite
  `7384 passed / 27 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; `scripts/build_variants.py --check`
  11 variants / 0 failures; canonical
  `9872624A311AB52D6A7112BFF5E3C7BB83B85103331E4455DECB64AA2325D25D`
  byte-identical; AST/BOM/zero-byte/UTF-8/ASCII clean on every touched file.
  **Count the suite on a clean tree:** `build_variants.py --all` also emits
  variants for any UNTRACKED profile on disk, and some profile checks are
  parametrized over the variants present, so another window's scratch profiles
  can inflate the number by a dozen tests that would not reproduce on a fresh
  clone. This receipt was re-measured after removing that side effect.
- Prior receipt (2026-07-27 @ `8424f369`, LANE 1 + LANE 2 landed):
  full Windows suite `7346 passed / 27 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; canonical `9872624A` byte-identical.
- Prior receipt (2026-07-27 @ `3acc7fed`, LANE 1 landed):
  full Windows suite `7291 passed / 27 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; canonical `9872624A` byte-identical.
- Prior receipt (2026-07-27 @ `a0141cdd`, B3 + B4 + B5 landed):
  full Windows suite `7158 passed / 27 skipped / 1 xfailed`; Bug Bible
  `17 passed / 24 skipped / 3 xfailed`; canonical `9872624A` (byte-identical --
  no chunk in the 8GB block touches a node, widget, link or schema; the
  `max_render_frames` TOOLTIP that B3 rewrote lives in Python `INPUT_TYPES`,
  never in the graph). Detail in HANDOFF_LOG.
- Every code chunk: focused tests, full Windows suite, Bug Bible,
  AST/JSON/BOM/zero-byte checks, commit, push, verify
  `HEAD == origin/v2.0-alpha`.
- Every node/widget/link/schema change edits `workflows/otr_canonical.json`
  in the same commit and runs `OTR_WorkflowValidator`, JSON round-trip,
  strict link/input, live widget-vector, and generated-variant audits.
- Reset selectively before every headless run; never blanket-kill Python.
  Every run loads the canonical workflow and writes directly to canonical
  episode/OBS paths. Asset existence, not resident VRAM, proves completion.
- One coder edits code or `workflows/otr_canonical.json` at a time; read-only
  audits and documentation may run in parallel. HANDOFF_LOG + this file are
  the only tracking surfaces (the otr-build-tracker artifact is RETIRED).

## Open risks

- Extensibility v1 is DONE, so it no longer constrains randomizer /
  dynamic_story sequencing. Deferred power-user tiers (client own-runner +
  staging, dependency manifest, standalone story_rules) are explicitly OUT of
  v1 and are a NEW block if the operator ever wants them -- not a reopening of
  CODER E. NO CLIENT BANK HAS EVER RUN LIVE: every wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven
  path end to end (fetch -> interpret -> writer -> cleanup -> tail -> publish).
  Treat the first live client-bank leg as a qualification, not a formality.
- CLIENT-AUTHORED PYTHON executes in-process (wave 3). The posture that must
  hold in every future change: `--activate` is the consent act; the seam fails
  LOUD (`UserBankExecutionError`) and never substitutes; client code never
  touches the canonical ledger; owner IDENTITY is verified so a bank can only
  run its OWN bundle; the shipped fetcher/interpreter registries are never
  widened to admit a client id. Do not relax any of these for convenience.
- The client-facing surface is now LIVE TEXT, not just docs: the
  `custom_source_bank` row's `guide_ref` is raised to the operator by
  `require_runnable_bank`, and the `source_bank` tooltip repeats it. Any future
  change to the activation path (folder name, CLI verb, restart behaviour) must
  update `nodes/story_packs/banks.json`, that tooltip and
  `docs/EXTENDING_OTR.md` together, or the product will confidently instruct
  clients to do the wrong thing.
- **`check_compatibility` is RESERVED, not wired (wave-4 decision, kibitz
  r3 codex `gpt-5.6-sol` high + r4 agy Gemini 3.6 Flash High, Claude judge).**
  No request type, no decision type, no runtime consumer exists, so activation
  does not inspect it -- not even for callability -- and `EXTENDING_OTR.md`
  now calls it a reserved name instead of "NOT YET WIRED". `COMPAT_ENTRY_ATTR`
  is left INERT in `BUNDLE_ENTRY_ATTRS` with a comment saying so. **Operator /
  planner decision flagged, NOW WITH A 2-of-2 RECOMMENDATION TO RIP
  (2026-07-24, operator-directed consult; codex `gpt-5.6-sol` high and Fable,
  independently, no shared context; Claude grounded both against the tree):**
  the argument that decided it is that Option A's stated benefit is FALSE --
  `BUNDLE_ENTRY_ATTRS` constrains what OTR-side code may request from
  `bundle_entry_point()`, it reserves nothing against clients, and activation
  provably ignores whatever a client puts under that name
  (`tests/test_otr_check_cli.py:335` asserts a bundle whose
  `check_compatibility` is a plain integer activates). The only artifact that
  reserves the name is the `EXTENDING_OTR.md` paragraph, which exists either
  way; the constant's sole executable effect is to legalize a call nobody
  makes. Verified blast radius if ripped: ~5 code sites, 2 test files, 3 docs;
  no workflow JSON, no routing, no source-payload consumer. Case AGAINST,
  stated by both: churn on landed green code for zero behaviour change, the
  constant is loudly commented inert and a test documents the inertness, and
  the plan of record already names the future consumer (randomizer
  eligibility), so it may be re-added within a wave or two. STILL NOT A CODER
  CHUNK -- the rip touches landed wave-3/4 code and the plan of record's
  "fetch_source + interpret_source + check_compatibility" line. Either ratify
  the inert constant or schedule the rip as a planner chunk. (The one piece
  already fixed @ `8c45172d`, correct under either answer: the `missing_module`
  quarantine message demanded a `check_compatibility` the code has never
  required. Both panelists found it independently. Proposed doctrine line: a
  name published to clients before its consumer exists lives in the
  client-facing DOC as "reserved, no contract, ignored if defined" and nowhere
  in executable code, because code that names an interface is read as
  enforcing it.)
- **The ledger-cleanup pass now runs on EVERY bank, not just client banks**
  (wave 6, `3d97a130`). It is a no-op on a complete ledger and costs no LLM
  call there, but two shipped-lane behaviours did change and are worth watching
  on the next live legs: (a) unsafe spoken language on a
  `content_owned_readonly` bank is now REPAIRED at the writer tail instead of
  reaching G9 untouched, so a leg that used to die at freeze may now ship a
  sanitized line; (b) a blank `meta.episode_title` is now filled at the tail
  instead of exploding later in `otr_credits_roll`. Both are the intended
  direction under THE LAW; neither has a live receipt yet.
- Lean-mean front/tail drift: MOVED to the Lean-mean campaign section of
  `ROADMAP.md` with the block (2026-07-29). The constraint itself still holds
  wherever it runs -- the tail's SW-1 re-survey is mandatory against the
  then-current writer, and the two campaigns never share a window.
- No code lands mid-sweep of an active qualification campaign (uniform-code
  confound -- the 420-rung lesson).
- The active campaigns may surface new lane defects; the campaign window owns
  admitting PBUGs (new-bug problem-statement rule applies).
- `dynamic_story` touches the writer, the visual-style authority and the
  canonical workflow; it re-derives the live JSON at build. It is now the only
  claimant on those surfaces (extensibility has released them).
- Generated-SFX R4 stays local/ignored evidence until the tracked R4.1 refit
  lands; it is not an executable queue.

## Tombstones (do not re-derive; records in HANDOFF_LOG + PROD_BUG_LOG)

Keep-6 bank rename (six de-versioned banks; default `scifi_news`,
local/offline-first) -- LLM veto rip + THE LAW -- roster trim + Sonnet-bake-off
rip (science_news family, `_v2` lanes, scifi_sonnet retired) -- v4 improvement
campaign banks #2-#5 PARKED (superseded by the rename + THE LAW; revive only
by operator decision; plan of record `docs/2026-07-17-v4-campaign/final.md`) --
codex56sol attempt telemetry + PBUG-20260712-17 root fix -- fresh two-matrix
bakeoff -- Qwen-Image still engine (removed 2026-07-23) -- word-fit ceilings /
candidate campaigns -- style-dropdown four-surfaces -- otr-build-tracker
artifact -- `tencent/hy3:free` panel seat (expired 2026-07-21) --
**the 45-word scene matrix, the 54-case visual-style sweep, and the entire
quick-wins block (CUT by the operator 2026-07-24: coding over matrices, bugs
triaged as a batch later; ENGINE_MATRIX survived as a Lean-Mean W6 sub-step,
CODER B and CODER C dissolved with the block)** --
**independent client-authored source banks v1 (all seven waves, CODER E,
2026-07-24 @ `30358ad1`; contract `docs/EXTENDING_OTR.md`; w7 closed by
assessment -- no widget was needed and none was invented)** -- the retired
Path-A/B user-source-lanes architecture.

## Pointers

- `ROADMAP.md` (dependency edges; lean-mean pin self-declares stale cites)
- `docs/PRODUCTION_SPRINT_LESSONS.md` (incl. lesson 24 lost-anchor; 25 bank-teardown)
- `docs/SOURCE_BANK_PREFLIGHT.md` -- add-a-bank gate + the Teardown protocol
- `docs/PROD_BUG_LOG.md` / `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/HANDOFF_LOG.md` (all completed-work history, newest at top)
- `docs/2026-07-23-video-failure-inventory.md` (campaign staging record)
- `docs/2026-07-15-720-bakeoff-verdict.md` (KEEP/IMPROVE + open items)
- `docs/2026-07-17-model-bakeoff-scoreboard.md` (writer-model verdict)
- `docs/EXTENDING_OTR.md` (LANDED client contract: add your own source bank)
- `docs/2026-07-24-independent-source-banks-v1-plan.md` (extensibility plan -- DELIVERED)
- `docs/2026-07-12-user-source-lanes-architecture.md` (SUPERSEDED -- Path-A/B decision log)
- `docs/2026-07-10-lean-mean-rip-final.md` (drift-check header 2026-07-15)
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-10-llm-first-story-edit-pass.md` (X1-X4 live remainder)
- `docs/2026-07-11-announcer-framing-defect.md` (OPEN)
- `docs/2026-07-11-timeline-cue-ledger.md`
- `workflows/otr_canonical.json`
