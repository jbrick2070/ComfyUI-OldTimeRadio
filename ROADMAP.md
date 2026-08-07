# OTR Roadmap

**Updated:** 2026-08-07

**Branch:** `v2.0-alpha`

**Purpose:** ordered release runway after `docs/GO_FORWARD_PLAN.md`

This is the only roadmap. It contains future work only. Completed source-pack,
Google TTS, portability-build, and prior sprint narratives live in git history
and dated evidence documents, not in the forward queue.

## North star

Ship OTR v2 as a friendly, fail-loud ComfyUI pack that generates complete
old-time-radio episodes with coherent stories, distinct voices, story-native
visuals, captions, credits, and a final publishable video across practical local
and cloud hardware paths.

## Ordered release runway

A coder-day is one focused engineering day. Live renders, provider waits, and
operator listening are separate elapsed time.

| Order | Campaign | Exit condition | Coding estimate |
|---:|---|---|---:|
| 1 | ~~Timeline Cue Ledger / generated spotted SFX~~ **RETIRED 2026-08-06** | Operator ruling "rip out SFX 100%" executed by `docs/2026-08-06-BUILD-SPEC-rip-sfx.md`; the five SFX engines, the bed compiler and the mux mix branch are gone, ids barred via `RETIRED_ENGINE_IDS` | - |
| 2 | ~~Other product expansion~~ **EMPTIED 2026-08-07** | Two items were PROMOTED into `docs/GO_FORWARD_PLAN.md` (`visual_storybased` = its item 2, multi-GPU upscale = its item 8); two were CUT or PARKED (see below). Nothing is left in this row | - |
| 3 | Lean-mean/dead-code campaign | Re-grounded deletion/consolidation waves land green; no dormant interstitial audio or duplicate authorities | 12-16 days |
| 4 | RunPod + AMD/Mac platform tests | Representative-machine acceptance smokes actually run, not just declared available | 3-5 days |
| 5 | Install path | Clean install/bootstrap/profile smoke/log collection on representative machines | 4-6 days |
| 6 | Product docs and v2 release | First-render guide, troubleshooting, accurate README, all release gates green; operator controls tag/promotion | 2-5 days |

**FULL RUNWAY ORDER, operator 2026-08-07.** GO_FORWARD and this file are ONE
ordered runway: GO_FORWARD 1-8 (qualification -> `visual_storybased` ->
reference A/B -> WAN 8-GB -> MiniMax H3 -> video matrix -> the 23 shipped
episodes -> upscaler), then rows 3-6 here (lean-mean -> RunPod/AMD/Mac ->
install -> docs and release). Row 4 was split out of the old combined
"RunPod and install path" because the operator ordered the platform tests as
their own step.

**SFX (row 1) IS RETIRED, 2026-08-06 (operator: "I do really want to rip out
SFX 100%").** It was parked by operator doubt on 2026-08-04; two days later the
rip executed (`docs/2026-08-06-BUILD-SPEC-rip-sfx.md`): the five SFX-bed
engines are deregistered and permanently barred
(`nodes/_otr_shared/public_engines.RETIRED_ENGINE_IDS`), the bed compiler and
the mux mix branch are deleted, and `tests/test_rip_sfx_bed_guard.py` trips on
any surface creeping back. The design docs below keep in-place RETIRED headers
as the historical record. The effective runway: product expansion -> lean-mean
-> RunPod -> release.

**LEAN-MEAN MOVED FROM ORDER 1 TO ORDER 3 ON 2026-07-29 (operator), and moved
OFF `docs/GO_FORWARD_PLAN.md` entirely at the same time.** Two reasons, and
they point the same way. First, the operator's: the randomizer and SFX come
first, and lean-mean comes after them. Second, the block's own nature: it is a
deletion campaign whose entire value IS its file-and-line kill inventory, which
is the most perishable thing a plan can carry -- so it must re-ground ONCE
against the final tree instead of ripping first and re-grounding after every
later block lands on freshly-rewired code. It sits ahead of rows 4 and 5 on
purpose: validating an install path and tagging a release against a tree still
full of dead code would have to be redone after the rip.

**Everything in `docs/GO_FORWARD_PLAN.md` precedes row 2 here (row 1 is
retired).** That file carries the ON DECK continuity-correctness queue, then
WAN 8-GB, Randomizer A and `dynamic_story`; when its queue is exhausted, a
window comes here.

**Release-runway planning range after GO_FORWARD:** roughly **28-46 coder-days** (figure predates the SFX retirement; the effective range is lower with row 1 gone)
before optional product-expansion campaigns, or about **6-9 one-coder weeks**
with normal integration margin. Hardware/provider qualification can extend
calendar time without consuming equivalent coding time.

## Portability status -- already implemented

The multi-platform workflow, profile, registry, and generated-variant features
are already in place. They are not a future coding campaign.

Release QA still runs representative CPU/no-GPU, NVIDIA, cloud, AMD, and Mac
acceptance smokes where matching hardware or credentials are available. That is
validation time, not planned coding time. A platform smoke creates coding work
only if it proves a real defect that must be root-fixed. Generated workflows
remain under `workflows/variants/` and derive from
`workflows/otr_canonical.json`; the generated `docs/ENGINE_MATRIX.md` (emitted by
`tools/engine_matrix.py`, drift-gated in the suite) makes the already-built
engine/backend support visible to users.

`dynamic_story` is owned by `docs/GO_FORWARD_PLAN.md`'s standing block
order; it is not duplicated in this longer-range queue.

## 1. Timeline Cue Ledger / generated spotted SFX

Foundation plan: `docs/2026-07-11-timeline-cue-ledger.md`.

Generated-SFX architecture evidence:
`docs/2026-07-11-sfx-engine-architecture/roundtable/pass04_final.md`.

**Status: RETIRED 2026-08-06 (operator ruling "rip out SFX 100%"; executed by
`docs/2026-08-06-BUILD-SPEC-rip-sfx.md`; see the note above the lean-mean
paragraph).** The paragraphs below are the historical design record only, not
a queue and not evidence awaiting revival. Reviving SFX would be a NEW design
against the post-rip tree -- the code these designs cite (the bed compiler,
the manifest fields, the five provider engines) no longer exists.
(Superseded status lines: PARKED by operator doubt 2026-08-04; before that,
implementation parked until the 720-word runway.)
**This campaign now runs BEFORE lean-mean, not after it** -- the 2026-07-29
operator direction reversed that dependency, so an older line reading "parked
until the lean-mean campaign lands" no longer holds anywhere.

The two dated designs are not separate implementation queues. Timeline Cue
Ledger owns C0/C1 and the blind usefulness gate. If that gate passes, the
generated-SFX design owns the renderer/mix/canonical continuation and
**supersedes the older curated-CC0-library renderer; no library fallback
survives**. SFX remains a derived cue artifact after spoken performance exists;
never resurrect `speaker_role="sfx"`, pseudo-dialogue cues, or the retired
pre-render subsystem.

Planned sequence:

1. **C0:** persist per-line WAV stems and ship the transcript drift report.
2. **C1:** define `cue-1`, build the timestamp-blind intent/placement passes,
   and run the budget-matched blind noun-detector control. Indistinguishable
   results stop the campaign before any audio engine work.
3. **R4.1/C2:** after a passing gate, ratify the current generated-SFX plan:
   ungated static selection between Stable Audio 3 Small-SFX and Medium;
   selected-profile hard failures with no fallback; ledger-bound semantic cue
   authoring; complete request/attempt/commit receipts; onset-exact generated
   stems written directly to canonical episode paths.
4. **C3a-C3c:** mix inside SceneSequencer at the raw scene-audio seam, preserve
   room tone, rebase/persist final ledger timing before `audio_done`, retire the
   accidental Google-video byproduct bed and any double-mix path, and wire the
   canonical workflow in the same activation commit. There is no post-video or
   Whisper/alignment lane.
5. **C4 later:** add VFX rows only after the SFX lane is green; VFX genuinely
   needs a post-video picture-aware pass.

Estimates void -- the campaign is RETIRED 2026-08-06; the Status line above
governs. No C0 promotion path exists on the post-rip tree.

## 2. Product expansion candidates

Choose these as separate campaigns after the core surface is stable:

- a non-gating episode content rating on the opening sequence and repeated in
  the credits, derived from the final canonical episode on a G-through-XXX
  scale; a future design may combine transcript/audio analysis with video-frame
  moderation, but the rating is advisory display metadata only and must never
  reject, reroll, rewrite, or block publication;
- ~~story-fitted visual style~~ **PROMOTED OUT OF THIS ROW 2026-08-07.** It is
  now `visual_storybased`, queue item 2 in `docs/GO_FORWARD_PLAN.md`, starting
  when the announcer-intro qualification ladder closes. The full spec lives
  there and is NOT duplicated here (same rule as `dynamic_story`). One line so
  this row is not misread: the ask is a dynamic visual style whose parameters
  the LLM decides from the story -- no presets, no seeds -- shipped as a TENTH
  dropdown entry peer to `anime` and `paper_origami`, with the existing nine
  packs and the seeded roll kept as the fail-closed floor;
- **richer brief-driven music-cue still prompts -- PARKED 2026-08-07**
  (operator: "richer cue stills, I dunno, do we need"). Driver's read, recorded
  so the question is not re-asked cold: it is a polish pass on the same visual
  surface `visual_storybased` is about to change underneath, so judging it now
  means judging a gap that may not survive that campaign. Revisit AFTER
  `visual_storybased` lands and see whether music beats still look generic;
- **CLOUD MUSIC LANE -- the one real provider gap (operator 2026-08-07).**
  The driver first CUT this whole bullet as scope creep against `CLAUDE.md`'s
  "100% local, offline-first, no cloud services" line. **That was wrong and the
  operator corrected it:** the pack ALREADY ships cloud lanes and treats them as
  opt-in rather than forbidden -- ElevenLabs TTS for voice, and OpenRouter /
  Comfy / Google slots for image and text. The local-only rule is about the
  DEFAULT path staying free and offline, not about refusing to have a cloud arm.
  **Verified 2026-08-07: music is the one modality with NO cloud option.**
  `nodes/stable_audio_theme.py` (`StableAudioTheme`) is the only music
  generator, and a tree-wide search for Lyria / Suno / Udio / any music API
  returns nothing. So voice has a cloud arm, images have a cloud arm, and music
  has none -- an asymmetry, not scope creep.
  Not scheduled and not scoped yet. When it is picked up, the existing provider
  rules travel with it: provider identity stays EXPLICIT, the lane FAILS LOUD
  rather than degrading silently, and local stays the default. Adding OTHER
  provider lanes beyond music is NOT wanted -- the operator's "provider
  additions, I don't think we need" stands for everything except this gap;
- **richer brief-driven music-cue still prompts -- PARKED 2026-08-07**
  (operator: "richer cue stills, I dunno, do we need"). Driver's read, recorded
  so the question is not re-asked cold: it is a polish pass on the same visual
  surface `visual_storybased` is about to change underneath, so judging it now
  means judging a gap that may not survive that campaign. Revisit AFTER
  `visual_storybased` lands and see whether music beats still look generic;
- ~~direct BYO Google music/image/video paths, followed by other providers~~
  **CUT 2026-08-07 (operator: "provider additions I don't think we need").**
  It also pulls against the pack's own scope rule in `CLAUDE.md` -- "100% local,
  open source, offline-first. No cloud services, no API keys, no paid services"
  -- so a campaign to ADD provider lanes works against the thing that makes this
  shippable to a stranger with no accounts. The cloud slots that already exist
  stay as opt-in bake-off arms; this was about building MORE of them, and it is
  not happening;
- ~~a new system-agnostic multi-GPU upscale stage~~ **PROMOTED 2026-08-07** to
  `docs/GO_FORWARD_PLAN.md` queue item 8, after the 23-episode disposition. The
  constraint travels with it: built against the profile and registry contracts,
  NEVER a resurrection of the retired NVIDIA-only node;
- a one-shot continuous multi-speaker segment role (podcast/trailer banter
  between two announcers, rendered as a single unbroken take). VibeVoice is the
  candidate model and has maintained ComfyUI nodes, but it is NOT a `char_voice`
  engine and must never be wired as one: the audio spine is per-line
  (`interface == "per_line"` -> `pack_audio_batch` -> SceneSequencer unbind), and
  VibeVoice's whole advantage is cross-turn prosody that only exists because it
  never cuts at line boundaries. Running it per-line discards the advantage;
  running it conversation-mode requires forced alignment to recover per-line
  timings, which is a new drift source in the exact place the ledger rule
  forbids a hole. Also weigh: a fourth isolated sidecar venv, 4 speakers max per
  generation, and a `commercial_clean` value that is genuinely unclear because
  Microsoft withdrew the upstream repo and weights (MIT code, community mirrors).
  Deferred by the operator 2026-08-06 -- "no for now"; revisit only as this
  distinct role, never as a voice-engine swap.

Each candidate gets its own scoped design, exact ownership table, tests, and
qualification ladder before entering GO_FORWARD.

## 3. Lean-mean campaign

**Moved here in full from `docs/GO_FORWARD_PLAN.md` on 2026-07-29 by operator
direction.** It is not cancelled, deferred-forever, or reduced -- it runs after
the randomizer and SFX, and this section is now its only home. GO_FORWARD's
CODER D and CODER G rows are struck through and their gates voided; do not
re-add a lean-mean row there.

Plan of record: `docs/2026-07-10-lean-mean-rip-final.md`, D-1..D-6 RATIFIED --
but its line pins are stale and must be re-grounded once after feature surfaces
stop moving. Its own header already declares five stale areas, and since it was
written the extensibility build added modules, moved the writer tail, and grew
the suite past 6,400.

### Re-ground gate -- OPERATOR PIN

**Both halves run a FULL `r2 -> r3 -> r4` arc, not the r3 default.** Pinned by
the operator 2026-07-24; their doubt is already settled and a later window must
not re-argue them down to r3 to save a round. The reason is the block's nature:
the question is not "do the line numbers still point at the right code", it is
"is this still the right code to delete" -- and that is an r2 question.

Panel: Claude codes and judges; kibitz = codex `gpt-5.6-sol` high + agy. Fable
gets a single final gate on a lean-mean epoch commit and nothing else (the
CLAUDE.md section-9 reality exception). Roughly six panel rounds of Codex spend
across the two arcs -- front-load them early in a credit week.

**The arc is the window's first job, not a formality: if r2 says the kill list
is wrong, the window's output is a NEW r2, not a rip.** No deletion before r4
converges at current HEAD.

Drift-check items fold into the r2 brief: SW-3 `news_ingest` re-survey, W6
keep-list adds, W7 tombstone re-triage, R-7 re-grep. SW-1's writer re-survey
belongs to the TAIL arc, against the then-current writer.

### Dependency order

Execute the ratified order, one pushed green chunk at a time:

`W0 -> W1 -> W2 -> W3 -> W4a -> W4b -> W7 -> W6 -> W5+SW4 -> C1-C5` (FRONT),
then `SW1/SW2/SW3 -> C6 -> C7 -> W8` (TAIL).

Required edges:

- W1 before SW3;
- W2 before C2;
- W2-W4 before W7;
- generated ENGINE_MATRIX before registry deletion/consolidation;
- cloud smoke before W8;
- user extensibility, Randomizer, and `dynamic_story` before the writer/widget
  structural split, preserving overlay quarantine, pack replay hashes, and the
  final live widget layout.

**Run the TAIL arc when the TAIL opens, not earlier.** Every block ahead of it
edits the very writer it then splits, so an arc run today would ground against
a writer that will not exist by the time it executes -- worse than not running
it, because it produces a confident stale plan.

Do not re-delete the old context helper already removed at `1a6ae8f1`, and mark
the earlier standalone interstitial-audio rip as satisfied when re-grounding.

### W2 carries a MANDATORY first chunk, and it is a MIGRATION

Record: `docs/2026-07-25-dormant-3d-rip-judgment.md` (2026-07-25 consult, codex
`gpt-5.6-sol` high + Claude judge). The operator asked whether the dormant 3D
talkers should be ripped; the answer is YES and W2 already said so ("delete,
NOT keep-dark"), so nothing was re-litigated. **But a LIVE fail-closed guard is
hiding inside the dormant code:** `otr_image_director._is_3d_engine:109-119`
raises for ANY non-empty UNREGISTERED engine (covered at
`tests/test_image_platform_c1.py:339-352`), and neither OTR_VideoDirector nor
the route freeze validates registry membership -- so deleting the 3D lock path
would silently delete a live protection.

**W2 chunk 1 is therefore a MIGRATION** of that validation to the
VideoDirector / route-freeze boundary, green and pushed on its own, BEFORE any
deletion. Also settled there: `triposr` goes as an unimplemented scaffold (it
never declared `requires_mesh_portrait`); the live mesh lane (`mesh_stage`,
`requires_mesh_fodder`, `directory_clip`, SilentComposite, `portrait_ledger`)
is NOT in scope; and W2 must pick its BOUNDARY explicitly -- adapters only, or
full lane retirement including the zero-declarer capability and the
`character_3d` family contract.

### ENGINE_MATRIX is a W6 SUB-STEP, not a separate chunk

`docs/ENGINE_MATRIX.md` -- ALREADY GENERATED AND COMMITTED (emitted by `tools/engine_matrix.py`; `--check` drift-gates it in the suite), so W6 only owes the README link. Original design note: emit from the three live CAPABILITIES registries per
the existing generator pattern (`build_variants.py` ~:276-338): write during
`--all` / explicit emit; `--check` regenerates in memory and FAILS on drift
without writing. Columns + stable ordering; link from README. The lean-mean doc
(`:301-304`) only needs W6's README policy line to link it, so this is an
ordering preference the operator set on 2026-07-10 -- **NOT a hard technical
dependency.** W6 executes without it; the README link is what suffers.
Estimate 0.5-1 d. (This is the one item that survived the 2026-07-24 quick-wins
cut, folded in here rather than kept standalone; CODER B and CODER C dissolved
with that block.)

### Shape

Dedicated window, multi-day, one campaign at a time. **Never interleave the
FRONT and TAIL campaigns in one window.** No code lands mid-sweep of an active
qualification campaign (the uniform-code confound -- the 420-rung lesson).

## 4. RunPod and installation

Build deployment and installation only from proven profiles:

- RunPod bootstrap/start/environment-check/smoke/log collection;
- Windows and Linux installers;
- deterministic model/download and install verification;
- `INSTALL.md`, `FIRST_RENDER.md`, `TROUBLESHOOTING.md`, and machine-profile
  guidance that match the generated workflows and engine matrix.

The install path must answer which workflow to load, which models or keys are
needed, what costs money, what stays local, and how to act on every loud failure.

## 5. README and v2 release

Feature recipes update README when their features land. This section is the
final README accuracy and product-release pass, after the workflows, installers,
and hardware claims are real. Ship v2 only when:

- canonical and generated workflows validate;
- representative local, cloud, and RunPod paths complete tiny episodes;
- source/story/visual lanes have current qualification receipts;
- full repo suite and Bug Bible are green;
- install and first-render docs match observed behavior;
- pending production-proven Bible candidates are resolved or explicitly held;
- the operator has completed the final listen/eyeball and authorizes the tag or
  promotion.

Pushes to `v2.0-alpha` remain normal green-chunk workflow. Tags and promotions
remain operator-gated.

## References

- `docs/GO_FORWARD_PLAN.md`
- `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/2026-07-10-lean-mean-rip-final.md`
- `docs/2026-07-11-timeline-cue-ledger.md`
- `docs/2026-07-11-sfx-engine-architecture/roundtable/pass04_final.md`
- `workflows/otr_canonical.json`
- `workflows/variants/`
