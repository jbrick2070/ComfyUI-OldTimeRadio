# ROUND r2 -- CODING PLAN: per-beat multi-clip coverage, per video path

**Repo:** ComfyUI-OldTimeRadio, `v2.0-alpha`. Code baseline `a1d810f1`; doc
baseline `3519d34c`. Suite `6454 / 27 / 1`; Bible 17; canonical `5377914B`.

r1 is JUDGED -- `docs/2026-07-25-multiclip-coverage-r1-judgment.md`. This round
produces the CODING PLAN. **Read the real files; every claim cites
`path:line`.** Do not re-litigate r1's settled decisions (section 1); DO
attack the plan.

---

## 1. SETTLED at r1 -- do not reopen

1. **ONE `ShotRow` per beat.** A beat never becomes N shots or N execution
   groups.
2. **Multi-clip is CONTAINED inside the beat's render.** The beat emits ONE
   clip with one start and one duration, so the manifest, SFX bed
   (`otr_master_audio_mux.py:170`), captions, timeline and `obs_publish` are
   untouched by construction.
3. **CUT the ExecutionGroup / provider-consumer DAG expansion.** Groups are
   per-role; clip order is intra-beat.
4. **Coverage planning is PURE and STATIC** -- fixed profile ceilings only.
   Never live VRAM (`eng_wan_ti2v.py:378-388` reads it), never mutable env.
5. **Forward only.** No mirror, no loop, no hold as a coverage mechanism on a
   moving-video lane. The `allow_mirror=False` seam already landed
   (`wrapper_bridge.py`, `eng_humo.py`, commit `a1d810f1`). Three faking
   mechanisms exist today and all three are out for these lanes: engine mirror
   (`wrapper_bridge.py:435`), composite loop-fill
   (`otr_silent_composite.py:244`), held-last-frame.
6. **CHAIN preferred, JUMP CUT acceptable, REUSE only if loop-closed,
   `still_*` lanes are one still.** Operator's ranked policy.
7. **AUDIO LANES ARE CUT AT PHRASE BOUNDARIES, not arbitrary frame counts.**
   Operator approved. This is the codebase's own parked fix -- see
   `otr_silent_composite.py:244-266`: *"The real fix is phrase-chunking
   (render the beat's correct duration so it never underruns) -- tracked as a
   follow-up."*
8. **Engine prepare/teardown happens ONCE per beat**, not per clip
   (`render_driver.py:2424-2458` currently does it per clip).
9. **First vertical slice = `ltx_8gb`** scene beats (discrete 9-frame min,
   161-frame cap, 8n+1 quantization, currently ping-pong-fills).

## 2. THE OPERATOR'S ARCHITECTURAL DIRECTIVE -- this round's central question

Operator, verbatim: *"I like the idea of per-beat architecture for true video
gen lanes, but remember EACH VIDEO PATH IS SEPARATE, so we have the same
per-beat video architecture duplicated in all video gen models but SLIGHTLY
DIFFERENT."*

He is right that the paths differ materially: `ltx_8gb` quantizes 8n+1 with a
161 cap; `wan_ti2v` quantizes 4n+1 against a live-VRAM budget; HuMo is a
capped audio-driven face; `ltx_av` caps at `_LTX_AV_MAX_FRAMES` (`:58`, 497);
Veo sells discrete 4/6/8-second durations (`eng_google_veo_video.py:245`);
`viz_*` lanes need nothing.

**The design tension you must resolve, explicitly:**

- Read literally -- each adapter re-implements partitioning, chaining and
  concatenation itself -- gives maximum per-path fidelity and **31 copies of
  every subtle bug**, which is the central-sprawl disease inverted.
- Read as "same SKELETON, per-path VALUES and HOOKS" -- one shared execution
  skeleton, each adapter declaring its own frame contract and overriding only
  what genuinely differs -- preserves "slightly different" with one place to
  fix a defect.

**THE OPERATOR HAS NOW RESOLVED THIS HIMSELF -- design to it.** Verbatim:

> "Sure -- if we can still have **each video model declare its own video
> prompts** and **reuse a phrase multi-clip beat splitter and
> putter-together for continuity**, great."

So the split of ownership is settled and is not an open question:

- **PER-ADAPTER (each video path owns it):** its own video PROMPTS, and its
  own frame-contract numbers (legal lengths, quantum, caps, continuity
  capability). This is the "slightly different" he means.
- **SHARED (one implementation, reused by every lane):** the phrase-aware
  multi-clip beat SPLITTER, and the ASSEMBLER that puts the clips back
  together for continuity.

Your plan must honour exactly that boundary. A design that pushes prompt
authorship into the shared splitter, or that duplicates the splitter/assembler
per adapter, is wrong on the operator's own instruction. Note this also lines
up with the still-plans build's separate per-engine layer-2 PROMPT HOOK -- say
whether these are the same hook or two, and if two, why.

**PROVE the shared skeleton can express the real differences.**
Walk at least `ltx_8gb`, `wan_ti2v`, HuMo, `ltx_av` and one cloud lane through
your proposed contract and show each one's "slightly different" is expressible
WITHOUT an escape hatch that lets an adapter fork the skeleton. If some lane
genuinely cannot be expressed, say so -- that is a finding, not a failure.

Name the exact declaration surface (fields, closed token sets, defaults) and
say what the post-registration audit checks so a new adapter cannot ship an
under-declared contract. Remember adapter imports are wrapped in
`try/except: pass` (`_otr_video_engines/__init__.py`) -- validation is a
POST-REGISTRATION audit, never decorator-time, or a typo silently deletes the
engine from the menu.

## 3. What the coding plan must specify

**A. The per-adapter frame contract.** Pure over frozen inputs (fps, canvas,
resolved profile, target frames). Returns legal render lengths -- discrete set
or min/max/quantum -- plus whether overshoot may be trimmed. No VRAM, no env,
no disk, no provider calls.

**B. The shared partitioner.** Where it runs (r1 says planning-time, after
audio timing exists and before stills are minted), what it emits, and
deterministic final-segment handling so segments sum to the beat exactly --
no gaps, no overlap.

**C. The phrase-chunk cut-point selector for audio lanes.** Inputs available:
the ledger's line index and `_cumulative_beat_start` (`render_driver.py:1460`,
`:1489+`), the frozen master mix. Specify how a cut point is chosen at a
speech pause, what happens when a single phrase alone exceeds the engine's
cap, and how per-clip audio slices are derived so they sum to the beat's exact
sample interval.

**D. The continuity declaration and the chain seam.** r1 adopted
`strict_first_frame | soft_reference | none`. Specify the per-engine
inventory, ONE canonical terminal-frame extractor (not per-engine -- canonical
clips already normalise format, `schemas.py:216`), where the frame is
persisted under `otr/episodes/<ep>/`, and that persistence is transactional
and fatal before publication (today it is best-effort,
`render_driver.py:3024-3032`).

**E. The still-spine interaction.** Clip 0 and every JUMP successor are
authored stills the spine validates up front; a CHAIN successor does not exist
at spine time and must be validated at the post-clip seam instead. Say exactly
how the spine distinguishes them without weakening it.

**F. The beat render lifecycle.** One prepare, N forward renders under that
lease, chain/jump between them, concatenate, one teardown, one emitted clip.
Name the failure policy at every step -- fail closed, LOUD, no silent
shortening.

**G. Test plan.** What is CPU-provable (partitioning arithmetic, cut-point
selection, contract validation, audit) versus what needs the live leg. The
suite is 6454 and every chunk runs it plus the Bug Bible.

**H. Chunk sequence.** Ordered, each independently green and pushable, each
with its acceptance. Name which chunk first produces real multi-clip video on
`ltx_8gb`.

## 4. Invariants (violating one is an automatic fail)

- THE LAW: an audit may improve a story, never fail one for length, language,
  style, visual vocabulary or quality.
- Fail closed. No shims, no fallbacks, no silent degradation.
- Never reverse or loop an audio-synced render.
- Per-adapter ownership; no central authority keyed on engine id.
- Any node/widget/link/schema change edits `workflows/otr_canonical.json` in
  the SAME commit. Unwired code is dead.
- Assets go straight to `otr\episodes\<ep>\`, final to `otr\obs\`; never tmp,
  never staged-to-move-later.
- UTF-8, no BOM, ASCII where practical, SFW.

## 5. Return format

VERDICT, then MUST-FIX, then SHOULD-FIX, then CUT. Each item: claim,
`path:line`, consequence if ignored. Attack the plan's feasibility and its
ordering, not its motivation.
