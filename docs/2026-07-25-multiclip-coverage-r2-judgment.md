# MULTI-CLIP COVERAGE -- r2 judgment (coding plan)

**Runs:** `kibitz-runs/2026-07-25-multiclip-coverage/r2/` (codex
`gpt-5.6-sol` high) and `kibitz-runs/2026-07-25-multiclip-coverage-agy/r2/`
(agy `Gemini 3.6 Flash (High)`), both pins verified, run independently.
Code baseline `a1d810f1`. Claude is the grounded panelist and sole judge.

**Both seats: VERDICT no.** Both accept the r1 architecture; both say the r2
brief was not yet an executable plan. Agreed.

## 1. THE SPLIT THAT MATTERS: how do we cut at phrases?

The operator approved phrase-based cutting for audio lanes. The panel split
hard on whether that is buildable.

- **codex: phrase boundaries DO NOT EXIST in the ledger.** `_line_index`,
  `_cumulative_beat_start` and the master slicer give LINE and BEAT bounds
  only (`render_driver.py:794`, `:1460`, `:326`). Build a frozen `PauseMap`
  from **voice-only line audio** via a deterministic alignment/VAD pass BEFORE
  master mixing, with sample offsets, confidence, source hash and algorithm
  version.
- **agy: CUT pause detection entirely.** Music and SFX beds break silence
  detection; acoustic pauses do not align with `8n+1` / `4n+1` frame quanta
  (a 3.12s pause is 78 frames -- illegal for both); and continuous speech
  longer than the cap has no fallback. Partition strictly by legal quantum.

**JUDGE CALL -- neither, quite. Phrase-awareness becomes a PREFERENCE OVER a
quantum-legal partition, not a constraint on it.**

Reasoning, and why each seat is half right:

1. **agy's music/SFX objection dies against codex's source choice.** agy
   assumes detection runs on the mixed master. codex derives the pause map
   from the **voice-only per-line audio, before master mixing** -- those stems
   are clean by construction. The objection is real about the wrong input.
2. **agy's arithmetic objection is real and survives.** A cut must land on a
   legal quantum boundary or the exact-sum requirement breaks. So the pause
   map must NOT choose cut points.
3. **Therefore: the partitioner enumerates the LEGAL cut points from the
   adapter's own frame contract, and the pause map only RANKS them.** Choose
   the legal boundary nearest a detected pause. No pause map, no nearby pause,
   or a phrase longer than the cap -> take the plain quantum cut, which is
   deterministic and honest.
4. This keeps the operator's intent ("split where phrase, not arbitrary")
   while making the audio analysis a QUALITY input that can be absent without
   breaking correctness. A cut 3 frames from a breath is perceptually a cut at
   the breath; a cut that breaks exact-sum arithmetic is a defect.

**Consequence for scope:** the pause map is a SEPARATE, optional, later chunk.
The first vertical slice partitions by quantum alone and is still correct.
That materially de-risks the block -- neither seat proposed this, and it is
the difference between shipping and blocking on a DSP pass.

## 2. Second split: one prompt hook or two?

- codex: TWO -- `build_jump_still_prompt` (frozen before minting) and
  `build_segment_prompt` (may depend on the prior clip's terminal artifact).
- agy: ONE -- `build_clip_prompt(request, clip_index, total_clips)`.

**JUDGE CALL: codex, two hooks.** The decisive argument is timing, not taste:
a still prompt MUST be frozen before the image phase mints anything, while a
CHAIN segment prompt can legitimately depend on an artifact that does not
exist until the previous clip renders. One hook would have to be called at two
different phases with two different contracts, which is how a seam rots.
agy's redundancy concern is answered by both hooks sharing one typed context.

codex is also right that `StillPlanRow.framing_geometry`
(`still_plan_helpers.py:141`) must NOT be repurposed as the execution hook --
it is stored authored text.

## 3. Convergence (adopted without argument)

1. **The `VideoEngine` Protocol has no signatures for any of this**
   (`registry.py:51-98`). Both. Add `frame_contract()` and the continuity
   capability to the Protocol; define a frozen `FrameContract`
   (`min_frames`, `max_frames`, `quantum`, `discrete_durations`,
   `allow_tail_trim`).
2. **The import-audit blindspot.** Both, independently: adapter imports are
   swallowed (`__init__.py:16-44`), so a broken adapter never registers and a
   POST-REGISTRATION audit cannot see the hole. Compare an independent
   expected roster (`registry.py:253` `CAPABILITIES`) against registered
   names; fatal in CI, quarantine in production.
3. **`prepare()` does not mean "weights loaded once"** (codex, sharper):
   `ltx_8gb.prepare()` only resolves node classes; the loaders live INSIDE the
   per-clip graph (`eng_ltx_8gb.py:328`, `:370`, `:408`). A naive hoist would
   still reload per clip and the feature would cost more than it delivers.
   Need a beat-session interface returning reusable model/CLIP/VAE handles,
   and the test must assert LOADER-CALL COUNT, not prepare-call count.
4. **Teardown in `finally`, lease always released** (agy explicit, codex
   equivalent). Hoisting prepare to beat scope leaks the lease on a mid-beat
   failure otherwise.
5. **Terminal-frame persistence must be SYNCHRONOUS AND FATAL.** Both.
   `persist_episode_clips` is best-effort today (`render_driver.py:3024-3035`,
   `except: pass`), which would let clip k+1 chain from a stale or missing
   frame.
6. **CHAIN successors' `init_image` must be a DEFERRED token**, not a path
   validated up front (agy, grounded at `render_driver.py:1495-1508` and
   `_assert_family_inputs_satisfiable` at `:2439`) -- otherwise the beat
   crashes on a missing file before clip 0 even renders.
7. **Discrete-duration lanes (Veo) need `allow_tail_trim`.** Both: 4/6/8s
   durations cannot sum to an arbitrary beat, so render the smallest covering
   duration and trim exactly at canonicalization.
8. **CHAIN seam arithmetic** (codex, and it is subtle): with `8n+1` contracts
   the successor's first frame DUPLICATES the predecessor's terminal frame if
   both are concatenated whole. The partitioner must solve
   `sum(render - drop_head - trim_tail) == target_visible_frames`, and trims
   must be applied BEFORE extracting the terminal frame, or continuity binds
   to a frame that is not in the assembled video.
9. **CUT the ExecutionGroup expansion.** Both, for the third round running.
10. **No new ComfyUI node** for splitting or assembly (codex) -- pure internal
    helpers keep the canonical workflow untouched.
11. **Cloud resume must not double-bill** (codex): Veo starts a prediction and
    only then polls (`eng_google_veo_video.py:370`, `:530`); persist the
    operation name under the segment transaction and reattach on resume.
12. **Colorspace on terminal-frame extraction** (agy): pin the bt709 matrix
    explicitly or chained stills drift in brightness/chroma over a long chain.

## 4. Ordering of record (revised for the pause-map deferral)

1. **Declaration surface + audit** -- `FrameContract`, continuity token, the
   Protocol signatures, the roster-vs-registered audit. No behaviour change;
   every adapter marked `single_only` until proven.
2. **Partitioner + CoveragePlan** -- pure, quantum-only, exact-sum, seam
   arithmetic with head/tail trims. CPU-provable in full, including
   property-based sweeps over beat lengths (agy's should-fix 2).
3. **Beat-session lifecycle** -- one load, N segments, teardown in `finally`,
   asserted by loader-call count.
4. **Transactional persistence + assembly** -- `.partial` -> validate -> hash
   -> atomic rename; ffprobe proves `frame_count == target_visible_frames`.
5. **`ltx_8gb` live slice** -- one beat over 161 frames, >= 2 forward-only
   clips, one heavy load, no ping-pong, `RESULT SUCCESS` + `obs_publish OK`.
6. **THEN** the pause map, as a quality ranking layer over already-correct
   partitioning.
7. Then further adapters by capability, audio lanes last.

## 5. Open for r3 (wiring)

- Whether the frozen render-mode options (resolution / reference mode for Veo,
  env ceilings for LTX/Wan) can be captured at the route lock that already
  landed (`57f4983a`) or need their own capture point.
- `profile_max_render_frames()` treats 0 as unlimited
  (`motion_common.py:442`); codex wants an absent ceiling REJECTED. Confirm
  that does not break the 8-GB WAN contract shipped at `f914f0a4`.
- The exact deferred-token spelling for a CHAIN `init_image` and every
  validator that must learn it.
