# PASS 07 REVIEW FOCUS: PRE-MORTEM / RED TEAM

You are one panelist. THIS pass is the PRE-MORTEM: assume it is six weeks
from now and the LTX-AV lane SHIPPED AND THEN FAILED PAINFULLY in
production. Work backwards: what killed it? Pass01-06 are LOCKED -- your
job is not to redesign but to find the failure the plan does not yet
survive, and the cheapest mitigation that fits the locked design.

Rank the kill-list (most likely x most damaging first). For each: the
failure story in one sentence, the earliest SIGNAL (log line / gate /
test that would have caught it), and the cheapest MITIGATION consistent
with pass01-06. Cover AT LEAST these candidates plus any you invent:

1. FALLBACK STORM: operator sets OTR_ENABLE_LTX_AV=1 but weights are
   absent/wrong-path on the box -> EVERY talk/music beat walks the
   chain every episode -> humo-heavy episodes, double render cost,
   nobody notices for a week because episodes still complete (the
   "never aborts" property hides it). What is the storm DETECTOR (e.g.
   N degrades of the same origin engine in one episode -> one screaming
   summary line / tracker count)?
2. RELOAD THRASH: consecutive same-engine clips re-load the 13-16 GiB
   transformer + 13.2 GB encoder PER CLIP (free_after_use semantics?)
   -> 3-clip episode pays 3x model load. Does the batch order clips by
   engine? Is keep-resident-across-consecutive-clips safe under the
   AS-3 lease, and who decides (wrapper_bridge policy vs adapter)?
   Find the existing behavior in the grounding and name the cheapest
   v1 answer (even if it is "accept the reload, record the cost in
   M0").
3. CANCEL MID-SAMPLE: operator cancels in Comfy Desktop during the
   transformer phase -> executor-thread state, lease held, VRAM
   resident, next render starts on a poisoned GPU. What does the
   existing machinery do (teardown finally-blocks? lease timeout?) and
   what is the v1 discipline (e.g. always-restart-after-cancel rule in
   the operator docs vs code)?
4. PARTIAL/CORRUPT DOWNLOADS + broken symlinks (HF resume, the
   cache+symlink pattern): earliest signal and cheapest gate (file
   size/hash check in assert_usable weights probe? M0 inventory only?).
5. MODULE-CACHE STALENESS: new adapter code on disk, Comfy Desktop
   still running old module -> dropdown shows engines but render uses
   stale code / engines missing entirely. Restart discipline per build
   (Desktop needs RESTART; headless boots fresh) -- where is it
   WRITTEN so the operator hits it (adapter docstring? M0 checklist?
   error message?)?
6. GPU CONTENTION with the ACTIVE acceptance-test window: M0 runs
   while the 30w acceptance render is live -> both fail mysteriously.
   The plan says "after the acceptance window" -- is a schedule note
   enough, or does M0's launcher check :8000 liveness first (the soak
   launcher pattern)?
7. NVML UNAVAILABLE: nvml_available() False on some driver state ->
   the 14.5 ceiling silently unenforced. Fail-open or fail-closed for
   THIS lane (the heaviest engine yet)?
8. PAD-TAIL ABUSE: a systematic timing bug upstream makes every beat
   exceed the cap -> every clip is 19.9s render + frozen tail; the
   per-clip LOUD line exists, but what aggregates it (same storm
   detector?)?
9. CAPTIONS/CREDITS/TIMELINE: pad-tail and trimmed clips join the
   compositor timeline + node-93 caption ledger + credits-tail cap
   (MASTER-WAV duration). Any interaction where a padded clip shifts
   captions or the credits gate? Name the M4 grep that proves none.
10. GOLDEN-FIXTURE ROT: the dark-lane goldens break on every unrelated
    driver change -> developers update them mechanically -> the guard
    is dead. Mitigation (scope the golden to the fields that matter?
    regenerate-with-review policy?).
11. SLICE-CACHE STALENESS: cache key is (start,dur,path) -- a re-run
    after a master re-render reuses stale slices (path unchanged).
    Cheapest key fix (mtime+size) and where.
12. DESKTOP NODE LAG (#13194/#13308): Desktop build lacks
    LTXVReferenceAudio while headless has it (or vice versa) ->
    operator look-QA renders differ from production renders. M0
    records both; what is the RUNTIME guard (assert_usable node gate
    runs per-process, so each build self-gates -- confirm that is
    sufficient)?

Rules: cite grounding or VERIFY-AT-BUILD; mitigations must be additive
and consistent with the locked design; prefer signals that land in
EXISTING grep surfaces (swap-log, ledger, tracker). Output: RANKED
numbered list (failure -> signal -> mitigation), then SHOULD-CONSIDER,
then OPEN-QUESTIONS. Terse.


# LTX-AV lane -- sprint plan after pass06 (all but pre-mortem/finishing LOCKED)

> Campaign docs/2026-06-10-ltx-av-lane/. Judgments pass01-06. Remaining:
> pre-mortem, finishing. NO production code this window.

## Mission

Additive audio-conditioned LTX-2.3 lane: `ltx_av_talk` (announcer_visual,
character_video; I2V lip-sync from the FLUX still + per-beat audio) and
`ltx_av_music` (music_visual; audio-reactive scene motion). All shipped
engines untouched.

## ARCHITECTURE (pass01) -- summary

eng_ltx_av.py: shared core + 2 thin MotionEngineBase adapters.
ltx_av_talk (announcer_visual, character_video; audio_driven_face;
required text_prompt+audio_ref+init_image; fallback humo -> humo_1.7B ->
latentsync -> still_kenburns; degrade aspect change = LOUD policy).
ltx_av_music (music_visual; audio_conditioned_video NEW; required
text_prompt+audio_ref; fallback ltx_video -> still_kenburns). Both dark;
ONE flag OTR_ENABLE_LTX_AV (usability; registration is import-time);
IN_PROCESS; fps 25; Sage gate; AS-3 lease; BUG-291 reclaim; V-12;
executor-thread. assert_usable: flag -> Sage -> node gate (MISSING_MODEL
naming the class; six-reason enum pinned) -> weights (message names the
exact artifact: encoder vs transformer vs projection vs VAE) -> av_dims
on template (None tolerated). STOP rule: pip-freeze sandwich + nodes
resolve, else SIDECAR_REQUIRED + STOP. Config envs: OTR_LTX_AV_CKPT,
OTR_LTX_AV_TEXT_ENCODER (humo pattern).

## I/O CONTRACTS (pass02) -- summary

_ref_path(request.audio_ref) tolerant; asset_refs["init_image"];
base_clip_ref ignored; talk fails closed pre-render on missing inputs.
Core ffmpeg-normalizes audio to s16le/44.1k/mono (episode tmp).
T = timing.target_frame_count; next_8n1 snap-UP (legacy :281 snaps DOWN);
render = min(next_8n1(T), LTX_AV_MAX_FRAMES [M0; init 497]); trim to T /
pad-by-last-frame, LOUD "[ltx_av] pad-tail rendered=<n> target=<T>" >2s.
Graph ends at video VAEDecode -> IMAGE batch ->
encode_frames_to_silent_mp4 (-an); joint-AV strip + fake-AV test;
CanonicalClip silent/bt709/yuv420p/fps25/int frames + engine_id/family
stamps; ffprobe zero audio streams. canvas = request.canvas; av_dims
RAISES w/ nearest-valid (#347 doc-error note). init image: in-graph
ImageScale+crop COVER+center-crop; pad+outpaint = M0 cell.

## PROMPTS (pass03) -- summary

Adapter-thin (text_prompt/negative_prompt only; AST-tested no brief
imports). Driver: ltx_av_music JOINS the :418 tuple verbatim (radio
override honored); ltx_av_talk SIBLING branch, NO radio override,
fallback = character_description-or-"a 1940s radio announcer" + "head
and shoulders at a period microphone" (no speech verbs) via
finish_visual_prompt(240, style_tail=False) + no-text clause; forbidden:
dialogue/stage directions/vocatives/captions. NEGATIVE =
_LTX_DEFAULT_NEGATIVE verbatim (extension only on M0 inert evidence,
music-only). No music tail v1 (M0 cell). Cap 240.

## WIRING (pass04) -- summary

Universal master-slice already feeds line-backed beats; ONE delta:
line-less shots slice from SHOT synthetic timing iff engine_id ==
ltx_av_music. Hash/seed safety proven by tests (ShotLock-stamped hash,
_seed_from_hash). _render_one passes request_template=request (TypeError
guard). ENGINE_FAMILY += both; SYNTH_FALLBACKS += both (belt-and-braces).
Flag-off = RENDER-TIME degrade (ShotLock never asserts; registry
docstring corrected). Force map role-guarded (LOUD-ignore), never
bypasses asserts; M4 smoke env documented. Announcer portrait alias
(ltx_av_talk-gated; object id VERIFY). Identity: post-restamp engine_id,
trail keeps origin; greps = format_swap_log + manifest engine_id +
pad-tail marker. No group pruning (no providers).

## TESTING (pass05) -- summary

tests/test_av_dims.py (pure unit incl. snap-up + cap cases);
tests/test_video_ltx_av.py (mirrors test_video_humo.py: registration/
dark/role-fit/schema/order/extraction/canonicalize/identity/fake-AV/
pad-marker/AST/cold-import/ascii/chains/SYNTH membership; node gate via
mocked NODE_CLASS_MAPPINGS; request_template=None tolerated);
tests/test_ltx_av_driver_wiring.py (dark-lane GOLDEN FIXTURES; flag-off
degrade; force guard; alias; synthetic slice gating; template
pass-through; ENGINE_FAMILY; canvas; prompt gates). Fallout: retry
-taxonomy sweep gains both chains; exact enumerations -> membership;
b7 auto-covers (loop var `imp` if edited). Forgot-it matrix: every edit
-> named failing test. Byte-identical: CPU structural; DEDICATED
forced-lane master-hash = M4 GPU. Pytest = no network/CUDA/weights/
forwards. M0 sheet checked in; parser test post-M0; LTX_AV_MAX_FRAMES
== sheet (drift test). Bug Bible: BUG-070/291 pins; new dims row at
ship (Three-File Contract).

## HARDWARE (LOCKED pass06)

JUDGE-VERIFIED sizes (HF API 2026-06-10): Kijai 22B distilled-1.1
fp8_scaled 23.5 GiB; QuantStack distilled GGUF Q2_K 11.6 / Q3_K_S 13.0 /
Q3_K_M 13.7 / Q4_K_S 15.6 / Q4_K_M 16.5 / Q5_K_S 17.3 / Q5_K_M 18.1 /
Q6_K 19.6 / Q8_0 23.7 GiB; gemma_3_12B_it_fp8_scaled encoder 13.2 GB;
audio VAE 365 MB; video VAE 1.45 GB; taeltx2_3 23.5 MB; text_projection
2.3 GB; dynamic LoRA 2.7 GB; NVFP4 dev 21.7 GB.

- DEAD-ENDS: full residency under 14500 MB NVML is DEAD for Q4_K_S/
  Q4_K_M/Q5+/fp8_scaled -- offload/block-swap rows only. Full-resident
  candidates: Q3_K_S (~1.5 GiB headroom) and Q3_K_M (BORDERLINE).
  Total NVML decides, never file size.
- M0 TABLE columns: lane / artifact / file GiB / aux in phase / encoder
  placement / offload setting / NVML idle-preload-peak-sustained-post
  (MiB, probe_used_mb + mid-render assert_vram_within_ceiling) / wall
  (1472x832 x ~6s) / frames / quality vs 2B (A/B) / PASS-WARN-FAIL /
  notes. Rows: Q3_K_S resident, Q3_K_M resident, Q4_K_M offloaded,
  L1 fp8 block-swap; optional taeltx-vs-full-VAE cell; FLUX-ordering
  verification row (lease released + below-ceiling before video).
- PASS BARS: NVML peak+sustained <= 14500 MB; wall <= 10 min/clip
  PASS, 10-15 WARN (ship-able opt-in, documented), > 15 FAIL (parked).
  Episode (~3 clips/30w): <= 30 min PASS; > 45 min dead. Quality >= 2B.
- ENCODER PHASING (inside ONE lease; never release between phases):
  acquire AS-3 -> text encode -> reclaim_idle_models("ltx_av
  text-encode phase") [_soft_free insufficient] -> load transformer ->
  sample -> decode -> teardown reclaim -> release +
  wait_until_below_mb(14500). M0 measures GPU-encode-then-reclaim AND
  CPU-offloaded encode; v1 default = the passing mode (prefer GPU
  encode); GGUF Q3 encoder = contingency row only.
- SYSTEM RAM: sheet records RAM, pagefile, peak commit + working set
  per lane; paging -> wall blowup (wall gate catches); pre-M0: RAM >=
  32 GB required for any block-swap row; disk-free check before pulls.
- L3 NVFP4: CUT from M0 (dev-only steps, 21.7 GB, #11864); stretch
  column for later; never gates the decision.
- Two-stage: base-only v1 CONFIRMED. FLUX: sequential, verified by an
  M0 row, no co-residency.
- download_ltx_2_3.ps1 disk note bumped >= 24 GiB; GGUF/encoder pulls
  reuse the cache+symlink pattern. ComfyUI-GGUF pack presence = M0
  inventory row (Manager-installed custom pack, not pip; pip-freeze
  sandwich still binds; if it can't hold, L2 is blocked and
  Q3-resident via the native loader path or L1 is the lane).

## Additive touch list (consolidated)

- NEW  nodes/_otr_video_engines/eng_ltx_av.py
- NEW  nodes/_otr_shared/av_dims.py
- NEW  tests/test_av_dims.py, tests/test_video_ltx_av.py,
       tests/test_ltx_av_driver_wiring.py (+ fixtures/ltx_av_dark/)
- EDIT nodes/_otr_video_engines/__init__.py; schemas.py; registry.py
       (docstrings); nodes/_otr_shared/role_compat.py;
       render_driver.py (deltas a-g + SYNTH_FALLBACKS)
- EDIT retry-taxonomy sweep + any exact-enumeration tests found
- EDIT scripts/download_ltx_2_3.ps1 (disk note) + sibling pull lines
       for GGUF/encoder as needed
- NEW  docs/2026-06-10-ltx-av-lane/M0_RESULTS.md (sheet)
- Docs/tracker + Bug Bible row at ship.

## Claims ledger / misreads -- see pass04/05 judgments (cumulative)

## Milestones

- M0 PROBE (GPU evening, AFTER the acceptance-test window; never
  concurrent with it): disk+pack inventory; node presence (Desktop +
  headless); pip-freeze sandwich; scratch IA2V render w/ real slice;
  output-audio hash probe; the M0 TABLE (above); prompt cells; P1
  matrix -> LIPSYNC | STYLIZED | INERT; INERT everywhere closes the
  lane with a finding.
- M1 ADAPTERS (CPU): eng_ltx_av dark + av_dims + schemas/role_compat/
  __init__/registry + driver deltas + 3 test files + goldens + fallout;
  suite + Bug Bible green.
- M2 GRAPH + LANE: winning-lane graph; pre-flight; lease+phasing;
  silent encode; trim/pad; max-frames pinned to sheet (drift test).
- M3 WIRING PROOF: slot asserts; flag-off degrade; force guard; alias;
  identity/manifest tests.
- M4 GATES: full suite + Bug Bible + byte-identical + DEDICATED
  forced-lane master-hash + live 30-word smoke (flag ON + force map);
  greps (swap-log, manifest engine_id, pad-tail, NVML <= 14.5); obs
  playable AAC only.
- M5 LOOK-QA + DOCS + Bug Bible row.

## Appendix: cut lanes

Yvann-Nodes (p01). New prompt/negative envs (p03). ASPECT_CHANGE kind +
group-prune wiring (p04). New usability reason / GPU pytest / framework
(p05). NVFP4-in-M0 (p06).

## Open questions (assigned)

- pass07 PRE-MORTEM: rank + mitigate the kill-list -- OOM mid-episode;
  fallback STORM (flag on, weights absent -> every beat degrades);
  model RELOAD THRASH between consecutive same-engine clips (does
  free_after_use force a 13-16 GiB reload per clip? batch ordering?);
  partial/corrupt downloads + broken symlinks; Comfy module-cache
  staleness (restart discipline Desktop vs headless); cancel mid-sample
  (zombie VRAM, lease stuck, executor-thread state); slice-cache key
  (frozen-master mtime); captions/credits/timeline interplay with
  pad-tail clips; Desktop node lag (#13194/#13308); golden-fixture rot;
  GPU contention with the ACTIVE acceptance render (hard schedule
  rule); NVML unavailable (nvml_available False -- ceiling unenforced?).
- pass08 FINISHING: convergence verdict + coder-window tickets
  (CW-LTX-AV-1..n) + M0 operator checklist + tracker row.
