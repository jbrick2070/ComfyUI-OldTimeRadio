# LTX-AV LANE -- CONVERGED SPRINT PLAN (BUILD-READY)

> STATUS: CONVERGED 2026-06-10 after an 8-pass grounded roundtable
> (GPT-5.5 + Gemini 3.1 Pro + DeepSeek v4 panels; Claude panelist +
> sole judge; every accepted claim grounded vs HEAD 56caa5b).
> History: docs/2026-06-10-ltx-av-lane/roundtable/ (pass00..08 plans,
> judgments, raw reviews, manifests).
> PLANNER-WINDOW artifact: no production code was written. Coding
> happens in coder windows via the tickets below. The 13 unpushed
> commits (3f55ef9..56caa5b) and the acceptance test belong to a
> different window and are UNTOUCHED.

## MISSION

Add an additive, audio-conditioned LTX-2.3 lane, dark by default:
- `ltx_av_talk` -- roles (announcer_visual, character_video): true
  lip-sync attempt from the FLUX still (I2V) + the per-beat audio
  slice of the FROZEN master.
- `ltx_av_music` -- role (music_visual,): audio-reactive scene motion
  ("visuals breathe with the track"; FFT precision not expected).
The shipped `ltx_video` 2B engine and every other engine stay EXACTLY
as-is. V-1 absolute: the new lane discards LTX's audio side entirely;
only OTR_MasterAudioMux emits audio; test_audio_byte_identical stays
green at every milestone.

## LOCKED DESIGN (passes 1-7 + pass08 folds F1-F9)

### Adapters (ONE new file: nodes/_otr_video_engines/eng_ltx_av.py)

Private shared core + two thin MotionEngineBase adapters:

| | ltx_av_talk | ltx_av_music |
|---|---|---|
| roles | announcer_visual, character_video | music_visual |
| family | audio_driven_face (reused) | audio_conditioned_video (NEW) |
| required_inputs | text_prompt, audio_ref, init_image | text_prompt, audio_ref |
| fallback_engine | humo (-> humo_1.7B -> latentsync -> still_kenburns) | ltx_video (-> still_kenburns) |
| degrade note | aspect change landscape->pillarbox, LOUD documented | aspect-stable, sync lost, LOUD |

Both: default_roles () dark; ONE flag OTR_ENABLE_LTX_AV (gates
USABILITY; @register at import is unconditional so the dark lane is
dropdown-visible and fails closed at render); ISOLATION_IN_PROCESS;
target_fps 25; engine_version "1"; BUG-070 assert_sage_not_patched;
AS-3 single-resident lease; BUG-291 reclaim_idle_models (never
unload_all); V-12 lazy heavy imports; heavy forward in the EXECUTOR
THREAD. Config envs: OTR_LTX_AV_CKPT, OTR_LTX_AV_TEXT_ENCODER,
OTR_LTX_AV_VAE (each with a shared-models-root default; F2).

assert_usable(host_caps, profile, request_template=None), in order:
1. flag gate (GATED_BY_FLAG);
2. Sage gate (BUG-070);
3. NVML REQUIRED: gpu_residency.nvml_available() False -> fail CLOSED,
   named (THIS lane only -- heaviest engine; grounded fail-open risk:
   probe_used_mb()->0, asserts no-op);
4. node gate: every required ComfyUI node class resolves in
   NODE_CLASS_MAPPINGS (lazy read, not module-scope); missing ->
   MISSING_MODEL naming the classes (six-reason enum is PINNED -- no
   new reasons);
5. weights: resolved REALPATH exists (broken symlinks fail) + size >=
   per-artifact floors (transformer >= 12 GiB, encoder >= 10 GiB,
   video VAE >= 1 GiB); message names the exact artifact;
6. av_dims on request_template.canvas when provided (None tolerated);
   av_dims violations are caught and re-raised as EngineUnusable with
   an existing reason + the dims message (F5; no raw ValueError
   escapes).

Isolation STOP rule: in-process IFF the cu130 pip freeze is IDENTICAL
before/after M0 setup AND all nodes resolve; otherwise declare
ISOLATION_SIDECAR_REQUIRED, STOP the sprint, write the finding.

### Dims validator (NEW nodes/_otr_shared/av_dims.py; F3-completed)

assert_ltx_dims(w, h, frames): W%32==0, H%32==0, frames%8==1,
frames >= 9 (_LTX_MIN_FRAMES grounded); RAISES (never rounds) naming
nearest-valid BOTH directions (floor/ceil multiples); message format
pinned for tests. next_8n1(n) = ((n+6)//8)*8 + 1 (snap UP -- the
legacy eng_ltx_video :281 formula snaps DOWN; never copy it).
Upstream silently rounds; "+1 on W/H" is an upstream doc error
(Lightricks/ComfyUI-LTXVideo #347). 1472x832 passes as-is.

### I/O contracts

- EXTRACTION: audio via tolerant _ref_path(request.audio_ref)
  (AudioRef {"path"} | str | .path; eng_humo.py:366-383); init_image =
  asset_refs.get("init_image",""); conditioning_refs NEVER satisfies
  required inputs; base_clip_ref ignored. ltx_av_talk fails closed
  pre-render (classified, before GPU) when either is empty.
- AUDIO NORMALIZE: core ALWAYS ffmpeg-normalizes the incoming
  audio_ref to s16le/44.1kHz/mono WAV in the episode temp dir (master
  slices already are; per-line variants are not). Node-accepted
  formats: M0 sheet.
- FRAMES: T = timing.target_frame_count (INTEGER authority; never
  derive from audio duration). render = min(next_8n1(T),
  LTX_AV_MAX_FRAMES [M0-measured; initial 497]). canonicalize TRIMS to
  exactly T, or PADS-BY-LAST-FRAME to T (cap case) with the LOUD
  marker "[ltx_av] pad-tail rendered=<n> target=<T>" when > 2s, AND
  stamps pad_tail_frames/padded_s structured on the clip (F6).
- OUTPUT: graph terminates at the video VAEDecode -> IMAGE batch ->
  wrapper_bridge.encode_frames_to_silent_mp4() (-an; "only the mux
  adds audio"). An audio-bearing container NEVER exists on disk; the
  joint-AV contingency strip is `-map 0:v:0 -an` guarded by a fake-AV
  unit test. canonicalize returns the CanonicalClip shape (has_audio=
  False, yuv420p, bt709, fps 25, integer frame_count) and STAMPS
  engine_id + family; ffprobe-asserts ZERO audio streams.
- CANVAS: request.canvas.w/h renders; default landscape 1472x832.
- INIT IMAGE (talk): in-graph core-node preprocess (ImageScale+crop)
  from resolve_aspect_transform math; v1 = uniform COVER +
  center-crop (no pad bars -- padding conditions bars into the gen);
  pad+outpaint is an M0 EXPERIMENT cell only (the M1 graph ships
  COVER+crop; `# TODO M0` marks the cell).

### Prompts

- Adapters are PROMPT-THIN: request.text_prompt/negative_prompt only;
  an AST test forbids brief-helper imports in eng_ltx_av.py.
- Driver composes (precedence: M4 creative > operator override >
  brief-composed). ltx_av_music JOINS the existing :418 no-creative
  branch verbatim (OTR_LTX_RADIO_PROMPT honored on opens; "vintage
  radio set" open clause; get_story_brief_ltx + finish_visual_prompt,
  240 cap, no-text clause). ltx_av_talk gets a SIBLING branch with NO
  radio override (scene prose fights portrait-I2V): fallback prompt =
  subject + "head and shoulders at a period microphone" (setting
  noun, NO speech verbs) finished at 240/style_tail=False + no-text.
  Subject = the cast row's character_description resolved from the
  LEDGER by char_id (NOT the M4 creative dict -- grounded absent
  there; F7), default "a 1940s radio announcer" when absent.
  FORBIDDEN in prompts: quoted dialogue/beat text, stage directions,
  vocative character names, caption text.
- NEGATIVE: _LTX_DEFAULT_NEGATIVE verbatim, one shared constant;
  extension only on M0 inert evidence (pre-agreed ", frozen pose,
  still image", music-only). No music motion-vocabulary in v1 (one M0
  cell tests a single motion-energy clause). Cap 240 everywhere.

### Wiring (render_driver deltas a-i, all additive; re-base line refs
### before editing -- F8)

(a) :387 landscape-canvas tuple += ltx_av_talk, ltx_av_music.
(b) :418 prompt gate: tuple += ltx_av_music; sibling ltx_av_talk
    branch (talk template above).
(c) synthetic-timing slice: line-less shots (b000 opening music) slice
    the frozen master from SHOT start_s/dur_s IFF engine_id ==
    ltx_av_music (the universal line-backed slice path already feeds
    everything else and stays byte-identical for existing engines).
(d) _render_one passes request_template=request to assert_usable
    (Protocol already declares the kwarg; TypeError guard for any
    legacy adapter).
(e) ENGINE_FAMILY += {"ltx_av_talk": "audio_driven_face",
    "ltx_av_music": "audio_conditioned_video"}.
(f) apply_engine_override validates (role, engine) via
    engine_fits_role; incompatible entries IGNORED with a LOUD
    warning; forcing never bypasses render-time asserts.
(g) announcer portrait alias: on role==announcer_visual + empty
    char_id + engine_id=="ltx_av_talk", populate
    asset_refs["init_image"] from the shipped non-cast announcer
    portrait recorded in ledger["images"] (object id VERIFY-AT-BUILD
    from the 435ba0a chain); missing -> classified pre-render fail ->
    chain walks (humo also starves LOUD) -> floor; trail records
    every hop.
(h) run_episode EPISODE SUMMARY (from runtime_fallback_decisions +
    clip pad fields): fallback_counts_by_from_engine, pad_tail_count,
    padded_s, nvml_available, max_vram_mb, final_engine_histogram;
    STORM lines: ">=2 degrades same ltx_av_* origin" ->
    "[ltx_av] FALLBACK_STORM from=<e> count=<n>/<beats>"; ">=2 clips
    pad >2s" -> "[ltx_av] PAD_TAIL_STORM count=<n> padded_s=<s>".
(i) _slice_master_audio cache key += master mtime_ns + size (the ONE
    shared-path bugfix this sprint ships; unit-tested; also fixes the
    latent HuMo-slice staleness).
(+) SYNTH_FALLBACKS += both names (belt-and-braces for the
    guarded-import edge).

Also: schemas.py FAMILIES += "audio_conditioned_video" with
FAMILY_REQUIRED_INPUTS entry ("text_prompt","audio_ref") (the sync
assert then passes); role_compat.py MUSIC_VISUAL supply +=
"audio_ref" (unconditional, M1); __init__.py guarded import;
registry.py docstring corrections (family list + the stale "ShotLock
calls assert_usable" claim -- grounded: enforcement is RENDER-TIME;
episodes never abort, beats never drop). NO Director edits (V-6
auto-dropdown). NO group pruning (lane has no provider groups).

Flag-off behavior (documented + tested): a dark-lane pick degrades AT
RENDER via the gated EngineUnusable -> LOUD restamp -> chain ->
episode completes.

Hash/seed safety: render_request_hash is ShotLock-stamped and
driver-read-only; request_seed = _seed_from_hash(hash) on the episode
path; attaching audio_ref moves nothing (tested).

### Hardware (judge-verified sizes; HF API 2026-06-10)

Kijai 22B distilled-1.1 fp8_scaled 23.5 GiB; QuantStack distilled GGUF
Q2_K 11.6 / Q3_K_S 13.0 / Q3_K_M 13.7 / Q4_K_S 15.6 / Q4_K_M 16.5 /
Q5_K_S 17.3 / Q5_K_M 18.1 / Q6_K 19.6 / Q8_0 23.7 GiB;
gemma_3_12B_it_fp8_scaled encoder 13.2 GB; audio VAE 365 MB; video VAE
1.45 GB; taeltx2_3 23.5 MB; text_projection 2.3 GB; LoRA 2.7 GB;
NVFP4 dev 21.7 GB.

- DEAD for full residency under the 14500 MB NVML ceiling: Q4_K_S,
  Q4_K_M, Q5+, fp8_scaled -- offload/block-swap rows only. Candidates:
  Q3_K_S (~1.5 GiB headroom), Q3_K_M (BORDERLINE). Total NVML decides,
  never file size.
- ENCODER PHASING inside ONE AS-3 lease (never release between
  phases): acquire -> text encode -> reclaim_idle_models("ltx_av
  text-encode phase") [_soft_free insufficient] -> load transformer ->
  sample -> decode -> teardown reclaim -> release +
  wait_until_below_mb(14500). M0 measures GPU-encode-then-reclaim vs
  CPU-offloaded encode; v1 default = the passing mode (prefer GPU).
- PASS BARS: NVML peak+sustained <= 14500 MB; wall <= 10 min/clip
  PASS, 10-15 WARN (ship-able opt-in, documented), > 15 FAIL (lane
  parked). Episode (~3 lane clips/30w): <= 30 min PASS, > 45 dead.
  Quality = OPERATOR A/B vs the named 2B proof clip
  (predicting_the_winner LTX open), labels LIPSYNC/STYLIZED/INERT +
  keep/no-keep, recorded in the sheet (F4 -- look-QA is the standing
  mechanism).
- L3 NVFP4 CUT from M0 (dev-only steps, 21.7 GB, issue #11864);
  stretch column later. Two-stage: base-only v1. FLUX co-residency:
  none (sequential; one M0 verification row proves lease-released +
  below-ceiling before video).
- System RAM: sheet records RAM/pagefile/peak commit per lane; paging
  -> wall blowup (the wall gate catches it); RAM >= 32 GB required
  for block-swap rows; disk-free check before pulls;
  download_ltx_2_3.ps1 disk note bumped >= 24 GiB; ComfyUI-GGUF pack
  presence is an M0 inventory row (Manager pack, not pip; the
  pip-freeze sandwich still binds).

### Pre-mortem detectors + disciplines (pass07)

Episode summary + the two STORM lines (above; M4 normal smoke FAILS on
any storm line). NVML fail-closed for this lane. Weight floors +
realpath. Post-cancel/post-OOM: before the next heavy shot, lease
absent AND below-ceiling, else ONE screaming line instructing restart;
operator rule "restart ComfyUI after any mid-render cancel" lives in
the M0 checklist + adapter docstring + ship notes (a live wedged PID
holds the lease ~120s; reclaim only frees dead PIDs). RELOAD THRASH
accepted v1 (per-clip lifecycle is the contract; the M0 2-clip row
prices it; keep-resident is future policy). M0 launcher HARD-GATE:
:8000 active-job check + no held lease + NVML idle; never during the
acceptance window. Desktop/headless PARITY table = SHIP gate for
look-QA (runtime self-gate is safety-sufficient). Module staleness:
restart discipline in checklist + docstring + error text. Captions/
credits: LOW (absolute timeline placement); existing duration_check/
captions greps + one manifest frame_count-vs-target note. Goldens =
SEMANTIC PROJECTION (engine_id, family, role, canvas, prompt source/
length-class, audio_ref presence, asset_refs keys, timing, seed) +
regen-note policy.

## TESTING (the forgot-it matrix is total: every edit -> a named test)

NEW tests/test_av_dims.py (pure unit: snap-up cases 25->25 26->33,
nearest-valid both directions, 1472x832 pass, 1450x832 raise, min 9,
cap+pad flag). NEW tests/test_video_ltx_av.py (mirrors
test_video_humo.py: registered_and_dark x2 [membership, never
cardinality], role fit incl. music audio_ref, required-vs-family
schema, assert_usable ORDER incl. NVML fail-closed + node gate via
mocked NODE_CLASS_MAPPINGS + weight floors + template-None tolerated,
ref extraction, deterministic build, canonicalize silent bt709 +
identity stamps, fake-AV strip + ffprobe zero audio [ffmpeg
skip-guard], pad-tail marker + structured fields, AST no-brief-import,
cold-import, ascii, 5-hop talk + 3-hop music chains, SYNTH_FALLBACKS
membership). NEW tests/test_ltx_av_driver_wiring.py (dark-lane
SEMANTIC-PROJECTION goldens under tests/fixtures/ltx_av_dark/ --
captured BEFORE the driver edits; flag-off degrade completes + trail
origin; force guard LOUD-ignores; announcer alias; synthetic slice
gating both ways; template pass-through + TypeError guard;
ENGINE_FAMILY; canvas tuple; prompt gates incl. no-radio-override on
talk; storm-line emission; slice-cache key unit). FALLOUT: retry
-taxonomy chain sweep += both chains [filename VERIFY]; pre-code
literal search converts exact enumerations to membership; b7 sweep
auto-covers the new file (loop var stays `imp` if the sweep is
edited). BYTE-IDENTICAL: CPU structural only; the DEDICATED
forced-lane master-hash run is M4 GPU (OTR_REGRESSION_RUNTIME
mechanics). Pytest = no network/CUDA/weights/real forwards. M0 sheet
parser test lands AFTER M0; from M2 a test pins LTX_AV_MAX_FRAMES ==
sheet. Bug Bible: BUG-070 + BUG-291 pins (lease-release mirror); NEW
row at ship "LTX dims round silently; OTR fails loud" under the
Three-File Contract.

## TOUCH LIST (complete)

NEW  nodes/_otr_video_engines/eng_ltx_av.py
NEW  nodes/_otr_shared/av_dims.py
NEW  tests/test_av_dims.py, tests/test_video_ltx_av.py,
     tests/test_ltx_av_driver_wiring.py, tests/fixtures/ltx_av_dark/
EDIT nodes/_otr_video_engines/__init__.py, schemas.py, registry.py
     (docstrings), nodes/_otr_shared/role_compat.py,
     nodes/_otr_video_engines/render_driver.py (deltas a-i + SYNTH)
EDIT retry-taxonomy sweep + any exact-enumeration tests found
EDIT scripts/download_ltx_2_3.ps1 (disk note; sibling pulls GGUF/
     encoder/VAE)
NEW  docs/2026-06-10-ltx-av-lane/M0_RESULTS.md (sheet: inventory,
     pip-freeze sandwich, GRAPH SPEC [F1], dims/ceiling/audio-format
     findings, lane table, parity table, prompt cells, P1 verdicts)
DOCS tracker row; Bug Bible dims row at ship; handoff note.

## TICKETS

### M0 -- OPERATOR PROBE (GPU evening; not a coder ticket)
Checklist order: (0) launcher gate -- :8000 idle + no lease + NVML
idle; NEVER during the acceptance window; (1) disk + pack inventory
(artifacts w/ realpath+size; ComfyUI-GGUF pack; node parity Desktop vs
headless); (2) pip-freeze BEFORE; (3) scratch IA2V graph OUTSIDE OTR
(official template); (4) one ~5s render with a REAL per-beat slice;
hash the output audio track (probe); (5) pip-freeze AFTER == BEFORE
(STOP rule); (6) the LANE TABLE rows: Q3_K_S resident / Q3_K_M
resident / Q4_K_M offloaded / L1 fp8 block-swap / 2-consecutive-clip
marginal cost / encoder GPU-vs-CPU mode / taeltx-vs-full-VAE; NVML
idle-preload-peak-sustained-post + wall + RAM commit per row; (7)
GRAPH SPEC capture (topology, node classes, widgets, terminal node,
talk-vs-music diffs) [F1]; (8) prompt cells (+/- speech verb; +/-
motion clause; optional long-prompt); pad-vs-crop experiment cell;
(9) NEGATIVE-INSTALL drill (one artifact aside -> exactly one
FALLBACK_STORM line); (10) P1 eyeball matrix a/b/c/d -> verdict
LIPSYNC | STYLIZED | INERT per role-shape + keep/no-keep; (11) fill
M0_RESULTS.md incl. parity table; restart-after-cancel rule noted.
INERT everywhere = write the finding, close the lane (CW-1/2
scaffolding may stay dark or be reverted -- operator's call; F9).

### CW-LTXAV-1 "Dark skeleton + contracts" (M1; CPU; may run BEFORE/
### parallel to M0)
av_dims.py + eng_ltx_av.py DARK skeleton (metadata, assert_usable
gates, extraction, NO graph -- render_clip raises a clean
NotImplemented-classified error while dark) + schemas/role_compat/
__init__/registry edits + test_av_dims + test_video_ltx_av (minus
graph-dependent cases) + fallout membership edits.
DONE: full suite + Bug Bible green; both engines dropdown-visible
dark; byte-identical untouched; init-image policy in code =
COVER+crop with `# TODO M0` on the outpaint cell.

### CW-LTXAV-2 "Driver wiring + goldens" (M1/M3; CPU)
FIRST capture the semantic-projection goldens from the PRE-DELTA tree
(F9 ordering), THEN driver deltas a-i (+SYNTH) +
test_ltx_av_driver_wiring + storm/summary emission tests + slice-key
unit. Re-base all line refs (F8).
DONE: suite green; goldens prove dark-lane bit-identity; flag-off
degrade test green; storm lines emit only when forced.

### CW-LTXAV-3 "Graph + lane" (M2; AFTER M0 GO; consumes M0 GRAPH
### SPEC + max_frames + winning lane + encoder mode)
Winning-lane graph in the shared core (node candidates from the M0
GRAPH SPEC; pre-flight names exact classes); lease + encoder phasing;
silent encode; trim/pad; LTX_AV_MAX_FRAMES pinned to sheet (drift
test); graph-dependent tests.
DONE: suite + Bug Bible green; M2 gates; STOP rule re-verified
(pip-freeze unchanged).

### CW-LTXAV-4 "Live gates + ship" (M4/M5; GPU; operator present)
Forced-lane 30-word smoke (OTR_ENABLE_LTX_AV=1 +
OTR_FORCE_ENGINE_MAP=announcer_visual=ltx_av_talk,character_video=
ltx_av_talk,music_visual=ltx_av_music); DEDICATED forced-lane
master-audio-hash run; acceptance greps: swap-log lines, manifest
engine_id, NO storm lines, duration_check OK, captions events line,
NVML <= 14.5; obs gains playable AAC finals only; then M5: docs,
tracker, Bug Bible dims row (Three-File Contract), parity-table ship
gate, operator look-QA package (3 eyeball frames per forced role).
DONE: all greps green; operator verdict recorded.

## CUT LANES (audit trail)

Yvann-Nodes scheduling lane (p01; revisit only on INERT-for-music).
OTR_LTX_AV_PROMPT/_NEGATIVE envs (p03). ASPECT_CHANGE FailureKind;
group-prune wiring (p04). New usability reason; GPU pytest; new
frameworks (p05). NVFP4 in M0 (p06). Timeline-assertion script;
keep-resident-v1; exact-byte weight checks (p07).

## KEY VERIFY-AT-BUILD LEDGER (coder checks before relying)

Announcer portrait object id in ledger["images"]; retry-taxonomy
sweep filename; ffmpeg/ffprobe skip-guard pattern; #13111 node class
names (mock targets); cheap_families assert_usable signatures
(TypeError-guard scope); _rt.restamp_shot_row/format_swap_log exact
formats (freeze greps); CanonicalClip extras field for pad metrics
(else add optional fields); character_description ledger path;
compositor placement function (absolute start_s cite); free_after_use
scope; Desktop-cancel exception path; :8000 busy endpoint;
BUG-265 scope before attaching its name to tests.
