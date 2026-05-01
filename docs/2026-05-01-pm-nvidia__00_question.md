# Question -- 2026-05-01

# Round-robin question -- 2026-05-01: post-mortem on three bug fixes + lingering symptoms

OTR SIGNAL LOST (ComfyUI radio-drama generator) on RTX 5080 Laptop / 16 GB VRAM Windows / torch 2.10 / CUDA 13 / Blackwell sm_120. Audio is king (rule C7, byte-identical between runs). 100% local, no cloud at runtime.

## Background: three fixes shipped today, one had to be reverted

**BUG-LOCAL-121 (commit `cee4ebb`) -- SHIPPED, intended as a defensive fallback:**
Added a filesystem fallback to `nodes/batch_humo_render.py::_resolve_radio_still_path`. The function now reads in order: (1) `ledger.radio_bookend_path`, (2) `ledger.meta.radio_bookend_path`, (3) `output/otr/stills/radio_bookend_<episode_id>.png` reconstructed from `ledger.episode_id`. Existence-check on disk at every layer. Returns None gracefully if all three fail.

Original hypothesis: radio bookend FLUX render was gated on `open_close=true` but ANNOUNCER role can exist without `open_close`, so the file was missing. Live log showed `[BatchHumoRender] line l001 speaker_role=announcer wanted radio still but it's missing`. The fix is purely additive (new fallback layer), 25/25 tests pass.

**BUG-LOCAL-123 (commit `fea2f23`) -- SHIPPED:**
Wrapped `nodes/video_composite.py::execute()` in try/finally that calls a new `_end_of_run_cleanup()` static method on every exit path. Cleanup pipeline: `comfy.model_management.unload_all_models()` -> `soft_empty_cache(force=True)` -> torch fallback (`gc.collect()` + `cuda.synchronize()` + `cuda.empty_cache()`).

Reason: ComfyUI's `model_management` deliberately keeps loaded models resident across prompts. After Run 1 finished cleanly (HuMo 16.5 GB + WanTE 6.4 GB + WanVAE + Whisper resident), Run 2's Mistral-Nemo `_prefill` SDPA attention OOM'd at the 1024-token full-script generate (3 spines at 150 tokens succeeded; 1024 didn't). Free-memory was 0 bytes when 4 GiB was requested. VideoComposite is the LAST node in every OTR workflow, so its finally block catches all paths. 18/18 tests pass.

**BUG-LOCAL-124 (commit `d180b62`) -- REVERTED today in `4095a55`:**
External static-analysis QA report flagged Node 12 (`OTR_SignalLostVideo`) `.video_path` output (an mp4 file path string) being wired into Node 51 (`OTR_BatchHumoRender`) `.ledger_json` AND Node 52 (`OTR_VideoComposite`) `.ledger_json` -- both expecting a JSON manifest. Recommended rewiring to Node 3 (`OTR_SceneSequencer`) `.scene_manifest_json` (slot 2, "previously unused"). Patch removed links 79 + 82 and added 86 + 87.

**Why it failed:** I read the source AFTER the live run crashed with `RuntimeError: BatchHumoRender: ledger path not found: []`. `OTR_SceneSequencer.scene_manifest_json` is a permanently-empty stub: `manifest = []` at `scene_sequencer.py:689`, never appended to, serialized at line 890 -> always returns `"[]"`. BatchHumoRender's `_load_ledger_with_path` then tried `Path("[]")` -> not on disk -> crash.

**The "misroute" was actually a working pattern.** BatchHumoRender's `_load_ledger_with_path` (lines 1694-1756 of `batch_humo_render.py`) explicitly handles `.mp4` paths arriving as `ledger_json` input via a multi-tier stem-fallback chain (exact match -> underscore-collapse variant -> directory scan with mtime + episode_id substring match). The type-tag "mismatch" on the graph editor was misleading; the consumer code knew what to do with it.

Memory note saved 2026-05-01: "Static-analysis QA reports can confidently prescribe wrong fixes if they don't verify what target outputs actually contain at runtime. ComfyUI STRING type is permissive -- type tag match doesn't mean 'this is the right input shape.' Always read the consumer's source before declaring something a misroute."

## Open symptoms that BUG-124 was supposed to fix but didn't (and they're still real)

The original Run 1 mp4 (`signal_lost_earth_is_splitting_open_beneath_the_paci_20260430_211312.mp4`) showed:
1. **Radar base layer plays straight through with HuMo clips overlaid violently** -- no scene-boundary cuts in the composite.
2. **All 3 announcer lines hit the "wanted radio still but it's missing" warning** despite `open_close=true` and an ANNOUNCER cast member. Each line fell back to `full_env_00154_.png` (cast still of the announcer's face).

Now that BUG-124 is reverted (those weren't caused by the link misroute), what ARE the actual root causes?

## Questions

### Q1: Sanity check BUG-121 (filesystem fallback for radio still resolver)

Is the layered resolver pattern correct? The function now checks ledger top-level path -> ledger meta path -> filesystem deterministic path (`output/otr/stills/radio_bookend_<episode_id>.png`). Each layer has an existence check, returns None gracefully if all fail. Is this the right defense-in-depth approach for a ComfyUI ledger-driven asset resolver?

Any failure modes I'm not catching? E.g.:
- What if `ledger.episode_id` is a non-string type or contains path-traversal characters?
- What if `otr_stills_dir()` resolution itself fails?
- What if the file exists but is a 0-byte / corrupt PNG that `Image.open()` will reject downstream?

### Q2: Sanity check BUG-123 (end-of-run VRAM cleanup in VideoComposite)

The pattern: wrap `execute()` in try/finally that calls `_end_of_run_cleanup()` on every exit path. Cleanup is: `unload_all_models()` -> `soft_empty_cache(force=True)` -> torch fallback. Each step has its own try/except + log so a single failure never breaks the run.

Concerns to address:
- Is calling `unload_all_models()` from a CONSUMER node (VideoComposite is just one node in a graph) the right pattern, or should this be at a different layer (e.g. ComfyUI hook, queue completion event)?
- Is `cuda.synchronize()` strictly necessary in the torch fallback path, or does `empty_cache()` already imply sync?
- Are there ComfyUI internal caches I should ALSO be clearing (e.g. clip vision encoder cache, IP-Adapter image embeddings) that `unload_all_models` doesn't touch?

### Q3: What ARE the real root causes of the 2 still-open symptoms?

**Symptom 1: No scene-boundary cuts in composited mp4.** Now that we know the manifest-misroute wasn't the cause (BatchHumoRender's stem-fallback handles mp4 paths correctly), what would make VideoComposite NOT cut between radar base layer and HuMo clips on scene boundaries?

Things to consider:
- VideoComposite reads `ledger.lines[]` and uses `start_s` / `dur_s` per entry to know when to overlay each HuMo clip. If those fields are missing or all 0.0, no cuts happen.
- BUG-LOCAL-106 (2026-04-29) added `dialogue_positions` tracking in scene-audio space to write authoritative `start_s` / `dur_s` back to ledger.lines[].
- VideoComposite's `master_mix_per_clip_mux` mode uses pillarbox + concat-demuxer + final mux pattern.
- The Run 1 mp4 was generated with `audio_source=humo_concat` (different mode -- the per_clip_mux failed strict_c7 and fell through). Could the symptom be specific to humo_concat mode?

**Symptom 2: Radio bookend FLUX render not happening / not stamping ledger.** Live log: `[BatchHumoRender] line l001 speaker_role=announcer wanted radio still but it's missing`. With `open_close=true`, the radio bookend SHOULD be rendered by `BatchFluxRender` (per the docstring at `batch_flux_render.py` line 455-467: "Rendering ALWAYS attempts under the new default"). The `radio_bookend_prompt` widget is `""` in the workflow JSON -> DYNAMIC mode. So it should fire.

Hypotheses for why it didn't:
- (a) FLUX render attempted but failed silently (caught by try/except at lines 481-502, only logs warning).
- (b) FLUX rendered the file but ledger stamping failed (subsequent ledger writes overwrote without re-merging).
- (c) The `episode_id` at FLUX render time was different from the `episode_id` at HuMo render time (e.g. ledger renamed mid-run).
- (d) Something else entirely.

What additional observability would let me prove which hypothesis is right on the next run?

### Q4: BUG-LOCAL-125 (scene_manifest_json is unpopulated stub) -- fix or deprecate?

Should `OTR_SceneSequencer` populate `scene_manifest_json` with real data (scene_id, shot_id, beat_id, start_s, dur_s per entry), or should the output be marked deprecated and removed from RETURN_NAMES? The slot exists, types-as-string, but always emits `"[]"`. There's no apparent consumer in the production graph that would benefit from reading it (the ledger.json on disk has all this data already).

Recommendation should consider: complexity cost of populating it, value to downstream nodes (currently zero), and the v2.0-beta release blocker calculus.

### Q5: How do I prevent the BUG-124 false-positive pattern in the future?

External static analysis (whether from a QA report, an LLM, or a teammate) flagged a "misroute" that turned out to be a working pattern with a defensive consumer. I shipped the fix without reading the target output's source. The QA report's claims about ChatGPT-style "ready to wire" outputs were confidently wrong.

Process question: short of a full integration test on every workflow change, what's the lightest-weight gate that would have caught this BEFORE it shipped? Some options I'm considering:
- (a) Pre-commit hook that runs the workflow JSON through a "shape check" against the target node's INPUT_TYPES + the source node's actual output implementation.
- (b) Required smoke test: any workflow JSON change must run a 30-second dry-run that exercises every modified link with synthetic inputs.
- (c) Mandatory "read the consumer code" rule before applying a recommended fix that touches graph wiring.

Which is most cost-effective for a one-developer project at the v2.0-alpha stage?

## For all questions

Prefer the smallest change with the largest payoff. Cite specific files / line numbers when relevant. Flag uncertainty rather than bluffing. If you don't have enough information to answer a sub-question, say so explicitly and identify what would unblock you.
