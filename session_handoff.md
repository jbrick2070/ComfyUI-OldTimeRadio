# Session Handoff -- OTR Audio + Voice-Casting Overhaul -- 2026-06-02

## Core goal
Build a model-agnostic, per-role audio engine registry + an upstream voice-casting
subsystem for the OldTimeRadio ComfyUI pipeline, on `v2.0-alpha`. Character voice =
`bark|chatterbox`, announcer = `kokoro|chatterbox`, music = `musicgen|stable_audio_music`,
each selectable per role; plus a deterministic voice bank/caster, a post-freeze cast-lock
node, and a frozen `ResolvedVoiceRequest` cache/identity contract. The legacy path stays a
permanent **byte-identical** fallback. Design is finished and reviewed; this session ends at
**ready-to-build**.

## Canonical spec (read this first, don't re-derive)
`docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` is the single source of truth --
invariants I-1..I-11, ComfyUI first-run invariants C-1..C-7, the wave/sprint build order, the
`ResolvedVoiceRequest v1` field list, the per-sprint test names, re-baseline triggers, and the
verify-at-build list. The handoff below is orientation + live decisions only; the plan has the
detail. (CLAUDE.md prime directives, git flow, and testing auto-load -- not repeated here.)

## Already shipped this overhaul (build on these; do not redo)
Sprints committed on `v2.0-alpha`: **A** `9b76d78` (audio engine registry + `_otr_audio_utils`),
**B** `1b5a39b` (delivery vector), **C** `c79cc51` (engine adapters: bark/kokoro/musicgen batch +
chatterbox/indextts2/stable_audio per-line), **C.1** `f49d4f9` (`_otr_script_prep.clean_spoken_text`),
**graph guards** `d06560a` (`tests/test_workflow_graph_integrity_guards.py`). Full `tests/` was green
at last run (3444 passed, 12 skipped). The `eng_indextts2.py` adapter exists but is now **dormant**
for v2 (see cuts).

## Decided this session (don't reopen)
- **Build runs in waves** (orchestration section of the plan): **R0a** (serial, first, ends on the
  one operator-GPU baseline capture) -> **Wave 0** (shared contracts, parallel) -> **Wave 1** (8 independent
  node/config files, Wave-0-only deps, parallel) -> **2a** (writer refactor + `OTR_CastLock`, serial) ->
  **2b** (opt-in workflow JSON + R0b smoke + push, serial) -> **Wave 3** (operator GPU: F pilots -> G1 -> I).
- **Parallel-safety is mandatory and runtime-state-aware:** worktree per agent, no shared venv,
  per-agent `OTR_AUDIO_CACHE_DIR`/`OTR_TEST_TMP`/`COMFYUI_TEMP_DIR`, no shared `otr_runtime.log`/`conftest`.
  `__init__.py`, the workflow JSON, and every regression+commit are serial gates. A `--no-parallel`
  path always exists; parallelism is optional.
- **Wave 1 was NOT independent as first drafted** -- the generic nodes import the request builder,
  quantizer, `prepare_text`, `assert_usable`, the adapter base/`pack_audio_batch`, and the cache
  protocol. Those all moved into **Wave 0** so 1a-1c depend on Wave 0 only. (This was the #1 blocker.)
- **Link migration is by two explicit tables, never blind slot index** (node 62 out[1] fans to 13
  consumers; theme outputs are renamed). 13-row consumer->source partition + theme name-map. (Blocker #2.)
- **FreezeCascade ports append at output indices 5,6 -- never insert** (insert shifts 0-4, breaks legacy
  link 110). `episode_seed` is a dedicated port read from the locked ledger; no widget.
- **Box-fresh-clean hardening:** no module-scope/import-time IO; YAML lazy-loaded; engine libs
  lazy-imported inside `generate()`; `INPUT_TYPES` hardcoded legacy-first. Absent default-ON model ->
  NAMED error at queue time + out-of-band fetch; **never network during `execute`**.
- **Engine defaults (Jeffrey, 2026-06-02) -- ship best-on-by-default, opt-out:** character voice
  **IndexTTS2 (#1) > Chatterbox (#2) > Bark (#3)**; music **Stable Audio 3 (#1, LOCAL, ComfyUI v0.22.0+
  native `Comfy-Org/stable-audio-3`, commercial-licensed) > MusicGen > Stable Audio Open**; announcer
  Kokoro|Chatterbox. IndexTTS2 + SA3 are the SHIPPED defaults; their F dep-isolation pilots are the
  promotion gate (no xformers/flash_attn/torch swap on sm_120; **no TensorRT** -- the NVIDIA-optimized
  HF collection is image/video-only, brittle on Blackwell + breaks byte-identical). Registry is
  model-agnostic -- a new engine = adapter + profile row + bank entries; users switch any role's engine
  via the node's `engine` dropdown.
- **Commercial = three-state, warn-not-block (I-8):** `true`->silent ship; `false` (known-gated:
  IndexTTS2 needs Bilibili authorization, MusicGen CC-BY-NC, Stable Audio Open NC) -> one-time
  non-blocking warning in `cast_report`+`audio_meta`, still renders; unknown/missing -> stop-ship.
  SA3 is `commercial_clean=true`. SA3 weights HF-gated (accept license + `HF_TOKEN`); IndexTTS2 weights
  are un-gated download.
- **Scope CUTS for the lean alpha** (re-addable later): license sub-fields + final-mux metadata scrubber
  deferred (boolean/warn gate only); manual voice overrides removed (`manual_voice_assignments_json`/
  `manual_override_sha` gone); migration slimmed to detect + reject-missing-`voice_ref_id` +
  version-mismatch->re-render (no 3-mode/quarantine); SFX out of v2.
- **Rejected (do NOT apply these reviewer suggestions):** keep `age_band` in bank/caster; keep
  `delivery_profile_id`+version as cache identity even though only `neutral` ships; do NOT build an
  `OTR_AudioTeardownJoin` node (the `finally`-teardown + existing `audio_done` gate already serialize);
  `OTR_CastLock` does NOT replace node 62 and cannot merge char_ids (char_id is stable identity, I-9).

## Immediate next steps (start Wave R0a -- all headless except (f))
1. `nodes/_otr_resolved_request.py` -- `_seed_to_int64(*parts)->int`, the `ResolvedVoiceRequest`
   shell (IN_KEY/IGNORED fields per the plan), and the AUDIO-batch contract asserts
   (`{waveform:[B,1,T], sample_rate}`, empty `[1,1,0]`). Full `tests/`, commit.
2. `nodes/_otr_determinism.py` (post-`import torch` flags, scoped `deterministic_inference(seed)` CM,
   SDPA MATH pin) + `scripts/run_comfy_otr.bat`/`.ps1` (env BEFORE python) + flip `allow_tf32`->False
   at `nodes/_otr_model_loader.py:243-244`. Commit.
3. Node 62 FreezeCascade: append dedicated `episode_seed` + v2-ledger output ports at indices 5,6;
   derive `episode_seed` internally; assert out[1] bytes unchanged for raw delegation. Wire into the
   workflow JSON. Commit.
4. Legacy seeding in the 4 legacy audio nodes (Bark/Kokoro/MusicGen/AudioGen): seed py+np+torch+cuda+
   Generator before forward, seed = `_seed_to_int64(sha256(frozen script_json bytes))`, no parse. Commit.
5. `config/legacy_invocation_manifest.json` -- frozen widget vectors + `widgets_sha256` for the 4 legacy
   nodes; add the contract test. Commit.
6. **Operator/GPU gate (needs Jeffrey + the RTX 5080):** render-twice legacy for bit-identity, capture
   `baseline_v2_audio_legacy_{sha,ledger_sha,audio_metadata_sha}` (audio only). Then Wave 0 opens.

After R0a: Wave 0 contracts (request builder+quantizer, `prepare_text`+version, `assert_usable`+6-class
enum, adapter base+`pack_audio_batch`+`supports_external_generator`, cache PROTOCOL, `_otr_class_registry`,
bank+sidecar schemas, `0d` profiles resolver), then the 8 Wave-1 files. See the plan's SSOT table for
file->wave->imports->tests.

## Open questions (flag to Jeffrey before they bite)
1. RESOLVED -- IndexTTS2 = shipped default voice, Stable Audio 3 = shipped default music (Jeffrey,
   2026-06-02). Do NOT re-cut. Residual operator checks before F: ComfyUI Desktop >= v0.22.0 (native SA3),
   SA3 HF license accepted + `HF_TOKEN` set, IndexTTS2 + SA3 weights present.
2. Commercial posture = warn-not-block; the SA3 default keeps the shipped stack commercial-OK on the
   music side (the IndexTTS2 voice still warns -> needs Bilibili authorization only if Jeffrey monetizes).
   Re-add the final-mux metadata scrubber + license sub-fields only when OTR output goes to public
   distribution.
3. Verify-at-build items (plan's last section) are unresolved by design -- resolve each in its owning
   wave: node 4 AudioEnhance device (CPU vs CUDA teardown), per-node `finally` teardown ordering, the
   `d06560a` guard not pinning node-62 output count, forceInput-no-widget-key in the builder, SceneSequencer
   sfx-None pure-prepend, litegraph builder schema fidelity.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps. Acknowledge when you're ready
to start."
