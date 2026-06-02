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
**graph guards** `d06560a` (`tests/test_workflow_graph_integrity_guards.py`).
**R0a (a-e) DONE 2026-06-02** on `v2.0-alpha`: `3ebf9f4` (a-c -- `_otr_resolved_request` shell
[`_seed_to_int64`, IN_KEY/IGNORED, cache_key, AUDIO asserts]; `_otr_determinism` + scoped
`deterministic_inference` + `scripts/run_comfy_otr.{bat,ps1}` + tf32-off at `_otr_model_loader.py:243-244`;
FreezeCascade node 62 gains `episode_seed`+`v2_ledger_json` at out indices 5,6, wired into
`workflows/otr_scifi_16gb_full.json`) and `892c280` (d-e -- legacy Bark/Kokoro/MusicGen/AudioGen seed
py+np+torch+cuda from the frozen `script_json` via `seed_all_rngs`; `_otr_legacy_manifest` +
`config/legacy_invocation_manifest.json` drift guard). Full `tests/` now **3495 passed, 12 skipped,
0 failed** (~32s). Only R0a **(f)** (operator-GPU render-twice bit-identity baseline) remains -- deferred.
The `eng_indextts2.py` adapter (under `nodes/_otr_audio_engines/`) exists but is now **dormant** for v2 (see cuts).

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

## Immediate next steps -- R0a (a-e) DONE; start Wave 0
R0a (a-e) is committed + green (see "Already shipped"); step (f) is operator-GPU (deferred).
**Wave 0 IN PROGRESS** (shared contracts; serial register + full `tests/` + commit). Pieces 1-2 DONE
(commit `d0decf1`): `build_resolved_request` + integer-tick `quantize_params` in `_otr_resolved_request.py`,
and `prepare_text` + `PREPARE_TEXT_VERSION` in `_otr_script_prep.py`. Full `tests/` 3525 pass / 12 skip.
**Resume at piece 3.** Plan SSOT table has file->imports->tests. The 8 Wave-0 pieces (1-2 done):
1. `nodes/_otr_resolved_request.py` -- ADD `build_resolved_request(...)` + integer-tick `quantize_params(...)`
   on top of the shipped shell (the shell -- `_seed_to_int64`, IN_KEY/IGNORED, `cache_key`, AUDIO asserts --
   is already in place; do NOT redo it).
2. `nodes/_otr_script_prep.py` -- ADD `prepare_text` (+`PREPARE_TEXT_VERSION`) on top of the shipped
   `clean_spoken_text`: strip asterisks/`♪`/delivery-tags, collapse ellipsis->one pause, KEEP `. , ?`.
   Golden examples test. Pure+deterministic (C7).
3. `nodes/_otr_audio_engines/registry.py` -- `assert_usable(engine,role)` + 6-class enum
   (`gated_by_flag|missing_model|missing_hf_token|incompatible_profile|noncommercial_blocked|malformed_config`),
   FAIL CLOSED.
4. `nodes/_otr_audio_engines/base.py` -- adapter base + `pack_audio_batch` + `supports_external_generator`.
5. `nodes/_otr_audio_cache.py` -- cache PROTOCOL only (interface; impl is Wave 1f).
6. `nodes/_otr_class_registry.py` -- mapping keys + display names + categories for the new nodes.
7. `config/*_schema.json` -- bank-entry + cache-sidecar schemas.
8. `config/audio_engine_profiles.yaml` + `nodes/_otr_engine_profiles.py` -- 0d resolver, lazy-load
   (module-cached, exception-wrapped), INPUT_TYPES legacy-first, NO module-scope IO (C-5).
Then Wave 1 (8 files), Wave 2a (CastLock + writer refactor), Wave 2b (opt-in JSON + 2 link tables).

## Operational notes (this session, 2026-06-02) -- read before resuming
- **Tests + git run via Desktop Commander on the Windows venv, NOT the sandbox** (sandbox has no torch;
  the bug-bible repo is not mounted there). Full suite:
  `cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio &
  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`
  (~32s, currently 3495 pass / 12 skip). conftest has a **KNOWN-FAIL-GUARD**: ANY new failing test is
  flagged a regression against a 0-fail baseline -- fix it, do not register it. To save context, redirect
  to a file and read only the tail summary.
- **Bug Bible repo NOT locatable** at the CLAUDE.md path (`...\comfyui-custom-node-survival-guide\`);
  `Documents\ComfyUI` looks OneDrive-virtualized (`dir` returns empty). Gated on in-repo `tests/` instead.
  Flag to Jeffrey / re-confirm the path.
- **Commit flow:** write `.git\COMMIT_EDITMSG` via the file tool, then in DC **cmd**:
  `git add <explicit paths>` (NEVER `-A` -- ~20 untracked planning docs + `custom_nodes.lnk` must stay
  unstaged) then `git commit -F .git\COMMIT_EDITMSG`. The CRLF->LF warning on JSON/`config` commits is the
  repo's existing eol policy (harmless). For workflow-JSON edits use a one-shot byte-replace py script
  (CRLF-preserving) that validates `json.loads` before writing -- do NOT reflow the whole file.
- **Deferred operator/GPU (batch to the very end, needs Jeffrey + RTX 5080):** R0a (f) render-twice
  bit-identity baseline, R0b box-fresh smoke, Wave 3 F/G1/I. Directive: no live testing until ALL headless
  sprints are built; the per-sprint pytest regression stays in every commit gate.

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
