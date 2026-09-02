VERDICT: yes-with-fixes -- the ONNX adapter is buildable, but section 3 and section 8 disagree on path/deps/providers, the torch call shape is specified two ways (byte-identity dies), and several tests/profiles the plan does not name will go red on the first pytest.

MUST-FIX BEFORE BUILD:
1. [S3 vs §8] Two specs, one implementor. §3 S3 puts the model at `<models>/TTS/KokoroTTS/onnx/model.onnx` and auto-picks ORT providers; §8 puts it at `<KokoroTTS>/onnx-community/onnx/model.onnx`, pins `CPUExecutionProvider`, and markers `kokoro-onnx` to 3.13-only. Strike §3 S1/S3/S4/S8 as superseded. One constant, duplicated in `nodes/_otr_kokoro_voice_prefetch.py` and `nodes/_otr_audio_engines/eng_kokoro.py` the same way `_KOKORO_MODEL_SUBDIR` already is (`eng_kokoro.py:38-39`, ` _otr_kokoro_voice_prefetch.py:77`), with the existing prefetch-dir test extended to the ONNX filename. Engine `load()` must look at that exact path; it is not specified today.

2. [§8 backends + S5 + invariant 1] Torch synthesis cannot move "VERBATIM" if the adapter also splits on `\n+`. Today `generate_voice` is one `KPipeline(text, voice=..., speed=self.speed, split_pattern=r"\n+")` loop (`eng_kokoro.py:233-235`). Pre-splitting then calling `synthesize` per segment is a different call shape and will fail the 5080 sha256 proof in §6. Fix: torch backend keeps that exact KPipeline call (full text + `split_pattern`). ONNX backend is the one that splits on `\n+` and calls `create` per segment. Adapter concat + 0.9 peak stays shared.

3. [§8 combo reorder] `_LEGACY_FIRST_ENGINES["char_voice"]` index 0 is pinned as `indextts2` in `nodes/_otr_engine_profiles.py:53-56`, `tests/test_engine_profiles.py:108`, `tests/test_batch_character_voices.py:108-112` and `:139-141` (`INPUT_TYPES()["required"]["engine"][1]["default"]`). Plan names only the last file. Also: CastLock `INPUT_TYPES` still defaults `voice_bank=_VOICE_BANKS[0]` (`"default"`) and `char_voice_engine="auto"` (`cast_lock.py:255-286`). Reordering the char combo to kokoro makes a menu-added `OTR_BatchCharacterVoices` default kokoro while a menu-added CastLock still resolves auto+default to indextts2, and there is still no char-side agreement check (`_otr_voice_node_common.py:953-964` is announcer-only). Smallest fix: do not reorder `_LEGACY_FIRST_ENGINES`; pin canonical node 81 `widgets_values` the same way node 80 is already special-cased (`tests/test_full_workflow_v2_audio_wiring.py:188-193`). If the ruling requires combo[0]=kokoro, also change CastLock's INPUT_TYPES defaults to `voice_bank="kokoro_builtin"` and `char_voice_engine="kokoro"` (string defaults only -- do not reorder `_VOICE_BANKS`).

4. [§8 prefetch root] Prefetch `_models_dir()` is three-up to `<comfy>/models` and the file says `folder_paths` is not importable (`_otr_kokoro_voice_prefetch.py:80-93`). Engine `_kokoro_model_dir()` uses `folder_paths.models_dir` (`eng_kokoro.py:48-58`). On this box weights live at `C:\ComfyUI-Models` (`nodes/_otr_gguf_backend.py:891-930`). A 326 MB fetch into the three-up tree will not be seen by `load()`. Duplicate `_models_root()` order in the prefetch file (env, existing `C:\ComfyUI-Models`, then `folder_paths` if importable, then three-up). Never import the engine at boot. verify: whether `folder_paths` is actually importable during `prestartup_script.py` (the current file says no).

5. [§8 find_spec vs load()] Engine selection must stay try-import (`from kokoro import KPipeline` / `import kokoro_onnx`). `tests/test_audio_engine_adapters.py:125-126` injects `sys.modules["kokoro"] = SimpleNamespace(...)` with no `__spec__`; `importlib.util.find_spec("kokoro")` raises `ValueError` on that entry. Do not share a find_spec helper into `load()`. Prefetch may use find_spec but must try/except (prestartup is never-fatal, `prestartup_script.py:87-98`).

6. [§8 create() kwargs] Pin against the 0.6.1 wheel with `inspect.signature` in the first coding hour, then write those names into the backend. Section 2 lists `trim` / `sentence_pause` / `clause_pause`; if 0.6.1 `create()` does not accept them, every ONNX line is `TypeError`. Same for `Kokoro.from_session(session, voices_path)` argument names. verify: 0.6.1 signatures (wheel not in the 5080 venv per §1). Empty-string segments after `\n+` split: skip them; do not call `create("")`.

7. [§8 load() errors] Missing package today is a bare `RuntimeError` (`eng_kokoro.py:149-152`), not `EngineUnusable`. Raise `EngineUnusable(self.name, role, EngineUsabilityReason.MISSING_MODEL, ...)` naming both pip lines. Missing ONNX file, failed npz build, and phonemizer/espeak failures at `create()` must be the same type -- a raw ORT/phonemizer traceback is not the C-7 contract. `unload()` must call backend `close()` then drop the session; `OTRVoiceNodeBase._teardown` already calls `adapter.unload()` (`_otr_voice_node_common.py:1530-1540`).

8. [§8 tests the plan omitted] Updating `profile_python_issue` off 3.13 will fail `tests/test_otr_provision_profile_routes.py:111-118` (`test_machine_readable_plan_check_rejects_kokoro_on_python_313`) and will change every `Python <=3.12` recipe that `scripts/otr_machine_matrix.py:289-290` currently stamps from a `(3, 13)` probe. Same change: `tests/test_engine_profiles.py:108`. The new allowed_voice_banks invariant will also fail `config/profiles/otr_rot_tts_bark.json` (`voice_bank=default`, `char_voice_engine=bark`) while `char_bark_v1.allowed_voice_banks` is `[bark_legacy]` (`config/audio_engine_profiles.yaml:46`). Fix or exclude that row in the same diff or the suite is red before ONNX exists.

9. [§8 process cache] "Selects once per process" plus tests that swap `sys.modules` and `OTR_KOKORO_BACKEND` needs an explicit reset seam, or re-read the env inside every `load()` (imports are cached; the choice is not expensive). Without that, `tests/test_kokoro_backend_select.py` as specified cannot be written.

10. [§8 OTR_KOKORO_ONNX_PROVIDERS] Empty or whitespace list must raise, not pass `providers=[]` into `onnxruntime.InferenceSession` -- empty means "all providers" and undoes the CPU pin. Invalid names fail loud.

SHOULD-FIX:
1. [S10] `nodes/story_packs/banks.json` has no ceiling field. The coverage test must map `source_bank_id` -> `_LEGACY_MAX_SPEAKING_CAST` (6, `_otr_casting.py:1176`) vs `MAX_SPEAKING_CAST` (10, `_otr_scifi_news_pro.py:351`). "Announcer both ways" is not a cast row; gender is `_seeded_announcer_gender` over `episode_seed` (`_otr_voice_bank.py:775-783`) -- use two seeds, not one roster. Bidirectional ENGLISH_VOICES vs `config/voice_reference_bank.json` (ids match filenames, e.g. `af_heart` at line 833-844) is a separate assertion.

2. [§8 IS_CHANGED] `OTRVoiceNodeBase.IS_CHANGED` fingerprints kokoro via the `.pt` bytes (`_otr_voice_node_common.py:66-101`) and does not include the ONNX model or `OTR_KOKORO_BACKEND`. A swapped `model.onnx` on an ONNX box will serve cached audio. Fold backend name + onnx file digest into the kokoro fingerprint, or document that ONNX model swaps require a new queue.

3. [§8 npz] Specify skip-and-log on a single corrupt `.pt` rather than aborting the whole npz (one bad voice would `EngineUnusable` every kokoro line). Windows `os.replace` of `_onnx_voices.npz`: on failure, `EngineUnusable`, do not swallow.

4. [§8 `close()`] verify: `onnxruntime.InferenceSession.close` exists on `>=1.20.1`. If not, `del session` is the unload. Teardown already swallows (`eng_kokoro.py:186-187`).

5. [domain / C-5] `nodes/_otr_audio_engines/_kokoro_backends.py` must not import `onnxruntime`, `kokoro`, or `kokoro_onnx` at module top -- `eng_kokoro` is imported at package init (`nodes/_otr_audio_engines/__init__.py:28`). Lazy inside `load`/`synthesize` only. Extend the `hf_hub_download` / `snapshot_download` ban (`tests/test_kokoro_voice_prefetch.py:113-116`) to the backends file.

6. [§8 marker tests] `tests/test_requirements_python_markers.py:31-33` keys by `Requirement.name`; `kokoro` and `kokoro-onnx` are distinct. Add the sibling assertions there; do not assume `_parsed()` can hold two `kokoro` lines.

OPTIONAL / NICE-TO-HAVE:
- Char-side CastLock vs node-81 engine agreement, cloned from the announcer guard.
- `OTR_KOKORO_ONNX_MODEL` filename override can wait; ship `onnx/model.onnx` only.
- CPU `intra_op_num_threads` cap so a 16-thread ORT session does not fight the video encode. [ASSUMPTION] default ORT thread count equals core count.
- 3.14 probe in `otr_machine_matrix.py` once `profile_python_issue` moves to `>= (3, 14)`.

CUT THESE:
1. §3 S4 "fail loud on cuda requested + CPU only" -- §8 already killed it: canonical stamps `voice_device=cuda` (`workflows/otr_canonical.json:1289-1296`) and `_voice_device_from_ledger` defaults to cuda (`_otr_voice_node_common.py:314`). Implementing S4 would refuse every 3.13 canonical run.
2. Separate `onnxruntime>=1.20.1` line in requirements (§3 S8) -- §8 correctly drops it; kokoro-onnx metadata already carries it. A second pin fights the 5080's existing `onnxruntime` / `onnxruntime-gpu` pair.
3. Process-lifetime backend singleton beyond `load()` -- the voice node already unloads every generate (`_otr_voice_node_common.py:872-874`). Caching the ORT session across prompts fights I-7; re-load is 0.9 s per §2.
4. Prefetching the ONNX model when neither package is importable (§3 S3) -- 326 MB then `EngineUnusable`. §8's find_spec pair is the gate.

[ASSUMPTION] §8 is the plan of record; claims above treat it as SoT except where §3 would still be coded by a reader who starts at the top. [ASSUMPTION] `Kokoro.create` / `from_session` match section 2; not re-checked against a local 0.6.1 tree (package absent on the 5080 venv). No new NODE_CLASS_MAPPINGS / INPUT_TYPES widget insert; combo reorder is string-valued so saved `widgets_values` do not shift.
