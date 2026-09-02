VERDICT: build-ready with fixes. Clean architectural separation between adapter and backends, but 5 concrete sequencing, interface contract, and Windows I/O defects must be resolved before build.

MUST-FIX BEFORE BUILD:
1. [S9 / S6 / nodes/_otr_voice_node_common.py:953-964] Character-side agreement check causes false-positive crash on preserve_ledger / char_voice_engine="auto".
   - Defect: Section 9 proposes adding a character-side agreement check in _otr_voice_node_common.py cloned from the announcer guard (:953-964). In nodes/cast_lock.py:1598-1611, under cast_voice_policy="preserve_ledger" (the default in CastLock and across 86 profiles) when char_voice_engine="auto", _stamp_voice_engine_selection stamps meta["char_voice_engine"] = "auto". A naive agreement check (stamped and stamped != engine) will compare "auto" against BatchCharacterVoices.engine (e.g. "indextts2" or "kokoro") and raise EngineUnusable(MALFORMED_CONFIG) on every default graph.
   - Fix: In _otr_voice_node_common.py, explicitly ignore "auto" in the character agreement guard: if stamped and stamped != "auto" and stamped != engine: raise EngineUnusable(...).

2. [S3 / S8 / S9 / nodes/_otr_kokoro_voice_prefetch.py / nodes/_otr_audio_engines/eng_kokoro.py] Path and folder structure discrepancy between prefetch download and engine loader for model.onnx.
   - Defect: Section 3 states model.onnx is placed at <models>/TTS/KokoroTTS/onnx/model.onnx. Section 8 states hf_hub_download(..., local_dir=<KokoroTTS>/onnx-community) with filename="onnx/model.onnx", which creates <models>/TTS/KokoroTTS/onnx-community/onnx/model.onnx (an extra onnx-community/ nesting level). If prefetch downloads into onnx-community/onnx/model.onnx while the engine looks for onnx/model.onnx, the engine will fail with EngineUnusable(MISSING_MODEL) despite the 326 MB download succeeding at boot.
   - Fix: Define a single shared relative path constant _KOKORO_ONNX_REL_PATH = os.path.join("onnx", "model.onnx") under TTS/KokoroTTS. In _otr_kokoro_voice_prefetch.py, set local_dir = os.path.join(models_dir, _KOKORO_MODEL_SUBDIR) and filename="onnx/model.onnx" so the file lands deterministically at <models>/TTS/KokoroTTS/onnx/model.onnx.

3. [S8 / S9 / prestartup_script.py / nodes/_otr_kokoro_voice_prefetch.py] Prestartup find_spec unhandled exception under test harnesses with fake modules.
   - Defect: Section 8 proposes gating prestartup prefetch using find_spec("kokoro") is None and find_spec("kokoro_onnx") is not None. As demonstrated in tests/test_audio_engine_adapters.py:119-136, test harnesses frequently fake sys.modules["kokoro"] = SimpleNamespace(...) (without __spec__). In Python, calling importlib.util.find_spec("kokoro") when sys.modules["kokoro"] has no __spec__ raises ValueError. If unhandled, this crashes prestartup during unit tests.
   - Fix: In _otr_kokoro_voice_prefetch.py, implement a safe module inspector _spec_exists(name: str) -> bool that wraps importlib.util.find_spec in try ... except (ValueError, AttributeError, Exception): return False.

4. [S2 / S8 / S9 / nodes/_otr_audio_engines/_kokoro_backends.py / nodes/_otr_audio_engines/eng_kokoro.py] NPZ voice dictionary key formatting and Windows file locking on session teardown.
   - Defect:
     a. kokoro-onnx's _style_for(voice, n) indexes self.voices[voice] where voice is the bare voice ID (e.g. "bm_george"). If the NPZ generator in eng_kokoro.py preserves the .pt extension in dictionary keys ("bm_george.pt"), kokoro-onnx will raise KeyError: 'bm_george' during synthesis.
     b. On Windows (WinError 32), np.load(npz_path) keeps an open zip handle on _onnx_voices.npz. When eng_kokoro.py attempts an atomic rebuild via os.replace(temp_path, target_npz) during a subsequent load, Windows raises PermissionError if an existing InferenceSession or uncollected NpzFile has the file open.
   - Fix:
     a. In the NPZ builder, strip the .pt suffix so keys match ENGLISH_VOICES exactly (voice_id = os.path.splitext(os.path.basename(f))[0]).
     b. In eng_kokoro.unload(), explicitly close/delete backend sessions and invoke gc.collect(). In the NPZ builder, catch PermissionError during os.replace on Windows and fall back to writing to a unique filename in tempfile.gettempdir() or reusing the existing valid NPZ.

5. [S9 / S6 / tests/test_full_workflow_v2_audio_wiring.py:171-195] Wiring test assertion failure from special-casing node 81 without exempting it from the derived defaults loop.
   - Defect: In tests/test_full_workflow_v2_audio_wiring.py:171-195, test_widget_vectors_exact derives expected widget defaults using _derive_widget_defaults(mapping[key]) for all nodes in NEW_NODE_IDS except node 80. NEW_NODE_IDS contains "OTR_BatchCharacterVoices": 81. Because Section 9 leaves _LEGACY_FIRST_ENGINES["char_voice"] with index 0 as "indextts2", _derive_widget_defaults(BatchCharacterVoices) will return ["indextts2"]. When canonical node 81 is updated to ["kokoro"], test_widget_vectors_exact will fail on node 81.
   - Fix: Update tests/test_full_workflow_v2_audio_wiring.py test_widget_vectors_exact to assert by_id[81]["widgets_values"] == ["kokoro"] and exclude nid in (80, 81) from the _derive_widget_defaults loop.

SHOULD-FIX:
1. [S9 / nodes/_otr_voice_node_common.py:760-777 / nodes/_otr_audio_engines/eng_kokoro.py] Unify caching invalidation via KokoroEngine.render_time_params().
   - Defect: Section 9 proposes adding backend name and ONNX file mtime to _provisional_identity_fingerprint. However, in canonical/unrouted runs where provisional_rows and routes are empty, IS_CHANGED in _otr_voice_node_common.py:767 returns "static" without calling _provisional_identity_fingerprint.
   - Fix: Implement render_time_params(self) on KokoroEngine returning {"backend": active_backend, "onnx_mtime": mtime_or_0}. Because _otr_voice_node_common.py:762-777 queries _get_engine(engine).render_time_params() and adds its contents to the cache hash before the "static" check, this guarantees cache busting across backend/model switches for all roles (announcer and character) automatically.

2. [S8 / nodes/_otr_audio_engines/eng_kokoro.py] Fallback self.role in eng_kokoro.py for standalone engine invocations.
   - Defect: _otr_voice_node_common.py:857 sets adapter.role = self.ROLE during generate(). If eng_kokoro.load() or generate_voice() is called directly in unit tests or CLI scripts, self.role may not be set, causing EngineUnusable(self.name, self.role, ...) to raise AttributeError.
   - Fix: Initialize self.role = "announcer_voice" in KokoroEngine.__init__ or use getattr(self, "role", "announcer_voice").

3. [S8 / requirements.txt / pyproject.toml / scripts/otr_provision.py:1710-1724] PEP 508 marker consistency on Python >= 3.14.
   - Defect: requirements.txt specifies kokoro-onnx>=0.6.1; python_version >= "3.13" and python_version < "3.14". On Python 3.14, neither backend is selected.
   - Fix: Ensure scripts/otr_provision.py profile_python_issue flags Python >= 3.14 with an actionable message noting that neither kokoro nor kokoro-onnx is packaged for 3.14 yet, and update tests/test_requirements_python_markers.py to assert mutual exclusion across 3.12, 3.13, and 3.14.

OPTIONAL / NICE-TO-HAVE:
1. [S8 / nodes/_otr_audio_engines/eng_kokoro.py] Support OTR_KOKORO_ONNX_MODEL env var override to allow lab testing of alternative quantized ONNX variants (model_q8f16.onnx) without modifying source files.
2. [S10 / tests/test_kokoro_char_voices.py] Add a quick unit assertion confirming that all 28 .pt voice files load with torch.load(weights_only=True) without NaN/inf values.

CUT THESE (over-engineering):
1. [S4 / S8] Custom GPU provider negotiation in ONNX Runtime: Safe to cut. Keeping ONNX Kokoro strictly on CPUExecutionProvider with thread cap min(4, os.cpu_count()) delivers RTF ~0.15, avoids CUDA/cuDNN DLL mismatch hell on portable Windows environments, and leaves all GPU memory for video pipelines.
2. [S8] Standalone onnxruntime entry in requirements.txt: Safe to cut. kokoro-onnx's package metadata already specifies onnxruntime>=1.20.1 as a direct dependency.
