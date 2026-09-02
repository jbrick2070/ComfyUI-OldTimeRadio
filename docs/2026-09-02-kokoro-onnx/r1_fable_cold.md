# r1 -- Fable 5.1, cold (anchor withheld), 2026-09-02

Substitute seat for the r1 arc round (operator: one strategic partner per round).
The driver grounded every claim below against the files before folding; the verdicts
that survived are in `driver_anchor.md` section 8. Two claims did not survive as
written: the `runtime_label` read at `_otr_voice_node_common.py:998-1001` does not
exist (it was a suggestion, kept as NICE), and the ComfyUI `main.py` line numbers were
not checked (the no-torch-at-prestartup rule is taken from the prefetch module's own
docstring and ComfyUI's documented boot order).

## 1. THE ARC

The one-engine-two-backends shape is the right one, and it is forced, not chosen: the operator's 09-01 product shape is ONE canonical JSON whose saved dropdowns render on every box, and `assert_usable` resolves the saved engine NAME (`registry.py:142-175`), so the same string `"kokoro"` has to mean torch on the 5080 and ONNX on the portable. A second registered engine (`kokoro_onnx`) would be cleaner internally but would need a different saved value per machine, which the ruling forbids.

What breaks as tentatively shaped:

1. The ship-scope edit is three nodes, not one, and a test pins one of them to INPUT_TYPES. Node 80 (CastLock) is `["default","auto_registry",true,"indextts2","kokoro","cuda"]`, but node 81 (BatchCharacterVoices) carries its own `engine` widget, `"indextts2"`, and there is NO cross-widget check for the char role -- the check at `_otr_voice_node_common.py:953-964` is announcer-only. Flip 80 without 81 and the char node renders indextts2 while the ledger and credits say kokoro: a silent wrong render. `tests/test_full_workflow_v2_audio_wiring.py:171-193` asserts node 80 literally AND every other node equals its INPUT_TYPES-derived default, so flipping 81's saved value also requires `_LEGACY_FIRST_ENGINES["char_voice"]` index 0 to become kokoro (`_otr_engine_profiles.py:47-56`). It does NOT touch `eng_indextts2.py`. Pinned-order tests: `tests/test_batch_character_voices.py:109,139`.

2. "Every generated variant" means editing profiles, not JSON variants: variants are generated from `slot_overrides` through `config/profiles/widget_mapping.json:59-100`; 79 profiles pin `char_voice_engine indextts2`, 86 pin `voice_bank default`, including the 5080's own bench and the profile the overnight loop actually runs. Edit by line, never a JSON round-trip; the 4060-proven profiles already say kokoro + kokoro_builtin.

3. Prestartup may not import torch, so the npz cannot be built there. The npz build belongs in the ENGINE at first `load()` (disk-only, allowed by C-7). Gate the ONNX model fetch with `importlib.util.find_spec` at prestartup only; inside `eng_kokoro.load()` keep try-import selection, because `tests/test_audio_engine_adapters.py:119-136` fakes `sys.modules["kokoro"]` with a `SimpleNamespace` and `find_spec` on such an entry raises `ValueError`.

4. The requirements marker must carry an UPPER bound or PBUG-20260901-04 repeats on 3.14: kokoro-onnx declares `<3.14`. Recommended: `kokoro-onnx>=0.6.1; python_version >= "3.13" and python_version < "3.14"`, complementary to `kokoro>=0.7.16; python_version < "3.13"`. Drop a separate `onnxruntime` line.

5. Four profiles pair `voice_bank: "default"` with `char_voice_engine: "kokoro"` and that raises at CastLock today: `cpu_floor`, `otr_mac_mps`, `otr_amd8_rocm`, `otr_amd16_rocm` (`char_kokoro_v1.allowed_voice_banks: [kokoro_builtin]`, `audio_engine_profiles.yaml:190`; `_resolve_char_engine` raises).

6. Byte-identity on the 5080 is a proof, not a property: move the torch path VERBATIM into a torch backend and prove it with a same-seed announcer line sha256 before/after on the 5080 venv.

7. No-network-in-render: `tests/test_kokoro_voice_prefetch.py:101-115` bans the literal strings `hf_hub_download` / `snapshot_download` anywhere in `eng_kokoro` source, including error messages. Pre-existing: on a fresh 3.12 box the TORCH path still networks mid-render for the 327 MB `.pth` at first `KPipeline()`.

Simpler internal layout (same arc): keep `eng_kokoro.py` as the thin registry adapter and add `nodes/_otr_audio_engines/_kokoro_backends.py` with two objects exposing `synthesize(text, voice_id, speed) -> float32 ndarray @ 24000` and `close()`.

## 2. THE FORKS

(i) Device policy -> (c), CPU by design, with a visible receipt; build the session yourself with `providers=["CPUExecutionProvider"]`. (a) is impossible under DONE WHEN: the canonical stamps `voice_device="cuda"` and `_voice_device_from_ledger` falls back to `"cuda"`.
(ii) fp32 `onnx/model.onnx`: the measured variant, size parity with the `.pth`; q8f16 exists for WebGPU/GPU.
(iii) Flip the canonical and the stranger-facing class variants; do NOT silently flip the operator's own bench: the overnight loop runs `otr_writer_bank_gate.py` with `DEFAULT_PROFILE = "otr_w45_still_flat"` (`scripts/otr_writer_bank_gate.py:59`), which pins char indextts2 -- a profile flip would move his dailies off Lemmy's qualified IndexTTS2 route. His call.
(iv) Build ONE npz from the `.pt` files, in the engine, lazily; pass voice ids as strings; keep the `.pt` as the identity source.

## 3. MUST-FIX / SHOULD-FIX / NICE

MUST-FIX: node 81 flip with node 80; `_LEGACY_FIRST_ENGINES["char_voice"]` kokoro first; try-import selection in `load()`, `find_spec` only at prestartup; ONNX prefetch gated and offline-respecting, copy-not-symlink, destination under `TTS/KokoroTTS`; no torch / numpy / kokoro imports at prestartup; explicit CPU provider; marker with the 3.14 upper bound plus test; `profile_python_issue` flags `>= (3, 14)` and the matrix regenerates; the four profiles get `kokoro_builtin`; the 5080 sha256 proof.

SHOULD-FIX: backends module; `create()` kwargs pinned; log before the 326 MB download; models dir via `folder_paths.models_dir` at prestartup with the three-up fallback; stamp the active backend where a human sees it; README 3.13 guidance rewritten; PBUG-20260901-04 follow-up line.

NICE: `hf_hub_download(local_dir=...)`; quiet phonemizer's mismatch logger; `OTR_KOKORO_ONNX_PROVIDERS` override; rename the prefetch module; prefetch the torch `.pth` for 3.12 boxes.

## 4. WHAT TO TEST

Unit: backend selection (four cases); ONNX session is CPU-only regardless of env / requested_device; ONNX ignores the ledger device but logs it; `create()` kwargs pinned and the output contract; npz cache (built, rebuilt on mtime, read-only fallback, missing id still raises); prefetch gating and no heavy imports in the prefetch module; the mid-render-fetch ban extended to the backends module; requirement markers per interpreter; every profile's `voice_bank` allowed for its `char_voice_engine`; canonical node 80 slots 3/4 equal nodes 81/82 in the canonical and every variant; the existing integrity gates.

Live proof A (5080, 3.12, torch): `backend=torch` in the log; a 1-act canonical leg publishes; same-seed announcer sha256 unchanged.
Live proof B (clean 3.13 portable): `pip install -r requirements.txt` succeeds with kokoro-onnx present and kokoro absent; first boot fetches the ONNX model into `models/TTS/KokoroTTS/...`; `backend=onnx provider=CPUExecutionProvider`; a 1-act canonical episode with kokoro announcer AND characters publishes; zero hub requests after boot (`HF_HUB_OFFLINE=1` for the render); per-line RTF under 0.3.

## 5. WHAT I COULD NOT VERIFY

kokoro-onnx internals beyond the driver's measurements; the 5080 venv's numpy (driver: 2.4.4); onnxruntime CPU determinism across thread counts; `folder_paths.models_dir` vs the three-up path on ComfyUI Desktop; the four broken profiles were read, not run; whether another pack ships `onnxruntime-gpu`; registry installs on 3.13 will not receive kokoro-onnx until the next pyproject bump.
