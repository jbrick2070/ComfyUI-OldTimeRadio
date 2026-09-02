# Driver anchor -- kokoro-onnx backend: the default voice that installs everywhere

Driver: Claude (Fable 5.1) on the RTX 5080 window, 2026-09-02. Queue item 2,
`docs/GO_FORWARD_PLAN.md` Section 1.1. Every claim below was checked against the real
Windows files or measured on this box today; the measurements are in section 2.

## 0. The problem, in one paragraph

`kokoro` (the torch package) cannot be pip-installed on Python 3.13 (PBUG-20260901-04:
0.7.16 pins `numpy==1.26.4`, every newer kokoro / misaki declares `Requires-Python <3.13`,
`misaki[en]` drags spacy with no 3.13 wheels). ComfyUI Desktop and the portable ship
Python 3.13, so the pack's DEFAULT announcer voice does not install on the mainstream
Windows path, and the README tells 3.13 users to switch three dropdowns to bark. The
operator ruled 2026-09-01: kokoro-onnx is the go-to, and "we can ship all audio lanes
with kokoro". Same Kokoro-82M model and voices, ONNX Runtime instead of torch, plain
wheels on 3.10-3.13, Windows / Linux / macOS.

## 1. What exists today (grounded)

* `nodes/_otr_audio_engines/eng_kokoro.py` (247 lines): `KokoroEngine`, `interface =
  "per_line"`, `roles = ("announcer_voice", "char_voice")`, `default_roles =
  ("announcer_voice",)`, `speed = 0.95`, `sample_rate = 24000`, lang `'b'` (British).
  `begin_episode` picks one announcer voice per episode through the voice bank and
  verifies `<models>/TTS/KokoroTTS/voices/<id>.pt` exists (C-7: never fetch in a render).
  `load()` builds `KPipeline(lang_code="b", device=requested_device or "cuda", repo_id=...)`
  -- the device is EXPLICIT (S4 rule, file header: "a device the host cannot provide
  fails LOUD in KPipeline -- never a silent downgrade to a 10x slower backend").
  `generate_voice` refuses a voice id with no local `.pt` (no-fallback rip), splits on
  `\n+`, concatenates, peak-normalizes to 0.9, returns `{"waveform": [1,1,T],
  "sample_rate": 24000}`. `unload` moves the model to CPU and empties the CUDA cache.
* `nodes/_otr_voice_node_common.py:838-872`: the voice node does `assert_usable(engine,
  role)` (registry-only, no IO), `adapter = get_engine(engine)`, sets
  `adapter.requested_device` from the CastLock ledger stamp (`meta.voice_device`), sets
  `adapter.role`, renders per line, and tears the adapter down in `finally`.
* `nodes/_otr_audio_engines/registry.py:214-219`: kokoro CAPABILITIES row --
  `device_backends ["cuda", "cpu", "mps"]`, `practical_without_gpu True`,
  `model_requirements ["kokoro-82m"]`, no sidecar, no vendor.
* `nodes/_otr_kokoro_voice_prefetch.py`: at PRESTARTUP (`prestartup_script.py:92`,
  `prefetch_at_boot`), fetches the 28 English voices (`ENGLISH_VOICES`, British +
  American) from `hexgrad/Kokoro-82M` as `voices/<id>.pt` into `<models>/TTS/KokoroTTS/`.
  Never fatal, honours `HF_HUB_OFFLINE=1` and `OTR_SKIP_KOKORO_PREFETCH=1`. Duplicates the
  model subdir constant on purpose (boot must not import the engine).
* `nodes/cast_lock.py:65-70`: `voice_bank` values include `kokoro_builtin` (a PRESET
  bank: `_resolve_char_engine` returns None for it, so the requested char engine is used
  with its own pool, no reference WAVs). `_CHAR_VOICE_ENGINES` includes `kokoro`.
* `workflows/otr_canonical.json` node 80 (`OTR_CastLock`) `widgets_values`:
  `['default', 'auto_registry', True, 'indextts2', 'kokoro', 'cuda']` = voice_bank,
  cast_voice_policy, (bool), char_voice_engine, announcer_voice_engine, voice_device.
  `tests/test_full_workflow_v2_audio_wiring.py:189` pins exactly that list.
* Profiles (`config/profiles/*.json`, `slot_overrides`): variants are GENERATED from
  profiles through `config/profiles/widget_mapping.json` (`char_voice_engine` ->
  CastLock AND BatchCharacterVoices `engine`; `announcer_voice_engine` -> CastLock AND
  AnnouncerVoice `engine`). 79 profiles pin `char_voice_engine indextts2` and 86 pin
  `voice_bank default`. The 5080's overnight loop (`scripts/otr_overnight_loop.sh:82-84`)
  runs `scripts/otr_writer_bank_gate.py` with its `DEFAULT_PROFILE = "otr_w45_still_flat"`
  (:59), which pins `voice_bank default / char indextts2 / announcer kokoro` -- so the
  dailies use the qualified IndexTTS2 (Lemmy) route for characters, and a CANONICAL flip
  changes nothing about tomorrow's obs; only a PROFILE flip would, and that is the
  operator's call. `otr_nvidia_8gb_haunted` (shipping) and `otr_5080_haunted_12b_overnight`
  already run kokoro on both slots with `kokoro_builtin`; `otr_4060_floor` pins bark.
  FOUR profiles pair `voice_bank default` with `char_voice_engine kokoro` (`cpu_floor`,
  `otr_mac_mps`, `otr_amd8_rocm`, `otr_amd16_rocm`) and `char_kokoro_v1.allowed_voice_banks`
  is `[kokoro_builtin]` (`config/audio_engine_profiles.yaml:190`), so CastLock raises
  `VoiceCastingError` for them today (`cast_lock.py` `_resolve_char_engine`) -- the exact
  Mac / AMD / CPU rows the ruling is about.
* `workflows/otr_canonical.json` node 81 (`OTR_BatchCharacterVoices`) carries its OWN
  `engine` widget (`['indextts2']`) and node 82 (`OTR_AnnouncerVoice`) `['kokoro']`; the
  cross-widget agreement check in `_otr_voice_node_common.py:953-964` covers the
  ANNOUNCER only, so flipping node 80 without node 81 renders indextts2 characters while
  the ledger says kokoro. The wiring test derives every node but 80 from INPUT_TYPES,
  and the char combo's first entry comes from `_LEGACY_FIRST_ENGINES["char_voice"]`
  (`nodes/_otr_engine_profiles.py:47-56`, index 0 `indextts2`; `legacy_first_engines()`
  feeds `build_engine_combo`).
* `tests/test_audio_engine_adapters.py:119-136` fakes `sys.modules["kokoro"]` with a
  `SimpleNamespace` (no `__spec__`), so backend selection inside `load()` must use
  try-import, not `importlib.util.find_spec` (which raises ValueError on such an entry).
* `tests/test_kokoro_voice_prefetch.py:101-115` bans the literal strings
  `hf_hub_download` / `snapshot_download` anywhere in `eng_kokoro`'s source (including
  error messages); the torch path's missing-voice message uses `huggingface-cli download`.
* `scripts/otr_provision.py:1710-1724` `profile_python_issue` flags any profile selecting
  kokoro on Python >= 3.13 ("cannot be installed on 3.13"); `scripts/otr_machine_matrix.py`
  consumes it and `tests/test_machine_matrix_drift.py` pins the generated output.
* The 5080 venv's numpy is 2.4.4 (kokoro-onnx needs >= 2.0.2).
* Voice bank (`nodes/_otr_voice_bank.py:880-899`): the kokoro announcer pick is a seeded
  draw over the whole kokoro pool within the chosen gender; characters for a preset
  engine draw from that engine's pool with the one-voice-per-character invariant.
* Dev box (RTX 5080, Python 3.12 venv): `kokoro 0.9.4` installed (torch path),
  `onnxruntime 1.29.0` and `onnxruntime-gpu 1.24.4` present, `kokoro-onnx` NOT installed.
  Weights on disk: `C:\ComfyUI-Models\TTS\KokoroTTS\kokoro-v1_0.pth` (327 MB) and 28
  `voices/*.pt`.

## 2. Measured today (the facts the design rests on)

* `kokoro-onnx 0.6.1` (PyPI): `Requires-Python >=3.10,<3.14`; deps `espeakng-loader>=0.2.4`,
  `numpy>=2.0.2`, `onnxruntime>=1.20.1`, `phonemizer>=3.4.0`; extra `[gpu]` adds
  `onnxruntime-gpu` (x86_64, not darwin). Wheel read: `Kokoro(model_path, voices_path)`
  and `Kokoro.from_session(session, voices_path)`; `create(text, voice: str | ndarray,
  speed=1.0, lang="en-us"|"en-gb", trim=True, sentence_pause=0.25, clause_pause=0.1)`
  returns `(float32 samples, 24000)`; `voices_path` must exist and is `np.load`ed (npz);
  `_style_for(voice, n)` returns `voice[min(n, len(voice)) - 1]`, i.e. a voice is an
  array `(510, 1, 256)` indexed by phoneme count; the session introspects the model's
  input names (`input_ids` or `tokens`) and dtypes, so any v1.0 export works; providers:
  `ONNX_PROVIDER` env wins, else every available provider when an accelerated
  onnxruntime distribution is installed, else CPU; tokenizer = `phonemizer` over the
  bundled espeak-ng (`espeakng_loader.get_library_path()`), `lang` `en-gb` supported.
* The pack's `.pt` voices load with `torch.load(weights_only=True)` as float32 tensors of
  shape `(510, 1, 256)` -- the SAME layout kokoro-onnx indexes. Converted all 28 to one
  npz in one pass; `Kokoro` accepted it (`get_voices()` = 28).
* ONNX model files on the Hub, `onnx-community/Kokoro-82M-v1.0-ONNX` (ungated), folder
  `onnx/`: `model.onnx` fp32 326 MB, `model_fp16.onnx` 163 MB, `model_q8f16.onnx` 86 MB,
  `model_quantized.onnx` 92 MB, `model_uint8.onnx` 177 MB, `model_q4.onnx` 305 MB,
  `model_q4f16.onnx` 155 MB. The repo's own README documents `input_ids` / `style` /
  `speed` and `voices/*.bin` as `(510, 1, 256)` float32.
* Scratch Python 3.13.12 venv (uv) on this box, CPU only (plain onnxruntime 1.29):
  `Kokoro(model.onnx fp32, voices npz)` loads in 0.9 s; `tokens input: input_ids`;
  bm_george, speed 0.95, lang en-gb: 7.72 s of audio in 1.23 s wall (RTF 0.16);
  bf_emma passed as an ndarray: 6.46 s in 0.90 s (RTF 0.14). Peaks 0.54 / 0.75 before
  normalization. WAVs in the scratchpad (`konnx/onnx_bm_george.wav`, `onnx_bf_emma.wav`).

* **5080 torch baseline, PRE-CHANGE (the byte-identity proof's left-hand side):** two
  fixed-seed announcer lines through the current `eng_kokoro` on the 3.12 venv, cuda,
  `torch.manual_seed(1234)`: `bm_george` sha256 `1aac53d5...539e15` (201600 samples),
  `bf_emma` `4811d6b8...488d1ed` (175800 samples); a second run reproduced BOTH digests
  exactly, so run-to-run the torch path is deterministic and sha-equality is the right
  proof. Receipt: `docs/2026-09-02-kokoro-onnx/5080_torch_baseline_sha256.json`.
* On the 5080, `ComfyUI-Installs\ComfyUI\ComfyUI\models\TTS` is a JUNCTION (reparse tag
  0xa0000003) onto `C:\ComfyUI-Models\TTS`, which is why the engine's `folder_paths.models_dir`
  path and the prefetch's three-up path hold the same files there. The fp32 ONNX model is
  now also at `<models>/TTS/KokoroTTS/onnx/model.onnx` on this box for the scratch test of
  the ONNX code path (`OTR_KOKORO_BACKEND=onnx`); the 5080 ships on torch regardless.

## 3. The shape (driver's position -- the panel critiques this)

> **Superseded where they disagree with sections 8 and 9 (r1 / r2 verdicts):** S1
> (selection details), S3 (model path and fetch gate), S4 (provider selection), S8
> (marker shape). Sections 8 and 9 are the plan of record; read them first.

**S1. Backend selection lives inside `eng_kokoro.py`; the registry row, the voice node
and the ledger do not change.** `load()` resolves the backend once per process:
`OTR_KOKORO_BACKEND` env (`auto` default | `torch` | `onnx`); under `auto`, `kokoro`
importable -> torch `KPipeline` exactly as today (the 5080 stays byte-identical); else
`kokoro_onnx` importable -> ONNX; else `EngineUnusable(MISSING_MODEL-class reason)`
naming BOTH installs (`pip install kokoro` on <=3.12, `pip install kokoro-onnx onnxruntime`
anywhere). `generate_voice` and `begin_episode` keep one code path; only the synthesis
call differs (`_synth_torch` / `_synth_onnx`), both returning float32 mono at 24 kHz
before the shared peak-normalize.

**S2. One voice source.** The ONNX backend reuses the `.pt` voices the boot prefetch
already places. At `load()` it builds `<models>/TTS/KokoroTTS/voices/_onnx_voices.npz`
from every `*.pt` present (rebuilt when the `.pt` set or mtimes change; `torch.load
(weights_only=True)` -- torch is always present under ComfyUI), hands that path to
`Kokoro(...)`, and passes each line's voice BY NAME. The cast ledger's voice ids
(`bm_george`, `bf_emma`, ...) are unchanged; no second voice download; the C-7 guard
(voice `.pt` must exist locally) stays exactly where it is.

**S3. One new asset: the ONNX model.** `onnx-community/Kokoro-82M-v1.0-ONNX`
`onnx/model.onnx` (fp32, 326 MB -- the same size class as today's 327 MB `.pth`),
placed at `<models>/TTS/KokoroTTS/onnx/model.onnx`. Fetched at PRESTARTUP by
`_otr_kokoro_voice_prefetch` (extended: `prefetch_kokoro_onnx_model()`, same never-fatal
/ offline-respecting shape), ONLY when `kokoro` is not importable in the boot
interpreter (that is the machine that will use ONNX). Never mid-render: a missing model
at `load()` is `EngineUnusable` with the offline fetch command, like a missing voice.
The `.pth` is not needed on an ONNX box. Open for the panel: fp32 (326 MB, exact CPU
quality) versus `model_q8f16.onnx` (86 MB) as the shipped default -- the driver's
position is fp32 first because it is the export kokoro-onnx targets and the size
matches the torch path; the smaller files are a later, measured option.

**S4. Device: the ONNX backend is CPU by design.** Reasons: an 82M model at RTF ~0.15
on CPU is faster than the render around it; the 8 GB tier's GPU is owed to the video
engine; `onnxruntime-gpu` needs matching CUDA / cuDNN DLLs a portable does not carry
and is not something a newcomer should have to debug for a voice. The CastLock
`voice_device` stamp (`cuda`) is logged ONCE at load as "ignored by the ONNX backend
(CPU by design)" -- loud, documented, deliberate, not a fallback. If a box does have an
accelerated onnxruntime installed, kokoro-onnx's own provider resolution uses it; the
pack neither requires nor configures that. README and the registry note say so. The
alternatives the panel should weigh: (a) fail loud on `cuda` requested + CPU only (the S4
wording), (b) warn and run CPU. The driver reads S4's target as the 10x-slower torch
CPU waterfall; RTF 0.15 is not that.

**S5. Synthesis parity, not byte parity.** `create(text, voice=<id>, speed=0.95,
lang="en-gb")`; split the line on `\n+` as today, concatenate, peak-normalize to 0.9.
Pronunciation differs between misaki (torch) and espeak-ng (ONNX); a machine runs one
backend consistently, so an episode is internally consistent. `sentence_pause` /
`clause_pause` stay at kokoro-onnx defaults (the torch path has its own pauses).

**S6. Ship scope (the ruling), same change:** `workflows/otr_canonical.json` node 80
widgets -> `['kokoro_builtin', 'auto_registry', True, 'kokoro', 'kokoro', 'cuda']`; every
variant regenerated (`scripts/build_variants.py --all` then `--check`); the profiles that
pin something else keep it (`16gb_full` indextts2, `otr_4060_floor` bark); the four
integrity checks run. `tests/test_full_workflow_v2_audio_wiring.py:189` updates. The
registry's `default_roles` are NOT touched (that would mean editing the hash-pinned
`eng_indextts2.py`); the shipped default is the canonical's SAVED value, which is what
a template loads. `voice_device` stays `cuda` (the torch backend honours it; the ONNX
backend logs it as ignored).

**S7. The generated "Voice engines" table.** `scripts/otr_machine_matrix.py` emits a
table into `docs/MACHINE_MATRIX.md` from `registry.CAPABILITIES` (device_backends,
requires_sidecar, requires_vendor, practical_without_gpu, model_requirements) plus a
"ships with the pack / install it yourself" column derived from a small explicit map in
the generator (kokoro, bark, musicgen, stable_audio_3 ship; indextts2 / chatterbox /
dia / google_tts / elevenlabs are install-it-yourself, with the pointer). README's
section 3 points at that table instead of hand-keeping it.

**S8. Dependencies.** `requirements.txt`: `kokoro-onnx>=0.6.1` and `onnxruntime>=1.20.1`
(all platforms, plain wheels); the existing `kokoro>=0.7.16; python_version < "3.13"`
line stays (torch path where it installs). `pyproject.toml` gets the same two lines on
the NEXT registry bump after alpha.15 resolves (Active or Flagged) -- never while a
version is Pending; the kokoro-onnx weights do not ship in the registry zip.

**S10. Every bank and media lane casts fully on kokoro, with the right genders
(operator requirement, 2026-09-02: "be sure all the voices and genders are mapped for
all media / source bank usage").** Grounded today: `config/voice_reference_bank.json`
carries all 28 English kokoro voices (15 female, 13 male; `roles` announcer + char;
`style_tags` preferred_announcer / british_leaning; `ref_path` = the `.pt` file);
`nodes/_otr_casting.py:149` `_VALID_GENDERS = {"male", "female", "other"}`; the bank
serves "other" through `gender_agnostic_fallback_ref` (`_otr_voice_bank.py:786`); cast
ceilings are 6 on the legacy banks (`_otr_casting.py:1176`) and 10 on `scifi_news_pro`
(`_otr_scifi_news_pro.py:351`); `tests/test_kokoro_char_voices.py` already proves depth
per gender and distinct voices per character. What the item ADDS: a coverage test that
walks every bank in `nodes/story_packs/banks.json`, casts that bank's ceiling with a
mixed gender roster (male / female / other, the announcer both ways) under `voice_bank
kokoro_builtin` + `char_voice_engine kokoro`, and asserts every cast row resolves to a
distinct kokoro `voice_ref_id` of the requested gender (or the documented fallback for
"other"), whose `.pt` name is in `ENGLISH_VOICES`; plus one assertion that every bank
entry with `engine kokoro` names a voice the prefetch fetches (no orphan ids either
way). The ONNX backend must also reject an id with no local `.pt` exactly as the torch
path does, so a bank/prefetch drift fails by name, never by a silent voice swap.

**S9. Not in scope:** Mac `mps` claims (kokoro's registry row already lists mps for the
torch path; the ONNX path is CPU); per-voice blending; streaming; the kokoro-onnx
`[gpu]` extra; a kokoro-onnx path for the 5080 (it keeps torch).

## 4. What must stay true (the invariants a reviewer should try to break)

1. The 5080 (3.12, `kokoro` importable) selects torch and produces byte-identical audio
   for a fixed seed before and after -- measured, not asserted (section 6).
2. No network inside a render: `load()` and `generate_voice` never download; missing
   model / voice -> `EngineUnusable` with an offline command.
3. The ledger contract is unchanged: same `voice_ref_id` names, same
   `{"waveform","sample_rate"}` return, same per_line interface, same teardown.
4. A box with neither package fails by NAME at the first voice line, not at import (the
   engine module imports nothing heavy at module scope -- C-5).
5. `prestartup_script.py` stays never-fatal.

## 5. Tests (unit, CPU, no network)

* `tests/test_kokoro_backend_select.py`: with fake `kokoro` / `kokoro_onnx` modules in
  `sys.modules`, `auto` picks torch when both exist, onnx when only onnx exists, raises
  a named `EngineUnusable` when neither; `OTR_KOKORO_BACKEND=onnx` forces onnx even with
  torch present; `=torch` with no torch package raises by name.
* ONNX synthesis contract: a stub `Kokoro` records `create(...)` kwargs -- voice by
  name, `speed == 0.95`, `lang == "en-gb"`, one call per `\n+` segment, concatenation
  and 0.9 peak; the returned dict shape; `requested_device="cuda"` logged as ignored,
  never raised.
* Voices npz cache: built from the `.pt` set, rebuilt on mtime change, every array
  `(510, 1, 256)` float32; the C-7 missing-voice refusal still fires before any backend
  loads.
* Prefetch: `prefetch_kokoro_onnx_model()` is a no-op when `kokoro` imports, when the
  model exists, under `HF_HUB_OFFLINE=1`; fetches exactly `onnx/model.onnx` from the
  pinned repo otherwise; never raises (the existing prefetch tests' shape).
* Workflow: the four integrity checks (`build_variants --check`,
  `test_widget_value_alignment`, `test_canonical_widget_input_parity`,
  `test_workflow_link_target_indexes`) plus the updated wiring pin.
* Matrix: the generated table is deterministic and `--check` stays in sync.

## 6. Live proofs (the DONE WHEN)

* **5080 (3.12, torch):** fixed-seed single-line render before and after the change,
  sha256-identical; then a 1-act canonical episode (any profile that runs kokoro) to
  `otr/obs/` with `obs_publish OK`.
* **4060 clean room (portable, Python 3.13.14):** `pip install kokoro-onnx onnxruntime`
  into the portable interpreter, pull the clone to the commit, boot (prefetch places the
  ONNX model and the voices), run the 1-act leg through `workflows/otr_canonical.json`
  with BOTH voice slots on kokoro, `RESULT SUCCESS` + `obs_publish OK` + the file in the
  clean room's obs; the server log shows `backend=onnx` and the CPU note once.

## 8. r1 (Fable, cold) -- what survived grounding and is now the plan of record

Every item below was checked against the files named in section 1 before folding.

* **Ship flip is nodes 80 AND 81, plus the combo order.** Node 80 ->
  `['kokoro_builtin', 'auto_registry', True, 'kokoro', 'kokoro', 'cuda']`; node 81 ->
  `['kokoro']`; node 82 stays `['kokoro']`. `_LEGACY_FIRST_ENGINES["char_voice"]` gets
  `kokoro` at index 0 (indextts2 second; nothing else reordered), so the INPUT_TYPES
  default and the saved value agree and the wiring test's derivation holds. Update
  `tests/test_full_workflow_v2_audio_wiring.py:188-189`, `tests/test_batch_character_voices.py`
  (:109, :139 pin the order) and, if the yaml rank/is_default is touched (it is NOT in
  this change), `tests/test_engine_profiles_rank_gate.py`. `eng_indextts2.py` untouched.
  New test: node 80 slots 3/4 equal node 81/82 `engine` in the canonical and every variant.
* **Profiles: fix the four broken rows, do not mass-flip.** `cpu_floor`, `otr_mac_mps`,
  `otr_amd8_rocm`, `otr_amd16_rocm` get `voice_bank kokoro_builtin` (line edits, never a
  JSON round-trip); regenerate variants (`--all`, then `--check`). New invariant test: for
  every profile with `char_voice_engine` set, `voice_bank` is in that engine's
  `allowed_voice_banks`. The 79 indextts2-pinned lab profiles stay; the stranger path is
  already kokoro through `config/machine_classes.json` (all four classes declare
  `char_voice kokoro`). Whether the 5080's own bench / overnight profiles
  (`otr_w45_*`, `otr_soak_*`, `otr_rot_*`, `otr_sbcov_*`, `otr_w45_still_flat`) move off
  the IndexTTS2 Lemmy route is the OPERATOR'S question, asked once, not decided here.
* **Backends split:** `nodes/_otr_audio_engines/_kokoro_backends.py` with two objects
  exposing `synthesize(text, voice_id, speed) -> float32 mono @ 24000` and `close()`;
  the torch code moves VERBATIM. `eng_kokoro.py` stays the thin adapter (seeded pick,
  C-7 guards, `\n+` split, concat, 0.9 peak) and selects once per process via
  `OTR_KOKORO_BACKEND=auto|torch|onnx` with try-imports (`from kokoro import KPipeline`,
  else `import kokoro_onnx`), logging ONE `[OTR.kokoro] backend=<x> provider=<p>` line. A
  forced backend that cannot import fails loud by name; never falls through.
* **Voices npz is built in the ENGINE at first `load()`, never at prestartup** (boot must
  not import torch / numpy -- verified in the installed ComfyUI, `main.py:230`
  `apply_custom_paths()`, `:236` `execute_prestartup_script()`, `:245` the "torch should
  never be imported before this point" check): from `torch.load(weights_only=True)` over the `.pt` set,
  written beside the voices with an atomic rename, rebuilt when any `.pt` is newer,
  temp-dir fallback when the voices dir is read-only. Voices passed BY NAME; the per-line
  `.pt` existence guard stays and an "id present in npz" check is added. The `.pt` stays
  the identity source (`_provisional_identity_fingerprint` hashes it; the bank names it);
  the npz is derived state and says so in its docstring.
* **ONNX model prefetch at prestartup, gated without importing anything heavy:**
  fetch only when `OTR_KOKORO_BACKEND == "onnx"` or (`find_spec("kokoro") is None` and
  `find_spec("kokoro_onnx")` is not None); honour `HF_HUB_OFFLINE=1` and
  `OTR_SKIP_KOKORO_PREFETCH=1`; log a line BEFORE the 326 MB download starts (a silent
  first boot reads as a hang under the 5-minute rule); resolve the models dir through
  `folder_paths.models_dir` when importable (ComfyUI applies custom paths before
  prestartup) with the three-up fallback; `hf_hub_download(..., local_dir=<KokoroTTS>/
  onnx-community)` so the file lands at `<KokoroTTS>/onnx-community/onnx/model.onnx`
  with no second HF-cache copy on an 8 GB laptop. The engine's missing-model message uses
  the `huggingface-cli download onnx-community/Kokoro-82M-v1.0-ONNX onnx/model.onnx
  --local-dir <...>/onnx-community` phrasing (the banned literals never appear in
  `eng_kokoro`). Pre-existing and recorded, not fixed here: on a fresh 3.12 box the torch
  path still fetches the 327 MB `.pth` at first `KPipeline()`.
* **Device (fork i) -> CPU by design, made visible:** build the session explicitly,
  `Kokoro.from_session(onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"]),
  voices_npz)` -- never kokoro-onnx's auto "all available providers" (a box carrying
  `onnxruntime-gpu` from another pack would try unqualified CUDA DLLs). One INFO line at
  load names the provider and the ignored ledger stamp. `OTR_KOKORO_ONNX_PROVIDERS` (a
  comma list) is the only override. S4 stays intact for the torch backend. Reason (a)
  cannot work: the canonical stamps `voice_device cuda` and `_voice_device_from_ledger`
  defaults to `cuda`, so a fail-loud ONNX path would refuse the canonical on every 3.13
  box, and flipping the canonical to `cpu` would move the 5080's torch path to CPU.
* **Model (fork ii) -> fp32 `onnx/model.onnx`**, the variant measured today, size parity
  with the `.pth`; q8f16 is an fp16-activation export for GPU/WebGPU, not for a CPU
  backend. `OTR_KOKORO_ONNX_MODEL` overrides the filename at most.
* **`create()` kwargs pinned explicitly:** `voice=<id>`, `speed=0.95`, `lang="en-gb"`,
  `trim=False` (the torch path does not trim), `sentence_pause=0.25`, `clause_pause=0.1`
  (they apply only when kokoro-onnx has to split a >510-phoneme chunk). A library default
  change cannot move the house cadence; an A/B listen against the torch path is a
  later, operator-ear decision.
* **Requirements markers (fork on install shape):** `kokoro-onnx>=0.6.1; python_version
  >= "3.13" and python_version < "3.14"`, complementary to the existing `kokoro>=0.7.16;
  python_version < "3.13"`, so each interpreter gets exactly one backend and 3.14 does not
  repeat PBUG-20260901-04 (kokoro-onnx declares `<3.14`). No separate `onnxruntime` line
  (kokoro-onnx's metadata carries it). `tests/test_requirements_python_markers.py` gains
  the sibling assertions (3.12 false / 3.13 true / 3.14 false; exactly one backend line
  true per interpreter). `pyproject.toml` mirrors both on the next bump.
* **Provisioner / matrix:** `profile_python_issue` flags kokoro at `>= (3, 14)` instead
  of `(3, 13)`, with the message rewritten (kokoro-onnx on 3.13, CPU); update
  `tests/test_otr_provision_profile_routes.py:106-108`; regenerate `docs/MACHINE_MATRIX.md`
  and the README block; README's hand-written "bark on 3.13" guidance (section 3, the
  Pick-the-graph rows, troubleshooting) is rewritten in the same change.
* **The 5080-unchanged proof is measured:** same-seed announcer line, sha256 before and
  after, `OTR_KOKORO_BACKEND` unset, on the 3.12 venv; printed in the commit message.
* **Also noted, not fixed here:** registry installs on 3.13 will not receive kokoro-onnx
  until the next `pyproject.toml` bump; until then the engine reports itself unusable
  naming both pip lines and the voice falls to the dropdown choice (bark).

## 9. r2 (Cursor, grok-4.6-high, coding plan) -- what survived grounding

Verified against the files before folding; these AMEND section 8 where they differ.

* **The torch call is ONE `KPipeline(text, voice=..., speed=..., split_pattern=r"\n+")`
  call over the FULL line, exactly as today (`eng_kokoro.py:233-235`).** The adapter
  does not pre-split for the torch backend (a different call shape would fail the 5080
  sha256 proof). Backend interface: `synthesize(text, voice_id, speed) -> float32 mono
  @ 24000`, each backend owning its own splitting: torch via `split_pattern`, ONNX by
  splitting on `\n+` itself, skipping empty segments, one `create()` per segment. The
  adapter keeps only the shared concat + 0.9 peak.
* **No `_LEGACY_FIRST_ENGINES` reorder.** Index 0 `indextts2` is pinned by
  `tests/test_engine_profiles.py:108`, `tests/test_batch_character_voices.py:108-112,
  139-141` and the "byte-identical default combo" comment, and reordering would make a
  menu-added BatchCharacterVoices default to kokoro while a menu-added CastLock (defaults
  `voice_bank default`, `char_voice_engine auto`, `cast_lock.py:255-286`) still resolves
  indextts2 -- with no char-side agreement check to catch it. Instead: node 81's saved
  value is pinned in `tests/test_full_workflow_v2_audio_wiring.py` the same way node 80
  is special-cased, AND the char-side agreement check is added to
  `_otr_voice_node_common.py` (cloned from the announcer guard at :953-964: CastLock's
  stamped `char_voice_engine` must equal the BatchCharacterVoices `engine` widget, else
  `EngineUnusable MALFORMED_CONFIG` naming both controls). That closes the silent-wrong-
  render hole r1 found without touching the default combo.
* **Prefetch destination = the engine's dir, resolved the same way.** The engine reads
  `folder_paths.models_dir/TTS/KokoroTTS` (`eng_kokoro.py:48-58`); the prefetch today
  walks three-up to `<comfy>/models` (`_otr_kokoro_voice_prefetch.py:80-93`). On the 5080
  both resolve to the install's `models/` (verified: `ComfyUI-Installs/ComfyUI/ComfyUI/
  models/TTS/KokoroTTS/voices` holds the 28 `.pt`; the headless launcher's
  `--extra-model-paths-config` adds `C:/ComfyUI-Models` as a search path but does not
  move `models_dir`). ComfyUI imports `folder_paths` before `execute_prestartup_script()`
  (`main.py:230/236`), so the prefetch resolves `folder_paths.models_dir` inside a
  try/except FIRST and falls back to three-up; that is what keeps a `--base-directory`
  (Desktop) install coherent. One relative-path constant for the ONNX file, duplicated in
  the prefetch and the engine exactly as `_KOKORO_MODEL_SUBDIR` already is, with the
  existing prefetch-dir test extended to it. The 5080's `C:\ComfyUI-Models` copy of the
  ONNX model made today is a scratch artifact, not the engine's path.
* **Errors are `EngineUnusable`, never raw tracebacks:** missing package (today a bare
  `RuntimeError`, `eng_kokoro.py:149-152`) -> `EngineUnusable(MISSING_MODEL)` naming
  both pip lines; missing ONNX file, npz build failure (`os.replace` included), and
  phonemizer / espeak / onnxruntime failures at synthesis -> `EngineUnusable` with the
  reason classified. A single corrupt `.pt` is skipped and logged when building the npz
  (that voice then fails by name at its line); the build never aborts every voice for one
  file. `unload()` calls the backend's `close()` (drop the session; `InferenceSession`
  has no close method) then the existing gc / empty_cache.
* **Selection is re-evaluated in every `load()`** (env re-read, try-imports; cheap), so
  tests that swap `sys.modules` and `OTR_KOKORO_BACKEND` need no reset seam, and there is
  no cross-prompt session singleton: the voice node unloads after every generate
  (`_otr_voice_node_common.py:872-874`, I-7) and a re-load is 0.7-0.9 s.
* **`OTR_KOKORO_ONNX_PROVIDERS`:** empty / whitespace raises (never `providers=[]`,
  which means "all"); unknown names raise. Default `["CPUExecutionProvider"]`, and
  `intra_op_num_threads` capped (min(4, cpu_count)) so a 16-thread session does not
  fight the video encode; the RTF is re-measured under that cap.
* **Tests the plan had omitted:** `tests/test_otr_provision_profile_routes.py:104-118`
  (pins "cannot be installed on Python 3.13" twice) and the matrix's `Python <=3.12`
  recipe suffix (`scripts/otr_machine_matrix.py:289`, probed at `(3, 13)`) move to a
  `(3, 14)` probe with the message rewritten; `config/profiles/otr_rot_tts_bark.json`
  pairs `voice_bank default` with `char_voice_engine bark` while `char_bark_v1`
  allows only `[bark_legacy]` (`audio_engine_profiles.yaml:46`) -- fixed to
  `bark_legacy` in the same change so the new invariant test is green from its first
  run; the marker test keys by `Requirement.name`, and `kokoro` / `kokoro-onnx` are
  distinct names, so the sibling assertions live beside the existing ones.
* **Cache fingerprint:** `_provisional_identity_fingerprint` hashes the `.pt` bytes
  (`_otr_voice_node_common.py:66-101`) and would serve cached audio across an ONNX model
  swap. The kokoro fingerprint gains the active backend name and the ONNX file's size +
  mtime (never a 326 MB hash per queue).
* **C-5:** `_kokoro_backends.py` imports nothing heavy at module top (`eng_kokoro` is
  imported at package init, `_otr_audio_engines/__init__.py:28`); the
  `hf_hub_download` / `snapshot_download` ban test is extended to the backends file.
* **CUT, agreed:** the S4 fail-loud fork; a separate `onnxruntime` requirement line; a
  process-lifetime session; fetching the ONNX model when neither package is importable.
* **Deferred as NICE:** `OTR_KOKORO_ONNX_MODEL` filename override (ship `onnx/model.onnx`
  only); the 3.14 probe in the matrix beyond the message change.

## 10. r3 (Antigravity, Gemini 3.7 Flash High, wiring) -- what survived grounding

* **The char-side agreement guard must ignore `auto`.** Verified: `cast_lock.py:1598-1611`
  stamps `meta["char_voice_engine"] = "auto"` when the request is `auto` and the
  resolver returns nothing (preset banks `kokoro_builtin` / `bark_legacy` do exactly
  that), so the guard is `if stamped and stamped != "auto" and stamped != engine: raise`.
  The announcer guard compares a resolved engine (`_resolve_announcer_engine`), which is
  why it never sees `auto`.
* **One ONNX path, the simple one:** `_KOKORO_ONNX_REL_PATH = os.path.join("onnx",
  "model.onnx")` under `TTS/KokoroTTS`, duplicated in the prefetch and the engine like
  `_KOKORO_MODEL_SUBDIR`; the prefetch calls `hf_hub_download(repo, "onnx/model.onnx",
  local_dir=<models>/TTS/KokoroTTS)` so the file lands at `<models>/TTS/KokoroTTS/onnx/
  model.onnx` (section 8's `onnx-community/` nesting is dropped). The missing-model
  message uses the matching `huggingface-cli download onnx-community/Kokoro-82M-v1.0-ONNX
  onnx/model.onnx --local-dir <models>/TTS/KokoroTTS` line.
* **`find_spec` at prestartup is wrapped:** `_spec_exists(name)` returns False on ANY
  exception (a `sys.modules` fake without `__spec__` raises ValueError); prestartup stays
  never-fatal.
* **npz keys are bare voice ids** (`os.path.splitext(basename)[0]`), never `bm_george.pt`.
* **No replace-in-place on Windows (supersedes section 8's atomic-rename mechanism).** `np.load(npz)` keeps a zip handle open for the
  session's life, so `os.replace` onto a live npz raises `PermissionError`. The npz is
  named by a digest of the `.pt` set (names + sizes + mtimes): `_onnx_voices.<digest>.npz`.
  A changed set produces a NEW file; a matching digest is reused; stale siblings are
  removed opportunistically when nothing holds them (errors ignored). Build order: npz
  first, session second; `unload()` drops the backend and the session before any rebuild.
* **Wiring test:** `test_widget_vectors_exact` asserts node 81 `== ["kokoro"]` explicitly
  and skips `nid in (80, 81)` in the derived-defaults loop (node 80's docstring rationale
  is rewritten: the canonical now selects kokoro for characters).
* **Cache busting through the hook that exists:** `_otr_voice_node_common.py:760-777`
  folds `get_engine(engine).render_time_params()` into `IS_CHANGED` for every role and
  returns the literal `"static"` when it is empty. `KokoroEngine.render_time_params()`
  returns `{}` under the torch backend (the 5080's caching behaviour stays byte-for-byte)
  and `{"backend": "onnx", "onnx_model": "<size>:<mtime>"}` under ONNX. The
  `_provisional_identity_fingerprint` change from section 9 is dropped in favour of this.
* **`self.role` fallback:** `getattr(self, "role", "announcer_voice")` wherever the engine
  raises `EngineUnusable`, so a direct call outside the voice node cannot AttributeError.
* **3.14:** `profile_python_issue` flags `>= (3, 14)` with "neither kokoro nor kokoro-onnx
  is packaged for Python 3.14 yet; use 3.13 (kokoro-onnx, CPU) or 3.12 (kokoro)"; the
  marker test asserts mutual exclusion across 3.12 / 3.13 / 3.14 (exactly one backend line
  true on 3.12 and 3.13, none on 3.14).
* **CUT, agreed:** GPU provider negotiation; a standalone `onnxruntime` requirement line.
  NICE deferred: `OTR_KOKORO_ONNX_MODEL`; a `.pt` NaN/inf sweep.

## 11. r4 (Sonnet 5, convergence) -- CONVERGED

No MUST-FIX open. Three documentation nits fixed above (the section-8 citation, the npz
supersession note); r2's S10 test guidance is already implemented and committed as
`tests/test_kokoro_covers_every_bank.py` (f2312570). Two low-stakes r1 items get a
disposition here: the PBUG-20260901-04 follow-up line is written with the live receipt
(proof B); the phonemizer "words count mismatch" logger is quieted only if it actually
fires in proof B's log. Record: `r4_sonnet_convergence.md`.

## 7. Review roster (operator: one strategic partner per round, Codex/Cursor sparingly)

r1 arc: Fable, cold (this anchor withheld). r2 coding plan: Cursor (`agent`) or Codex if
its CLI is on the box (it is not on PATH today). r3 wiring: Antigravity (`agy`).
r4 convergence: one Sonnet 5 read of the finished diff (Sonnet QA is the post-code gate
regardless). Every claim is grounded against the files before it is folded in.
