<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The promotion path still depends on incompatible in-process voice adapters, and the provided Stable Audio adapter is not the SA3-native path the plan says is conflict-free.

MUST-FIX BEFORE BUILD:
1. [Objective / Options B-D / F-G1 / eng_indextts2.py / eng_chatterbox.py] The plan says IndexTTS2/Chatterbox cannot be installed in the main venv, but both adapters still import and run `indextts` / `chatterbox` inside the ComfyUI process. That cannot be promoted on the stated torch 2.10+cu130/numpy 2.x stack. Concrete fix: choose either A/D for the first sprint and leave Bark as the voice default, or implement B fully by replacing these adapters with main-process sidecar clients. The ComfyUI process must not import `indextts`, `chatterbox`, their torch, or their numpy at all.

2. [Stable Audio 3 / D5 / Verify-at-build / eng_stable_audio.py] The existing music adapter is not Stable Audio 3 native. It imports `stable_audio_tools`, defaults to `"stabilityai/stable-audio-open-1.0"`, and names the engine `"stable_audio_music"`, while the blocker says `stable-audio-tools` will not resolve and the intended default is ComfyUI-native `stable_audio_3`. Concrete fix: do not promote `eng_stable_audio.py` as SA3. Add/replace with a `stable_audio_3` adapter that wraps the ComfyUI-native SA3 path, uses local checkpoint/profile resolution, and has no `stable_audio_tools` dependency.

3. [C-6 / C-7 / D / D5 / registry.py / eng_indextts2.py / eng_chatterbox.py / eng_stable_audio.py] Missing deps/models/tokens are not surfaced through the six-class fail-closed taxonomy. Current adapters raise plain `RuntimeError` or may call model loaders directly; `registry.assert_usable()` also treats default engines as usable without disk/token checks. Concrete fix: queue-time validation must classify absent dependency/model/token as `EngineUnusable(..., MISSING_MODEL|MISSING_HF_TOKEN|MALFORMED_CONFIG, ...)`, and adapters must not emit unclassified `RuntimeError` for expected install/model absence.

4. [CLEAN-BREAK directive vs C-5 / C-6 / I-1 / I-10 / D / R0b / H-I] The document gives mutually exclusive default/fallback instructions: “No permanent legacy fallback,” “new engines default,” “literal legacy-first fallback list,” “internal build default stays legacy until F,” “opt-in workflow,” and “legacy permanent fallback” all coexist. This is not implementable without different builders choosing different defaults. Concrete fix: collapse this into one explicit state machine: pre-promotion defaults, promotion defaults, whether legacy adapters remain selectable, and whether legacy node instances are forbidden. Remove superseded lines instead of relying on prose supersession.

5. [Options B / F] The sidecar option is not specified enough to build, but it is the only plausible path for IndexTTS2/Chatterbox on this main venv. Concrete fix before any sidecar build: define the IPC contract, process lifecycle, request/response schema, audio transfer format, seed fields, timeout/error mapping, local model-path validation, stderr capture, Windows process cleanup, and VRAM residency policy. If sidecar is not selected, explicitly defer IndexTTS2/Chatterbox promotion.

6. [Options B / F] The plan assumes a GPU sidecar with torch 2.8-cu128 may work on sm_120, but the blocker only says “MIGHT.” A sidecar using torch 2.8 that cannot run Blackwell still fails the release goal. Concrete fix: make F prove GPU execution on sm_120 before declaring sidecar viable, or specify CPU-only voice sidecar as the first supported mode with performance limits accepted.

7. [G1 / Output-correctness gates / eng_indextts2.py / eng_chatterbox.py / eng_stable_audio.py] External-generator determinism is not enforced. The adapters pass `generator=` through `supported_kwargs()`, which can silently drop it, and the shown classes do not define a checked `supports_external_generator` value. Concrete fix: bit-exact mode must fail closed if the real callable does not accept and use a bound generator. The pilot should set a verified capability in the adapter/profile, and runtime should reject engines without it.

8. [Offline-first / C-7 / D5 / eng_chatterbox.py / eng_stable_audio.py] Runtime model loading may fetch from the network. `ChatterboxTTS.from_pretrained(device="cuda")` and `get_pretrained_model(model_id)` are not shown as local-only and have no path/hash preflight. Concrete fix: require profile-resolved local model directories/checkpoints, validate them before execute, pass local-only flags if the library supports them [ASSUMPTION], and fail with a named error instead of fetching.

9. [I-7 / D / eng_stable_audio.py] Stable Audio teardown is insufficient for the VRAM gate. `eng_stable_audio.py.unload()` only sets `_model = None`; it does not `del`, `gc.collect()`, or `torch.cuda.empty_cache()`. Concrete fix: every promoted GPU music path must release model references and empty CUDA cache in the node `finally` before emitting `done` / video gate. If using a sidecar, process termination is the teardown.

10. [D5 / Engine matrix / eng_stable_audio.py] Engine naming is inconsistent. The execution plan and objective refer to `stable_audio_3`, while the grounded adapter registers `stable_audio_music`. Concrete fix: use one canonical engine id everywhere: profiles, workflow widgets, registry, cache keys, release metadata, and tests. For SA3, that should be `stable_audio_3` if that is the product default.

11. [Objective / Engine matrix / I-8] IndexTTS2 is planned as the shipped character default while `eng_indextts2.py` marks `commercial_clean = False` and the document says Bilibili written authorization is required for commercial use. That may be intentional for personal/local use, but it contradicts the request for a commercial-clean path. Concrete fix: either state that the shipped default is non-commercial-warning-only, or do not make IndexTTS2 the commercial/default voice; use Chatterbox/Bark until a clean engine is validated.

SHOULD-FIX:
1. [eng_chatterbox.py] `load()` hardcodes `device="cuda"`. That blocks CPU-sidecar fallback and may fail on machines where the GPU sidecar is invalid. Fix: make device profile-driven and support CPU mode if B is retained.

2. [eng_indextts2.py] `OTR_INDEXTTS2_DIR` is allowed to be empty and then passed as `IndexTTS2("", "")`. Fix: validate directory, config file, and expected weight files before constructing the model, and classify failures as `MISSING_MODEL`.

3. [eng_chatterbox.py] `supported_kwargs()` is given both `cfg=0.5` and `cfg_weight=0.5`. If the real API accepts both, both will be sent; if they are aliases, behavior may be ambiguous. Fix: pilot the real signature and keep one canonical parameter.

4. [eng_indextts2.py / eng_chatterbox.py] Both adapters reseed global torch/CUDA RNG inside `generate_voice()`. The plan says `deterministic_inference` saves/restores RNG in `finally`, but that code is not shown. Verify: RNG state is restored around every call; otherwise audio generation can perturb later engines.

5. [eng_indextts2.py `_as_waveform`] Return normalization does not enforce `[B,C,T]`, mono/stereo policy, dtype, CPU/GPU placement, or sample-rate consistency. Fix: centralize this through `pack_audio_batch`/canonical audio before returning to consumers.

6. [Stable Audio 3 verify-at-build] The plan assumes ComfyUI Desktop has ComfyUI >= v0.22.0 and native SA3 nodes available. Verify: actual installed ComfyUI Desktop version and node availability before making SA3 the music default.

7. [R0b / F] Box-fresh smoke uses stubs and “no weights,” while promotion defaults intentionally require weights/tokens. Fix: add a separate promoted-default validation mode that proves missing default weights produce the named queue-time error without breaking ComfyUI import or graph validation.

8. [I-7 / E.5] The “gate_signal edge into first video loader” is necessary but not sufficient if ComfyUI preloads models before execution due to graph validation or module globals. Verify: video model load occurs only after the gate input is consumed, not during import/validation.

9. [F] “OFFLINE only here” is ambiguous for pilots that need first-time model acceptance/download. Fix: separate one-time acquisition step from offline render pilot; the pilot itself should run with network disabled.

10. [D / registry.py] `engines_for_role()` sorts alphabetically after default status. If multiple adapters accidentally claim the same role default, dropdown order becomes arbitrary by engine name rather than failing. Fix: add a guard test that exactly one default exists per role at promotion.

OPTIONAL / NICE-TO-HAVE:
- Add a tiny sidecar health-check command returning engine version, model hash, device, dtype, and deterministic capability before any line generation.
- Record the sidecar venv lockfile hash in `audio_meta`/baseline metadata.
- For Windows sidecars, use explicit UTF-8 JSON lines, absolute paths, and kill the whole process tree on timeout.

CUT THESE (over-engineering):
1. [Wave orchestration / parallel-safety rules] Cut parallel multi-worktree execution for the first sprint. The current blocker is dependency architecture, not implementation throughput; serial work avoids shared venv/cache/log races and reduces review surface.

2. [R0a legacy baseline steps d-f, if Clean-Break truly supersedes them] Do not build a legacy invocation manifest and legacy audio baseline if the first sprint is SA3-only or sidecar architecture and legacy is not being promoted. Keep only the guard that proves existing Bark/Kokoro/MusicGen still render.

3. [H native stereo deferred / stereo-policy machinery] Keep music stereo preservation only where needed for SA3; defer broader native-stereo voice abstractions until a stereo voice engine exists.

4. [Slim migration framework beyond reject-and-rerender] The plan already says old registry ledgers can be rejected. Do not build additional migration/quarantine paths until there is a real pre-v2 archive requirement.