# OTR SHIP LIST -- Synthesis of 71 confirmed + 51 disputed findings
Date: 2026-09-01. Sources: 71 CONFIRMED (grounded, not refuted), 51 DISPUTED (grounded, refuted by a second reader). Duplicates merged by file+line or root cause; every original file:line retained. Severities are as the grounders assigned them; nothing softened.

---

## 1. REGISTRY FLAG -- ranked causes and the exact first change

State of the listing, live-checked 2026-09-01: `GET /nodes/comfyui-old-time-radio/versions` returns exactly two rows, `2.0.0-alpha.13` (2026-08-30T07:42:36Z) and `2.0.0-alpha.14` (08-30T08:02:10Z), **both `NodeVersionStatusFlagged`**. The node record is `NodeStatusActive` but carries **no `latest_version` key at all** (a healthy pack such as comfy-sidebar carries `latest_version.status = NodeVersionStatusActive`). That missing key is exactly why ComfyUI Manager reports "not a CNR node / cannot resolve install target". There is no Active version and no rollback target: the earlier node hard-delete destroyed alpha.8's row. (registry-flag-02, pyproject.toml:3)

The gate is a **secret scanner**, not a generic linter: `services/registry/registry_svc.go:1403` calls `sendScanRequest(s.config.SecretScannerURL, ...)`, and ANY non-empty response body sets `Flagged` (1434-1436); empty sets Active (1418-1423).

### Ranked candidate causes

1. **`README.md:164` ships an `hf_`-token-shaped literal.** The line is `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` -- `hf_` plus 38 characters, inside gitleaks' documented huggingface rule window `hf_[a-zA-Z0-9]{34,40}`. This is the only shipped string in the bundle that matches a published secret-detection pattern. (registry-flag-01, high)
2. **The scanner's ruleset changed under a tree that did not.** Every other structural candidate was ruled out by byte-diffing the FLAGGED alpha.14 against the ACTIVE alpha.8: wildcard CORS and the unauthenticated POST render routes (`__init__.py:430`), the prestartup `sys.modules` swap / `sys.path.insert` / boot HF download (`prestartup_script.py:42,91,94`), `ctypes.WinDLL` (`nodes/_otr_gguf_backend.py:1019`) and `__import__` (`nodes/_otr_video_engines/eng_ltx25.py:287`), and the four dotenv-shaped `workflows/variants/*.env.json` files carrying 64-hex `master_hash` values -- **all present, in identical counts, in the version the scanner marked Active.** They cannot be the delta.
3. **`.comfyignore` itself ships inside the scanned zip and narrates the pack's own scanner triggers** (`.comfyignore:10` "comfy-cli's security scan flagged exec() calls...", `:21` "security scanner is built to flag.", plus a tabulation `subprocess 33/33  Popen 9/9  base64 5/5  urlopen 2/2 ... ctypes 2/2  marshal 1/1`). Low probability as a cause, but it is build metadata shipping to end users and it hands a reviewer a list of the pack's own hits. (registry-flag-06, low)
4. **`workflows/external_examples/video_humo_native_unlimited_workflow.json:1021`** redistributes a third party's absolute path with a 32-hex account identifier (`/data/ComfyUI/personal/445aa9a89d7fa5567450f91c998092da/output/...`). Present in alpha.8 too, so not the delta, but it is third-party content carrying someone else's identifier through this pack. (registry-flag-12, low)

### Exact minimal change to try first

**Publish `2.0.0-alpha.15` containing exactly two edits plus the dependency batch:**

1. `README.md:164` -> `hf_your_token_here` (18 chars after the prefix, below every scanner floor; the same placeholder already used at README.md:240 and :246; reads identically to a human).
2. `pyproject.toml` version bump (this is what fires the publish Action) plus, in the SAME edit because pyproject is a release trigger and must not be touched twice: add `"pycairo>=1.24"`, `"pillow>=10.0"`, `"aiohttp>=3.9"`, and change `"bitsandbytes>=0.42.0"` to `"bitsandbytes>=0.42.0; sys_platform != 'darwin'"`.

**If alpha.15 comes back Flagged, run the control:** republish the alpha.8 tree (commit e44235f5) byte-identical as alpha.16. The node hard-delete freed every old version string, so that content can be resubmitted unchanged. Active -> the trigger is in the alpha.9+ delta and can be bisected. Flagged -> the scanner's ruleset moved, no archaeology on this repo will find it, and that result is the evidence to hand Comfy-Org.

Do not version-delete anything (soft delete burns the string permanently); node-delete is the only clean slate, and it needs the operator's browser session.

---

## 2. SHIP BLOCKERS (must fix before the next registry version)

* **`pyproject.toml:22` -- `"bitsandbytes>=0.42.0"` carries no environment marker anywhere in pyproject.toml or requirements.txt (grep for platform/sys_platform/darwin returns nothing in either file), and it sits in the static `[project].dependencies` list the registry reads literally.** Fix: `"bitsandbytes>=0.42.0; sys_platform != 'darwin'"` in both files; the loader already fails loud when bnb is missing while `quant_policy` demands it (`nodes/_otr_model_loader.py:806-812`), so the marker costs no safety. Hits: Mac (install cannot resolve), probably AMD/ROCm. Owner: 5080. (deps-drift-01; the sibling mac-amd-01 was refuted only on the wheel-availability wording, not on the missing marker.)
* **`pyproject.toml:3` -- no Active registry version and no rollback target; the node record has no `latest_version`, so Manager cannot resolve an install target at all.** Fix: the alpha.15 / alpha.16 sequence in section 1. Hits: every machine that installs from the registry. Owner: 5080. (registry-flag-02)
* **`config/profiles/otr_g4_ltx_8gb.json:40` -- the only two shipping profiles that name the genuinely-8GB `ltx_8gb` engine (`:4` status=shipping, `:24` engine=ltx_8gb) pair it with `llm.vram_ceiling_gb: 14.5`, a Mistral-Nemo/12B writer budget nearly 2x an 8 GB card; `config/profiles/otr_w45_ltx_8gb.json` has the identical mismatch at the same three lines.** Fix: ship an 8GB-appropriate writer (GGUF row, ceiling ~6.8) paired with `ltx_8gb`, and rewrite the checkpoint-missing text at `nodes/_otr_video_engines/eng_ltx_8gb.py:733,1088,1115,1280` to name `Lightricks/LTX-Video` + `ltxv-2b-0.9.8-distilled.safetensors` + a models-root destination instead of `scripts/download_ltx_0_9_8.ps1`, which never ships. Hits: 4060, registry. Owner: 4060 owns the 8GB profile once proven; 5080 owns the `nodes/` error text. (profiles-lanes-02)
* **`workflows/otr_canonical.json:1376` saves `indextts2` as the default character-voice engine, and `config/voice_reference_bank.json:4` states outright that `VALIDATE_INPUTS deliberately accepts at queue time` while `ref_path` (e.g. `:18` `models/TTS/refs/indextts2/vz_bill_boerst.wav`) is verified only at render time -- and zero .wav files ship anywhere in the pack, with no `download_url`/`repo_id` in any of the 20 indextts2 rows.** So a box-fresh graph validates clean, writes an entire episode, and then dies at the voice stage. Fix: either ship license-clean reference WAVs, or make the shipped default profiles use kokoro/bark for `char_voice_engine` (as `otr_nvidia_8gb_haunted` and `otr_4060_12b_gguf_offload` already do), or pre-flight the resolved `ref_path` files in `OTR_CastLock` BEFORE the writer LLM call. Hits: all, 4060, RunPod, Mac. Owner: 5080 (workflows/, config/). (profiles-lanes-03 + gap-cold-start-default-graph-01)
* **`nodes/_otr_workflow_validator.py:293` -- `_detect_host()` maps Darwin->"mac", Windows->"win", and everything else to "any"; Linux can never produce "linux".** Both `config/profiles/otr_amd8_rocm.json:5` and `otr_amd16_rocm.json:5` declare `"platform": "linux"`, and the stamp assertion at `:373-377` then aborts the prompt on every Linux host and suggests `cpu_floor`. Executed and confirmed: faked `platform.system()` -> Linux yields `{'platform':'any'}`. Fix: add an explicit `"linux" if sysname == "Linux"` branch; give `nodes/_otr_shared/host_caps.py:22-50` the same token; add a test that every profile's declared platform is reachable from `_detect_host` on the OS it names. Hits: AMD, RunPod. Owner: 5080. (mac-amd-02 + gap-rocm-linux-never-detected-01)
* **`nodes/_otr_model_loader.py:740` -- `load_llm()` runs `gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()` unconditionally (lines 737-740), with no `is_available()` guard and no try/except, unlike every neighbouring block in the same function (`:700`, `:710-717`, `:743`) and its own failure-path cleanup at `:1158-1163`.** `empty_cache()` is a safe no-op off CUDA; `ipc_collect()` calls `_lazy_init()` and raises "Torch not compiled with CUDA enabled". This kills the canonical graph's writer stage on Mac before device selection is consulted. Fix: wrap the pair in `if torch.cuda.is_available():` plus try/except -- byte-identical behaviour on the 5080. Hits: Mac (and any CPU-only fallback). Owner: 5080. (gap-cold-start-default-graph-03, merged with mac-amd-06 and gap-model-loader-unguarded-cuda-wash-02, same lines)

---

## 3. HIGH (fresh install silently degrades, or a lane crashes)

* **`pyproject.toml:14` / `nodes/_otr_shared/scope_draw.py:629` -- `pycairo` is declared only in `requirements.txt:116`, never in the static pyproject list (lines 14-46) that the registry reads.** `import cairo` is unguarded at `scope_draw.py:629` and `nodes/_otr_video_engines/eng_viz_mandala.py:224`. Same drift class carries `pillow>=10.0` (`requirements.txt:118`) and `aiohttp>=3.9` (`:120`), added 2026-08-31 and never propagated; PIL is imported 22 times across shipped nodes, 16 unguarded (e.g. `nodes/otr_credits_roll.py:612,705,738,911,1122,1199`). This is a literal repeat of the kokoro/pyloudnorm cold-install bug the project already paid for once. Fix: add all three in the alpha.15 bump. Hits: all, RunPod, registry. Owner: 5080. (deps-drift-02 + node-load-01 + mac-amd-08 + deps-drift-03 + node-load-02 + gap-image-pillow-not-in-pyproject)
* **`nodes/_otr_audio_engines/eng_chatterbox.py:76`, `eng_dia.py:78`, `eng_indextts2.py:183` -- all three sidecar TTS engines resolve their worker as `os.path.join(_REPO_ROOT, "scripts", "_otr_*_worker.py")`, and `scripts/` stopped shipping at alpha.12 (alpha.8 zip: 119 files under scripts/; alpha.14: 0).** Their "not installed" messages then point at Windows-only `.ps1` installers that also do not ship (`eng_chatterbox.py:94-95`, `eng_dia.py:99`, `eng_indextts2.py:199`). Fix: either negate the six needed files back into `.comfyignore`, or refuse cleanly at registration naming the GitHub repo instead of a path the install does not contain. Hits: all, RunPod, Mac, registry. Owner: 5080. (registry-flag-08 + hardcoded-paths-04)
* **`nodes/_otr_openrouter_backend.py:452` -- 16 user-facing message sites cite `docs/` files that `.comfyignore` strips from every install** (also `:1140`; `nodes/OTR_LedgerScriptWriter.py:2114,2134,2421,2442,2469,2484`; `nodes/_otr_model_catalog.py:287,294,334,341,1386,1397`; `nodes/_otr_comfy_backend.py:441,501`). Fix: point at the GitHub URL, or inline the two setup steps, or carve `docs/openrouter-setup.md` + `docs/comfy-credits-setup.md` out of the exclusion. Hits: all. Owner: 5080. (dead-code-01)
* **`nodes/_otr_video_engines/eng_ltx_8gb.py:733` (also `:1088`, `:1115`, `:1280`) and `nodes/_otr_upscale_engines/eng_spandrel_esrgan.py:249` -- missing-model errors tell the user to run `scripts/download_ltx_0_9_8.ps1` and `python scripts/ensure_upscale_models.py`; `unzip -l otr_alpha14.zip | grep -c scripts/` = 0.** The spandrel engine is the ONLY shipped upscale engine, and it already holds `_model_sha256` as a class constant, so the URL + path + hash can be inlined verbatim. Hits: 4060, RunPod, all, registry. Owner: 5080. (dead-code-02 + gap-image-upscale-error-cites-unshipped-scripts)
* **`nodes/_otr_image_engines/flux_gen1.py:238` -- `flux_gen1` is the only image engine with `default_roles = ROLES` (`:95`), i.e. the shipped default for every image role, and its `assert_usable` is `return self.name` with no disk check and no host_caps check.** Its three siblings (z_image_turbo, lumina_image, flux2_klein) all do a real presence check. A fresh install therefore reaches a deep unnamed crash inside `wrapper_bridge`/`CheckpointLoaderSimple` instead of greying out. Hits: all, Mac, 4060, RunPod, AMD. Owner: 5080. (gap-image-flux-default-noop-assert-usable)
* **`README.md:16` (repeated at `__init__.py:411-413`) -- the headline quick start is "Workflow -> Browse Templates -> EXTENSIONS -> comfyui-old-time-radio -> otr_canonical", but `example_workflows/` contains exactly one file, `otr_4060_floor.json`.** The template gallery is populated from `example_workflows/`, and `pyproject.toml [tool.comfy]` declares no template override. Every new user's first documented action fails. Fix: add `example_workflows/otr_canonical.json`, or lead with the drag-the-JSON path. Hits: all, registry. Owner: 5080. (fresh-install-docs-01)
* **`nodes/_otr_gguf_backend.py:708` and `:1182` -- `default_layers = DEFAULT_N_GPU_LAYERS if policy.device == "cuda" else 0`, with `DEFAULT_N_GPU_LAYERS = -1` at `:67`.** `config/profiles/otr_mac_mps.json:36` sets `"device": "mps"`, so Apple Silicon offloads ZERO layers and runs the Q4_K_M 12B writer entirely on CPU while Metal idles. Fix: `in ("cuda", "mps")`; keep 0 for cpu; `OTR_GGUF_N_GPU_LAYERS` stays the override. Confined to the gguf backend, cannot reach the 5080. Hits: Mac. Owner: 5080. (mac-amd-05 + gap-gguf-ngpu-layers-mps-forced-zero-03)
* **`nodes/_otr_gguf_backend.py:1028` -- `from llama_cpp import Llama` is the ONLY local writer lane a Mac profile allows (`config/profiles/otr_mac_mps.json:52` lane_allowlist is gguf/openrouter/comfy_credits/google_api), llama-cpp-python is declared in neither manifest, and the failure text at `:1039-1040` gives only the CUDA 12.4 wheel index.** `README.md:103` repeats that same CUDA-only command in the very paragraph (`:96-99`) that names `otr_mac_mps`, `otr_amd8_rocm`, `otr_amd16_rocm`, `cpu_floor` as the audience. Fix: branch the message on `sys.platform` (keep the Windows 0.3.33 pin and DLL note verbatim -- they are hard-won), add per-platform install lines to README, and declare it as an optional extra rather than a base dependency. Hits: Mac, AMD, 4060, registry. Owner: 5080. (mac-amd-04 + docs-gguf-mac-rocm-01 + fresh-install-docs-06)
* **`nodes/_otr_shared/capability_profiles.py:496 -- `_fit_reason` reads device_backends, practical_without_gpu, requires_vendor, required_toolchain and requires_sidecar, and nothing else; `needs_fp8_te` / `needs_fp4_te` are shape-validated at `:449-450` and never consulted.** fp8 and NVFP4 engines therefore qualify on the ROCm tiers whose own `dtype_policy` says otherwise. Fix: two clauses keyed on the profile's `dtype_policy` (`no_fp8`, `no_fp8_no_fp4`) returning new REASON_NEEDS_FP8 / REASON_NEEDS_FP4. Hits: AMD, Mac. Owner: 5080. (mac-amd-03)
* **`nodes/otr_credits_roll.py:616 -- the font resolver is `os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")` plus 4 Windows paths, 3 bare names and 2 fixed Linux paths; zero macOS candidates and no `OTR_*_FONT` override in the file.** Nothing resolves -> `CreditsDataError` at `:633-635`, no bitmap fallback by design, so the credits tail silently drops every Mac render. Fix: add an `OTR_CREDITS_FONT` override checked first plus macOS candidates; align with `video_engine.py`'s platform-branched resolver. Hits: Mac, RunPod. Owner: 5080. (hardcoded-paths-01)
* **`nodes/_otr_shared/cloud_media_backend.py:202 -- `resolve_cache_root()` defaults to `Path(__file__).resolve().parents[2] / "otr" / "cache" / "cloud_media"`, i.e. INSIDE the installed pack, and `:379-380` puts `billing_ledger.jsonl` there.** A registry-managed install directory is replaced wholesale on update: real-money billing state silently disappears, or the write fails outright on a read-only mount. The module's own comment claims it wires the ComfyUI output-base helper; the code never calls it. Fix: route through `nodes/_otr_paths.otr_shared_cache_dir()`, keeping `OTR_CLOUD_MEDIA_CACHE_DIR` as the override. Hits: RunPod, Mac, AMD, registry. Owner: 5080. (runtime-writes-03)
* **`config/profiles/cpu_floor.json:14-16` and `config/profiles/otr_mac_mps.json:14-16` route announcer_image, music_image AND character_image to `google_image`, a metered paid API, with the mere presence of an env key as the only gate (`nodes/_otr_google_api/client.py:43-56`).** `cpu_floor.json:3` display_name is "CPU floor (no GPU; Mac ships AS this tier)", and `nodes/_otr_workflow_validator.py:350/362/374` actively steers no-CUDA and Mac hosts to it. Fix: surface an explicit profile-level confirmation, or route through a budget ceiling. Hits: Mac, all. Owner: 5080. (gap-google-cloud-surface-01)
* **`nodes/_otr_image_engines/eng_google_image.py:289 -- the BYO-key Google lane calls `create_interaction(...)` directly with no `session.reserve`, no `session.bill`, no `ledger_append`,** unlike every Comfy-Cloud Partner row (`eng_cloud_image.py:340-342` -> `nodes/_otr_shared/cloud_media_invoke.py:774-815`). A runaway multi-beat episode on cpu_floor/otr_mac_mps racks up unbounded paid calls with zero record in this app. Fix: wire it through the same reserve/bill/ledger path, or document plainly that BYO-key spend is outside OTR's ledger entirely. Hits: all, Mac. Owner: 5080. (gap-google-cloud-surface-02)
* **`nodes/_otr_janitor.py:90` -- `_otr_boot_sweep()` runs synchronously on the node-registration path (`__init__.py:581`) and `_entry_mtime` rglobs every descendant of every top-level entry: measured 21,440 `stat` calls, 6.69 s, on every ComfyUI boot, growing without bound.** Its cause is the same defect as the next item. Hits: all, 4060, RunPod, 5080. Owner: 5080. (performance-03)
* **`nodes/_otr_janitor.py:127` -- `sweep_shared_tmp` iterates only top-level entries (`for entry in tmp_root.iterdir():`) and judges each by its NEWEST child (`_entry_mtime`, `:82-99`), so `tmp/audio_slices` -- one long-lived directory that `nodes/_otr_video_engines/render_driver.py:499` writes every per-beat slice into -- can never be swept.** Measured: 21,289 files, 9.3 GB. Proven by execution against a scratch tree: a 40h-old slice beside a fresh one is never selected. Fix: descend one level and apply the cutoff per file; this also removes most of the boot cost above. Hits: all, RunPod, 4060. Owner: 5080. (performance-06 + gap-janitor-audio-slices-dir-granularity-04)
* **`nodes/_otr_video_engines/eng_ltx_video.py:1471 -- `_build_graph` defines the 8 GiB unet (`:1471`), the ~6 GiB text encoder (`:1478`) and the VAE (`:1482`) INSIDE the per-clip graph, and `:1666` runs it fresh per beat with the results local; `eng_ltx_av.py:901-902` is the same shape.** Four sibling adapters already avoid this via a `prepare()` override + `external_results` (`wrapper_bridge.py:410-419`), and `eng_ltx25` already has the episode-scoped `_encoder_scope` / `_encoder_cache_key` pattern to lift verbatim. Hits: all, RunPod, 5080. Owner: 5080. (performance-02)

---

## 4. MEDIUM / LOW, grouped by theme

### Paths and platform
* `nodes/_otr_openrouter_backend.py:780` -- catalog cache defaults to `<repo>/models/openrouter_models.json`, inside the pack; silently and permanently disables caching on a read-only install. MEDIUM. Owner: 5080. (runtime-writes-05)
* `nodes/video_engine.py:991` -- `_find_ffmpeg()` runs `shutil.which` first (fine) but its fallback list is 100% Windows (`%LOCALAPPDATA%\...\ffmpeg.exe`, `C:\ffmpeg\bin\ffmpeg.exe`); nothing for `/opt/homebrew/bin` or `/usr/local/bin`. LOW. Owner: 5080. (hardcoded-paths-07)
* `nodes/_otr_upscale_engines/_resolve.py:63` -- `mps` is rejected as `MALFORMED_CONFIG` while `capability_profiles.py:163` accepts `upscale_stage.device` as any non-empty string; refusal happens at render, not profile-load. LOW. Owner: 5080. (mac-amd-11)
* `tools/make_registry_icon.py:171` -- `out = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\assets\otr_icon.gif"`; `tools/` ships, nothing imports it, but it is a personal-path leak in the shipped tree. LOW. Owner: 5080. (hardcoded-paths-06)
* `docs/RUNPOD_INSTALL.md:87` -- the Step 3 rationale still describes the pre-fix failure ("falls back to `C:\ComfyUI-Models`, which on Linux becomes a literal directory nothing scans"); `nodes/_otr_gguf_backend.py:916-918` now guards that literal on `legacy.is_dir()`. The instruction is still correct; only the reason is stale. MEDIUM. Owner: 4060 (fresh-install path). (fresh-install-docs-05)

### Encoding
* `nodes/_otr_hf_auth.py:99` -- the token-file read is wrapped in `except OSError`, but a BOM-stamped or non-UTF-8 `~/.cache/huggingface/token` raises `UnicodeDecodeError` (a ValueError), which escapes a best-effort fallback. Fix: widen to `(OSError, UnicodeDecodeError, ValueError)` or open with `utf-8-sig`, `errors="replace"`. LOW. Owner: 5080. (encoding-os-03)

### Dead code and dead advice
* `nodes/_otr_content_safety.py:39` -- a full profanity/weapon/nudity word list (`:47-56`) ships with ZERO callers anywhere in `nodes/`, `__init__.py`, `tools/`, `viewer/`, `config/`, and its `to_dict()` is broken (NameError). Its own docstring justifies it by a test file that no install receives. LOW. Owner: 5080. (hygiene-01)
* `viewer/index.html:214` (also `:227`, `:261`) -- fetches `/ledger?latest=1`, `/list`, `/ledger?path=`; the shipped pack registers only `/otr/latest_ledger`, `/otr/video_render_single`, `/otr/video_render_soak` (`__init__.py:436,540,553`). The server that answered those routes stopped shipping at alpha.12. Fix: repoint at `/otr/latest_ledger`, or add `viewer/` to `.comfyignore`. MEDIUM. Owner: 5080. (registry-flag-09)
* `nodes/_otr_image_engines/lumina_image.py:413` -- the missing-model error tells the user to set `OTR_ENABLE_LUMINA=1`, while the class sets `requires_flag = None` and the module docstring (`:28-31`) says that flag is vestigial and read by nothing. Fix: delete the clause. MEDIUM. Owner: 5080. (gap-image-lumina-dead-flag-in-error)
* `nodes/_otr_image_engines/eng_cloud_image.py:278` -- partner-pin error says "re-pin via scripts/otr_pin_partner_nodes.py"; `scripts/` never ships. MEDIUM. Owner: 5080. (gap-image-cloud-error-cites-unshipped-scripts)
* `config/source_banks/banks.json:177` + the matching `OTR_LedgerScriptWriter` tooltip -- `custom_source_bank`'s guide tells the user to run `otr_check bank <path> --activate` and read `docs/EXTENDING_OTR.md`; both are stripped from the bundle. MEDIUM. Owner: 5080. (gap-content-data-integrity-01)
* `nodes/_otr_google_api/models.py:157` -- `google_api_model_cache.json` is read at `:157`, `:174` and `:227-236` and written nowhere in the shipped codebase; the cache-backed dropdown and structured-output fallbacks are permanently dead. LOW. Owner: 5080. (gap-google-cloud-surface-04)
* `config/profiles/cpu_floor.json:75` -- `preflight.required_keys: ["OTR_GOOGLE_API_KEY"]` (identical in `otr_mac_mps.json:73-76`) is shape-validated at `capability_profiles.py:204` and enforced by nothing; only `required_models` has a live consumer. Fix: wire it, or remove the field so it stops implying a guarantee. MEDIUM. Owner: 5080. (gap-google-cloud-surface-03)
* `config/source_banks/_corpus/monkeys_paw.txt:1` -- an orphaned duplicate of the real, wired `sources/monkeys_paw.txt`; nothing under `nodes/` reads `_corpus/`. LOW. Owner: 5080. (gap-content-data-integrity-02)

### Shipped-surface hygiene
* `_START_HERE.md:3` -- ships to every installer and contains only maintainer build routing ("The active build plan lives OUTSIDE this repo at `C:\Users\jeffr\Documents\otr-video-roundtable\`"). Add to `.comfyignore` or replace with a real quick start. MEDIUM. Owner: 5080. (fresh-install-docs-03)
* `SKILL.md:3` -- "For Cowork and any other AI assistant working on this repo", 142 lines of sprint-commit process, ships to every installer. Add to `.comfyignore`. LOW. Owner: 5080. (fresh-install-docs-04)
* `.comfyignore:10` -- ships inside the zip and narrates the pack's own scanner triggers; add `.comfyignore`, `.gitignore`, `.gitattributes` to itself and move the flag-streak prose to `docs/`. LOW. Owner: 5080. (registry-flag-06)
* `workflows/external_examples/video_humo_native_unlimited_workflow.json:1021` -- third-party graphs with a stranger's 32-hex identifier; add `workflows/external_examples/` to `.comfyignore` (~380 KB saved, nothing in `nodes/` loads them). LOW. Owner: 5080. (registry-flag-12)
* `__init__.py:390` -- every node-load failure is logged twice (`log.warning(...)` then `print(f"[OldTimeRadio] Skipped ...")`); same double-emit at `nodes/_otr_video_engines/dmd_sampler.py:120-121`. Pick one channel. LOW. (hygiene-02)
* `nodes/_otr_model_catalog.py:1922` -- print-only, no `import logging` in the file; same in `nodes/_otr_shared/cloud_media_invoke.py:293` and `:765` (both literally labelled WARNING but sent via print) and `nodes/_otr_story_routing.py:597`. LOW. (hygiene-03)
* `nodes/_otr_readiness.py:110` -- `num2words` is optional-imported and declared in neither manifest, so spoken numbers silently degrade to raw digits on any install that lacks it. LOW. (deps-drift-08)
* `nodes/_otr_kokoro_voice_prefetch.py:151` -- `hf_hub_download` with no explicit timeout, called synchronously from `prestartup_script.py:87-96` before any node registers; a slow-but-not-offline connection delays ComfyUI reaching listening state. LOW. (node-load-03)
* `nodes/_otr_image_engines/lumina_image.py:405` -- the only local image engine with no `folder_paths` auto-discovery; `z_image_turbo` and `flux2_klein` both find their default-named file in `models/diffusion_models`, lumina requires `OTR_LUMINA_CKPT`. MEDIUM. (gap-image-lumina-no-folder-paths-autodiscovery)
* `nodes/_otr_image_engines/flux2_klein.py:293` -- `"unet": ("UnetLoaderGGUF",)` depends on the ComfyUI-GGUF pack, which is named nowhere in README, node_list.json, requirements.txt, pyproject.toml or _START_HERE.md, and gets no pack-name hint because `wrapper_bridge._PACK_FOR_PREFIX` (`:89-93`) maps only `ADE_` and `VHS_`. MEDIUM. (gap-image-flux2klein-undocumented-gguf-dependency)

### Tests (all positive, recorded so the dimension is not mistaken for unaudited)
* `tests/conftest.py:75` -- 12,733 tests collect with zero import/collection errors; the shared `folder_paths` stub at `:75-86` pre-empts the documented order-dependent AttributeError class. (test-health-02)
* `tests/test_ltx25_recipe_matches_lab_golden.py:33` -- 5080-only absolute-path fixtures are all guarded with existence checks + `pytest.skip`, so they skip cleanly off-5080 rather than failing red (same in `tests/test_cast_voice_coverage.py:28-29,89`). (test-health-04)
* `tests/test_image_platform_c1.py:943` -- no test writes into the repo tree or `otr/episodes|obs` outside `tmp_path`; the five files referencing those strings use them only in assertions. (test-health-05)
* `config/profiles/widget_mapping.json:1` -- `build_variants.py --check` reports "check: 90 variants, 0 failures"; all five widget/link integrity suites pass 317/317; all 13 managed node types exist in `otr_canonical.json`. The widget machinery is clean. (profiles-lanes-09)

---

## 5. OPTIMIZATIONS worth doing, ranked by 4060 / RunPod benefit

1. **Janitor: descend one level in `sweep_shared_tmp` (`nodes/_otr_janitor.py:127`) and stop rglob-ing in `_entry_mtime` (`:90`).** Reclaims a measured 9.3 GB of `audio_slices` and removes ~6.7 s from every ComfyUI boot. Biggest single win on a RunPod pod (billed by the hour, small disk) and on the 4060. Three-line change.
2. **`eng_ltx_video` / `eng_ltx_av` prepare() override (`eng_ltx_video.py:1471`, `eng_ltx_av.py:901-902`).** Stops reloading ~14 GiB of unet + text encoder + VAE per segment; the `external_results` contract (`wrapper_bridge.py:410-419`) and the ltx25 encoder-scope pattern already exist in-tree. Largest render-time win on any card, and it does not raise the peak VRAM envelope.
3. **Bound `_shared/cache` and `mesh_cache` (`nodes/_otr_paths.py:245`, `eng_mesh_stage.py:533`) -- 27 GB measured, no eviction policy at all.** Add an opt-in `OTR_SHARED_CACHE_MAX_GB` LRU, default OFF on the dev box and SET in the RunPod profile so the janitor's auto-delete contract is not silently widened. Both tiers are content-addressed and cheap to regenerate.
4. **`images_to_uint8` (`nodes/_otr_video_engines/wrapper_bridge.py:593`) -- three full float32 copies per decoded batch (~1.4 GB of transient host RAM per beat where ~450 MB would do).** Keep the first copy, make the rest in-place with `np.multiply(..., out=arr)` / `np.add(..., out=arr)`. Host-RAM relief matters most on an 8 GB laptop and a small pod.
5. **Memoize `OTR_LedgerScriptWriter.INPUT_TYPES` (`nodes/OTR_LedgerScriptWriter.py:2024`) -- profiled at 11.5 ms warm per `/object_info` call, 104 `os.stat` + 12 JSON re-parses, the only node in the pack with a non-zero warm INPUT_TYPES.** A 5-10 s TTL or an mtime-keyed cache changes no behaviour; the builders are already documented as network-free.
6. **Hoist the bars overlay buffer (`nodes/otr_post_upscale_procgen_blend.py:614`) -- a full 1920x1080 PIL allocation plus a numpy copy per frame to paint ~9% of the frame; 4,095 frames on the measured episode.** Draw the strip into a small image and blit into one reusable `np.zeros`.
7. **Cache the negative in `lookup_git_commit` (`nodes/_otr_ledger.py:883`) -- `_GIT_COMMIT_CACHE` is set only on success (`:879`), so every registry install (no `.git`) pays a subprocess spawn per ledger stamp and logs a WARNING for its normal state.** Use an `_UNSET` sentinel; drop the log to DEBUG.

---

## 6. MAC / AMD REALITY

**Mac today: nothing runs.** `pip install` of the pack cannot resolve on macOS at all because `pyproject.toml:22` pins `bitsandbytes>=0.42.0` with no marker in the static list the registry reads. Past that, in order of encounter: `nodes/_otr_model_loader.py:740` raises "Torch not compiled with CUDA enabled" inside `load_llm` before device selection; the only allowed local writer (`otr_mac_mps.json:52`) is GGUF, whose install guidance is a CUDA 12.4 wheel index (`nodes/_otr_gguf_backend.py:1039-1040`, `README.md:103`) and which then runs entirely on CPU because `n_gpu_layers` is forced to 0 off cuda (`_otr_gguf_backend.py:708,1182`); no local image engine declares mps, so stills require a paid `OTR_GOOGLE_API_KEY` (`otr_mac_mps.json:14-16,80`) with no ledger or ceiling; the upscale stage rejects `mps` outright (`_otr_upscale_engines/_resolve.py:63`); and the credits tail cannot resolve a font (`nodes/otr_credits_roll.py:616`).

**AMD/ROCm today: the install may succeed, the render cannot start.** `nodes/_otr_workflow_validator.py:293` can never emit "linux", so both `otr_amd8_rocm.json:5` and `otr_amd16_rocm.json:5` fail their own platform assertion on the host they target and are told to switch to `cpu_floor`. Even once that is fixed, `capability_profiles.py:496` never reads `needs_fp8_te`/`needs_fp4_te`, so fp8 and NVFP4 engines still qualify on tiers whose `dtype_policy` forbids them.

**Minimal path to an audio-only Mac episode (writer + TTS + credits, no video, no stills):**
1. `pyproject.toml:22` -- add `; sys_platform != 'darwin'` (install becomes possible).
2. `nodes/_otr_model_loader.py:737-740` -- guard the wash with `if torch.cuda.is_available():` (writer stage stops crashing).
3. `nodes/_otr_gguf_backend.py:708` and `:1182` -- offload on `("cuda", "mps")` (writer stops being CPU-only).
4. `nodes/_otr_gguf_backend.py:1039` + `README.md:103` -- platform-branch the llama-cpp-python install text (a Metal build line for Darwin).
5. `pyproject.toml` -- add `pycairo`, `pillow`, `aiohttp`; note `pycairo` also needs the macOS system cairo library, which belongs in the mac launch notes.
6. `nodes/otr_credits_roll.py:616` -- add an `OTR_CREDITS_FONT` override plus `/System/Library/Fonts` candidates.
Voice: use kokoro or bark, never the shipped `indextts2` default (`workflows/otr_canonical.json:1376`), which needs reference WAVs that do not exist.

**To add still images on Mac, one of two:** supply `OTR_GOOGLE_API_KEY` and accept metered, unledgered spend, or qualify ONE small local image engine on Apple Silicon and add `"mps"` to its `device_backends` row in `nodes/_otr_image_engines/registry.py` (all local rows -- `:118`, `:123`, `:136`, `:151`, `:157` -- are cuda-only today).

**Minimal AMD path:** the `_detect_host` linux branch (blocker above) plus the `needs_fp8_te`/`needs_fp4_te` clauses in `_fit_reason`. Both are small and both are prerequisites to anything else being measurable.

---

## 7. LOWEST-FRICTION 8 GB LANE

**Name: `otr_nvidia_8gb_haunted`** -- the `animatediff15_*` lane, documented at `README.md:78` as "the 8 GB default". It is the only 8 GB video lane in the pack with a fully documented, ungated, auto-fetching download path, and -- critically -- it is one of the two profiles that already use kokoro/bark for `char_voice_engine`, so it side-steps the indextts2 reference-WAV blocker in section 2. The `ltx_8gb` alternative is currently unusable per profiles-lanes-02, and the two H3 8 GB profiles self-label "UNQUALIFIED LAB/HISTORICAL ... legal floor not yet proven on a physical 8GB card" (`config/profiles/otr_nvidia_8gb_h3.json:3`).

**Model files, as documented in `README.md:88-91` (that table is the authority; sizes verbatim):**

| File | Size | Source |
|---|---|---|
| SD 1.5 base checkpoint | 1.99 GB | auto-fetch, ungated, no token |
| `v3_sd15_mm` motion module | 1.56 GB | auto-fetch (AnimateDiff v3) |
| `v3_sd15` domain adapter | 0.10 GB | auto-fetch |
| Kokoro voices | 0.30 GB | auto-fetch (also prefetched at boot by `prestartup_script.py:87`) |
| Writer `gemma-4-E2B-it` | ~9.6 GB | auto-fetch to `<models_root>\LLM\converted\` |
| `musicgen-small` | ~2.2 GB | auto-fetch |

Total first-run download: **~15.8 GB**. No gated repo, no HF token required for any row.

**Node-pack dependency:** ComfyUI-AnimateDiff-Evolved, which provides the `ADE_*` classes this lane samples through (`README.md:78`). `wrapper_bridge._PACK_FOR_PREFIX` already names it by URL if it is missing, so that failure is self-explaining.

**Exact steps a new user does:**
1. Install ComfyUI-AnimateDiff-Evolved (ComfyUI Manager, or clone into `custom_nodes/`).
2. Install OTR. **Registry install is broken today** (section 1) -- until alpha.15 is Active, clone the repo into `custom_nodes/` on branch `v2.0-alpha`, then `pip install -r requirements.txt` into ComfyUI's python.
3. Restart ComfyUI. Confirm `[OldTimeRadio]` lines in the console and the OTR nodes in the node menu -- that local check is the only trustworthy install test.
4. Drag `workflows/variants/otr_nvidia_8gb_haunted.json` onto the canvas (do NOT use `workflows/otr_canonical.json`, which is stamped for the 5080: cuda, bnb_nf4, 14.5 GB ceiling, indextts2). Note that `README.md:16`'s Browse-Templates route does not offer it -- `example_workflows/` contains only `otr_4060_floor.json` (section 3).
5. Queue Prompt. First run downloads ~16 GB and is slow; subsequent runs are not.
6. Confirm success the way the operator does: an episode lands in `otr/obs/`.

---

## 8. DISPUTED -- operator to rule (51)

Each line: id, anchor, the claim, then the refutation.

1. registry-flag-03, pyproject.toml:39 -- kokoro drags a bare `torch` into the registry dep list, breaching the "torch deliberately absent" contract | refuted: `spandrel~=0.4.1` at :28 already does the same and shipped in the ACTIVE alpha.8, so it is neither new nor the flag.
2. registry-flag-04, __init__.py:430 -- wildcard CORS on a route returning an absolute `fullpath`, plus two unauthenticated POST render routes taking caller-supplied paths | refuted: byte-identical in the Active alpha.8, so not the flag delta (the security merit was not addressed either way).
3. registry-flag-05, prestartup_script.py:42 -- boot-time `sys.modules` swap, `sys.path.insert(0, ...)` and an HF download before any node loads | refuted: identical in the Active alpha.8.
4. registry-flag-10, nodes/_otr_gguf_backend.py:1019 -- `ctypes.WinDLL` by path and `__import__("sys")` at eng_ltx25.py:287 are scanner-visible | refuted: already audited and dismissed in `.comfyignore`'s own 2026-08-28 block; identical counts in alpha.8.
5. registry-flag-11, workflows/variants/otr_8gb_ltx.env.json:4 -- dotenv-shaped filename beside an `env` map and a 64-hex value trips generic secret rules | refuted: the same four files shipped in this exact shape in the Active alpha.8.
6. deps-drift-04, nodes/_otr_gguf_backend.py:1024 -- llama-cpp-python is load-bearing and declared nowhere, with an exact 0.3.33 pin only in a code comment | refuted: deliberately optional; the error text IS the install contract, and a base pin would force a heavyweight compile on every install.
7. deps-drift-05, nodes/_otr_audio_engines/eng_stable_audio.py:36 -- `stable_audio_tools` imported and declared nowhere | refuted: legacy opt-in adapter, `default_roles = ()`, fails with a named error by design.
8. deps-drift-06, nodes/_otr_model_catalog.py:1809 -- `huggingface_hub` imported directly at two unguarded sites, undeclared | refuted: hard transitive of the pinned `transformers>=5.10.4`, guaranteed present.
9. deps-drift-07, nodes/_otr_model_loader.py:436 -- `packaging.version` imported unguarded, undeclared | refuted: `transformers` hard-depends on packaging; unreachable failure.
10. hardcoded-paths-03, nodes/_otr_audio_engines/eng_chatterbox.py:72 -- Windows `.venv/Scripts/python.exe` default with no POSIX branch (same at eng_dia.py:74, eng_indextts2.py:176) | refuted: deliberate; `scripts/otr_provision.py:1293-1300` documents the env var as the POSIX path and explicitly rejects the os.name branch.
11. hardcoded-paths-05, nodes/_otr_hf_env.py:49 -- `_DEFAULT_HF_HOME = r"C:\ComfyUI-Models\huggingface"` as tier-3 fallback | refuted: `prestartup_script.py:57-60` sets HF_HOME cross-platform before any node import, so the literal is unreachable in the shipped boot path.
12. models-fetch-ltx25-01, config/profiles/otr_w45_ltx25_foley_plus.json:81 -- five LTX 2.5 weights with zero fetch source in README, MODEL_INVENTORY or the fetcher's LANES dict | refuted: a pinned fetch source for all five exists elsewhere in the repo; the finding only checked three files.
13. hf-env-hardcoded-default-01, nodes/_otr_hf_env.py:49 -- same literal, no `.is_dir()` guard unlike `_models_root()` | refuted: both shipped call sites (`_otr_model_loader.py:842`, `:1709`) run after ComfyUI has already set HF_HOME.
14. eng-ltx25-unshipped-remedy-01, nodes/_otr_video_engines/eng_ltx25.py:743 -- offers "rerun the pinned provisioner with --packs-only", which never ships | refuted: the primary remedy in the same sentence is the patch file, and `patches/` DOES ship.
15. runpod-required-models-naming-drift-01, scripts/otr_fetch_lane_weights.py:336 -- two competing sources of truth for the RunPod model manifest | refuted: in-flight Codex file, and the fetcher itself is already correct.
16. workflow-json-01, config/profiles/otr_4060_floor.json:4 -- the single shipped example workflow is generated from a `status: draft` profile | refuted: `status` is validated but is not an application gate anywhere.
17. profiles-lanes-05, .comfyignore:79 -- all 113 profiles ship undifferentiated, 71 of them draft and 40 of those harness legs | refuted: profiles are imported and consumed at runtime, unlike `scripts/`, so the exclusion precedent does not transfer.
18. profiles-lanes-06, config/profiles/otr_nvidia_8gb_h3.json:3 -- the only two 8GB + non-AnimateDiff profiles are self-labelled unproven lab experiments | refuted: both are already `status: draft` and already say so in `display_name`; the proposed fix is the current state.
19. profiles-lanes-07, README.md:78 -- AnimateDiff is the only 8GB lane with a documented auto-fetch path | refuted: not a defect; README is accurate (retained as ground truth in section 7).
20. profiles-lanes-08, scripts/download_ltx_0_9_8.ps1:14 -- hardcodes the operator's absolute repo and venv paths | refuted: `scripts/` never ships, so `affects: registry` is wrong.
21. profiles-lanes-10, config/profiles/otr_runpod_starter.json:4 -- `status: draft` while `docs/RUNPOD_INSTALL.md` calls it the proven newcomer starter | refuted: status is not a gate; documentation and field disagree harmlessly.
22. fresh-install-docs-02, scripts/otr_pod_provision.sh:158 -- silently falls through to an NVIDIA-named profile on any Linux host without `nvidia-smi` | refuted: in-flight Codex file, and `scripts/` is outside the ship surface.
23. fresh-install-docs-07, README.md:289 -- "OS: Windows or Linux" omits macOS though `otr_mac_mps` ships | refuted: the sentence describes what is TESTED, and README:292 already flags Mac/AMD variants as unverified drafts.
24. fresh-install-docs-08, nodes/OTR_LedgerScriptWriter.py:2114 -- user-facing strings cite unshipped `docs/` | refuted as a separate item: these six are tooltips, not errors, and the root cause is already covered by dead-code-01 in section 3.
25. encoding-os-01, nodes/story_orchestrator.py:822 -- a raw RSS headline is logged with no UTF-8 console safety net anywhere in the package | refuted: logging handlers do not kill the process the way the historical raw `print` in prestartup did; wrong hazard class.
26. encoding-os-02, nodes/video_engine.py:1812 -- fixed `otr_video_audio.wav` in the shared temp dir, non-unique across concurrent processes | refuted: written at :1838 and consumed at :1971 within one call; the collision window is narrow.
27. performance-01, nodes/otr_image_gen_dispatcher.py:2156 -- the full image checkpoint reloads per still, 33-35 times an episode | refuted: single-resident discipline is deliberate (`flux_gen1.py:261-263`) and is what keeps 8 GB working.
28. performance-04, nodes/otr_post_upscale_procgen_blend.py:1013 -- three sequential full-episode x264 encodes where one filter graph would do | refuted: the bars pass was deliberately isolated (BUG-4, 2026-06-20) to protect a historically fragile `filter_complex`, and `captions_ass_path` is pinned by a 2026-07-04 widget migration.
29. performance-09, nodes/_otr_video_engines/render_driver.py:4238 -- weights torn down and reloaded per BEAT because BeatSession is beat-scoped | refuted: that bracket is a GPU lease plus patcher detach, not a disk reload; `motion_common.py` says so explicitly.
30. performance-12, nodes/_otr_video_engines/render_driver.py:2063 -- whole-ledger indexes rebuilt inside a whole-ledger walk, O(n^2) over beats and lines | refuted: measured cost is milliseconds; cleanup, not a render win (the finding says so itself).
31. test-health-01, tests/test_llm_timeout_queue_halt_smoke.py:150 -- two unconditional skips | refuted: both cite tracked docs and raise NotImplementedError if unskipped; the finding's own verdict is "no action needed".
32. test-health-03, tests/conftest.py:31 -- CUDA masked before any torch import, but no test carries `requires_cuda` yet | refuted: positive finding, zero ship impact.
33. test-health-06, tests/test_canonical_headless_api.py:419 -- three `os.name != "nt"` skips leave PowerShell harness contracts with no non-Windows equivalent | refuted: real Windows-only harness contracts, not masked defects; belongs in portability planning, not a ship list.
34. mac-amd-01, pyproject.toml:22 -- bitsandbytes has no macOS wheel at any released version | refuted on the wheel-availability claim specifically; **the missing environment marker is separately CONFIRMED as deps-drift-01 and is listed as a blocker.**
35. mac-amd-07, workflows/otr_canonical.json:524 -- the pack's entry-point graph is CUDA-hardcoded in five widgets (`:524` cuda, `:526` bnb_nf4, `:527` 14.5, `:1293` indextts2, `:2137` fp8_ok) | refuted: README:290-291 already points non-NVIDIA users at the variants, so "nothing points a new user at them" is false. (Note: the fix proposed -- naming the concrete variant FILE in the validator's failure text -- survives on its own merits.)
36. mac-amd-09, config/profiles/otr_mac_mps.json:23 -- Stable Audio 3 pins the CUDA-only `dpmpp_3m_sde_gpu` sampler and is the Mac music engine | refuted: facts accurate, but the Mac lane is unproven end to end anyway, so this is not the gating item.
37. mac-amd-10, nodes/_otr_image_engines/registry.py:157 -- no local image engine declares mps, so a Mac install cannot mint a still without a paid key, contradicting "No cloud services required" | refuted on the contradiction: the marketing line is not falsified for the platforms actually shipped. (The underlying fact is used in section 6.)
38. mac-amd-12, nodes/_otr_hf_env.py:49 -- unguarded Windows HF_HOME fallback on every platform | refuted: structurally unreachable in the shipped boot path (third report of the same line).
39. runtime-writes-01, nodes/_otr_user_banks.py:143 -- user-authored story banks live at `<repo>/user_packs/source_banks` with no env override, destroyed on every pack update | refuted on severity only; the fact is accepted.
40. runtime-writes-02, nodes/_otr_user_banks.py:576 -- activation snapshots content-hashed into `<repo>/user_packs/.snapshots` | refuted: same root cause as 39, and only written on explicit activation.
41. runtime-writes-04, nodes/_otr_hf_env.py:49 -- HF_HOME default has no platform guard unlike `_models_root()` | refuted: `os.environ['HF_HOME']` is checked first and is always set by prestartup.
42. runtime-writes-06, tools/make_registry_icon.py:171 -- hardcoded operator path and Windows fonts at module scope with no `__main__` guard | refuted as a runtime defect: nothing imports it. **The path leak itself is CONFIRMED as hardcoded-paths-06 in section 4.**
43. gap-cold-start-default-graph-02, nodes/_otr_model_loader.py:1372 -- the VRAM admission gate is priced against the saved 14.5 widget, never the real card, so an 8 GB box admits a ~12 GB model | refuted on mechanism: the OOM is caught at `:966-1010`, not a raw uncaught traceback as claimed.
44. gap-cold-start-default-graph-04, nodes/_otr_rolls.py:188 -- "roll (any eligible bank)" can land on either RSS-dependent bank with no offline fallback | refuted: the failure is loud and correctly typed, and network banks are a documented design choice (the finding's own fix is "document it").
45. gap-upscale-mps-rejected-as-malformed, nodes/_otr_upscale_engines/_resolve.py:63 -- `mps` reported as MALFORMED_CONFIG rather than not-yet-supported | refuted: documented deliberate deferral, and unreachable today. (Retained at LOW in section 4 as the schema/resolver mismatch, mac-amd-11.)
46. gap-image-capabilities-device-metadata-unenforced, nodes/_otr_image_engines/ideogram4_local.py:601 -- `host_caps` is accepted-but-unread in all five local image `assert_usable` methods | refuted: enforcement lives in the profile-derivation step, not the adapter; the claim that nothing catches a mis-declaration is architecturally wrong.
47. gap-viewer-orphaned-unreachable-05, viewer/index.html:214 -- unreachable page calling routes never registered anywhere | refuted on the wording: those routes belong to `scripts/serve_ledger.py`. **The shipping half is CONFIRMED as registry-flag-09 in section 4.**
48. gap-tools-audit-model-license-crashes-06, tools/audit_model_license.py:31 -- hardcodes `docs/model-license-audit-targets.txt` and raises an unhandled traceback on any registry install (executed and reproduced) | refuted on severity: maintainer-only tool, nothing invokes it.
49. gap-tools-engine-matrix-check-fails-07, README.md:517 -- README tells users to run `tools/engine_matrix.py --check`, which always fails on a fresh install because `docs/ENGINE_MATRIX.md` never ships (executed, exit 1) | refuted: that sentence describes a suite test, not a user instruction.
50. gap-tools-make-registry-icon-hardcoded-08, tools/make_registry_icon.py:171 -- no `__main__` guard, so import alone would load `C:\Windows\Fonts\arialbd.ttf` and write to the operator's path | refuted: nothing imports it (duplicate of 42).
51. gap-indextts2-queue-time-vs-render-time-09, nodes/_otr_audio_engines/eng_indextts2.py:160 -- `requires_flag = None` means the graph queues clean and fails mid-render | refuted as engine-specific: it is the uniform house pattern across every audio engine. **The consequence for a fresh install is nonetheless CONFIRMED as a blocker via profiles-lanes-03 / gap-cold-start-default-graph-01.**

---

## 9. GO_FORWARD_PLAN entry -- 2026-09-01

```
## 2026-09-01 -- Ship audit synthesis (71 confirmed, 51 disputed)
DECISIONS
1. Registry flag: the only shipped string matching a published secret rule is
   README.md:164 (hf_ + 38 chars). Every structural suspect (CORS, prestartup,
   ctypes/__import__, *.env.json) is byte-identical in the ACTIVE alpha.8 and is
   therefore ruled out. There is no Active version and no rollback target.
2. alpha.15 = ONE push, 5080-owned: README.md:164 -> hf_your_token_here;
   pyproject.toml version bump + pycairo/pillow/aiohttp added + bitsandbytes
   marked "; sys_platform != 'darwin'". pyproject is a release trigger: touch
   it once, batched.
3. If alpha.15 is Flagged, publish the alpha.8 tree (e44235f5) byte-identical as
   alpha.16 as the control. Flagged again = the scanner ruleset moved; that is
   the evidence for Comfy-Org, not more archaeology here.
4. Six blockers stand: bitsandbytes marker, no Active version, ltx_8gb profile
   pairs an 8GB engine with a 14.5GB writer (otr_g4_ltx_8gb.json:40 and
   otr_w45_ltx_8gb.json), indextts2 default needs reference WAVs that do not ship
   (voice_reference_bank.json:4 / otr_canonical.json:1376), _detect_host never
   emits "linux" (_otr_workflow_validator.py:293), unguarded ipc_collect
   (_otr_model_loader.py:740).
NEXT STEPS
5. 5080: alpha.15 push, then the four one-line platform guards (ipc_collect,
   n_gpu_layers mps, linux branch, needs_fp8_te/fp4 clauses). Each must print the
   5080's own before/after per 0B before pushing.
6. 5080: stop citing scripts/ and docs/ in shipped error text -- 8 LTX sites,
   3 TTS worker paths, spandrel, cloud image, 16 docs/ sites.
7. 4060: own the fresh-install proof. Run otr_nvidia_8gb_haunted end to end from a
   clean clone and publish to otr/obs; it is the only lowest-friction 8GB lane and
   the only one that avoids the indextts2 blocker.
8. Janitor fix first among optimizations: 9.3 GB reclaimed, 6.7 s off every boot,
   three lines (_otr_janitor.py:90 and :127).
9. Record the six blockers in docs/PROD_BUG_LOG.md by APPEND once each is proven
   on a live leg; promote to the Bug Bible only what has a live artifact.
```