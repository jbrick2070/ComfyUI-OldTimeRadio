# Novelty delta: what the 2026-09-01 ship audit found that prior QA had not

Scope: 71 classified findings from the 2026-09-01 ship audit, each checked against docs/PROD_BUG_LOG.md, the docs tree, kibitz/handoff artifacts, the Bug Bible coverage index, and git history.

## 1. Counts by classification

| Classification | Count | Share |
|---|---:|---:|
| NEW | 32 | 45% |
| ADJACENT_KNOWN | 25 | 35% |
| KNOWN_OPEN | 14 | 20% |
| KNOWN_FIXED_REGRESSED | 0 | 0% |
| **Total** | **71** | |

NEW severity spread: 2 blocker, 10 high, 10 medium, 10 low (3 of the low NEW items are positive findings, not defects).

**Two caveats on the counts.** First, the audit reported several defects twice under different ids, and the classifier split the duplicates: `gap-rocm-linux-never-detected-01` (NEW) and `mac-amd-02` (ADJACENT_KNOWN) are the same `_detect_host()` line; `gap-model-loader-unguarded-cuda-wash-02` / `gap-cold-start-default-graph-03` (both NEW) and `mac-amd-06` (ADJACENT_KNOWN) are the same unguarded CUDA wash; `mac-amd-05` (NEW) and `gap-gguf-ngpu-layers-mps-forced-zero-03` (ADJACENT_KNOWN) are the same `n_gpu_layers` gate. Deduplicated, the true NEW count is closer to 29. Second, the pyproject/requirements drift cluster (`deps-drift-01/02/03/08`, `node-load-01/02`, `mac-amd-08`, `gap-image-pillow-not-in-pyproject`) is one defect class spread across 8 ids and three classifications.

## 2. NEW findings

| id | file:line | Defect | Severity |
|---|---|---|---|
| gap-cold-start-default-graph-03 | nodes/_otr_model_loader.py:740 | load_llm() calls torch.cuda.ipc_collect() unguarded, so the canonical graph crashes on any CUDA-less torch build | blocker |
| gap-rocm-linux-never-detected-01 | nodes/_otr_workflow_validator.py:293 | _detect_host() can never emit "linux", so a stamped ROCm workflow always fails its own platform assertion on the host it targets | blocker |
| registry-flag-01 | README.md:164 | Ships an hf_-token-shaped placeholder literal while the registry gate is a secret scanner | high |
| dead-code-02 | nodes/_otr_video_engines/eng_ltx_8gb.py:733 | LTX-8GB / cloud-partner / upscale error text tells users to run scripts/ files excluded from the bundle | high |
| performance-03 | nodes/_otr_janitor.py:90 | Boot sweep stats 21,440 files at every ComfyUI boot (6.69 s measured) and grows without bound | high |
| performance-06 companion: gap-janitor-audio-slices-dir-granularity-04 | nodes/_otr_janitor.py:102 | Sweep runs at directory granularity, so audio_slices/ never loses its individual stale files | high |
| mac-amd-05 | nodes/_otr_gguf_backend.py:708 | n_gpu_layers forced to 0 on any non-cuda device, so the 12B writer runs CPU-only on Apple Silicon | high |
| runtime-writes-03 | nodes/_otr_shared/cloud_media_backend.py:202 | Cloud-media budget/billing ledger defaults to a path inside the pack directory | high |
| gap-image-upscale-error-cites-unshipped-scripts | nodes/_otr_upscale_engines/eng_spandrel_esrgan.py:249 | The only shipped upscale engine's missing-model error names a script .comfyignore strips from every install | high |
| gap-google-cloud-surface-01 | config/profiles/cpu_floor.json:14 | cpu_floor and otr_mac_mps route every image role to the paid Google API with no budget ceiling or opt-in | high |
| gap-google-cloud-surface-02 | nodes/_otr_image_engines/eng_google_image.py:289 | The Google BYO-key lane has no spend ceiling, cost estimate, or ledger entry at all | high |
| gap-model-loader-unguarded-cuda-wash-02 | nodes/_otr_model_loader.py:739 | The memory wash calls empty_cache()/ipc_collect() with no is_available() guard and no try/except | high |
| registry-flag-09 | viewer/index.html:214 | Shipped viewer fetches /ledger and /list, routes served only by the excluded scripts/serve_ledger.py | medium |
| hardcoded-paths-04 | nodes/_otr_audio_engines/eng_chatterbox.py:94 | "Not installed" error points at a Windows-only .ps1 installer that no longer ships | medium |
| fresh-install-docs-05 | docs/RUNPOD_INSTALL.md:87 | The doc's rationale for setting OTR_COMFYUI_MODELS_ROOT describes a failure mode the code already fixed | medium |
| fresh-install-docs-06 | README.md:103 | Only llama-cpp-python install command is CUDA-specific, in the section that names Mac/AMD/CPU as its audience | medium |
| performance-08 | nodes/OTR_LedgerScriptWriter.py:2024 | INPUT_TYPES does 104 os.stat calls and 12 JSON re-parses on every /object_info request | medium |
| performance-10 | nodes/_otr_video_engines/wrapper_bridge.py:593 | images_to_uint8 makes three full float32 copies of every decoded batch (~1 GB transient host RAM per beat) | medium |
| gap-image-cloud-error-cites-unshipped-scripts | nodes/_otr_image_engines/eng_cloud_image.py:278 | Cloud image partner-pin error points at an unshipped scripts/ file | medium |
| gap-image-lumina-no-folder-paths-autodiscovery | nodes/_otr_image_engines/lumina_image.py:405 | The only local image engine that will not find its own default weight dropped in the standard models folder | medium |
| gap-image-flux2klein-undocumented-gguf-dependency | nodes/_otr_image_engines/flux2_klein.py:293 | The ComfyUI-GGUF pack dependency is undocumented everywhere and unnamed on failure | medium |
| gap-google-cloud-surface-03 | config/profiles/cpu_floor.json:75 | preflight.required_keys is declared per profile and enforced nowhere in the runtime path | medium |
| registry-flag-06 | .comfyignore:10 | .comfyignore ships inside the scanned zip and narrates the pack's own scanner-trigger investigation | low |
| registry-flag-12 | workflows/external_examples/video_humo_native_unlimited_workflow.json:1021 | A vendored example workflow ships a stranger's absolute path with a 32-hex user id | low |
| hardcoded-paths-06 | tools/make_registry_icon.py:171 | Hardcodes the operator's absolute Windows username path as the output target | low |
| node-load-03 | nodes/_otr_kokoro_voice_prefetch.py:151 | Boot-time voice prefetch runs synchronously in prestartup with no network timeout | low |
| hygiene-02 | __init__.py:390 | Node-load failures are emitted twice, through print and the logger | low |
| hygiene-03 | nodes/_otr_model_catalog.py:1922 | Shipped modules use print() exclusively, with no logging import at all | low |
| performance-11 | nodes/_otr_ledger.py:883 | Shells out to git on every ledger stamp and never caches the failure | low |
| profiles-lanes-09 | config/profiles/widget_mapping.json:1 | Positive: widget-mapping and variant-generation machinery is clean | low |
| test-health-02 | tests/conftest.py:75 | Positive: 12,733 tests collect with zero import/collection errors | low |
| test-health-05 | tests/test_image_platform_c1.py:943 | Positive: no test writes into the repo tree or otr/episodes|obs outside tmp_path | low |

## 3. KNOWN_FIXED_REGRESSED

**None.** Zero of the 71 findings is a defect that a prior commit claimed to fix and that has since come back. Every previously recorded item the audit re-surfaced was still openly unfixed, not silently regressed. That is a meaningful negative result: the fix history in this repo is holding.

## 4. KNOWN_OPEN and ADJACENT_KNOWN, grouped

### KNOWN_OPEN (14) - already on record, still unfixed

- **Registry distribution:** registry-flag-02 (no Active version, no rollback target) - GO_FORWARD_PLAN.md:1043-1063 item J, raised three times by 2026-08-31.
- **Manifest drift:** deps-drift-01 (bitsandbytes, no platform markers) - 2026-07-09 01_AUDIT_AUDIO_LLM.md:690; node-load-01 (pycairo) and gap-image-pillow-not-in-pyproject (Pillow) - commit 017660de plus PBUG-20260829-02/-04's deferred-pyproject rule.
- **Bundle-relative references:** dead-code-01 (OpenRouter messages cite docs/) - the .comfyignore comment added by 9925513f already states docs/ appears only in error strings.
- **Profiles and lanes:** profiles-lanes-02 (no shipping 8 GB profile with a matching 8 GB writer) - GO_FORWARD_PLAN.md "THE 8 GB PROFILE FAMILY CANNOT RUN ITS OWN WRITER"; profiles-lanes-03 (indextts2 ref WAVs fail at render) - RUNPOD_INSTALL.md:375-380.
- **Performance and disk:** performance-05 (full-frame PIL alloc per bars frame) - 2026-07-08 scope-accel pass08_final_plan.md; performance-06 (audio_slices never swept) - _otr_janitor.py:83 docstring plus HANDOFF_LOG.md:10148; performance-07 (no eviction on _shared/mesh_cache, 27 GB) - 2026-07-08-source-banks-v2-plan.md:195,511 and eng_mesh_stage.py:531-533.
- **Mac / non-CUDA:** mac-amd-04 (llama-cpp-python undeclared, CUDA-only guidance) - 01_AUDIT_DEPS_CLOUD.md:315-321; mac-amd-11 (upscale rejects mps) - _resolve.py:9-10 docstring.
- **Engine contract:** gap-image-flux-default-noop-assert-usable - flux_gen1.py:14-19 and :233-238 state the no-op behavior as deliberate.
- **Test health:** test-health-04 (guarded lab-golden fixtures) - the test file's own docstring, lines 15-19.

### ADJACENT_KNOWN (25) - the class is recorded, this instance is not

- **pyproject/requirements drift, new packages:** deps-drift-02 (pycairo), deps-drift-03 (pillow, aiohttp), deps-drift-08 (num2words), node-load-02, mac-amd-08 - PBUG-20260829-02/-04 "THE COLD-INSTALL PAIR" and commit 017660de.
- **scripts/ exclusion fallout:** registry-flag-08 (three TTS Path-B workers), gap-content-data-integrity-01 (guide_ref points at otr_check and docs/EXTENDING_OTR.md) - 2026-08-30 USABLE-BY-OTHERS brief item 2, and commit a02d186f, whose verification checked imports only.
- **Internal docs shipping to installers:** fresh-install-docs-03 (_START_HERE.md), fresh-install-docs-04 (SKILL.md) - the 9925513f "stop shipping internal files" sweep that missed these two root files.
- **README and template discoverability:** fresh-install-docs-01 - 2026-08-23-workflow-discoverability-PROBLEM.md; docs-gguf-mac-rocm-01 - 01_AUDIT_DEPS_CLOUD.md:234.
- **Platform portability:** hardcoded-paths-01 (Mac font resolver), hardcoded-paths-07 (Windows-only ffmpeg candidates), mac-amd-02 (_detect_host no linux), mac-amd-03 (needs_fp8/fp4 never consulted), mac-amd-06 (unguarded ipc_collect), gap-gguf-ngpu-layers-mps-forced-zero-03 - the 2026-07-09 platform-portability set (01_AUDIT_DEVICE_GREP.md, SWITCH_SPEC.md, platform-portability-final.md).
- **Runtime writes and cloud caches:** runtime-writes-05 (OpenRouter cache under the repo) - CLAUDE.md 6A models-root ruling; gap-google-cloud-surface-04 (google_api_model_cache.json read but never written) - the 2026-07-08 Google BYO lane plan, which specified the writer that was never built.
- **Cold start:** gap-cold-start-default-graph-01 (indextts2 ref WAVs absent from README) - MODEL_ASSET_INDEX.md:129 and RUNPOD_INSTALL.md:117 record the constraint, README does not.
- **Singletons:** encoding-os-03 (UnicodeDecodeError uncaught) - 2026-08-28 hf-token RESEARCH_REPORT.md:279; hygiene-01 (_otr_content_safety broken to_dict) - 2026-08-22-dead-symbol-inventory.md:43; performance-02 (ltx_video/ltx_av encoder reload) - Bible 12.117, which fixed only the ltx25 sibling; gap-content-data-integrity-02 (_corpus/monkeys_paw duplicate) - HANDOFF_LOG.md:5307-5314; gap-image-lumina-dead-flag-in-error - PBUG-20260817-02.

## 5. Verdict

The audit did find real nuance prior QA had missed, but the novelty is concentrated in three places rather than spread evenly: the two blockers, `gap-rocm-linux-never-detected-01` (`_detect_host()` maps only Darwin and Windows, so every stamped ROCm profile fails its own platform assertion on the exact Linux box it targets) and `gap-cold-start-default-graph-03` / `gap-model-loader-unguarded-cuda-wash-02` (an unguarded `torch.cuda.ipc_collect()` in load_llm's memory wash that kills the canonical graph on any CUDA-less torch), are both single-line defects that the dedicated 2026-07-09 device-guard audit walked past while flagging neighbors in the same functions. The most valuable pattern-level discovery is the cluster of five dangling-reference findings (`dead-code-02`, `gap-image-upscale-error-cites-unshipped-scripts`, `gap-image-cloud-error-cites-unshipped-scripts`, `hardcoded-paths-04`, `registry-flag-09`): the 2026-08-28 commit that stopped shipping `scripts/` verified Python imports only, so every user-facing error string and the shipped viewer's fetch targets still point into a tree that no registry install contains, and `gap-google-cloud-surface-01/02` adds a genuinely unexamined risk that two profiles route all image work to a paid API with no ceiling. Honest limits on that claim: 45% NEW overstates the yield, since deduplication drops it to roughly 29 and three of those are positive no-defect findings, and the largest ADJACENT_KNOWN cluster is simply the pyproject drift the project already understands and has a written rule for.