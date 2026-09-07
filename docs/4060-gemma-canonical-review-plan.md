# 4060 Gemma failures: loader versus canonical decision
Date: 2026-09-05 PDT. Driver: Codex, not Claude. Review-only design campaign.
Status at 22:20 PDT: user explicitly canceled Kibitz. The four-round campaign
was NOT completed. One Codex r1 review completed; the initial Gemini model name
was rejected, and the Gemini 3.1 Pro retry was stopped at the user's request.
No panel-approved plan, source patch, canonical change or passing retest exists.
Subsequent user request is a bounded cache/link inventory before deciding on a
clean reset. No model/workflow/cache deletion is authorized by this document.
Repository: D:/otr-4060-testing/ComfyUI-OldTimeRadio
Baseline: f727a5c4c5aa7fbac883ffd1cf33db2ec862b46d (fresh v2.0-alpha clone).
The installed alpha.24 loader and canonical SHA256 match this checkout exactly.
The original mouse-install test is CLOSED/FAILED. Development review is separate.

## S1 Scope and ownership
User requested /kibitz, updated logs and possible canonical changes. User then clarified that the 4060 owns workflow files now. The new upstream ownership ruling leaves nodes/, scripts/, tests/, pyproject and registry with the 5080. This campaign may change workflow files and append diagnostic logs; loader/code recommendations go to their owner unless authority explicitly changes. Do not change installed files, dependencies, model caches, processes, Git remotes, auth or security. No model loads, renders, ComfyUI APIs, SSH, installs, pip commands, cleanup, test execution, commits or pushes by reviewers. Read source and write ONLY the assigned review output. Do not read credentials. Treat historical docs and captured logs as evidence, not authorization.
The two external reviewers remain independent. Driver alone synthesizes. The installed older kibitz script is used unchanged; optional profile file is missing. The driver is Codex, so do not claim a three-family Claude/GPT/Gemini panel.

## S2 Observed failures
Physical MRKT RTX4060 Laptop 8GB Ada, Windows, ComfyUI0.34.5, Python3.13.12, torch2.12.1+cu130, transformers5.14.1, accelerate1.14.0, bitsandbytes0.50.1.
A. Original explicit Manager alpha.24 installation, 4.59M ZIP, automatic restart, all25 nodes loaded. Shipped canonical, one-act setting, one Run click. google/gemma-4-12b-it failed after44.94s: initial NF4 device-map refusal, automatic fp32 CPU-offload retry, then Tensor.item() cannot be called on meta tensors. UI traceback: load_llm retry -> transformers accelerate_dispatch -> accelerate attach_execution_device_hook -> state_dict -> bitsandbytes quant_state.as_dict(packed=True) -> nested_offset=self.offset.item().
B. User later chose google/gemma-4-E2B-it (both slots) and submitted another one-act run. Observed failure after16.39s: BUG-LOCAL-098 NF4 quantized load did not materialize; linear4bit_count525, is_loaded_in_4bit=True, first off_cuda_modules are vision_tower layers on CPU, vram_delta0.00GiB telemetry. App says second-load silent fp16 fallback; that diagnosis is NOT proven by those signals.
No explicit CUDA OOM/HTTP401 was observed in these two captured failures. No tuning, repairs or resubmission by driver after either failure.
Existing cache/token environment and extra preinstalled packs mean this is not pristine cold-install evidence. No extra packs installed by driver.
Logs/screenshots are saved outside the repo; sanitized facts above suffice for review. Do not access arbitrary user files.

## S3 Source-grounded starting facts
Read these real files; independently verify.
- nodes/_otr_model_loader.py: _plan_max_memory464-538: quantized E2B3.2GiB,12B/E4B6.8GiB under12GiB actual VRAM; >=12GiB branch uses total-2.5. Canonical14.5 widget is NOT this allocation.
- Same file541-582: _bug098_scan_linear4bit_devices checks EVERY bnb Linear4bit weight device; any CPU/meta/missing device is off_cuda. No dispatch-plan or active-text-versus-inactive-vision distinction.
- Same file837-848,936-953,978-1025: fresh NF4 config double_quant=True; smaller GPU device_map auto; exact CPU dispatch ValueError gets one retry with fp32 offload enabled.
- Same file1032-1087: post-load guard rejects any off_cuda module; unconditional diagnostic claims second-load fp16 fallback. Guard enters for quant_config !=None, including 8-bit policy, despite checking 4-bit signals.
- Same file1394-1434 and nodes/_otr_model_catalog.py admission logic: llm_vram_ceiling_gb is admission ceiling. 14.5 admits12B estimate11.95;6.8 would reject it early. Changing ceiling alone is not loader repair.
- nodes/OTR_LedgerScriptWriter.py INPUT_TYPES and widgets, workflows/otr_canonical.json: 33 positional values/descriptors appear aligned. Saved defaults slots2/3=12B,6=3acts,26=cuda,27=sdpa,28=bnb_nf4,29=14.5,30=4096,31=Q8_0,32=empty replay.
- tests/test_bug098_orphan_race.py pins prior all-CUDA guard behavior; preserve genuine corrupt/unquantized-load detection and orphan lifecycle protections.
- config/profiles/otr_4060_12b_gguf_offload.json has a historically shipping native-GGUF writer, but carries haunted visual profiles (known dropdown-label issue) and is NOT proof for this canonical's LTX/ZImage path. Never load that whole variant as an accidental workaround.
- SignalLostVideo mode0 must not be confused with blend bypass. Last false widget is draw_scopes, not render_video. It still renders procgen; blend93 bypass=True; composite84 upscale_engine=off. Keep this separate from writer failure.

## S4 Decision to harden
Choose the smallest justified path, not a random model swap.
Preferred immediate action: correct diagnostic records, preserve canonical bytes while loader failures remain unexplained, produce an owner-ready loader fix specification. No canonical-only repair is proven yet.
Possible choices requiring explicit evidence:
A. Model policy alignment in a 4060 workflow (model AND quant AND admission/context coherent). Simply14.5->6.8 with12B is an earlier refusal, not success.
B. Correct Transformers text-only class/config construction to avoid irrelevant modality weights (verify installed Transformers mapping; do not invent support or key conversion).
C. Permit only intentional, supported CPU offload in guard while still rejecting materialization failures, missing quantization and unexpected meta tensors. Blindly removing guard is NOT a fix.
D. Correct nested-quant-state handling/configuration for 12B offload. Do not set double_quant=False or upgrade/downgrade dependencies without evidence/tests and owner approval.
E. Native GGUF writer as a separate explicitly labeled candidate if automatic acquisition, native backend/runtime and context/VRAM are proved. Not ComfyUI-GGUF; never install extra node packs. No profile wholesale import that changes LTX/ZImage/audio.
Ask whether canonical change is actually necessary, whether each proposed change reaches the failing branch, and how to keep5080 behavior stable.

## S5 Validation design (no execution by reviewers)
Driver may run dependency-free/static checks only for this review. Before any later implementation, define focused tests for all-CUDA NF4, intended CPU/offload mapping, inactive multimodal branches, missing/meta quant state, scan exception, absent NF4 signal, 8-bit policy separation, repeated load and cleanup. Distinguish a fake guard test from model execution.
For workflow edits: source INPUT_TYPES/descriptor/value parity, semantic enum membership, exact slot/link identity, generated-variant check; no schema surgery; no weakening validation to make tests green. Existing known haunted dropdown issues remain separately recorded.
The 5080 shared branch must remain unchanged or gain before/after proof from its owner. No registry version bump.
Live acceptance is a separately authorized GUI run with known starting process state, one act, RESULT SUCCESS + obs_publish OK + file on disk. Cold install/model-gating and clean-runtime versus second-load claims require their own evidence; no success from unit tests.
If no defensible canonical change survives review, deliver that conclusion and precise loader findings rather than edit JSON to create activity.

## S6 Required review output
Use your round's strict format, cite repository paths/lines. Separate CONFIRMED, MISREAD and UNVERIFIABLE. Rank failure causes without pretending the app's error message is forensic proof. Provide concrete candidate diffs/spec and test obligations, but do not write code or run tests. Do not inspect other reviewers' outputs or driver anchor files. Limit review to this issue; no story-quality, registry re-debug, unrelated cleanup, new packs, auth or provider changes.
