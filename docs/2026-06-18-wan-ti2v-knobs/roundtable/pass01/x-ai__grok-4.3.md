<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Spec proposes Lightning/4-step/LoRA recipes and cross-platform guarantees that have no implementation path in the provided engine code and leave multiple hard constraints unresolved.

MUST-FIX BEFORE BUILD:
1. [HARD CONSTRAINTS 2] GGUF portability on MPS/AMD is explicitly an open question with no decision; code defaults to GGUF via _loader_mode + UnetLoaderGGUF in _node_candidates when extension matches. Fix: resolve to a single default (safetensors + UNETLoader) for the floor and update _ckpt_path/_loader_names/_node_candidates to enforce it.
2. [Candidate recipes B/C + Questions 3] Lightning LoRA and 4/6-step distill variants have zero support in _build_graph (no LoRA loader node, no strength input, no cfg=1.0 path, no alternate scheduler wiring). Fix: either delete B/C from the spec or supply the exact additional node classes + W() wiring that would be added to _build_graph.
3. [Current recipe + _build_graph] Default length stated as 25 but code hardcodes _TI2V_MIN_FRAMES=33 and calls quantize_frames_4n1 with that floor on every render_clip path. Fix: change spec default to length=33 (or remove the parenthetical).
4. [Questions 3 + HARD CONSTRAINTS 3] Lightning LoRA license must be Apache-2.0/MIT and commercial_clean=True is already set in the class, yet no license guard exists in assert_usable or _installed. Fix: add explicit license check (or drop all Lightning candidates from the 8GB floor).
5. [Questions 4] uni_pc is the code default in _build_graph and env; spec questions portability of uni_pc/sa_solver/MoEKSampler on MPS/AMD with no resolution. Fix: restrict floor default to euler/lcm (core nodes only) and document the sampler whitelist.

SHOULD-FIX:
1. [Problem + Current recipe] VRAM numbers (8.2 GB engine / 13.1 GB peak) are CUDA-only from 5080 smoke test; no MPS/AMD equivalent exists. Add a cross-platform measurement requirement before promoting any recipe.
2. [assert_usable + _aux_loader_files] VAE guard only checks basename string; does not verify the file actually decodes 5B latents. Add a minimal decode probe when OTR_TEST_MODE is set.
3. [render_clip] free_after_use + keep set omits any LoRA patcher; if B/C are added later this will silently drop the LoRA weights.

OPTIONAL / NICE-TO-HAVE:
- Add OTR_WAN_TI2V_SAMPLER whitelist validation at assert_usable time.
- Expose shift/cfg/step bounds per candidate so env overrides cannot produce invalid 4-step combos.

CUT THESE (over-engineering):
1. Candidate C (6-step distill) -- duplicates the 4-step lever with no reliability difference shown; safe to drop when the goal is "most solid floor".
2. MoEKSampler mention -- absent from _node_candidates and not a core node; safe to remove from Questions 4.
3. All 720p references (even as dropped) -- already declared out of scope; delete to shrink the document.

[ASSUMPTION] Lightning nodes and LoRA loader classes are assumed to be core ComfyUI nodes available on MPS/AMD exactly like Wan22ImageToVideoLatent; verify against actual /object_info.