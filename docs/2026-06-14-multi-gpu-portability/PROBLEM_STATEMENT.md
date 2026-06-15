<!-- ============================================================================
  HEADER NOTE (added by the research window, 2026-06-14)
  This file is the OPERATOR'S AUTHORED SEED for the multi-GPU hardware-tier
  portability dispatch. Everything below the horizontal rule is Jeffrey's
  problem statement, preserved VERBATIM (no edits to wording, structure, or
  scope). It is the source of truth for this dispatch window.

  Companion files in this same directory:
    - RESEARCH_FINDINGS.md  -- the R1-R7 results + Q1-Q5 answers against real code
    - TIER_MATRIX.md        -- the populated capability matrix from section 3

  Status: research + hardening only. Implementation is GATED (see section 10).
============================================================================ -->

---

# Problem Statement — Multi-GPU Portability / Hardware-Tier Targets

- **Project:** ComfyUI-OldTimeRadio (OTR) — local/offline generative pipeline, v2.0-alpha
- **Authored:** 2026-06-14 (seed for a dedicated dispatch window)
- **Status:** DRAFT problem statement. Research + hardening only. Implementation is GATED (see §10).
- **Suggested repo path:** `docs/2026-06-14-multi-gpu-portability/PROBLEM_STATEMENT.md`

-----

## 0. How to use this file (dispatch instructions)

This file seeds a fresh window dedicated to the portability problem. The receiving window should:

1. Read `docs/GO_FORWARD_PLAN.md` IN FULL, skim the `otr-build-tracker` dashboard, run `git log --oneline -12` + `git status` on `v2.0-alpha`, then read this file.
1. Do the **research-against-real-code** pass in §7. Report findings.
1. Resolve the **open questions** in §6 against actual code, not assumptions.
1. **STOP and discuss** before writing any implementation. No code until the operator confirms.
1. Treat every "current stack" claim below as *to-be-verified* — the appendix is from memory, not from the repo.

-----

## 1. Context

OTR currently runs against a **single hardware target**: Lenovo Legion Pro 7i Gen 10, RTX 5080 (Blackwell, compute `sm_120`, 16 GB VRAM), torch built for CUDA 13.0 (`cu130`). The video layer is a model-agnostic engine (shell nodes) with **3–4 video models hardcoded** — by deliberate scoping decision, there is no input-translation/adapter layer yet (that is a v2.5 item).

The pipeline is effectively locked to Blackwell + CUDA + a ~16 GB VRAM budget. Before/around public release, it needs to run on a wider hardware spread without that lock.

## 2. Problem

> The pipeline assumes Blackwell-class CUDA hardware (`sm_120`, `cu130`, 16 GB) throughout. We need documented, working execution tiers for older NVIDIA cards and for non-CUDA hardware, without breaking the shipping Blackwell path and without violating the existing invariants.

Specifically, produce variants (or a single tier-parameterized build) that run on:

- **RTX 40-series** (Ada Lovelace, `sm_89`)
- **RTX 30-series** (Ampere, `sm_86`)
- **Hardware-agnostic non-CUDA** — AMD (ROCm) and/or Apple Silicon (Metal/MPS), with a CPU fallback where unavoidable. This tier must not require CUDA or Blackwell at all.

## 3. Goals — target tier matrix

|Tier     |Arch         |Compute    |torch build (verify)|VRAM span (typical)|Expected posture                        |
|---------|-------------|-----------|--------------------|-------------------|----------------------------------------|
|T0 (ship)|Blackwell    |sm_120     |cu130               |16 GB              |Reference; must stay green              |
|T1       |Ada (40xx)   |sm_89      |cu12x               |8–24 GB            |Target: runs with documented changes    |
|T2       |Ampere (30xx)|sm_86      |cu11x/cu12x         |8–24 GB            |Target: runs with documented changes    |
|T3a      |AMD          |ROCm       |rocm wheels         |varies             |Target: runs, possibly reduced model set|
|T3b      |Apple Silicon|MPS (Metal)|mps-enabled         |unified mem        |Target: runs, possibly reduced model set|

Per-tier VRAM/throughput caps (resolution, frame count, batch, offload depth) are an output of the research, not fixed here.

## 4. Constraints & invariants (must hold across all tiers unless explicitly waived LOUD)

- 100% local / offline. No network at render time.
- Determinism (seed-keyed) — **see §6, open question on cross-backend determinism.**
- Every in-render fallback is LOUD (no silent degradation).
- Dependency isolation (V-12 discipline).
- UTF-8 no BOM. SFW.
- Single resident heavy engine ≤ 14.5 GB on the Blackwell/16 GB tier — this number is **16 GB-specific** and must become a tier parameter, not a constant, for T1–T3.
- Do not touch the frozen audio spine (byte-identical master, mux-LAST, `test_audio_byte_identical` stays GREEN).
- prod/main stays GATED. Docs only; do not commit/push unprompted.

## 5. Known portability blockers (to confirm against code)

1. **torch/CUDA build pinning.** `cu130`/Blackwell wheels won't run on older CUDA, and non-CUDA tiers (ROCm/MPS) need an entirely different torch. Need a per-tier build/wheel matrix.
1. **Custom/compiled kernels** — flash-attention, sage-attention, triton, any `sm_120`-compiled extensions. These are the hardest blockers; ROCm/MPS frequently lack them. Each must have a pure-pytorch fallback (LOUD) or be tier-gated.
1. **dtype divergence** — fp8 is Blackwell-favored; bf16 is broad on Ampere+; MPS is fp16-centric with op gaps; ROCm dtype coverage varies. Need a single dtype-selection chokepoint per tier.
1. **VRAM / unified-memory tiering** — offload thresholds, block-swap counts, tiling, and resolution/frame defaults are tuned for 16 GB. T1–T3 span 8–24 GB plus Apple unified memory; these must be parameterized.
1. **Cross-backend determinism** — bit-identical output across CUDA/ROCm/MPS is generally **not achievable**. Likely need to redefine "deterministic" as *within-tier reproducible*. Flag LOUD as an accepted limitation (confirm with operator).
1. **Per-backend model coverage** — some of the 3–4 hardcoded video models may have no acceptable path on MPS/ROCm. A non-CUDA tier may ship a reduced model set (confirm acceptable — §6).
1. **Custom-node ecosystem assumptions** — many ComfyUI custom nodes in the dependency tree are CUDA-only. Audit which deps hard-require CUDA.

## 6. Open questions to resolve early (before any implementation)

- **Q1 — Determinism scope.** Is *within-tier* reproducibility sufficient, with cross-tier parity explicitly waived? (Recommended yes.)
- **Q2 — Reduced model set.** For T3a/T3b, is a smaller/quantized video-model set acceptable if the full set can't run, or must all tiers be model-complete?
- **Q3 — One build vs N builds.** Tier-parameterized single codebase (preferred for maintenance) vs separate per-tier branches/builds? Decide before refactoring the device path.
- **Q4 — Floor target.** What's the minimum VRAM the non-Blackwell tiers must support (8 GB? 12 GB?) — this sets the cap math.
- **Q5 — Where the chokepoint lives.** Is there a single device/dtype selection point today, or is `cuda`/dtype assumed throughout? (Drives effort estimate — §7.)

## 7. Research tasks against the real code (core deliverable for the dispatch window)

Run these on `v2.0-alpha` and report findings before proposing changes:

- **R1.** Inventory every hard dependency on Blackwell/CUDA/cu130: grep requirements/lockfiles and source for torch version pins, `cu130`, `sm_120`, "blackwell", and any compiled-extension imports (flash-attn, sage-attn, triton, custom CUDA).
- **R2.** Map GPU-touching stages and flag backend-sensitive ops (attention impls, VAE tiling, fp8/bf16/fp16 usage, anything calling `.cuda()` or hardcoding `device="cuda"`).
- **R3.** Catalog the 3–4 hardcoded video models and each one's backend/dtype requirements. Mark per tier: runs as-is / needs swap or quantized variant / blocked.
- **R4.** Find every place the 14.5 GB / 16 GB budget is baked in (offload thresholds, block-swap counts, resolution/frame/batch defaults). List them as candidates for tier parameters.
- **R5.** Locate the device-selection and dtype-selection code paths. Determine whether a single chokepoint exists or whether device/dtype is assumed throughout. This answers Q5 and sizes the refactor.
- **R6.** Determinism audit: where seeds are set, where backend nondeterminism enters. Propose the per-tier definition of "deterministic."
- **R7.** Dependency audit: list every dep that hard-requires CUDA (custom nodes included) and whether a ROCm/MPS/CPU equivalent exists.

## 8. Non-goals / out of scope

- **No input-translation / canonical-schema adapter layer.** Models stay hardcoded per the v2.5 deferral. This work is *device/backend* portability, not *model-input* abstraction.
- **No changes to the frozen audio spine.**
- **No other sprints.** Story-spine, story-pipeline, broader audio stack remain PARKED.
- **No cross-backend bit-identical determinism** (pending Q1 confirmation).
- **ComfyUI Cloud node verification** is a separate roadmap item; reference only if it informs a "rent-a-GPU" tier, otherwise out of scope here.

## 9. Definition of done (acceptance assertions)

- [ ] A capability/tier matrix exists and is backed by real-code findings (R1–R7 complete).
- [ ] For each tier (T1, T2, T3a, T3b): a clear verdict — runs as-is / runs with documented changes / blocked, with reasons.
- [ ] The single device/dtype chokepoint is identified (or its absence documented), with a prioritized change list and effort estimate.
- [ ] The 16 GB-specific budget constant is enumerated everywhere it appears and a parameterization plan exists.
- [ ] Q1–Q5 answered against code/operator, not assumed.
- [ ] All invariants restated per tier; any that can't hold are flagged LOUD with rationale.
- [ ] T0 (Blackwell) regression posture unchanged; `test_audio_byte_identical` still GREEN.

## 10. Position in roadmap / gating

Pre-release order (T0 ship gated behind these): (1) 3D polish, (2) ComfyUI Cloud node verification, (3) story-render prompt optimizations. **This portability work is GATED until those land.** Research and problem-hardening (§6–§7) may proceed in parallel now in this dispatch window; implementation does not start until the operator lifts the gate.

-----

## Appendix — Assumed current stack (VERIFY against repo; do not trust)

- Hardware target: RTX 5080, Blackwell `sm_120`, 16 GB VRAM (Lenovo Legion Pro 7i Gen 10).
- torch: `cu130`.
- Video layer: model-agnostic engine, shell-node architecture, 3–4 video models hardcoded.
- Branch: `v2.0-alpha`. prod/main GATED.
- Invariants per §4. Source of truth: `docs/GO_FORWARD_PLAN.md`.

<!-- ============================================================================
  FOOTER (added by the research window, 2026-06-14)
  End of operator's verbatim seed.
  Companion deliverables for this dispatch:
    - RESEARCH_FINDINGS.md  -- R1-R7 findings + Q1-Q5 answers, file:line evidence
    - TIER_MATRIX.md        -- populated capability matrix (section 3)
  All three files are doc-only; implementation remains GATED (section 10).
============================================================================ -->
