<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The adapter registry and fallback chains are solid, but the candidate evaluation ignores severe license, aspect-ratio, and rasterizer-compilation blockers in the 3D space.

MUST-FIX BEFORE BUILD:
1. [Class B & D] Kill native mesh-gen (Hunyuan3D-2) and Splatting (TRELLIS/LGM) immediately. 
   * Defect: You cannot render meshes or splats to a video file locally without a rasterizer (`nvdiffrast` or `diff-gaussian-rasterization`). There are absolutely no Python 3.12 / sm_120 (Blackwell) Windows wheels for these. Your own `CAPABILITIES` dict correctly flags `hunyuan3d_talk` and `trellis_talk` as requiring `cu128_toolkit`. 
   * Fix: Move B and D to explicit NO-GOs.
2. [Class C] Kill SV3D and Zero123.
   * Defect: Output contract violation. These models output low-res squares (e.g., 576x576 or 256x256). Padding a 256x256 spinning object to your required 1472x832 landscape canvas results in an unusable postage-stamp video. Additionally, SV3D is Non-Commercial (violates commercial-clean constraint).
   * Fix: Move C to explicit NO-GOs.
3. [Class A] DepthAnythingV2 license violation.
   * Defect: DepthAnythingV2 is CC-BY-NC 4.0 (Non-Commercial). This violates your "commercial-clean strongly preferred" constraint and risks the ledger.
   * Fix: Downgrade to DepthAnythingV1 (Apache 2.0) or use Marigold (Apache 2.0) for the depth map generation.

PANEL DELIVERABLES & RANKING:

**RECOMMENDED FIRST TEST CASE: [Class F] LTX-Video + Camera-Control LoRA (or prompt-directed orbit)**
*   **Feasibility:** 100% wheel-only. Zero new dependencies. You already run it.
*   **VRAM envelope:** 12.5 GB (fits the 14.5 GB ceiling, verified in `CAPABILITIES`).
*   **Slot fit:** Perfect for `announcer_visual` and `music_visual`.
*   **Integration cost:** < 1 day. It uses the exact same `image_to_video` or `text_to_video` family already defined in `registry.py`. No new adapter needed, just a new registry row with the LoRA appended to `model_requirements`.
*   **Determinism & License:** Deterministic. LTX is OpenRAIL/MIT-ish (commercial clean).
*   **Failure-mode fit:** Drops cleanly into the existing LOUD fallback chain (e.g., `ltx_video_orbit` -> `humo_1.7B` -> `still_kenburns`).
*   **Output Contract:** Natively generates 1472x832 landscape (unlike Class C).

**RUNNER-UP: [Class A] 2.5D Depth Parallax (DepthAnythingV1 + Comfy core warp)**
*   **Feasibility:** Pure PyTorch/transformers wheels. No compilation.
*   **VRAM envelope:** < 4 GB (registers as `light` or `medium`).
*   **Slot fit:** Excellent for `scene_broll` or `music_visual`.
*   **Integration cost:** 1-2 days. It cleanly extends the `static_motion` family pattern seen in `cheap_families.py` (`StillKenBurnsFamily`), just replacing the ffmpeg pan with a Comfy depth-warp node chain.
*   **Determinism & License:** Deterministic. Apache 2.0 (if V1 is used).

**EXPLICIT NO-GOs:**
*   **Class B (Hunyuan3D-2):** Rasterizer requires compilation.
*   **Class C (Orbit video models):** Aspect ratio mismatch (square vs 1472x832) and NC licenses.
*   **Class D (Splatting/TRELLIS):** Rasterizer requires compilation.
*   **Class E (World Models):** Flash-attn dependency requires compilation on sm_120.

SHOULD-FIX:
1. [registry.py] If you build Class A (Depth Parallax), ensure it registers under the `static_motion` family (already supported by `VideoEngine` protocol) rather than inventing a new family, to keep the registry core clean.

CUT THESE:
1. [Class B] Searching for a "wheel-only texture/vertex color path" for Hunyuan3D-2. It is over-engineering a dead end. Even if you get colored vertices, you cannot project them to a 2D camera view natively in pure PyTorch without massive performance penalties or a compiled rasterizer. Cut the research time.

[ASSUMPTION]: You listed "torch 2.10.0+cu130". PyTorch 2.10 does not exist yet (latest is 2.6). I am assuming this is a typo for a future/custom build, which reinforces the fact that niche 3D compilation wheels (nvdiffrast, spconv) absolutely do not exist for your stack.