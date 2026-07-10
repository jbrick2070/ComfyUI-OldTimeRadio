# OTR Platform Portability -- Campaign Brief (2026-07-09)

**Window role: DEEP CODE REVIEW + ANALYSIS ONLY. Zero code writing. Zero edits to `workflows\otr_canonical.json`. Output = docs only.**
A separate coder window owns the code (serialize via `docs/GO_FORWARD_PLAN.md` -- do not edit that file from this window).

Doc convention (per `.gitignore:241`, `docs/2026-*/` folders are ignored on purpose): campaign WORKING artifacts (anchor review, audits, roundtable/kibitz passes) go in the local folder `docs\2026-07-09-platform-portability\`; this brief and the promoted FINAL plan (`docs\2026-07-09-platform-portability-final.md`) are flat tracked files that get committed and pushed.

## 1. Operator goal

Make the OTR pipeline runnable on many platform tiers, each as a **saved copy of the ONE canonical JSON with explicit switches set differently**. Target platforms:

| Tier | Notes |
|---|---|
| NVIDIA 50-series 16 GB | Current baseline (RTX 5080 laptop, sm_120, torch 2.10 cu128) |
| NVIDIA 40-series | 12-16 GB class |
| NVIDIA 30-series | 8-12 GB class |
| AMD 16 GB (ROCm) | Linux |
| AMD 8 GB (ROCm) | Linux |
| Apple Silicon Mac (MPS) | Unified memory; no CUDA, no sage/flash-attn, no fp8 |
| CPU-only | "Really cheap" tier -- honest scope, even if slow / reduced lanes |
| Comfy Cloud | Feasibility question -- see Open Questions |
| Linux + NVIDIA | Should fall out of the CUDA tiers; verify no Windows-only assumptions |

## 2. Hard requirements (operator directives)

1. **ONE canonical JSON** (`workflows\otr_canonical.json`) stays the single source of truth. A platform variant = that JSON saved with different switch values. No forked graphs.
2. **Explicit switches, NO auto-detection.** No runtime probing that silently changes behavior. Validation-not-detection is allowed and wanted: a node may FAIL LOUD if the host cannot satisfy the switch (e.g. `device=cuda` on a Mac), but it must never fall back silently. This continues the VRAM-rip philosophy (tier/OOM auto-gates ripped 2026-07-03 @ 4fa85282; do not reintroduce them).
3. **1080p is the output ceiling for now.** Upscalers are acknowledged future work: the switch spec may RESERVE one `upscale_stage` switch (default `off`), but no upscaler design/eval in this campaign. Note `nodes\rtx_upscale.py` exists and is RTX-coupled -- audit it as a portability liability, do not extend it.
4. **Cheap CPU-only tier must exist.** Campaign decides its honest scope (plausibly: story/LLM via llama.cpp, TTS lanes, stills, parallax/visualizer motion; probably no diffusion video). State what it CANNOT do, plainly.
5. Switches live as **widgets in the canonical JSON**. `widgets_values` is POSITIONAL -- new switches APPEND at the END only (BUG-LOCAL-097). The switch spec must give exact widget name, node, position, allowed values, default.
6. **NO FALLBACKS** ethos holds (rip-sfx-broll 6bad6e5b). Every variant either runs its configured lane or fails loud.

## 3. Grounding anchors (verified in repo at HEAD 5d28749a)

- Canonical workflow: `workflows\otr_canonical.json` (~37 KB). Branch `v2.0-alpha`.
- **Video engines** (`nodes\_otr_video_engines\`): character_3d, cloud_video, google_omni_video, google_veo_video, google_vid_sfx, humo, ltx_av, ltx_video, mesh_stage, still_parallax, triposr, visualizer, viz_camera, viz_mandala, viz_rainbow, wan_i2v, wan_ti2v.
- **Audio engines** (`nodes\_otr_audio_engines\`): bark, chatterbox, cloud_elevenlabs, cloud_sonilo, dia, google_lyria, google_tts, indextts2, kokoro, musicgen, stable_audio, stable_audio_3.
- **27 CUDA-touching files in `nodes\` alone** (grep `torch.cuda|sageattention|nvidia-smi|.cuda(|cuda:0`), notably: `_otr_model_loader.py`, `_otr_gguf_backend.py`, `_otr_loader_backends.py`, `_otr_vram_levers.py`, `_otr_memory.py`, `_otr_shared\gpu_residency.py`, `_otr_sys_specs.py`, `_otr_determinism.py`, `vram_guardian.py`, `rtx_upscale.py`, `_otr_video_engines\motion_common.py`. Also audit `scripts\` (`_consult_nvidia.py`, `kill_otr_zombies.ps1`, `setup_cloud.sh`) and `prestartup_script.py`.
- **Prior decisions that bind:**
  - VRAM rip complete @ 4fa85282 -- tier system + OOM ceiling gone; "tier JSONs are the only protection." This campaign IS the parked "tier clones" phase, widened from VRAM tiers to a platform matrix.
  - LTX-AV GGUF bakeoff (2026-06-26): distilled-1.1 Q3_K_M = dev-match winner on 16 GB. Quant ladder is a proven per-tier lever.
  - LTX-AV VideoVAE enc/dec split @ ae8ec55e -- VRAM headroom lever, another switch candidate.
  - Dep conflicts: indextts2/chatterbox pin old deps vs the Blackwell venv; SA3-native is conflict-free. Per-platform dep feasibility is part of the matrix, not just VRAM.
  - Cloud lanes exist and are platform-independent: Google engines live; ElevenLabs+Sonilo BUILD_PLAN @ d2fc8d77 (not coded). Cloud lanes may be the honest answer for weak tiers -- say so where true.

## 4. Campaign shape (fixed)

**Step 0 -- Claude anchor review FIRST.** Read the real Windows files (Desktop Commander / file tools -- NEVER the Linux sandbox mount, it lags and shows phantom corruption). Write `00_ANCHOR_REVIEW.md`: your own portability assessment + proposed switch architecture, before any panel sees anything.

**Step 0b -- Sonnet fan-out (optional, up to 5 subagents, model=sonnet).** Mechanical audit only -- per CLAUDE.md section 9 this is NOT Fable work. Suggested split: (1) device/dtype/attention grep audit across nodes+scripts, (2) per-video-engine hardware coupling table, (3) per-audio-engine + LLM/GGUF lane coupling table, (4) canonical JSON widget inventory -- which existing widgets are hardware-relevant, (5) deps/venv + Comfy Cloud feasibility research. Each subagent MUST be told: Windows paths + Desktop Commander, read-only.

**R1 -- `/roundtable` (cloud frontier).** High-level arc: switch architecture, platform matrix, variant strategy. Panel: GPT + Gemini + DeepSeek/Grok class (`~latest` aliases) **plus `tencent/hy3:free` via `--models` (until 2026-07-21)**. Reasoning effort none, 10-12k max tokens. Run LIVE, no dry-run; report actual spend after.

**R2 -- `/kibitz` coding plan.** (Local Codex + Antigravity only -- no claude CLI panelist.) Plan the FUTURE implementation: how `_otr_model_loader` / `_otr_gguf_backend` / engines take a device+dtype+quant policy from switches.

**R3 -- `/kibitz` wiring.** Exact widget spec against the real `otr_canonical.json`: names, nodes, append-only positions, per-variant value table, validator implications.

**R4 -- `/kibitz` convergence.** Stop when no new must-fix. You are the judge at every round: anchor first, ground every panel claim against real files, discard misreads, synthesize. Watch for Codex trying to co-commit during kibitz runs -- if it commits, reset/discard its commit.

## 5. Deliverables

Working (local folder `docs\2026-07-09-platform-portability\`, gitignored by design):

- `00_ANCHOR_REVIEW.md` -- Claude's grounded pick.
- `01_AUDIT_*.md` -- merged subagent findings (coupling tables).
- `PLATFORM_ENGINE_MATRIX.md` -- engine x platform: runs / runs-degraded (quant/res/steps) / cloud-lane-only / impossible, with the file-level reason.
- `SWITCH_SPEC.md` -- every switch: widget name, node, position (append-only), allowed values, default, fail-loud behavior.
- `VARIANT_MANIFEST.md` -- the saved-JSON set (seed: `otr_nv50_16gb` = canonical defaults, `otr_nv40_12gb`, `otr_nv30_8gb`, `otr_amd16_rocm`, `otr_amd8_rocm`, `otr_mac_mps`, `otr_cpu_only`, `otr_comfy_cloud`) with full switch-value table per variant.
- `RISKS_OPEN_QUESTIONS.md` -- anything needing an operator decision.
- Roundtable/kibitz artifacts under `roundtable\` per skill convention. UTF-8 no BOM, ASCII where practical.

Promoted (flat, tracked, committed + pushed):

- `docs\2026-07-09-platform-portability-final.md` -- the converged plan: go/no-go summary on page one, then matrix + switch spec + variant manifest + sprint breakdown for a later coder window.

## 6. Open questions the campaign must answer (seed)

1. **Comfy Cloud:** OTR is a custom node pack -- can Comfy Cloud run arbitrary custom nodes at all? Research docs.comfy.org. If not, define what `otr_comfy_cloud` honestly means (partner-API lanes? exportable subset? NO-GO?). Do not hand-wave this.
2. **Attention backend switch:** sage/flash vs sdpa/math per platform -- where does each import live, and is it switchable without code forks?
3. **ROCm reality:** which engines have hard CUDA APIs (streams, fp8, custom kernels) vs plain `to(device)`? AMD 8 GB may be quant-ladder + stills-heavy.
4. **MPS gaps:** ops missing on MPS for each engine family; unified memory as an advantage (big models, slow).
5. **CPU tier scope:** what is honestly viable (llama.cpp story lane, kokoro/bark TTS?, musicgen-small?, stills?) and what is not (diffusion video).
6. **Determinism/seeds across backends** (`_otr_determinism.py` is CUDA-touching).
7. **Windows-vs-Linux assumptions** outside torch: paths, PowerShell-only scripts, prestartup UTF-8 handling.
8. Does the 1080p ceiling need a per-tier `res_cap`/`frame_budget` switch pair, or one shared cap with per-tier frame budgets?

## 7. Guardrails (inherit, non-negotiable)

- Repo CLAUDE.md operator directives apply in full (filesystem split, PowerShell quoting rule, 60s MCP ceiling, git policy).
- READ-ONLY on code and on the canonical JSON. Docs-only writes.
- Commit AND push each green tracked-doc chunk to `v2.0-alpha` same session -- **stage by explicit path only** (another window may be mid-work). Verify HEAD==origin, no 0-byte files, no BOM. (The working folder is gitignored -- only the flat promoted docs land in git.)
- Spend autonomy < $20: run panels live, state actual spend after.
- Never "dummy" -- "placeholder"/"stub". SFW. No profanity.

## 8. Definition of done

R4 converges with no new must-fix; every surviving claim is file-grounded; the promoted final doc is complete enough that a coder window can implement without re-deriving; operator gets a one-page go/no-go summary at the top, including honest NO-GO calls (especially Comfy Cloud and CPU video).

---

## Appendix: paste-in kickoff prompt for the fresh window

Platform-portability review campaign. ANALYSIS ONLY -- no code writing, no edits to workflows\otr_canonical.json.

Read, in order: (1) CLAUDE.md at repo root -- operator directives win; (2) docs\2026-07-09-platform-portability-brief.md -- your full mission brief.

Then execute the campaign exactly as the brief specifies: Step 0 your own grounded anchor review first (real Windows files via Desktop Commander, never the Linux mount); Step 0b up to 5 Sonnet subagents for the mechanical coupling audit (Windows paths, read-only); R1 /roundtable cloud frontier panel + tencent/hy3:free via --models, live, report spend; R2-R4 /kibitz (Codex + Antigravity), you are anchor + judge every round. Working artifacts in docs\2026-07-09-platform-portability\ (gitignored by design); promote the converged plan to docs\2026-07-09-platform-portability-final.md and commit+push it to v2.0-alpha, staging by explicit path only.

Goal: one canonical JSON + explicit switches (no auto-detect, fail-loud), saved as per-platform variant JSONs -- NVIDIA 50/40/30, AMD 16/8 GB ROCm, Mac MPS, CPU-only cheap tier, Comfy Cloud (feasibility!), 1080p ceiling, upscalers deferred to a reserved off-by-default switch.

Start now with the task list, then Step 0.
