# OTR resource-optim + portability -- r1-HARDENED (Codex + AntiGravity CONVERGED; Claude verified)

Honest answer to "are we leaving headroom on the 5080, and can a 32 GB-RAM box run it?":
**No on both, today** -- and the "HuMo lost quality" has a concrete config root cause. All findings
below were grounded against the real files.

## A. HuMo: stable + HQ + reliable (the operator's actual goal) -- ROOT CAUSE FOUND
- The shipping profile `config/profiles/16gb_full.json` pins `"video_render_engine": "humo_1.7B"` and
  `other_beats_visual: "humo_1.7B"`. The 14B keystone is NOT the production default. THAT is the
  perceived quality drop -- not a runtime downgrade (render_shot has no fallbacks; it raises loud).
- WHY 1.7B: the 14B does not fit 14.5 GB alongside the umt5 text-encoder (~5.2 GB resident, the CS-4
  finding) -- so 1.7B was chosen FOR reliability. This is the stable-vs-HQ tradeoff, in a config file.
- PATH TO STABLE + HQ: free the VRAM so 14B fits, then promote it:
  1. lazy umt5-TE detach (CS-4-open already names this) / a smaller or quantized TE / a smaller HuMo
     14B quant -- whichever lands the 14B render-window peak under a real envelope (target <= 13.5 GB
     so the OS keeps headroom).
  2. measure it via the JOB-3 bakeoff Phase B (14B under the AV stack). If 14B fits with margin,
     flip `16gb_full.json` role+slot overrides 1.7B -> 14B (SAME-change profile edit; re-validate).
  3. until it fits, 1.7B STAYS the reliable default (de-blue cfg already applied); 14B = opt-in.

## B. 5080 headroom -- nominal, not real
- Define and enforce VRAM ENVELOPES on the measured render-window peak over N runs:
  green <= 13.0 GB, yellow <= 14.0 GB, red > 14.0 GB. The OS/desktop/browser eat 1.0-2.5 GB VRAM, so
  the 14.5 GB ceiling leaves ~0 real headroom -> random CUDA OOM.
- Heavy engines ride the edge (grounded): ltx_video Q3_K_M ~14.8 GB (registry.py:162, RED);
  wan_i2v ~14.5 GB (registry.py:195 / eng_wan_i2v.py:289, RED); ltx_av ~13.7 GB (eng_ltx_av.py:10-11,
  yellow). LEVER: drop one quant step where quality holds (the 2026-06-26 LTX ladder: Q3_K_M -> a
  tuned Q2/Q3 mix) OR lower the default ceiling. HuMo 1.7B is the cheap one; 14B is the peer to test.
- ONE ceiling source of truth: reconcile workflow node-62 `vram_ceiling_gb=14` vs runtime 14500 MB
  (motion_common.py:40 / wrapper_bridge.py:37) -- they disagree; pick one and derive the other.
- Per-subsystem candidate matrix (video / audio / image / LLM): current default, alternatives, peak
  VRAM/RAM, CPU-viable?, quality gate, license, wired-in-canonical. One table per subsystem.

## C. Portability (32 GB system RAM / modest-or-no GPU) -- 5080-only today
- `8gb_lite` + `cpu_floor` are `"status": "draft"` and BOTH degrade video to `still_kenburns`
  (Ken Burns stills, NO motion). So even when promoted, a modest box gets a STILLS radio drama
  (station card / visualizer / still + burned captions), not HuMo/LTX motion -- set that expectation.
- HARD BLOCKER for CPU: the writer LLM `OTR_LedgerScriptWriter` is `exempt_node_types` in
  widget_mapping.json:72 -> profiles CANNOT swap its `creative_writing_model`/`technical_model`, and
  the loader defaults to CUDA + 4-bit NF4 bitsandbytes (_otr_model_loader.py:105/114/193). A CPU-only
  host would choke on the LLM before rendering. FIX: un-exempt the two writer slots + override them to
  an Ollama-GGUF (gemma local) or remote backend in `cpu_floor.json` / `8gb_lite.json`.
- Host-detect is a HARD ABORT: the validator raises ValueError when profile != host
  (_otr_workflow_validator.py:289-315) instead of auto-applying the suggested tier. FIX: auto-degrade
  to the suggested profile (or launch-script applies it) rather than failing.
- PROMOTION GATES per tier (draft -> shipping): a stamped run renders end-to-end; asset exists; peak
  <= the tier's vram_budget; documented user-visible degradation. + wall-clock budget for CPU (bark /
  musicgen on CPU are very slow).

## CUT (both agents) / scope
CUT: remote/OpenRouter + Comfy-Credits LLM optimization (0 local VRAM, opt-in); character_3d
(triposg/hunyuan3d, dark + cu128-gated); audio MODEL replacement (keep an installability/residency
audit only -- indextts2 uses a pinned worker venv, fails closed if missing; don't reopen the spine).

## Open for r2/r3/r4 (if hardened to a build spec)
The quant-ladder pick per heavy engine + the chosen envelope numbers; the umt5-TE-detach mechanism +
14B promotion procedure; the widget_mapping un-exempt + lower-tier LLM override wiring; the
auto-degrade validator change; the per-tier promotion-gate test harness.
