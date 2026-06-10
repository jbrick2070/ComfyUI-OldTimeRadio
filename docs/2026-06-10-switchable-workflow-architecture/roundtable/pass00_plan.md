# Problem Statement -- One Switchable OTR Workflow vs. Many (the model / hardware / toolchain matrix)
*Draft for roundtable + coder scoping -- 2026-06-10. Operator preference: ONE architecture with switches.*

## The problem
OTR must run across a matrix: **models** (HuMo 1.7B/14B, LTX, Wan, latentsync, Flux + image peers, the audio engines), **hardware tiers** (8GB vs 16GB NVIDIA, Apple Silicon / MPS, CPU), and **toolchains** (CUDA cu128 sidecars vs Mac vs CPU). Today this shows up as multiple workflow JSONs (e.g. `otr_scifi_16gb_full.json` + an implied 8GB / Mac variant) PLUS a separate headless/soak submission path. Two costs follow:
1. **Onboarding friction** -- a new user faces too many choices and scripts; time-to-first-value is too long and drop-off is high.
2. **Drift** -- variants fall out of sync. Concrete, recent example: captions + procgen credits + the LTX radio open were enabled in the headless path but NOT baked into the production JSON, so real renders silently lost them. Every extra workflow file is another thing that can drift.

## Two decisions -- usually conflated; separate them
- **Decision A -- Workflow architecture (the core question here):** ONE canonical workflow JSON that adapts via switches, vs. SEPARATE per-tier / per-arch JSONs. *Operator preference: one with switches.*
- **Decision B -- Setup / onboarding UX (the delivery layer):** how a user picks their config -- an interactive CLI wizard (rich / InquirerPy), a local Gradio/Streamlit page, or a smart auto-detector. This sits ON TOP of Decision A: it *sets* the switches; it is not the switch mechanism. (The earlier write-up covered Decision B well but skipped Decision A -- this statement makes both explicit so they can be decided separately.)

## Why "one architecture with switches" is the strong default (the operator instinct, grounded)
- **OTR is ALREADY switch-based.** `OTR_VideoDirector` / `OTR_ImageDirector` per-role engine dropdowns + `OTR_ENABLE_*` env flags + the registry/fallback resolver. "One graph, many configs" is the existing grain, not a rewrite.
- **Fewer source-of-truth files = less drift.** The recent captions/credits parity bug came from a second path diverging from the production JSON. Consolidating shrinks that failure surface -- this is the single biggest argument for one graph.
- **One validation + determinism target** instead of N.

## The hard parts (why this needs a real plan, not just a toggle)
1. **Combinatorial validity + test cost.** Every switch combination must be a valid graph that passes the OTR converter/validator contract (`widgets_values` == serialized slots) AND renders. Full cartesian testing explodes -- you need representative PROFILES, not every combo.
2. **Toolchain splits cannot all live in one env.** cu128 sidecars (LTX/Wan/latentsync/3D) vs Apple-Silicon/MPS vs CPU are environment-level, not just graph toggles. A single graph can express "engine off -> fail-closed/fallback," but it cannot make a cu128 node run on a Mac. So some switching is a PROFILE / env / sidecar concern that lives ABOVE the graph.
3. **VRAM single-residency.** Switches must never let two heavy engines co-resident; the enable-set has to respect the <=14.5 GB single-heavy budget (the 8GB tier must auto-exclude 14B, etc.).
4. **Determinism across switch states** (seed-keyed) must hold per profile.
5. **Switch mechanism choice.** Env flags vs ComfyUI node-bypass/mute vs Director widgets -- which is canonical? Manual node rewiring by users is OUT (that is the friction we are removing). Prefer: switches = env flags + Director widgets that the registry/resolver read; the graph STRUCTURE stays fixed.
6. **Legibility for "vibe coder" users.** One mega-graph can become an unreadable hairball. The switches must be FEW, named, and high-level (a hardware tier + a couple of role choices), never dozens of node toggles.

## Strawman shape (one canonical workflow + a thin profile layer)
- **ONE canonical workflow JSON; structure fixed.** Engine selection + enablement come from (a) a single HARDWARE-TIER / PROFILE selector + `OTR_ENABLE_*` flags the Director/registry read; (b) Director per-role dropdowns; (c) the fallback resolver, which handles capability / VRAM / missing-toolchain by failing closed to a supported engine -- LOUD.
- **A PROFILE layer (Decision B) on top:** auto-detect (NVIDIA+VRAM / MPS / CPU) -> propose a profile -> confirm-or-customize (CLI wizard first; Gradio later) -> write the env/flags -> launch. Same graph every time; only the profile changes.
- **Toolchain differences (cu128 sidecars, MPS) are handled by the sidecar/venv + fail-closed gates, NOT by forking the graph.** On a Mac, the cu128-dependent engines are simply unavailable in that profile and fall back; the graph is identical.
- **2D-vs-3D becomes just another capability switch** in the same graph (the `character_3d` 3D-plan platform must-fixes already point this way).

## Open questions for the panel / coder
1. Can 8GB / 16GB / Mac truly share ONE graph with only profile/flag differences, or is there any STRUCTURAL difference that forces a variant? If one exists, name the single node/edge that differs.
2. Canonical switch mechanism: env flags + Director widgets vs ComfyUI bypass -- and how do we keep `widgets_values` converter-valid across every profile?
3. How do we test the matrix affordably -- a fixed set of representative profiles (e.g. 16GB-full, 8GB-lite, Mac-MPS, CPU-floor), each with a render/validate gate, instead of full cartesian?
4. Where does the PROFILE live (a committed `profiles.json`? env? wizard output?) and how is it applied at load AND at headless submit so BOTH use the SAME source of truth -- i.e. structurally kill the drift bug?
5. Does the single-graph rule extend cleanly to `character_3d`, or does 3D justify the one allowed exception?

## Decision criteria
Drift-resistance (single source of truth) - time-to-first-value (onboarding) - maintainability - testability (matrix cost) - VRAM safety - determinism.

## Working recommendation
Adopt ONE canonical switchable workflow -- it matches the operator preference, the existing platform grain, AND it directly attacks the drift bug. Express switches as profile + `OTR_ENABLE_*` flags + Director widgets the resolver reads; keep toolchain splits in the sidecar/profile layer, not in separate graphs. Layer the onboarding UX on top (auto-detect -> confirm -> launch; CLI wizard first, Gradio later). Validate via a small fixed profile set, each gated. Net: the profile (not a hand-edited graph) is the single source of truth that production AND headless both consume.

## Reconciliation + grounded cross-platform audit (2026-06-10, panel input integrated)

**The "separate JSONs win" panel answer is right about the USER ARTIFACT, wrong if it means hand-maintaining separate ARCHITECTURES.** It even concedes the key move: *"you still maintain one master graph; you just Save-As with the dropdowns set before you ship."* So the two positions collapse into ONE design:
- **Author / maintain: ONE canonical switchable master graph** (operator preference; single source of truth).
- **Distribute: GENERATED per-tier snapshot JSONs** (`otr_16gb_pc.json`, `otr_24gb_mac.json`, `otr_8gb_lite.json`) -- produced by a BUILD step that loads the master, sets the switches for that profile, and exports. Zero-friction "click Run" files for users, but NEVER hand-edited -> no drift.
- This answers the panel's question (patch-JSON-on-the-fly vs ship-a-mac-JSON): **neither -- GENERATE the tier files from the master at ship time.** Runtime patching reintroduces a second source of truth; hand-maintained mac JSONs reintroduce drift. A generator script is the only option that keeps one source AND ships pre-dialed files.

**Model-asset acquisition (Decision B, expanded).** The installer must be environment-aware, not dump 40 GB blindly:
- Parse `extra_model_paths.yaml` + respect `HF_HOME` BEFORE downloading. OTR ALREADY uses both (`HF_HOME=C:\ComfyUI-Models\huggingface`, `extra_model_paths.yaml -> C:\ComfyUI-Models`), so power users' shared drives are honored for free.
- Pull from a per-profile model MANIFEST via `huggingface_hub` (resumable, checksum, skip-existing).
- Confirm-before-dump: "~16 GB to <detected path> -- Enter to continue, or type a custom path."

**Grounded cross-platform audit (panel claims checked against the REAL code -- several already handled):**
- **ffmpeg encoder: mostly ALREADY Mac-safe.** The composition path (`otr_silent_composite`, `otr_caption_burn`, `otr_post_upscale_procgen_blend`, `rtx_upscale`, `wrapper_bridge`) all encode with **`libx264`** (universal CPU); the master mux (`otr_master_audio_mux`) uses **`-c:v copy`**. The ONLY `h264_nvenc` is in `video_engine.py` (procgen) and it sits behind `_check_nvenc()` -> auto-falls-back to libx264. So the panel's "crashes at muxing on Mac" worry does NOT apply to OTR's actual code. (Optional polish: add `h264_videotoolbox` as a Mac hardware fast-path; libx264 already works.)
- **No MPS path yet (the real Mac gap).** OTR has `torch.cuda.is_available()` + `sys.platform=='win32'` checks + `OTR_VideoProbe(cuda_available)`, but NO `torch.backends.mps` routing -- on a Mac the engines fall to **CPU** (works, slow), not Metal. If Mac is a real shipping tier, MPS routing is genuine engineering, not a toggle.
- **cu128 / attention is profile-level, confirmed.** The BUG-070 Sage gate + the LTX/Wan/latentsync/3D sidecars are cu128 -- they CANNOT be one-graph'd onto Ampere/Ada/Mac. Standardize the in-process engines on `sdpa` / `xformers` (Ampere/Ada-safe); keep cu128 engines as profile/sidecar switches that fail-closed off-profile (already the design).

**Refined recommendation:** ONE master switchable graph + a GENERATOR that exports pre-dialed per-tier snapshots + an environment-aware installer (manifest + `extra_model_paths.yaml`/`HF_HOME` + confirm-before-dump). That single design satisfies BOTH panel answers AND the operator preference, and it structurally kills the drift bug (the snapshots and the headless path are both generated from the one master, never hand-edited).
