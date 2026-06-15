# SA3 settings bake-in — judgment (Claude as judge)

Panel: Opus-4.8, Sonnet-4.6, GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro (~$0.47).
CONVERGED → baked. Defaults are still env-overridable; the operator never NEEDS to set them.

## BAKED VALUES (the deliverable)
| knob | old | NEW | rationale |
|---|---|---|---|
| `OTR_SA3_CONTEXT_S` | 30.0 | **12.0** | Gemini's key insight (multi-reviewer): a 30s context for a 4s cue makes it an aimless mid-song fragment. Context = the LONGEST cue (12s) → each cue is a coherent SLICE of one tight phrase (opening=whole 12s, closing=resolving 2nd half start@4s, interstitial=middle start@4s). |
| `OTR_SA3_CFG` | 6.0 | **7.0** | SA3's native default; 6.0 is slightly weak for prompt adherence (Gemini + my knowledge). |
| `OTR_SA3_STEPS` | 100 | **100** (kept) | Stable Audio's reference step count; SDE sampler benefits from steps; cues are short so cost is trivial. (Panel split 50–100; kept quality-first since determinism/cost aren't the complaint.) |
| `OTR_SA3_SAMPLER` / `SCHEDULER` | dpmpp_3m_sde_gpu / exponential | **kept** | Stable Audio's NATIVE sampler = best quality. Gemini's ODE-swap-for-determinism is moot: the byte-identical golden already PASSES with the SDE_gpu sampler, so determinism holds; switching to dpmpp_2m is image-diffusion thinking and would likely sound worse on SA. |
| `_SA3_NEG_DEFAULT` | …harsh clipping, digital distortion, muddy mix, out of tune, low quality | **vocals…crowd noise, modern pristine mix, digital distortion** | Ban vocals/speech (primary) + push toward vintage ("modern pristine mix"). Dropped "harsh/muddy/out-of-tune/low-quality" — they fight the wanted eerie/theremin/analog-tape texture (Opus, GPT, Gemini, Sonnet consensus). |
| denoise | 1.0 hardcoded | **1.0 (`OTR_SA3_DENOISE`)** | Correct for from-empty generation; now env-overridable for parity (Sonnet/GPT/Grok). |
| seconds_start mapping | intro→0 / outro→tail / else→middle | **+ opening→0, closing→tail** | also-match the literal cue words "opening"/"closing" (the prompts carry "intro"/"outro", but belt-and-braces — Opus/GPT/Grok). |
| genre anchors | — | **+ "instrumental"** | reinforce instrumental-only in the anchor so a music model doesn't bleed vocals (Gemini). |

## ACCEPTED (folded)
- context=12, cfg=7, negative refinement, opening/closing detection, "instrumental" anchors,
  safe env parsing (`_env_float`/`_env_int` fall back LOUD, don't crash on a bad override — GPT),
  `OTR_SA3_DENOISE`, a `log.warning` when context < cue dur (Sonnet), stale docstring fixed (GPT).

## REJECTED / overruled (with reason)
- **Switch to dpmpp_2m/karras for determinism** (Gemini) — REJECTED: the SDE_gpu sampler already
  passes the byte-identical golden, so determinism is not broken; the SDE sampler is Stable Audio's
  native/reference and the quality priority.
- **Cut steps to 50** (Gemini/Opus) — DOWNGRADED: kept 100 (quality-first; SDE benefits from steps;
  short-cue cost is negligible). Operator can lower `OTR_SA3_STEPS` if they want speed.
- **"You MUST A/B before baking"** (Opus/GPT/DeepSeek/Sonnet refrain) — NOTED but the operator
  explicitly asked to bake my best judgment so they don't have to tweak; they A/B live afterward.
- **ConditioningStableAudio is single-conditioning / the call is wrong** (Opus MUST-FIX #1) — MISREAD:
  the same 4-arg `(pos,neg,start,total)→(pos,neg)` call shipped originally and SA3 has rendered as the
  default since 2026-06-03; ComfyUI's node takes both conditionings.
- **Cut the era map to one entry / assert clip not None** (Sonnet) — KEPT: briefs are news-driven and
  can span eras; the CLIP fallback is pre-existing and out of scope here.

## Still operator-gated
A/B listen (the values are a strong starting point, tune `OTR_SA3_CFG`/`CONTEXT_S` to taste), then
RE-BASELINE the `test_audio_byte_identical` golden (the music bytes changed intentionally).
