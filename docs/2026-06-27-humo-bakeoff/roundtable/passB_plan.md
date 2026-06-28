# Step B pivot -- roundtable decision (GPT-5.5 + Gemini-3.1-pro + Claude judge)

Panel: GPT-5.5 (yes-with-fixes), Gemini-3.1-pro (no/keep-1.7B), DeepSeek empty. Spend ~$0.11.
Claude grounded both against the code + Step A data.

## DECISION: PROCEED -- build + test a 17B-GGUF leg, Q3 FIRST.
Gemini's "no" rests on a FULLY-RESIDENT 17B (11.86 + umt5 4.7 + whisper 1.5 ~= 18 GB OOM).
That misreads the execution path: the bakeoff runs via HTTP /prompt + the OTR_BakeoffReclaim
TWO-STAGE encoder evict (NOT eng_humo.render_clip, whose "fully resident / no free_after_use"
docstring Gemini cited). Step A EMPIRICALLY showed the 14B two-stage peaks ~15.86 GB NVML
(weights + ~2 GB), not weights+all-encoders. Scaling from that anchor by UNET weight:
- Q3_K_M (8.4 GB, ~5.6 GB lighter than the ~14 GB fp8 14B) -> ~10-11 GB NVML -> fits <=13.5 well.
- Q5_K_M (11.86 GB, ~2 GB lighter) -> ~13.5-14 GB -> borderline.
So test Q3 first (both agents independently said Q3-first); escalate to Q5 only if Q3 fits
AND the operator wants more quality. Keep humo_1.7B de-blue hardening as the FALLBACK if the
smoke fails or Q3 quality < 1.7B.

## CONFIRMED wiring (both agents converge; grounded)
1. **Bakeoff-only IDs, NOT the OTR "17B" namespace.** GPT MUST-FIX (CONFIRMED): `HuMo17BEngine`
   is the 1.7B tier (`name="humo_1.7B"`, `_HUMO_17B_UNET="humo_1.7B_fp16.safetensors"`, envs
   `OTR_HUMO_17B_*`). Use leg ids `humo_17b_gguf_q3` / `humo_17b_gguf_q5`; do NOT touch
   `OTR_HUMO_17B_*`.
2. **GGUF loader.** Emit `{"class_type":"UnetLoaderGGUF","inputs":{"unet_name":<gguf basename>}}`
   (no weight_dtype) -- mirror eng_wan_i2v. The builder produces this leg by building the 14B
   topology then SWAPPING the unet node class+inputs (do not call eng_humo for the loader).
3. **LoRA-free.** Omit the LoraLoaderModelOnly node; wire `ModelSamplingSD3.model <- unet`
   directly (the 14B-shaped lightx2v LoRA won't apply to 17B). Drive via the env-set
   OTR_HUMO_LORA_NAME=none at build time (eng_humo._build_graph already skips lora on that).
4. **Sampling:** LoRA-free `steps=20`, `cfg=1.0`, `ModelSamplingSD3 shift=8.0`,
   `uni_pc`/`simple`/`denoise=1.0`. 25 steps only if Q-fits + operator says quality is close.
5. **Smoke FIRST (the real risk):** a 33f (`_HUMO_MIN_FRAMES`) min-smoke exercising
   LoadAudio->AudioEncoderLoader->AudioEncoderEncode->WanHuMoImageToVideo->KSampler->VAEDecode.
   ANY model-type / cross-attn / shape error on the GGUF-loaded model = HARD FAIL the 17B path
   (both agents flag this as the #1 risk -- a custom node may reject a GGUF-loaded UNET).
6. **Meter = NVML** (Step A: torch under-reports); reuse the Step-A reset+probe + NVML peak.
   Gate: NVML <=13.5 target / <=14.5 hard.
7. **Loader-agnostic runner** (Codex r3, CONFIRMED): assert_checkpoints + build_manifest must
   read loader_class/loader_param from meta (UnetLoaderGGUF/unet_name), not hardcode UNETLoader.
8. **Isolated legs:** one resident leg per boot (the runner already boots-per-leg) so Q3/Q5
   peaks don't contaminate each other.

## Build order
1. Builder: add `humo_17b_gguf_q3` leg (env-set lora=none+unet=gguf+steps=20 around the 14B
   topology; post-swap unet node -> UnetLoaderGGUF; meta loader_class/param). 2. Runner:
   loader-agnostic checkpoint/manifest. 3. 33f Q3 smoke (hard-fail on any audio/cross-attn
   error). 4. If smoke OK: full 49f Q3 leg -> NVML fit + clip for operator eyeball vs 14B/1.7B.
   5. Q5 only if Q3 fits + operator wants more. KILL-GATE: smoke fail OR Q3 NVML>14.5 ->
   stop the 17B path, keep 1.7B + de-blue.

## Judgment
ACCEPTED (grounded): the full wiring above; Q3-first; NVML meter; loader-agnostic runner;
bakeoff-only IDs (the OTR "17B" namespace is the 1.7B tier). REJECTED: Gemini's keep-1.7B
"no" -- its OOM rests on a fully-resident assumption the two-stage HTTP harness defeats (Step
A data). VERIFY-AT-BUILD: the 33f smoke is the gate -- the GGUF-loaded UNET may be rejected by
WanHuMoImageToVideo's audio cross-attn; if so, the 17B-GGUF path is dead and we keep 1.7B.
