# How much MOTION can the local lip-sync lanes actually do?

**A research question, and the operator was explicit about the standard:**
*"I want to research how much motion they can actually do, based on research not
assumption."* So: no analogies, no "audio-driven models are probably
talking-head only". Evidence or an admitted unknown.

**The decision this feeds.** The action-prompting scope
(`docs/2026-08-27-action-prompting/SCOPE.md`) gives body/camera motion to the
three CLOUD audio-in lanes alongside their dialogue. The six LOCAL lip-sync
lanes are the same shape. They are currently OUT only because the operator said
"all the cloud". If they can hold motion, they belong in; if they cannot, the
exclusion needs to be a finding rather than an oversight.

## THE LANES

| lane | family | LOCAL weights actually loaded | notes |
|---|---|---|---|
| `humo` | audio_driven_face | `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` (`eng_humo.py:209`) | **HuMo 14B**, Wan 2.1 base, Kijai fp8 e4m3fn scaled. Portrait. |
| `humo_14B_169` | audio_driven_face | same 14B unet, wide render aspect | **HuMo 14B**, 16:9 |
| `humo_1.7B` | audio_driven_face | `humo_1.7B_fp16.safetensors` (`eng_humo.py:1283`) | **HuMo 1.7B** fp16. Portrait. Its knobs live in the `OTR_HUMO_17B_*` namespace. |
| `humo_1.7B_169` | audio_driven_face | same 1.7B unet, wide render aspect | **HuMo 1.7B**, 16:9 |
| `minimax_h3_audio_in` | audio_conditioned_video | unet `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (`eng_minimax_h3.py:288`); clip `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`; video VAE `minimax_h3_video_vae_fp16.safetensors`; audio VAE `minimax_h3_audio_vae_fp32.safetensors` | **MiniMax H3, REF2VA** branch -- a DIFFERENT unet from the silent `minimax_h3_video` lane, which uses `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (`:277`). Pruned int8 convrot. |
| `ltx_audio_in` | audio_conditioned_video | unet `ltx-2.3-22b-dev-Q3_K_M.gguf` (`eng_ltx_av.py:576`, env `OTR_LTX_AV_UNET`); fp16 sibling `ltx-2.3-22b-dev.safetensors`; distill LoRA `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`; text encoder `gemma_3_12B_it_fp4_mixed.safetensors` | **LTX 2.3 22B dev**, Q3_K_M GGUF as shipped. NOT LTX 2.5 -- that is the silent/foley family. |

**These are the LOCAL checkpoints on this box, not the vendors' reference
models.** Quantisation matters to the question: a pruned int8 or Q3_K_M model may
hold less motion than the fp16 the model card was written about. Say which you
are citing.

## WHAT THE REPO ALREADY PROVES -- and it inverts the question

**Motion direction is WIRED ON EVERY ONE OF THEM. Nobody is sending any.**

Four of six do not list `text_prompt` in `required_inputs`, which reads like
"these lanes take no prompt". That reading is WRONG -- `required_inputs` says
what the ROLE must supply, not what the adapter uses. Every adapter takes a text
prompt into its graph and falls back to a default when none arrives:

* **HuMo** -- `eng_humo.py:831`
  `positive = plan.get("text_prompt") or "a person speaking, subtle facial motion"`,
  encoded at `:848` and fed to the sampler at `:862`/`:883`. **Its own default
  asks for SUBTLE FACIAL MOTION** -- so on a HuMo beat with no authored prompt,
  the lane is actively being told to keep still.
* **MiniMax H3 audio-in** -- `eng_minimax_h3.py:692`
  `return get("text_prompt") or self._DEFAULT_PROMPT`. And its own
  `PROMPT_STYLE_DIRECTIVE` (`:110-114`) says: *"Name the subject, then ONE ACTION
  AND ITS SPEED as flowing prose, never a keyword list."* **The engine is asking
  for action and we are not giving it any.**
* **LTX audio-in** -- `eng_ltx_av.py:858`, and its directive (`:66-70`) already
  contemplates motion with a constraint: *"Describe motion that does not turn the
  head away or leave frame."*

So the question is NOT "can we send motion" -- the wire exists on all six. It is
**how much motion each model can render before the lip-sync degrades**, which is
a capability question, not a plumbing one.

## THE ACTION MACHINERY ALREADY EXISTS AND IS SWITCHED OFF

**The operator was right to ask.** `nodes/_otr_motion_clause.py` is a complete
per-beat motion-clause system -- it derives a real motion clause from each beat's
dialogue and cast, writes it to `ledger['video']['shots'][i]['motion_clause']`,
and the render path reads it read-only. It came out of the
`docs/2026-07-29-motion-floor-brief.md` campaign and carries a 2026-08-17
"kinetic amendment" that OPENED its length cap from 70 to 130 chars on operator
direction, because *"at 70 the writer could not say what a body does without
running out of characters, so the cap was itself a damping force."*

**It is opt-in and DEFAULT OFF** (`_otr_motion_clause.py:16,30`, gate
`OTR_LTX_MOTION_CLAUSE`; consumer at `otr_video_render_batch.py:578-599`). With
the flag unset the batch pass writes STATIC FALLBACKS only, so the composed
prompt stays byte-identical to a no-motion build.

**It was NOT enabled on the foley leg** -- zero `motion_clause` lines in that
leg's server log -- and the env var appears in no profile, launcher, or workflow
in this repo. So the ho-hum action is at least partly a switch nobody turned on,
not a system nobody built.

**This changes the first research question.** Before asking what these models
CAN do, establish what they do with the motion clause ENABLED, because that is
free and already written. Only then ask whether the clause needs to go further.

## WHAT IS NOT KNOWN, AND MUST BE RESEARCHED

For EACH of the three model families (HuMo, LTX-2.3 audio-in, MiniMax H3):

1. **Does the model accept body/camera motion direction at all, or is the text
   conditioning effectively face-only?** Cite the model card, paper, or the
   upstream node's own documentation.
2. **What breaks first as motion increases** -- lip-sync accuracy, identity
   drift, the face leaving frame, or nothing?
3. **Is there a documented ceiling** -- a recommended motion vocabulary, a
   "keep the subject centred" constraint, a frame budget that limits travel?
4. **Does motion direction interact with the driving audio?** HuMo is
   audio-DRIVEN (`audio_driven_face`); LTX/H3 are audio-CONDITIONED. Those are
   different mechanisms and may answer differently.
5. **Which of the four HuMo variants differ?** 1.7B vs 14B, portrait vs wide
   (`humo_1.7B` / `humo_1.7B_169` / `humo` / `humo_14B_169`). A 14B may hold
   motion a 1.7B cannot.

## WHAT COUNTS AS EVIDENCE

**Admissible:** the model card or paper; the upstream ComfyUI node's own
documentation or source; a live A/B on this box at the SHIPPED recipe; the
`vram-recipe-lab` if it already probed motion on these lanes.

**NOT admissible:** reasoning by analogy from other models; "audio-driven
implies static"; a claim with no citation; a single cherry-picked clip.

**Web lookup IS allowed** (operator ruling 2026-08-15, the RSS precedent) -- so
model cards and papers are fair game, and preferred over speculation.

## THE LIVE PROBE, IF IT COMES TO ONE

Cheapest honest test, and it must be A/B not single-arm: same still, same audio,
same seed, same shipped recipe -- one arm with the current low-motion default,
one with an explicit action clause. Judge lip-sync quality AND motion, and judge
them separately. **Render to a NEW directory** -- an audition re-run in place
destroys the record that cites its hash. **The recipes are not on the table**:
this measures what the SHIPPED recipe does with a different prompt, and never
changes the recipe to get a better number.

## THE HONEST DEFAULT IF THE RESEARCH IS INCONCLUSIVE

Leave the six local lanes OUT of action prompting and say WHY in one line. An
unproven capability that degrades lip-sync on the lanes whose entire job is
lip-sync is a worse trade than a static-but-correct talking head. But record it
as "not established", never as "not possible".
