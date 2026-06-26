# R1 judgment (Claude, sole judge)

Panel: openai/gpt-5.5-20260423, google/gemini-3.1-pro-preview-20260219,
deepseek/deepseek-v4-pro-20260423. Spend this pass ~$0.2840.

## Accepted (grounded CONFIRMED -> folded into pass01)
- **Character-beat mis-routing (all 3, independent) -- the headline.** One engine
  serving character + scene beats under family `audio_conditioned_video` mis-routes
  character beats: (a) portrait clobbered by scene still, (b) ambient master audio
  leaked onto face beats via `_uses_ambient_master_audio` (CONFIRMED family-keyed
  @730), (c) scene prompt applied to talking heads. FIX = role-driven routing
  (pass01 Part B0-B3).
- **Capability choice (Gemini F2, DeepSeek, GPT #2):** gate on `init_image in
  required_inputs`, not `accepts_still`. CONFIRMED: station_card inherits
  accepts_still=True (base @447) but must stay excluded (@863-865). Folded (Part B1, F2).
- **VALIDATED_ENGINES (GPT #5):** CONFIRMED @289 lists both legacy names + drives
  the dropdown gate. Folded (Part A) with GPU-validation provenance + smoke-confirm.
- **Missing-still policy split (GPT #3, DeepSeek #5):** required-still fail loud
  pre-GPU vs optional-still degrade to text. Folded (Part B1).
- **OTR_ENABLE_LTX_I2V scope (Gemini #3, GPT #4, DeepSeek #5):** keep scoped to
  ltx_video only. Folded (Part B4, F4).
- **Canvas clamp (Gemini SHOULD):** driver has only the id string -> just use the
  explicit name `("ltx_audio_in",)`, no registry lookup. Folded (Part C.3).
- **Docstring drift (GPT #2 should):** rewrite the "two adapters" narrative. Folded.
- **Pre-merge vs post-merge split (GPT #5):** soak gates the soak, not the code.
  Folded (Part E).
- **Keep ltx_audio_in name (unanimous):** rename cut. Folded (F3).

## Rejected / reframed
- **Gemini CUT "take the narrow safe-mirror" -- REJECTED.** Grounded: the narrow
  mirror (add ltx_audio_in to the ltx_video scene-still branch) STILL clobbers the
  character portrait, because that branch sets the scene still unconditionally for
  the engine. Narrow does not escape the defect; role-driven is required for
  correctness. (GPT + DeepSeek agree robust.)
- **GPT #3 "missing still may continue for ltx_audio_in" -- partial.** ltx_audio_in
  REQUIRES init_image (render_clip raises) -> it must fail LOUD pre-GPU, not
  continue. Encoded as the required-vs-optional split.

## Verify-at-build (downgraded from assertions)
- Per-engine required_inputs/accepts_still/family/render_aspect enumeration
  (station_card route none; ltx_video init_image optional; wan_i2v/flux_still/
  flat_still/mesh_stage unchanged). [GPT #7/#8, my anchor]
- default-role resolution consults default_roles for ltx_audio_in. [my anchor #3]
- no workflow widget beyond node-87 names the two. [my anchor ASSUMPTION]
- no runtime/validator consumer of the _api.json. [GPT #4]

## Convergence
R1 surfaced ONE large, real defect (role mis-routing) with strong cross-model
agreement; pass01 reframes the whole routing to role-driven. NOT converged yet --
advance to R2 (coding plan) to harden the still_route helper signature, the
classifier edge cases, and the test matrix before building.
