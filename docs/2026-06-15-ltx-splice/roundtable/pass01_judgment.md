# Roundtable pass 01 — judgment (Claude as judge/grounder)

Panel: GPT-5.5 + Gemini-3.1-pro-preview + DeepSeek-v4-pro (3 models, ~$0.18). Grounded against
`eng_ltx_video.py`, `registry.py`, `eng_ltx_av.py`. Verdict from all 3: NOT build-ready as written —
and they were right. Convergence: this pass found material spec-gaps (now folded); the gaps were
specification, not architecture. Recommend proceed to coding after operator nod (or one cheap pass02).

## ACCEPTED (grounded CONFIRMED → folded into pass01 plan)
- Phase 0 must rip `ltx_orbit` ONLY; the 2B/T5 loader deletion moves into the Phase 1 commit (else the
  engine bricks before it can go green). All 3 panelists; confirmed (those are the live `ltx_video` loaders).
- Killed the self-contradiction: §3.0 delete vs §3A "keep ltx_orbit". Now delete-only; drop from `__all__`.
- Graph rewrite is explicit now: node keys `unet/videovae/te/lora`; replace every `W("checkpoint",0)`→`W("lora",0)`
  and `W("checkpoint",2)`→`W("videovae",0)` in BOTH builders. (GPT/Gemini, grounded.)
- `render_clip` `keep={"checkpoint"...}` / `results.get("checkpoint")` (lines 805/809) must follow the new
  model node key or VRAM leaks. (Gemini, grounded.)
- Canvas defaults `_LTX_DEFAULT_W/H` are 768/512 (lines 168-169) → set 832/480. (GPT/DeepSeek, grounded.)
- Path/usability rewrite (`_ckpt_path`,`_text_encoder_name`,`_installed`,`load`,`assert_usable`) for GGUF+Gemma+
  LoRA+VAE, mirroring `eng_ltx_av.py`. (GPT/DeepSeek/Gemini, grounded.)
- LoRA: drop the `"22b"` gate; always include; fail closed if missing. (GPT, grounded.)
- Registry `CAPABILITIES["ltx_video"].model_requirements` 2B→new models; re-measure VRAM. (GPT/Gemini.)
- Flag: `OTR_ENABLE_LTX_VIDEO` already defaults ON (line 326) — keep as kill switch; BUG-413 ≠ flag. (DeepSeek, grounded — corrected my plan.)
- Motion-acceptance smoke (frame-diff across roles), BUG-413 i2v-seed regression assert, license grep test,
  stale-comment cleanup, all-roles GPU smoke. (GPT, accepted.)

## OPERATOR DECISIONS (panel could not decide)
- (A) Sampler = **distilled** — the frozen mini JSON is the source of truth; gated by the motion smoke. (`ksampler` stays env fallback.)
- (B) per-beat = **scene/b-roll** — `LtxVideoEngine.roles` already covers it; do NOT add `character_video` (HuMo keeps faces).
- (NEW) LTX i2v init = **full-frame FLUX still, never a portrait** (operator) — folded into §5.

## REJECTED / DOWNGRADED (grounded MISREADS — kept the proven mini values)
- Gemini "VAEDecodeTiled temporal_size 4096 = guaranteed OOM, use 64": REJECTED — b001 ran 4096 in production
  with no OOM; mini is source of truth. 64 kept only as a verify-at-build fallback for longer-than-tested shots.
- Gemini "Gemma device=default = guaranteed OOM": DOWNGRADED — mini ran `default` fine (1 shot). `device="cpu"`
  + lighter projection-ckpt accepted as a verify-at-build perf tweak IF many-shot VRAM pressure appears.
- GPT "ModelSamplingLTXV required": REJECTED for our recipe — the frozen mini omits it and looks good; verify-at-build only.

## CUT (over-engineering — panel agreed)
- No GGUF-quant widget appended to the JSON unless a knob is truly wired. No full JSON round-trip for the Phase 0
  rip (no JSON change). No `ltx_av` sampler alignment this sprint.

## STILL OPEN (verify-at-build, see plan §8)
device cpu vs default; temporal 4096 vs 64; ModelSamplingLTXV; `commercial_clean` flip pending the profile-filter check.
