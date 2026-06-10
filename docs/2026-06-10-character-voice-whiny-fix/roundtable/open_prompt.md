# Open review: make the AI character voices sound RIGHT

You are one independent expert on a review panel. The DOC below is a working plan for fixing "whiny" / pleading / thin character voices in an automated radio-drama generator (ComfyUI pipeline: IndexTTS2 zero-shot voice cloning for characters, Kokoro announcer, per-line 8-dim emotion vector derived from text keywords, a reference-clip bank with seeded casting). GROUNDING excerpts from the real code are attached after the DOC.

This review is deliberately open-ended: we want your independent, creative read of the PROBLEM, not just a line-edit of the plan.

1. **Blind spots.** What plausible causes of a thin / pleading / "whiny" character sound does the DOC miss entirely? Consider the whole chain: reference-clip acoustics (duration, SNR, mic coloration, pitch, read style), zero-shot cloner behavior on weak refs, emotion conditioning, text preparation and punctuation, seeds, sample rates and resampling, loudness and mixing against music beds, and the psychoacoustics of how a lone voice reads in a drama mix.
2. **Craft.** What would a voice director or audio post engineer do here that software engineers tend not to think of? Concrete, automatable ideas preferred (one-person shop, no actors, no per-line manual work).
3. **Critique the plan.** Anything in P0-zero..P4 that is wrong, mis-ordered, riskier than it looks, or not worth doing? Anything cheaper that dominates an existing step?
4. **Wildcards.** Up to 3 unconventional ideas worth a 30-minute experiment each, even if they sound odd. Mark speculation as speculation.

Rules: be specific and falsifiable where possible; cite the DOC or GROUNDING when you rely on it; mark claims about IndexTTS2 internals you are not certain of as UNSURE. Do not rewrite the plan; return findings. Hard constraints that hold regardless: deterministic re-render (same code + config + seed = same bytes; version-bumped changes are fine), frozen master-mix byte-identity for a fixed config, 16 GB VRAM with a single resident heavy model, fully local render path (no cloud TTS), no new TTS engines inside this fix.
