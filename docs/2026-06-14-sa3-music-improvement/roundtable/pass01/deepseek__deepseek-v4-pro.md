<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The document is a design spec, not a build-ready plan. It asks questions instead of answering them, and the questions it asks rest on unverified assumptions about SA3 behavior. The panel must converge on concrete answers before a coder can act.

MUST-FIX BEFORE BUILD:
1. [§3.A, §4.1] Prompt grammar is undefined. The document asks "what is the highest-leverage SA3 prompt structure" but does not provide the template. Without a concrete template, the coder cannot implement `compose_music_prompt`. The plan must deliver the exact template (genre, instruments, BPM, key, mood, production descriptors) and the period→genre/instrument mapping table. The grounding excerpt shows the current code uses abstract mood-language (e.g., "minor mode, unresolved tension") — the plan must specify the replacement SA3-shaped template.
2. [§3.B, §4.2] Negative prompt is undefined. The document asks what SA3's negative prompt should contain but does not decide. The current code in `eng_stable_audio_3.py` uses `""` (empty string). The plan must specify the exact negative prompt string.
3. [§3.B, §4.3] Sampler inputs are not decided. The document asks whether `steps=100, cfg=6.0, dpmpp_3m_sde_gpu/exponential` is reasonable but does not converge on values. The plan must specify the final `steps`, `cfg`, `sampler`, `scheduler`, and `denoise` values.
4. [§3.C, §4.4] Duration/timing strategy is not decided. The document asks whether to render longer and trim vs. render exact short cues, but does not converge. The current `CUE_DURATIONS` are 12/8/4 seconds. The plan must specify: (a) whether to change `CUE_DURATIONS`, (b) whether to render a longer bed and trim, and (c) how `seconds_total` conditioning relates to the wanted clip length.
5. [§3.D, §4.5] Model choice is not decided. The document asks whether `stable_audio_3_small_music` is sufficient or whether a larger checkpoint should be used. The plan must specify the exact checkpoint name and confirm it fits in 16 GB VRAM.
6. [§3.E, §4.6] Seed strategy is not decided. The document asks about best-of-N at author time but does not converge. The plan must specify whether to use single-seed or best-of-N, and if best-of-N, the N value and selection criteria.
7. [§2, §5] The frozen audio spine constraint is stated but the plan provides no verification mechanism. The plan must specify how to verify that the spine remains byte-identical after changes (e.g., "run `test_audio_byte_identical` before merge"). [ASSUMPTION: the test exists and covers the music path.]

SHOULD-FIX:
1. [§1.5] The document notes "Two music paths may diverge" and the theme node lists `_LEGACY_FIRST_FALLBACK = ("musicgen","stable_audio_music")`. The plan does not address whether the saved `otr_scifi_16gb_full.json` uses musicgen or SA3 for music. The coder must verify which engine is actually in use; if musicgen is still the default in saved ledgers, the SA3 prompt changes may not affect the operator's output. [ASSUMPTION: the operator is using SA3, but this must be confirmed.]
2. [§3.A] The plan says "period→genre/instrument mapping (e.g., 1950s sci-fi radio → 'vintage orchestral sci-fi, theremin, brass, timpani, eerie strings, analog tape warmth')" but does not provide the full mapping table. The coder needs the complete mapping for all period descriptors used in the Meta brief.
3. [§3.C] The document notes "very short total-seconds (esp. 4s) is a known weak spot" for SA3. If the decision is to render longer and trim, the plan must specify the trim/fade logic (where does it live? in `generate_clip`? in the theme node?).
4. [§1.6] The determinism contract says "per-cue seed via `_seed_to_int64(music_seed_base, slot)`" and "KSampler builds its generator internally." The plan does not address whether the new sampler inputs (steps, cfg, sampler, scheduler) could break determinism. The plan must specify how to verify determinism after changes (e.g., render twice, compare waveforms).

OPTIONAL / NICE-TO-HAVE:
- The document's §4.6 asks about best-of-N seed selection at author time. If the panel decides this is out of scope for the immediate fix, state that explicitly.
- The document mentions "A/B listen plan" in §6(f) but does not define it. Consider specifying: generate 3 cues with old vs. new prompt/sampler, blind-listen, pick winner.

CUT THESE (over-engineering):
1. [§4.5] The question about "larger SA3 music checkpoint" may be premature. The document itself notes SA3 was chosen for dependency-clean operation. If `stable_audio_3_small_music` is the only ungated, local, 16 GB-viable checkpoint, the panel should converge on it and move on. [ASSUMPTION: no larger ungated SA3 music checkpoint exists or fits in 16 GB. Verify: check HuggingFace for `stable_audio_3_medium_music` or similar.]
2. [§4.6] The best-of-N seed selection scheme is over-engineering for this roundtable. The operator's complaint is about prompt/input quality, not seed luck. Single-seed determinism is sufficient; best-of-N can be a follow-up.

HIDDEN DEPENDENCIES / SEQUENCING:
1. [§3.A → §3.B] The prompt grammar and negative prompt are coupled: the negative prompt must complement the positive prompt's style. If the positive prompt specifies "vintage orchestral," the negative should exclude modern production artifacts. These must be designed together.
2. [§3.B → §3.C] The sampler inputs and duration are coupled: SA3's `seconds_total` conditioning interacts with `steps` (more steps may be needed for longer durations). These must be tuned together.
3. [§3 → §6.f] The test plan depends on all other decisions being made first. The A/B listen plan cannot be defined until the prompt, sampler, and duration are fixed.
4. [§1.5 → §2] The "two music paths may diverge" issue must be resolved before testing: if the operator's saved ledger uses musicgen, the SA3 changes will not affect their output. [ASSUMPTION: the operator is using SA3, but the document does not confirm this.]

ASSUMPTIONS:
- The operator is actually using SA3, not musicgen. The document notes the default flipped 2026-06-03, but the operator's saved ledger may have an explicit engine override. [Verify: check `otr_scifi_16gb_full.json` for the music engine field.]
- `test_audio_byte_identical` exists and covers the music path. [Verify: check the test suite for this test.]
- No larger ungated SA3 music checkpoint exists. [Verify: check HuggingFace for `stable_audio_3_medium_music` or similar, and confirm VRAM requirements.]