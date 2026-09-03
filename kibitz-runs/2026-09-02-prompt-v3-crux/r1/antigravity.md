VERDICT: build-ready as-is? no. The plan pairs an expanded prompt payload with an unchanged 77-token CLIP admission cliff, re-introduces negative tokens into SD1.5 positive conditioning, and replaces character wardrobe recurrence with total scene stagnation while bypassing the real root cause of empty beat intents.

MUST-FIX BEFORE BUILD:
1. [D7 / D1 / D5] Arithmetic collapse of the 77-token CLIP window.
   - DEFECT: D7 retains the hard refusal gate `windows == 1` / `tokens <= 77` (`ghost_signal_author.py:1361-1367`), while D1 and D5 expand mandatory prompt components to include the pack's authored `positive_tail` (e.g., `recur_frac` alone is 227 chars / ~45 tokens), the full `meta.story_brief` crux (~25 tokens), and two `story_brief_terms.setting` entries (~10 tokens). Style + Crux alone consumes 70-80 tokens, leaving zero tokens for the beat subject or world motion. Preflight `assert_shell_fits` (`ghost_signal_author.py:1409-1442`) will fail immediately on boot, or D7's dynamic drop order will prune the beat's extracted nouns on virtually every shot.
   - FIX: Keep the front style anchor strictly budgeted (use `compact_style_cue` at <=10 tokens or a tight 2-4 word descriptor) and contract the crux clause to a compact subject kernel (<=15 tokens) so that `style + crux + setting <= 35` tokens, leaving at least 34 tokens for the beat subject, world motion, and punctuation before hitting `GHOST_AUTHOR_TOKEN_TARGET` (69 tokens).

2. [D6] Re-injecting negative exclusion tokens into positive SD1.5 conditioning.
   - DEFECT: D6 specifies keeping "'no people' for the music bed and 'no faces' for the figure beats" as positive law words. This directly violates the hard architectural law established in `ghost_signal_prompt.py:107-113` ("There is no `no people` here and there never will be... a positive clause that attends to an absent human is a request for the model to think about one"). In SD1.5 CLIP conditioning, negative phrases in the positive channel attend directly to the semantic concepts "people" and "faces", actively triggering deformed human bodies and phantom faces.
   - FIX: Purge "no people" and "no faces" from positive prompt generation entirely. Enforce exclusions strictly through the negative conditioning channel (`LANE_HYGIENE_NEGATIVE` and `compose_ghost_negative`). Keep the positive law purely affirmative ("unbroken shot") or remove positive law tokens completely.

3. [D2 / D1] Narrative arc destruction and visual stagnation via total character erasure.
   - DEFECT: D2 eliminates all character presence ("The character disappears unless the story is about a person... No motif_for_character on v3") and substitutes the static `meta.story_brief` crux on every beat. This misinterprets the operator's guidance (Rule 1 & Rewrite 4: "characters moving through that world -- small, in it, never a coat in close-up"). Eliminating characters turns an ensemble radio drama into 27 consecutive shots of an empty environmental backdrop (e.g., repeating the floating landmass 27 times), destroying visual progression and creating extreme prompt-inertia stagnation.
   - FIX: Reframe D2 so `character_video` beats place figures in spatial context within the world (e.g., "figures navigating the <setting>", "a researcher observing the <beat_subject>") without the rigid wardrobe/prop allow-list, preserving character agency and shot variety across the episode arc.

4. [D4 / D10] The spoken dialogue noun fallacy vs deferring narrative intent.
   - DEFECT: D4 builds an allow-list scraper for spoken dialogue nouns while D10 defers diagnosing why `lines[*].beat_intent` is empty. Radio drama dialogue is inherently conversational and emotional ("What did you find?", "I don't trust them", "We need to leave"), containing zero concrete physical nouns. In dialogue-heavy scenes, D4 will extract nothing and collapse into bare crux duplication (`<crux>, <crux>`). The original v2 failure ("humans and bags") occurred precisely because `beat_intent` was dropped upstream (`story_scaffold_enabled: False`), leaving the prompt author blind to scene action.
   - FIX: Do not rely on spoken dialogue noun scraping as the primary beat visual. Prioritize passing the scene/action narrative context into the prompt author, or fall back to the scene setting description rather than bare crux repetition when spoken dialogue lacks physical objects.

5. [D9 / Q5] Replay-time prompt re-composition destroys provenance and cache integrity.
   - DEFECT: D9 proposes a "replay-time prompt re-composition switch" where prompts are mutated at replay time while seeds remain frozen. In `otr_shot_lock.py:2547-2566` and `render_driver.py:2889-2936`, `request_sha256` and `render_request_hash` are the foundational cryptographic keys for request admission, validation, and ComfyUI caching. Mutating prompt text inside the replay branch creates a corrupted state where cached request hashes no longer match the rendered prompt text, violating ComfyUI caching contracts (Domain Profile Rule 4).
   - FIX: Keep replay execution strictly immutable. Implement v3 comparison as a distinct execution run (generating a fresh execution plan with updated hashes and `prompt_version="ghost_signal_v3"`) pinned against the frozen seeds of the v2 run, rather than mutating text inside the replay harness.

SHOULD-FIX:
1. [D3 / Q2] Creative suppression via deterministic slotting.
   - DEFECT: The operator explicitly mandated: "Fewer variables per prompt. Get creative." D3 inverts the pipeline by having Python assemble slots 1, 2, and 3 via rigid regex/concatenation, demoting the LLM to an optional single-clause decorator for "world motion" that is discarded on failure. This mechanical string templating guarantees repetitive, monotonous prompt syntax across episodes.
   - FIX: Allow the LLM author to creatively synthesize the crux, setting, and beat action into a cohesive visual sentence, using Python as a post-generation schema and budget validator rather than a rigid mad-libs assembler.

2. [D4 / Q3] Bounded allow-list vs open POS extraction risks.
   - DEFECT: D4 proposes an allow-list discipline without defining vocabulary boundaries. A closed 40-word allow-list fails across diverse sci-fi subgenres, while an unconstrained POS extractor feeds abstract nouns, proper nouns, and technical jargon directly into SD1.5, triggering severe text/caption hallucinations.
   - FIX: Constrain noun extraction to concrete physical nouns by passing candidates through an explicit stoplist that rejects abstract concepts and names, followed by `validate_drawable_beat` to reject lettering triggers before prompt assembly.

3. [D7] Dynamic trimming violates the "never trim protected components" invariant.
   - DEFECT: D7 introduces an in-flight cascading drop order ("the beat's third noun phrase, its second, the setting's second term..."). `finalize_ghost_prompt_v2` is explicitly contracted: "IT NEVER TRIMS AND NEVER REPAIRS" (`ghost_signal_author.py:1306-1308`). Adding dynamic trimming inside finalization obscures budget breaches and complicates debugging.
   - FIX: Enforce prompt length and token limits upstream at author/composition time so `finalize_ghost_prompt_v2` remains an immutable validation gate.

4. [D5] Setting stutter between crux and setting terms.
   - DEFECT: D5 threads the first two `story_brief_terms.setting` entries alongside `meta.story_brief`. Because `meta.story_brief` already incorporates the primary setting ("A claustrophobic research station monitors..."), appending `subsurface research station, remote reservoir` introduces redundant repetition ("research station... subsurface research station"), wasting 10-15 tokens.
   - FIX: Deduplicate setting terms against words already present in the crux before appending, or only append setting terms when the crux lacks an explicit location.

OPTIONAL / NICE-TO-HAVE:
1. [D8] Enhanced slot observability: Record per-slot token counts in `observability.prompt_slots` so log receipts explicitly prove which slot consumed what portion of the 77-token budget.
2. [Section 1, Lines 51-54] Empirical stability sweep: Run the probe runner over identical seeds to isolate whether frame jitter is driven by prompt token count or motion module weights (`mm-p_0.5.pth` vs v3).

CUT THESE (scope / over-engineering):
1. [D4] Spoken dialogue noun scraping: Safe to cut. Spoken dialogue is what characters say, not what the visual scene depicts. Relying on an NLP scraper introduces brittle regexes and POS tagging overhead.
2. [D6] Positive framing law floor ("no people", "no faces"): Safe to cut. Exclusions belong in the negative prompt, and framing guidance is already deprecated.
3. [D9] ShotLock replay-time prompt re-composition hook: Safe to cut. Running a dedicated test pipeline with pinned seeds achieves the exact same visual A/B comparison without adding mutating bypasses to the production replay engine.

[ASSUMPTION] Assumes `meta.story_brief` is consistently populated with high-quality, descriptive text across all narrative engines outside the studied `scifi_news_pro` bank.
[ASSUMPTION] Assumes the AnimateDiff v2 motion module (`mm-p_0.5.pth`) maintains coherent temporal motion when human figures are entirely absent and prompts describe only environmental "world motion".
[ASSUMPTION] Assumes the 10 beat rewrites from a single episode (`the_last_reading`) reflect universal operator preferences across all 9 visual styles and drama formats.
