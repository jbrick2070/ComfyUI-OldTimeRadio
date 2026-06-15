<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-4.6-sonnet-20260217 -->

VERDICT: yes-with-fixes. The document is internally consistent and the code matches most claims, but several concrete numeric recommendations rest on unverifiable model-specific assertions, one structural logic bug exists in the code, and two specification ambiguities would produce wrong runtime behaviour.

---

MUST-FIX BEFORE BUILD

1. [_sa3_clip_window / Question 3b] **Outro placement logic is wrong for the stated use case.**
   The document says outro→tail, meaning `seconds_start = ctx - dur` so the cue occupies the final `dur` seconds of the context window. The code does `start = max(0.0, ctx - dur)` which is correct only when `ctx >= dur`, and that is guaranteed by `ctx = max(context_s, dur)`. However, the document also says the interstitial is 4 s and context is 30 s. For a 4 s outro: `start = 30 - 4 = 26.0`. That means the VAE decodes 4 s of latent conditioned as seconds 26–30 of a 30 s piece. That is correct for "tail" semantics. **But** the interstitial formula `(ctx - dur) / 2` for a 4 s cue gives `start = 13.0`, placing it at the exact midpoint of a 30 s piece. If the roundtable recommends `CONTEXT_S = 47`, the interstitial start becomes `21.5 s` — more than halfway through a 47 s piece, which is not "middle" by any musical definition (it is past the climax in most forms). The document must either fix the formula to `(ctx - dur) / 2` being explicitly validated as musically correct for the chosen `CONTEXT_S`, or adopt a percentage-based anchor (e.g. 40 % of `ctx`). **Concrete fix:** pin `CONTEXT_S` to a value where `(ctx - dur) / 2` lands in a musically neutral zone, or change the interstitial formula to `start = ctx * 0.35` (before the structural midpoint).

2. [generate_clip / Constraints] **`EmptyLatentAudio().generate(dur, 1)` signature is unverified.**
   The code calls `audio_nodes.EmptyLatentAudio().generate(dur, 1)` with positional args `(duration, batch_size)`. Verify: ComfyUI's `EmptyLatentAudio` node INPUT_TYPES and `generate` method signature — if the node expects `seconds` as a named kwarg or the argument order differs, this silently produces wrong-length latents. The document never cites the actual node signature. **Fix:** check `comfy_extras/nodes_audio.py` in the target ComfyUI revision and add an integration test that asserts `audio["waveform"].shape[-1] == int(dur * sample_rate)`.

3. [Question 1 / CFG] **The CFG recommendation has no empirical grounding for SA3 small specifically.**
   The document asks whether 6.0 or 7.0 is correct and notes "SA3's common default is ~7." The code default is 6.0. The plan says Claude will bake the converged value — but no A/B data, no SA3 small model card citation, and no listening test result is provided. Baking either value without evidence is a gamble. **Fix:** before baking, run at minimum one seed × three CFG values (5.5, 6.0, 7.0) on each cue length and record whether harshness arises. Do not bake a value the roundtable "recommends" without a single empirical data point on this specific model.

4. [Question 2 / Steps] **Same problem: the 50–60 step claim for dpmpp_3m_sde is unverified for SA3 small.**
   `dpmpp_3m_sde` is a 3rd-order multistep SDE solver; its convergence rate on SA3's latent space is not the same as on image diffusion. The document asserts "50–60 steps look/sound identical" without a citation or test. If the model was trained with a specific number of inference steps (common for flow-matching or consistency variants), deviating silently degrades quality. **Fix:** verify SA3 small model card or training config for recommended inference steps before cutting from 100.

---

SHOULD-FIX

5. [generate_clip / determinism] **`KSampler().sample(... 1.0)` — the trailing `1.0` is denoise.**
   Passing `denoise=1.0` is correct for full generation, but it is hardcoded and not env-overridable, while every other knob is. If a future operator wants to test partial denoising (Question 6 mentions "a denoise value"), there is no hook. This is a minor inconsistency with the stated "env-overridable" constraint. **Fix:** add `denoise = float(os.environ.get("OTR_SA3_DENOISE", "1.0"))` and pass it.

6. [Question 3 / CONTEXT_S] **The document says SA Open is trained up to ~47 s but gives no source.**
   [ASSUMPTION] This figure is taken from the Stability AI SA3 model card or community reports. If the actual training ceiling is different (e.g. 95 s for some SA variants), the recommendation to use 40–47 s as an upper bound is wrong. **Fix:** verify the exact `seconds_total` training range from the SA3 small model card before baking `CONTEXT_S`.

7. [_sa3_clip_window] **`ctx = max(float(context_s), dur)` silently expands context for the 12 s intro if `context_s` is ever set below 12.**
   If an operator sets `OTR_SA3_CONTEXT_S=10` and the intro is 12 s, `ctx` becomes 12, `seconds_start=0`, `seconds_total=12` — the structural context collapses to exactly the cue length, defeating the whole BUG-408 fix. No warning is emitted. **Fix:** add a `log.warning` when `context_s < dur` so the operator knows the context was silently clamped.

8. [Negative prompt / Question 5] **"out of tune" in the negative prompt may suppress intentional microtonal or atonal content.**
   For 1950s atomic-age sci-fi, theremin and eerie strings are often deliberately "out of tune" by Western standards. The document itself warns "do NOT blanket-ban dissonant" but does not apply the same logic to "out of tune." **Fix:** replace "out of tune" with "severely out of tune, pitch instability" or remove it and rely on the positive prompt's specificity.

9. [_sa3_augment_prompt] **Era-keyword matching is first-match-wins with no logging.**
   If the prompt contains both "1940" and "1960" (e.g. a brief that spans eras), the first matching tuple wins silently. For a deterministic production pipeline, silent genre selection is a traceability gap. **Fix:** log the selected genre clause at DEBUG level (parallel to the existing `log.info` in `generate_clip`).

---

OPTIONAL / NICE-TO-HAVE

- Question 6 mentions BPM hints: SA3 small does respond to tempo language in the prompt ("slow 60 BPM pulse," "stately 80 BPM"). Adding a per-cue BPM hint to the prompt-composition contract (not the negative prompt) could improve rhythmic coherence for the 12 s opening cue without any code change.
- The `_phash` in the log covers prompt content but not the `seconds_start`/`seconds_total` values; those already appear on the same log line, so this is fine as-is.
- Consider logging `context_s` source ("env" vs "default") to make A/B tuning sessions easier to reconstruct from logs.

---

CUT THESE (over-engineering)

1. [load() / CLIPLoader fallback] The `if clip is None: clip = comfy_nodes.CLIPLoader().load_clip(...)` branch. The document states the checkpoint is `stable_audio_3_small_music.safetensors` from Comfy-Org, which bundles the conditioner. [ASSUMPTION] If the Comfy-Org packaging always includes the clip, this branch is dead code that adds a silent fallback obscuring a real packaging error. Either assert `clip is not None` and raise `MISSING_MODEL`, or document explicitly when this branch fires. Dead silent fallbacks in a fail-closed system are contradictory.

2. [_SA3_PERIOD_GENRE tuple / era matching] Three era tuples covering 1920–1970 for a show that is exclusively "1950s atomic-age sci-fi" per the use-case section. The 1920s/1930s and 1960s/1970s branches will never fire for "Signal Lost." Cut to one default genre string (the 1940/1950 entry promoted to `_SA3_DEFAULT_GENRE`) and remove the loop entirely. Saves complexity with zero loss for this production.