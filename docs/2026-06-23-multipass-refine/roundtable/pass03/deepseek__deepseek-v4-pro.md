<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no — the plan's scoring scheme uses `build_sq_data` after grounding, making `ungrounded_crisis_density` impossible to measure, and several critical interfaces are missing or misordered.

MUST-FIX BEFORE BUILD:

1. [step-2] Scoring function `score_story_candidate` claims to use `sq_by_beat` from `build_sq_data`, but `build_sq_data` mutates beat intents by grounding generic crisis nouns (see `ground_crisis_nouns` in `_otr_story_quality_l12.py`). This makes `ungrounded_crisis_density` always zero and the metric useless. Scoring must be performed on the raw outline intents BEFORE any grounding; `build_sq_data` should be called only on the winner after selection, or the scoring must compute ungrounded density independently without mutation.

2. [sequencing-F2] When `OTR_STORY_BEST_OF_N` is active, the downstream F2 block (the existing L1/L2 `build_sq_data` call in `OTR_LedgerScriptWriter.run()`) would run again on the winner, duplicating grounding and conflict assignment. This violates "no double freeze" and may skew the composer's inputs. The L1/L2 block must be conditional: skip it if best‑of‑N already grounded the winner's beats, or restructure so grounding happens once.

3. [diversity-injection] The plan states each candidate outline is generated with "a structural-diversity constraint in the prompt". `generate_outline` and `OutlineRequest` provide no mechanism to inject a per‑candidate prompt variation — the request is a frozen dataclass and the prompt is built by `_build_user_prompt` with no configurable diversity parameter. Without this hook, the loop cannot produce the promised structural diversity; either the OutlineRequest must be extended or a custom system‑prompt overlay must be passed to `generate_outline`.

4. [episode-seed-missing] The plan's "seed-keyed determinism" relies on an `episode_seed` for per‑candidate RNG (`sha256(f"{episode_seed}:{n}")`), but the writer's `seed` widget was removed (BUG-LOCAL-269/270) and no other deterministic episode‑level seed is available. The best‑of‑N loop cannot be seed-keyed as described; either a stable seed must be derived from the remaining inputs (e.g., hashed news article + style) or a new hidden input must be added.

5. [local-only-gate] The plan requires disabling best‑of‑N (N=1) when the creative model is a paid/remote provider (OpenRouter, Comfy Credits). The writer must detect the provider before the loop; currently no provider‑detection point exists where `run()` can gate on `resolved["creative_writing_model"]` being a remote handle (e.g., `openrouter:slot-a`). Implementation must probe the loaded model's provider (via `_otr_model_loader.request_slot` or the cache entry) and fall back to a single call.

6. [flag-gate] The feature flag `OTR_STORY_BEST_OF_N` is not defined or read anywhere. An environment variable (or equivalent) must be added, read at the top of `run()`, and used to gate the entire best‑of‑N path; default OFF.

SHOULD-FIX:

7. [scoring-grades] The `score_story_candidate` function and its mapping to a letter grade (B‑or‑better = pass) are not specified. Definition must be provided before building, along with the interpretation of fields like `character_want_clarity`.

8. [telemetry-key] The meta key `meta.story_quality.best_of_n` may collide with existing `story_quality` dict populated by L1/L2; the plan must define a safe merge or a sub‑key.

9. [v1-scope] The v1 "B+ loop" is described but is out of scope for the v0 build target; treat it as future work to avoid mixing.

OPTIONAL / NICE-TO-HAVE:

- The plan's tie‑break uses "deterministic seed tie‑break" — this can be implemented once a stable seed source is established.

CUT THESE (over-engineering):

- v0 does not need the post‑composition B+ loop (v1); that can be deferred entirely.