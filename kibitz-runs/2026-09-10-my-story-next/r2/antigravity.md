VERDICT: build-ready as-is? no.
Undefined response-local receipt interface in OpenRouter backend, ungrounded helper classification in Sprint B, downstream act-count mismatch breaking interstitial frame generation in Sprint A, and missing finalizer/reseal protocol wiring for My Story content authorship.

MUST-FIX BEFORE BUILD:
1. [Sprint B -- Original creative-model credit] / [Operator contract]
   DEFECT: The plan specifies that OpenRouter requires a fresh response-local receipt carried through retries and model-gone fallback, and that the scheduler journals executed model identities per call. However, `OpenRouterBackend.generate` (`nodes/_otr_openrouter_backend.py:1189-1408`) and `make_openrouter_generate_fn` (`nodes/_otr_openrouter_backend.py:1648-1707`) return only `str`. Every prompt pass across OTR expects `generate_fn(...) -> str`. The plan defines no interface or signature for how `OpenRouterBackend.generate` communicates the actual server-resolved slug (which may change during `OpenRouterModelGoneError` fallback at line 1389) back to `SlotScheduler` without breaking the `str` return contract.
   FIX: Define an explicit receipt-passing channel. Add a response-local receipt carrier to `OpenRouterBackend.generate` (e.g., accepting an optional `receipt_out: dict | None = None` parameter or attaching a lightweight receipt attribute `_otr_receipt` onto the returned `str`), and update `_SlotScheduler._make_generate_fn` (`nodes/OTR_LedgerScriptWriter.py:715-743`) to extract `executed_identity` and `reported` directly from that response-local receipt immediately upon return.

2. [Sprint A -- item 2] & [Sprint A -- item 6]
   DEFECT: Downstream act-count mismatch breaks interstitial cue generation.
   In `nodes/_otr_my_story.py:1118`, interstitial requirement is calculated as:
   `inter_wanted = (act_count - 1) if (include_act_breaks and act_count > 1) else 0`
   This reads `act_count` from `resolved["act_count"]` (the requested act count, e.g. 1 in live prompt `e11e5d86-2dda-41fa-b942-bb927a754b83`). Under the proposed policy, a 2-act treatment is accepted. Because `act_count` is 1, `inter_wanted` evaluates to 0, and `_pass_frame` instructs the model to generate 0 interstitial cues. Then in `_assemble` (`nodes/_otr_my_story.py:897`), `_assemble` iterates over all accepted acts (`acts = 2`) and expects an interstitial after act 1, but finds `len(frame.music_inter) == 0`. It silently drops the interstitial cue between acts.
   FIX: In `nodes/_otr_my_story.py:1118`, compute `effective_acts = len(treatment.acts)` and derive `inter_wanted = (effective_acts - 1) if (include_act_breaks and effective_acts > 1) else 0`. Pass `inter_wanted` derived from the actual accepted acts to `_pass_frame`. In `_assemble` (`nodes/_otr_my_story.py:897-906`), if `include_act_breaks` is True and `inter_seq >= len(frame.music_inter)`, emit a placeholder cue row with an empty prompt so `StableAudioTheme` can compose it from the shared brief downstream.

3. [Sprint A -- item 6] & [Sprint A -- item 1]
   DEFECT: Deterministic attribution insertion will be blocked by `_make_frame_validator`.
   Item 6 requires: "Add the existing deterministic attribution sentence before sealing when absent".
   However, `_pass_frame` (`nodes/_otr_my_story.py:727`) binds `post_validator=_make_frame_validator(attribution, inter_wanted)`.
   In `_make_frame_validator` (`nodes/_otr_my_story.py:695-699`):
   ```python
   spoken = " ".join(list(model.announcer_intro) + list(model.announcer_outro))
   if _norm_ws(attribution) not in _norm_ws(spoken):
       return "the attribution sentence is missing. Include it VERBATIM..."
   ```
   If the LLM omits or paraphrases the attribution sentence, `_make_frame_validator` returns an error, triggering up to 3 repair retries and ultimately raising `StructuredCallExhaustedError`. The run aborts before Python ever reaches the code that would deterministically add the attribution sentence.
   FIX: Remove the verbatim attribution presence check from `_make_frame_validator`. In `run_my_story_episode` (immediately after `_pass_frame`), check if `_norm_ws(attribution)` is present in `frame.announcer_intro` or `frame.announcer_outro`; if absent, append `attribution` to `frame.announcer_outro` deterministically in memory before `frame.model_dump()` is sealed in `accepted_artifacts` and passed to `_assemble`.

4. [Sprint B -- Original creative-model credit]
   DEFECT: Ambiguous helper partitioning and undefined finishing-model schema.
   The plan mandates: "Separate concept/drafting/framing helpers from ledger_clean and cast repair. The journal proves calls returned, not accepted authorship; use the exact printed wording 'Story generation models used: <identities>' for Original, with a separate finishing-model record."
   `OTR_LedgerScriptWriter` runs 12+ helpers (`nodes/OTR_LedgerScriptWriter.py:1684, 4065, 4314, 4589, 5177`, `nodes/_otr_writer_tail.py:807, 1130, 1244, 1388, 1396`). The plan does not define:
   (a) How helpers like `generate_title`, `story_brief_reflection`, and `produced_story_summary` are classified (are they framing or finishing?).
   (b) The schema, field name, and target container for the "separate finishing-model record" (e.g. `meta["finishing_models"]`).
   (c) Deduplication, delimiter, and formatting rules for `<identities>` in `credits_source_line`.
   FIX: Define two explicit, frozen sets of helper names:
   `GENERATION_HELPERS = {"build_news_briefs", "build_original_briefs", "lock_cast", "generate_outline", "dramatic_state", "build_continuity_ledger", "compose_line", "generate_title", "compose_news_coda", "compose_announcer_outro", "story_brief_reflection", "produced_story_summary"}`
   `FINISHING_HELPERS = {"ledger_clean", "ledger_cleanup", "cast_coverage_repair"}`
   Define the exact metadata schema: `meta["finishing_models"] = [{"helper": h, "model_id": m, "reported": bool}, ...]`. Format `<identities>` as unique executed model IDs (or display labels) from successful `GENERATION_HELPERS` calls, joined by `", "` in first-executed order.

5. [Sprint C -- credits hero title containment]
   DEFECT: Unhandled token overflow in hero title wrapping and missing vertical advancement math.
   In `nodes/otr_credits_roll.py:941-948`, when a hero title reaches `_PT_HERO_MIN`, `d.text((x, y), hero, fill=_rgba(_TEAL), font=fh)` draws the full title as a single line, causing the text to cross into Column 2.
   Existing `_wrap` (`nodes/otr_credits_roll.py:697-708`) splits by whitespace and does not wrap or split an overlong token wider than `max_w` (`cur = wd`).
   Furthermore, drawing wrapped lines requires advancing `y` line-by-line (`y += len(lines) * _fh(fh)`), which must also be reflected in the scratch measurement pass in `_draw_col1` (`nodes/otr_credits_roll.py:846-850`).
   FIX: Implement a dedicated title-wrapping function in `nodes/otr_credits_roll.py` that measures words against `col1_w = int(_COL1_W * sx)`. For any single token whose rendered width exceeds `col1_w`, bisect the token character-by-character into fitting sub-tokens without losing glyphs. In `_flow_col1`, iterate through the wrapped lines, drawing each line at `(x, y + i * _fh(fh))`, and advance `y += len(lines) * _fh(fh)` before subtitle and metadata strips are rendered.

SHOULD-FIX:
1. [Sprint A -- item 1] & [Sprint A -- item 2]
   DEFECT: Undefined schema for normalized act numbers and fidelity telemetry in `meta.my_story`.
   Item 2 dictates: "Ordered slot numbers may be normalized before acceptance with original numbers recorded... Preserve requested/planned/actual counts separately in meta.my_story."
   Currently, `meta["my_story"]` (`nodes/_otr_my_story.py:1026-1035`) does not declare dedicated keys for requested vs planned vs actual act counts, nor does `ActPlan` (`nodes/_otr_my_story.py:245-250`) declare a field to store original un-normalized slot numbers.
   FIX: Add explicit fields to `meta["my_story"]`:
   `meta["my_story"]["counts"] = {"requested_acts": act_count, "planned_acts": len(treatment.acts), "actual_acts": len(acts), "requested_characters": raw_requested, "planned_characters": interp.cast_plan.planned, "actual_characters": len(treatment.cast)}`.
   Add `original_n: int | None = None` to `ActPlan`. When `act.n` is normalized to `1..N`, record the original integer in `plan.original_n` and persist `original_act_numbers` in `meta["my_story"]`.

2. [Sprint A -- item 5]
   DEFECT: `_read_config_context` ignores nested `text_config` and `resolve_context_cap` hardcodes Qwen override to 8192.
   In `nodes/_otr_model_catalog.py:689-707`, `_read_advertised_context` only inspects top-level config keys (`max_position_embeddings`, `n_positions`, `n_ctx`). For Qwen 2.5/3.5 and vision-language models, context configuration is nested under `data.get("text_config")`. Additionally, `CURATED_CONTEXT_OVERRIDES` (`nodes/_otr_model_catalog.py:1828`) forces `"Qwen/Qwen3.5-4B": 8192`.
   However, if `resolve_context_cap` returns Qwen's native 262,144 context, `check_vram_fit` (`nodes/_otr_model_catalog.py:2077`) or `_assert_policy_admits_vram` (`nodes/_otr_model_loader.py:1836`) will estimate an enormous KV cache and raise `VRAMFitFailedError` ("pre-load VRAM veto").
   FIX: In `_read_advertised_context`, check `data.get("text_config", {})` when top-level keys are absent. In `check_vram_fit`, ensure KV estimation prices the requested `load_config.n_ctx` or clamped runtime budget rather than the raw 262,144 native context.

3. [Sprint A -- item 5]
   DEFECT: Transformers cache reuse key in `_otr_model_loader.py` lacks effective context cap.
   In `nodes/_otr_model_loader.py:2110`, the cache hit check is:
   `_hit = _try_cache_hit_locked(normalized, slot, policy_key=_policy.cache_key())`
   `_policy.cache_key()` contains hardware and quantization parameters, but does not capture `ctx_verdict.value`. If an episode loads a model at 8192 context and a subsequent run requests 16384 (or vice versa), the second run reuses the cached model entry with the old context cap.
   FIX: Incorporate `ctx_verdict.value` into `_try_cache_hit_locked` or `policy.cache_key()` so any change in effective context capacity invalidates resident reuse and triggers reload.

4. [Sprint A -- item 1]
   DEFECT: Treatment validator rejects on listener character gender discrepancy.
   In `nodes/_otr_my_story.py:551-554`:
   ```python
   want = stated.get(member.name.strip())
   if want and member.gender != want:
       return "%s is %s in their notes but %r here; honour what they stated" % (member.name, want, member.gender)
   ```
   Item 1 specifies: "record unresolved fidelity differences honestly without claiming their wishes were all satisfied."
   Rejecting the treatment forces an unnecessary 3-attempt repair cycle when the model alters a character's gender.
   FIX: Remove `member.gender != want` check from `_make_treatment_validator`. In `run_my_story_episode`, record any discrepancies between listener notes and accepted treatment in `meta["my_story"]["fidelity_notes"]`.

5. [Sprint A -- item 1]
   DEFECT: `MyStoryInputTooLongError` misattributes blame to `_longest_field`.
   In `nodes/_otr_my_story.py:430-437`, `_call` wraps capacity errors and blames `_longest_field(bundle)`:
   `"Shorten it -- '%s' is the longest field -- or pick a model with a larger context."`
   If the prompt overflowed due to system prompts or fixed schemas rather than user input, naming the longest field is inaccurate and unhelpful.
   FIX: Remove `_longest_field` from `nodes/_otr_my_story.py`. Format `MyStoryInputTooLongError` to report the actual token shortfall from the underlying `GenerationContextOverflowError` (`prompt_tokens`, `context_cap`, `min_output_tokens`).

OPTIONAL / NICE-TO-HAVE:
1. [Sprint A -- item 6]: Disclose unplaced interstitial cues. When `len(frame.music_inter) > len(acts) - 1`, record the surplus cue proposals in `meta["my_story"]["unplaced_music_cues"] = frame.music_inter[len(acts)-1:]` so telemetry accounts for all model output.
2. [Sprint C -- credits hero title containment]: Add an automated test with an unbroken 100-character alphanumeric token to verify that `_wrap_hero_title` bisects without dropping characters and respects `col1_w`.

CUT THESE (over-engineering):
1. [Sprint A -- item 5] / [A2 shared capacity honesty] -- Comprehensive rewrite of GGUF/transformers VRAM estimators.
   *Why safe to cut*: The live blocker was an act count validator refusal (`_make_treatment_validator: len(acts) != act_count`), not a VRAM OOM or token overflow. Local Qwen-3.5-4B already functions at 8192 context under `CURATED_CONTEXT_OVERRIDES`. A general overhaul of `_otr_model_catalog` and `_otr_model_loader` VRAM estimation is outside the scope of proving the My Story ledger.
2. [Sprint B -- Original creative-model credit] -- Custom display-name mapping dictionary for model IDs.
   *Why safe to cut*: Using the concrete model ID or existing catalog display label directly avoids maintaining a synthetic model name mapping table.

[ASSUMPTION]: In Sprint B, the "separate finishing-model record" is assumed to be stored in `meta["finishing_models"]`, as `credits_source_line` on the video card only accommodates a single summary line.
[ASSUMPTION]: In Sprint A item 6, deterministic attribution addition is assumed to append to `model.announcer_outro` so spoken script lines are preserved without displacing intro dialogue.
