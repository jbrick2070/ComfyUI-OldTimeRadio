<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The scoring loop mutates the candidates, corrupting the winner before it even reaches the downstream pipeline, and the RNG seeding is physically disconnected from the LLM.

MUST-FIX BEFORE BUILD:

1. [Section F2 / v0 Step 2] Defect: Double-mutation of `beat.intent`. Step 2 calls `build_sq_data` to score each candidate. `build_sq_data` mutates `beat.intent` in-place (appending fallback stakes/consequences). If the winner then flows into the existing F2 block as planned ("everything downstream... runs on the winner unchanged"), F2 will run `build_sq_data` a *second* time on the winner, double-appending the text to the intents.
   Fix: Deepcopy the candidate `outline` before passing it to `build_sq_data` for scoring, OR cache the mutated `outline` and `sq_by_beat` for the winner and explicitly bypass the F2 `build_sq_data` call.

2. [Section D.3 / v0 Step 1] Defect: Phantom `episode_seed` variable. The plan dictates `sha256(f"{episode_seed}:{n}")`. There is no `episode_seed` in `OTR_LedgerScriptWriter.run()`; the legacy `seed` widget was removed entirely in BUG-LOCAL-269.
   Fix: Use `cast_seed` as the deterministic base: `hashlib.sha256(f"{cast_seed}:outline:{n}".encode()).hexdigest()`.

3. [v0 Step 1 / generate_fn] Defect: The computed RNG hash is never actually wired into the LLM. `_OTRO.generate_outline` and `_build_truncating_generate_fn` do not accept a seed argument, meaning hashing a string does absolutely nothing to vary the LLM's output.
   Fix: Inside the N-loop, explicitly call `import torch; torch.manual_seed(int(hash_val, 16) % (2**64))` and `random.seed(...)` immediately before calling `generate_outline`.

4. [Local-only gate] Defect: Vague "paid/OpenRouter" check will leak Comfy Credits billed runs. The writer supports two remote billing lanes.
   Fix: Explicitly check `if resolved["creative_writing_model"].startswith(("openrouter:", "comfy:")):` to force N=1.

SHOULD-FIX:

5. [v0 Step 1] Defect: `generate_outline` can raise `OutlineFailedError` if a candidate exhausts its internal retries. If candidate 2 of 5 fails, it will crash the entire node instead of scoring the successful candidates.
   Fix: Wrap the `generate_outline` call in a `try/except OutlineFailedError` inside the N-loop. If it fails, `continue` and score the candidates that succeeded.

CUT THESE (over-engineering):

1. "each FULLY validated by `validate_outline_against_budget` + `stamp_dialogue_slot_ids`" in Step 1. Cut this requirement from the loop. `_assemble_outline` inside `generate_outline` already calls both of these before returning the outline. Doing it again in the caller is redundant.