<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan has converged into a highly sound, deterministic structural selector. A few minor implementation traps (division by zero, global imports, sorting math) remain that would break the build or violate existing invariants.

MUST-FIX BEFORE BUILD:
1. [v0 - Score each candidate] **Division by zero in pure scorer**. If a candidate generates empty intents or zero voiced beats, `total voiced-intent words` or `fraction of beat intents` will raise `ZeroDivisionError`.
   *Fix*: Explicitly clamp denominators: `total voiced-intent words = max(1, ...)` and `total voiced beats = max(1, ...)`.
2. [v0 - Selector] **Violation of module import invariant**. The plan calls `torch.manual_seed(...)`, but `OTR_LedgerScriptWriter.py` strictly forbids module-level torch imports (see L440: `import torch # local import; never load torch at module import`).
   *Fix*: Add `import torch` locally inside the `select_best_outline` function.
3. [v0 - Keep-best comparator] **Mixed sort directions**. Python's `min()`/`max()` cannot natively mix `asc` and `desc` on a tuple without negating the numeric values.
   *Fix*: Specify the exact sort key for `min()`: `key=lambda c: (c.score.ungrounded_crisis_density, -c.score.distinct_conflict_nouns, -c.score.premise_grounding, c.index)`.
4. [v0 - Keep-best comparator] **Fallback determinism**. "fall through to ONE normal generate_outline" does not specify the seed or hint.
   *Fix*: Specify that the never-fail fallback must use the `i=0` seed and `diversity_hint=""` so the fallback remains deterministic.

SHOULD-FIX:
1. [v0 - Local-only gate] **Missing env var read**. The plan says `OTR_STORY_BEST_OF_N: unset/0/1`, but doesn't specify how it's read.
   *Fix*: Explicitly state `import os; n_raw = os.environ.get("OTR_STORY_BEST_OF_N", "0")` inside `run()` to match the file's existing convention for env vars (e.g., `OTR_STORY_QUALITY_V2`).

OPTIONAL / NICE-TO-HAVE:
- Log the winning candidate's index and scores in the LOUD success message so operators can see the selector working without digging into the JSON ledger.

CUT THESE:
1. [v1 -- the operator's holistic "B+ until good" loop] - Unanimously deferred by the panel. Safe to cut from this implementation plan entirely to keep the PR focused solely on v0.

VERIFY-AT-BUILD checklist:
1. `generate_outline`'s RNG is actually re-seeded per call by the global torch/random seed (no generator threading bleed).
2. `count_ungrounded_crisis` on RAW intents is non-zero on a real outline (else the metric can't discriminate).
3. Flag-off: exactly one `generate_outline` call; no `meta.story_quality.best_of_n` key; prompt byte-identical.
4. `build_sq_data` runs exactly once (on the winner); `_enrich_intent` never double-appends.
5. Local gate clamps N=1 on `openrouter:`/`comfy:` BEFORE the first candidate call.