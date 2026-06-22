<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan contains a circular dependency in the data model, undefined state-passing for episode reruns, and relies on regex for a parsing task that will catastrophically fail on the provided corpus.

MUST-FIX BEFORE BUILD:

1. **[4. DEFECT 3] Circular dependency in `init_lines_from_outline` coercion.**
   * *Defect*: The plan instructs to "force `speaker_role='character'` if `char_id` resolves to a known cast id" inside `init_lines_from_outline`. Grounding D explicitly states that in this function, `char_id` is derived *from* `role` (lines 761-766). You cannot coerce `role` based on `char_id` before `char_id` is derived.
   * *Fix*: Remove the coercion from `init_lines_from_outline`. Restrict the coercion strictly to the `role_mismatch` repair guard in `_otr_ledger_reviewer.py` (rejecting `expected="announcer"` if the existing row has a cast `char_id`) and `set_lines` (where external updates are applied).

2. **[3. DEFECT 2] State loss during `needs_full_rerun` escalation.**
   * *Defect*: The plan proposes injecting a coherence hint into the outline/DramaticState for a full rerun. However, `needs_full_rerun` clears the ledger and restarts the episode. Because the workflow JSON is FROZEN (Invariant 1), you cannot wire a new hint port into the outline node. The hint will be destroyed on reset.
   * *Fix*: Implement a state-preservation mechanism in `_otr_freeze_cascade.py`. Before emitting the rerun signal, extract the hint and write it to a reserved key in the ledger's `meta` (e.g., `meta["coherence_hints"]`). The cascade must inject this specific `meta` key into the *new* blank ledger it initializes for the rerun. Modify `_otr_outline.py` to read `meta["coherence_hints"]` and append it to the `_build_beat_user_prompt` system instructions.

3. **[2. DEFECT 1] Catastrophic regex failure on unbalanced quotes (b015).**
   * *Defect*: The plan requires stripping action outside a "matched quote pair" but explicitly excludes b015 (malformed quoting). b015 has three quote marks (`Well... expected." tightens... "I do hope...`). Standard regex or span detectors cannot reliably identify "outside matched quotes" when quotes are unbalanced; they will match across lines or swallow legitimate dialogue.
   * *Fix*: Define the Tier 3 detection primitive with a hard structural abort: `if text.count('"') % 2 != 0: return text`. Only attempt to split by `"` and evaluate the non-quote segments if the quote count is even. For the segments outside quotes, apply `_NARRATION_VERBS` from `_otr_line_hygiene.py` to classify as action.

SHOULD-FIX:

1. **[2. DEFECT 1] Orphan punctuation after deterministic strip.**
   * *Defect*: The plan requires the stripped line to remain well-formed, but stripping a trailing action like `"<spoken>." <action>` will leave trailing spaces or commas if the action was preceded by them.
   * *Fix*: After the Tier 3 strip removes the action span, run `.strip(" ,;-")` on the resulting string and assert that the final character is in `_TERMINAL_PUNCT` (from `_otr_line_hygiene.py`). If it is not, abort the strip and return the original text.

2. **[7. Open Questions] Strong-model NO-OP assert definition.**
   * *Defect*: The plan asks for the assert that zero strips/rerolls fire, but doesn't define how to measure it across the pipeline without altering return signatures.
   * *Fix*: Assert against the ledger's `meta` field at the end of the suite: `assert "coherence_hints" not in ledger.meta` and `assert "stage_direction_stripped" not in [row.meta.get("CODE_STAGE_DIRECTION") for row in ledger.lines]`.

CUT THESE:

1. **[3. DEFECT 2] "Outline re-intent" vs "episode escalation" debate.**
   * *Why it is safe to cut*: The critic runs on the generated *lines* (`_otr_story_critic.py`). You cannot "re-intent" the outline without regenerating the lines anyway. Building a partial-rewind mechanism violates the "ONE gate path" invariant. Cut "outline re-intent" entirely and exclusively use the existing `needs_full_rerun` episode escalation.