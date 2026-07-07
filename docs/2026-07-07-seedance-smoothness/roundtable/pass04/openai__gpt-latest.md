<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes — implementation is narrowly scoped and aligned, but helper output is still under-specified enough to make hashes/tests diverge.

MUST-FIX BEFORE BUILD:
1. [Helper Contract / Smooth-Motion Clause] Defect: exact conditioned prompt bytes are ambiguous. The plan says “Append the smooth-motion clause” but does not specify separator, trailing-whitespace handling, or whether any non-regex whitespace normalization is allowed. This directly affects `conditioned_sha8`, excerpts, and byte-identical idempotence tests. Concrete fix: specify exact assembly, e.g.:
   - Do not normalize whitespace except for excerpt generation.
   - `original_sha8` is computed from the input prompt exactly as received.
   - After softener substitutions, build:
     `conditioned = softened.rstrip() + "\n\n" + SEEDANCE_SMOOTH_MOTION_CLAUSE`
   - `conditioned_sha8` is computed from that exact string.

2. [Softener Order / Tests] Defect: `softeners_applied: list[str]` is required and tests demand stable softener names, but the exact names and counting semantics are not specified. Different implementors could use regex text, replacement text, enum-like IDs, or include duplicates per occurrence. Concrete fix: define exact IDs and semantics:
   - `dynamic_dolly_push`
   - `handheld_dolly`
   - `whip_pans`
   - `white_hot`
   - `rapid_zooms`
   - `aggressively`
   - `standalone_handheld`
   Include each ID once, in softener order, iff that rule made at least one substitution.

3. [Helper Contract] Defect: unchanged/idempotent metadata path is implied but not explicit. If the stable marker exists, implementors may return only `changed=False` or may compute hashes/excerpts inconsistently. Concrete fix: state that even when unchanged, metadata must include the full schema with:
   - `changed=False`
   - `original_sha8 == conditioned_sha8`
