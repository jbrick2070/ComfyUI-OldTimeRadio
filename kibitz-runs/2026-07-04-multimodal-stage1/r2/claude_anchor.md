R2 ANCHOR REVIEW (Claude, code-grounded) -- STAGE1_SUBPLAN.md v2 (coding/implementability)

VERDICT: yes-with-fixes. The v2 loader API (load_pack / get_pack_prompt[_or_none]),
hand-rolled stdlib validator, and granular seams are implementable + testable.
Residual coding risks below.

MUST-FIX:
1. [section 3, byte-identity mechanism] Prefer RUNTIME-IMPORT comparison over AST
   extraction. The lab used AST because its mirror was dependency-incomplete; we
   are IN-REPO and the suite already imports node modules (conftest sets
   OTR_TEST_MODE). So the byte-identity test should `import nodes._otr_outline` etc.
   and assert `module._SYSTEM_PROMPT == pack["outline_system"]` on the REAL runtime
   value. This is robust to implicit string concatenation and `+`-joins (which a
   naive `ast.Constant` extractor would miss -- e.g. STYLE_TAIL_DEFAULT is two
   adjacent literals). FOLD: runtime-import equality, not AST, for the pins.

2. [section 4, composite/interpolated seams] Any seam whose runtime string carries
   a RUNTIME VARIABLE is not a static constant and must NOT be authored (treat like
   line_grounding -> stays Python). MUST verify `announcer_outro_resolved_tail`
   (`_otr_line_composer.py:3517`) is a static literal vs an f-string with vars
   BEFORE authoring it. If interpolated, drop that seam from Stage 1.

3. [section 5, workflow no-diff GATE] Do NOT pin a sha256 constant in a pytest (it
   goes stale the moment a later stage legitimately edits the JSON). Implement the
   gate as a chunk verification step: `git diff --quiet -- workflows/otr_scifi_16gb_full.json`
   during the green-chunk check, not as a suite assertion.

SHOULD-FIX:
1. [section 3, object_pairs_hook] Confirm the dup-key hook fires on NESTED objects
   (prompt_stages) too -- it does (object_pairs_hook runs per-object during parse);
   note it so the implementer relies on it, not a top-level-only check.
2. [section 4] Grep-confirm exact lines for `_ANNOUNCER_OUTRO_SYSTEM` and that
   `outline_system` `_SYSTEM_PROMPT` at :532 is the one the router imports (vs the
   3 secondary re-defs at 1102/1115/1130) before authoring.

OPTIONAL: cache the loaded pack behind the fixtures state-digest (lab pattern) once
a hot consumer exists (Stage 1b) -- not needed while dormant.
