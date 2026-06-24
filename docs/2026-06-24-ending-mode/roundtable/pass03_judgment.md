# R3 judgment

Panel: Gemini-3.1-pro returned; GPT-5.5 + DeepSeek-v4-pro ERRORED (empty content,
finish_reason=length -- reasoning-token exhaustion at max_tokens=2000, the known
issue). Spend $0.084. R4 raises --max-tokens.

ACCEPTED (Gemini + Claude anchor, grounded CONFIRMED):
- SEQUENCING circularity (Gemini): inject at the beat-PROMPT inside
  generate_outline is circular -- beat_roles + crisis-noun grounding run LATER in
  the writer (build_sq_data), after generate_outline returns. RESOLUTION: resolve
  style+ending in the WRITER post-outline (premise = outline.premise), inject at
  the LINE COMPOSER's final-character-beat request (role known, dialogue written
  there). OutlineRequest needs NO new fields.
- Keep meta.style (early, back-compat surfaces) distinct; optional gated override
  on flag-ON (Claude anchor 3).
- env-only gate => no workflow JSON change (Claude anchor 4).

VERIFY-AT-BUILD: line-composer request shape; the announcer-flag reach into
_assemble_outline; the core bet (weak model honors the concrete ending) -> the
§J A/B.

Convergence: design is build-ready; R4 = final residual-defect check.
