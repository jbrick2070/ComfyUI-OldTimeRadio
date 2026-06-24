# R2 judgment

Spend this pass: ~$0.5939 (campaign running ~$1.19). Panel: GPT-5.5, Gemini-3.1-
pro, DeepSeek-v4-pro. VERDICTS: GPT no (under-specified fixes), Gemini yes-with-
fixes, DeepSeek (model-gate emphasis). Convergence on the build shape: HIGH.

ACCEPTED + grounded:
- K1 gate must run IN-LOOP after compose_line, before last_lines append (GPT) --
  CONFIRMED the late-scrub would pollute downstream prompts. LineRequest lacks the
  grounded palette -> add `grounded_nouns` (Gemini) -- CONFIRMED (LineRequest has
  allowed_people/things only). Object match by head-noun, reroll hint from
  offending tokens only (GPT) -- accepted (prevents banning premise-legit nouns).
- K2 pre-outline select needs script_brief/news_seed, not outline.premise (GPT +
  Gemini) -- CONFIRMED (F2 uses outline.premise today).
- K3 `_enrich_intent` treats non-irreversible as personal_stake -> needs a
  role-keyed map, not a widened condition (GPT) -- CONFIRMED; + 200-char truncate
  order bug.
- Announcer: F3 "State this outcome plainly" (composer 2819, is_resolved_ending_
  change 2785) conflicts with the grammar -> add `ending_tag` param, force
  resolved=False for unresolved/revelation/quiet (GPT + Gemini) -- CONFIRMED.

CORRECTED (my overstatement): K7 ARC_PHASE_GUIDANCE is NOT globally dead --
`_otr_outline._phase_summary` (1233) uses it in the OUTLINE beat prompts; it is
dead only in the LINE COMPOSER (position shadow). _position_for fills position
with "phase, beat N of M" minus the directive (Gemini, grounded).

DEFERRED (panel-unanimous CUT for this build): K4 climax-position (breaks
validator + ending_template target + outro final-line assumption), model-
capability gate (K1 net ships first), K11 render profiles, K8 _PERSONAL_COST
(harmless fallback). 

INVARIANTS GUARDED: in-loop gate is deterministic (count+token reroll, seed-
stable); object match by head-noun avoids false-fails; all new fields defaulted
empty (byte-identical when contract absent); audio byte-identity unaffected
(text-path; golden re-baseline already flagged).
