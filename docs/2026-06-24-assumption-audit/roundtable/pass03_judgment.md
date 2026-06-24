# R3 judgment

Spend this pass: ~$0.6067 (campaign running ~$1.80). Panel: GPT-5.5, Gemini,
DeepSeek. R3 = wiring/sequencing; no new ASSUMPTIONS, only integration precision.

ACCEPTED + grounded:
- Gate placement: after the COMMON `cleaned` (4163 exchange / 4206 compose),
  before `last_lines.append` (4222) -- covers the use_exchange bypass. CONFIRMED.
- Contract sequencing: `script_brief` set at `build_news_briefs` (~2658); build
  the contract AFTER that, before D.5; input script_brief|news_seed. CONFIRMED.
- F2 still re-selects from outline.premise -> replace with the pre-built contract.
  CONFIRMED.
- K5 collapse is UNSAFE now: `resolved["style"]` feeds build_news_briefs
  (2666/2796/2924) + cast + meta.style/visual_plan.style. ADD `meta.story_contract`,
  DEFER the collapse. CONFIRMED -> downgraded K5 to "add, not collapse".
- grounded palette must include `outline.premise` (R3) -- accepted.
- Build 4 fallback path (`_resolved_outro_fallback`, 4263 area) also needs
  ending_tag, else a failed LLM close still states the outcome -- accepted.
- Build 3: CUT consequence enrichment -- `assign_beat_roles` never assigns
  CONSEQUENCE under the climax-last validator (unreachable). Accepted.

CORRECTIONS: K7 dead only in the line composer (outline `_phase_summary` 1233
uses ARC_PHASE_GUIDANCE). Folded.

INVARIANTS GUARDED: gate deterministic + in-loop; all new fields defaulted empty
(byte-identical when contract absent); K5 ADD-not-overwrite preserves every
downstream style consumer; audio byte-identity unaffected.

CONVERGENCE: the substance is stable across R1->R3 (no new kills in R3, only
wiring). R4 is the residual-defect confirmation pass.
