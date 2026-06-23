<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. Core logic is sound, but Pydantic serialization and cascade sequencing will break the build.

MUST-FIX BEFORE BUILD:
1. [L5a / _otr_freeze_cascade.py] **Critic sequencing impossible.** The plan assumes the critic internally aborts when `too_many_edits` is set. Grounding (`_otr_freeze_cascade.py:598`) proves `too_many_edits` is a terminal failure that completely halts the cascade. The ledger is rolled back, and `run_story_critic` (line 755) is never called. *Fix*: Do not try to force the critic to run on a failed, rolled-back ledger. Fix the downstream consumer to handle a missing `meta.story_critic_report` gracefully when `reviewer_verdict == "too_many_edits"`.
2. [Data Model / _otr_outline.py] **Pydantic serialization breaks frozen schema.** The plan claims adding `conflict_object: str = ""` to `Beat` means "serialization unchanged." This is false. Pydantic's `model_dump()` includes default values unless explicitly excluded. This will leak into the JSON outline and break the frozen ledger schema validator. *Fix*: Define the new fields in `Beat` as `Field(default="", exclude=True)` so they exist in memory for the composer but vanish during serialization.
3. [L2 / _otr_line_composer.py] **Missing structured cost field.** The plan asks to "VERIFY a structured character cost/fear field exists". Grounding confirms it does NOT. `LineRequest` only receives pre-rendered strings (`character_voice_card`, `all_voice_cards`). *Fix*: The deterministic fallback table keyed by `(speaker, domain)` is mandatory, not a fallback. Wire it directly.

SHOULD-FIX:
1. [L5a / _otr_ledger_scrub.py] **Telemetry overwrite.** Grounding (`_otr_ledger_scrub.py:1003`) shows `_meta["story_quality"] = {...}` blindly overwrites the dictionary. If any other pass (or a previous scrub) wrote to `meta.story_quality`, it is destroyed. *Fix*: Use `_meta.setdefault("story_quality", {}).update({...})`.

OPTIONAL / NICE-TO-HAVE:
- [L1a] When filtering "ANNOUNCER" from `allowed_roster` to build `allowed_people`, ensure you also filter the exact string `"NARRATOR"` (cited in `Beat.speaker` grounding as the standard SFX/music speaker).

CUT THESE (over-engineering):
1. [L1 crisis-noun repair] "Cap = max(1, floor(total_voiced_beats * 0.2))". Safe to cut. If a noun is ungrounded and deterministically repairable via the palette, repair all instances. A partial repair just leaves residual hallucinations in the prompt.