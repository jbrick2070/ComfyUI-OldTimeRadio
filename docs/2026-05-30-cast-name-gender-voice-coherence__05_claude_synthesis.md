# Cast NAME ↔ GENDER ↔ VOICE coherence — problem statement, round-robin, recommendation
**2026-05-30 · branch v2.0-alpha · synthesis of ChatGPT (gpt-5.5) + Gemini (gemini-3.1) + NVIDIA (llama-3.3-nemotron)**

## Problem
Generated casts are incoherent: a male-coded name lands on a female slot with a
female voice, and vice-versa (real run: `MALIK HIBBERT`→female, `PHYLLIS OKAFOR`→male).

## Root cause (from live code, `nodes/_otr_casting.py`)
- NAME is rolled from a **gender-blind** flat pool (`config/cast_pools.py FIRST_NAMES`).
- GENDER is planned 40/40/20, shuffled, and bound to slots **positionally** (`:601-608`) —
  it never looks at the name.
- VOICE correctly follows gender (`python_assign_voice_preset`).
- **So the only thing actually wrong is the NAME.** Gender quota and voice mapping are fine.

**Resolved fact (the key uncertainty both Gemini & NVIDIA flagged):**
`python_assign_voice_preset` consumes **exactly one `rng.choice` per slot**, fixed regardless
of gender (gender/timbre filtering uses no RNG; only the final tie-break draws — `:785`).
→ A fix that doesn't change gender or RNG order is byte-identical for already-coherent seeds.

## Round-robin outcome
| | Recommends | Key point |
|---|---|---|
| **ChatGPT** | Approach A: tag names by gender, **re-draw name after gender is known** | 100% coherence, simplest; but **reorders RNG** |
| **Gemini** | Approach D: **post-roll alignment**, keep RNG order, swap gender↔slot | A breaks C7 byte-identity for *all* historical seeds; D is a no-op when already coherent; flags 14.5 GB VRAM ceiling |
| **NVIDIA** | Approach D (or A with RNG state-capture) | Concurs with Gemini's RNG critique; quantifies LLM-naming OOM risk |

**All three reject LLM-driven naming (your creative idea) as the *default* fix** — extra
LLM calls on the 16 GB card risk OOM and reproducibility drift. They park it as an optional
"creative mode" for later, not the first fix.

**Consensus:** tag the name pools by gender; do NOT involve the LLM; preserve the 40/40/20
quota and the voice mapping.

**Disagreement:** ChatGPT redraws names (reorders RNG → breaks every historical seed);
Gemini/NVIDIA swap gender→slot (preserves byte-identity for coherent seeds, but swapping
gender also changes the voice for fixed slots, and can't reach 100% coherence when the rolled
name-gender mix ≠ the quota mix).

## Recommended synthesis — "name repair" (best of all three)
Since **only the NAME is wrong**, fix the name, not the gender:

1. **Tag the pool** (all three agree): add `FIRST_NAMES_BY_GENDER = {male, female, unisex}`
   in `config/cast_pools.py`; keep flat `FIRST_NAMES` as a compat alias.
2. **Leave the existing flow byte-for-byte untouched** — name roll, gender shuffle, and voice
   assignment all keep their exact current RNG order (Gemini's correctness win).
3. **After** the slot has its gender, if the rolled first-name's gender tag ≠ the slot gender,
   **replace only the first name** with one drawn from the matching-gender pool, using an
   **isolated RNG** (`random.Random(f"{cast_seed}:{char_id}")`) so the main `cast_rng` sequence
   is never perturbed. Gender, voice, and quota are untouched.

**Why this wins over both A and D:**
- Coherent seeds → no name mismatches → **zero changes → byte-identical audio** (C7 holds).
- Incoherent seeds → only the bad *name string* changes; gender + voice + downstream RNG stay put.
- **100% name coherence** (unlike D, which is quota-limited), with **no RNG reorder** (unlike A),
  **no gender/voice churn**, **zero LLM/VRAM cost**.
- Plugs in as ~15 lines after the cast loop in `_otr_casting.py`; voice code unchanged.

**Non-stereotype knob (your "may want intentional mismatches" requirement):**
`OTR_NAME_CROSS_GENDER_RATE` (default `0.0` = strict repair). At >0, a deterministic isolated-RNG
roll *allows* a mismatched name to stand that fraction of the time — never touches `cast_rng`.
Plus `OTR_OTHER_NAME_POLICY=unisex|all` for "other" slots. No new ComfyUI widget (env-only).

## Your LLM idea (gender→voice→matching name, multi-round) — verdict
Creatively strong, and easy to bolt on **later** as an opt-in `OTR_NAME_MODE=llm` that calls the
existing local writer for names only, with the tagged-pool repair as the deterministic fallback.
But per all three models it's the wrong *first* move: OOM risk + nondeterminism on the 16 GB card.
Recommendation: ship the deterministic name-repair now; add LLM naming as an optional mode if you
want richer/period-flavored names after the baseline is coherent.

## Minimal implementation + tests
- `config/cast_pools.py`: `FIRST_NAMES_BY_GENDER` + `gender_of_first_name(name)` lookup.
- `nodes/_otr_casting.py` (after the gender-bound loop): isolated-RNG name repair on mismatches.
- Tests: (1) determinism — same `OTR_CAST_SEED` twice → identical cast; (2) **coherent-seed
  byte-identity** — a seed that was already coherent produces the exact same cast as before the
  change; (3) coherence — every binary-gender slot's name tag matches its gender at rate 0.0;
  (4) quota 40/40/20 unchanged; (5) voice-uniqueness invariant still passes; (6) audio
  byte-identical regression on a coherent seed.

## Open questions for Jeffrey
1. **Backward compat:** the recommended path preserves byte-identity for already-coherent
   historical seeds. Confirm that's wanted (it is the safest); if you don't care about historical
   seeds at all, ChatGPT's simpler "redraw after gender" Approach A is also fine.
2. **Cross-gender rate default:** keep `0.0` (strict) or set a small default (e.g. `0.05`) so the
   cast occasionally has intentional non-stereotypical names?
3. Want the optional **LLM naming mode** scoped as a follow-up, or is deterministic enough?

*Raw transcripts: `docs/2026-05-30-cast-name-gender-voice-coherence__{01_chatgpt,02_gemini,03_nvidia}.md`*
