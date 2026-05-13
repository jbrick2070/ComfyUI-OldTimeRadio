# Round-robin design review: cast-casting LLM prompt for OTR

## Context

I'm building a "cast contract" architecture for an open-source AI radio
drama generator (https://github.com/jbrick2070/ComfyUI-OldTimeRadio).
The whole pipeline runs **locally on a single RTX 5080 16GB laptop**.

**The pipeline must be LLM-agnostic.** Supported models:
  - Mistral-Nemo 12B (current default)
  - Gemma family
  - Qwen family
  - **Future ROADMAP item:** a fine-tuned "1903 old-timey period model"
    — likely smaller, narrower, and more prompt-sensitive than the
    general-purpose 12B family.

The user (Jeffrey) wants swap-the-model freedom and is explicit that
he'll be running the same prompts against this whole pool. The period
model is the gating constraint — design for the weakest expected
model in the pool, not the strongest.

The cast-contract architecture inverts our current order:

  - **Today** (broken): one big LLM call produces outline + cast names +
    beat structure together. The LLM tends to repeat names across runs
    and sometimes drops cast members entirely, so a 3-character episode
    renders as audio with only the announcer talking. We need to fix this.
  - **Target**: ledger created first → cast fully locked
    (`name`, `character_description`, `gender`, `voice_preset` all
    populated) → THEN the outline / story LLM call runs against the
    already-locked cast and cannot mutate it.

This review is specifically about **the new "casting LLM call"** that
fills `character_description`, `gender`, and `voice_preset` for the
characters whose names Python has already chosen.

## Decisions already locked (please don't relitigate these)

  - **Names** are picked by Python from a curated pool (~110 first
    names, ~50 last names, organized by aesthetic pillars). Names
    are NEVER picked by the LLM — that's the anti-repetition fix.
  - **ANNOUNCER** is always present, sits at `char_id=c01`, gender
    rolled 50/50 male/female, voice picked from a 4-preset announcer
    pool. **Pre-baked by Python before the LLM call. LLM never sees
    the announcer row.**
  - **LEMMY** (a recurring easter-egg engineer character) rolls in at
    11% per episode via `SystemRandom`. When he hits, his row is
    fully pre-baked: name=LEMMY, gender=male, voice=v2/en_speaker_8,
    fixed character_description. **LLM never sees the LEMMY row either.**
  - **Open characters** (the 1-5 pool-filled rows) are what the LLM
    actually has to write. Names are pre-set; the LLM fills
    description + gender + voice_preset.
  - **Voice pool** is pre-filtered before the LLM call: any voice
    already taken (by LEMMY or ANNOUNCER) is removed from the
    "available voices" list the LLM sees, so collision-by-prompt is
    impossible. The LLM still has to avoid duplicates within its own
    response (validator + reroll handles that).
  - **Gender distribution target**: 40% male / 40% female / 20% other
    written into the prompt as a **soft suggestion only**, no Python
    validator. We accept that across many episodes the realized split
    will drift; the upside is the LLM looks at each name (LEMMY -> male
    obvious, ALICE -> female obvious, JESSE -> ambiguous LLM call) and
    picks gender naturally.
  - **Validator + reroll** mirrors our existing outline LLM caller:
    pydantic schema validates the JSON response, on failure we reroll
    up to 3 attempts (1: fresh at temp 0.7, 2: fresh at temp 0.8,
    3: "repair" prompt that hands the validation error back, temp 0.3).
    This reroll loop is the **primary model-agnostic safety net** —
    we lean on it rather than on prompt cleverness.
  - **Per-model shims** (if any model needs special handling like a
    forced system role, `<json>` tags, etc.) live in a config wrapper
    at the model loader layer, NOT in the prompt itself. The prompt
    text stays one canonical version across all supported models.

## Past failure mode worth knowing about

When we previously fed long instruction-heavy prompts to the local
story LLM (any of them, but Mistral-Nemo is what we tested), two
things happened:

  1. The LLM fell back to the same boring story across runs, reaching
     for the most common dramatic tropes and the most common character
     names regardless of input variety.
  2. Cast members got dropped or merged. The LLM stopped respecting
     individual character constraints when there was a lot of
     surrounding instruction to keep track of.

**Brevity target for any local LLM call: <250 tokens of prompt total.**
If a prompt exceeds 400 tokens that's a signal to cut, not justify.

## Proposed casting prompt (the actual question)

Single user message, no system prompt, schema embedded:

```
Cast a radio drama. Story: <news_seed truncated to 500 chars>
Style: <user-selected style, e.g. "noir mystery">

Names: ALICE, BOB, NORA

Voices: v2/en_speaker_0 (male warm 40s), v2/en_speaker_1
(male gruff 50s), v2/en_speaker_4 (female bright 30s),
v2/en_speaker_6 (female throaty 40s), v2/en_speaker_9
(male neutral 30s)

Aim ~40% male, ~40% female, ~20% other.

JSON only:
{"cast":[{"name":"ALICE","character_description":"<short>",
"gender":"male|female|other","voice_preset":"<id>"}, ...]}
```

The brevity moves I made:

  - **Killed the system prompt entirely.** Role-frame ("Cast a radio
    drama") sits at the start of the user message instead.
  - **Killed "you are a casting director" framing.** Verb carries it.
  - **Killed "voices already taken: ..." block.** Python pre-filters,
    LLM never sees taken voices.
  - **Killed gender hints next to names.** Whole point of the design
    is the LLM looks at the names and decides.
  - **Killed "do not invent new characters / do not rename" prose.**
    Schema enforces it (validator catches name drift, reroll handles).
  - **One-line per voice entry.** No bullets, no full sentences.
  - **No bounds on character_description length.** Schema can enforce;
    prose doesn't need to repeat.

## What I want you to evaluate

Please give me your honest assessment on each of these. **Frame your
answers in terms of "instruction-tuned local 7B-14B class models in
general"** — not specific to Mistral-Nemo or any one family. If a
recommendation only helps one family and could regress others, flag
that explicitly. Push back where you think I'm wrong; the goal is the
best prompt, not validation of mine.

1. **Will instruction-tuned local 7B-14B class models reliably emit
   valid JSON from this prompt?** Or is one or more model family
   likely to add prose preamble ("Sure, here's the cast:"), wrap in
   markdown fences, or break the schema in some predictable way?
   If yes: which families are most at risk, and what's the minimum
   prompt tweak that fixes it without regressing on the other
   families?

2. **Is killing the system prompt entirely the right call across
   model families?** Some models (Mistral) tolerate user-only well;
   others (Gemma) historically expect a system role. Should I be
   relying on the model loader's chat template to inject a default
   system role, or should the prompt itself include a one-liner that
   works everywhere? Trade-off worth the ~10 token cost?

3. **Should I include a one-shot example** in the prompt (one example
   cast member showing the expected output shape)? Would that improve
   reliability on the WEAKER end of the model pool (smaller, more
   period-specific) enough to justify the ~80-100 tokens?

4. **The gender 40/40/20 hint** — is "Aim ~40% male, ~40% female,
   ~20% other" the right phrasing across model families? Other
   framings I considered: "Try for roughly...", "Distribute genders...",
   "Pick gender per name; aim for...". Or omit and rely entirely on
   the LLM's name-reading judgment?

5. **The voice pool format** — `v2/en_speaker_0 (male warm 40s)` —
   am I burying the preset ID in parentheses next to the description?
   Should the format be `v2/en_speaker_0: male, warm, 40s` or
   `v2/en_speaker_0 - male warm 40s` or something else? Optimal across
   instruction-tuned 7B-14B parsing this and picking one?

6. **Per-character vs single call** — I'm doing one call for ALL open
   characters in a single response. Reasoning: lets the LLM see all
   names together to differentiate them creatively (no two female 40s
   scientists), and validates collisions in one pass. Counter-argument:
   per-character calls give more focused attention and avoid the LLM
   "running out of ideas" on character N when N is large (max 5 open
   characters). Worth splitting? Does the answer change for smaller
   models?

7. **Validation strategy** — pydantic catches: (a) wrong name returned
   (LLM invented or renamed), (b) voice not in available pool, (c)
   duplicate voice within response, (d) gender not in
   {male|female|other}, (e) JSON parse failure. Reroll: 3 attempts,
   last is repair-prompt with validation error appended at temp 0.3.
   Same pattern as our outline caller. Anything material missing for
   model-agnostic robustness?

8. **Pitfalls specific to local 7B-14B class models** that I'm
   probably underweighting? Cross-family things — temperature
   sensitivity, attention drift on the voice list when it's >5
   entries, JSON closing-brace counting, chat-template gotchas,
   anything you've seen reported in real deployments.

9. **The character_description field is unbounded in the schema right
   now** (just a string). Should I bound it (e.g. 20-150 chars)?
   Smaller / weaker models often produce paragraph-length blurbs when
   given an unbounded string field — bounding it reduces audio prompt
   bloat downstream. Worth bounding?

10. **The "1903 old-timey" fine-tuned model is a roadmap item.** It'll
    likely be smaller and narrower than the 12B-class models we use
    today. Is there anything about the prompt design above that
    specifically locks in 12B-class assumptions and would break on a
    smaller / period-narrow model? What should I change now to
    future-proof?

11. **Anything I haven't asked about that you'd flag.** This is the
    most important question.

## What you have access to

Just this document. The actual codebase isn't necessary for the prompt
review — assume the validator + reroll pattern works (it already does
for our outline call). Focus on the prompt design itself, on
cross-model-family reliability concerns, and on anything specific to
the planned period-trained model that would force a redesign.

Thank you.
