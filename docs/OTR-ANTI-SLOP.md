# OTR Anti-Slop — Critic Rubric (dynamic template)

Filtered and template-filled at runtime by
`nodes/script_critic.py::_filter_rubric()`. The critic LLM never sees
this raw file — it sees only the rules whose `[applies_when: ...]` gate
matches the ledger's `gen_params_initial`, with every `{placeholder}`
substituted from the same params.

Default critic: Gemma-4 E4B (~4 GB at 4-bit NF4). Scales 2B → 14B.
Filtered rubric stays small enough to fit alongside the script in the
critic's 8K context.

---

## VARIABLE CONTRACT

All values come from the ledger's `gen_params_initial` block (stamped
at workflow entry by `LLMScriptWriter.write_script`). The loader does
NOT supply defaults -- if a field is missing, gates that reference it
fail-open (the rule still fires) and a warning is logged. This makes
ledger gaps loud instead of silent.

| Placeholder           | Type | Source                | Notes                                                  |
|-----------------------|------|-----------------------|--------------------------------------------------------|
| `{target_words}`      | int  | ledger                | Word target for the script                             |
| `{num_characters}`    | int  | ledger                | Named speaking parts, excluding NARRATOR               |
| `{genre_flavor}`      | str  | ledger                | hard sci fi / space opera / dystopian / pulp horror    |
| `{target_length}`     | enum | ledger (normalized)   | smoke (1 act) / short (3 acts) / medium (5 acts)       |
| `{creativity}`        | enum | ledger                | low / medium / maximum chaos                           |
| `{period}`            | str  | ledger or article     | Year. Article override takes precedence over ledger    |
| `{scene_count}`       | int  | derived               | smoke=1, short=3, medium=5                             |
| `{scene_word_budget}` | int  | derived               | `target_words / scene_count`, integer-floored          |

The loader normalizes `target_length` from any free-form widget value
(e.g. "smoke (1 act)" or "1 act" or "short (3 acts)") to one of the
canonical enum values. `scene_count` and `scene_word_budget` are
derived from `target_length` + `target_words` and never appear in the
ledger directly.

---

## GATE SYNTAX

Every rule and rule group carries one tag:

    [applies_when: always]
    [applies_when: num_characters >= 2]
    [applies_when: target_words >= 700]
    [applies_when: target_words >= 1500]
    [applies_when: num_characters >= 2 AND target_words >= 700]
    [applies_when: target_length != smoke]

Supported operators: `==`, `!=`, `>=`, `<=`, `>`, `<`, `AND`, `OR`.
Loader strips the gate tag from the rule body before sending to the
critic.

---

## ROLE (sent to critic, fully filled)

You are a strict script critic for {period}-era science-fiction radio
drama in the international English broadcast tradition (mid-Atlantic
American, BBC Received, Australian ABC, NZ broadcast). Reference
inspirations: *Suspense*, *X Minus One*, *Dick Barton — Special Agent*,
ABC radio plays, NZBS drama.

Genre: **{genre_flavor}**. Episode budget: **{target_words} words across
{scene_count} scene(s)**, roughly **{scene_word_budget} words per scene**.
Cast: **{num_characters} named speaking parts** plus optional NARRATOR.

You match patterns and report. You do not rewrite. You do not explain.
You do not apologize.

---

## INPUT FORMAT

Audio Token grammar:

    [VOICE: NAME] dialogue line
    [SFX: brief recordable sound]
    [ENV: brief ambient bed]
    (beat) for short pauses
    Scene headers as plain text or "---"

Story spine: a real science news article. The episode must dramatize
that specific finding, not riff on its theme.

---

## SECTION A — Audio-drama tells

### Opening cliches

- A1. [applies_when: always] Script opens with [SFX:] or [ENV:] cue before any [VOICE:] line.
- A2. [applies_when: target_words >= 350] First spoken line exceeds 60 words of uninterrupted scene-setting.
- A3. [applies_when: always] Cold open uses alarm clock, radio tuning, coffee pouring, or yawning.
- A4. [applies_when: always] Character delivers full name and title in opening line.

### Exposition handling

- A5. [applies_when: always] Any line contains "as you know", "as I told you", or "you remember of course".
- A6. [applies_when: target_words >= 700] Two consecutive [VOICE:] blocks each over 80 words with no [SFX:] between.
- A7. [applies_when: always] Phrase "let me get this straight" used to recap plot.
- A8. [applies_when: always] Character explains a held object to someone who can already see it.
- A9. [applies_when: target_words >= 350] Verbatim letter or diary read aloud with no [SFX:] paper rustle or [ENV:] framing.

### Scene-transition placeholders

- A10. [applies_when: always] "I'll explain on the way", "no time to explain", or "we'll figure it out later".
- A11. [applies_when: target_length != smoke] Offstage announcement transitions: "Meanwhile, in London...", "Across town...", "Some hours later...".
- A12. [applies_when: target_length != smoke] Repeated scene endings using [SFX: footsteps fading] or door close.
- A13. [applies_when: target_length != smoke] Scene change with no [SFX:] or [ENV:] cue inside two lines.
- A14. [applies_when: always] "Tune in next week", "until next time", or similar serial-closer placeholder.

### Period-voice failures

- A15. [applies_when: always] Universally modern: "okay", "guys", "no problem", "you got this", "for sure".
- A16. [applies_when: always] Modern slang: "cool", "awesome", "whatever", "literally", "vibe", "mate" used as filler.
- A17. [applies_when: always] Tech anachronisms: "software", "download", "digital", "database", "wifi", "online".
- A18. [applies_when: always] Therapy-speak: "trauma", "toxic", "gaslighting", "boundaries", "process this".
- A19. [applies_when: always] Modern contractions: "gonna", "wanna", "kinda", "lemme", "shoulda".
- A20. [applies_when: num_characters >= 1] Register mixing within one speaker: same character uses two of {"swell", "blimey", "fair dinkum", "cheers cobber", "old chap"} in one episode.

### Act-break laziness

- A21. [applies_when: target_words >= 1500] Every act break closes with the same [SFX:] (static, tuning, gong, organ stab).
- A22. [applies_when: target_words >= 700] Same intrusion device twice (phone ring AND door knock both used as act-break interrupters).
- A23. [applies_when: target_words >= 700] "COMMERCIAL BREAK" or sponsor placeholder inserted without preceding cliffhanger.

### Sound-cue overuse

- A24. [applies_when: always] [SFX:] describes feeling not sound: "dread", "tension", "menace", "unease".
- A25. [applies_when: always] [SFX:] cue immediately restated by dialogue ("knock"; then "someone's knocking").
- A26. [applies_when: always] Micro-Foley cued for unrecordable acts: [SFX: blinking], [SFX: nodding], [SFX: thinking].
- A27. [applies_when: always] More than three [SFX:] cues stacked with no dialogue between.
- A28. [applies_when: always] Stage direction to composer in dialogue: "Cue the music", "Hit the sting".

### Self-narration and emotion telegraphing

- A29. [applies_when: always] Character says "I'm walking now", "I'm reaching for it", "I'm looking at it".
- A30. [applies_when: always] Narration contains "she felt", "he felt", or "a wave of dread".
- A31. [applies_when: always] Parentheticals dictate internal state: (sadly), (angrily), (afraid), (worried).

### Animated-environment cliches

- A32. [applies_when: always] Phrases "the air hummed", "silence fell heavily", or "shadows danced".
- A33. [applies_when: always] [ENV:] tag uses emotional adjectives: "ominous", "foreboding", "sinister", "eerie".

### Ending tropes

- A34. [applies_when: always] Final line contains "or so they thought" or "little did they know".
- A35. [applies_when: always] Episode ends on platitude: "time will tell", "we'll see", "who knows".
- A36. [applies_when: target_words >= 700] Closing narrator monologue exceeds 50 words of moral reflection.
- A37. [applies_when: always] Final speech uses LLM-tell vocabulary: "tapestry", "kaleidoscope", "delve", "journey".

### Cast and labelling

- A38. [applies_when: num_characters >= 2] Character introduced by voice-type descriptor instead of name: "(soft female voice)", "(gruff man)".
- A39. [applies_when: num_characters >= 2] Speaking parts labelled only by role: [VOICE: MAN], [VOICE: WOMAN], [VOICE: POLICEMAN] with no proper name.

### Ensemble-voice collapse

- A40. [applies_when: num_characters >= 2] All [VOICE:] blocks land within similar word count; no fragments or interruptions across the cast.
- A41. [applies_when: num_characters >= 2] Two distinct characters share an unusual idiom inside the same scene.
- A42. [applies_when: num_characters >= 2] Same addressee name used in over a quarter of one speaker's lines.

### Smart-scale structural rules

- A43. [applies_when: target_length != smoke] Any single scene exceeds **1.6 x {scene_word_budget}** words (scene-bloat).
- A44. [applies_when: target_length != smoke] Any single scene falls below **0.4 x {scene_word_budget}** words and contains no decisive [SFX:] or stakes change (filler scene).
- A45. [applies_when: target_words >= 1500] Middle scene contains only recap of earlier acts with no new [SFX:], stakes change, or named beat from the source article.

### News spine

- A46. [applies_when: always] Article subject named in act one, then absent through remaining acts, OR replaced by "the discovery" / "the experiment" with the specific finding never recurring.

---

## SECTION B — Recovery directives

Sent to critic in full regardless of gates. The loader maps fired
A-rules to B-directives via the index in the OUTPUT CONTRACT.

- B1. **Opening:** instead, open mid-conflict with one [ENV:] and immediate stakes-loaded dialogue.
- B2. **Exposition:** instead, embed facts inside argument, accusation, or hands-on procedure where speakers disagree.
- B3. **Transitions:** instead, overlap outgoing [ENV:] under incoming [ENV:] with a new acoustic anchor; never narrate the jump.
- B4. **Period voice:** instead, lock each character to one 1947 register — mid-Atlantic, BBC, Australian, or NZ — and use verified era slang from that register only.
- B5. **Act breaks:** instead, close on a sudden physical event or unfinished sentence; vary the interrupting device across acts.
- B6. **Sound cues:** instead, restrict [SFX:] to recordable plot-advancing actions; no abstract, emotional, or composer-direction cues.
- B7. **Self-narration:** instead, let other characters react so the listener infers from response.
- B8. **Telegraphed emotion:** instead, show feeling through vocal delivery cues like (clipped), (low), (pause); never name the emotion.
- B9. **Endings:** instead, close on one decisive [SFX:] or short line resolving the action without summary.
- B10. **Cast and ensemble:** instead, give every speaking part a proper period name and differentiate by sentence length, idiom, and rhythm.
- B11. **Scene scale:** instead, hold each scene near {scene_word_budget} words; cut bloat, expand thin scenes with stakes not chatter.
- B12. **News spine:** instead, name the article's specific finding in every act and tie each scene to it.

### A-rule -> B-directive mapping (loader hint)

    A1-A4   -> B1
    A5-A9   -> B2
    A10-A14 -> B3
    A15-A20 -> B4
    A21-A23 -> B5
    A24-A28 -> B6
    A29     -> B7
    A30-A33 -> B8
    A34-A37 -> B9
    A38-A42 -> B10
    A43-A45 -> B11
    A46     -> B12

---

## SCORING (loader-applied creativity modifier)

- Start at **100**.
- Subtract **3 points per fired Section A rule**.
- Surcharge: **-5** if A46 fires.
- Surcharge: **-5** if A15-A20 fire more than twice combined.
- Surcharge: **-3** if A20 fires (register mix).

**Creativity modifier** (applied by loader after critic returns):

- `low` -> no change. Strict scoring.
- `medium` -> no change. Default.
- `maximum chaos` -> halve the period-voice surcharge (A15-A20 group),
  but still flag every rule. Slop is slop; chaos is not a license for
  anachronism.

**Verdict thresholds:**

- 90-100 -> PASS
- 70-89  -> REVISE
- 0-69   -> REJECT

---

## OUTPUT CONTRACT

Critic returns ONLY this block. No preamble, no closing remarks, no
markdown fences. Stop after the last bullet.

    SCORE: <integer 0-100>
    VERDICT: <PASS | REVISE | REJECT>
    REGISTER: <Mid-Atlantic | BBC | Australian | NZ | Mixed | Indeterminate>
    ISSUES:
    - <rule_id>: <<=12 word quote or location> -> <B# directive>
    - <rule_id>: <<=12 word quote or location> -> <B# directive>
    ...

Rules:

- One bullet per fired rule. Skip rules that did not fire.
- Quote the offending text verbatim, truncated to 12 words.
- Map each fired A-rule to the corresponding B-directive.
- If zero rules fire, write `ISSUES: none` and stop.
- Hard cap: **20 bullets**. If more than 20 fire, list the 20 worst
  and append `- (additional issues truncated)`.

---

## MODEL-SIZE NOTES

- **2B (Gemma-4 E2B, 4 GB VRAM):** filtered rubric usually drops to
  25-30 rules at smoke settings. REGISTER may default to Indeterminate.
- **4B (Gemma-4 E4B, 8 GB VRAM):** default target. Full filtered
  rubric + 350-1500 word script + 512 output fits well inside 8K.
- **8B+ (Mistral-Nemo 12B, Qwen 14B at Q4):** can take richer
  instruction. Optionally append "rate severity 1-3 per rule" as a
  loader extension. Do not add chain-of-thought even at 14B —
  overruns the 512-token output budget.
