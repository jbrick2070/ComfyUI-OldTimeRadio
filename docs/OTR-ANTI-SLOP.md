# OTR Anti-Slop — Audio Drama Tells

**Status:** placeholder — Jeffrey is having Gemini generate the full
audio-drama-tuned list. Once that lands, replace the contents below
with Gemini's Section A + Section B output. The wiring (writer
prompt include + critic gate consumption) already reads this file
verbatim, so a paste-in is the only step needed when the real list
arrives.

**Loaded by:**

- `nodes/story_orchestrator.py` — `LLMScriptWriter` system prompt
  pulls this file in at script-write time so the writer sees the
  rejection rubric up-front.
- `nodes/script_critic.py` — `LLMScriptCritic` reads the same file
  as the rejection rubric the critic scores against. Same source of
  truth; no list drift between writer and critic.

---

## Section A — Audio-drama tells (placeholder list)

The following is a starter set lifted from the AISM bullets already
in the writer system prompt plus the OTR-specific tells we've seen
on real episode runs. Replace with Gemini's expanded list when it
arrives.

- Every scene opens with a sound cue (static pop, door creak,
  thunder rumble) instead of in-medias-res dialogue.
- Two characters back-to-back monologuing exposition with no
  interruption, no question, no objection.
- "I'll explain on the way" or "There's no time to explain" used
  as a transition placeholder when the writer can't think of a
  real bridge.
- Every act break collapses into the same static-pop or
  radio-tuning sound cue.
- Animated-environment cliches: "the air itself seemed to hum",
  "the walls pulsed with menace", "the lights flickered as if alive".
- Telegraphed emotion: "she felt a cold dread", "his heart raced
  with terror" — audio drama can't see internal states; show via
  voice or action.
- Period-inappropriate vocabulary in 1940s setting ("okay", "guys",
  "you got this", "no problem").
- Characters narrating their own physical actions ("I'm walking
  toward the door now").
- Every female character introduced via voice description before
  a single line of dialogue.
- Scenes ending on "we'll see" / "time will tell" / "or so they
  thought" / "little did they know".
- Ensemble-voice collapse: every character uses the same vocabulary
  register, sentence length, hedging patterns.
- Epilogue lectures: the final scene is a character explaining the
  moral of the story instead of dramatizing the resolution.
- News-spine buried: the science article that seeded the episode is
  mentioned in the opening then never connects to the plot.
- Names that scream sci-fi ("Zara", "Kade", "Vox", "Nyx") in a
  1940s-styled drama.
- Dialogue tags converging on the same verb ("said" 30 times in
  a row, or worse "exclaimed" / "intoned" 30 times in a row).

## Section B — Recovery rubric (placeholder)

For each major category in Section A, what the critic should look
for as the GOOD alternative:

- **Opening**: instead of a sound cue, open mid-conversation with
  a stake the listener can hear in the first two lines.
- **Exposition**: instead of monologue, hand exposition through
  conflict — one character interrupting, correcting, or refusing
  the other.
- **Transitions**: instead of "I'll explain on the way", end the
  scene on a concrete decision; open the next scene already
  acting on it.
- **Act breaks**: instead of static-pop, vary the tonal cue per
  act — silence, music sting, location whoosh, voice fade.
- **Environment**: instead of "the air hummed", describe one
  specific sound a character can react to.
- **Emotion**: instead of telling the listener someone is afraid,
  give them a voice break, a stutter, a held breath, a refusal.
- **Period voice**: replace anachronisms with 1940s-appropriate
  alternatives ("certainly", "indeed", "see here", "now then").
- **Self-narration**: cut. If a character has to say "I'm walking",
  use footsteps Foley.
- **Endings**: instead of "or so they thought", end on a concrete
  unresolved object — a ringing phone, an open door, a name
  not spoken.
- **Voice differentiation**: every named character gets one verbal
  tic the others don't share.

---

## How the critic uses this list

The `LLMScriptCritic` node reads this file at run time, builds a
rejection rubric from the bullets above, and asks the critic LLM
(default `google/gemma-4-E4B-it`, separate from the writer model)
to score the script against each bullet. Output goes to
`ledger.script_gates[]` with per-tell findings.

The critic is **advisory by default** — it stamps findings into
the ledger but does NOT block downstream audio render unless
`block_on_reject=True` is set on the node. This lets us collect
critic data across runs and tune thresholds before making it
authoritative.
