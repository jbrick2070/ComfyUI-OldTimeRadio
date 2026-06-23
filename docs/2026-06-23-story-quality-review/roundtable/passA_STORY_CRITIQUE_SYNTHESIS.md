# Roundtable A -- Story Critique Synthesis (panel's thoughts on the stories)

**Panel (live, parallel):** GPT-5.5 (gpt-5.5-20260423), Gemini-3.1-pro, DeepSeek-v4-pro, Grok-4.3.
**Spend:** ~$0.1108. Raw reviews: `passA/`. Judge anchor: `../STORY_REVIEW.md`.
**Method:** panel read the 18 frozen transcripts COLD (no access to my review); I grounded every claim against the
real ledgers + my dump and discarded the artifacts (below).

---

## 1. The headline -- unanimous convergence

All four models, independently, reached the SAME three conclusions:

1. **The root cause is the BEAT PLANNER, not the writer model and not the line composer.** Every premise
   (classroom AI, fossils, spiders, coal law, astronomy) collapses into one scene: 2-3 people in a sealed room
   fighting over a lever / key / drive / console while a gauge climbs and something counts down. The planner
   launders every news brief into the same "physical-sabotage standoff." (GPT, Gemini, DeepSeek, Grok all rank
   this #1.)
2. **A flag-and-reroll QA/critic gate WILL NOT WORK.** All four explicitly: a weak model told "this is
   incoherent / too flat, reroll" just regenerates the same standoff with synonyms. This is the strongest
   possible confirmation of the operator's instinct -- the fix must change what the model is ASKED to produce,
   upstream, not score-and-retry what it already produced.
3. **Swapping to a bigger/frontier writer alone will not fix it.** All four note the 3 "strong" gemma episodes
   still follow the identical template. The planner is the constraint, not the writer's prose ability. (This
   matches my anchor finding that local gemma-12b out-wrote frontier grok -- model choice changes sentence
   texture, not dramatic architecture.)

The implication is decisive: **move the lever UPSTREAM into the outline/beat planner, make it deterministic and
structural, and do NOT build another critic-reroll gate.**

---

## 2. Failure modes (merged + ranked, grounded)

1. **Generic console-standoff template (ROOT, unanimous).** "Pull the lever or I'm tearing your terminal keys
   out of the dock" (EP2); "I need those keys before that mahogany table becomes a funeral pyre" (EP8, a UN
   legal case); "Override failed! Taking manual control" (EP12, spider conservation). The conflict grammar is
   fixed: sealed location + timed danger + contested object + override/purge/vent + announcer reports the news
   outcome.
2. **Resolution by announcer fiat (root).** The decisive action happens OFF-stage; the announcer narrates the
   news result, often ignoring the scene. EP7 ends "...pushing a button?" -> ANNOUNCER "the once-smoldering
   remnants now bear signs"; EP8 "I'm not waiting for permission" -> "The gavel has fallen." No character pays
   the price on-stage. (GPT, DeepSeek, Grok.)
3. **Dialogue is procedural threat-noise with no subtext (root/symptom).** Characters announce dashboard states,
   issue ultimata, and explain all stakes in one breath; they never listen, lie, plead, or change. "give me the
   clearance code or I'll have to let the pulse hit both our heads" (EP14). Lines are interchangeable between
   characters -- no distinct voice despite the existing `speech_signature` field. (DeepSeek, Grok, GPT.)
4. **Action-narration spoken as dialogue + meta/prompt leak (symptom, sharp).** Whole lines that are stage
   directions: "Jettisoning module, bracing for impact." / "Initiating dark mode on mainframe." (EP16);
   "...Fingers dancing on the controls" (EP4). And a literal director instruction frozen as a spoken line:
   **EP18 "Nia's voice should maintain its warmth and calculation, not shift to a more urgent or aggressive
   tone."** TTS would speak all of this. (GPT, Gemini.)
5. **Character-identity drift / turn-skew (root).** One speaker slot carries contradictory positions and voices
   multiple people: EP1 c03 orders the seizure, stands down, AND reconciles; EP13 one slot speaks ~10 of 14
   lines as several townsfolk. The line composer is not bound to a per-beat "who speaks / what they want / what
   they control." (GPT.)
6. **Unearned escalation to lethal/irreversible force (root).** Zero-to-self-destruct in three lines, even in a
   fossil-preservation premise: "I'll bring this cave down on us all, Peter. You know I will." (EP6). No
   established space, relationship, or stake to make the threat mean anything. (Gemini.)
7. **Severe under-length as a SYMPTOM of early collapse.** Episodes stop at 154-168 words (EP4/9/16) because the
   composer "exhausts its four reusable move types and has nowhere else to go" (Grok). Length is downstream of
   the monotony, not an independent defect (see section 4).

---

## 3. The converged lever set (UPSTREAM, deterministic, weak-model-robust)

Ranked by panel convergence x leverage. All target the planner/composer, none is a flag-and-reroll critic gate.

- **L-A. Premise-specific conflict palette + a banned generic-crisis-word list in the beat planner** (MOST
  converged: Gemini #1, GPT #3, Grok #1). Inject a hard denylist into the planner prompt -- override, purge,
  lever, console, lockdown, core, vent, switch, key, drive, countdown, manual control -- and/or cap such beats
  at <=2. Map the news domain -> allowed conflict objects (classroom: lesson plan/parent board/demo; legal:
  injunction/leaked memo/testimony; astronomy: observation time/peer review/instrument). **Why it survives a
  weak model: small models obey explicit NEGATIVE token constraints even when they ignore "be subtle."** Cheapest
  high-impact lever; this is the one to prototype first.
- **L-B. A phased beat structure with carried state, replacing the freeform 14-18 beats** (GPT, Grok, DeepSeek).
  Converge on 4 phases -- Setup / Pressure / Reversal / Decision -- each beat with ONE job, naming who acts and
  whether their leverage rose or fell, and the next beat must change exactly one state. Forces a real turn
  instead of a threat loop. Why it survives a weak model: removes the need to invent escalation; each beat is a
  short slot with a visible carried state.
- **L-C. Required non-standoff beats: a personal-stake/relationship beat AND an on-stage climax beat** (DeepSeek,
  Grok, GPT). Mandate (i) at least one beat where a personal cost/relationship surfaces before any override, and
  (ii) a final beat that DRAMATIZES the decisive action with a sensory consequence -- not an announcer summary.
  Why: a required slot is enforceable on a weak model; a quality nudge is not.
- **L-D. Bind a cast ledger into the beats before line composition** (GPT, DeepSeek, Grok). OTR already has
  CastLock + a `dramatic_state` with per-character wants -- the gap is they are not BINDING each beat (who
  speaks, their want, their current leverage) and the composer drifts. Stamp a hard cast card per composer call;
  give each character a distinct, enforced register. (VERIFY in code -- section 5.)
- **L-E. Separate action from dialogue at the composer via schema** (Gemini #2, GPT #4). Have the composer emit
  `{internal_action, spoken_dialogue}` and send ONLY `spoken_dialogue` to the ledger/TTS. A deterministic "trash
  can" for stage directions kills failure mode #4 (action-narration + meta-leak) WITHOUT a reroll. This is the
  clean replacement for the operator's candidate (a) and for L7. Note byte-identity: changing composer output
  shape is audio-affecting -- gate behind the flag / re-baseline deliberately.
- **L-F. Deterministic transcript sanitizer (regex, NOT an LLM gate)** (GPT #6, Gemini). Hard-fail/repair
  prompt-leak tokens ("voice should", "tone", lowercase "announcer:" inside dialogue), unbalanced quotes. This
  is hygiene, not craft, and it is deterministic -- acceptable under the no-QA-gate rule.

---

## 4. Adjudicated disagreement -- length / beat count

The panel split, and the split resolves cleanly against the evidence:

- GPT: do NOT chase 883 words; use FEWER, tighter turns (9-12); "longer incoherence is worse" (EP11 at 430w is
  still overstuffed threat-monologue).
- DeepSeek: INCREASE to ~30 beats to hit length + allow development.
- Grok: enforce length via per-phase beat minimums (e.g., 3-4-3-4).

**Judge call:** length is a SYMPTOM of the structural collapse, not a lever. 30 freeform beats (DeepSeek) would
just be more standoff; padding to 883 words makes incoherence longer (GPT). The right move is Grok's: a PHASED
structure with per-phase minimums so length emerges from coherent development, with quality-of-beat over
quantity. **I am revising my own anchor here: I ranked raw length #1; the panel is right that structure is the
lever and length follows.** Drop "hit 883 words" as a goal; keep "each phase must develop" as the mechanism.

---

## 5. Grounding corrections (claims I DISCOUNTED as artifacts -- judge duty)

- **Mojibake ("kÄkÄpÅ", "El NiÃ±o", "Youâ€™re") -- NOT a confirmed story defect; it was MY packet-build
  artifact.** I assembled the panel packet with PowerShell `Get-Content -Raw` (default ANSI in PS 5.1) over a
  UTF-8 file, which double-mangled non-ASCII before the panel saw it. My direct python read of the real ledgers
  shows clean "kākāpō"/"El Niño". Downgrade to verify-at-build (confirm the real frozen ledger + TTS path is
  clean UTF-8); do not chase it as a craft fix.
- **"Characters are nameless c02/c03 line slots" (GPT #1) -- partly MY artifact.** My transcript labeled
  speakers by `char_id`, not the cast NAME; the real ledger HAS names (Nia, Dmitri, Ming, Charlie... appear in
  the dialogue). REJECT "nameless." But the DEEPER claim -- identity drift + skewed turn distribution + the
  composer not bound to per-beat identity/want (failure mode #5) -- is CONFIRMED in EP1/EP13 and stands.
- Everything else in section 2/3 is CONFIRMED against the ledgers.

---

## 6. What WON'T work (unanimous -- carry into the build constraints)

1. Another critic/QA gate that flags-then-rerolls. Weak model regenerates the same standoff. (All 4.)
2. Soft prompt nudges ("avoid cliche", "show don't tell", "withhold the objective"). Already tried; ignored. (All 4.)
3. Swapping writer model as the primary fix. The planner is the constraint; even "strong" gemma episodes are the
   same scene. (All 4.) [Model choice is still a free secondary win -- gemma-12b > grok here -- but not THE fix.]
4. Post-hoc subtext injection. "The underlying beats contain no material for subtext to be built from" (Grok).

---

## 7. Carry-forward CODE-VERIFY items for Roundtable B (wiring round)
1. Does the outline/beat planner (`_otr_outline._build_beat_user_prompt` / `_assemble_outline`) have any phase
   structure or banned-word capability today? Where would a denylist + domain palette inject?
2. Does `dramatic_state` carry per-character wants into EACH beat, and does CastLock bind names/leverage into
   the composer call, or is identity left implicit (failure mode #5)?
3. Composer output shape (`_otr_line_composer`): feasibility of an `{internal_action, spoken_dialogue}` split;
   byte-identity blast radius.
4. Is there an existing climax/turn beat slot, or is the outline a flat escalation list?
5. Real-ledger encoding + the prompt-leak path that let EP18's director note reach spoken `text`.
