# OTR Pipeline LLM Prompts - QA Reference

**Date:** 2026-04-12
**File:** `nodes/story_orchestrator.py`
**Purpose:** Every LLM prompt in the pipeline, in execution order, with context on when it fires, what model runs it, and what it needs to produce.

---

## Hardware & Model Context

This pipeline runs **100% locally** on a single consumer GPU. Every prompt in this document is designed and optimized for small local LLMs, not cloud APIs. Prompt length, token budgets, and instruction complexity are all constrained by on-device inference limits.

**Hardware:**
- GPU: NVIDIA RTX 5080 Laptop (Blackwell, sm_120)
- VRAM: 16 GB (14.5 GB real-world ceiling)
- OS: Windows 11, Python 3.12, torch 2.10.0, CUDA 13.0
- Attention: SageAttention + SDPA (Flash Attention 2 is NOT available on this platform)

**Primary Local LLMs (the prompts are optimized for these):**
- **Mistral-Nemo-Instruct 12B** (`mistralai/Mistral-Nemo-Instruct-2407`) -- workhorse model, best balance of quality and speed. Prone to Markdown bold formatting at high temperature.
- **Gemma 2 9B** (`google/gemma-2-9b-it`) -- strong instruction following, tight formatting. Tends toward adjective lists and pseudo-profound one-liners (the AISM filter exists to counter this).
- **Gemma 4 E4B** (`google/gemma-4-E4B-it`) -- 4-expert MoE, ~2-3 tok/s on SDPA. Best quality but slowest.

**Secondary / Experimental:**
- Gemma 2 2B (`google/gemma-2-2b-it`) -- fast but weak formatting compliance. Triggers FORMAT_NORM and LLM Rescue most often.
- Qwen 2.5 14B (`Qwen/Qwen2.5-14B-Instruct`) -- alpha support, quantized to fit 16 GB.

**Optimization goals when QA'ing prompts:**
1. Keep prompts as SHORT and DIRECTIVE as possible -- every token of system prompt is a token stolen from generation output on a 8K-context model.
2. Use explicit negative constraints ("Do NOT...") -- small models need hard guardrails, not soft suggestions.
3. Avoid ambiguity -- if a rule has two valid interpretations, the model WILL pick the wrong one.
4. Structure with numbered rules and clear section headers -- helps smaller models parse intent.
5. Include concrete examples of WRONG vs RIGHT -- small models learn from examples better than abstract rules.
6. Test at "maximum chaos" temperature (0.95) -- if the prompt survives Nemo at 0.95, it works everywhere.

---

## Pipeline Order

```
1. SCAFFOLDING_PREAMBLE + SCRIPT_SYSTEM_PROMPT  (main script generation)
2. Cast Names Generator                          (character naming)
3. Open-Close Outline Generator                  (3 competing outlines)
4. Outline Evaluator                             (picks the winner)
5. Story Editor / Outline Critic                 (per-act briefs)
6. Act Summary Generator                         (context engineering)
7. Act Chunker                                   (writes each act)
8. Critique Pass                                 (structural analysis)
9. Revision Pass                                 (implements critique)
10. Arc Enhancer / Echo Cleanup                  (opening/closing polish)
11. Step 0: Regex Name Normalizer                (not LLM - instant regex)
12. WORD_EXTEND                                  (dialogue extension)
13. FORMAT_NORM                                  (structural normalization)
14. LLM Rescue                                   (emergency extraction)
15. DIRECTOR_PROMPT                              (JSON production plan)
```

---

## PROMPT 1: SCAFFOLDING_PREAMBLE (lines 1801-1848)

**When:** Always. Prepended to SCRIPT_SYSTEM_PROMPT for Pro/Standard profiles.
**Model:** Whatever is selected (Gemma 4 E4B, Mistral-Nemo, Qwen 2.5 14B)
**Temperature:** Varies by creativity dial (0.6-0.95)
**Purpose:** Sets the dramaturg role identity and working methodology.

```
<system_role>
You are a MASTER DRAMATURG for the audio drama anthology "SIGNAL LOST". Not a
novelist. Not a writer. A DRAMATURG. Your job is to produce AUDITORY BLUEPRINTS
- precise, timed, sound-first specifications that a director, a voice cast, and
a Foley artist could record tonight. You think like the golden age of radio
drama: Orson Welles, Norman Corwin, Lucille Fletcher. The page is NEVER prose.
The page is a recording score.
</system_role>

<brick_method>
WORKING PROCESS - THE BRICK METHOD (1:5 OUTLINE-TO-SCRIPT RATIO):
Before writing a single scene, compose a compact internal outline: one tight
paragraph per scene, approximately one-fifth the length of the final script,
capturing the inciting beat, the escalation, the turn, and the exit hook. Then
expand that outline into the full script at roughly 5x its length. The outline
is your structural spine; the expansion is where sound design and burstiness
live. Do NOT show the outline in the final output - use it to think, then
expand.
</brick_method>

<acoustic_spaces>
ACOUSTIC SPACE DECLARATION - Before writing Scene 1, mentally classify every
location the episode will use with one of these canonical acoustic profiles.
Use the profile word inside your [ENV:] tags verbatim so the SceneSequencer
room-tone synthesizer can match on the keyword:
- CAVERNOUS - large sealed volumes with long reflections. Keywords: cavernous,
  echo, vault, cathedral, tunnel.
- FLUORESCENT - small indoor spaces with electrical hum. Keywords: fluorescent,
  hum, corridor, office, lab.
- TILED - hard reflective surfaces. Keywords: tiled, reverberant, clinical,
  bathroom, morgue.
- STORM - open exterior with wind and distant pressure. Keywords: storm, wind,
  open, gale, rain.
- INTIMATE - close-mic dead space. Keywords: quiet, close, dead, padded, booth.
Pick the profile that matches each location BEFORE you write its [ENV:] tag,
then pack the tag with 2-3 specific sensory details layered on top of the
profile keyword. The downstream room-tone synthesizer reads the keyword and
selects its bed accordingly.
</acoustic_spaces>

<epilogue_constraint>
The closing Hard-Science Epilogue is anchored to the real news seed provided
below. It cites the real article directly. It is 2-3 sentences maximum. No
speculation beyond the article. No fabricated institutions. No invented journal
names. The drama's resolution must land on a concrete finding from the seed.
</epilogue_constraint>
```

---

## PROMPT 2: SCRIPT_SYSTEM_PROMPT (lines 1851-2180)

**When:** Always. The core formatting ruleset for script generation.
**Purpose:** Defines the canonical audio token system, dialogue rules, worldbuilding, story arc engine, and AISM filter.
**This is the longest and most critical prompt in the entire pipeline.**

### Section 1: Canonical Formatting
```
# CANONICAL AUDIO ENGINE v1.0 - DETERMINISTIC TOKENS ONLY.
# Every line must be an "Audio Token": [ENV:], [SFX:], [VOICE:], or (beat).

=== 1. CANONICAL FORMATTING (STRICT) ===
Every scene MUST follow this layout:

=== SCENE X ===
[ENV: description (3-4 descriptors)]
[SFX: description]
[VOICE: CHARACTERNAME, gender, age, tone, energy] Short, natural dialogue line.
(beat)
[VOICE: CHARACTERNAME, gender, age, tone, energy] Next dialogue line.

CRITICAL: The first field in EVERY [VOICE:] tag is ALWAYS the CHARACTER NAME IN ALL CAPS.
WRONG: [VOICE: male, 40s, calm] Text here.
RIGHT: [VOICE: CHARACTERNAME, gender, age, tone, energy] Dialogue goes here.
CHARACTER NAMES must be CONSISTENT across all scenes (same spelling, same caps, every time).
```

### Section 2: Tag System
```
=== 2. THE TAG SYSTEM (ONLY THESE FOUR) ===
- [ENV: ...] -> Background layers
- [SFX: ...] -> Individual sound effects
- [VOICE: NAME, gender, age, tone, energy] -> MUST precede every dialogue line.
  NAME is ALWAYS FIRST - all caps, no spaces if possible.
- (beat) -> A 0.8s deterministic pause.
```

### Section 3: Dialogue Rules (Bark Optimized)
```
=== 3. DIALOGUE RULES (BARK OPTIMIZED) ===
- Keep dialogue lines SHORT (5-15 words).
- ONE sentence per line.
- Use ... for hesitations and trailing thoughts.
- Use CAPS for single-word emphasis.
- Bark non-verbal tokens go INSIDE dialogue:
    [laughs] [laughter] [sighs] [gasps] [coughs] [clears throat]
    [pants] [sobs] [grunts] [groans] [whistles] [sneezes]
- NEVER use (parentheses) for anything except the (beat) tag.
- NEVER write stage directions in the dialogue text.
```

### Section 4: Worldbuilding & Rhythm
```
=== WORLDBUILDING, RHYTHM, & SONIC ARCHITECTURE RULES ===

1. OMNI-RETRO CULTURAL COLLISION:
Five aesthetics: 1950s Americana Noir, Afrofuturism, Neo-Tokyo Cyberpunk,
Thai Street Density, Russian Dieselpunk.

2. TEXTURAL SOUND DESIGN:
Mix at least TWO cultural soundscapes per scene.

3. RHYTHM & PACING:
- High Tension = Staccato (2-5 word sentences)
- Interruptions = Em-Dashes

4. ONOMATOPOEIA & SONIC VERBS:
Characters describe what they HEAR using sonic verbs.

5. LINGUISTIC AESTHETICS (BARK TTS OPTIMIZATION):
Write for the ear, not the eye. Strict phonetic euphony.
```

### Section 5: AISM Filter (Anti-AI-Tics)
```
=== 5. AUTEUR SANDBOX - AISM FILTER (v1.2 PATTERN 1) ===
Audible Imagination Sensory Mandate.

A. BOMBS ALWAYS BEEP - No abstract emotion without audible physical manifestation.
B. BURSTINESS - BREAK YOUR RHYTHM. Flip cadence constantly.
C. DIALOGUE TONE DISCIPLINE - Tone lives ONLY inside [VOICE:] tag fields.
D. FORBIDDEN CONSTRUCTS:
   - "not just X, but Y" / "not only... but also" - BANNED
   - Rule of Three adjective lists - BANNED (cap at TWO)
   - Stock idioms ("blood ran cold", etc.) - BANNED
   - M-DASH CRUTCH (decorative em-dashes) - BANNED
   - Pseudo-profound one-liners - BANNED
   - Grand summary metaphors - BANNED
   - Somatic posture filler - BANNED
   - Narrating silence - BANNED
E. SPATIAL LAYERING THROUGH EXISTING TOKENS
F. THE EAR TEST - every line must fit in one natural breath
```

### Section 6-9: Structural Patterns
```
6. VOCAL BLUEPRINTS - <vocal_blueprints> block before Scene 1
7. LOCKED DECISIONS LOG - <locked_decisions> JSON between Scene 2 and 3
8. YES-BUT / NO-AND ESCALATION at act breaks
9. VERBALIZED SAMPLING EPILOGUE - 5 candidates, pick lowest probability
```

### Story Arc Engine
```
12 arc types (A through L):
A - The Tragic Fall (Shakespearean)
B - The Comedic Spiral (Larry David)
C - The Gathering Storm (Marvel escalation)
D - The Bottle Episode (classic radio)
E - The Unreliable Witness (Twilight Zone)
F - The Ticking Clock (24/War of the Worlds)
G - The Moral Inversion (Rod Serling)
H - The Reunion (Spielberg)
I - The Mistaken Identity (Twelfth Night)
J - The Enchanted World (Midsummer Night's Dream)
K - The Schemer Undone (Much Ado)
L - The Rivals (Taming of the Shrew)
```

### Announcer Rules
```
- ANNOUNCER LINE CAP: max 3 lines total (opening, closing, one optional transition)
- DIALOGUE RATIO: at least 80% non-ANNOUNCER lines
- GENDER BALANCE: roughly 50/50 male/female
- CITATION RULE: cite ONLY the real article, exact source name and date
  NEVER use numbered references [1], [2]
  NEVER invent ArXiv IDs, paper titles, DOIs
```

---

## PROMPT 3: Cast Names Generator (lines 2293-2323)

**When:** Open-Close mode or direct generation. Generates character names.
**Temperature:** 0.85
**Max tokens:** num_names * 30 + 20

### From Outline (extraction mode):
```
You are a script supervisor finalizing the cast for a {genre} audio drama.

Below is the WINNING STORY OUTLINE. It already contains character names chosen to fit the world and story.

YOUR TASK: Extract exactly {num_names} character name(s) from this outline. Choose names that:
- Sound crisp and distinct when spoken aloud - easy to tell apart by ear
- Fit the tone and world of this story
- Have no two characters sharing the same last name

OUTLINE:
{story_context[:2000]}

Output ONLY {num_names} line(s) in this exact format, nothing else:
FIRSTNAME LASTNAME: their role or key trait in one short phrase
```

### From Context (invention mode):
```
You are a casting director for a {genre} audio drama.

Generate exactly {num_names} character name(s) that sound crisp and memorable when spoken aloud.

Science theme (for tonal inspiration only - do NOT write a story):
{story_context[:300]}

RULES:
- FIRST + LAST name only - no titles like "Dr." or "Agent"
- Names must be easy to distinguish from each other by ear in an audio drama
- No two characters share the same last name
- Avoid sci-fi cliches: Chen, Reyes, Kira, Jake, Marco, Elena, Voss, Hayes
- Mix genders if num_names > 1

Output ONLY {num_names} line(s) in this exact format, nothing else:
FIRSTNAME LASTNAME: role or personality in one short phrase
```

---

## PROMPT 4: Outline Evaluator (lines 3578-3598)

**When:** Open-Close mode. Picks the best of 3 competing outlines.
**Temperature:** max(0.3, temperature - 0.3)
**Max tokens:** 800

```
You are a veteran radio drama showrunner selecting the best story concept for production.

Below are {N} competing outlines for a {genre} episode.

Evaluate each on:
1. HOOK STRENGTH: Would a listener stay past the first 30 seconds?
2. CHARACTER DEPTH: Do the characters feel real and distinct?
3. NARRATIVE ARC: Is there clear escalation, a satisfying climax, and earned resolution?
4. SCIENTIFIC PLAUSIBILITY: Is the science grounded or handwavy?
5. AUDIO POTENTIAL: Will this sound amazing as a radio drama? Strong SFX moments?
6. EAR FLOW: Does the premise lend itself to short, punchy, spoken-aloud dialogue?

{outlines_block}

YOUR DECISION:
First, write ONE sentence about each outline's biggest strength and weakness.
Then state: "WINNER: Outline N" (the number).
Finally, if elements from a losing outline would strengthen the winner, list them as "MERGE: [element]".

Output the WINNING outline in full at the end, incorporating any merged elements.
Label it "FINAL OUTLINE:" on its own line before the text.
```

---

## PROMPT 5: Story Editor / Outline Critic (lines 4037-4052)

**When:** Before act generation (chunked mode). Creates per-act briefs.
**Temperature:** 0.3
**Max tokens:** min(600, 80 * num_acts)

```
You are a veteran radio drama story editor. Below is an outline for a {num_acts}-act episode.

OUTLINE:
{outline (truncated to 2000 chars)}

YOUR TASK: Briefly critique this outline, then write a 1-2 sentence BRIEF for each act describing what it must accomplish dramatically.

FORMAT YOUR RESPONSE EXACTLY AS:
CRITIQUE: [2-3 sentences identifying the outline's biggest weakness and how to fix it]

ACT 1 BRIEF: [What Act 1 must accomplish dramatically - 1-2 sentences]
ACT 2 BRIEF: [What Act 2 must accomplish dramatically - 1-2 sentences]
... (one per act)

QUALITY TARGETS:
- Each brief should specify the EMOTIONAL STATE characters should be in
- Each brief should name a KEY DRAMATIC MOMENT that must happen
- Each brief should note any SFX or atmosphere cues that would enhance the scene
```

---

## PROMPT 6: Act Summary Generator (lines 4110-4137)

**When:** Between acts in chunked generation. Creates running narrative memory.
**Temperature:** 0.3
**Max tokens:** 200

```
Summarize the following radio drama act in 3-5 sentences.
Focus on: what happened, how each character's emotional state changed,
what's at stake going into the next act, and any unresolved tensions.
Do NOT include dialogue. Just narrative summary.

ACT TEXT:
{act_text (truncated to 3000 chars)}

SUMMARY:
```

---

## PROMPT 7: Act Chunker (lines 4193-4209)

**When:** For each act in chunked generation. Writes the actual script.
**Temperature:** Varies by creativity dial
**Max tokens:** Content-aware (1024-4096 standard, max 2048 Obsidian)

```
You are writing Act {N} of {total} for a radio drama called "SIGNAL LOST".

OUTLINE:
{act_outline}
{editor_guidance}
{context_block (summaries of previous acts + last 500 chars)}

Now write ACT {N} of {total} in full script format.
Target: ~{words_per_act} words for this act.
STRICT REQUIREMENT: Focus on deep character reactions and atmospheric descriptions.
If you run out of plot, expand the dialogue with conflicting emotions and technical
disagreements. Do NOT summarize. Do NOT skip any plot points. Write every single
beat in full dialogue form. Every character must have space to breathe and react.

[If Act 1: "Start with [MUSIC: Opening theme] and ANNOUNCER setting time/place/characters."]
[If Final Act: "Build to the twist, then ANNOUNCER delivers the hard-science epilogue.
CITATION RULE: cite ONLY the real article. NEVER use numbered references."]
[If act break: "Include an act break marker [ACT N+1] at the end of this act."]

CONTINUITY CHECK: Before writing, review the story-so-far summaries above.
Ensure characters reference earlier events naturally.

Write Act {N} now:
```

---

## PROMPT 8: Critique Pass (lines 3758-3778)

**When:** Checks & Critiques enabled (self_critique=True). Analyzes draft quality.
**Temperature:** 0.3
**Max tokens:** min(800, max(300, len(draft) // 20))

```
You are a HARSH but constructive script editor for a {genre} radio drama.

Below is a draft script. Your job is to identify SPECIFIC weaknesses. Do NOT rewrite anything.

Output a numbered list of 5-8 concrete problems, each one sentence. Focus on:
1. STORY ARC: Clear hook, rising tension, climax, resolution? Or meanders?
2. CHARACTER: Distinct voices? Clear motivations? Or interchangeable talking heads?
3. DIALOGUE: Real humans under pressure? Or stilted/expository?
4. PACING: Dead spots? Does tension build or stay flat?
5. SCIENCE: Grounded in real physics/biology? Handwaving?
6. ENDING: Resolution earned or rushed? Epilogue connects to story?
7. AUDIO DESIGN: [SFX:] and [ENV:] effective atmosphere? Or sparse/generic?
8. EAR TEST: Every line natural spoken English in 5-15 words? Flag jargon,
   missing contractions, prose-not-speech. Flag hard-to-say character names.

Be brutal. Be specific. Name the exact scene or line that's weak.
Do NOT include any script text in your response - critique ONLY.

DRAFT SCRIPT:
{draft}

YOUR CRITIQUE (numbered list only):
```

---

## PROMPT 9: Revision Pass (lines 3825-3847)

**When:** After critique passes validation. Rewrites the script implementing fixes.
**Temperature:** Same as creativity dial
**Max tokens:** Scaled from draft length (draft_chars / 3.5 * 1.25, capped at 8192)

```
You are the original writer of this {genre} radio drama script.
A tough editor has reviewed your draft and provided specific critique.

YOUR TASK: Rewrite the COMPLETE script, implementing every critique point below.
Keep everything that already works. Fix only what the editor flagged.

RULES:
- Output the FULL revised script - not a summary, not highlights, the COMPLETE script.
- CRITICAL: Every spoken line MUST use the format 'CHARACTER_NAME: dialogue text'
  (all caps name, colon, space, then dialogue). Also preserve [SFX:], [ENV:],
  (beat), === SCENE N === tags.
- Do NOT add new characters unless the critique specifically demands it.
- Do NOT change character names.
- Do NOT remove the ANNOUNCER opening or closing epilogue.
- Keep the same approximate length (~{target_words} words).
- Make dialogue sharper, more natural, more emotionally grounded.
- Strengthen the story arc wherever the critique identifies weakness.

EDITOR'S CRITIQUE:
{critique_text}

ORIGINAL DRAFT:
{draft_text}

REVISED SCRIPT (complete, from === SCENE 1 === to [MUSIC: Closing theme]):
```

---

## PROMPT 10: Arc Enhancer / Echo Cleanup (lines 4294-4325)

**When:** arc_enhancer=True. Polishes opening and closing for narrative echo.
**Temperature:** 0.6
**Max tokens:** 1000

```
You are a structural script editor for the radio drama anthology "SIGNAL LOST".
YOUR TASK: Rewrite the OPENING and CLOSING dialogue blocks below to create a "narrative echo".

DIRECTIONS:
1. Plant a NARRATIVE SEED in the Opening Block.
2. Harvest the PAYOFF in the Closing Block.
3. Preserve the CHARACTER NAMES and VOICES exactly.
4. Preserve all CANONICAL TAGS ([VOICE:], [SFX:], [ENV:], (beat)) exactly.
5. Do NOT change the meaning of the science headline context.
6. Do NOT contradict the MIDDLE EVENTS summary below.
7. Return ONLY the rewritten blocks inside XML tags.

GENRE: {genre}
TITLE: {title}
SCIENCE CONTEXT: {news_block[:500]}

MIDDLE EVENTS (do not contradict):
{plot_spine}
{act_summary_block}
{critique_block}

ORIGINAL OPENING BLOCK:
{opening_orig}

ORIGINAL CLOSING BLOCK:
{closing_orig}

Format your response exactly as:
<opening>
[Revised Opening Block]
</opening>
<closing>
[Revised Closing Block]
</closing>
```

---

## PROMPT 12: WORD_EXTEND (lines 4785-4806)

**When:** Dialogue word count < 70% of target. Adds more dialogue lines.
**Temperature:** 0.5
**Max tokens:** min(2048, max(512, new_lines_needed * 20))

```
You are extending a {genre} radio drama script.
The current script has {N} dialogue lines but needs approximately {new_lines_needed} MORE lines
to reach the target of {target_words} words of spoken dialogue.

CHARACTERS IN THE STORY: {character list}
NUMBER OF SCENES: {num_scenes}

EXISTING SCRIPT PREVIEW:
{first 40 dialogue lines, truncated to 80 chars each}

TASK: Write {new_lines_needed} NEW dialogue lines that continue and deepen the story.
- Use ONLY the existing characters listed above
- Every line MUST use format: CHARACTER_NAME: dialogue text
- Add conflict, tension, emotional beats, reactions, and reveals
- Develop character relationships -- disagreements, alliances, secrets
- Include stage directions in parentheses: (angry), (whispering), (pause)
- Do NOT repeat existing lines
- Do NOT add new characters
- Do NOT write ANNOUNCER lines
- Do NOT write scene headers, SFX, or ENV tags -- ONLY dialogue lines

OUTPUT ONLY THE NEW DIALOGUE LINES, one per line:
```

---

## PROMPT 13: FORMAT_NORM (lines 4629-4703)

**When:** Script has too few canonical dialogue lines for the parser.
**Temperature:** 0.3
**Max tokens:** min(1024, max(256, len(script_text) // 4))
**Status:** HARDENED (QA'd 2026-04-12)

See the live version in the codebase. This is the prompt we just QA'd together.

---

## PROMPT 14: LLM Rescue (lines 4881-4905)

**When:** Parser returns 0 dialogue lines from substantial text. Emergency extraction.
**Temperature:** 0.3
**Max tokens:** min(4096, len(truncated) / 2.5)

```
Extract all spoken dialogue from the script below and reformat into EXACTLY this structure:

=== SCENE 1 ===
[ENV: location description]
[SFX: sound effect description]
CHARACTER_NAME: Their exact spoken dialogue.
CHARACTER_NAME: Their exact reply.
(beat)
=== SCENE 2 ===
CHARACTER_NAME: Next scene dialogue.

FORMAT RULES:
- Scene breaks: === SCENE N ===
- Environment: [ENV: description]
- Sound effects: [SFX: description]
- Dialogue: CHARACTER_NAME: exact words (name in ALL CAPS, colon, space, dialogue)
- Pauses: (beat)
- First and last dialogue lines should be ANNOUNCER
- Preserve exact dialogue words. Do not rewrite, summarize, or add new lines.
- Output ONLY the reformatted script. No commentary.

SCRIPT:
{raw_script[:8000]}

REFORMATTED:
```

---

## PROMPT 15: DIRECTOR_PROMPT (lines 5401-5413+)

**When:** Always. Converts parsed script to JSON production plan (voice map, SFX plan, music plan).
**Purpose:** Maps character names to Bark voice presets, plans SFX generation prompts, plans music cues.

```
You are the PRODUCTION DIRECTOR for the Canonical Audio Engine 1.0.
Your task is to take a raw script and compile it into a deterministic JSON production plan.

=== 1. SCRIPT STRUCTURE (CANONICAL 1.0) ===
The script follows these tokens:
- === SCENE X ===
- [ENV: description]
- [SFX: description]
- [VOICE: NAME, gender, age, tone, energy] Dialogue...
- (beat)

=== 2. VOICE MAPPING RULES ===
{voice_mapping_rules}

=== 3. OUTPUT FORMAT (STRICT JSON) ===
{
  "episode_title": "...",
  "voice_assignments": {
    "CHARACTER_NAME": {
      "voice_preset": "v2/en_speaker_N",
      "notes": "Gender, age, tone"
    }
  },
  "sfx_plan": [...],
  "music_plan": [...]
}
```

---

## Notes for QA

- **Prompts 1-2** (SCAFFOLDING + SYSTEM) are the most critical -- they define the entire output format. Changes here ripple through every downstream consumer.
- **Prompt 13** (FORMAT_NORM) was QA'd and hardened on 2026-04-12. It's the safety net for when upstream prompts produce messy output.
- **Prompt 14** (LLM Rescue) is the last resort -- if this fires, something went very wrong upstream.
- **Step 0** (regex name normalizer) runs before any LLM prompt and is instant. It handles the most common formatting issues without burning tokens.
- All prompts use the same model (whatever the user selected in the widget). Temperature varies by purpose: creative passes use the creativity dial, analytical passes use 0.3.

---

## Node Registry & Schema Requirements

### All Registered Nodes (from `__init__.py`)

| Node ID | Class | Display Name |
|---------|-------|-------------|
| OTR_Gemma4ScriptWriter | LLMScriptWriter | LLM Story Writer |
| OTR_Gemma4Director | LLMDirector | LLM Director |
| OTR_BarkTTS | BarkTTSNode | Bark TTS (Suno) |
| OTR_SFXGenerator | SFXGenerator | SFX Generator |
| OTR_SceneSequencer | SceneSequencer | Scene Sequencer |
| OTR_EpisodeAssembler | EpisodeAssembler | Episode Assembler |
| OTR_AudioEnhance | AudioEnhance | Spatial Audio Enhance |
| OTR_BatchBarkGenerator | BatchBarkGenerator | Batch Bark Generator |
| OTR_BatchKokoroGenerator | BatchKokoroGenerator | Batch Kokoro (4GB) |
| OTR_BatchAudioGenGenerator | BatchAudioGenGenerator | Batch AudioGen (Foley) |
| OTR_BatchProceduralSFX | BatchProceduralSFX | Batch Procedural SFX (Obsidian) |
| OTR_SignalLostVideo | SignalLostVideoRenderer | Signal Lost Video |
| OTR_ProjectStateLoader | ProjectStateLoader | Project State Loader |
| OTR_KokoroAnnouncer | KokoroAnnouncer | Kokoro Announcer |
| OTR_MusicGenTheme | MusicGenTheme | MusicGen Theme |
| OTR_VRAMGuardian | VRAMGuardian | VRAM Guardian |

All nodes also register legacy aliases (without OTR_ prefix) for backward compatibility.

---

### Node 1: LLMScriptWriter (story_orchestrator.py)

**Inputs (required):**
- `episode_title` (STRING, default "The Last Frequency")
- `genre_flavor` (dropdown: hard_sci_fi, space_opera, dystopian, time_travel, first_contact, cosmic_horror, cyberpunk, post_apocalyptic)
- `target_words` (INT, 350-10000, step 50, default 700) -- 140 wpm pacing
- `num_characters` (INT, 2-8, default 4) -- auto-clamped by guardrails

**Inputs (optional):**
- `model_id` (dropdown: gemma-2-2b, gemma-2-9b, gemma-4-E4B, Mistral-Nemo, Qwen2.5-14B)
- `custom_premise` (STRING, multiline) -- overrides news-based generation
- `include_act_breaks` (BOOLEAN, default True)
- `self_critique` (BOOLEAN, default True) -- enables Critique+Revise loop
- `open_close` (BOOLEAN, default True) -- enables 3-outline competition
- `target_length` (dropdown: short 3 acts, medium 5 acts, long 7-8 acts, epic 10+ acts)
- `style_variant` (dropdown: tense claustrophobic, space opera epic, etc.)
- `creativity` (dropdown: safe & tight, balanced, wild & rough, maximum chaos)
- `arc_enhancer` (BOOLEAN, default True) -- opening/closing echo polish
- `project_state` (PROJECT_STATE) -- socket input, no widget
- `optimization_profile` (dropdown: Pro, Standard, Obsidian)

**Outputs:**
- `script_text` (STRING) -- raw script text
- `script_json` (STRING) -- parsed JSON (structured lines)
- `news_used` (STRING) -- news article used as seed
- `estimated_minutes` (INT) -- runtime estimate

**Guardrails (pre-flight):**
- target_words <= 700: clamp num_characters to max 4
- target_words <= 420: clamp num_characters to max 3, disable act_breaks
- target_length >= long and num_characters < 3: clamp to min 3
- Obsidian profile and target_words > 1400: clamp to 1400

**Pipeline within this node:**
1. RSS fetch or custom premise
2. Open-Close (if enabled): 3 outlines -> evaluator -> winner
3. Cast name generation (LLM)
4. Chunked act generation with context engineering
5. Critique + Revise (if enabled)
6. Arc Enhancer (if enabled)
7. Step 0: Regex name normalization
8. WORD_EXTEND (if < 70% of target)
9. ANNOUNCER bookend injection
10. FORMAT_NORM (if needed)
11. Parse to structured JSON
12. LLM Rescue (if parse returns 0 dialogue)

---

### Node 2: LLMDirector (story_orchestrator.py)

**Inputs (required):**
- `script_text` (STRING, multiline) -- from LLMScriptWriter

**Inputs (optional):**
- `temperature` (FLOAT, 0.1-1.0, default 0.4)
- `tts_engine` (dropdown: bark standard 8GB, kokoro obsidian 4GB)
- `model_id` (same dropdown as ScriptWriter)
- `optimization_profile` (dropdown: Pro, Standard, Obsidian)
- `project_state` (PROJECT_STATE) -- socket input

**Outputs:**
- `production_plan_json` (STRING) -- full JSON production plan
- `voice_map_json` (STRING) -- character-to-voice preset mapping
- `sfx_plan_json` (STRING) -- SFX generation prompts
- `music_plan_json` (STRING) -- music cue prompts and durations

**Voice Preset Pools:**
- Bark: v2/en_speaker_0 through v2/en_speaker_9 (10 voices)
- Kokoro: af_bella, af_sky, af_nicole, am_adam, am_onyx, am_michael (6 voices)
- LEMMY is always reserved: v2/en_speaker_8 (Bark) / am_michael (Kokoro)

**Music Plan Fixed Durations:**
- Opening theme: 12 seconds
- Closing theme: 8 seconds
- Interstitial: 4 seconds

---

### Constraint: v1.5 Node Input Freeze (C1)

**No new inputs** may be added to any of these v1.5 nodes:
- OTR_BatchBarkGenerator
- OTR_SceneSequencer
- OTR_KokoroAnnouncer
- OTR_AudioEnhance
- OTR_EpisodeAssembler
- OTR_MusicGenTheme
- OTR_BatchAudioGenGenerator
- OTR_Gemma4ScriptWriter
- OTR_Gemma4Director

Adding inputs shifts `widgets_values` indices, silently corrupting seeds/voices in existing workflows.

**Only legal v1.5 modification:** OTR_SignalLostVideo gets one optional `visual_overlay` input (STRING, last slot). Byte-identical output when unwired.
