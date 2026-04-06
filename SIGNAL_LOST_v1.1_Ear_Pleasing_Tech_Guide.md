# SIGNAL LOST – Technical Guide for Coders: Omni-Retro Sonic Architecture

The definitive blueprint for `gemma4_orchestrator.py` v1.1. This guide engineers an Omni-Retro / Multiversal Pulp aesthetic — 1950s Americana Noir mashed against Afrofuturism, Neo-Tokyo Cyberpunk, Thai street density, and Russian Dieselpunk — and forces every line into Bark-TTS-optimized spoken English.

## 1. The Omni-Cultural Naming Engine (Python)

Procedural deterministic naming via `_generate_character_profile`. The hardcoded `_FIRST_NAMES` and `_LAST_NAMES` pools draw from all five aesthetic pillars. All names are 1-2 syllables, hard consonants, easy to say aloud — Bark TTS handles them cleanly.

The 5-Pillar Pool:

- 1950s Americana Noir: Vance, Carter, Stone, Margot, Nora, Sully, Mac, Hayes, Blake, Cole, Drake, Quinn, Reese, Kane.
- Afrofuturism: Malik, Zuri, Chidi, Ayo, Oya, Kael, Tariq, Nia.
- Neo-Tokyo Cyberpunk: Ren, Akira, Kenji, Yuki, Sora, Jiro, Rei, Hiro.
- Thai Density: Krit, Mali, Niran, Sunan, Dao, Pim, Som.
- Russian Dieselpunk: Lev, Anya, Dmitri, Sergei, Volkov, Mira, Yuri.

Casts naturally emerge as collisions like `[VANCE, ZURI, AKIRA, VOLKOV]` — a lived-in multiversal world before a single line is spoken. The Director phase overrides any names Gemma proposes with these procedural names; the upper-case result becomes the immutable `character_id`.

## 2. The 5-Pillar Worldbuilding (Gemma 4 Phase)

`SCRIPT_SYSTEM_PROMPT` instructs Gemma to mix at least TWO cultural soundscapes per scene via `[ENV:]` and `[SFX:]` tags. The sonic palette:

- 1950s Americana: crackling radio static, humming neon, theremin swells, revolver clicks.
- Neo-Tokyo: high-pitch digital buzzing, mag-lev trains, synthetic rain, holographic ad jingles.
- Thai: monsoon rain on tin roofs, distant temple gongs, sizzling street woks, sputtering tuk-tuks.
- Russian Dieselpunk: brutalist echoes, heavy diesel machinery, hydraulic hisses.
- Afrofuturism: analog synth swells, polyrhythmic drum-circle static, deep bass hums.

Forbidden: `[ENV: a futuristic city street]`. Required: `[ENV: heavy Thai monsoon on tin roofs, Neo-Tokyo mag-lev train screams overhead, deep dieselpunk engine idling]`.

## 3. Rhythm, Sonic Verbs & Euphony

Three locked rules in the system prompt:

- **Staccato under tension.** 2-to-5 word sentences during action. Em-dashes (—) for interruptions. Golden-age radio pacing.
- **Sonic verbs.** Characters describe what they hear: snap, hiss, thud, crack, groan, click, roar. "The hull is groaning. Hear that snap?" — never "The ship is breaking."
- **Linguistic euphony.** Write for the ear, not the eye. If a sentence takes more than one breath, break it up. Vary lengths — punchy fragment, flowing sentence, harsh stop. The "Mind's Ear" test: read every line internally before emitting it.

## 4. Pipeline Enforcement

The pipeline enforces ear flow at three checkpoints:

- **Open-Close Evaluator** rejects outlines that imply long monologues or jargon-heavy exposition; demands premises that fit 5-15 word spoken lines and offer strong SFX moments.
- **Critique Pass** explicitly reads every line aloud; flags any line >15 words, missing contractions, written-prose feel, or hard-to-say character names.
- **Revision Pass** rewrites flagged lines while preserving canonical formatting and procedural character_ids.

The result: a canonical JSON reaching `batch_bark_generator.py` populated with culturally collided characters, soundscapes mixing two pillars per scene, and dialogue engineered to sound like an alternate-universe 1950s broadcast transmitting from the year 2150.
