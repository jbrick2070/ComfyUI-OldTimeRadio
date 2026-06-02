# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-06-01 -- Visual/Audio quality go-forward (post first-full-episode)

## Core goal
The remote-LLM lanes are shipped and PROVEN: the first full episode of the day rendered end-to-end
("Seven Worlds Our Size", claude-opus-4.8 via OpenRouter -> 88.7 MB OBS .mp4, 0 crashes). After
watching it, Jeffrey produced a prioritized **visual + audio quality go-forward** (this doc's
Immediate next steps). The job for the next session: execute that list -- aspect/prompt/caption/
audio polish on specific workflow nodes -- everything is a "do," ordered by leverage. No new lane
work; this is output-quality polish on the working pipeline.

## Tech stack & constraints
Windows, RTX 5080 (14.5 GB VRAM ceiling), venv `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`,
branch `v2.0-alpha`, ComfyUI Desktop at localhost:8000 (Desktop GUI run = full video branch, ~2.5 h/ep;
the headless `_otr_soak_driver.py` renders audio-only and SKIPS HuMo for fast iteration). All CLAUDE.md
rules apply and auto-load -- esp. **PD3: every node change (aspect, prompt, duration, caption) MUST be
wired into the canonical workflow JSON** `workflows/otr_scifi_16gb_full.json` (29 nodes), and **PD1:
audio byte-identical** (caption/visual edits are audio-neutral; audio-graph edits need the byte-identity
gate). Git via Desktop Commander cmd + `.git\COMMIT_EDITMSG` then `git commit -F`; one push attempt;
verify local==origin. Run full `tests/` after every code change. Don't duplicate BUG_LOG.md / ROADMAP.md /
the problem-statement doc here -- they auto-load.

## What's done & decided (this session)
- **Shipped BUG-LOCAL-297 -> 303 on `v2.0-alpha`** (all committed + pushed, full `tests/` green = 3400):
  297 Comfy endpoint path (`/proxy/openrouter/api/v1/...`); 298 headless MusicGen converter widget-drift;
  299 Comfy lane routing (`comfy:slot-a/b` now route to the backend, 4 sites); 300 freeze-cascade
  safety/quality split (`meta.freeze_block_class`; Bark renders a "quality" block, halts only on
  "structural"); 302 story-QA reject LLM **gated OFF by default** (`OTR_ENABLE_STORY_QA`, opt-in);
  303 `BeatEdit` accepts `index`->`beat_index` alias. Details in BUG_LOG.md (auto-loads).
- **Comfy Credits lane PROVEN live** (real credit-billed calls). Key fact: Comfy bills LLM text via the
  **OpenRouter partner node** -> the Comfy Credits Activity shows "openrouter" + the model, charged to
  Comfy credits. `comfy:slot-a`=claude-opus-4.7; `openrouter:slot-a`=claude-opus-4.8 (own OpenRouter key,
  billed to the OpenRouter account, NOT Comfy). They sit adjacent in the writer dropdown -- easy to mix up.
- **OpenRouter own-key lane PROVEN end-to-end:** "Seven Worlds Our Size" -> OBS
  `signal_lost_seven_worlds_our_size_20260601_181200_procgen_blended.mp4` (88.7 MB), 2:32:55, 0 errors.
- **Added P11 (caption -65%) + P12 (Bark TTS eval, XTTS-v2 lead) to**
  `docs/2026-05-31-otr-consolidated-problem-statement.md` (Stream D). The go-forward below SUPERSEDES/
  expands P11 (captions) and overlaps P12 (TTS) -- fold these in there if formalizing.
- **Findings:** HuMo renders ~32 s/it vs the documented 14-18 s/it NORMAL_VRAM baseline (~2x slow) --
  cause is ComfyUI's new "DynamicVRAM" re-staging the 3321 MB HuMo model EVERY clip (`Model WAN21_HuMo
  prepared ... 3321MB Staged` per clip). Captions are Arial **52 px** at `nodes/_otr_captions.py:66`.

## State of the art
- **HEAD `19f5974`** on `v2.0-alpha`, local==origin. Canonical workflow: `workflows/otr_scifi_16gb_full.json`.
- **Node-ID -> class map** (for editing the go-forward; CONFIRM the *-flagged ones against the workflow JSON):
  - Node 1  = `OTR_LedgerScriptWriter` (writer; Bark-hygiene item #4 + script-stream split)
  - Node 7  = `OTR_EpisodeAssembler` (audio assembly; LUFS-normalize free-win)
  - Node 11 = `OTR_BatchBarkGenerator` (Bark TTS)
  - Node 14 = `OTR_MusicGenTheme`
  - Node 15 = `OTR_BatchAudioGenGenerator` (the AudioGen/SFX branch -- target for pushed parentheticals)
  - Node 20 = `OTR_VideoPlan` *(Jeffrey's label -- CONFIRM the actual video-plan node/class)*
  - Node 21 = shot duration -- likely `OTR_FixedShotDurationStub` *(CONFIRM)*
  - Node 23 = `OTR_BatchFluxRender` (environment keyframe -- item #3)
  - Node 51 = `OTR_BatchHumoRender` (HuMo video; lip-sync; the ~2.5h render)
  - Node 55 = `OTR_BatchLTXRender` (LTX motion)
  - Node 59 = `OTR_BatchFluxPortraitRender` *(Jeffrey's "FLUX portrait" -- CONFIRM; item #1 aspect + prompt)*
  - Captions = `nodes/_otr_captions.py:66` (`sdh_standard` Arial size **52** -> P11 target ~18; `otr_crt` 50 @ `:80`)
- **Scratch left in repo root this session** (untracked; relocate/remove per P10): `_otr_soak_driver.py`,
  `_otr_dump_scripts.py`, `_otr_smoke_launch.bat`, `_otr_wf_audit.py`, `_otr_obs_peek.py`,
  `_otr_crash_find.py`, `_otr_run_review.py`, `_otr_last_prompt.json`, `_otr_regress*.log`.
- Env knobs added this session: `OTR_ENABLE_COMFY_CREDITS=1` (persistent, setx), `OTR_ENABLE_OPENROUTER=1`
  + `OPENROUTER_API_KEY` (persistent). New opt-in flags: `OTR_ENABLE_STORY_QA` (default off),
  `OTR_STORY_QA_HARD_REJECT`, `OTR_COMFY_MIN_OUTPUT_TOKENS`, `OTR_BARK_HALT_ON_QUALITY_BLOCK`.

## Immediate next steps
Jeffrey's go-forward, ordered by leverage. Each node edit must be wired into the workflow JSON (PD3).

**SHIP NOW (low-hanging, in order):**
1. **Node 59 portrait -> landscape + in-world.** Set render aspect to **16:9** (kills the pillarbox bars
   eating ~40% of frame on monologues) and swap the prompt from studio-portrait to an active radio-room
   character shot. *The aspect change is the single biggest visual win.*
2. **Node 20 VideoPlan rule.** Every prompt carries **one action + one camera/framing + one motion source**;
   forbid two consecutive shots sharing a setup/angle (fixes the parked opening). Route composition cues
   to FLUX, motion cues to LTX.
3. **Node 23 environment.** Append action language so stills stop being pretty-but-dead.
4. **Node 1 Bark hygiene.** Send Bark clean spoken text only; push "(long breath)"/"(riffling pages)" to
   pause markers + SFX cues on the AudioGen branch (Node 15). **Quick check FIRST:** confirm those
   parentheticals currently reach Bark vs. already living only on the transcript card.
5. **Node 21 duration, selectively.** Drop establishing/B-roll holds to ~6.5-7.5 s, but let dialogue shots
   match their TTS line length -- HuMo lip-syncs to the line, so a hard cap clips the audio.
6. **Captions (= P11).** Smaller labels, less-opaque box, no all-caps, break lines on clause boundaries
   (the episode split "...the lights / go dark come spring"). `_otr_captions.py:66`.

**FREE WINS (settings, not architecture):**
- Normalize audio to ~**-16 LUFS / -1 dBTP** in AudioEnhance or EpisodeAssembler (Node 7) -- current peaks
  ~-3 dB, mean ~-24, feels quiet on phones.
- HUD corner text: make legible or remove (reads as noise).
- Closing transcript: ~40-50% dark scrim behind the text columns so green-on-lamp/CRT stops washing out.
- Casting: **Mina** (female name) is on male `speaker_7` -- swap if not deliberate (cf. BUG-269 cast RNG).

**LONG-RANGE (worth the build):**
- **Audio-driven shot duration.** Retire `FixedShotDurationStub`; VideoPlan sets each shot's length from
  its TTS line + handles. The principled cure for both parked shots and clipped VO (#5 only patches it).
- **Optional remote-video gate `OTR_VeoBranchGate`** mirroring the Flux/LTX gates + the OpenRouter
  remote-LLM pattern -- route only hero shots to a remote model as a v2.1/premium path; HuMo/LTX stay
  the local default.
- **Shot-type library** (establishing / OTS / hands-insert / CRT-readout / medium-CU) VideoPlan samples
  from -- turns "no two adjacent alike" into a system; sets up `crt_lower_third` / `classified_transcript`
  caption styles.

**Paste-ready prompt strings (use verbatim):**
```text
# -- Node 59 -- FLUX portrait prompt (also set render aspect to 16:9 landscape) --
cinematic medium close-up, character inside a retro radio control room / starship comms bay, urgently working analog dials, paper notes, microphone, and glowing green CRT monitors, face lit by screen glow, heavy shadows, 35mm film grain, tense old-time-radio sci-fi atmosphere

# -- Node 23 -- environment prompt (append to existing cinematic base) --
include a clear subject action: hands adjusting radio knobs, papers sliding across the desk, over-the-shoulder view of a CRT readout, signal waveform pulsing, warning lights flickering, dynamic depth of field -- no empty static room

# -- Node 20 -- OTR_VideoPlan prompt rule --
Every visual prompt must contain: (1) one physical action, (2) one camera/framing instruction, and (3) one motion source (flickering light, drifting smoke, moving paper, waveform pulse, or camera push-in). No two consecutive shots may share the same setup or camera angle. Send composition cues (action, framing, subject) to the FLUX prompt and motion cues to the LTX prompt.

# -- Node 1 -- OTR_LedgerScriptWriter / Bark rule --
Do not put emotional acting directions inside dialogue. Never write lines like "(long breath)" or "(sighs sadly)" for the TTS to perform. Output three separate streams: (a) short clean spoken lines for Bark, (b) explicit pause markers, and (c) SFX cues such as "heavy exhale", "paper rustle", "radio static swell", "chair creak" routed to the AudioGen/SFX branch.
```

## Open questions
- **#4 Bark parentheticals:** do "(long breath)"/"(riffling pages)" currently reach Bark, or already live
  only on the transcript card? Resolve before building the 3-stream split.
- **Mina casting:** is female "Mina" on male `speaker_7` a bug or deliberate?
- **P12 TTS:** confirm **XTTS-v2** was the "Express" model Jeffrey meant (then tighten P12).
- **Caption size:** -65% -> ~18 px may be too small; validate legibility on a real frame, or fall back to
  -50% (~26 px).
- **Node IDs 59 / 20 / 21:** confirm the class names against `workflows/otr_scifi_16gb_full.json` before editing.
- **HuMo ~2x slowdown:** investigate ComfyUI DynamicVRAM per-clip HuMo re-staging (separate from the above).

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps. Acknowledge when you're ready to start."
