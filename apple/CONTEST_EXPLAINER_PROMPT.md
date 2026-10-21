# Prompt: write the explainer script for Radio Play Studio

Paste everything below the line into a writing model (or hand it to a person). It carries every
fact the script needs; the writer must not invent any others. Written 2026-10-20 for Jeffrey A.
Brick's entry to Comfy's Open Source Higgsfield challenge.

---

You are writing the voice-over script and shot list for a short explainer video about a contest
entry. Read the whole brief before writing. Use ONLY the facts given here; if something you want to
say is not in this brief, leave it out. Do not round, inflate or guess any number.

## 1. The job

Write a **demo/explainer video script, 1:45 to 1:55 long** (the contest's hard limit is under 2
minutes; aim for 1:50). Deliver it as a table with one row per shot: shot number, start-end time,
what is on screen, the exact narration words (or "no narration" when the episode's own sound
plays), and any on-screen text. After the table, give: a word count of the narration, the total
running time, and a checklist proving every required name (section 6) appears on screen or in the
narration.

Narration pace: about 140 words per minute, so roughly 160-190 spoken words in total, because
about 25-35 seconds of the video should be the episode itself playing with no voice-over.

## 2. Who is watching, and what they judge

- The audience is the judging panel of **Comfy's Open Source Higgsfield challenge** (Comfy is the
  company behind ComfyUI). They are technical and have watched many entries.
- The rules for the video: **under 2 minutes**; show **the build in action and how it was built**;
  **name every model and tool used**. Judges "judge the work, not the video editing" -- so clarity
  beats flourish.
- Categories entered: **Most Innovative** (the video must say WHY it is innovative), **Open
  Weights Only** (every model must be open weights; no closed APIs), and the **LTX bonus** (LTX is
  the core video model). Best Overall is considered automatically.
- Content rules: safe for work; no unlicensed IP; no real-person likenesses in the generated
  footage. (The built-in story is the entrant's own published fiction, which he owns.)

## 3. What the entry is

**Radio Play Studio** -- "Press one button: your own GPU writes, casts, voices, scores and films a
whole radio play."

- It is an open take on Higgsfield's Cinema Studio for a different medium: instead of directing
  one shot at a time, **one press produces a whole story** -- an old-time-radio-style play with a
  beginning, middle and end, written, cast, voiced, scored and filmed.
- It is a small web app (Python standard library, no build step, MIT licence) that drives the
  entrant's own ComfyUI node pack, **Old-Time Radio** (`comfyui-old-time-radio` on the Comfy
  registry, 23 nodes), through the **Comfy SDK** and **comfy-api-proxy**. It converts the pack's
  shipped workflow into an API graph against the server's live node definitions.
- It runs **entirely on the user's own machine**: a 16 GB NVIDIA laptop GPU (RTX 5080 Laptop), no
  cloud, open weights only.

What the user does (this is the "in action" part):
1. Opens the app in a browser. Two cards: **"A classic, or surprise me"** (Shakespeare scenes,
   public-domain classics, or a roll of the dice) and **"Tell my own story"**.
2. Chooses language, number of acts, cast size and visual style.
3. In "Tell my own story" there is **one card per character** (name, he/she, who they are, how they
   look) plus three boxes: plot, setting, and who the story is by. **Leave every box empty and it
   tells the author's own built-in story** (below).
4. Presses **Make the episode**. A progress bar shows "about N minutes left"; it **learns how long
   each stage takes on that user's own machine**, follows the video node's own frame counter, and
   a page reload picks the run back up.
5. The finished episode plays: 1920x1080, voices, music, foley, burned-in captions.

## 4. How it works, in order (for the "how it was built" part)

1. **Writer -- Gemma 4 12B** (int8, run by ComfyUI itself): turns the idea or the user's boxes into
   a cast, a treatment and the script, act by act. In "Tell my own story" it is held to the user's
   own words (names, events, ending) by a source-correction pass.
2. **Cast lock and shot lock**: each character keeps one voice and one look for the whole play;
   each beat's camera, expression and motion are planned before filming.
3. **Stills -- FLUX 3 Klein** (Black Forest Labs, open weights): a picture for every beat. In
   "Tell my own story" each character's still is painted from that character's photos (the same
   open weights, through ComfyUI's reference conditioning), so faces hold from beat to beat.
4. **Voices -- Kokoro**: a distinct voice for every character and the announcer.
5. **Music -- Stable Audio 3**: the theme and the act-break music.
6. **Video -- LTX 2.5** (the 16 GB "foley" lane): films every beat from its still, **generating
   its own sound effects** that are mixed under the voices.
7. **Upscale -- ffmpeg** (Lanczos + unsharp, no model): to 1080p. (It replaced NVIDIA's closed RTX Video
   Super Resolution so the whole stack is open weights.)
8. **Finish**: captions burned in, everything muxed with ffmpeg, published as one MP4.

Platform: ComfyUI 0.42.1 + the Old-Time Radio pack; the app talks to it through comfy-sdk 0.5.0
and comfy-api-proxy 0.1.6. The workflow file is `otr_16gb_foley_open.json` (shipped in the pack).

## 5. Why it is innovative (the line the judges must hear; say it at least twice)

- **Other tools make clips. This makes a whole story** -- written, cast, voiced, scored and
  filmed, start to middle to end, from one press.
- **Characters stay themselves**: one voice and one look per character across every beat.
- **"Tell my own story"** turns a person's own writing into a produced radio play -- here, the
  entrant's own published fiction.
- **All local, all open weights** -- on one 16 GB laptop GPU.

## 6. Every name that MUST appear (spoken or on screen)

Gemma 4 12B; FLUX 3 Klein; LTX 2.5; Kokoro; Stable Audio 3; ffmpeg; ComfyUI; the
Old-Time Radio node pack; Comfy SDK; comfy-api-proxy. (Also allowed on an end card: ffmpeg.)
A compact way: one on-screen "stack" card listing all ten while the narration names the four that
matter most to the story (Gemma writes, FLUX paints, LTX films with sound, Kokoro speaks).

## 7. The footage available (write shots that use only this)

- **Screen recording of the app**: the two cards, the cast cards filling via "Load the example",
  the progress bar with minutes left, a reload that keeps the run.
- **The demo episode**: "Tell my own story" with nothing typed -- the built-in story, three acts,
  rendered on the open reference workflow My Story runs (`otr_16gb_foley_open_flux3_ref`), with
  the built-in story's own photos of its two leads. Story: Jeffrey, a shy former
  VJ preserved as a mind in a virtual afterlife in 2095, wins a day with MagnoliaRex, an electronic
  music artist; the day never ends -- Athens festival, a world tour, a talking cat in Cairo with a
  clue, and finally a frozen, empty Seoul as the machines that keep them alive fail. She says "I
  love you," plays him a duet made from his recorded voice, and at 11:59 p.m. says "Bon voyage,
  Jeffrey! Have a wonderful time!" Use a 25-35 second excerpt (Seoul ending recommended) with its
  own sound, no narration over it.
- **A simple diagram** (to be drawn): browser -> Radio Play Studio -> Comfy SDK -> comfy-api-proxy
  -> ComfyUI + Old-Time Radio pack -> models.
- **End card**: the repo link https://github.com/jbrick2070/ComfyUI-OldTimeRadio/tree/main/contest/radio-play-studio
  and #OpenSourceHiggsfield.

Do NOT use or describe: any older episode made with a different upscaler or still model; cloud
services; anything about cost or API keys.

## 8. Tone and style

- Plain, warm, confident. Short sentences. A nod to old-time radio is welcome in the opening line
  or the sign-off (an announcer's cadence), but no gimmick that eats time.
- Say what it DOES, not what it "leverages". No hype words ("revolutionary", "game-changing").
- First person is fine ("I built..."); the entrant is Jeffrey A. Brick.
- Let the episode breathe: the strongest proof is the play itself.

## 9. Suggested structure (adjust timings, keep the order)

1. 0:00-0:08 -- Hook + one-line promise.
2. 0:08-0:30 -- The app in action: cards, choices, press, the learning progress bar.
3. 0:30-1:05 -- The episode plays (own sound), with one short line before it about "Tell my own
   story" and the built-in story.
4. 1:05-1:35 -- How it is built: the diagram and the stack card; name the models in pipeline order.
5. 1:35-1:50 -- Why it is innovative, open weights, local; end card with link and hashtag.

## 10. Hard checks before you hand it back

- Running time 1:45-1:55; narration about 160-190 words.
- All ten names from section 6 appear.
- "Whole story, not a clip" is said; "open weights" and "on your own GPU / no cloud" are said.
- No number appears that is not in this brief.
- Photos are described only as the built-in story's own pictures of its characters.
