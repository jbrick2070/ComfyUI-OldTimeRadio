# Open Source Higgsfield -- submission form, draft answers

Form: https://form.typeform.com/to/V1dSQrkS (12 questions, read 2026-10-19; nothing was entered).
**The operator fills and submits it.** Personal fields are left for him. Everything below is a
draft to paste from; every number is from a logged run on 2026-10-19.

## Before submitting (blockers)

- [x] **Demo video under 2 minutes** (the form says so): https://youtu.be/UeQQeVMaMVs (1:53, posted 10-22).
- [x] **Social post** tagging #OpenSourceHiggsfield: https://www.instagram.com/reel/DezJQFfOLT3/ (Instagram reel, 10-22).
- [x] **Code link that will not move:** operator 2026-10-20, "merge the contest branch into main
      now so the code link does not move" -- the app is on `main`.
- [x] **Open weights -- DECIDED (operator 2026-10-20: "we skip VSR, we stay open weight";
      "flux 3 and esrgan"; "we skip the ref images").** The submission runs
      `workflows/otr_16gb_foley_open.json` (main 4c43ea10): FLUX 3 Klein stills from text, the open
      1080p step done by ffmpeg (Lanczos + unsharp, no model) instead of NVIDIA RTX VSR -- switched
      from Real-ESRGAN on 10-21 ("yes, switch to off"; "I want to be 100% open"): the repo's own
      benchmark measured it the most faithful of the three, at ~37 ms a frame, and Real-ESRGAN was
      running when the machine went down twice. **Tick Open Weights Only.**
      **10-21, REFERENCE STILLS BACK (operator: "use reference photos ... no VSR"):** My Story runs
      the open graph's reference twin, `workflows/otr_16gb_foley_open_flux3_ref.json` (main
      a9244e8a): character stills on FLUX 3 Klein with references (same open weights), announcer and
      music stills from text, the same ffmpeg 1080p step. The demo episode is rendered on it. Two
      things to say honestly: the bundled photos are of the author (his consent, his call against the
      "no real-person likenesses" rule), and they were blended once with a closed image tool before
      shipping -- a picture INPUT, not a model in the pipeline, which stays all open weights.
- [ ] **Demo render on that exact workflow** (operator, 10-20: "we need another run with ESRGAN for our
      demo to be precise") -- then cut the under-2-minute video from it.
- [x] README: the demo video link is in (10-22).

## Q1. Tell us about yourself (required)
Name / Comfy app email / Country -- **operator.**

## Q2. Usernames
Social handles, Discord, Reddit -- **operator.**

## Q3. How familiar were you with ComfyUI before this challenge?
**Operator.** (He has published a custom node pack on the Comfy registry since 2026-08.)

## Q4. Project name/title
Radio Play Studio

## Q5. One-line description (100 characters or less)
> Press one button: your own GPU writes, casts, voices, scores and films a whole radio play.

(90 characters.)

## Q6. Submission links
- **GitHub repo URL:** https://github.com/jbrick2070/ComfyUI-OldTimeRadio/tree/main/contest/radio-play-studio
  Public, MIT (`LICENSE` in the folder and at the repo root), setup in the README.
- **Demo video URL:** https://youtu.be/UeQQeVMaMVs
- **Live link to try it (optional):** **operator's call.** An episode is about an hour of GPU on a
  16 GB card and judges expect 5-10 runs, so a live link is not practical on the home box. If he
  wants one, the 5-minute stills workflow (`--workflow workflows/otr_16gb_still.json`) behind a
  password and a tunnel is the only realistic shape -- the app is loopback-only today and would need
  a login added first. Recommended: leave blank.
- **Social post link:** https://www.instagram.com/reel/DezJQFfOLT3/

## Q7. Which parts of ComfyUI did you use?
Comfy SDK (comfy-sdk 0.5.0) driving a self-hosted ComfyUI through comfy-api-proxy (the Comfy API
v2); ComfyUI core 0.42.1 with its native MOSS-Image, LTX 2.5 and Gemma 4 support; a custom node
pack on the Comfy registry (comfyui-old-time-radio, 22 nodes) whose shipped workflow the app
converts to an API graph against the server's live /object_info.

## Q8. List all generative models used
- **Gemma 4 12B** (int8, run by ComfyUI itself) -- writes the story, the cast and every line.
- **FLUX 3 Image Klein** (Black Forest Labs, open weights under the FLUX Kommunity License; int8
  model, Qwen3-VL-4B encoder, Flux 2 VAE) -- a still for every beat; in Tell my own story each
  character's still is painted from that character's photos (the same weights, through ComfyUI
  core's ReferenceLatent).
- **LTX 2.5** (mix4x8 DiT + w4a8 encoder, 16 GB foley lane) -- films every beat, with its own sound.
- **Kokoro** -- a voice for every character and the announcer.
- **Stable Audio 3** -- the theme music.
- No upscale model: **ffmpeg** scales to 1080p (Lanczos + unsharp).

## Q9. Other tools or techniques used
- comfy-api-proxy (Comfy API v2 bridge) and a standard-library Python web app (no build step).
- Cast lock (each character keeps one face and one voice) and shot lock (each beat's camera,
  expression and motion planned before filming).
- Reference stills (`otr_16gb_foley_open_flux3_ref`, what Tell my own story runs): a portrait
  per character, and every still conditioned on the portraits of everyone in the frame (the
  speaker, the characters the shot names, the person spoken to).
- Cast cards: one card per character in Tell my own story, each with up to three photos.
- A progress bar that learns each stage's time on the user's own machine and follows the video
  node's own "frames rendered" bar: within 4 points of the wall clock on a 68-minute episode.
- Captions burned in (SDH), ffmpeg mux, every weight fetched tokenless from public repos.

## Q10. Which category or categories does your submission best fit?
- **Most Innovative** -- nobody else's entry makes a whole story: one press yields a written,
  cast-locked, shot-locked radio drama with voices, music and foley.
- **Open Weights Only** -- every model is open weights (workflow `otr_16gb_foley_open`).
- **LTX bonus** -- LTX 2.5 is the core video model (every beat is filmed by it).
- (Best Overall: every entry is considered automatically.)

## Q11. Is your stack fully open?
Yes -- open weights only (operator 2026-10-20: "we stay open weight"). Every model runs locally
from public weights: Gemma 4 12B, FLUX 3 Klein, LTX 2.5, Kokoro and Stable Audio 3; the 1080p step
is ffmpeg, no model.
No partner nodes, no closed APIs, no Comfy Cloud calls, no closed upscaler (RTX VSR is swapped for
ffmpeg's Lanczos scaler). The app and the node pack are MIT. Licence note for honesty: FLUX 3 Klein's open
weights come under the FLUX Kommunity License (non-commercial above US$5M a year).

## Q12. Agreements -- **operator.**

## Anything else for the judges?
> Radio Play Studio is an open take on Higgsfield's Cinema Studio for a different medium: instead
> of directing one shot at a time, one press writes, casts and directs a whole radio play. Every
> character keeps the same face and voice across beats, and LTX films each beat with its own sound
> under the voices. It runs on a 16 GB laptop GPU through the Comfy SDK and comfy-api-proxy, with
> no cloud. Example episodes: a scene from A Midsummer Night's Dream, and "Tell my own story": the
> author's own fiction, the tale of Jeffrey and MagnoliaRex, from his published book.
