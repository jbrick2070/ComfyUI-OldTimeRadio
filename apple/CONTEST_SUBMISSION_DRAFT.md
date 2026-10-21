# Open Source Higgsfield -- submission form, draft answers

Form: https://form.typeform.com/to/V1dSQrkS (12 questions, read 2026-10-19; nothing was entered).
**The operator fills and submits it.** Personal fields are left for him. Everything below is a
draft to paste from; every number is from a logged run on 2026-10-19.

## Before submitting (blockers)

- [ ] **Demo video under 2 minutes** (the form says so), posted somewhere linkable.
- [ ] **Social post** tagging #OpenSourceHiggsfield (X, IG, Reddit, TikTok...) -- the operator posts.
- [x] **Code link that will not move:** operator 2026-10-20, "merge the contest branch into main
      now so the code link does not move" -- the app is on `main`.
- [x] **Open weights -- DECIDED (operator 2026-10-20: "we skip VSR, we stay open weight";
      "flux 3 and esrgan"; "we skip the ref images").** The submission runs
      `workflows/otr_16gb_foley_open.json` (main 4c43ea10): FLUX 3 Klein stills from text, the open
      Real-ESRGAN x2 upscale instead of NVIDIA RTX VSR. **Tick Open Weights Only.**
- [ ] **Demo render on that exact workflow** (operator: "we need another run with ESRGAN for our
      demo to be precise") -- then cut the under-2-minute video from it.
- [ ] README: replace "Demo video: to be recorded" with the link.

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
- **Demo video URL:** -- after recording (under 2 min).
- **Live link to try it (optional):** **operator's call.** An episode is about an hour of GPU on a
  16 GB card and judges expect 5-10 runs, so a live link is not practical on the home box. If he
  wants one, the 5-minute stills workflow (`--workflow workflows/otr_16gb_still.json`) behind a
  password and a tunnel is the only realistic shape -- the app is loopback-only today and would need
  a login added first. Recommended: leave blank.
- **Social post link:** -- after posting.

## Q7. Which parts of ComfyUI did you use?
Comfy SDK (comfy-sdk 0.5.0) driving a self-hosted ComfyUI through comfy-api-proxy (the Comfy API
v2); ComfyUI core 0.42.1 with its native MOSS-Image, LTX 2.5 and Gemma 4 support; a custom node
pack on the Comfy registry (comfyui-old-time-radio, 22 nodes) whose shipped workflow the app
converts to an API graph against the server's live /object_info.

## Q8. List all generative models used
- **Gemma 4 12B** (int8, run by ComfyUI itself) -- writes the story, the cast and every line.
- **FLUX 3 Image Klein** (Black Forest Labs, open weights under the FLUX Kommunity License; int8
  model, Qwen3-VL-4B encoder, Flux 2 VAE) -- a still for every beat, from text.
- **LTX 2.5** (mix4x8 DiT + w4a8 encoder, 16 GB foley lane) -- films every beat, with its own sound.
- **Kokoro** -- a voice for every character and the announcer.
- **Stable Audio 3** -- the theme music.
- **Real-ESRGAN x2** (open weights, via spandrel) -- the 1080p upscale.

## Q9. Other tools or techniques used
- comfy-api-proxy (Comfy API v2 bridge) and a standard-library Python web app (no build step).
- Cast lock (each character keeps one face and one voice) and shot lock (each beat's camera,
  expression and motion planned before filming).
- Reference stills (an optional workflow, `otr_16gb_foley_flux3_ref`, not the submitted one): a
  portrait per character, and every still conditioned on the portraits of
  everyone in the frame (the speaker, the characters the shot names, the person spoken to).
- Cast cards: one card per character in Tell my own story (a photo on a card is used only by the
  reference workflow).
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
from public weights: Gemma 4 12B, FLUX 3 Klein, LTX 2.5, Kokoro, Stable Audio 3 and Real-ESRGAN.
No partner nodes, no closed APIs, no Comfy Cloud calls, no closed upscaler (RTX VSR is swapped for
Real-ESRGAN). The app and the node pack are MIT. Licence note for honesty: FLUX 3 Klein's open
weights come under the FLUX Kommunity License (non-commercial above US$5M a year).

## Q12. Agreements -- **operator.**

## Anything else for the judges?
> Radio Play Studio is an open take on Higgsfield's Cinema Studio for a different medium: instead
> of directing one shot at a time, one press writes, casts and directs a whole radio play. Every
> character keeps the same face and voice across beats, and LTX films each beat with its own sound
> under the voices. It runs on a 16 GB laptop GPU through the Comfy SDK and comfy-api-proxy, with
> no cloud. Example episodes: a scene from A Midsummer Night's Dream, and "Tell my own story": the
> author's own fiction, the tale of Jeffrey and MagnoliaRex, from his published book.
