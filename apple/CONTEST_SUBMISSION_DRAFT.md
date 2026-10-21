# Open Source Higgsfield -- submission form, draft answers

Form: https://form.typeform.com/to/V1dSQrkS (12 questions, read 2026-10-19; nothing was entered).
**The operator fills and submits it.** Personal fields are left for him. Everything below is a
draft to paste from; every number is from a logged run on 2026-10-19.

## Before submitting (blockers)

- [ ] **Demo video under 2 minutes** (the form says so), posted somewhere linkable.
- [ ] **Social post** tagging #OpenSourceHiggsfield (X, IG, Reddit, TikTok...) -- the operator posts.
- [ ] **Code link that will not move:** the app is on branch `claude/sharp-heisenberg-no3hzt`
      (PR #8). Either merge to `main` (operator's call) or link the branch.
- [ ] **Upscaler decision (for the Open Weights Only category):** the 16 GB foley workflow ships
      `upscale_engine = rtx_vsr` (NVIDIA RTX Video Super Resolution, a closed NVIDIA SDK that runs
      locally). The open choices are `spandrel_esrgan` (Real-ESRGAN x2, open weights, slower) and
      `off` (Lanczos + unsharp). Either switch the demo/default, or do not tick Open Weights Only.
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
  (after the merge; until then `/tree/claude/sharp-heisenberg-no3hzt/contest/radio-play-studio`).
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
- **FLUX 3 Image Klein** (Black Forest Labs, FLUX Kommunity License; int8 model, Qwen3-VL-4B
  encoder, Flux 2 VAE) -- every still. Character stills are painted from each character's
  reference portrait (reference latents), so every face holds from shot to shot; a person's own
  photo can be the reference for a character's portrait (workflow `otr_16gb_foley_flux3_ref`).
- **LTX 2.5** (mix4x8 DiT + w4a8 encoder, 16 GB foley lane) -- films every beat, with its own sound.
- **Kokoro** -- a voice for every character and the announcer.
- **Stable Audio 3** -- the theme music.
- Upscale: **Real-ESRGAN** if switched (see blockers); today's default is NVIDIA RTX VSR.
- (The plain `otr_16gb_foley` workflow paints from text with **MOSS-Image SFT**, Apache-2.0; the
  submission uses the reference workflow above.)

## Q9. Other tools or techniques used
- comfy-api-proxy (Comfy API v2 bridge) and a standard-library Python web app (no build step).
- Cast lock (each character keeps one face and one voice) and shot lock (each beat's camera,
  expression and motion planned before filming).
- Reference stills: a portrait per character, and every still conditioned on the portraits of
  everyone in the frame (the speaker, the characters the shot names, the person spoken to).
- Cast cards: one card per character in Tell my own story, with an optional photo.
- A progress bar that learns each stage's time on the user's own machine and follows the video
  node's own "frames rendered" bar: within 4 points of the wall clock on a 68-minute episode.
- Captions burned in (SDH), ffmpeg mux, every weight fetched tokenless from public repos.

## Q10. Which category or categories does your submission best fit?
- **Most Innovative** -- nobody else's entry makes a whole story: one press yields a written,
  cast-locked, shot-locked radio drama with voices, music and foley.
- **Open Weights Only** -- only if the upscaler is switched (see blockers).
- **LTX bonus** -- LTX 2.5 is the core video model (every beat is filmed by it).
- (Best Overall: every entry is considered automatically.)

## Q11. Is your stack fully open?
**Operator decision 2026-10-20: (a) keep FLUX 3 Klein with references and say so plainly.**
Open weights only: every model runs locally from public weights, no partner nodes, no closed APIs,
no Comfy Cloud calls; the app and the node pack are MIT. One honest caveat: FLUX 3 Image Klein's
weights are open but under the FLUX Kommunity License (free for non-commercial use; commercial use
only under US$5M a year), not an open-source licence -- it is what keeps every face consistent.
The upscaler must also be open for this answer (see blockers: Real-ESRGAN or off, not RTX VSR).

## Q12. Agreements -- **operator.**

## Anything else for the judges?
> Radio Play Studio is an open take on Higgsfield's Cinema Studio for a different medium: instead
> of directing one shot at a time, one press writes, casts and directs a whole radio play. Every
> character keeps the same face and voice across beats, and LTX films each beat with its own sound
> under the voices. It runs on a 16 GB laptop GPU through the Comfy SDK and comfy-api-proxy, with
> no cloud. Example episodes: a scene from A Midsummer Night's Dream, and "Tell my own story": the
> author's own fiction, the tale of Jeffrey and MagnoliaRex, from his published book.
