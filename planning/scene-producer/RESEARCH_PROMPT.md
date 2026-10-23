# Research prompt: a scene-by-scene "micro producer" for Radio Play Studio

For Gemini Deep Research (operator idea, 2026-10-22). Paste everything below the line.

---

I build an open-source, local-first AI radio-drama pipeline on ComfyUI ("Old Time Radio" node pack,
and a Comfy SDK web app called Radio Play Studio). Today one button writes, casts, voices, scores and
films a whole episode: an LLM writes the script and a per-beat "shot lock" (expression, body motion,
one camera move, and the beat's foley sounds); FLUX 3 Klein paints a still per beat from character
reference portraits; LTX 2.5 films each beat with its own sound; Kokoro voices; Stable Audio scores.
It runs on a 16 GB laptop GPU.

I want to add a SCENE-BY-SCENE MICRO PRODUCER: after the script is written and before anything is
filmed, the user sees every scene and shot as an editable card -- who is in it, where, the camera
angle and move, lens/shot size, the action, the sound, the duration, continuity from the previous
shot -- can change any of it, lock it, reorder or split shots, regenerate one card, and then film
only what changed. My own earlier prototype (Veo-Moovie-Maker, a Gemini/Veo web app) had a Shot type
with: sequenceOrder, actionPrompt, environment, action, camera, negativePrompt, charactersInvolved,
duration, isContinuation, referenceImageNotes, startingFrame, endingFrame, plus a Cast Bank of
characters with a reference image each.

Please research, with sources and dates:

1. EXISTING TOOLS that do shot-by-shot pre-visualisation / storyboarding for AI video, open and
   closed: e.g. Higgsfield Cinema Studio, LTX Studio, Runway, Kling, Pika, Luma Dream Machine boards,
   Krea, Freepik, Google Flow, Katalist, Storyboarder, Boords, Plot, Shai, Saga, and any ComfyUI
   custom nodes or workflows for storyboards, shot lists or multi-shot consistency (search GitHub,
   the Comfy registry, Reddit r/comfyui and r/StableDiffusion, Hugging Face Spaces). For each: the
   per-shot fields it exposes, how the user edits, how character and location consistency is kept,
   whether it re-renders only changed shots, licence, and whether code is available.
2. THE BEST DATA MODEL for a shot card: compare the fields above with professional shot lists and
   screenplay/storyboard standards (shot size, angle, movement, lens, blocking, screen direction /
   180-degree rule, continuity notes, sound cues). Propose a minimal schema that an LLM can fill and a
   human can edit, and say which fields actually change what a video model renders.
3. LLM SHOT PLANNING: papers and repos on script-to-shot-list generation, multi-shot video
   consistency, and agentic "director" pipelines (e.g. Mora, MovieAgent, Anim-Director, StoryAgent,
   DreamFactory, VideoDirectorGPT, VideoStudio, FilmAgent, CineVerse, and anything newer in 2026),
   with what each found works -- especially for small local LLMs (4B-12B).
4. UI PATTERNS: how the good ones present 20-60 shot cards so editing stays fast (timeline vs grid vs
   script-linked), and how they show "locked" vs "will regenerate".
5. FOR A RADIO DRAMA specifically: anything on pairing shot planning with dialogue timing and sound
   design (foley per shot, room tone, continuity of ambience across shots).

Deliver: a comparison table of tools, a recommended shot-card schema (JSON) with a one-line reason
per field, three UI layouts worth copying with links, and a short list of open-source code I could
legally reuse (with licences). Flag anything you could not verify.
