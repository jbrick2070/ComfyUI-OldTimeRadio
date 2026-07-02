<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

NAME: The Living Portrait
PITCH: A single character headshot breathes and subtly moves with idle life, then speaks episode dialogue with flawless lip-sync, all born from one still and impossible to fake locally.
CHAIN: (1) Recraft/Flux2 Pro generates a pristine 1940s character portrait (cloud). (2) Tripo image-to-3D converts that portrait into a detailed head mesh (cloud). (3) [SPECULATIVE] Meshy rig & animate adds a looping idle animation (gentle sway, blink) and exports a video clip (cloud). (4) Kling audio-driven lip-sync overlay drives the mouth from the episode’s TTS line (cloud). (5) Final composite onto a period backdrop (local trivial).
WHY CLOUD-ONLY: Tripo’s single‑image‑to‑3D and Meshy’s cloud rigging run on models that require far more than 16 GB VRAM; the idle animation generation is also a cloud‑only heavy compute step.
IDENTITY/CONTINUITY ANGLE: The same 3D head mesh is reused for every beat of that character, guaranteeing identical facial structure; only the lip‑sync changes per line.
AUDIO-REACTIVE ANGLE: Lip‑sync is directly driven by the beat’s TTS audio; the idle animation can be tempo‑matched to background music for extra synchronisation.
RISK: Meshy’s idle animation may look unnatural or loop obviously, breaking the illusion of a living portrait.
COST SKETCH: Medium – one Tripo mesh + one Meshy animation per character, then cheap Kling lip‑sync per beat; reused across episodes.

NAME: Voice‑Painted Stage
PITCH: The entire radio‑theater set pulses, shifts, and breathes with every sound, turning the episode audio into a living backdrop that never repeats.
CHAIN: (1) Recraft generates a reference still of a 1940s radio‑studio stage (cloud). (2) For each beat, Seedance 2.0 reference‑to‑video takes the mixed episode audio chunk (dialogue + music + foley) as audio‑ref and the stage still as identity preservation, outputting a video where the set reacts dynamically to the sound (cloud). (3) Character portrait or lip‑sync videos are overlaid onto stage screens or as cut‑outs (local compositing).
WHY CLOUD-ONLY: Seedance 2.0 is a huge video diffusion model with audio‑ref conditioning—impossible on a 16 GB local GPU.
IDENTITY/CONTINUITY ANGLE: The same reference stage image is used across beats; Seedance’s identity preservation keeps the layout and props recognisable despite audio‑driven motion.
AUDIO-REACTIVE ANGLE: The video motion is inherently driven by the episode audio via the audio‑ref input.
RISK: Seedance may introduce flickering or lighting inconsistencies between beats, breaking the sense of a continuous stage.
COST SKETCH: High – each beat demands a full Seedance video generation, which is credit‑intensive.

NAME: Identity‑Locked Repertory
PITCH: Every line of dialogue is performed by the exact same face in a unique, era‑appropriate scene, as if the actor stepped from still to still, locked forever to the master portrait.
CHAIN: (1) Recraft creates a master reference portrait for each character (cloud). (2) For every character_video beat, Seedance 2.0 reference‑to‑video receives that portrait, the beat’s TTS audio line, and a text scene prompt (e.g., “dim detective office, venetian blinds”), producing a video of the character speaking/gesturing in the described setting with lip‑sync and identity preserved (cloud). (3) Optionally, Recraft remove/replace background reins in stray elements (cloud). (4) Assemble beats locally.
WHY CLOUD-ONLY: Seedance 2.0’s simultaneous audio‑ref, identity preservation, and text‑to‑scene generation is a colossal compute task far beyond any local 16 GB card.
IDENTITY/CONTINUITY ANGLE: The master portrait enforces identical facial appearance across all beats, solving the show’s hardest visual consistency problem.
AUDIO-REACTIVE ANGLE: Video is directly driven by the line audio—lip‑sync and emotional expression come from the audio‑ref.
RISK: Seedance may misinterpret the scene prompt, producing jarring backgrounds or inconsistent props that require heavy post‑processing.
COST SKETCH: High – each character beat fuels one full Seedance generation, making it the most credit‑hungry option.