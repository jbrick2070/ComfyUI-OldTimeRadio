# Radio Play Studio -- demo video storyboard (motion graphics, 1:50)

Operator 2026-10-20: "make a video in the same style as their intro video -- motion graphics, 3D
intro into walking through the nodes and videos appear ... they love the 3D rotating motion
graphics." Style taken from Comfy's launch video for the challenge (35 s, viewed on
comfy.org/open-source-higgsfield): a typewriter code line on black; hot orange backdrop; lime-yellow
heavy condensed all-caps type; app UI on dark purple panels; punchy kinetic cuts. Their logo and
their font are NOT used (ours: Impact, a system font; colours only).

Palette: orange #E8641E, lime #E4FF3A, deep purple #211927, black, white.
Rule: **the episode footage is pure output** -- shown unedited, tagged "UNEDITED OUTPUT". Every
other second is motion graphics made for this video.
Workflow shown and rendered: `workflows/otr_16gb_foley_open.json` (the submitted one).
Narration: Kokoro (open weights, the pack's own voice engine), plain American male, ~170 words.

| # | Time | Picture | Narration |
|---|---|---|---|
| 1 | 0:00-0:05 | Black. Typewriter, lime monospace: `> radio_play_studio --one-press` then a blinking cursor. | (none; a soft radio-dial sweep) |
| 2 | 0:05-0:13 | Hard cut to orange. 3D title RADIO PLAY STUDIO rotates in on its Y axis, extruded lime letters with depth and a soft shadow; subline types on: ONE PRESS. A WHOLE STORY. | "Press one button, and your own GPU writes, casts, voices, scores and films a whole radio play." |
| 3 | 0:13-0:30 | Dark purple. Three app screenshots arrive as tilted 3D panels, one after another: the two cards (A classic / Tell my own story); the cast cards; the progress bar "about N min left". Lime callout words pop beside each: PICK A STORY / YOUR OWN CAST / LEARNS YOUR GPU. | "Pick a classic, or tell your own story -- a card for every character. Press make. The bar learns how long each stage takes on your machine, and a reload never loses the run." |
| 4 | 0:30-1:00 | The episode, full frame, its own voices, music and foley. Corner tag: UNEDITED OUTPUT · 3-ACT EPISODE · RTX 5080 LAPTOP 16 GB. (The Seoul ending of the built-in story.) | (none) |
| 5 | 1:00-1:30 | The 3D node fly-through: the real 23-node graph as lime-edged cards on a purple floor, links glowing as signal flows left to right. The camera glides along it; at each stage a model label pops up in 3D and its output flies out of the card -- a script page at Story Writer, a waveform at the voices, a note at Theme Music, a still at Image Generator, a video clip at Video Render, a 1080p frame at Silent Composite. | "Under the hood it's my Old-Time Radio node pack on ComfyUI, driven through the Comfy SDK and comfy-api-proxy. Gemma 4 writes the story. FLUX 3 Klein paints every beat. Kokoro gives each character a voice, Stable Audio 3 scores it, LTX 2.5 films every beat with its own sound, and ffmpeg takes it to 1080p, with no model at all." |
| 6 | 1:30-1:43 | Orange. Kinetic type stack: OTHER TOOLS MAKE CLIPS / THIS MAKES A WHOLE STORY, then OPEN WEIGHTS ONLY with the ten names rotating in on a 3D ring. | "Other tools make clips. This makes a whole story -- beginning, middle and end, every face and voice held from shot to shot. All open weights, all on one laptop." |
| 7 | 1:43-1:50 | End card: RADIO PLAY STUDIO (3D, settles), the repo URL, #OpenSourceHiggsfield, BY JEFFREY A. BRICK. | "Radio Play Studio. Open source, on your own GPU." |

Required names, all present (spoken in 5, on screen in 6): Gemma 4 12B, FLUX 3 Klein, LTX 2.5,
Kokoro, Stable Audio 3, ffmpeg, ComfyUI, Old-Time Radio node pack, Comfy SDK, comfy-api-proxy.
(10-21: the 1080p step is ffmpeg, no model -- Real-ESRGAN was dropped.)

Build: Blender 5.2 (scenes 1, 2, 5, 6, 7; Eevee on the GPU after the demo render frees it), app
screenshots from the built-in browser (scene 3), ffmpeg for the cut, Kokoro for narration.
Output: output/otr/episodes/demo_cuts/radio_play_studio_demo.mp4 (1920x1080, 30 fps).
