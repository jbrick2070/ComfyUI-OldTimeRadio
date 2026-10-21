# Radio Play Studio -- demo video script (under 2 minutes)

The form requires: under 2 minutes, "your build in action and how you built it"; Most Innovative
also requires saying why it is innovative. Judges "judge the work, not the video editing".
Every model and tool must be named. Times are targets; total 1:55.

Footage needed (all on the 5080):
- Screen capture of the app at http://127.0.0.1:8765 (or the CLI) -- see shots 2-4.
- `midsummer_highlight.mp4` and `magnolia_highlight.mp4` -- cut from the finished episodes with
  ffmpeg (no re-render; see the plan file).
- The README's "Measured timings" table for shot 6, if a still frame of text helps.

| # | Time | Picture | Narration (voice-over, plain and short) |
|---|---|---|---|
| 1 | 0:00-0:08 | Title card: "Radio Play Studio -- press one button" | "Press one button, and your own GPU writes, casts, voices, scores and films a whole radio play." |
| 2 | 0:08-0:22 | The app: the two cards, "A classic, or surprise me" / "Tell my own story"; pick Shakespeare, Midsummer, 1 act; the Still pictures picker on MOSS-Image | "Pick a classic, or tell your own story. Choose a language, the acts, the cast size, the look." |
| 3 | 0:22-0:32 | Press Make the episode; the bar with "about N min left"; reload the tab -- it picks the run back up | "It runs on your machine. The bar learns how long each stage takes on your GPU -- within four points of the clock -- and a reload never loses it." |
| 4 | 0:32-0:58 | `midsummer_highlight.mp4` (Bottom and Titania) | (let the episode's own voices and foley play; no narration) |
| 5 | 0:58-1:24 | "Tell my own story": press **Load the example**, the three cast cards fill (Jeffrey, MagnoliaRex, the Cat), then `magnolia_highlight.mp4` | "Or tell your own. A card for each character -- even a photo. This is my own published novel, turned into a three-act radio play, and every face stays the same from shot to shot." |
| 6 | 1:24-1:46 | A simple diagram: browser -> Radio Play Studio -> Comfy SDK -> comfy-api-proxy -> ComfyUI + Old-Time Radio pack; then the model list | "Built on the Comfy SDK and comfy-api-proxy, driving my Old-Time Radio node pack on ComfyUI. Gemma 4 writes it. MOSS-Image and FLUX 3 Klein paint every beat -- Klein drawing each one from the characters' portraits. Kokoro gives each character a voice. Stable Audio 3 scores it. And LTX 2.5 films every beat with its own sound." |
| 7 | 1:46-1:55 | End card: repo URL, #OpenSourceHiggsfield | "Every face and voice stays the same from shot to shot. Open weights, MIT, no cloud -- a whole story, not one clip." |

Why it is innovative (say it in shots 1, 5 and 7; the line is the point):
- Other entries make clips. This makes a whole story: written, cast, voiced, scored and filmed,
  start to middle to end, from one press.
- Characters stay locked: the same face and the same voice in every beat.
- "Tell my own story" turns a person's own writing into a produced radio play.

Model and tool names to say or show (all of them, per the rules): Gemma 4 12B; MOSS-Image;
Kokoro; Stable Audio 3; LTX 2.5; ffmpeg (the 1080p step on the submitted graph, no model --
10-21; RTX VSR and Real-ESRGAN are both out of the contest graph); Comfy SDK;
comfy-api-proxy; ComfyUI; the Old-Time Radio node pack.

Note: both demo episodes were rendered with FLUX 3 Klein stills -- Midsummer before the MOSS
switch, MagnoliaRex with reference stills (2026-10-20) -- so name FLUX 3 Klein in shot 6.
`magnolia_highlight.mp4` is the Athens scene (0:30-0:56 of the episode, no Cat); the alternate
`magnolia_highlight_ending.mp4` shows the open humanoid-Cat defect.
