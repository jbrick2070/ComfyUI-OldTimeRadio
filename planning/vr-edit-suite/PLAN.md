# OTR VR Edit Suite - concept brief

Status: idea, 2026-10-30. Nothing built. Baseline `main` 439ba9854. This is a design with more than one defensible answer, so it gets the full review arc before any code (CLAUDE.md, 2026-08-17 amendment).

## What it is

A VR room you stand inside to review and fix an episode's pictures **before the final OTR render**. You look at each shot's still (or its filmed clip) wrapped around you, say what you want in plain words, point at the part you mean, and an AI assistant makes the change. When you are happy, you leave the headset and the episode publishes with your edits.

It does two kinds of work:

1. **Color and look (fast, no AI needed to run).** Exposure, white balance, contrast, curves, film looks, LUTs. Runs live in the headset, so a knob turn shows instantly. The AI's job is only to set the starting values from what you said ("1940s tungsten, a little faded") and to keep the look the same across shots.
2. **Picture edits (slower, AI does the work).** "Add rain on the window", "make her coat red", "remove the chair", "paint this shot again". Sent to ComfyUI on the 5080; the result fades in a few seconds later.

"Strategic" means it works on the **whole episode**, not one picture at a time: it shows every shot of a scene around you, flags the ones that do not match (a face that changed, a shot much brighter than its neighbors), and lets one decision apply to a scene or the whole episode.

## Where it sits: it is a VR front end for the Scene Producer

The Scene Producer (`planning/scene-producer/PLAN.md`, schema in `nodes/_otr_producer_direction.py`) already does the hard part. A producer run writes, voices and paints the episode, then **stops before video** (`ExecutionBlocker`). It leaves a bundle with `cards.json`, one card per shot. The next run carries a `direction_json` saying which cards to paint again, which to keep, which takes to use, and only what changed is rendered again. Audio, cast, voices and timing never move.

So the VR suite does not need its own pipeline. It reads the same bundle and writes the same direction:

```
write + voice + paint      VR review           film           VR review          publish
  (stills stage)    -->  stills in VR  -->  (video stage) --> clips in VR --> (publish stage)
                         edits -> direction_json        edits -> direction_json
```

| VR action | What it becomes | Exists today? |
|---|---|---|
| "Paint this again" / pick take 2 | `takes.stills` | Yes |
| Change what a shot shows | `cards.<shot>.picture_text` | Yes |
| Change how it moves | `cards.<shot>.motion_text` | Yes |
| "This one is fine" | `kept.stills` / `kept.clips` | Yes |
| Note on a scene | `scenes.<id>.note` | Yes |
| Color look per shot / scene / episode | **new `grade` section** | No - schema change |
| Region edit ("make this coat red") | **new edit request with a mask** | No - schema change |

The first five need no change to the graph. A VR viewer that only does those is a new front end and nothing else.

## The interface: an HTML page on the Comfy SDK, never the node canvas

Operator, 2026-10-30: fine-tuning a look by wiring nodes is not fun. The person in the headset never sees a node. The page is plain HTML/WebXR and talks to ComfyUI the same way Radio Play Studio does:

- **Submit** through the Comfy SDK and `comfy-api-proxy` (:8189 in front of ComfyUI on :8000). A stage is one SDK job; its id is the ComfyUI prompt id, so the page can re-attach after the headset sleeps (`client.jobs.get(id)`).
- **Hear back** without polling by hand: ComfyUI's own `/ws` websocket streams progress and "executed" events, so the room can show "painting shot 4..." and swap the picture in the moment it lands. Whether the proxy can also call a webhook out to the page is not checked yet; the websocket is enough on one machine.
- **Read** pictures and `cards.json` through core `/view`, as the Producer plan already does. No new server route.
- **Graphs stay hidden.** The page fills `direction_json` and the existing widgets on `workflows/otr_canonical.json`; it never builds its own graph (CLAUDE.md section 0).

## The decisions the review has to settle

1. **Grade the still before filming, or the clip after?** Grading the still means the video model films the graded picture, so the look is baked in and costs a re-film to change. Grading after (at publish, as an ffmpeg filter or LUT) is cheap to change and never re-films, but the video model never sees it. Lean: grade after, at publish, with an option to bake a strong look into the still.
2. **Which engine does a region edit?** `admit_prompt_direction` refuses a run that paints with a different image engine than the bundle's stills (a mix of two looks). A region edit with a separate edit model would break that rule on purpose. Either the rule learns about an edit layer, or region edits go through the same engine as a masked repaint. Faces are the risk: the MOSS edit LoRA does not hold identity; Klein-ref is the engine that does.
3. **A wrap-around picture, or a flat screen in a VR room?** OTR stills are flat 16:9 frames. Wrapping them 360 degrees needs outpainting and seam-matching for every shot. A large curved screen in a dark room (a screening room) needs none of that and is how a real colorist works. Lean: screening room first; a 360 "step inside" view later as an extra.
4. **Where does it run?** The page is served from the ComfyUI server and opened in the headset's browser (WebXR). No app store. It mirrors the direction schema the way Radio Play Studio does, or becomes a mode of Studio.
5. **Voice.** The headset browser's speech input, or local Whisper on the 5080. Cloud is optional and off by default; the suite has to work offline.

## Limits that already apply

- 16 GB, one job at a time. An AI edit waits in the ComfyUI queue; it does not run beside an episode render.
- Color work runs in the headset (a WebGL shader), so it costs the 5080 nothing.
- An edit never changes audio, cast or timing (the Producer's rule).
- A partial result never publishes (the Producer's rule).
- Generated content is not filtered; the AI applies what you ask.

## Smallest useful first version

A screening room in the headset browser that loads one stills bundle and shows the cards scene by scene. You can:

- say or tap **keep**, **paint again** or **change the picture**, and the page writes a `direction_json` and submits the next stage;
- set a **look** with a voice prompt plus a few knobs, previewed live, stored in the page only until decision 1 is settled.

No schema change and no new node for this version. It shows whether reviewing in VR beats reviewing on the flat page before anything shared is touched.

## Not in scope

- Reopening story or music quality (both closed by operator directive).
- Changing the shipped default workflow. The suite is opt-in, like the Producer.
