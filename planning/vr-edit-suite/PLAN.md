# OTR VR Staging Room - concept brief

Status: idea, 2026-10-30. Nothing built. Baseline `main` ff78c374. This design has more than one defensible answer, so it goes through review before any code (CLAUDE.md, 2026-08-17 amendment).

## What it is

A VR room for making **the big staging calls** on an episode's shots **before the final OTR render**: where the camera stands, the angle, how wide or tight the shot is, where each character stands and faces, and whether a shot should be a different shot entirely. You walk around the scene, grab the camera, move a character, say "lower, looking up at him, she's in the doorway behind", and an AI assistant turns that into the next take of that shot.

Operator, 2026-10-30: not a colorist's tool. Fine work (color, small fixes) is out of scope. The suite is for the decisions that change what the picture *is*.

**The big choices, in the operator's words (2026-10-30):** choosing the starting object (what the shot is *of*), the perspective angle, the image mode or style (anime, cinematic and the rest of the nine in `nodes/_otr_visual_styles.py`), and the overall color craft or cinematic look. The first two are per shot. Style and look are per episode, and every style must craft the whole episode, stills and video alike.

"Strategic" means it works on a **whole scene**, not one picture: the shots of a scene sit around you in order, so you can see that three shots in a row are the same medium two-shot, or that the camera jumps sides between two shots of the same conversation, and fix the pattern instead of one frame.

## Where it sits: a VR front end for the Scene Producer

The Scene Producer (`planning/scene-producer/PLAN.md`, schema in `nodes/_otr_producer_direction.py`) already does the hard part. A producer run writes, voices and paints the episode, then **stops before video** (`ExecutionBlocker`). It leaves a bundle with `cards.json`, one card per shot. The next run carries a `direction_json` saying which cards to paint again, keep, or change, and only what changed is rendered again. Audio, cast, voices and timing never move.

The suite reads the same bundle and writes the same direction:

```
write + voice + paint    VR staging        film          VR review        publish
  (stills stage)    -->  of the stills --> (video stage) --> of the clips --> (publish)
                         -> direction_json                  -> direction_json
```

| VR action | What it becomes | Exists today? |
|---|---|---|
| "Paint this again" / pick take 2 | `takes.stills` | Yes |
| New framing, angle or blocking, described | `cards.<shot>.picture_text` | Yes |
| New camera move for the clip | `cards.<shot>.motion_text` | Yes |
| "This one is fine" | `kept.stills` / `kept.clips` | Yes |
| Note on a whole scene | `scenes.<id>.note` | Yes |
| Exact camera position and character marks | **new `staging` per shot** | No - schema change |

Framing today lives only in prompt words: the image engines (`nodes/_otr_image_engines/`) take text and, for Klein, reference pictures; none takes a depth or pose control image. So the first five rows are buildable now. The last row is the real new idea.

## The interface: an HTML page, never the node canvas

Operator, 2026-10-30: tuning by wiring nodes is not fun. The person in the headset never sees a node.

- The page is WebXR, opened in the headset's browser, and served by **Radio Play Studio's own server** (`contest/radio-play-studio/radio_play_studio/server.py`, port 8765), which already reads Producer bundles (`producer.py`) and submits stages to ComfyUI through the Comfy SDK (`runner.py`). The VR room becomes another view of Studio, not a second app that mirrors the schema again.
- Progress already reaches Studio through its runner (ComfyUI `/history`, SDK jobs). The room shows "painting shot 4..." and swaps the picture in when it lands.
- Graphs stay hidden: Studio fills `direction_json` on `workflows/otr_canonical.json` as it does today (CLAUDE.md section 0).

## The decisions the review has to settle

1. **How does a staging change reach the picture?** Three ways, in rising cost:
   - **A. Words.** The room turns your moves into framing words in `picture_text` ("low angle, wide, she stands in the doorway behind him"). Works today, but the engine may not obey, so the result is approximate.
   - **B. A rough layout picture.** The room renders a grey-box view from the camera you placed (stand-in figures, the set as blocks) and Klein uses it as a reference picture along with the cast portraits. No new engine type, but untested whether Klein follows a layout reference rather than copying it.
   - **C. A depth or pose control.** The grey-box view becomes a depth or pose map driving a ControlNet-style input. Most exact, but it is a new engine input that no shipped engine has, and it collides with the rule below.
   Lean: A first (free), then measure B on a handful of shots before anyone builds C.
2. **The same-engine rule.** `admit_prompt_direction` refuses a later run whose image engines differ from the bundle's (`producer_image_engines`), to avoid two looks in one episode. Path C, or any separate "staging" engine, breaks that rule on purpose. Either the rule learns about a staging layer or staging stays inside the same engine. Faces are the risk throughout: Klein-ref holds identity, the MOSS edit LoRA does not.
3. **What are the characters in the room?** Flat cut-outs of each shot's still (cheap, shows the current picture), or 3D stand-ins. `eng_mesh_stage.py` already turns each character's portrait into a cached GLB (Hunyuan3D-2mv) and drives Blender, so real 3D stand-ins are closer than they look, but that engine is opt-in (`OTR_ENABLE_MESH_STAGE`) and unproven as a staging tool.
4. **Where does the set come from?** Without a 3D set, "stand in the doorway" has no doorway. Options: a blank stage with marks, a set guessed from the still (depth estimate), or a few stock rooms. This decides how much of the room is real.
5. **Does staging carry to the video stage?** A camera move set in VR becomes `motion_text` words today. A real camera path would need the video engines to take a path, which none do.
6. **Can the style change after the first run?** `visual_style` is a writer widget, resolved once and written to the ledger meta (`OTR_LedgerScriptWriter.py` ~4179). A Producer replay passes the writer through, so today a later stage cannot switch anime to cinematic. Options: pick style in the room *before* the first paint (a style preview wall of one shot in each of the nine), or make style a Producer field that repaints every still. The second is a whole-episode repaint plus the same-engine rule again.
7. **Voice.** Headset browser speech input, or local Whisper on the 5080. Cloud is optional and off by default; the suite works offline.

## Limits that already apply

- 16 GB, one job at a time. A repaint waits in the ComfyUI queue; nothing runs beside an episode render.
- The room itself (moving stand-ins, flying the camera) runs in the headset and costs the 5080 nothing; only "paint it" does.
- An edit never changes audio, cast or timing, and a partial result never publishes (both Producer rules).
- Generated content is not filtered; the assistant applies what you ask.

## Smallest useful first version

The stills of one scene in a curved row around you, in order, in the headset browser via Studio. For each shot you can say or tap **keep**, **paint again**, or describe a new framing; the assistant writes the `picture_text` (path A) and Studio submits the next stage. No schema change and no new node. It answers the first question cheaply: is planning shots in VR better than on the flat Studio page?

## Not in scope

- Fine retouching (operator, 2026-10-30).

## Side quest: color looks

Operator, 2026-10-30: color shifts can be a side quest. Not part of the staging room's first version. When it comes back, the open question is the same one as before: grade the still before filming (the video model films the look; changing it costs a re-film) or grade the clip at publish (cheap to change; the video model never sees it). The earlier lean was grading at publish. Color runs in the headset as a live preview, so it costs the 5080 nothing until it is applied.
- Reopening story or music quality (both closed by operator directive).
- Changing the shipped default workflow. The suite is opt-in, like the Producer.
