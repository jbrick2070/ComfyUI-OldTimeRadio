# OTR VR Director's Pass - concept brief

Status: 2026-10-30. The CORE is built and proven live (below); the per-shot pass is ADVANCED MODE, designed and measured, not built. Reviewed once (r1, at the end). Baseline `main` 396f8920.

## The core: the deck's picks as a fast pop-up (built 2026-10-30)

Operator, 2026-10-30, while the per-shot pass was being built: *"the highest layer, they are just choosing the same options as on the app / Stream Deck, if this is a true fast-paced pop-up ... advanced mode you are working on now is ok but not the core."*

So the first thing in the headset is the Stream Deck's choices, not the shot editor: **Machine, Video, Language, Story bank, Visual style, Acts, then MAKE**, one at a time, in big tiles. Ten seconds of picking, then the episode is made. It is `/direct` on Radio Play Studio's own server (`contest/radio-play-studio/radio_play_studio/static/direct.html`), the same routes the page and the deck use (`/api/options`, `/api/episodes`, the event stream, `/episodes/<run>/<file>`), and nothing on the server changed but the one route that serves the page.

- **Machine and Video come first** because they pick the workflow, and the workflow decides what the other four offer. Switching the way of filming re-reads the choices for that workflow (a few seconds; the page shows "one moment" and keeps every pick that the new workflow still offers). Staying on the server's own workflow costs no read at all.
- **My Story is skipped**, as on the deck: it needs typed text, which stays on the full page.
- **Inputs:** arrow keys and Enter; a gamepad's stick, d-pad, A and B (the Gamepad API); any MIDI knob or fader scrubs the row and any pad or key picks (Web MIDI, so the APC40 works as it is); a click or tap on a tile.
- **WebXR:** the same picks drawn as tiles on a ring around the viewer, in raw WebGL with no library: look at a tile to put the cursor on it, or push the thumbstick; pull the trigger to pick; squeeze to go back. The MAKE tile and the progress tile sit straight ahead. "Enter VR" appears only when the browser reports an immersive session. It is a view of the one state, so the flat page and the headset never disagree. **Tethered only** (decision 5 below still stands): the browser on the PC opens `localhost`, which WebXR counts as secure.
- **Proven live 2026-10-30** in the PC browser, end to end: all six picks in about ten seconds; a workflow switch (foley to stills) re-read the choices and kept every pick; MAKE posted `{language, bank, style, acts, workflow}`; the page followed the run from the event stream, and a reload mid-run found it again at 78%; the run became `transgression_shroud_20261030_231638__anim__stmo__msim__koko__pubd__cg412__sa3_final.mp4` in `otr/obs/` (one act, anime, still motion, rolled bank; 13:46 on the 5080), and the page went to "Done. Press play." with it loaded. The headset leg (an immersive session on a tethered Quest) is not yet run on hardware; the WebGL path is written against the WebXR API as specified and shares every code path with the flat view except the drawing.
- **Reviewed:** Codex (`gpt-5.6-sol`, read-only, briefed to refute) on the first commit found seven defects, all real, all folded the same night (`74d11e16` then the fix commit); a second Codex pass reviewed the fixes.
- **Tests:** `tests/test_rps_direct.py` -- the route, the loopback rule, unique ids, every id the script asks for, and the posted fields pinned to the deck's `DIAL_FIELDS` so one contract covers both.

- **How it reaches people:** from GitHub, the way Studio always has. `.comfyignore` keeps `contest/` out of the registry zip (since 2.3.39, operator 2026-10-23), so `/direct` needs no registry version and must never be the reason for one. A 2.3.48 bump was staged on the wrong premise on 2026-10-30 and reverted before any push.

The per-shot director's pass below is the advanced mode that opens once an episode exists.

## What it is

A **short VR session for directing an episode before the final OTR render**: put the headset on for about four minutes, make the big calls, take it off. Think Dragon's Lair: mostly watching, with a handful of moments where you make a quick call and the episode goes that way.

Operator, 2026-10-30: headsets hurt, and even a normal person is worn out after an hour on a flight. So the session is a **brief pop-in to direct**, not a place to work. Not a colorist's tool either: the calls are the big ones.

**The big calls, in the operator's words:** choosing the starting object (what the shot is *of*), the perspective angle, the image mode or style (anime, cinematic and the rest), and the overall color craft or cinematic look. Subject and angle are per shot. Style and look are per episode.

## How a session goes

1. **Before you put it on** (no one waiting): the system picks the decision points worth your time, about 6 to 10 per episode. Good candidates: the first shot of each scene, a run of shots with the same framing, a place the camera jumps sides in a conversation. For each one it **paints two or three alternatives in advance** (wide from the doorway / low on him / over her shoulder), so nothing in the session waits on the GPU.
2. **In the headset** (about four minutes): the episode's stills play in order with the dialogue. At each decision point it pauses and shows the alternatives side by side. You pick one, and it moves on.
3. **Afterward** (headset off): your picks become the Producer direction, and Studio submits the next stage. Anything you asked for that was not already painted is rendered now and waits for the next pass or the flat Studio page.

## Three ways to make a call

All three work at every decision point, and you can mix them:

- **Pick.** Look at an alternative and confirm, or point and pull the trigger.
- **Speak or type a direction.** An AI assistant is built in: say "lower, looking up at him, she's in the doorway behind" or type it like a prompt. It turns that into framing words for the shot (`picture_text`). If a painted alternative already matches, it picks that one; otherwise it queues a new take for after the session, so you never wait in the headset. The same assistant handles episode-level calls: "make the whole thing cinematic", "storybook".
- **Physical inputs.** The headset's controllers (thumbstick to move between alternatives, trigger to pick, a button for "keep as is"), a gamepad, or a hardware control surface. The browser's Web MIDI API reads a MIDI controller such as the APC40 (the FaderFace side project already maps one): pads pick an alternative, a fader sweeps between framings, a knob sets how much the assistant may change. Physical buttons mean no hunting for menus with your eyes while the headset is on.

## Where it sits: a front end for the Scene Producer

The Scene Producer (`planning/scene-producer/PLAN.md`, schema in `nodes/_otr_producer_direction.py`) already does the hard part. A producer run writes, voices and paints the episode, then stops before video. It leaves a bundle with `cards.json`, one card per shot, and the next run carries a `direction_json`. Only what changed is rendered again. Audio, cast, voices and timing never move.

What exists today (checked in r1):

| Call | What it becomes | Status |
|---|---|---|
| New framing, subject or angle, in words | `cards.<shot>.picture_text` on a **Stills** run | Works. A changed text rebuilds the prompt and its hash, which misses the still cache and repaints (`_otr_producer_direction.py:718`, `otr_image_gen_dispatcher.py:147`). A Video or Publish run with unpainted text is refused by design (`:1254`). |
| Paint again, same words | `takes.stills` | Works |
| Keep | `kept.stills` / `kept.clips` | Works |
| Scene note | `scenes.<id>.note` | Works |
| Change the *action* in the clip | `cards.<shot>.motion_text` | Works, but it is the action, not the camera. It replaces `creative["motion"]` and leaves the frozen `creative["camera"]` alone (`otr_shot_lock.py:2420`). |
| Camera move for the clip | none | Not today. No video engine takes a camera path (`_otr_video_engines/registry.py:70`). |
| Two or three alternatives per shot | none as such | See decision 1 |
| Episode style or look | none | A replay cannot change it (decision 4) |

## The decisions still open

1. **How are the alternatives painted?** A card holds one `picture_text`. Two ways:
   - **Sequential Stills runs**, one per alternative (A's framings, then B's, then C's), then a final run that sets each pick. The pick costs nothing, because the dispatcher's cache finds the earlier paintings. **Measured 2026-10-30, settled** (below). No schema change.
   - **A new `options` field per card** that paints several framings in one run. Cleaner, and it saves the fixed cost of each extra run, but it is a schema change to the direction, the freezer and `cards.json`.
   Decision: sequential runs for the first version. `options` only if the fixed cost per run starts to matter.

   **The measurement.** Project `c80906f6f093` (My Story, `otr_16gb_foley_open_flux3_ref.json`), shot `shot_shot_000_b3` (`flux3_klein`), through Radio Play Studio's own API, on one ComfyUI boot (0.43.2, booted after the last change to the dispatcher and direction code). Five Stills runs:

   | Run | Bundle | Picture text | Result | Run time |
   |---|---|---|---|---|
   | 1 | 12-stills | framing A (low angle) | painted, `8d2b05da` | 95 s |
   | 2 | 13-stills | framing B (wide, doorway) | painted, `b9876c08` | 79 s |
   | 3 | 14-stills | A again | **reused** `8d2b05da` from two bundles back | 66 s |
   | 4 | 15-stills | edit removed | stayed on A (see below) | 71 s |
   | 5 | 16-stills | original text typed back | **reused** `d0bf59bc` from bundle 11, five back | 72 s |

   The reuse is a cache hit, not an identical repaint. On runs 3 and 5 the server log goes from resolving the shot to the next object in about 20 ms, with no `portrait_ledger` line for a new picture, and the ledger's `cache_index` did not grow (30 entries after run 2, 30 after run 3). Every other shot was reused on every run. So **a run where nothing new is painted costs about 65-70 s** of fixed work (graph, replay import, freeze), and each newly painted still adds about 10-30 s on the 5080.

   **What run 4 taught.** An edit is cumulative: each bundle freezes the text it painted, and a direction with no edit for a card keeps the latest bundle's text, not the writer's first one. To go back, the room sends the old text explicitly. The VR pass always sends each pick's full text, so this costs nothing, but "undo" means restoring the text, not deleting the edit.

   For the Dragon's Lair pass, three alternatives on 8 decision points is three runs of about 70 s each plus 24 new stills, roughly 6-10 minutes of GPU before the headset goes on, and a free final run for the picks.
2. **Who picks the decision points?** Rules (first shot of each scene, repeated framing, the camera crossing sides) or the assistant reading the shot list. Rules first: they are checkable and cheap.
3. **Words, or a layout picture?** Words (path A) work today, but the engine may not obey exactly. A grey-box layout picture used as a Klein reference (path B) is an experiment, not a free feature: Klein takes at most three references, one primary plus two extras (`flux3_klein.py:93`), and today those are the speaker's and other cast portraits (`otr_image_gen_dispatcher.py` ~2265). A layout picture would push out a face, and enters the same reference chain with no "this is layout" role. There is no direction field to carry it either. A depth or pose control (path C) exists in no shipped engine.
4. **Style and look after the first run.** A Producer replay cannot change `visual_style`: the writer returns from its replay branch before style is chosen or stamped (`OTR_LedgerScriptWriter.py` ~3346-3439, stamp at ~4179), and the direction has no style field. There are ten choices, not nine: nine style packs plus `visual_storybased` (`_otr_rolls.py:428`). Options: choose style in a pre-pass (one shot painted in each style, pick in the headset, then the real first run), or a new direction field that overrides the frozen style and repaints every still. The procedural video lanes show no visible style change at all (writer tooltip ~2278), so style matters most on the still-driven lanes.
5. **How does the headset reach Studio?** Studio only answers on loopback: the CLI refuses any other bind (`cli.py:67`), and the server refuses any other Host header (`server.py:299`). That boundary is deliberate, since Studio has no login. WebXR also needs a secure context, and plain HTTP on a LAN address is not one.
   - **Tethered (recommended first):** the headset runs as a PC headset (Link or Air Link), and the browser on the PC opens Studio at `localhost`. Localhost counts as secure, and nothing about Studio's security changes.
   - **Standalone headset over Wi-Fi:** needs HTTPS, a login, and lifting the loopback rule. That is a security decision for the operator, not a default.
6. **Voice, offline.** Nothing exists yet. The browser's built-in speech input in Chrome sends audio to a cloud service, so it does not meet the offline default. Local Whisper on the 5080 does, but cannot run while a render holds the GPU; during a four-minute session nothing else is rendering, so that is fine. Typed prompts and physical inputs need no speech at all.
7. **Which workflow?** Studio submits its selected workflow (default `otr_16gb_foley_open.json`, `config.py`), a generated variant of `workflows/otr_canonical.json`. The suite changes no workflow; it only fills `direction_json` the way Studio does today.

## Limits that already apply

- 16 GB, one job at a time; Studio runs one stage at a time (`server.py:108`). Alternatives are painted before the session, never during it.
- An edit never changes audio, cast or timing, and Publish refuses a card that is not filmed as it stands (both Producer rules, confirmed in r1).
- We add no content filter. Cloud engines can still refuse a prompt on their own, and the Producer already shows a refused still as a failed card.

## First version

One scene, tethered, in the headset browser through Studio:

- two alternatives for each of that scene's decision points, painted by sequential Stills runs (decision 1), with the cache measured;
- pick with a glance or a controller, or type a direction that becomes `picture_text`;
- the picks submit as one Stills direction, and the episode continues on the flat Studio page.

This tests the operator's actual question: is a four-minute Dragon's Lair pass a faster, better way to make the big calls than the flat page? Speech, MIDI, style choice and any layout picture come after, each as its own step.

**The spatial version comes later**, and only if the first one earns it. r1's recommendation, kept: one camera, two stand-ins, and a small versioned `staging` payload, proving that moving them produces a predictably different still (through a measured Klein layout reference, or by falling back to words) before any scene-wide room is built. `eng_mesh_stage.py` is not a ready source of stand-ins: it renders a turntable clip of one figure, not posable characters in a shared set, and its result is a clip, not a GLB that Studio can see.

## Not in scope

- Fine retouching (operator, 2026-10-30).
- Reopening story or music quality (both closed by operator directive).
- Changing any shipped workflow. The suite is opt-in, like the Producer.

## Side quest: color looks

Operator, 2026-10-30: color shifts can be a side quest. Same open question as before: grade the still before filming (the video model films the look; changing it costs a re-film) or grade the clip at publish (cheap to change; the video model never sees it). Earlier lean: at publish.

## Review receipt

- **r1, 2026-10-30.** Driver: Claude Opus 5.5, which wrote its own grounding first (Studio localhost-only, Klein three-reference cap, `visual_style` frozen on replay). Contrarian: Codex `gpt-5.6-sol`, reasoning high, read-only, briefed to refute; tree unchanged. Verdicts: decisions 2 and 6 held; 1, 3, 5 and 7 refuted, plus the "first five rows" and the "served by Studio" claims; 4 unverifiable. Overall: the concept is worth building with changes; the curved-wall V1 was not, because it reproduced the Studio board without testing anything new. Every claim above was checked against the files before it went in.
- **Folded in:** loopback and secure-context limits, the Klein reference cap, `motion_text` being the action and not the camera, ten style choices and the replay freeze, Studio's default workflow being a variant, the mesh stage not being a stand-in source, the absence of any voice code, and the spatial slice as the later step.
- **Driver's call against r1:** Codex's spatial slice as the *first* version predates the operator's pop-in and Dragon's Lair direction. The first version is the short picking pass instead, because that is the question the operator asked; the spatial slice stays as the next step.
- **Struck from the earlier draft:** "Klein-ref holds identity" (Klein's own file says references anchor content and do not lock structure), and the `OTR_ENABLE_MESH_STAGE` gate (stale: it gates nothing).
- **Depth:** one round. Every r1 row turned into a fact, a measurement to make (decision 1's cache) or an operator call (decision 5). A second round would only confirm.
