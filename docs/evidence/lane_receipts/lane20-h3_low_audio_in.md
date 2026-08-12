# VIDEO_LANE_PREFLIGHT receipt -- lane 20, `h3_low_audio_in` (`minimax_h3_audio_in`)

`VIDEO_LANE_PREFLIGHT receipt: minimax_h3_audio_in | 2026-08-12 | smoke receipt
output/otr/episodes/_lane_smokes/lane20_h3_low_audio_in/ | verdict PASS`

**The cheapest lane in the campaign, because lane 19 paid for it.** This is a
SECOND registration out of the module lane 19 wrote, not a new engine: same
canvas, same grid, same 24->25 conversion, same boot contract, same preflight,
same V-1 self-probe. It rendered live on the FIRST attempt.

## Matrix row

7/7 GREEN on arrival, and lane 19 stayed 7/7 through the refactor. No
`EXPECTED_RED` entries to delete -- a new engine either satisfies every gate at
registration or it must not register.

## What this lane actually is, and the one sentence that gets it wrong

`MiniMaxH3ReferenceToVideo` conditions on REFERENCE media: a portrait presented
to the tokenizer as `<Picture 1>` and the beat's own audio as `<Audio 1>`, on
the **ref2va** DiT -- a different 21 GB repack from lane 19's fl2va, one token
apart in the filename.

**It loads an audio VAE and it still emits silence.** That VAE is a required
input of the reference node, which uses it to ENCODE the reference audio into
the conditioning. No `VAEDecodeAudio` exists in this graph any more than in lane
19's. "Loads an audio VAE" and "emits audio" are different claims and only the
first is true here -- the smoke proves it from both ends: the server log shows
`Requested to load MiniMaxH3AudioVAE` (576 MB staged), and the delivered
container reports `nb_streams=1`.

## The four differences from lane 19, each a decision

### 1. Continuity is `soft_reference`, and lane 19's STRICT is not inherited

This is the contract value the shared base deliberately does NOT provide.

Lane 19 wires `first_frame`, which the FL2VA node pins as a keyframe at
`resolved_frame_index 0` and re-injects every step -- a real first-frame lock,
so it earns `strict_first_frame`. **`MiniMaxH3ReferenceToVideo` has no
`first_frame` input at all.** Its `ref_images` are IDENTITY references and
nothing pins frame 0 to any of them. Claiming STRICT here would promise the
coverage planner a seam this node cannot deliver, and the visible cost of a
wrong strict claim is a jump at a join the plan said was seamless. A jump cut is
honest -- the same reasoning `google_veo_video` records for its own endpoints.

### 2. The mouth policy, which fails at PLAN time when it is not extended

`render_driver._is_character_face_beat` tested `engine_id == "ltx_audio_in"` by
EQUALITY. It is now a membership test over `_AUDIO_IN_CHARACTER_ENGINES`.

**Registering this family without that change does not degrade anything -- it
makes every H3 character beat fail before a single weight loads.**
`mouth_owner_for_beat` REFUSES an audio-in beat that is neither a character face
nor a cabinet role, because nothing would then decide whether the still it is
about to animate has lips. The test asserts that consequence, not the mapping:
it raises `MouthPolicyError` with `is_character_face=False` and returns
`MOUTH_HUMAN` with it True.

**The wrong fix, named so nobody reaches for it:** minting a new family to dodge
the check. A family outside `content_oracle.MOTION_FAMILIES` makes frozen clips
motion-EXEMPT -- trading a loud plan-time refusal for a silent quality hole.

### 3. The scene still must never OVERWRITE the reference it lip-syncs

**This one was WRONG in the first draft, and the post-coding QA caught it.**

`_engine_scene_init_required` overwrites `init_image` with the beat's wide scene
still for any non-face engine that declares `init_image`. `ltx_audio_in` is
excluded from it by name; this lane was not. On a lane whose `init_image` IS the
reference the model lip-syncs -- presented to the tokenizer as `<Picture 1>`,
with the lane's own prompt saying "a medium close shot of `<Picture 1>` speaking
directly to camera" -- that does not fail. It renders the wrong identity,
silently, on every beat. Both audio-in lanes are excluded now.

**The comment I wrote was false about the code beneath it, and the test I wrote
to prove it was LEXICAL, so it passed anyway.** The draft claimed this lane
"does NOT take the scene-still spine"; in fact it does, exactly like
`ltx_audio_in`, because `_still_spine_requires_scene` returns True for the whole
`audio_conditioned_video` family. That is fine -- the SPINE decides which stills
get MINTED and a per-beat scene still is worth having. The two questions are now
separate assertions so they cannot be confused again: the spine one asserts
True, and the overwrite one is a behavioural check on the real branch condition.

**Third lexical-check failure of the session**, and the first one that let a
real defect through rather than merely failing on a comment. The seed test was
rewritten the same way in the same pass -- it grepped the module for the literal
"43" and now asks the graph instead.

### 4. It HARD-requires both inputs

`required_inputs = ("audio_ref", "init_image")`, each with its own named
refusal before anything is staged. The audio one says why it matters rather than
that a field is empty: without the beat's audio this lane would render an
UNCONDITIONED clip while its receipt claimed lip-sync.

## THE ONE UNVERIFIED THING FROM THE HANDOFF, NOW SETTLED

The handoff flagged the `COMFY_AUTOGROW_V3` reference sockets as the single
thing to prove before building around it. **Settled by reading the runtime, then
confirmed by the render.**

In the lab's API graph those sockets serialize DOTTED
(`"ref_images.ref_image_0"`) because ComfyUI's prompt EXECUTOR flattens the
schema and reassembles the dict before calling the node. This adapter calls the
node class directly through `wrapper_bridge`, which bypasses the executor
entirely -- and a V3 node's `FUNCTION` is `EXECUTE_NORMALIZED`, a plain
passthrough to `execute(*args, **kwargs)`. So `execute` must receive the dict it
iterates: `ref_images={"ref_image_0": Wire(...)}`. `_iter_wires` recurses dicts,
so the wires nested inside resolve normally. Pinned by two tests -- one on the
shape, one asserting the nested wires are reachable, because a socket that looks
connected and resolves to nothing would render with no references at all and
still succeed.

## The refactor, and what it was NOT allowed to do

Lane 19 shipped and rendered live at `be4aadff`. The split into
`_MiniMaxH3Base` + two subclasses had one hard constraint: **lane 19's behaviour
must not move.** Its graph is composed from `_sampler_spine` + its own
`_conditioner_nodes` and is node-for-node what it was; its refusal ordering
(Sage -> boot -> weights -> node classes, with canvas and length checked BEFORE
anything is staged) is unchanged; and its own 41-test suite still holds the
whole lane-19 contract, now through the MRO.

**One defect the refactor introduced, found and fixed before the push.**
`session_identity` read the module-level `H3_RECIPE_RECEIPT` -- lane 19's
receipt -- so the moment a second adapter shared that method, lane 20's identity
described lane 19's recipe. It stayed DISTINCT between the lanes anyway, because
the DiT token in the same tuple differs, so nothing would have reused the wrong
session; it would simply have been a receipt that lies about which recipe
produced the handles. It is exactly the hardcoded-per-lane-value-in-a-shared-base
that this class's own docstring warns about, written by the same hand in the same
change. Found by COMPARING the two identities, which is a check neither lane's
own tests would have made -- both would have passed in isolation.

`_UNET_DEFAULT` has **no default on the base**, deliberately. A third adapter
must SAY which repack it loads rather than inherit whichever happened to be
first -- the two H3 DiTs are the same size and one token apart, which is exactly
the pair a shared default gets wrong silently. `_weight_rows()` raises a named
`EngineUnusable` when it is unset, and a test proves it.

## G8.1 solo smoke

| Item | Value |
|---|---|
| Boot | the named **`h3`** contract: Sage-free, `--disable-pinned-memory`, `--reserve-vram 12` |
| Harness | `_otr_single_engine_smoke.py --engine minimax_h3_audio_in --frames 129 --portrait <png> --audio <wav>` |
| Prompt id | `5614ecf0-2684-458c-ba1d-bdafc56b015e` |
| Reference inputs | `lane20_portrait.png` (1024x1024) + `lane20_voice.wav` (10.0 s, 44.1 kHz mono PCM), both from the lab's fixture set |
| Recipe | `minimax_h3_ref2va_int8_res_multistep_20step_v1` |
| Wall time | **239.4 s** -- within a second of lane 19's 242.6 s, as expected for the same stack |
| Canvas PROBED | **864x480** -- equals the declaration |
| Frames PROBED | **129 packets counted**, and 129 was the ask -- no trim |
| Rate / duration | **25/1**, duration **5.160000 s** = 129/25 EXACTLY |
| Codec / pixfmt / colour | h264 / yuv420p / bt709 |
| Audio | **`nb_streams=1`** -- ZERO audio streams, on the lane that LOADS an audio VAE |
| Audio VAE PROVED loaded | server log: `Requested to load MiniMaxH3AudioVAE ... 576MB Staged` |
| Extension | `native_frame_count: 129`, `extension_mode: none` |
| Peak, ABSOLUTE | **6,678 MB**, `VramPeakProbe` maximum, cold (lane 19: 6,315 MB -- the audio VAE is the difference) |
| Peak, NET | **not claimed.** No pre-queue baseline was sampled, and the cost-row surface is NET (L7) |
| Artifact | `.../lane20_h3_low_audio_in/minimax_h3_audio_in_864x480_f129_h3boot_smoke.mp4` |
| sha256 | `e81fa0d73da23f6ae1cdc5cd22b8afc02d55954e88dd1084f0328dc4ab7c3f91` |

**One attempt, no failed gates.** Lane 19's three attempts bought this: the
boot, the staging, the canvas and the length checks were all already right.

## Deliberately NOT done here

**No lip-sync QUALITY claim.** The clip is machine-proven -- right canvas, right
length, right rate, silent, conditioned on a real audio reference that a real
audio VAE encoded. Whether the mouth actually matches the voice is an EAR/EYE
judgment and it is the operator's, exactly as the lab's own lip-sync receipts
record ("machine; human pending").

**No cost row**, and the manifest says so in words for this lane too.

**Chaining is not claimed.** `soft_reference` means the planner will jump-cut
between segments on this lane, and that is the honest answer until something
measures the seam.
