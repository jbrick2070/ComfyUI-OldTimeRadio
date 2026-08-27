# Action-based prompting for moving video lanes -- OPERATOR SCOPE

Operator, 2026-08-27: *"we need to be sure all non audio-in moving video models
use action based prompting so we make the best of our render cycle -- no sense
in rendering a video that looks like a silly pan."* Then: *"also probably for
cloud lanes too."* Then: *"actually all the cloud should have motion too, they
can handle the load."*

**THE PROBLEM, MEASURED.** On the published foley episode
`signal_lost_ink_and_martyrdom_20260827_071626`, **8 of 12 character beats** fell
through to the deterministic placeholders `restrained visible reaction, subtle
natural body motion, stable mid-shot`, and only 2 beats got a usable writer
directive. That is not a regression introduced by the prompt-policy fix -- the
leg BEFORE it missed 12 of 13 -- but the fallbacks are now generic filler where
the old ones at least carried the beat's own text. Two thirds of a render cycle
is currently driven by a fixed string that asks for as little motion as English
can express.

**THE M4 ASK IS PART OF THE PROBLEM.** `_build_nonverbal_batch_prompt`
(`nodes/otr_shot_lock.py`) currently asks for *"a restrained facial expression"*.
It steers the writer toward stillness on lanes whose entire value is motion.

## THE SCOPE, BY LANE

The predicate is the engine's FAMILY, not its name and not where it runs.

### A. MOVING, NOT AUDIO-IN -- action REPLACES the dialogue

These already take the nonverbal branch. They need the ask and the fallbacks
turned action-forward.

| lane | family | where |
|---|---|---|
| `minimax_h3_video` | image_to_video | local |
| `ltx25_video` | image_to_video | local |
| `ltx25_foley_plus` | image_to_video | local -- CONFIRM |
| `ltx25_mime` | image_to_video | local |
| `ltx_video` | text_to_video | local |
| `ltx_8gb` | image_to_video | local |
| `wan_ti2v` | image_to_video | local |
| `fastwan_8gb` | image_to_video | local |
| `animatediff15_v3_haunted_video` | text_to_video | local |
| `mesh_stage` | image_to_video | local -- CONFIRM |
| `cloud_vidu_q2_pro_fast_720p` | image_to_video | cloud |
| `cloud_wan_i2v` | image_to_video | cloud |
| `google_omni_video` | text_to_video | cloud |
| `google_veo_video` | text_to_video | cloud |

### B. AUDIO-IN CLOUD -- action is ADDED ALONGSIDE the dialogue

**This is a different code path and must not be confused with A.** These lanes
are driven by the audio; the dialogue is what the mouth is doing and it must
NOT be removed. Motion direction is added for the BODY and the CAMERA.

| lane | family |
|---|---|
| `cloud_kling_avatar` | audio_driven_face |
| `cloud_seedance_2` | audio_conditioned_video |
| `cloud_wan_i2v_audio` | audio_conditioned_video |

**OPEN QUESTION FOR THE OPERATOR:** the LOCAL audio-in lanes (`humo`,
`humo_1.7B`, `humo_1.7B_169`, `humo_14B_169`, `ltx_audio_in`,
`minimax_h3_audio_in`) are the same shape as B. The operator said "all the
cloud", so B is cloud-only as written. If the intent was "every audio-in lane
gets body/camera motion alongside its dialogue", say so and B grows by six.

### C. EXCLUDED, and why

* `still_flat`, `still_pan`, `still_motion` -- a pan or a hold. No generative
  motion for a prompt to direct.
* `still_word` -- the words ARE the picture; it is the one non-audio lane that
  deliberately keeps the dialogue.
* `viz_camera`, `viz_green`, `viz_mxc_cpu`, `viz_mxc_mandala` -- abstract
  audio-reactive visualizers with no subject to act.
* `word_razzle` -- `roles=()`, so it can never serve a character beat, and its
  subject is lettering on a period poster, not a performer.

## WHAT THIS IS NOT

**It is NOT the fix for the inaudible foley bed.** Both panel lanes were
explicit: the bed sits ~37-58 dB under the master, and even the two loudest
stems are 37-38 dB down, so no amount of richer action closes that gap. The bed
level is a separate item requiring an amendment to RULING 2. Better action makes
there be more foley WORTH hearing; it does not make the existing foley audible.
Keep the two items apart or one will be used to excuse the other.

## OPERATOR RULINGS, 2026-08-27 (these settle the scope)

1. **`ltx25_foley_plus` IS IN, and needs BOTH.** *"foley plus needs action of
   course and it needs foley sound prompting however it meets the model's
   requirements."* Action for the picture AND matched sound wording for the
   audio half -- one positive string serves both on LTX 2.5.
2. **`mesh_stage` is OUT.** *"that just does a rotating 3d model, there's no
   action."* Its motion is a Blender camera orbit the prompt does not direct.
3. **EVERY lane, local AND cloud, AUDIO-IN INCLUDED.** *"when I said cloud I
   meant all cloud lanes inc cloud audio in should take better action. I was not
   changing my direction on local -- local needs a lot more action too."* So the
   A/B split below is about MECHANISM, not about who is in scope: everyone is in
   scope.
4. **The expectation:** *"every video lane should have appropriate motion
   prompting per its spec baked in by now."*

## THE AUDIT AGAINST RULING 4 -- it is NOT baked in for people

The motion plan landed for the RADIO CONSOLE and never reached the CAST.

| lane | baked-in default today | verdict |
|---|---|---|
| announcer / music bookends | pack-owned `VisualStyle.motion_registers` -- *"Tuning dial needle sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille trembles. Dust motes drift."* (`render_driver.py:1327,1469`) | REAL motion, per pack |
| `humo`, `humo_1.7B`, `humo_1.7B_169`, `humo_14B_169` | `"a person speaking, subtle facial motion"` (`eng_humo.py:831`) | DAMPED -- says *subtle* |
| `minimax_h3_video`, `minimax_h3_audio_in` | `"subtle natural motion, cinematic light"` (`eng_minimax_h3.py:334`) | DAMPED -- says *subtle* |
| `ltx25_video`, `ltx25_foley_plus`, `ltx25_mime` | `"a vintage radio broadcast scene"` (`eng_ltx25.py:993`) | NO motion -- a scene noun |
| `ltx_audio_in` | `"a vintage radio broadcast scene"` (`eng_ltx_av.py:858,1001`) | NO motion -- a scene noun |
| ShotLock nonverbal fallback (any non-audio character lane) | `"restrained visible reaction, subtle natural body motion, stable mid-shot"` | DAMPED -- says *restrained* and *subtle* |
| LTX character append (`render_driver.py:3025-3029`) | `"stable centered subject, full face clearly visible, generous headroom"` | ANTI-motion -- it asks for STABLE and CENTERED |

**Four of the character defaults contain the literal word "subtle" or
"restrained", and two contain no movement language at all.** On the LTX
character path the driver then appends *stable centered subject*, which actively
asks for stillness. That is the "silly pan" the operator is seeing, and it is
authored, not emergent.

## THE SEPARATE, ALREADY-BUILT MACHINERY -- and why it is off

`nodes/_otr_motion_clause.py` derives a per-beat motion clause from the beat's
dialogue and cast (`docs/2026-07-29-motion-floor-brief.md`; 2026-08-17 kinetic
amendment opened its cap 70 -> 130 chars because *"the cap was itself a damping
force"*). It is opt-in on `OTR_LTX_MOTION_CLAUSE`, **default OFF**
(`_otr_motion_clause.py:16,30`; consumer `otr_video_render_batch.py:578-599`),
set in no profile, launcher or workflow, and was NOT active on the foley leg.

Operator's account of why: *"it's default off because we did a hardcoded plan to
add motion to almost everything, I thought."* The audit above says that hardcoded
plan covered the bookends and stopped there. So there are two candidate paths --
turn the clause on, or fix the baked-in per-lane defaults -- and they are not
mutually exclusive. Decide deliberately rather than doing both by accident.
