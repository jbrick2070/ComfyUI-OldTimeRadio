# Item 3b -- every OTHER video lane's prompt, audited

**Operator, 2026-09-02:** *"once you are done with this it's worth reviewing all
other lane video prompts too to see we aren't stuffing too much character
description; we should have real action and motion that applies to the action at
hand unless it is a TTS speech video that can't handle it"* -- and, on budgets:
*"obviously AnimateDiff can't take that much text, other video models can, so
maybe it's OK they have more words."*

This is the read-only audit. No code changed. It runs after Prompt v3 Half A is
judged on the Ghost lane, because the same shape either works there or it does
not, and there is no reason to change eleven lanes on a theory.

---

## 1. Every lane shares ONE assembly, and it leads with the face

`motion_common.compose_parts` orders the leaves
**`appearance, setting, expression, motion[, camera]`**
(`nodes/_otr_video_engines/motion_common.py:851-886`). Ten of the eleven lanes
that own a formatter call it or its LTX 2.5 sibling `_ltx25_parts`
(`eng_ltx25.py:2540-2570`); the eleventh (`humo*`) builds its own envelope.

`appearance` is not a short tag. The render driver resolves it from the cast row
through `_appearance_for_char`, which reads `portrait_prompt` / `appearance` /
`description` / `character_description`
(`render_driver.py:3128-3140`, `otr_shot_lock.py:135`).

**Measured on the last published episode ("The Faded Ledger", tonight 21:08):**

| cast row | words in the appearance leaf |
|---|---|
| c01 announcer | 13 |
| c02 archivist | **83** |
| c03 consultant | **51** |

That 83-word leaf opens with *"40s, meticulous film archivist. Face:
heart-shaped, heavy-lidded almond eyes with faint periorbital lines, thin
straight nose, set jaw..."* and it is the FIRST thing in the prompt, on every
beat that character appears in.

**On `wan_ti2v` that is 83 of a 100-word cap** (`WAN_MAX_WORDS = 100`,
`eng_wan_ti2v.py:1371`, self-checked and hard-truncated at
`:1395-1398`). Eighty-three per cent of the lane's entire prompt budget is a
face, leaving seventeen words for the setting, the expression, the motion and
the camera **combined** -- and anything past 100 is chopped mid-sentence, so the
camera clause is the part that silently disappears. That is the operator's
sentence about the Ghost lane, reproduced on a different lane with a hard cap
doing the damage.

## 2. The roster, and which lanes the finding applies to

Every registered video engine, its family, and whether it owns a formatter
(generated from the live registry):

| lane | family | own formatter | appearance in the prompt? |
|---|---|---|---|
| ltx25_foley_plus | image_to_video | yes | **NO -- already dropped** |
| ltx25_mime | image_to_video | yes | **NO -- already dropped** |
| ltx25_video | image_to_video | yes | yes |
| ltx_8gb | image_to_video | yes | yes |
| ltx_video | image_to_video | yes | yes, deliberately (see 4) |
| fastwan_8gb | image_to_video | yes | yes |
| wan_ti2v | image_to_video | yes | yes, inside a 100-word cap |
| minimax_h3_video | image_to_video | yes | yes |
| google_omni_video | text_to_video | yes | yes -- load-bearing (see 4) |
| google_veo_video | text_to_video | yes | yes -- load-bearing (see 4) |
| humo, humo_1.7B, humo_1.7B_169, humo_14B_169 | audio_driven_face | yes | exempt (see 3) |
| minimax_h3_audio_in | audio_conditioned_video | yes | exempt (see 3) |
| ltx_audio_in | audio_conditioned_video | no | exempt (see 3) |
| animatediff15_v3_haunted_video | text_to_video | no | Ghost lane -- item 3 proper |
| animatediff15_v3_stillin_lab_video | text_to_video | no | Ghost lane -- item 3 proper |
| cloud_* (5 lanes), mesh_stage, still_*, viz_*, word_razzle | various | no | no formatter; out of scope |

## 3. The exemption the operator named, made precise

*"unless it is a TTS speech video that can't handle it."* The code already has an
exact test for this and it is not a hand-written list: a lane receives the spoken
line when its FAMILY is in `AUDIO_IN_FAMILIES`
(`audio_driven_face`, `audio_conditioned_video` -- `mouth_policy.py:62`), read by
`_lane_preserves_dialogue` (`otr_shot_lock.py:929-957`).

Those lanes are exempt from any motion increase, and their reasons are recorded
per lane rather than assumed:

* **`minimax_h3_audio_in`** -- PASS at 5.17s on seed 43 and the IDENTICAL prompt
  FAILED on seed 42; its recorded failure is the jaw and collar reshaping as the
  head approaches yaw. One weight shift, one small head tilt, stop before
  profile.
* **`humo*`** -- the documented first failure is hands or props near the face.
  A tiny push may REPLACE the lean; never accompany it.

**These lanes keep their appearance leaf too.** They are lip-syncing a face, and
the face is the subject.

## 4. Two lanes where appearance is genuinely load-bearing

* **`google_omni_video` and `google_veo_video` are `text_to_video`.** There is no
  conditioning still, so the appearance leaf is the ONLY identity anchor. It
  stays.
* **`ltx_video` restates it on purpose**, and its own comment says why: *"do not
  assume the init still is always present, because T2V remains a supported path
  on this lane."* That is a real reason, but it is a per-BEAT fact, not a
  per-lane one. The honest fix here is conditional -- carry the appearance when
  the request has no init image, drop it when it does -- rather than either
  blanket answer.

## 5. The precedent is already in the tree, and it was proven live

`ltx25_foley_plus` and `ltx25_mime` already drop the leaf, via
`_ltx25_parts(include_appearance=False)`, and the reasoning at
`eng_ltx25.py:2547-2560` generalises exactly:

> identity is already carried by the conditioning STILL, whose scene_character
> row mints the face unobstructed, so the text was redundant as well as harmful

Those two lanes had a second, sharper reason -- the joint latent SPEAKS the
prompt, so a `character_description` beginning "30s, Queen of the Fairies"
rendered a woman saying *"Queen of the Fairies"* out loud (proven live
2026-08-28). The silent I2V lanes have no mouth to protect, which is why they
kept it.

**But redundancy alone is enough of a reason.** Every `image_to_video` lane is
conditioned on a still that already shows the face. Restating 83 words of that
face in the text buys nothing and, on `wan_ti2v`, costs the camera clause.

## 6. What the audit proposes (to run AFTER Half A is judged)

1. **Drop `appearance` on the silent I2V lanes** -- `ltx25_video`, `ltx_8gb`,
   `fastwan_8gb`, `wan_ti2v`, `minimax_h3_video` -- exactly as the foley and mime
   siblings already do, and for the recorded reason: the still carries identity.
2. **Make `ltx_video` conditional on the init image** rather than unconditional,
   so its genuine T2V path keeps the anchor and its I2V path stops paying for it.
3. **Leave the appearance leaf alone** on the two Google text-to-video lanes and
   on every audio-in lane.
4. **Spend the recovered words on the crux**, the same kernel Prompt v3 builds
   for the Ghost lane -- the story's own object in the story's own setting.
   On `wan_ti2v` that is 83 words of budget handed back inside a 100-word cap.
5. **Per-lane budgets stay per-lane**, per the operator: AnimateDiff's window is
   77 SD1 tokens and the LTX/WAN lanes take far more prose. Nothing here
   proposes one shared length.

**None of this is done yet, and none of it should be until Half A has been seen
on screen.** The measurement is the deliverable; the change follows the eye.
