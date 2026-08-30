# SPEC — a HAUNTED lane that takes an image

**Status: SPEC ONLY. No code written. Awaiting the review arc.**
Every fact below was measured on the 5080 on 2026-08-30, not recalled. Where
something is unverified it says so.

---

## 1. What this is

A **new sibling lane** — `animatediff15_v3_haunted_i2v` — that animates *from a
still* instead of from text alone, on the AnimateDiff haunted stack.

**It is a NEW lane. The golden and haunted text-to-video lanes are untouched.**
That is not caution for its own sake: the existing ladder is deliberately
additive, and each rung was added beside its parent rather than inside it —

```
GhostSignalEngine            eng_ghost_signal.py:290          mm-p_0.5.pth
  └─ GhostSignalV3Engine     eng_ghost_signal_official.py:72  v3_sd15_mm.ckpt
       └─ GhostSignalV3HauntedEngine  :120                    + v3_sd15_adapter
```

The haunted lane's own docstring says a sibling "sets all three together (G1.3 —
a per-artifact constant travels WITH the lane that owns it, never as a module
constant a sibling silently inherits)". This spec follows that rule exactly.

## 2. Why it is worth building

**Image-conditioned video on the cheap lane.** Today the only image-to-video
lanes are expensive: MiniMax H3 is **~39.6 GB** of weights and LTX 2.5 is
**gated**. The haunted stack is **~3.9 GB** and fully ungated. Adding a still
input to it would give image-to-video on an 8 GB card with weights a user can
obtain without a token.

**And the still lane already exists.** Six engines already declare
`family = "image_to_video"` — `eng_minimax_h3`, `eng_ltx25`, `eng_ltx_8gb`,
`eng_ltx_video`, `eng_mesh_stage`, `eng_cloud_video`. The pipeline already
generates and hands over a first frame for those. **This lane subscribes to
that existing contract rather than inventing one.**

## 3. The contract

Measured from `eng_minimax_h3.py`, the reference implementation:

```python
family          = "image_to_video"
required_inputs = ("init_image",)
accepts_still   = True
```

against the haunted lane today (`eng_ghost_signal.py`):

```python
family          = "text_to_video"
required_inputs = ("text_prompt",)
optional_inputs = ()
accepts_still   = False
```

So the new lane declares the H3 shape and inherits everything else — motion
module, domain adapter, receipt, cadence, canvas — from
`GhostSignalV3HauntedEngine`.

**Open contract question for the panel:** should `init_image` be REQUIRED (H3's
choice, a hard dependency on the still lane) or OPTIONAL with a text-only
fallback? Required is simpler and matches the reference; optional makes the lane
usable when no image model is configured, which is the haunted lane's current
selling point. These lead to different profiles and different failure modes, and
I do not think the answer is obvious.

## 4. Two routes to actually consuming the image

### Route A — latent init (img2img). NEEDS NOTHING NEW.

Encode the still, repeat it across the batch, sample at `denoise < 1.0` so the
motion module moves a picture that is already there.

Verified present on this box, no new pack required:

```
VAEEncode          present
RepeatLatentBatch  present
KSampler           present
ImageScale         present
```

* **Pro:** ships today. No new dependency, no new download, no version bump of
  anyone else's pack. The haunted lane's install story is unchanged.
* **Con:** the weakest form of conditioning. A low denoise holds the image but
  suppresses motion; a high denoise moves well but drifts off the still. The
  usable band has to be found by eye and there is no guarantee it is wide.
* **The dial:** `denoise`, plus the existing
  `OTR_GHOST_HAUNTED_LORA_STRENGTH` (frozen at 1.0 and, per its own comment,
  **never swept by eye**).

### Route B — SparseCtrl. NEEDS AN ADE UPGRADE. **Blocked today.**

SparseCtrl is AnimateDiff v3's own image-conditioning module — built for exactly
this, and strictly better than latent init.

The weights are ungated and sit in the SAME repo the lane already pulls from:

```
guoyww/animatediff   v3_sd15_sparsectrl_rgb.ckpt        1.99 GB
                     v3_sd15_sparsectrl_scribble.ckpt   1.99 GB
```

**But the nodes to drive them are not installed.** Measured against the live
server: **143 `ADE_` classes, ZERO SparseCtrl among them.** Installed pack is
`ComfyUI-AnimateDiff-Evolved` at `9257651`, version **1.6.0**, 2026-07-28.

* **Pro:** the correct mechanism; designed for keyframe conditioning.
* **Con:** requires upgrading a third-party pack that the SHIPPING 8 GB profile
  depends on (PBUG-20260829-09). An ADE upgrade is a change to the one external
  dependency our only proven 8 GB lane rests on, and the 4060 is the only box
  that can prove it still works afterwards.
* **UNVERIFIED:** which ADE version first ships SparseCtrl nodes, and whether
  that version still supports the v3 loader path this lane uses. Nobody has
  checked. That is a prerequisite, not a detail.

**Recommendation for the arc to pressure-test:** build Route A first because it
is free and reversible, keep the engine's image-consumption behind one seam, and
treat Route B as a later swap behind that same seam — not as a fork.

## 5. Weights and install

Route A adds **nothing**. The lane uses the artifacts already pinned:

```
v1-5-pruned-emaonly-fp16.safetensors   2.0 GB   checkpoints
v3_sd15_mm.ckpt                       1.67 GB   animatediff_models
v3_sd15_adapter.ckpt                  0.10 GB   loras
```

Route B adds 1.99 GB, ungated, from a repo already in the fetcher.

**The still itself comes from the existing image lane** — `z_image_turbo` is
ungated, auto-downloading, and already named in the 8 GB profiles' image roles
where it currently sits inert because the haunted lane refuses stills.

## 6. What this touches

| surface | change |
|---|---|
| `eng_ghost_signal_official.py` | NEW class beside the haunted one; no edit to existing classes |
| `_otr_video_engines/registry.py` | one registration |
| `_otr_shared/public_engines.py` | one row |
| `content_oracle.py` | family mapping — `image_to_video`, not `text_to_video` |
| profiles | a NEW profile; existing ones untouched |
| `otr_canonical.json` | **nothing** unless a profile is promoted to default |

## 7. Risks, stated plainly

1. **The still may fight the look.** The haunted lane's character comes from the
   v3 domain adapter at strength 1.0 — deliberately grimy. A clean `z_image_turbo`
   still may be dragged toward that look, or may fight it and produce neither.
   **This is the risk most likely to make the lane not worth shipping**, and it
   cannot be settled by reading code — it needs frames looked at by eye.
2. **Route A's denoise band may be too narrow to be useful.** Same nature: an
   empirical question, cheap to answer, not answerable in advance.
3. **An ADE upgrade (Route B) endangers the proven 8 GB lane.** Only the 4060
   can prove otherwise.
4. **A second image consumer changes VRAM shape on 8 GB.** The haunted lane's
   headroom today assumes no image model is resident. `z_image_turbo` plus SD1.5
   plus the motion module has not been measured together on 8 GB.

## 8. What NOT to do

* Do not add `accepts_still` to the existing haunted lane. Its docstring names
  the exclusions — "no IPAdapter, no ControlNet, no reference folder" — as
  deliberate, and the golden lane is the reference a comparison is measured
  against.
* Do not upgrade AnimateDiff-Evolved to chase Route B before Route A has shown
  the look is worth having.
* Do not promote this into `otr_canonical.json` until an episode has published
  on it. A dropdown row is a promise the model will load.

## 9. Open questions for the arc

1. `init_image` required or optional? (§3)
2. Route A now vs waiting for Route B — is a weaker mechanism that ships today
   better than the right one that needs a dependency bump?
3. One seam for image consumption, or two engines?
4. Does the haunted look survive a conditioning still at all? — the question
   that decides whether any of this is worth building.
