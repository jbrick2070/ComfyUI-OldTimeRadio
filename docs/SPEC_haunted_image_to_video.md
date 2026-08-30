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

---

## 10. REVIEW ROUND 1 — Antigravity, grounded 2026-08-30

Reviewed by `agy` against the real files; every claim below was then verified by
me against those files before being folded in. Its verdict: **do not build as
specified — run the disqualifying probe first.** I accept that.

### Errors it found in this spec, all CONFIRMED by re-reading the code

1. **`required_inputs = ("init_image",)` IS WRONG AND WOULD CRASH AT RENDER.**
   `GhostSignalEngine._assert_required_inputs` (`eng_ghost_signal.py:727`) hard-
   requires BOTH `text_prompt` and `negative_prompt`, with the comment "Ghost
   owns its whole subject, so an empty prompt is a blank picture, not a
   default". Declaring only `init_image` lets the compat layer admit a textless
   call that then dies mid-render.
   **Correct: `required_inputs = ("text_prompt", "init_image")`.**

2. **"Six engines already declare image_to_video" was wrong.** It is **12
   declarations across 8 files** — I counted files with `grep -l` and reported
   them as engines. `eng_wan_ti2v.py` and three separate engines inside
   `eng_cloud_video.py` were missed.

3. **File attribution in section 3 is loose.** The live haunted lane is
   `GhostSignalV3HauntedEngine` in `eng_ghost_signal_official.py:120`. The
   VALUES I quoted are correct — verified that the subclass overrides none of
   `family`, `accepts_still`, `required_inputs` or `still_plan`, inheriting all
   four from the base — but they should be attributed to the base class.

### The surface I missed that would have SILENTLY produced nothing

**`still_plan` is a required class attribute and mine would have been empty.**
`GhostSignalEngine` declares `still_plan: tuple = ()` (`:351`) because it needs
no stills. `eng_minimax_h3` declares `still_plan = _H3_STILL_PLAN` (`:453`).
**An I2V engine with an empty `still_plan` causes the image dispatcher to mint
ZERO stills** — so the lane would register, load, and render text-to-video under
an image-to-video receipt. That is the worst failure shape in this codebase: a
receipt that lies. Section 6's table missed it entirely.

> **CORRECTED IN ROUND 2 (Codex, grounded, and I checked the call path myself).**
> The *conclusion* survives; the *mechanism* above is wrong, and I folded it in
> without verifying it. `engine_consumes_still` in
> `nodes/otr_image_gen_dispatcher.py` reads `accepts_still` FIRST and returns it
> whenever it is not `None`; `required_inputs` is only the fallback for engines
> that predate the flag. So a class declaring `accepts_still = True` gets its
> stills minted no matter what `still_plan` says — an empty plan does NOT zero
> out the dispatcher. What an empty plan *does* do is lie in the other
> direction: `still_plan_helpers.py` defines `()` as a real declaration meaning
> "this lane needs no images", so shipping it would misstate the lane's pixel
> contract and skip the plan-authored kind/identity behaviour. The sibling still
> needs a truthful non-empty scene plan. Same fix, different reason.

### Section 6's table was too short. VERIFIED additional surfaces:

| surface | why it is not optional |
|---|---|
| `registry.py` `CAPABILITIES` dict | `tests/test_capability_profiles.py:393` asserts `set(vreg.CAPABILITIES) == set(vreg.all_engine_names())`. Registering without it FAILS CI. |
| engine class `still_plan` | non-empty `StillPlanRow` entries, or no stills are minted (above) |
| `tests/fixtures/still_plan_matrix.json` | parity test; needs regeneration |
| `public_engines.py` `_PUBLIC_LABEL` | 1:1 bijection assert |
| `content_oracle.py` `_FAMILY_FALLBACK` | bare-script family resolution |
| `docs/ENGINE_MATRIX.md` via `tools/engine_matrix.py` | `test_engine_matrix_doc` asserts every registered engine is documented |
| frame-contract tests | sweep new engines automatically |

### Its answers to the four open questions

1. **`init_image` REQUIRED**, not optional — a silent I2V→T2V fallback would
   produce unconditioned video under an I2V receipt and mask image-dispatcher
   failures. The prompt-only lane already exists as a separate dropdown row, so
   the fallback buys nothing. **Adopted.**
2. **Neither route should be committed today.** Route A is weak conditioning;
   Route B needs an ADE bump that endangers the only proven 8 GB lane.
   **Adopted** — see the probe below.
3. **Two engines, not one seam.** Recipes stamp receipts, and hiding SparseCtrl
   behind Route A's engine id would make the receipt lie about what conditioned
   the pixels. This reverses my recommendation, and it is right: the receipt
   argument is the same one that governs the haunted-vs-clean lane split.
   **Adopted.**
4. **The disqualifying probe** (its design, and it is cheap): one still from
   `z_image_turbo`, `VAEEncode` → `RepeatLatentBatch(16)` → `KSampler` with the
   v3 motion module + adapter at strength 1.0, at denoise **0.35 / 0.50 / 0.65 /
   0.80**, decoded to four clips. Disqualified if ≤0.50 shows zero macro-motion
   with texture boil, or ≥0.65 loses the still's identity within 2–3 frames.

### Its technical hypothesis for WHY Route A likely fails — INFERRED, not verified

Repeating one still latent across the window gives the motion module's temporal
attention identical cross-frame keys and queries, which suppresses motion
trajectory while leaving high-frequency texture free to boil. SD1.5 AnimateDiff
has **no image-conditioning channel** without SparseCtrl or IP-Adapter, unlike
Wan/LTX which condition through DiT patches.

**This is reasoning, not measurement, and I am recording it as such.** But it
predicts exactly the failure the probe is designed to detect, which is a good
sign for both.

### Where this leaves the spec

**Route A is now a PROBE, not a build.** Write the standalone probe script,
look at four clips, and only then decide whether a lane exists at all. That
inverts the spec's original recommendation and is better: the question that
decides everything — does the haunted look survive a conditioning still — is
answered by four clips and an eye, not by an engine.

## 11. REVIEW ROUND 2 — Codex, grounded 2026-08-30

Independent end-to-end read of this spec at commit `07a47008`. I verified its
decision-changing claims against the real files before folding; what follows is
what survived, not what was asserted. Its engine roster matched the live
`_VIDEO_REGISTRY` name-for-name, which is why I trust the rest of it.

### 11.1 The central premise of this spec was false

**Route B was never blocked by an AnimateDiff-Evolved upgrade.** This spec chose
Route A (latent init) largely to avoid disturbing the proven ADE 1.6.0 install
the 8 GB haunted lane depends on. That trade does not exist. SparseCtrl is not
ADE's to ship — ADE's own README, line 6, names `ComfyUI-Advanced-ControlNet`
and says it *"Includes SparseCtrl support."* Verified locally: the only file in
the installed ADE tree that even mentions SparseCtrl is its README, and **ACN is
not installed here at all.**

So the real Route B cost is *installing one new node pack*, not upgrading a
working one — materially cheaper and lower-risk than this spec assumed, and it
removes the main argument for preferring a mechanism AnimateDiff v3 does not use
for image conditioning. Route A stays in scope only as a cheap hypothesis probe;
it is not the recommended build.

**Corollary:** matching version numbers are not compatibility proof. ACN's
DinkLink layer checks an ADE boundary and warns the cross-pack API can change,
so ACN + this exact ADE commit still has to be qualified, not assumed.

### 11.2 The blocker is upstream of both routes, and it is already logged

Neither route has an 8 GB product path today, and the reason has nothing to do
with video conditioning. **PBUG-20260829-03**: on the 4060, activating the image
role native-aborts the ComfyUI process at the Z-Image sampler's step 0/8 —

    Fatal Python error: Aborted

CUDA error 2 (out of memory) while ComfyUI's DynamicVRAM streamed the 6.2 GB
Z-Image UNET onto a card still holding OTR's residents. This is worse than an
ordinary OOM: it calls native `abort`, so OTR never gets to catch it, write a
receipt, or degrade. Its own amendment records that the shipping haunted profile
survives *only* because its text-to-video engine never activates the image role.

An image-to-video haunted lane activates exactly that role. **Until the canonical
image phase can generate and reclaim its Z-Image bundle on 8 GB, this lane has no
4060 story regardless of how good its conditioning is.** That measurement comes
first, before any engine work, and an actual OOM decides it — no estimate gate.

### 11.3 Contract corrections

* **`required_inputs = ("text_prompt", "init_image")`.** Round 1 was right that
  `("init_image",)` alone would crash. Round 2 adds why the obvious repair is
  also wrong: `negative_prompt` **cannot** be added to `required_inputs` — it is
  not in the closed request-token vocabulary in `schemas.py`. Its non-blank
  requirement stays a renderer invariant, enforced inside the Ghost renderer.
* **H3 is not a template to copy.** It gets away with `("init_image",)` because
  it owns a prompt fallback and a dedicated first-frame conditioner. Ghost has
  no text fallback, so it cannot inherit that shape.
* **Cache identity is prompt-only today.** `shot_cache_identity` hashes prompt,
  negative prompt, shot/seed/canvas and artifacts — **not** the init image and
  **not** denoise. An I2V identity that treats two different stills as the same
  request is a false contract and must not ship.

### 11.4 Surfaces section 6 missed

Verified present in the repo and enforced by tests:

* `workflows/variants/<profile>.json` **and** `<profile>.launch.md` — generated
  for every committed profile (70 launch files exist today).
* `docs/ENGINE_MATRIX.md` — regenerate via `tools/engine_matrix.py`; parity is
  enforced.
* `tests/fixtures/still_plan_head_parity.json` — regenerate the roster.
  (Round 1 named `still_plan_matrix.json`; **no such file exists.**)
* `tests/test_frame_contract.py` — its unbounded-engine set is a literal roster,
  not automatic.
* `scripts/otr_w45_campaign.py` — every new local engine joins the roster or
  gets a deliberate `PROFILE_EXCEPTIONS` mapping.
* `docs/evidence/video_evidence_manifest.json` — needs an explicit
  `admission_unenforced` statement until a real measurement lands. No estimated
  cost may ever kill a render.
* `nodes/_otr_video_engines/registry.py` — a `CAPABILITIES` row is required;
  `@register` on the class does the registration. This spec's "one registration"
  wording was inaccurate.
* `workflows/otr_canonical.json` needs **no** structural edit while this stays an
  internal engine/profile addition. A new widget would immediately drag in
  canonical + `widget_mapping.json` + all variants as same-change edits.

### 11.5 Factual corrections to earlier sections

* **12 effective `image_to_video` engines, not six.** Verified against the live
  registry: `cloud_vidu_q2_pro_fast_720p`, `cloud_wan_i2v`, `fastwan_8gb`,
  `ltx25_foley_plus`, `ltx25_mime`, `ltx25_video`, `ltx_8gb`, `ltx_video`,
  `mesh_stage`, `minimax_h3_video`, `wan_ti2v`, `word_razzle`.
* **The golden and clean-v3 Ghost lanes are not selectable siblings.** They are
  unregistered/tombstoned bases kept as inheritance references.
  `animatediff15_v3_haunted_video` is the one surviving AnimateDiff lane.
* **Haunted adapter strength is not frozen at 1.0** —
  `OTR_GHOST_HAUNTED_LORA_STRENGTH` overrides it, and no value has been
  qualified by eye.
* **"SparseCtrl is strictly better" is not established.** It is the purpose-built
  mechanism; comparative quality on *this* haunted stack is empirical.
* **Upstream warns about exactly our plan:** AnimateDiff recommends the animated
  image come from the same SD 1.5 model used for animation. A Z-Image still fed
  to a haunted SD 1.5 checkpoint violates that, and cross-model subject drift may
  dominate even when SparseCtrl works correctly.

### 11.6 Revised recommendation

1. **Measure the image phase on the 4060 first** (PBUG-20260829-03). Clean
   server, real 512x288 canonical beat; record baseline / post-writer reclaim /
   Z-Image peak / post-decode reclaim / video-stage peaks. If it still aborts,
   stop — there is no 8 GB lane to build.
2. **Then qualify ACN 1.6.0 in isolation** through the real
   `workflows/otr_canonical.json`, on an unregistered probe path with no
   selectable profile row.
3. **Then build the SparseCtrl lane** as its own engine id with its own recipe,
   receipt and evidence row — never swapped in behind Route A's identity, which
   would make old renders unreproducible.
4. Route A's 0.35/0.50/0.65/0.80 denoise sweep remains valid, but it can only
   falsify Route A. It says nothing about whether the purpose-built mechanism is
   worth building.

**Both reviewers agree on the one thing that matters: do not build this as
originally specified.** Round 1 said run the cheap experiment first. Round 2
says that experiment is aimed at the wrong stage — measure the image phase
before touching the video engine at all.
