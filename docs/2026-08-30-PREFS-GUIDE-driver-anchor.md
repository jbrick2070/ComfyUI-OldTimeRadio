# Driver anchor — "which ComfyUI preferences actually matter" guide

**Driver:** Claude (Cowork, 5080). **Written BEFORE fan-out**, from measurements
taken on two live machines today, so the panel is attacking real numbers rather
than a proposal.

**The proposed deliverable:** a short guide answering "when standing up OTR on a
new machine or a rented pod, which settings must be reproduced, and which are
noise?" Today a fresh pod ran the pack with **zero** settings reproduced, which
is what prompted the question.

---

## What was measured (CONFIRMED — these are facts, not claims)

**5080, local:** `user/default/comfy.settings.json` — 15,885 bytes, **51 keys**.

Grepping those 51 for anything render-adjacent (`vram|memory|cache|device|cuda|
precision|batch|preview|timeout|queue|backend|sampler|tile`) returns **four**:

    Comfy.Queue.History.Expanded      True
    Comfy.Queue.QPOV2                 False
    Comfy.Queue.ShowRunProgressBar    False
    VHS.AdvancedPreviews              Always

The other 47 are chrome — `Comfy.ColorPalette: light`, `Comfy-Desktop.AutoUpdate`,
`Comfy-Desktop.SendStatistics`, pip/torch install mirrors, and similar.

**Pod, RTX 5090:** `GET /api/userdata/comfy.settings.json` → **HTTP 404**. No
settings file exists. It nonetheless loaded all 25 `OTR_` node classes
(1036 → 1061 classes) after a git clone and a restart.

**What the pod DOES carry that the local box's settings file never mentions:**

    argv      ['main.py', '--listen', '0.0.0.0', '--port', '8188',
               '--enable-cors-header']            <- no --highvram on a 31.4 GiB card
    packages  comfy-aimdo 0.4.10                  <- the DynamicVRAM component that
                                                     native-aborted the 4060 in
                                                     PBUG-20260829-03
              comfy-kitchen 0.2.10
              comfyui-frontend-package 1.45.19
    torch     2.10.0+cu130                        <- matches the 5080 exactly
    ComfyUI   0.26.2

---

## The driver's claims — ATTACK THESE

**CLAIM 1 (INFERRED).** None of the 51 settings affect a headless API render.
The four that pattern-matched are queue-panel UI state and a VideoHelperSuite
*preview* toggle; a render submitted over `/prompt` never reads them.
*Weakest point:* I pattern-matched key NAMES. I did not trace any key to a
consumer. A setting could matter under a name my regex missed.

**CLAIM 2 (INFERRED).** The real levers are **launch args, env vars, and
package versions** — not `comfy.settings.json`. Evidence is that the pod ran
with no settings file at all, while the missing `--highvram` on a 31.4 GiB card
is a genuine difference.
*Weakest point:* "it ran" is not "it rendered correctly". No episode has yet
rendered on the pod, so this claim is proven only up to node registration.

**CLAIM 3 (INFERRED).** Copying `comfy.settings.json` between machines is
cargo-cult work that would feel productive and change nothing.
*Weakest point:* if any OTR node reads a ComfyUI setting, this is wrong.
**Unverified: I have not grepped the OTR codebase for settings reads.**

**CLAIM 4 (UNVERIFIABLE today).** `comfy-aimdo` matters more than every UI
preference combined, because it is the component that called native `abort()`
and killed a whole 4060 episode. But its behaviour on 31.4 GiB is untested.

---

## Questions for the panel

1. **Is CLAIM 3 false?** Does anything in `nodes/` read ComfyUI settings — via
   `/api/userdata`, `app.ui.settings`, `folder_paths`, or an extension API? A
   single reader makes the guide's core recommendation wrong.
2. **What is missing from the "real levers" list?** Candidates I have not
   checked: `extra_model_paths.yaml`, `--disable-smart-memory`,
   `--reserve-vram`, `PYTORCH_CUDA_ALLOC_CONF`, `HF_HOME`, attention backend
   selection, and whatever `comfy-kitchen` is.
3. **Should the guide recommend pinning `comfy-aimdo`, or disabling it?** It has
   one confirmed production kill (PBUG-20260829-03, native abort at Z-Image
   step 0/8 on 8 GB). Pinning a component that aborts the process is a real
   decision with more than one defensible answer.
4. **Is a prefs guide even the right artifact,** or should this be a machine
   provisioning checklist that happens to mention settings? The pod evidence
   suggests the latter.
5. **What would falsify CLAIM 1 cheapest?** Ideally one grep, not a render.

## THE ACCEPTANCE CRITERION (operator, 2026-08-30) — this outranks everything above

> *"I just don't want to brick someone with some setting we don't know it really
> does, but it only works on a 5080."*

**The failure mode this guide exists to prevent is not a useless recommendation.
It is a recommendation that is TRUE ON THE AUTHOR'S HARDWARE AND FALSE
ELSEWHERE.** A guide that omits a useful setting costs someone a slow render. A
guide that confidently recommends `--highvram` costs an 8 GB owner a dead
install, and they will have no idea it was our advice that did it.

This is `CLAUDE.md` §0B applied to documentation: a change for one machine must
prove the other machine is unchanged, **measured, not asserted**. The same rule
must bind what we write, not just what we ship.

Concretely — `--highvram` is being added to a 31.4 GiB pod today because it is
obviously right there. It is obviously WRONG on the 4060's 8 GB. If the guide
says "set `--highvram`" without binding it to VRAM, the guide is the bug.

**So every recommendation in the final guide must carry the hardware it was
proven on, and must say what happens on the other two.** The project now has
three real cards to test against:

    4060    8 GB     the portability floor — where a bad default bricks
    5080   16 GB     where every recommendation will be written
    5090   31.4 GB   rented, where "just give it more VRAM" looks free

**And the operator's own caveat stands: the proof is in the testing.** The panel
can only tell us whether the REASONING is grounded — whether anything actually
reads these settings. It cannot tell us whether a value survives different
hardware. Only running it on all three can, and no recommendation should ship
as advice until it has.

**Panel: treat any claim in this document that lacks a named card as
unsupported, and say so.**

## Out of scope

Story quality, the visual authored-path bug, PBUG-11, PBUG-20, and the
duration-tail work. This is exclusively about what a new machine must reproduce.
