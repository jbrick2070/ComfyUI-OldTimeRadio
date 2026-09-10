# Per-device shipping JSONs -- proposed recipes, PENDING OPERATOR APPROVAL

**NOTHING HERE HAS BEEN APPLIED.** `workflows/otr_canonical.json` is untouched.
This is the reviewable form of a per-device assembly pass run 2026-09-09: four
no-friction combinations, each adversarially verified, three of them corrected
on that verification. It exists so the operator can approve or reject exact
widget values rather than a description of them.

**The goal these serve, in the operator's words:** publish "no friction
mac/amd/nvidia jsons" -- one graph per device whose dropdowns already hold a
combination that works there and needs no manual download, no licence click and
no token.

**No friction is a hard filter, not a preference.** Auto-download is fine.
`HF_TOKEN`, a licence click, a manual fetch, an installer script or a
third-party node pack all disqualify a slot. GGUF is out by operator ruling --
it does not auto-download.

---

## The prerequisite that applies to EVERY one of them

**`ffmpeg` AND `ffprobe`, and pip cannot supply the second.**
`pip install -r requirements.txt` brings `imageio-ffmpeg`, which carries an
ffmpeg binary and **not** ffprobe. Without ffprobe the still lanes fail loud at
`validate_silent_clip_contract`. So every device needs a real ffmpeg install
(`brew install ffmpeg`, a distro package, or `ffdl install`) -- README.md:86.

This is not any one combination's fault and it is not avoidable by picking
different engines. It is the one hand step in every "no-friction" JSON, and it
should be stated on the tin rather than discovered.

## A MONOSPACE FONT, and on Linux that is not automatic

`video_engine._load_font` measures text by opening a font FILE by absolute
path; `_otr_captions` then hands libass a family NAME to draw with. Those are
two different resolvers, joined only by the arithmetic that centres the title
(`x = centre - tw // 2`). When the measurement side finds nothing it falls back
to PIL's bitmap default, which ignores the requested size -- so the title is
still DRAWN at full size, at a position computed from a width roughly ten times
too small, and lands off the right edge of the frame.

**That is not hypothetical: it shipped on eight published macOS episodes and was
fixed 2026-09-09.** `f0aa8e6a` added the macOS branch -- the non-Windows list
held only two Debian/Ubuntu paths, so macOS matched nothing at all. The
follow-up added the Fedora/Arch/openSUSE layouts plus a bare-name tier that
lets PIL walk the platform font directories itself, so any distribution
carrying DejaVu or Liberation now resolves whatever layout it uses.

* **Windows / macOS: nothing to do.** `consola.ttf` and `Menlo.ttc` ship with
  the OS.
* **Linux, including the ROCm lane: verify it, do not assume it.** A MINIMAL or
  container image may carry no fonts at all. Install `dejavu-sans-mono` (or
  `liberation-mono`) if the render logs `no monospace TTF found`, or point
  `OTR_VIDEO_FONT` at a TTF path directly.
* **If you set `OTR_CAPTION_MONO_FONT`, set `OTR_VIDEO_FONT` to the matching
  file.** The first changes what libass draws; only the second changes what PIL
  measures. Setting one alone re-opens exactly the defect above.
* **Prefer DejaVu over Liberation on Linux, and it is not a style preference.**
  The ASS side names `DejaVu Sans Mono` for every non-Mac, non-Windows host,
  while the measuring side will accept Liberation Mono if that is what it
  finds. A Liberation-only box therefore measures one family and draws another
  -- quietly, with no warning, because both halves individually succeed. The
  standing fix is for the drawn family to be derived from the face actually
  resolved instead of a parallel hand-kept map; until that lands, install
  DejaVu or set both env vars to the same file.

**AMD/Linux is the lane most exposed to this**, for the same reason its whole
column is 0-of-68: no one has ever run it, so nothing has forced the question.

---

## What every recipe shares

| slot | value | why |
| --- | --- | --- |
| voice | `kokoro` both slots, bank `kokoro_builtin`, policy `auto_registry` | the only frictionless voice with receipts on every machine class |
| music | `stable_audio_3` | auto, ungated; the only music engine with published receipts |
| upscaler | `off` | see the note below -- it is unmeasurable, not proven |
| image | parked and dormant | every recipe's video lanes declare `accepts_still = False`, so no image engine is invoked and its download is never paid |

**The upscaler status line should say what is true:** the ledger records no
upscaler at all (`otr_master_audio_mux.py:1028-1033`), so on 8 GB it is
*unmeasurable*, not proven. On the Mac it is now `measured` -- 96 frames flat at
1.13 GB, ~24 min projected for a full episode -- but still not `proven`, because
no published episode has run the stage.

---

## 8 GB NVIDIA (RTX 4060 class)

| slot | value |
| --- | --- |
| writer | `Qwen/Qwen3.5-4B`, both slots |
| video x3 | `viz_mxc_cpu (16:9) (audio-reactive, no scene image)` |
| image | `z_image_turbo` x3 (dormant) |
| voice / music / upscale | the shared block above |

**Download ~12.2 GB. No node packs.**

**The correction the verifier made, and it is the important one:**
`llm_quant_policy` must be **`bnb_nf4`**, not `none`. The 2.99 GiB / 14.47 tok/s
receipt this card is famous for **IS an NF4 receipt**
(`docs/4060_DRILL_LOG.md:4515`); bitsandbytes installs on every non-darwin box.
The policy widgets flip as a SET:

```
    llm_device            mps  -> cuda
    llm_quant_policy      none -> bnb_nf4
    llm_vram_ceiling_gb   10.0 -> 6.8        (the 8 GB class's own value)
    llm_attn_impl         sdpa -> sdpa       (unchanged)
```

**What the all-`viz_mxc_cpu` collapse costs, said plainly:** the three video
roles were designed to look different (`docs/PROD_BUG_LOG.md:12577`). One engine
in all three slots is one look for the whole episode. It is the right ship-now
call on receipts; it is not free.

---

## 16 GB+ NVIDIA (RTX 5080 class)

| slot | value |
| --- | --- |
| writer | `google/gemma-4-E2B-it`, both slots, `quant_policy` stays `none` |
| announcer / music / character visual | `still_flat (16:9)` / `still_motion (16:9)` / `still_pan (16:9)` |
| image | `z_image_turbo` x3 |
| voice / music / upscale | the shared block above |

**Download ~32.7 GB. No node packs. THREE device values change, not four.**

* node 92 `OTR_VideoRenderBatch` -- **leave alone.** Its `engine` widget is
  diagnostic-only and inert in `mode=episode`.
* node 88 `OTR_ImageDirector` -- unchanged; `fp8_ok` is inert while only the
  bf16 z_image file is present.
* node 87 indices 0/1/2 take the **decorated** menu labels, not bare ids.

**Status is "components proven, tuple pending", and the pending list has TWO
items:** no single published run has carried all seven widgets together, AND no
published 16 GB episode has used E2B through `quant_policy: none` with
`native_text_decoder` -- the August receipts ran `bnb_nf4` under the
pre-2026-09-06 composite loader.

---

## Apple Silicon (M4 / 16 GB)

This is the one recipe the verifier did **not** correct.

| slot | value |
| --- | --- |
| writer | `Qwen/Qwen3.5-4B` -- `mac16-tight`, and the smallest viable Mac writer |
| video x3 | the zero-download procedural lanes |
| image | `sd15` |
| voice / music / upscale | the shared block above |

**Download ~12.2 GB.**

**Its weakest slot is the image slot, and it is frictionless only because it is
asleep.** There is NO frictionless image engine on this Mac: `sd15` is a hand
fetch. With procedural video lanes nothing invokes it, so the JSON is honestly
no-friction -- but the moment a user picks a still-consuming lane, that stops
being true. Say so in the JSON's own notes.

**Do not swap the writer for `gemma-4-E2B-it`.** It measures ~10 GB at bf16
against Qwen's ~9 -- larger, despite the smaller download.

---

## AMD ROCm / Linux

**Ship ONE AMD JSON (>=16 GB VRAM), not two.** Do not ship an <=8 GB AMD JSON at
all.

`workflows/otr_canonical.json` already IS this combination. Five edits:

```
  1. node 1  llm device            mps -> cuda   (ceiling 14.5)
  2. node 87 device_policy         mps -> cuda
  3. node 88 dtype                 fp8_ok -> no_fp8      (cosmetic; lane dormant)
  4. node 80 CastLock voice device mps -> cuda
  5. the validator's profile_id stays EMPTY  <-- LOAD-BEARING
```

**Edit 5 is the one that matters.** An empty `profile_id` is what keeps
`otr_amd16_rocm.json`'s `viz_mxc_mandala` and `still_motion` from ever reaching
the shipped graph -- and `viz_mxc_mandala`'s pycairo is pinned
`sys_platform == 'win32'`, so stamping that profile would break the music lane
on a stock Linux install. Do not stamp the AMD profile onto this JSON.

**Download 12.20 GiB.** No token, no licence click, no manual fetch, no
installer, no node pack, no GGUF.

**AND NOT ONE CELL OF IT HAS AN AMD RECEIPT.** Nothing in this repo has ever run
on AMD hardware; the matrix's AMD column is 0-of-68 for exactly that reason.
Every value above is arithmetic plus the absence of a known blocker. That is a
reasonable thing to ship as a starting point and an unreasonable thing to
describe as proven.

---

## What to check before applying any of this

1. The device widgets flip **as a set**. `llm_device` alone is not enough, and
   `OTR_VideoDirector.device_policy` is decorative -- nothing reads it (see
   README's device-widget table). The two that bite are
   `OTR_CastLock.voice_device` (which drives voice AND music) and
   `OTR_SilentComposite.upscale_device`.
2. Widget values are POSITIONAL. Any edit goes through
   `workflows/otr_canonical.json` and then the four gates in CLAUDE.md
   section 0.
3. Saved labels must be a live dropdown option. A stale label renders red and
   can resolve to index 0 -- it happened once already on 2026-09-09.
