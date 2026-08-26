# SOAK LEG GUIDE -- the key items to change per leg, and how

**Written 2026-08-25** so the ground covered building the LLM x image x
upscale sweep isn't relearned the next time a soak campaign starts.
Companion to `docs/LLM_PREFLIGHT_GUIDE.md` (whether a model belongs in the
dropdown at all) -- this doc is about *driving* a leg once it does.

---

## 0. The one rule everything else follows

**The video/image engine widgets are MANAGED. A headless `--set` cannot touch
them.** `patch_creative` (`nodes/_otr_workflow_apply.py:731`) explicitly
refuses anything outside `CREATIVE_WHITELIST` -- this is deliberate (the
BUG-08.06 stranded-COMBO class: a raw widget poke can save a value no live
menu recognizes). The **only** sanctioned lever for an engine pick is a
**capability profile**'s `role_overrides` / `upscale_stage`, applied through
`apply_profile_to_workflow`. That is not a workaround -- it is doing the exact
same thing a human clicking the dropdown and saving the graph would do,
through the same applier the UI itself would go through.

Practical consequence: **to vary an engine across legs, write one small profile
JSON per leg (or per axis) rather than reaching for `--set`.**

---

## 1. What CAN go on the CLI directly (`CREATIVE_WHITELIST`)

`nodes/_otr_workflow_apply.py:681`. These are content/model-selection dials,
not managed engine widgets, and the runner (`scripts/otr_canonical_api_run.py`)
already has dedicated flags for the most-used ones:

| flag | widget | notes |
|---|---|---|
| `--source-bank` | `source_bank` | pin, don't `roll`, if the sweep needs a specific writer path -- see §4 |
| `--visual-style` | `visual_style` | |
| `--creative-model` / `--technical-model` | `creative_writing_model` / `technical_model` | **applied AFTER the profile** -- see §2 |
| `--num-characters` | `num_characters` | |
| `--act-count` | `act_count` | `'1'` = exactly 3 voiced beats, see §5 |
| `--title` / `--premise` | `episode_title` / `custom_premise` | **never use `--title` for a soak leg label** -- PBUG-20260817-05: it becomes the on-screen title card and the published filename. Label your own leg in your own harness's console/receipt, not the widget. |

Anything else goes through `--set NODE.widget=value`, still gated by the same
whitelist -- and still refused if it names an engine widget.

## 2. Order of application, and why the CLI wins

`build_api_prompt` (`scripts/otr_canonical_api_run.py`) applies the profile
**first**, then `_apply_writer_shortcuts` (which is what `--creative-model` /
`--technical-model` go through) **second**. So a profile's own
`llm.creative_model` / `llm.technical_model` are silently overridden the
moment you pass the CLI flags -- **verified empirically** (2026-08-25): a
profile pinning `Mistral-Nemo`/`Mistral-Nemo` produced a leg whose actual
prompt carried whatever `--creative-model`/`--technical-model` said instead.

Practical consequence: a soak profile that only needs to vary the STILL/IMAGE/
UPSCALE surface does not need a real `llm` block at all -- leave it at any
harmless default and drive the model choice from the CLI. Do not try to encode
the LLM under test in the profile; it will be silently ignored the moment a
CLI override is present, and a reader could be misled into thinking it wasn't.

## 3. The widget map (canonical, verified against the live tree 2026-08-25)

Read live via `INPUT_TYPES()`, not memorized -- indices drift when a widget is
added. Re-derive with the probe pattern in §7 before trusting these numbers on
a HEAD that has since moved.

**Node 1, `OTR_LedgerScriptWriter`** (creative dials -- CLI-reachable):
`episode_title`(0) `num_characters`(1) `creative_writing_model`(2)
`technical_model`(3) `custom_premise`(4) `include_act_breaks`(5)
`act_count`(6) ... `source_bank`(22) `visual_style`(23) ...

**Node 87, `OTR_VideoDirector`** (engine dials -- PROFILE ONLY). Note both the
video AND image per-role pickers live on THIS node, not on `OTR_ImageDirector`:
`announcer_video_model`(0) `music_video_model`(1) `character_video_model`(2)
`announcer_image_model`(3) `music_image_model`(4) `character_image_model`(5)
... `device_policy`(12) `dtype_policy`(13) `max_render_frames`(14)

Value FORM differs by widget: the three `*_video_model` widgets store the
live menu label **with its suffix**, e.g. `'still_flat (16:9)'`; the three
`*_image_model` widgets store the **bare engine id**, e.g. `'z_image_turbo'`
(no suffix at all). A profile's `role_overrides` writes the bare internal id
either way -- `_director_option_value` in `_otr_workflow_apply.py` resolves it
to the exact live label for the video widgets automatically. You only need to
know the bare ids to author a profile; you only need the suffixed form if you
are hand-editing `widgets_values` directly instead (not recommended -- use a
profile).

**Node 88, `OTR_ImageDirector`**: granularity/seed/dtype only. **No engine
picker lives here** -- do not go looking for `announcer_image_model` on this
node, it is a mirror consumer (`otr_meta_brief_image_prompt.py`), not the
authority.

**Node 84, `OTR_SilentComposite`** (upscale -- PROFILE ONLY, via
`upscale_stage`): `upscale_engine`(5), choices are **exactly two**:
`'off'`, `'spandrel_esrgan'` -- there is no third upscale engine to rotate in.
`upscale_device`(6) is a free-form STRING; `'cuda'` or `'cpu'`.

## 4. Which source bank actually reaches BOTH LLM slots

Found by the 2026-08-25 sweep-design fan-out, worth restating here because it
is easy to get backwards: an unpinned `source_bank='roll (any eligible bank)'`
can land on a lane that never calls `technical_model` at all, which silently
proves nothing about that slot. **`scifi_news_pro` is the only bank verified to
drive both the creative and technical writer closures** (creative: pitch/
treatment/script at `_otr_scifi_news_pro.py:4409/4416/4456`; technical: P0
dossier/cast_aliases/news_read/safety cleanup/`_pass_casting` at `:4342/4434/
4441/4474/4533`). **Pin it explicitly for any leg meant to exercise both
slots.**

## 5. `act_count='1'` is the fast, high-coverage shape

At `act_count=1` there are **exactly 3 voiced beats** -- one each for
announcer, music and character. That 1:1 mapping onto the three per-role
engine widgets means a SINGLE one-act leg can already exercise three DIFFERENT
still engines and three DIFFERENT image engines in one render (one per role),
without needing three separate legs. This is the mechanism behind "one act,
lots of variety" -- use it before reaching for more legs.

## 6. The genuinely LOCAL engine pools (verified on disk 2026-08-25)

Do not assume a dropdown entry is local from its name alone -- `ideo` and
`google_image` are both cloud (`nodes/_otr_image_engines/eng_cloud_image.py`,
`eng_google_image.py`), despite sitting in the same combo as the true local
rows with no visual distinction beyond the source file.

* **Still/video (no video):** `still_flat`, `still_motion`, `still_pan`,
  `still_word` (+ `word_razzle`, a 5th word-display variant, if wanted).
* **Local image models (5):** `z_image_turbo`, `flux_gen1`, `flux2_klein`,
  `lumina_image`, `ideogram4_local`.
* **Upscale (2, exhaustive):** `off`, `spandrel_esrgan`.
* **Curated local LLM rows (7, per `docs/LLM_PREFLIGHT_GUIDE.md`):**
  `mistralai/Mistral-Nemo-Instruct-2407`, `google/gemma-4-E2B-it`,
  `google/gemma-4-E4B-it`, `google/gemma-4-12b-it`,
  `unsloth/gemma-4-12b-it-GGUF`, `unsloth/Qwen3-8B-GGUF`,
  `google/gemma-2-2b-it`. Pass the FULL dropdown label including the size
  suffix to `--creative-model`/`--technical-model` (e.g.
  `'google/gemma-4-12b-it (11.9 GB)'`) -- `validate_model_id` resolves it.

## 7. How to re-derive the widget map when it drifts

```python
import os, sys
os.environ["OTR_TEST_MODE"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes")
from importlib import import_module
pkg = import_module("ComfyUI-OldTimeRadio")
cls = pkg.NODE_CLASS_MAPPINGS["OTR_VideoDirector"]  # or any node
it = cls.INPUT_TYPES()
# walk required/optional, skip forceInput entries -- that ordering IS the
# widgets_values index order.
```

Then cross-check against the REAL canonical's `widgets_values` for that node
type in `workflows/otr_canonical.json` -- if the lengths disagree, the widget
map has drifted and needs re-deriving before trusting any index above.

## 8. A worked example -- the LLM x image x upscale sweep

`scripts/otr_llm_image_upscale_sweep.py` + `config/profiles/otr_soak_llmsweep_0{1..7}.json`
is a complete, working instance of every rule above: 7 legs, cyclic LLM
pairing (every row plays creative once and technical once), 3 distinct stills
+ 3 distinct local image engines per leg via `role_overrides`, alternating
`upscale_stage.engine`, `--creative-model`/`--technical-model` on the CLI,
`scifi_news_pro` pinned, `act_count=1`. Read it as the template for the next
sweep rather than starting from a blank page. Its 7 profiles are registered in
`scripts/build_variants.py`'s `LANE_PRESETS` as SOAK INSTRUMENTS -- **any new
soak profile must be added there too**, or `build_variants.py --check` never
notices it and it also never *accidentally* ships as a user-facing variant
(the exclusion is what keeps it out, not an oversight).

## 9. Always validate offline before booking a GPU leg

`--dry-run` REQUIRES `--offline-schemas` alongside it, or the "dry" run still
calls the live `/object_info` endpoint and fails with a `ConnectionError` if no
server is booted -- which looks like a crash, not a validation pass. **A driver
that treats `dry_run` as an automatic pass regardless of exit code is a false
green** (caught in the first cut of the sweep script above: every leg
"passed" by never actually running). Check for the real completion marker
(`"DRY_RUN complete"` in stdout) and a zero exit code, not the flag alone.
