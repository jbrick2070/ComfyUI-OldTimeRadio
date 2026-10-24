# Radio Play Studio: how it's filmed, a different writer, and the icon's colours

Driver anchor, 2026-10-23. Claude drives; reviewers are asked to break it.

The Studio is `contest/radio-play-studio/`: a local web page (port 8765) in front
of the pack's shipped workflows. Its form is `radio_play_studio/form.py` (FIELDS),
the page is `radio_play_studio/static/index.html`, the server is
`radio_play_studio/server.py`, and the run logic is `radio_play_studio/runner.py`
(`Studio.workflow_for`, `build_graph_from`, `photo_stills`, `form`). Workflows are
generated from `config/workflow_matrix.json` rows by `scripts/build_variants.py`.

## What the operator asked for (2026-10-23, his words, typos fixed)

1. "I don't need this box, Source to adapt -- you chose My Story for that."
2. "Put language first."
3. "Can't you make the colour scheme more old time radio icon (registry icon)?"
4. "I thought on the web page you were to expose a few video model options."
   Earlier the same day he named them: **TTS + Foley video**, **TTS w/ video**,
   **TTS w/ stills**. He also said "keep it fixed" once, then reversed it with #4.
5. "An option, maybe nested, to try a different LLM, but hidden unless selected.
   It should include OpenRouter, Comfy, Google and all slots, but only optional."

1-3 are done in the working tree (not pushed): Source to adapt hidden on the page
(kept for the CLI's --source-ref), Language is FIELDS[0], and a single dark theme
from the icon (#12092f indigo, #f82890 pink, #50f898 waveform green for the lamps,
a faint violet grid, the h1 in heavy upper-case sans with a pink shadow). Every
text pair clears WCAG AA.

## Facts the design rests on (verify them)

- The Studio runs one workflow per server: `--workflow`, default
  `otr_16gb_foley_open.json` (open weights: FLUX 3 Klein stills, LTX 2.5 foley,
  1080p by ffmpeg, no NVIDIA RTX VSR). A My Story run switches to the reference
  twin `<name>_flux3_ref.json` when it exists (`config.reference_twin`), so the
  characters are painted from portraits/photos with `flux3_klein_ref`.
- The 16 GB siblings differ from foley_open beyond the video lane:
  `otr_16gb_video` = ltx25_high_video + moss_image stills + rtx_vsr upscale;
  `otr_16gb_still` = still_motion + moss_image stills + rtx_vsr upscale. Neither
  has a `_flux3_ref` twin, and both carry a closed SDK (RTX VSR) the Studio's
  README promises it avoids ("every model stays open weights").
- Render time on the 5080, 1 act (planning/regression/ROUNDS.md): still 1,543 s,
  video 8,849 s, foley 12,090 s.
- The page's Still pictures box writes all three still dropdowns, so a filming
  choice that switches graphs keeps the person's still pick.
- The writer node (`nodes/OTR_LedgerScriptWriter.py`, INPUT_TYPES ~2600-2800) has
  `creative_writing_model` and `technical_model` (local models plus the handles
  `openrouter:slot-a/b`, `comfy:slot-a/b`, `google_api:slot-a/b`), and six slot
  dropdowns (`openrouter_slot_a_model` ... `google_api_slot_b_model`) whose first
  choice is an "(enable X)" sentinel until that service's credential is present.
  OpenRouter and Google read keys from ComfyUI's own environment
  (OPENROUTER_API_KEY; OTR_GOOGLE_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY).
  Comfy credits need OTR_ComfyCredential's hidden `api_key_comfy_org`: a
  signed-in ComfyUI app session or `extra_data.api_key_comfy_org` on /prompt;
  the Studio submits through comfy-api-proxy and has no session of its own.
- Operator rulings: cloud lanes are opt-in and never a hard dependency; local and
  offline is the default; the registry zip must stay at 0 scan findings (the
  Studio is not in the zip). Story quality is not being chased -- this is choice,
  not a quality push.

## Proposed design (the anchor -- break it)

**A. "How it's filmed"** -- a three-way choice near the top of The show, right
after Language, with his three names and the time each takes:
TTS + Foley video (default), TTS w/ video, TTS w/ stills. Each maps to a whole
workflow, never to a single swapped dropdown (a graph is tuned for its lane).
To keep the open-weights promise and My Story's faces, add four matrix rows,
generated like every other variant: `otr_16gb_video_open`,
`otr_16gb_video_open_flux3_ref`, `otr_16gb_still_open`,
`otr_16gb_still_open_flux3_ref` (FLUX 3 Klein stills, ffmpeg 1080p, no RTX VSR).
Offered only when the server runs the default graph and the siblings exist; any
other --workflow keeps today's fixed behaviour. `photo_stills` becomes per
filming choice.

Alternative considered: the Studio patches the upscale engine and stills at
submit time on the existing video/still graphs -- no new workflows, but the
Studio would run a graph that exists nowhere on disk.

**B. "Try a different writer"** -- a closed `<details>` at the end of The show.
Inside: Creative writer and Technical writer pickers (live choices from
ComfyUI), and, only for a picked cloud handle, that slot's model picker. A slot
still on its "(enable X)" sentinel shows one line saying which key ComfyUI needs.
Comfy slots are listed but marked as needing a ComfyUI sign-in the Studio cannot
supply. Nothing in it changes a run unless the box is opened and changed. The
README's "no cloud calls" line becomes "none unless you pick a cloud writer".

**C. Colours** -- as above; one dark theme.

## Questions for the reviewers

1. Is A's four-new-graphs route right, or is the submit-time patch (or something
   else) better? What breaks: the registry (new shipped graphs), tests that count
   workflows, the Scene Producer (a project records its workflow on first run),
   the photo warning, the 8 GB / other --workflow servers?
2. B: where does a closed panel hide a cost the person must see (a paid model, a
   missing key, a wrong technical model)? Should the technical writer be exposed
   at all? Is listing Comfy slots that cannot work from the Studio worse than
   hiding them?
3. C: does a single dark theme lose anything the light one carried?
4. Names and microcopy: are his three names clear on a page that speaks nine
   languages? (He chose them; suggest, do not overrule.)
5. Anything in the anchor that is factually wrong against the files.

## Decisions after the round (2026-10-23)

Roster: Codex (gpt-6-astra) and agy (Gemini, via kibitz --no-profiles) on the
files, Fable on judgment. The operator then reshaped the brief mid-round: a
machine picker first ("AMD / Mac / NVIDIA 8/16/24 ... then it prepopulates"),
cloud machines ("don't forget cloud too", "Comfy Cloud / Google Cloud"), a key
box ("enter your API or say where you need to store it"), mix and match for
stills, and A/B as one writer choice.

- **A is now "your machine, then how it's filmed",** and every choice is a
  workflow that already ships (`MACHINES` in config.py). That settles the
  four-new-graphs (Codex) versus submit-time-patch (agy) split: neither is
  needed. Ways without a reference twin run My Story without references; the
  photo warning says so per workflow.
- **Foley stays the default** (the operator's standing contest ruling), against
  Fable's "default to stills".
- **B is one creative-writer choice** with slot-A models for OpenRouter, Google
  and Comfy; the technical writer stays the workflow's (all three reviewers).
  Comfy stays in: agy and Fable cut it as unreachable, but the Comfy SDK's
  `submit(api_key=...)` sends `extra_data.api_key_comfy_org`, so the page can
  hold the key. A cloud writer on its "(enable ...)" line is refused before
  submit (agy). The panel sends nothing unless touched, and its summary always
  names the writer (Codex, Fable).
- **Stills mix and match:** every engine but the custom-model row, cloud ones
  marked paid.
- **One dark theme** stays (operator), declining Codex's light option.
