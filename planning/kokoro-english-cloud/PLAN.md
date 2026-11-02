# Kokoro English on a box without spaCy's model (Comfy Cloud first)

Status: BUILT 2026-11-02. Proven locally (real fetch, model hidden, English line voiced from the 3.7.1 copy); cloud leg pending.

## The defect

On Python 3.12 the torch `kokoro` package speaks English through misaki, and
misaki's English G2P needs spaCy's `en_core_web_sm`. The pack does not declare
it: the model is not on PyPI, and a direct-URL requirement is what the
registry's pattern scan flags (`contains_custom_url_dependency`; 2.3.45 and
2.3.46). So `_kokoro_backends.py:828-845` refuses English by name and names the
one line to run. That is right for a desktop install, where the operator can
run it. On Comfy Cloud nobody can: `pipDependencies` in the build spec only pin
versions and never add a package, and the worker image is built from the pack's
own requirements. Result, measured 2026-10-31 to 11-01: every English cloud
episode falls back to Bark; Hindi on Kokoro succeeded (job eaef5ad2) because the
espeak rows never touch spaCy.

## What is measured

* `spacy/en_core_web_sm` on Hugging Face is spaCy's own mirror, MIT, one file
  `en_core_web_sm-any-py3-none-any.whl` (12.8 MB) at commit
  `8fb6f8155286360bf5486fe399e2738d1df7ded2`, sha256
  `86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889`.
  It is version **3.7.1** (`spacy_version >=3.7.2,<3.8.0`); this box has 3.8.0
  under spaCy 3.8.11, the cloud image spaCy 3.8.18.
* The wheel unzips to `en_core_web_sm/` plus `en_core_web_sm-3.7.1.dist-info/`.
  With that folder on `sys.path`, `importlib.util.find_spec` (our gate) and
  `importlib.metadata.distribution` (misaki's `spacy.util.is_package`) both
  find it, so misaki's own `spacy.cli.download` branch is never reached.
* A/B through misaki's G2P on eight script lines, model 3.8.0 vs 3.7.1:
  **7 of 8 phoneme strings identical.** The one difference is a heteronym after
  a dash: "-- record it" tagged as a verb by 3.8.0 (ɹəkˈɔɹd) and a noun by 3.7.1
  (ɹˈɛkəɹd). So the mirror is close but not byte-identical to a desktop box.

## The design

Same shape as the Kokoro voice and ONNX prefetch the operator already approved
(2026-08-24): fetch once at prestartup, never inside a render, never fatal.

1. `nodes/_otr_kokoro_voice_prefetch.py`: new `prefetch_spacy_english_model()`.
   Runs only when it can matter: the torch `kokoro` package will be selected
   (the inverse of `onnx_backend_wanted()`), `en_core_web_sm` is not already
   importable, and no extracted copy exists. Honours `HF_HUB_OFFLINE` and
   `OTR_SKIP_KOKORO_PREFETCH`. `hf_hub_download` by repo id, filename and the
   pinned commit (no web address in the source), checks the sha256, and unzips
   into `<models>/TTS/spacy/en_core_web_sm-3.7.1/` with a zip-slip check
   (every member must resolve inside the destination). Called from
   `prefetch_at_boot` in its own try, like the other two fetches.
2. `nodes/_otr_audio_engines/_kokoro_backends.py`, the English branch: when
   `find_spec("en_core_web_sm")` is None, look for that extracted folder; when
   it is there, append it to `sys.path` (append, not insert: an installed model
   always wins) and continue. The refusal stays, reworded to name both fixes:
   the one line to run, or a restart with internet so the boot fetch runs.
   One helper owns the path, used by both files.
3. Nothing in `requirements.txt` or `pyproject.toml` changes, so the scan
   oracle has nothing new to read; it still runs before the bump.

Blast radius: a box that already has the model (this 5080, any desktop install
that ran the one line) is untouched, because the installed package is found
first and the prefetch returns before any network. Python 3.13 (ONNX backend)
is untouched: the prefetch does not run and its English path never loads misaki.
Only boxes that today refuse English Kokoro change, from a refusal to speaking.

## Contrarian round (Sonnet subagent, briefed to refute, 2026-11-02)

Verdict SOUND WITH FIXES. Every finding below was checked against the files
and is folded in:

* **Gate the fetch on the torch package, not on "not ONNX".** The inverse of
  `onnx_backend_wanted()` is also true when neither backend is installed. The
  prefetch runs only when `OTR_KOKORO_BACKEND` is not `onnx` and
  `_spec_exists("kokoro")`, matching how `select_backend_name`
  (`_kokoro_backends.py:136`) prefers torch.
* **Keep the logs clean.** spaCy 3.8 warns W095 (model built for 3.7) on every
  load. Filter that one warning around the `KPipeline` construction, and only
  when the extracted copy is the one in use; the boot fetch logs one line
  naming the 3.7.1 model.
* **No shared import between the two files.** The prefetch module loads flat
  at prestartup and by path in `tests/test_standalone_module_loads.py`; the
  backends are a package module. Follow the existing convention
  (`_KOKORO_MODEL_SUBDIR`, prefetch:105): duplicate the small path constant in
  both and pin them equal with a test.
* **Atomic extraction.** Verify the sha256 on the downloaded wheel, unzip into
  a temp folder beside the target, then rename it into place. The gate
  trusts the folder only when `en_core_web_sm/meta.json` is inside it.
* **Append to `sys.path` once**, not on every `load()`.
* **The refusal keeps the substring** `spacy download en_core_web_sm`
  (`tests/test_argon_sinks_are_closed.py:46`), and that test also hides the
  extracted folder so it still sees a refusal.
* **Unproven on the cloud: that prestartup can fetch.** The boot log shows
  `HF_HOME=/app/ComfyUI/models/huggingface`, and Kokoro's own weights landed
  there during the Hindi run, so `models/` is writable and the worker has
  network; the cloud leg below is what proves the boot-time fetch itself.
* **Only the repo's scan replica was read.** `hf_hub_download` is already in
  this module and is not a replica rule; archive extraction and `sys.path`
  edits are not modelled, so the private scanner remains a residual risk.

## Proof

* Unit: prefetch receipt paths (present, offline, skip env, ONNX box, sha
  mismatch, zip-slip refused); the backend finds the extracted copy and refuses
  with the new message when it is absent. Assert the call from
  `prefetch_at_boot` exists (wiring, not only the helper).
* Live, local: with the installed model hidden (a scratch venv without it),
  English Kokoro speaks from the extracted copy.
* Live, cloud: re-cut the build on the fix, run an English one-act with both
  engines on Kokoro, publish to obs.

## Alternatives weighed

* **spaCy's GitHub release wheel (3.8.0) fetched at boot.** Byte-identical to a
  desktop box, but it puts a web address in the source, which is the exact
  pattern the registry flagged. Rejected unless the scan is proven to read only
  the dependency files.
* **Fall back to espeak English when the model is missing.** No fetch at all,
  but it changes how every cloud English line sounds: an ear decision, not a
  fix.
* **Ship the 12.8 MB wheel inside the pack.** No network, but 12.8 MB more in
  every registry zip for a lane most installs never need, and a vendored model
  to keep current.
* **Put the wheel in the build spec's model list.** Lands it in the image with
  no cold fetch, but only for this one build; the pack still needs step 2 to
  find it, and every other font-less or fresh 3.12 install stays broken.
