# Preflight

The checks that decide whether a new lane actually works, before you spend an
evening finding out it does not. One section per kind of thing you can add.

Each section names the test file that enforces it. **A gate with a test is a
promise; a gate without one is advice**, and this document says which is which
rather than pretending they are the same.

Run the section for what you built, then the render at the bottom. The render is
the part that counts.

---

## Video lane

Enforced by `tests/test_lane_preflight_matrix.py`.

- **Weights resolve, or refuse by name.** Through `folder_paths` or a documented
  environment pin. A missing file raises a named `EngineUnusable` from
  `assert_usable` -- never an `os.path.exists` on a hardcoded default, never a
  silent substitution.
- **`accepts_still` is declared explicitly.** True or False in the class body.
  Not inherited by accident, not omitted. This one value decides whether the
  image model downloads at all, so getting it wrong costs a stranger either a
  19 GB surprise or a broken render.
- **The clip contract holds.** A silent clip is still a valid clip; a directory
  of frames is validated as a directory. Both paths have assertions.
- **SageAttention is not patched underneath you.** `assert_sage_not_patched`.
- **The dropdown label resolves.** `exact_menu_option_for` -- the public label a
  human picks must map to your internal id.
- **The still plan is a real `StillPlanRow`** if your lane consumes stills.
- **The registration surface is complete.** This is the part people miss, and it
  is not a test you can write after the fact:
   - a shortcode, checked by
     `tests/test_shortcodes.py::CompletenessTests::test_every_video_engine_id_has_a_code`;
   - exactly one of the bookend rosters in
     `nodes/_otr_video_engines/render_driver.py` -- `ENGINES`, `BOUNDED`,
     `SELF_COMPOSED`, `NOT_TEXT_DRIVEN`, `KNOWN_RED`. They are asserted disjoint
     by `tests/test_bookend_scene_prompt_roster.py`, so exactly one is correct
     and "none" is a failure.
   - regeneration of every generated document that now has a row for you.

## Image engine

Enforced by `tests/test_image_gen_preflight_matrix.py`.

- **Weights resolve or refuse by name**, as above.
- **The engine declares its device backends honestly** -- from a measurement, not
  an assumption.
- **It is inert until something consumes its still.** Verify by switching a
  video role to a still-consuming lane; do not conclude your engine is broken
  because the canonical's procedural lanes ignore it.
- **A refusal classifier is optional but supported.** An engine can declare
  `is_model_refusal = True` on its own exception type, and the dispatcher reads
  it at runtime. There is a live example if you need one. *(No test covers this
  path -- it is a real mechanism with no gate.)*

## Voice engine

Enforced by `tests/test_tts_voice_preflight_matrix.py`.

- **Declare the reference contract**: `voice_ref_kind`, `voice_ref_field`,
  `sample_rate`.
- **Resolve reference paths through the real resolver.** `resolve_voice_ref_path`
  in `nodes/_otr_audio_engines/base.py`, or `_resolve_ref_to_disk` in
  `nodes/_otr_voice_node_common.py`. A repo-root join produces a path that has
  never existed.
- **Bound every protocol read.** If your engine talks to a worker process, route
  reads through `read_protocol_line`, never a bare `readline()`. An unbounded
  read turns a stalled worker into a hang that holds VRAM with nothing in the
  log, and is indistinguishable from a slow render.
- **A route tier is a decision, not a default.** Qualified, provisional and
  unrouted all exist; the dormancy gate in `nodes/cast_lock.py` tests both the
  approved and provisional keys.
- **Qualification still requires a person.** No code path promotes a voice route
  to qualified on its own, and that is deliberate.

## Episode language row

An episode language is a registry extension, not a voice engine. It must bind
one complete row across writer instructions, spoken chrome, credits, caption
font/wrap policy, source exclusions and a real Kokoro voice pool. Voice
eligibility comes only from each bank entry's explicit `languages` list; a
prefix is never evidence.

Run the focused registry/writer/voice/painter suite, the full regression and
the shared Bug Bible regression. Qualification still ends with a canonical
one-act episode in `otr/obs/`, followed by an inspection of native text,
glyphs and voices. The complete checklist is
[MULTILINGUAL.md](MULTILINGUAL.md).

## Music engine

**There is no preflight and no gate for music.** The adapters that declare
`roles = ("music",)` are not covered by a preflight matrix. If you
add one, follow the voice section's weight and declaration rules and be aware
you are working without a net.

One known open defect to avoid inheriting: the legacy fallback tuple in
`nodes/stable_audio_theme.py` lists fewer engines than the profiles table does,
with a different engine at the start of the tuple.

## Upscale engine

**No preflight of its own.** `tests/test_upscale_weight_resolution_gate.py`
borrows the video weight-resolution gate, and its own comment explains why: the
upscale namespace had test files and not one asked whether the engine's
checkpoint was reachable. Follow the video section's weight-resolution and
`accepts_still` rules.

## Writer LLM

The add-your-own page is [LLM_PREFLIGHT.md](LLM_PREFLIGHT.md) -- same job as
this file's engine sections, for a language model. You can pick anything the
dropdown will take. What the pack ships is Qwen 3.5 as one transformers
row (NVIDIA NF4, Mac / CPU full).

There is still no preflight matrix test. Schema tests that must stay green
against the real catalog:

`tests/test_model_catalog_schema.py`,
`tests/test_model_catalog_scan.py`,
`tests/test_loader_backend_protocol.py`,
`tests/test_chat_template_kind_dispatch.py`,
`tests/test_context_window_precondition.py`,
`tests/test_qwen_quant_twins.py`.

Green here means the row is well-formed, not that it can write an episode.
Then the render at the bottom.

## Cloud partner rows

**This section needs the git clone**, and so does every "Enforced by
`tests/...`" line above it: `tests/` and `scripts/` are both stripped from the
registry bundle. The CHECKLIST reads fine either way -- it is RUNNING the named
tests and commands that wants the clone.

**The contract lives only in a test**, `tests/test_cloud_engine_is_a_three_part_rule.py`:
a `cloud_` id prefix, a `provider_side` attribute, and a `cloud_` node key. Read
that file before adding one — nothing else documents it.

## Your own source bank

The bank shape and the two function signatures are in
[EXTENDING.md](EXTENDING.md). The gates:

- **The fetcher binds.** `python scripts/otr_check.py bank <path>` proves your
  functions can accept what the writer will pass, **without calling them**.
  The keywords are in [EXTENDING.md](EXTENDING.md).
- **Freeze policy resolves.** A bank whose policy cannot resolve fails closed.
- **The pipeline is registered in both tables** if you add one -- `LANE_SPECS`
  and `INLINE_PIPELINES`.
- **Roll-pool membership is two flags, not one.** `runnable: true` **and**
  `defaults.auto_select` not false.
- **Cameo policies exclude banks explicitly**, by frozenset, and the exclusion
  sets are asserted to match each other.

## Removing a lane or a bank

Atomic, in one commit, or it is worse than leaving it: registry row, the module,
both pipeline tables, the runner, cameo policy, every test, and a bare-token grep
that returns exactly the survivors you enumerated. A half-removed lane passes
every check and exists nowhere.

---

## The render, which is the only real gate

Everything above proves a part. **One full run through
`workflows/otr_canonical.json` that lands a file in `otr/obs/` proves the lane.**

Green tests are not a lane. A test that calls your helper directly proves your
helper and says nothing about whether anything calls it.

If it took more than five minutes and nothing reached `otr/obs/`, go read the leg
log — do not wait it out.
