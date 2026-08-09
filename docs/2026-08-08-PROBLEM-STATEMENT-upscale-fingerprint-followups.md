# PROBLEM STATEMENT -- upscale-stage follow-ups: the composite cache fingerprint

**Date:** 2026-08-08
**Branch / HEAD at framing:** `v2.0-alpha` @ `36d695f6`
**Queue position:** item 8 follow-up chips 1 and 2 (see `docs/GO_FORWARD_PLAN.md`,
"TOMBSTONE -- SYSTEM-AGNOSTIC UPSCALE STAGE"). Items 1/3/4/5/6/7 are blocked on
the operator; item 9's remaining chunks are held until LEMMY Phases 2-4 land in
the concurrent Codex window.
**Files this chunk may touch:** `nodes/otr_silent_composite.py`,
`nodes/_otr_upscale_engines/*`, `tests/*`, plus comment-only edits in
`nodes/video_engine.py`, `nodes/_otr_paths.py`, `nodes/_otr_memory.py`,
`nodes/otr_post_upscale_procgen_blend.py`.
**Files this chunk MUST NOT touch** (owned by the concurrent Codex window until
LEMMY lands): `config/cast_pools.py`, `nodes/_otr_dialogue_policy.py`,
`nodes/_otr_line_composer.py`, `nodes/_otr_compose_exchange.py`,
`nodes/production_ledger.py`, `nodes/_otr_voice_node_common.py`.

---

## 0. Why this chunk, and what was declined on the way here

The handoff queued a "ledger-field repro leg" first: deliberately drop
`meta.source_bank`, run a Shakespeare still, and show daggers becoming bananas,
as the live artifact that would qualify a Bug Bible promotion.

**That leg was declined, and the reason belongs in this record.** The fail-open
is not a defect; it is the specified behaviour, in three independent places:

* `nodes/_otr_banana_route.py:612-614` -- *"A ledger with no `meta.source_bank`
  (hand-built harness requests) is NOT excluded -- the global default applies."*
* `docs/2026-08-06-BUILD-SPEC-banana-route.md:83-84` -- the same sentence, in the
  reviewed and shipped spec.
* `tests/test_banana_route.py:601-602` -- `""` is pinned as NOT excluded.

A live artifact of intended behaviour cannot qualify a Bible promotion, because
the admission rule requires a *bug*. One genuine observation survived the
grounding and is recorded for the operator rather than acted on: the two
consumers of that field disagree on failure direction -- credits `_require` it
and die loud (`nodes/otr_credits_roll.py:565`), the banana gate treats absence
as licence to substitute -- and `nodes/otr_credits_roll.py:130-133` records that
**676 of 1178 named on-disk episodes predate the stamp**. Whether the gate
should fail toward the conservative branch is a design call, not a coder call.

---

## 1. DEFECT A -- the model fingerprint is hardcoded to one engine (chip 1)

`OTR_SilentComposite.IS_CHANGED` (`nodes/otr_silent_composite.py:1268-1359`) is
the node's ComfyUI cache key. Its docstring claims it fingerprints **"ALL
external inputs regardless of engine"**. Section 5 does not:

```python
# nodes/otr_silent_composite.py:1341-1356
parts.append(("engine", str(upscale_engine), str(upscale_device)))
if upscale_engine == "spandrel_esrgan":
    model_path = None
    try:
        import folder_paths  # type: ignore
        try:
            model_path = folder_paths.get_full_path(
                "upscale_models", "RealESRGAN_x2plus.pth")
        except Exception:
            model_path = None
    except (ImportError, ModuleNotFoundError):
        model_path = None
    if model_path and os.path.isfile(model_path):
        st = os.stat(model_path)
        parts.append(("model", model_path, st.st_mtime_ns, st.st_size))
```

Both the engine id and the checkpoint filename are string literals. The registry
was built precisely so a second engine ships "its own row here; zero per-profile
edits" (`nodes/_otr_upscale_engines/registry.py:104`). Engine #2 will register,
appear in the dropdown, run -- and contribute **no model bytes to the cache
key**. Swapping its weights in place would then reuse a stale composite.

The engine already owns this fact:
`nodes/_otr_upscale_engines/eng_spandrel_esrgan.py:55` declares
`_model_filename = "RealESRGAN_x2plus.pth"`, and `:70` declares
`_model_sha256`. `IS_CHANGED` restates both instead of asking.

**Severity is latent, not live.** Today `CAPABILITIES` holds exactly two rows
(`off`, `spandrel_esrgan`), so no shipped configuration is mis-fingerprinted.
This is a defect that fires on the next engine, which is the whole point of the
namespace.

## 2. DEFECT B -- the fingerprint and the loader resolve the model differently

Found while grounding chip 1; not in the original chip text.

`IS_CHANGED` resolves via `folder_paths.get_full_path(...)` only. The adapter's
`load()` resolves via a **two-stage candidate list**
(`eng_spandrel_esrgan.py:100-128`): every `folder_paths.get_folder_paths(
"upscale_models")` entry, **then** a repo-relative fallback
`Path(__file__).resolve().parents[4] / "models" / "upscale_models"`.

Two resolvers, one file. They agree on the ordinary ComfyUI path and diverge
when the checkpoint exists only under the repo-relative fallback: `load()`
succeeds, `IS_CHANGED` fingerprints nothing, and replacing that file never
invalidates the cache. `folder_paths` raising a non-import error diverges the
same way -- the adapter logs and continues to the fallback (`:108-111`), the
fingerprint gives up.

The narrowness is stated honestly: this needs a box where the file is reachable
only through the fallback. It is still two answers to one question, and that
`parents[4]` path has already drawn blood once -- `867f16c3` fixed a test that
false-passed the moment the operator populated the real models dir, because it
isolated `get_folder_paths` but not the fallback (GO_FORWARD item 8 chip 5).

## 3. DEFECT C -- the pinned SHA is not in the fingerprint

`_model_sha256` was `""` at ship and was pinned in `8250e01c`. A re-pin changes
what the engine will *accept* without changing any file the fingerprint reads,
so a cached composite survives a verification-contract change. Cheapest possible
fix: fold the engine's declared digest into the parts tuple.

## 4. COVERAGE GAP -- `IS_CHANGED` has no tests at all

`grep -l IS_CHANGED tests/` returns `test_audio_cache_wiring.py`,
`test_route_freeze_wiring.py`, `test_route_freeze.py`,
`test_otr_workflow_validator.py`, `test_workflow_validator_paths.py` -- none of
them `OTR_SilentComposite`. The eleven test files added by the item 8 ship cover
the registry, the widget positions, the pipeline and the adapter, but not the
cache key. Any fix here lands with the first tests this classmethod has had.

## 5. CHIP 2 -- stale RTXUpscale prose (comments only)

`nodes/rtx_upscale.py` was DELETED in the item 8 ship and `OTR_RTXUpscale` added
to `DELETED_NODE_TYPES`. Prose still refers to it as if it were live:
`nodes/video_engine.py:2086` (a user-visible tooltip), plus docstrings and body
comments in `nodes/_otr_paths.py`, `nodes/_otr_memory.py` and
`nodes/otr_post_upscale_procgen_blend.py`. Zero behaviour change; the audit
sweep is to confirm the list is complete rather than trusting the chip text.

---

## 6. THE DRIVER'S PROPOSED SHAPE (attack this)

Single-source model identity in the adapter; make the node ask.

1. Give the upscale namespace one optional, duck-typed member --
   working name `model_fingerprint_parts() -> tuple` -- returning the
   `(label, path, mtime_ns, size)` tuples plus the declared digest for whatever
   files that engine actually consumes. `off` declares nothing; the node uses
   `getattr(engine, ..., None)` so an engine without the member is legal, matching
   how `requires_flag` stays vestigial rather than mandatory.
2. Extract the adapter's candidate-list resolution out of `load()` into a
   non-raising `resolve_model_path()`, and have `load()` call it, so Defect B
   closes by construction rather than by keeping two lists in step.
3. `IS_CHANGED` section 5 becomes: append engine identity, then
   `_get_upscale_engine(upscale_engine)` and extend with whatever the engine
   reports. Keep the whole-body `except -> float("nan")` fail-open (Bug Bible
   06.02/06.07) and keep the module-top import fallback at
   `otr_silent_composite.py:36-67` working -- its `_NullOff` stub must not start
   raising.

**Questions the panel is asked to break, not bless:**

* Is a duck-typed optional member right here, or should `UpscaleEngine` gain a
  required method with an `off` no-op -- given the Protocol is
  `@runtime_checkable` and `test_protocol_parity` pins it as a structural
  superset of the audio core?
* Does moving resolution out of `load()` change any raise site or reason code?
  `load()` currently raises `EngineUnusable(MISSING_MODEL)` with the candidate
  list in the message; that message is asserted in the shipped tests.
* Is calling `get_engine()` inside `IS_CHANGED` safe at ComfyUI queue time?
  `register` instantiates at import (`engine_registry_base.py:148`), so the
  singleton exists -- but `IS_CHANGED` runs on every queue and must stay cheap
  and side-effect-free. Does reading a declared attribute off a shared singleton
  risk anything if a render is concurrently holding the descriptor?
* Should the digest of a **missing** model be fingerprinted as absence, or should
  a declared-but-missing checkpoint return `nan` and force re-execution?
* Is there a fifth site in the chip-2 prose sweep the driver has not listed?

## 7. WHAT "DONE" LOOKS LIKE

* One atomic commit, pushed to `v2.0-alpha`, HEAD == origin.
* No second engine invented to prove the point -- the tests use a fake adapter.
* Suite at or above the `36d695f6` baseline **9465 passed / 111 skipped /
  1 xfailed**, zero failures; Bug Bible 17/24/3 at survival-guide `7a5fb88`.
* `git diff -- workflows/` EMPTY. This chunk has no widget, no link and no
  schema change, so the canonical JSON must not move.
* Sonnet 5 QA on the diff, then the Fable final gate, then the suite, then push.
