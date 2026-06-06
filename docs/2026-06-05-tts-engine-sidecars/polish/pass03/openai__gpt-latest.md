<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. `_render_per_line` still treats a non-empty but nonexistent clone ref as usable, so PD1 fallback is bypassed and the worker can hard-fail instead of rendering.

MUST-FIX BEFORE BUILD:
1. [_otr_voice_node_common.py::_render_per_line] Non-empty stale `voice_ref_path` bypasses both `_resolve_clone_ref_path()` and bark fallback. The code only checks `not voice_ref`; it never verifies that an existing cast `voice_ref_path` / `ref_path` resolves to a file before sending it to `adapter.generate_voice()`. This contradicts the surrounding comments saying fallback applies when there is no “usable reference clip” / “no reference clips are installed on disk,” and can break PD1 always-renders. Concrete fix: for adapters where `_engine_requires_voice_ref(adapter)` is true, normalize and validate `voice_ref` before the missing-ref block:
   - call `_resolve_ref_to_disk(voice_ref)` when `voice_ref` is non-empty;
   - keep it only if the resolved path exists;
   - otherwise set `voice_ref = None`;
   - then run the existing `_resolve_clone_ref_path(...)` and fallback logic.
   This also ensures the worker receives an absolute on-disk path instead of an unresolved relative path.

2. [_otr_voice_node_common.py::_render_per_line + eng_chatterbox.py::ChatterboxEngine] Chatterbox advertises `roles = ("char_voice", "announcer_voice")`, `requires_voice_ref = True`, and `missing_ref_fallback = "bark"`, but the fallback branch is hard-gated by `self.ROLE == "char_voice"`. For `announcer_voice`, a missing/unusable ref after resolution will skip the advertised metadata fallback and call the clone adapter with `voice_ref=None`. Concrete fix: either remove `announcer_voice` from `ChatterboxEngine.roles` until announcer refs/fallback are implemented, or make the fallback branch metadata-driven for any role:
   `if _engine_requires_voice_ref(adapter) and not voice_ref and fb_name: ...`
   with a role-usability check for the fallback engine. [ASSUMPTION] This assumes bark is intended/usable as the announcer fallback; if not, Chatterbox must not advertise announcer support yet.

SHOULD-FIX:
1. [_otr_voice_node_common.py::_resolve_clone_ref_path] The gender-agnostic last resort chooses from bank entries before checking whether their `ref_path` exists. If the bank has a mix of stale and valid refs, the deterministic choice can pick a missing file and return `None` even though another valid ref for the same engine exists. Concrete fix: build `role_cands` / fallback `cands` from entries whose resolved `ref_path` exists, or resolve/filter immediately before `Random.choice()`.

2. [eng_chatterbox.py::generate_voice / eng_dia.py::generate_voice] `_load_wav()` failures after an `ok:true` response escape as raw `soundfile`/IO exceptions, not named engine errors. The temp file is removed in `_load_wav()`’s `finally`, but C-7 named fail-closed errors are not preserved for missing/corrupt output files. Concrete fix: wrap the final `_load_wav(resp["out_path"], ...)` call in `try/except Exception` and raise `RuntimeError("Chatterbox worker output load failed: ...")` / `RuntimeError("Dia worker output load failed: ...")`; optionally close/reap the worker if a corrupt ok:true output is considered protocol-corrupt.

OPTIONAL / NICE-TO-HAVE:
1. [_otr_sidecar.py::_env_float] Reject non-finite values (`inf`, `nan`) explicitly, not just non-positive values, before passing them to `queue.get(timeout=...)`.
2. [eng_chatterbox.py / eng_dia.py] Replace `tempfile.mktemp()` with `mkstemp()` or `NamedTemporaryFile(delete=False)` to avoid the usual name-race footgun.

CUT THESE (over-engineering):
1. None. The sidecar lifecycle helpers and response-shape checks are doing real failure containment; I would not cut them before build.