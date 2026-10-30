# Known test failures

The set is **keyed by machine** (`tests/conftest.py::MACHINE_FACTS`, read with
the first label of the hostname, upper-cased) and **every entry carries the
reason it fails**: a nodeid and a substring of the failure MESSAGE measured for
it on that box. On that machine the listed tests are marked `xfail(strict=True)`
at collection, so the chunk gate ends green when they fail as recorded, and a
listed test that starts PASSING fails the suite until it leaves the table and
this file (the strict pattern praised below). The guard
(`pytest_sessionfinish`) adds the reason check: a listed test that fails WITHOUT
its signature is reported as a regression, so an entry cannot hide a different
break behind a known name. A run cut short counts only what started.

Every entry must also appear here, with its cause, tracking item and exit
condition, in the same commit (`tests/test_known_fail_guard.py` holds this file
and the table together). A machine with no row has an empty set and no
environment change. Do not fill a row with a remembered list: a guessed ledger
hides a real break. The session banner says what the guard thinks the box is:
`[OTR conftest] ... machine=MRKT known_failures=19 OTR_COMFYUI_MODELS_ROOT='...'`.

## IDREAM (the RTX 5080, the dev box): empty

No known failure. Every weight, reference WAV and git ref the suite reaches for
is there, and the pack's default models root (`C:\ComfyUI-Models`) is its root.
The 2026-09-26 chunk gate at `88211ed2` was 16965 passed, 0 failed; the
2.3.43 release run (2026-10-28) had one red, `test_output_tree_containment`'s
remote voice reference, which is not in any set.

## MRKT (the RTX 4060 laptop): 19, recorded 2026-10-30

Measured twice on the same day: identical on 2.3.43 (`d6cb10e74`) and on HEAD
(`bc523b2c4`), full suite, so none is this box's code. Four more failed those
runs and are NOT listed, because they were not this machine's: the HuMo boot
contract passes once the suite reads this box's own models root (the table
sets it: `C:\Users\jeffr\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models`,
where the demo launcher already points the server); the two kokoro cache-key
tests were torch-backend-only tests on an ONNX install, fixed in the same
change; and the ledger path-rebase warning was a real Python 3.13 bug in
`nodes/production_ledger.py`, fixed in the same change.

The MRKT chunk gate, from the pack root:

```
$env:PYTHONUTF8=1
D:\otr-4060-testing\.otr-live-test-venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
```

### The voice references are not on this box (15)

`config/voice_reference_bank.json` names its clone references under
`models/TTS/refs/indextts2/` (`lemmy_algenib_cockney_v1.wav`,
`vz_pd_librivox_mark_f_smith.wav`, `vz_bill_boerst.wav`, ...) and kokoro's
identities as per-voice `.pt` tensors. No `TTS/refs` directory exists on this
box under either root, and kokoro runs on the ONNX backend here (Python 3.13,
`kokoro_onnx` installed, no torch `kokoro`), which keeps its voices in one
file and has no `.pt` per voice. Each resolver fails open as it is built to:
`_resolve_clone_ref_path` returns None, `_bank_identity_fingerprint` returns
NaN, and casting falls back to the bank's stand-ins. Exit: the reference tree
is copied from the 5080 (`C:\ComfyUI-Models\TTS\refs\indextts2\`) under this
box's shared models root -- the clone-reference rows then pass; the two kokoro
`.pt` rows need a torch kokoro install with its voices, or the fingerprint
learning the ONNX voices file (a `nodes/` change, the 5080's). Tracking:
[GO_FORWARD_PLAN](GO_FORWARD_PLAN.md) section 0c, portability (the 4060 owns
the fresh-install path).

- `tests/test_cast_lock_voice_ref_completeness.py::test_castlock_stamps_the_same_reference_the_render_path_would_open`
- `tests/test_cast_lock_voice_ref_completeness.py::test_unspecified_gender_names_the_real_render_reference_without_changing_identity[None]`
- `tests/test_cast_lock_voice_ref_completeness.py::test_unspecified_gender_names_the_real_render_reference_without_changing_identity[]`
- `tests/test_cast_lock_voice_ref_completeness.py::test_unspecified_gender_names_the_real_render_reference_without_changing_identity[   ]`
- `tests/test_cast_lock_voice_ref_completeness.py::test_the_render_fallback_still_works_for_a_row_castlock_never_stamped`
- `tests/test_voice_casting_identity.py::test_a_clone_engine_takes_the_ordinary_draw_and_reaches_a_real_file`
- `tests/test_voice_casting_identity.py::test_an_assigned_row_is_fingerprinted_rather_than_static`
- `tests/test_voice_casting_identity.py::test_the_reserved_recording_is_delivered_to_the_character_it_names[indextts2-auto_registry]`
- `tests/test_voice_casting_identity.py::test_the_reserved_recording_is_delivered_to_the_character_it_names[indextts2-preserve_ledger]`
- `tests/test_voice_casting_identity.py::test_the_reserved_recording_is_delivered_to_the_character_it_names[chatterbox-auto_registry]`
- `tests/test_voice_casting_identity.py::test_the_reserved_recording_is_delivered_to_the_character_it_names[chatterbox-preserve_ledger]`
- `tests/test_voice_casting_identity.py::test_the_reservation_outranks_a_catalogue_assignment[auto_registry]`
- `tests/test_voice_casting_identity.py::test_the_reservation_outranks_a_catalogue_assignment[preserve_ledger]`
- `tests/test_voice_casting_identity.py::test_a_pt_voice_tensor_is_fingerprinted_too` (kokoro `.pt`)
- `tests/test_voice_identity_fix.py::test_a_frozen_ledger_route_does_not_override_the_stamped_voice` (kokoro `.pt`: both sides NaN, and NaN never equals NaN)

### The upscale gate reads the 5080's layout (3)

`spandrel_esrgan._resolve_model` looks in ComfyUI's `folder_paths` (absent in a
bare test) and then `<four parents up from the engine file>/models/upscale_models`,
which is `D:\models\upscale_models` for this checkout and is empty; it never
reads the configured models root. The other test reads
`scripts/_otr_headless_model_paths.yaml`, whose `upscale_models` roots are
`C:/ComfyUI-Models/upscale_models/` and
`C:/Users/jeffr/Documents/ComfyUI/models/upscale_models/`, neither of which
exists here. The weight itself, `RealESRGAN_x2plus.pth`, IS on this box under
the shared models root. Exit, per test: the first and third pass when the
engine consults the configured models root (a `nodes/` change, the 5080's) or
this checkout sits inside a ComfyUI install; the second passes when one of the
yaml's roots exists here with a weight in it. Tracking:
[GO_FORWARD_PLAN](GO_FORWARD_PLAN.md) section 0c, portability.

- `tests/test_upscale_weight_resolution_gate.py::test_every_registered_upscale_engine_can_reach_its_checkpoint`
- `tests/test_upscale_weight_resolution_gate.py::test_the_headless_boot_maps_upscale_models_somewhere_that_has_weights`
- `tests/test_upscale_weight_resolution_gate.py::test_the_pinned_esrgan_filename_matches_what_is_on_disk`

### The git ref `s29-clean-slate-gate` is only on the 5080's clone (1)

The forbidden sweep diffs the tree against that ref (`git diff
s29-clean-slate-gate -- *.py`); it is on neither this clone nor origin
(`git ls-remote origin` has no such name), so git exits 128 before the sweep
runs. Exit: the 5080 pushes the ref (`git push origin s29-clean-slate-gate`)
and this clone fetches it. Tracking: [GO_FORWARD_PLAN](GO_FORWARD_PLAN.md)
section 0c, portability.

- `tests/test_b7_forbidden_sweep.py::test_forbidden_sweep_runs_clean`

## Two gates (operator 2026-09-17)

A ~10 minute full suite is accepted as the **chunk gate** on a mature row.
Iteration uses the **working gate** (seconds).

Working gate, from the pack root:

```
$env:PYTHONUTF8=1
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_working_gate.py
```

Add the nodeid you are in:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_working_gate.py tests\test_cast_lock.py::test_auto_registry_stamps_voice_refs
```

Print the default set: `scripts\otr_working_gate.py --list`

Chunk gate, once per green row (~10 min):

```
$env:PYTHONUTF8=1
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
```

`scripts/otr_working_gate.py` DEFAULT_NODEIDS holds closed-row pins still
worth re-running plus any OPEN [GO_FORWARD](GO_FORWARD_PLAN.md) CODE verify
that is already green. Drop a path in the same commit that closes its row.
Do not add a still-red verify -- that is the row, not the gate.

Worktree-only (not a live-checkout finding):
`test_installed_pack_not_stale` and the two `test_w45_campaign_bank_pinning`
rows.

## The strict xfail opened 2026-08-12 is CLOSED

`tests/test_writer_model_field_shadowing.py::
test_no_writer_model_field_shadows_a_BaseModel_attribute` was marked
`xfail(strict=True)` on 2026-08-12 for PBUG-20260812-02, and the marker was
deleted the same day when the field was fixed.

**The mechanism worked exactly as designed, and is worth keeping as the pattern.**
`strict=True` meant that fixing `CastShape.register` made the test PASS, which
failed the suite and forced the marker out in the same change. The bug could not
be quietly forgotten and the marker could not outlive its cause. That is the
lane matrix's strict unexpected-pass discipline applied to a single defect.

What replaced it is stronger than the original rule: the general check now
sweeps EVERY pydantic model reachable under `nodes/` (92 of them) and asserts no
field default can fail `json.dumps`, with a companion test proving the sweep
actually found the models so a broken import cannot make it vacuously green.
See `apple/PROD_BUG_LOG.md` PBUG-20260812-02 (status FIXED).

When a nodeid enters `MACHINE_FACTS`, document it here IN THE SAME COMMIT: the
machine, the nodeid, why it fails, the tracking item, and the exit condition.
When a machine's row empties again, say so under its heading, as IDREAM's does.

(Restored 2026-08-07: the conftest referenced this file in seven places but it
was never created -- the guard's own instructions pointed at a dead path.)
