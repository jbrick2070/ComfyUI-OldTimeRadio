VERDICT: build-ready as-is? yes. The standalone lab is green, passes all test cases, and is ready to lock as a standalone fixture prototype prior to the gated production transplant.

MUST-FIX BEFORE BUILD:
None — plan converged.

SHOULD-FIX:
1. [nodes.py:98-108] / [_state_files()] Top-level __init__.py omitted from state digest:
   Defect: The state files collector `_state_files()` only adds `nodes.py` and folders `src/` and `fixtures/`, completely missing the top-level `__init__.py`. Edits to the entrypoint file will not trigger `IS_CHANGED` cache invalidations in ComfyUI.
   Fix: Append `LAB_ROOT / "__init__.py"` to the `paths` list in `_state_files()`.

2. [contracts.py:161-183] / [LedgerWritingSpec] Missing consistency validator:
   Defect: `LedgerWritingSpec` does not validate that its top-level identifiers (`source_bank_id`, `story_model_id`) match those nested inside `story_input` or `prompt_profile`. This allows mismatched specifications to pass validation if built manually.
   Fix: Add a Pydantic `model_validator` to verify that `self.story_input.source_bank_id == self.source_bank_id` and `self.story_input.story_model_id == self.story_model_id`.

3. [nodes.py:142-160] / [_validate_public_domain_manifests()] Path traversal vulnerability:
   Defect: The public domain manifest validator checks file existence using `base / rel` without verifying that `rel` is relative to the manifest directory and doesn't contain directory traversal sequences (e.g. `..`).
   Fix: Raise a `RuntimeError` if any path component in `text_files` or `image_files` is absolute or contains `..`.

4. [preview.py:26-58] / [interpret_fixture_material()] Potential TypeError on string concatenation:
   Defect: Concatenating `material.source_title` directly with a string literal assumes the field will never be None. If parsed as None, it triggers a `TypeError`.
   Fix: Use an f-string: `f"Characters orbit the care, recovery, and meaning of a media artifact: {material.source_title or 'untitled'}"`.

OPTIONAL / NICE-TO-HAVE:
- Document in `nodes.py` (or the preview output json) that `_find_source_packet()` returns a single shared fixture placeholder for all models of a source bank rather than model-specific packets.

CUT THESE:
1. [TRANSPLANT_MANIFEST.md:96-120] / [Future Transplant Script] Automation script `plan_transplant.py`:
   - Why safe to cut: For a localized transplant of ~10 target files, creating a custom regex-based python patching script is complex, fragile, and prone to parsing errors. Manual code integration is simpler, safer, and easier to audit.

VERIFY-AT-BUILD checklist:
1. Confirm that `requirements.txt` contains `pydantic>=2.0,<3.0` and that Pydantic is successfully installed in the target production environment.
2. Confirm that `build_legacy_news_mirror()` output keys match the canonical key contract of the production ledger nodes in the target `ComfyUI-OldTimeRadio` codebase before final transplant.
3. Verify that the production workflow JSON edits append widgets without shifting existing widget indices.

[ASSUMPTION] We assume that Pydantic is already present in the target environment or will be installed as part of standard custom node setup.
[ASSUMPTION] We assume that ComfyUI's execution engine hashes widget values in addition to `IS_CHANGED` return values for cache key calculation.
[ASSUMPTION] We assume the production repository `ComfyUI-OldTimeRadio` is in a sibling directory or that the transplant path is configured correctly.
