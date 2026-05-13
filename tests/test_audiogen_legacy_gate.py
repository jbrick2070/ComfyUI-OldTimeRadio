"""S25/AG-2..4 acceptance: legacy ledger.sfx[] obeys the C2 safety
gate; legacy model_id repair is gone; DeprecationWarning fires.

Mirrors the pattern in tests/test_audiogen_writeback_hardening.py
(source-level pins instead of full-stack invocation) so test
collection works without transformers / AudioGen on the test box.

  AG-2 (BUG-LOCAL-217): legacy sfx[] writeback now gates wav_path on
                        save proof + os.path.isfile, same as the v2
                        lines[] writeback C2 already fixed.
  AG-3:                 non-empty legacy ledger.sfx[] fires a
                        DeprecationWarning so the operator sees the
                        legacy path being exercised.
  AG-4 (BUG-LOCAL-218): the `model_id in ["3", "3.0"] -> medium`
                        silent repair is deleted (combo-list at
                        INPUT_TYPES is the loud-fail surface).
"""
from __future__ import annotations

import pathlib
import re

import pytest


_AUDIOGEN_SRC = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes" / "batch_audiogen_generator.py"
).read_text(encoding="utf-8")


# -------------------------------------------------------------------
# AG-2: C2 ghost-path gate landed on the legacy sfx[] loop
# -------------------------------------------------------------------


def test_legacy_sfx_loop_has_os_path_isfile_gate():
    """The legacy ledger.sfx[] writeback must check os.path.isfile
    on cache_path before stamping row[\"wav_path\"], mirroring the
    v2 lines[] gate that C2 landed. Source-level pin: there must be
    at least 2 os.path.isfile(cache_path) call sites in this file
    (one for lines[], one for sfx[])."""
    sites = re.findall(r"os\.path\.isfile\(cache_path\)", _AUDIOGEN_SRC)
    assert len(sites) >= 2, (
        f"AG-2: expected >=2 os.path.isfile(cache_path) call sites "
        f"(legacy sfx[] + v2 lines[]); found {len(sites)}. The "
        f"legacy loop's C2 gate may be missing."
    )


def test_legacy_sfx_loop_blanks_wav_path_on_failure():
    """The failure branch of the legacy sfx[] writeback must stamp
    row['wav_path'] = '' (§6.16), not omit the field or stamp the
    cache_path string."""
    # The legacy loop's blank-on-failure stamp.
    pat = re.compile(r"row\[\"wav_path\"\] = \"\"", re.MULTILINE)
    assert pat.search(_AUDIOGEN_SRC), (
        "AG-2: legacy sfx[] writeback must stamp row['wav_path'] = '' "
        "on the failure branch per §6.16 (matching the v2 lines[] gate)."
    )


def test_legacy_sfx_loop_stamps_sfx_render_status():
    """The legacy loop now stamps sfx_render_status alongside wav_path
    so audit_post_freeze_writeback's enum check can fire on this
    path too."""
    assert 'row["sfx_render_status"]' in _AUDIOGEN_SRC, (
        "AG-2: legacy sfx[] writeback must stamp row['sfx_render_status']."
    )


# -------------------------------------------------------------------
# AG-3: DeprecationWarning on non-empty legacy sfx[]
# -------------------------------------------------------------------


def test_legacy_sfx_non_empty_emits_deprecation_warning():
    """A non-empty ledger.sfx[] means a legacy v1 producer is still
    in the graph. The CD-3 audit will decide whether the path stays
    or gets deleted in S26; the warning surfaces the legacy path so
    the operator sees what's exercised."""
    # Look for the warnings.warn(...) call + DeprecationWarning class.
    pat = re.compile(
        r"(?:_w\.warn|warnings\.warn)\s*\(\s*\n.*ledger\.sfx\[\] is non-empty",
        re.MULTILINE | re.DOTALL,
    )
    assert pat.search(_AUDIOGEN_SRC), (
        "AG-3: a non-empty legacy ledger.sfx[] must trigger a "
        "DeprecationWarning. Source-level pin missing."
    )
    assert "DeprecationWarning" in _AUDIOGEN_SRC, (
        "AG-3: warnings.warn must use DeprecationWarning, not the "
        "default UserWarning class."
    )


# -------------------------------------------------------------------
# AG-4 (BUG-LOCAL-218): silent model_id repair deleted
# -------------------------------------------------------------------


def test_model_id_silent_repair_removed():
    """The combo-list at INPUT_TYPES.optional.model_id is the only
    allowed surface for the model id. The downstream silent repair
    `if str(model_id) in ["3", "3.0"]: model_id = ...` is gone.

    Source-level pin: NO line in active code (i.e., not a comment)
    should rewrite model_id from a bare numeric string.
    """
    # Match a non-comment line: 'if ... model_id ... in ["3", "3.0"]'
    # The forensic comment carries the same pattern but is prefixed
    # by `#`, so the match must start at indentation followed by `if`.
    for line in _AUDIOGEN_SRC.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if (
            stripped.startswith("if ")
            and "model_id" in stripped
            and '["3", "3.0"]' in stripped
        ):
            pytest.fail(
                f"AG-4: silent model_id repair found in active code: "
                f"{line!r}. BUG-LOCAL-218 fix was reverted."
            )


def test_model_id_input_combo_list_intact():
    """The combo-list constraint at INPUT_TYPES must list both
    audiogen-medium + audiogen-small. With the silent repair gone,
    the combo-list is the only enforced surface; missing values
    would let bad widget vectors land at runtime."""
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator
    it = BatchAudioGenGenerator.INPUT_TYPES()
    model_spec = (it.get("optional") or {}).get("model_id")
    assert model_spec is not None, (
        "AG-4: model_id must remain in INPUT_TYPES.optional."
    )
    choices = model_spec[0]
    assert isinstance(choices, list), (
        f"AG-4: model_id type must be a combo-list, got {type(choices).__name__}"
    )
    assert "facebook/audiogen-medium" in choices
    assert "facebook/audiogen-small" in choices
