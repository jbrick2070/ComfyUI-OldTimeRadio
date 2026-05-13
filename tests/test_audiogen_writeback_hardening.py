"""C2 (S24, 2026-05-13) -- AudioGen writeback hardening pins.

Three bundled bugs covered:
  (a) writeback never stamps sfx_render_status -> stamp per item with
      one of: "ok_cache" / "ok" / "fallback_silence" /
      "fallback_output_shape" / "error".
  (b) short-output fallback wrote silence to canonical cache_path,
      poisoning future cache hits. The fallback now writes to
      <cache_dir>/_fallback/ instead (or skips).
  (c) _save_wav returned None on both success and failure; writeback
      stamped sfx_wav_path unconditionally. Now returns bool; writeback
      gates sfx_wav_path on save_ok AND os.path.isfile.

Source-level pins so collection works without transformers/AudioGen.
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pytest


_AUDIOGEN_SRC = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes" / "batch_audiogen_generator.py"
).read_text(encoding="utf-8")


# -------------------------------------------------------------------
# (c) _save_wav returns bool
# -------------------------------------------------------------------


def test_save_wav_returns_bool():
    """The signature must declare ``-> bool``."""
    from nodes.batch_audiogen_generator import _save_wav
    sig = inspect.signature(_save_wav)
    # Source-level check on the annotation (inspect resolves type
    # hints to strings under ``from __future__ import annotations``)
    assert sig.return_annotation in (bool, "bool"), (
        f"C2: _save_wav must declare -> bool; got {sig.return_annotation!r}"
    )


def test_save_wav_source_has_explicit_return_true_and_false():
    """The body must have explicit ``return True`` (success path) and
    ``return False`` (exception path).  Pre-C2 the function fell off
    the end on both paths and returned None."""
    assert "return True" in _AUDIOGEN_SRC, (
        "C2: _save_wav must explicitly return True after os.replace."
    )
    assert "return False" in _AUDIOGEN_SRC, (
        "C2: _save_wav must explicitly return False from the except branch."
    )


# -------------------------------------------------------------------
# (b) short-output fallback uses _fallback/ subdir, not canonical
# -------------------------------------------------------------------


def test_short_output_fallback_uses_fallback_subdir():
    """The fix routes the fallback save to a sibling ``_fallback/``
    directory under the original cache_dir, instead of overwriting
    the canonical cache_path. Source-level pin."""
    # The new code path must contain the literal substring.
    assert '"_fallback"' in _AUDIOGEN_SRC, (
        "C2: short-output fallback must save to a _fallback/ subdir, "
        "not the canonical cache_path. Either the dir name changed or "
        "the canonical-cache-poisoning fix was reverted."
    )


def test_short_output_fallback_stamps_fallback_output_shape_status():
    """The render-status field must take the value
    ``"fallback_output_shape"`` on the short-output path."""
    assert '"fallback_output_shape"' in _AUDIOGEN_SRC, (
        "C2: short-output path must stamp _render_status="
        "fallback_output_shape so downstream consumers see the "
        "degraded state."
    )


def test_short_output_fallback_does_not_save_to_cache_path():
    """The post-fix flow guards the canonical _save_wav(item['cache_path']...)
    call inside an `else:` branch off `short_output_fallback`.  This
    source-level check confirms the guard exists."""
    # Look for the structural pattern: short_output_fallback flag +
    # else branch carrying _save_wav to cache_path.
    assert "short_output_fallback" in _AUDIOGEN_SRC, (
        "C2: short_output_fallback flag missing -- the fix was reverted."
    )
    # The canonical save must be guarded.
    pat = re.compile(
        r"else:\s*\n[^\n]*save_ok = _save_wav\(\s*\n[^\n]*item\[\"cache_path\"\]",
        re.MULTILINE,
    )
    assert pat.search(_AUDIOGEN_SRC), (
        "C2: canonical _save_wav(item['cache_path'], ...) must be "
        "guarded inside an else branch off short_output_fallback. "
        "Either the branch shape changed or the guard was removed."
    )


# -------------------------------------------------------------------
# (a) sfx_render_status stamped on writeback
# -------------------------------------------------------------------


def test_writeback_stamps_sfx_render_status_on_every_row():
    """The ledger.lines[] writeback block must put sfx_render_status
    into line_fields unconditionally."""
    assert '"sfx_render_status": render_status' in _AUDIOGEN_SRC, (
        "C2: writeback must stamp sfx_render_status on every row. "
        "Either the field key changed or the stamp was removed."
    )


def test_writeback_defaults_status_for_pre_stamped_cache_hits():
    """Items that came from cache might not have explicit
    ``_render_status`` (the cache-hit branch stamps "ok_cache" now,
    but a future refactor might skip it). The writeback fallback
    expression should be:

        item.get("_render_status") or (
            "ok_cache" if item.get("had_cache_hit_at_resolve")
            else "ok"
        )
    """
    pat = re.compile(
        r'render_status = item\.get\("_render_status"\) or \(\s*\n'
        r'\s*"ok_cache"',
        re.MULTILINE,
    )
    assert pat.search(_AUDIOGEN_SRC), (
        "C2: writeback's render_status fallback expression missing. "
        "The default must distinguish ok_cache (had_cache_hit) from ok."
    )


def test_writeback_gates_sfx_wav_path_on_save_proof():
    """The writeback must check both save_ok / had_cache_hit AND
    os.path.isfile(cache_path) before stamping sfx_wav_path."""
    assert "os.path.isfile(cache_path)" in _AUDIOGEN_SRC, (
        "C2: writeback must verify cache_path exists on disk before "
        "stamping sfx_wav_path. Either the check was removed or the "
        "function name changed."
    )


def test_writeback_blanks_sfx_wav_path_on_failure():
    """On save-failure / silence fallback, sfx_wav_path must land
    "" not be omitted from the dict (§6.16 convention)."""
    # The else branches should set the empty string.
    assert 'line_fields["sfx_wav_path"] = ""' in _AUDIOGEN_SRC, (
        "C2: failure paths must stamp sfx_wav_path=\"\" per §6.16, "
        "not omit the field."
    )


# -------------------------------------------------------------------
# (a-fallback) ImportError silence-fallback stamps fallback_silence
# -------------------------------------------------------------------


def test_import_error_fallback_stamps_fallback_silence():
    """S17.2 + C2 (S24): the allow_silence_fallback=True path stamps
    _render_status="fallback_silence" on each affected item so the
    writeback ledger row carries the degraded state."""
    pat = re.compile(
        r'render_queue\[idx\]\["_render_status"\] = "fallback_silence"',
        re.MULTILINE,
    )
    assert pat.search(_AUDIOGEN_SRC), (
        "C2: ImportError silence-fallback must stamp "
        "_render_status='fallback_silence' on every affected item."
    )
