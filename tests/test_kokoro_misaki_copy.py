"""The pack's copy of misaki's Mandarin phonemizer (nodes/_otr_audio_engines/_misaki).

misaki does not install on Python 3.13, so the ONNX Kokoro carries misaki
0.9.4's own Mandarin code. What keeps the copy honest: its files hash to
PROVENANCE.json; it phonemizes exactly as misaki does wherever misaki is
installed, and as the goldens recorded from misaki where it is not; and every
symbol it produces is one Kokoro's model has.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import warnings

import pytest

COPY = pathlib.Path(__file__).resolve().parents[1] / "nodes" / "_otr_audio_engines" / "_misaki"

#: Plain sentences, full-width punctuation, book-title and corner brackets,
#: numbers cn2an spells out (a date, a decimal, a price), and a borrowed Latin
#: run the legacy path passes through untouched.
SAMPLES = [
    '\u665a\u4e0a\u597d\u3002\u8fd9\u91cc\u662f\u5931\u843d\u7684\u4fe1\u53f7\uff0c\u6e29\u8fea\u8bf4\uff1a\u975e\u5e38\u611f\u8c22\uff01',
    '\u6863\u6848\u5728\u54ea\u91cc\uff1f\u6211\u4e0d\u77e5\u9053\u3002\u91cc\u5fb7\u8239\u957f\u4e5d\u70b9\u56db\u5341\u4e94\u5206\u5230\u3002',
    '\u300a\u660e\u5929\u300b\u5979\u8bf4\uff1a\u300c\u4e0b\u96e8\u4e86\u300d\uff0c2026\u5e749\u670828\u65e5\uff0c\u4ef7\u683c\u662f3.5\u5143\u3002',
    'Hello \u4e16\u754c, OTR \u7535\u53f0 is live.',
    '\u3010\u65b0\u95fb\u3011\u6211\u4eec\uff08\u542c\u4f17\uff09\u5728\u7b49\u5f85\uff1b\u660e\u5929\u89c1\uff0e',
]

#: misaki 0.9.4 ZHG2P() on SAMPLES, recorded 2026-09-28 (jieba 0.42.1).
GOLDEN = [
    'wa\u2193n\u0282a\u2198\u014b xau\u2193. \uab67\u0264\u2198li\u2193 \u0282\u0268\u2198 \u0282\u0268\u2192lwo\u2198 t\u0264 \u0255i\u2198nxau\u2198, w\u0259\u2192nti\u2197 \u0282wo\u2192: fei\u2192\uab67\u02b0a\u2197\u014bka\u2193n\u0255je\u2198!',
    'ta\u2198\u014ba\u2198n \u02a6ai\u2198 na\u2193li\u2193? wo\u2193 pu\u2198 \uab67\u0268\u2192tau\u2198. li\u2193t\u0264\u2197 \uab67\u02b0wa\u2197n\uab67a\u2193\u014b \u02a8jou\u2193tj\u025b\u2193n s\u0268\u2198\u0282\u0268\u2197u\u2193f\u0259\u2192n tau\u2198.',
    '\u201cmi\u2197\u014bt\u02b0j\u025b\u2192n\u201d t\u02b0a\u2192 \u0282wo\u2192:  \u201c\u0255ja\u2198y\u2193 l\u0264\u201d , \u025a\u2198li\u2197\u014b\u025a\u2198 ljou\u2198nj\u025b\u2197n \u02a8jou\u2193\u0265e\u2198 \u025a\u2198\u0282\u0268\u2197pa\u2192\u027b\u0268\u2198, \u02a8ja\u2198k\u0264\u2197 \u0282\u0268\u2198 sa\u2192ntj\u025b\u2193n u\u2193\u0265\u025b\u2197n.',
    'Hello \u0282\u0268\u2198\u02a8je\u2198, OTR tj\u025b\u2198nt\u02b0ai\u2197 is live.',
    '\u201c\u0255i\u2192nw\u0259\u2197n\u201d wo\u2193m\u0259n (t\u02b0i\u2192\u014b\uab67\u028a\u2198\u014b) \u02a6ai\u2198 t\u0259\u2193\u014btai\u2198; mi\u2197\u014bt\u02b0j\u025b\u2192n \u02a8j\u025b\u2198n.',
]



def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_g2p():
    pytest.importorskip("jieba")
    pytest.importorskip("pypinyin")
    pytest.importorskip("cn2an")
    pytest.importorskip("ordered_set")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import jieba
        from nodes._otr_audio_engines._misaki.zh import ZHG2P
    jieba.setLogLevel(logging.WARNING)
    return ZHG2P()


def test_the_copy_is_the_files_its_provenance_names():
    provenance = json.loads((COPY / "PROVENANCE.json").read_text(encoding="utf-8"))
    for name, record in provenance["files"].items():
        assert _sha(COPY / name) == record["vendored_sha256"], name
    unchanged = [n for n, r in provenance["files"].items() if r["change"].startswith("none")]
    assert sorted(unchanged) == ["LICENSE", "transcription.py"]
    for name in unchanged:
        record = provenance["files"][name]
        assert record["upstream_sha256"] == record["vendored_sha256"], name


def test_the_installed_misaki_is_the_one_the_copy_was_taken_from():
    """When misaki moves on, this says so before its phonemes drift from ours."""
    misaki = pytest.importorskip("misaki")
    root = pathlib.Path(misaki.__file__).resolve().parent
    provenance = json.loads((COPY / "PROVENANCE.json").read_text(encoding="utf-8"))
    for name in ("zh.py", "transcription.py"):
        assert _sha(root / name) == provenance["files"][name]["upstream_sha256"], name


def test_the_copy_phonemizes_exactly_as_misaki_does():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        zh = pytest.importorskip("misaki.zh")
    ours, theirs = _copy_g2p(), zh.ZHG2P()
    for text in SAMPLES:
        assert ours(text) == theirs(text), text


def test_the_copy_matches_misakis_recorded_output():
    """The line above needs misaki, which does not install on Python 3.13 --
    the boxes this copy is for. The goldens hold it there."""
    ours = _copy_g2p()
    assert [ours(text)[0] for text in SAMPLES] == GOLDEN


def test_every_symbol_is_one_kokoro_has():
    config = pytest.importorskip("kokoro_onnx.config")
    for text, phonemes in zip(SAMPLES, GOLDEN):
        latin = set(c for c in text if c.isascii() and c.isalpha())
        unknown = sorted(set(c for c in phonemes if c not in config.DEFAULT_VOCAB) - latin)
        assert unknown == [], (text, unknown)


def test_only_the_legacy_frontend_is_carried():
    ZHG2P = type(_copy_g2p())
    with pytest.raises(ValueError, match="legacy frontend"):
        ZHG2P(version="1.1")
    assert ZHG2P()("   ") == ("", None)
