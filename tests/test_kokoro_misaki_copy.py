"""The pack's copy of misaki's Mandarin and Japanese phonemizers (nodes/_otr_audio_engines/_misaki).

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
    assert sorted(unchanged) == ["LICENSE", "data/ja_words.txt", "num2kana.py", "transcription.py"]
    for name in unchanged:
        record = provenance["files"][name]
        assert record["upstream_sha256"] == record["vendored_sha256"], name


def test_the_installed_misaki_is_the_one_the_copy_was_taken_from():
    """When misaki moves on, this says so before its phonemes drift from ours."""
    misaki = pytest.importorskip("misaki")
    root = pathlib.Path(misaki.__file__).resolve().parent
    provenance = json.loads((COPY / "PROVENANCE.json").read_text(encoding="utf-8"))
    for name in ("zh.py", "transcription.py", "cutlet.py", "num2kana.py", "data/ja_words.txt"):
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



# --------------------------------------------------------------------------- #
# Japanese: misaki's Cutlet route, the one JAG2P() takes by default
# --------------------------------------------------------------------------- #
#: Kanji, kana and katakana; corner brackets and full-width digits; the three
#: curly quotes mojimoji maps after NFKC; half-width katakana and full-width
#: Latin; interjections with a small tsu and long vowel marks.
JA_SAMPLES = [
    '\u3053\u3093\u3070\u3093\u306f\u3002\u3053\u3061\u3089\u306f\u5931\u308f\u308c\u305f\u4fe1\u53f7\u3067\u3059\u3002',
    '\u30a2\u30fc\u30ab\u30a4\u30d6\u306f\u3069\u3053\uff1f\u300c\u660e\u65e5\u300d\u3068\u5f7c\u5973\u306f\u8a00\u3063\u305f\u3002\uff12\uff10\uff12\uff16\u5e74\uff19\u6708\uff12\uff18\u65e5\u3002',
    '\u2018Hello\u2019 \u3068 \u201c\u30a6\u30a7\u30f3\u30c7\u30a3\u201d\u304c\u8a00\u3063\u305f\u3002',
    '\uff8a\uff9d\uff76\uff78\uff76\uff80\uff76\uff85\u3068\u5168\u89d2\uff21\uff22\uff23\u3001\u4fa1\u683c\u306f3.5\u5186\u3002',
    '\u3048\u3063\u30fc\uff01\u3084\u3063\u305f\u30fc\uff01\u3059\u3054\u3063\u30fc\u3044\u2026\u2026',
]

#: misaki 0.9.4 JAG2P() on JA_SAMPLES, recorded 2026-09-28 (fugashi 1.5.2,
#: unidic-lite 1.0.8 -- the only UniDic on that venv, as the copy pins).
JA_GOLDEN = [
    'komba\u0274\u03b2a. ko\u02a8i\u027ea \u03b2a \u026f\u0255ina\u03b2a \u027ee ta \u0255i\u014b\u0261o\u02d0 des\u0268.',
    'a\u02d0kaib\u026f \u03b2a doko? \u201cas\u0268\u201d to kano\u02a5o \u03b2a i\u0294ta. \u0272i se\u0274 \u0272i\u02a5\u0268\u02d0\u027eok\u026f ne\u0274 k\u02b2\u0268\u02d0 \u0261a\u02a6\u0268 \u0272i\u02a5\u0268\u02d0ha\u02a8i \u0272i\u02a8i.',
    '` Hello \' to \u03b2end\u02b2i " \u0261a i\u0294ta.',
    'ha\u014bkak\u026fkatakana to \u02a3e\u014bkak\u026f ABC, kakak\u026f \u03b2a sa\u0274. \u0261o e\u0274.',
    'e\u0294\u02d0! ja\u0294ta! s\u0268\u0261o\u0294\u02d0 i......',
]


def _copy_cutlet():
    for name in ("fugashi", "jaconv", "unidic_lite"):
        pytest.importorskip(name)
    from nodes._otr_audio_engines._misaki.cutlet import Cutlet
    return Cutlet()


def test_the_japanese_copy_phonemizes_exactly_as_misaki_does():
    pytest.importorskip("pyopenjtalk")            # misaki.ja imports it at module top
    ja = pytest.importorskip("misaki.ja")
    ours, theirs = _copy_cutlet(), ja.JAG2P()
    for text in JA_SAMPLES:
        assert ours(text) == theirs(text), text


def test_the_japanese_copy_matches_misakis_recorded_output():
    ours = _copy_cutlet()
    assert [ours(text)[0] for text in JA_SAMPLES] == JA_GOLDEN


def test_every_japanese_symbol_is_one_kokoro_has():
    """Every phoneme misaki makes is in the vocabulary. What the text carries
    through unchanged (Latin letters after NFKC, the three quotes the mojimoji
    table maps) is not a phoneme; both backends drop it at tokenization."""
    import unicodedata
    config = pytest.importorskip("kokoro_onnx.config")
    for text, phonemes in zip(JA_SAMPLES, JA_GOLDEN):
        carried = set(unicodedata.normalize("NFKC", text)) | set("`'\"")
        unknown = sorted(set(c for c in phonemes if c not in config.DEFAULT_VOCAB) - carried)
        assert unknown == [], (text, unknown)


def test_the_mojimoji_table_is_what_mojimoji_does_after_nfkc():
    """The copy drops mojimoji (no Python 3.13 wheel). Its two calls follow
    NFKC; over the Basic Multilingual Plane (the full range was measured once,
    2026-09-28) han_to_zen then changes nothing and zen_to_han changes exactly
    the three characters the copy's table maps."""
    import unicodedata
    mojimoji = pytest.importorskip("mojimoji")
    from nodes._otr_audio_engines._misaki import cutlet
    changed = {}
    for cp in range(0x10000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        text = unicodedata.normalize("NFKC", chr(cp))
        half = mojimoji.zen_to_han(text, kana=False)
        if half != text:
            changed[text] = half
        assert mojimoji.han_to_zen(half, digit=False, ascii=False) == half, hex(cp)
    assert {ord(k): v for k, v in changed.items()} == cutlet.ZEN_TO_HAN_AFTER_NFKC


def test_the_japanese_copy_pins_unidic_lite():
    unidic_lite = pytest.importorskip("unidic_lite")
    tagger = _copy_cutlet().tagger
    assert unidic_lite.DICDIR in tagger.dictionary_info[0]["filename"]



# --------------------------------------------------------------------------- #
# Randomized parity against misaki itself, where misaki is installed
# --------------------------------------------------------------------------- #
#: Kana, katakana, kanji, digits of both widths, half-width katakana with its
#: voicing marks, both widths of punctuation, the curly quotes, newlines.
JA_POOL = '\u3042\u3044\u3046\u3048\u304a\u304b\u304d\u304f\u3051\u3053\u3055\u3057\u3059\u305b\u305d\u3063\u3083\u3085\u3087\u3093\u30fc\u30a2\u30a4\u30a6\u30a8\u30aa\u30ab\u30ad\u30af\u30b1\u30b3\u30c3\u30e3\u30e5\u30e7\u30f3\u30f4\u65e5\u672c\u8a9e\u96fb\u6ce2\u653e\u9001\u591c\u4eba\u7269\u5e74\u6708\u5186\u5206\u6642\u56de\u500b0123456789\uff10\uff11\uff12ABCabc\uff21\uff42\uff8a\uff9d\uff76\uff78\uff9e\uff9f\u3001\u3002\uff01\uff1f\u300c\u300d\u2018\u2019\u201c\u201d\u3000 .,!?%\uff05\uff5e\u301c\u2026\n'
#: Common hanzi, digits, Latin, both widths of punctuation, brackets, newlines.
ZH_POOL = '\u6211\u4f60\u4ed6\u7684\u662f\u5728\u6709\u4e0d\u4e86\u4eba\u8fd9\u4e2d\u5927\u6765\u4e0a\u56fd\u4e2a\u5230\u8bf4\u4eec\u4e3a\u5b50\u548c\u4f60\u5730\u51fa\u9053\u4e5f\u65f6\u5e74\u7535\u53f0\u4fe1\u53f7\u6863\u6848\u8239\u957f0123456789ABCabc%\u3001\u3002\uff0c\uff01\uff1f\uff1a\uff1b\u300a\u300b\u300c\u300d\uff08\uff09\u201c\u201d .,!?\n'


def _outcome(fn, text):
    """The phonemes, or the failure: misaki itself raises on a few nonsense
    kana sequences, and the copy must raise the same way there."""
    try:
        return fn(text)
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__, str(exc))


def _random_lines(pool, seed, count=300):
    import random
    rng = random.Random(seed)
    return ["".join(rng.choice(pool) for _ in range(rng.randint(1, 60))) for _ in range(count)]


def test_the_japanese_copy_matches_misaki_on_random_text():
    pytest.importorskip("pyopenjtalk")
    ja = pytest.importorskip("misaki.ja")
    ours, theirs = _copy_cutlet(), ja.JAG2P()
    for text in _random_lines(JA_POOL, 28):
        assert _outcome(ours, text) == _outcome(theirs, text), ascii(text)


def test_the_mandarin_copy_matches_misaki_on_random_text():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        zh = pytest.importorskip("misaki.zh")
    ours, theirs = _copy_g2p(), zh.ZHG2P()
    for text in _random_lines(ZH_POOL, 29):
        assert _outcome(ours, text) == _outcome(theirs, text), ascii(text)


def test_readiness_builds_the_phonemizer_not_just_the_import(monkeypatch):
    """An import alone passes a box whose MeCab dictionary will not open; the
    readiness check builds the phonemizer, so that box refuses at the gate."""
    from nodes._otr_audio_engines import _kokoro_backends as kb
    built = []

    def _builds():
        built.append(True)

    def _cannot_open():
        raise RuntimeError("Failed initializing MeCab")

    monkeypatch.setitem(kb.OWN_G2P_IMPORT, "j", "._misaki")     # an import that works
    monkeypatch.setitem(kb.OWN_G2P, "j", _builds)
    assert kb.own_g2p_ready("j") is True and built == [True]
    monkeypatch.setitem(kb.OWN_G2P, "j", _cannot_open)
    assert kb.own_g2p_ready("j") is False
    assert kb.own_g2p_ready("k") is False
