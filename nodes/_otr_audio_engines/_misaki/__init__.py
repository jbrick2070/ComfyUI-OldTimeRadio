"""A vendored subset of misaki 0.9.4 (hexgrad, Apache-2.0) for the ONNX Kokoro.

WHY (2026-09-28). ComfyUI Desktop and the portable build run Python 3.13, where
misaki does not install (misaki 0.9.4 and misaki-fork 0.9.6 both declare
Requires-Python <3.13), so Kokoro there runs as kokoro-onnx. The libraries
under misaki's Mandarin and Japanese phonemizers install on 3.13, so the ONNX
backend carries misaki's own code for those two rows and feeds kokoro-onnx the
phonemes the torch pipeline feeds its model.

WHAT IS HERE (PROVENANCE.json has each file's upstream and vendored sha256):

* Mandarin -- ``zh.py``, misaki's ``ZHG2P`` on the one path hexgrad/Kokoro-82M
  uses (``version=None``, the legacy frontend); the version '1.1' branch, which
  imports a frontend not copied here, is removed and ``__init__`` refuses any
  other version. ``transcription.py`` byte for byte (misaki adapted it from
  stefantaubert/pinyin-to-ipa under MIT; that notice is at its top). Needs
  jieba, pypinyin, cn2an and ordered-set.
* Japanese -- ``cutlet.py``, the route ``JAG2P()`` takes by default (adapted by
  misaki from polm/cutlet under MIT; notice at its top), with ``num2kana.py``
  and ``data/ja_words.txt`` byte for byte. Two changes, each measured: mojimoji
  (no Python 3.13 wheel) is gone, because after NFKC its han_to_zen is a no-op
  and its zen_to_han a three-character table; and the MeCab dictionary is
  pinned to unidic-lite, so a ``unidic`` package without its downloaded
  dictionary cannot take MeCab over. Needs fugashi, jaconv and unidic-lite.
* ``LICENSE`` -- misaki's Apache-2.0 licence. (unidic-lite's dictionary is
  installed by the user, under its own BSD terms; it is not copied here.)

The sources are misaki's and keep its IPA and CJK literals, so unlike the rest
of this package they are not ASCII. tests/test_kokoro_misaki_copy.py pins the
copy: the hashes against PROVENANCE.json, the phonemes against misaki itself
wherever misaki is installed and against recorded goldens where it is not, and
the mojimoji table against mojimoji.
"""
