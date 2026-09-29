"""A vendored subset of misaki 0.9.4 (hexgrad, Apache-2.0) for the ONNX Kokoro.

WHY (2026-09-28). ComfyUI Desktop and the portable build run Python 3.13, where
misaki does not install (misaki 0.9.4 and misaki-fork 0.9.6 both declare
Requires-Python <3.13), so Kokoro there runs as kokoro-onnx. The libraries
under misaki's Mandarin phonemizer -- jieba, pypinyin, cn2an, ordered-set --
install on 3.13, so the ONNX backend carries misaki's own Mandarin code and
feeds kokoro-onnx the phonemes the torch pipeline feeds its model.

WHAT IS HERE (PROVENANCE.json has each file's upstream and vendored sha256):

* ``transcription.py`` -- byte for byte. misaki adapted it from
  stefantaubert/pinyin-to-ipa under MIT; that notice is at its top.
* ``zh.py`` -- misaki's ``ZHG2P`` on the one path hexgrad/Kokoro-82M uses
  (``version=None``, the legacy frontend). The version '1.1' branch, which
  imports a frontend not copied here, is removed and ``__init__`` refuses any
  other version; nothing else changed.
* ``LICENSE`` -- misaki's Apache-2.0 licence.

The sources are misaki's and keep its IPA and CJK literals, so unlike the rest
of this package they are not ASCII. tests/test_kokoro_misaki_copy.py pins the
copy: the hashes against PROVENANCE.json, and the phonemes against misaki itself
wherever misaki is installed and against recorded goldens where it is not.
"""
