# 4060 clean-room install: friction log (2026-09-01)

Method: build ComfyUI from the official portable zip in a folder that has never seen OTR,
follow README literally, count every stop. The Desktop install on the same box was left
untouched. Profiles used are headless stand-ins for saved dropdown values (one canonical
JSON, different dropdowns), not new in-app profiles.

Box: MRKT, RTX 4060 Laptop 8 GB, Windows 11, 32 GB RAM, ~44 MB/s to Hugging Face.
Clean room: C:\OTR-CleanRoom, fresh ComfyUI_windows_portable_nvidia v0.34.0
(Python 3.13.14, torch 2.13.0+cu130), fresh HF cache, OTR by git clone at d4e1d5af
(registry install impossible: both registry versions Flagged, latest_version null).
The existing Desktop install on the same box was not touched.

## Friction points, in the order a new user meets them

F1  BLOCKER  `pip install -r requirements.txt` fails on Python 3.13 (the interpreter both
    ComfyUI Desktop and the portable ship today). Cause: `kokoro>=0.7.16`. kokoro 0.7.16
    pins numpy==1.26.4 (no 3.13 wheel); kokoro 0.8+/0.9.x and every misaki>=0.7.5 declare
    Requires-Python <3.13, and misaki[en] pulls spacy/thinc/blis with no 3.13 wheels.
    pip resolves all-or-nothing, so NONE of the 18 requirements install. Verified on the
    box (pip freeze before == after) and against PyPI metadata. Kokoro is the shipped
    default announcer voice and the 8 GB class char voice. --ignore-requires-python does
    not rescue it (numpy pin, then spacy source build). Workaround used for this test:
    install everything except kokoro, run voices on bark.
F2  BOX QUIRK (not OTR) pip on the portable Python raised WinError 448 "untrusted mount
    point" under C:\Users\jeffr\AppData; fixed by pointing TEMP/TMP/PIP_CACHE_DIR at a plain
    folder. Recorded, not counted against the pack.
F3  OK  Stock boot (portable .bat semantics, no PYTHONUTF8): prestartup 23.6 s (fetches 28
    Kokoro voice files at boot even though kokoro cannot be installed), import 4.6 s,
    "All 25 nodes loaded", routes registered. No emoji/cp1252 crash.
F4  DOC  README "git clone" step: fine (94 MiB pack, under a minute). README's Browse
    Templates quick start: example_workflows/ carries only otr_4060_floor.json (audit
    fresh-install-docs-01).
F5  MODELS  No 8 GB profile exists for LTX 2.5, HuMo 1.7B or H3; the only proven 8 GB lane
    is AnimateDiff. The LTX 2.5 and HuMo 1.7B tiers are fetchable only through the pod
    provisioner (scripts/otr_provision.py, which does not ship); the Windows fetcher
    (scripts/otr_fetch_lane_weights.py) has no ltx25 / humo_1.7B lane and defaults its
    models root to C:\ComfyUI-Models rather than the running install's models tree.
    For this test the weights were fetched with the provisioner's own pins (bytes + sha256
    verified): LTX 2.5 set 23.9 GB, HuMo 1.7B set 13.6 GB, z_image int8 set 14.6 GB.
F6  H3 is operator-only (private repacks); not a from-nothing candidate.

## Server launch notes
Detached launches from an SSH session die with the session (Start-Process children were
killed at session end). Task Scheduler (schtasks) is the reliable way to keep a headless
server alive on this box.

F7  GOOD  With weights absent, the pack's own preflight refused the queue BEFORE the writer
    ran and named every missing file ("PREFLIGHT FAIL: profile requires model file(s) the
    running server cannot see: LTX-2.5-Distilled-Q3_K_M.gguf, ..."), with the restart hint.
    One wrinkle: the hint cites scripts/_otr_headless_model_paths.yaml, which registry
    installs do not carry.
F8  BLOCKER (LTX 2.5 only)  The LTX 2.5 lane needs ComfyUI-GGUF cloned at a pinned commit
    AND a patch applied from patches/ (git apply) AND its requirements installed, then a
    restart. README section 2b lists only ComfyUI-AnimateDiff-Evolved; nothing tells a
    Windows user about ComfyUI-GGUF or the patch until the render fails. Every LTX 2.5 core
    class (LTXV*, ManualSigmas, ResizeImageMaskNode, VAEDecodeTiled) is already in stock
    ComfyUI v0.34.0; only UnetLoaderGGUF / CLIPLoaderGGUF are missing.
F9  GOOD  HuMo 1.7B needs NO extra node pack: WanHuMoImageToVideo, AudioEncoderLoader and
    AudioEncoderEncode are all in stock ComfyUI v0.34.0. Its four weights are ungated
    Comfy-Org repackages. On node-pack friction it is the lowest-friction meaty video lane.
F10 BOX QUIRK  pip's console-script PATH scan tripped WinError 448 on a junction in this
    box's PATH (an OpenAI Codex app folder); --no-warn-script-location avoids it.

## Leg results

(in progress -- see the dated entries appended below)
