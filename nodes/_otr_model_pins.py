"""The exact bytes OTR downloads for three model sources it does not ship.

Kokoro (its model, its config and its voice files), the Kokoro ONNX export,
and the Whisper model the foley lane uses to find speech were fetched at each
repository's CURRENT revision, with no check on what arrived. Each source is
now pinned to one commit, and every file to its size and SHA-256 (the LFS
digest the Hub publishes; for the few small files that are not LFS objects,
the digest of bytes whose git blob id was checked against the Hub's tree
listing). The repositories are dormant -- Kokoro-82M's last commit is
2025-04-10, the ONNX export's 2025-02-08, faster-whisper-base's 2023-11-23 --
and the pins are those commits, so an install that already has these files
holds exactly these bytes.

Two things follow from a commit pin, both wanted:

* ``hf_hub_download`` with a 40-hex revision returns a cached file with no
  network call at all, where a ``main`` lookup asks the Hub every time.
* Nothing can change what is fetched: not a new commit upstream, not a
  redirected or corrupted transfer, not a stale cache entry. A file whose
  digest is wrong is refused, never used, and there is no fallback to another
  revision.

Stdlib only, and no package-relative import, so ``prestartup_script.py`` can
load it by file before ComfyUI imports any node package. ``huggingface_hub`` is
imported inside ``fetch``.
"""
from __future__ import annotations

import hashlib
import os
import threading

KOKORO = "kokoro"
KOKORO_ONNX = "kokoro_onnx"
WHISPER_BASE = "faster_whisper_base"

#: source -> (repo id, commit, {filename: (size in bytes, sha256)}).
SOURCES = {
    KOKORO: ("hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987", {
        "config.json": (2351, "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"),
        "kokoro-v1_0.pth": (327212226, "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"),
    "voices/af_alloy.pt": (523425, "6d877149dd8b348fbad12e5845b7e43d975390e9f3b68a811d1d86168bef5aa3"),
    "voices/af_aoede.pt": (523425, "c03bd1a4c3716c2d8eaa3d50022f62d5c31cfbd6e15933a00b17fefe13841cc4"),
    "voices/af_bella.pt": (523425, "8cb64e02fcc8de0327a8e13817e49c76c945ecf0052ceac97d3081480e8e48d6"),
    "voices/af_heart.pt": (523425, "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff"),
    "voices/af_jessica.pt": (523435, "cdfdccb8cc975aa34ee6b89642963b0064237675de0e41a30ae64cc958dd4e87"),
    "voices/af_kore.pt": (523420, "8bfbc512321c3db49dff984ac675fa5ac7eaed5a96cc31104d3a9080e179d69d"),
    "voices/af_nicole.pt": (523430, "c5561808bcf5250fe8c5f5de32caf2d94f27e57e95befdb098c5c85991d4c5da"),
    "voices/af_nova.pt": (523420, "e0233676ddc21908c37a1f102f6b88a59e4e5c1bd764983616eb9eda629dbcd2"),
    "voices/af_river.pt": (523425, "e149459bd9c084416b74756b9bd3418256a8b839088abb07d463730c369dab8f"),
    "voices/af_sarah.pt": (523425, "49bd364ea3be9eb3e9685e8f9a15448c4883112a7c0ff7ab139fa4088b08cef9"),
    "voices/af_sky.pt": (523351, "c799548aed06e0cb0d655a85a01b48e7f10484d71663f9a3045a5b9362e8512c"),
    "voices/am_adam.pt": (523420, "ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34"),
    "voices/am_echo.pt": (523420, "8bcfdc852bc985fb45c396c561e571ffb9183930071f962f1b50df5c97b161e8"),
    "voices/am_eric.pt": (523420, "ada66f0eefff34ec921b1d7474d7ac8bec00cd863c170f1c534916e9b8212aae"),
    "voices/am_fenrir.pt": (523430, "98e507eca1db08230ae3b6232d59c10aec9630022d19accac4f5d12fcec3c37a"),
    "voices/am_liam.pt": (523420, "c82550757ddb31308b97f30040dda8c2d609a9e2de6135848d0a948368138518"),
    "voices/am_michael.pt": (523435, "9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37"),
    "voices/am_onyx.pt": (523420, "e8452be16cd0f6da7b4579eaf7b1e4506e92524882053d86d72b96b9a7fed584"),
    "voices/am_puck.pt": (523420, "dd1d8973f4ce4b7d8ae407c77a435f485dabc052081b80ea75c4f30b84f36223"),
    "voices/am_santa.pt": (523425, "7f2f7582fa2b1f160e90aafe6d0b442a685e773608b6667e545d743b073e97a7"),
    "voices/bf_alice.pt": (523425, "d292651b6af6c0d81705c2580dcb4463fccc0ff7b8d618a471dbb4e45655b3f3"),
    "voices/bf_emma.pt": (523420, "d0a423deabf4a52b4f49318c51742c54e21bb89bbbe9a12141e7758ddb5da701"),
    "voices/bf_isabella.pt": (523440, "cdd4c37003805104d1d08fb1e05855c8fb2c68de24ca6e71f264a30aaa59eefd"),
    "voices/bf_lily.pt": (523420, "6e09c2e481e2d53004d7e5ae7d3a325369e130a6f45c35a6002de75084be9285"),
    "voices/bm_daniel.pt": (523430, "fc3fce4e9c12ed4dbc8fa9680cfe51ee190a96444ce7c3ad647549a30823fc5d"),
    "voices/bm_fable.pt": (523425, "d44935f3135257a9064df99f007fc1342ff1aa767552b4a4fa4c3b2e6e59079c"),
    "voices/bm_george.pt": (523430, "f1bc812213dc59774769e5c80004b13eeb79bd78130b11b2d7f934542dab811b"),
    "voices/bm_lewis.pt": (523425, "b5204750dcba01029d2ac9cec17aec3b20a6d64073c579d694a23cb40effbd0e"),
    "voices/ef_dora.pt": (523420, "d9d69b0f8a2b87a345f269d89639f89dfbd1a6c9da0c498ae36dd34afcf35530"),
    "voices/em_alex.pt": (523420, "5eac53f767c3f31a081918ba531969aea850bed18fe56419b804d642c6973431"),
    "voices/em_santa.pt": (523430, "aa8620cb96cec705823efca0d956a63e158e09ad41aca934d354b7f0778f63cb"),
    "voices/ff_siwis.pt": (523425, "8073bf2d2c4b9543a90f2f0fd2144de4ed157e2d4b79ddeb0d5123066171fbc9"),
    "voices/hf_alpha.pt": (523425, "06906fe05746d13a79c5c01e21fd7233b05027221a933c9ada650f5aafc8f044"),
    "voices/hf_beta.pt": (523420, "63c0a1a6272e98d43f4511bba40e30dd9c8ceaf5f39af869509b9f51a71c503e"),
    "voices/hm_omega.pt": (523425, "b55f02a8e8483fffe0afa566e7d22ed8013acf47ad4f6bbee2795a840155703e"),
    "voices/hm_psi.pt": (523351, "2f0f055cea4f1083f4ef127ece48d71606347f6557dbb961c0ca5740a2da485b"),
    "voices/if_sara.pt": (523425, "6c0b253b955fe32f1a1a86006aebe83d050ea95afd0e7be15182f087deedbf55"),
    "voices/im_nicola.pt": (523341, "234ed06648649f9bd874b37508ea17560b9c993ef85b4ddb3e3a71e062bd2c12"),
    "voices/jf_alpha.pt": (523425, "1bf4c9dc69e45ee46183b071f4db766349aac5592acbcfeaf051018048a5d787"),
    "voices/jf_gongitsune.pt": (523351, "1b171917f18f351e65f2bf9657700cd6bfec4e65589c297525b9cf3c20105770"),
    "voices/jf_nezumi.pt": (523420, "d83f007a7f01783b77014561a7d493d327a0210e143440e91c9b697590d27661"),
    "voices/jf_tebukuro.pt": (523435, "0d6917904438aec85f73a6fa1f7ac2be6481aae47c697834936930a91796c576"),
    "voices/jm_kumo.pt": (523425, "98340afd68b1cee84fe0cd95528cfa6d4b39e416aa75a9df64049d52c8b55896"),
    "voices/pf_dora.pt": (523425, "07e4ff987c5d5a8c3995efd15cc4f0db7c4c15e881b198d8ab7f67ecf51f5eb7"),
    "voices/pm_alex.pt": (523425, "cf0ba8c573c2480fc54123683a35cf1e2ae130428e441eb91f9149bdb188a526"),
    "voices/pm_santa.pt": (523430, "d42103169c5c872abbafb9129133af7e942bb9d272c3cc3b95c203e7d7198c29"),
    "voices/zf_xiaobei.pt": (523435, "9b76be63dab4f4f96962030acc0126a9aee9728608fbbe115e2b58a2bd504df6"),
    "voices/zf_xiaoni.pt": (523430, "95b49f169bf1640f4f43c25e13daa39f7b98d15d00823e83ef5c14b3ced59e49"),
    "voices/zf_xiaoxiao.pt": (523440, "cfaf6f2ded1ee56f1ff94fcd2b0e6cdf32e5b794bdc05b44e7439d44aef5887c"),
    "voices/zf_xiaoyi.pt": (523430, "b5235dbaeef85a4c613bf78af9a88ff63c25bac5f26ba77e36186d8b7ebf05e2"),
    "voices/zm_yunjian.pt": (523435, "76cbf8bad35901d011d9628a2fdceb7b4f1f127e7a3269cb393b3941eb7fc417"),
    "voices/zm_yunxi.pt": (523425, "dbe6e1ce7c3dbaf2f5667432947b638b1c6831ccbe154c4610dbcc44f431e27b"),
    "voices/zm_yunxia.pt": (523430, "bb2b03b08e84d64e1214440ce3b624987fac177f2eeb5bab8571799a3d980acd"),
    "voices/zm_yunyang.pt": (523435, "5238ac22e0c7f8b6cdd2eddd6e444b8a700b73c4674d9a047d59a94ff96379a2"),
    }),
    KOKORO_ONNX: ("onnx-community/Kokoro-82M-v1.0-ONNX", "1939ad2a8e416c0acfeecc08a694d14ef25f2231", {
        "onnx/model.onnx": (325532232, "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"),
    }),
    WHISPER_BASE: ("Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66", {
        "config.json": (2309, "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a"),
        "model.bin": (145217532, "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9"),
        "tokenizer.json": (2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
        "vocabulary.txt": (459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
    }),
}


class PinMismatch(RuntimeError):
    """A pinned file on disk, or one just fetched, is not the pinned bytes."""


def repo_and_revision(source: str) -> "tuple[str, str]":
    repo, revision, _files = SOURCES[source]
    return repo, revision


def pin_for(source: str, filename: str) -> "tuple[int, str] | None":
    """``(size, sha256)`` for a pinned file, or None when it has no pin row."""
    return SOURCES[source][2].get(filename)


def filenames(source: str) -> "list[str]":
    return sorted(SOURCES[source][2])


#: (normcased path, size, mtime_ns) -> verdict. A 327 MB file is hashed once
#: per process, and again only if it is replaced.
_VERDICTS: dict = {}
_VERDICTS_LOCK = threading.Lock()


def _digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def file_matches(path: str, size: int, sha256: str) -> bool:
    """True when ``path`` holds exactly ``size`` bytes with digest ``sha256``."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    if st.st_size != size:
        return False
    key = (os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns)
    with _VERDICTS_LOCK:
        known = _VERDICTS.get(key)
    if known is None:
        try:
            known = _digest(path) == sha256.lower()
        except OSError:
            return False
        with _VERDICTS_LOCK:
            _VERDICTS[key] = known
    return known


def mismatch_message(source: str, filename: str, path: str) -> str:
    repo, revision = repo_and_revision(source)
    size, sha256 = pin_for(source, filename)
    return ("%s is not the pinned %s/%s at commit %s (%d bytes, sha256 %s). Delete it "
            "and restart ComfyUI to fetch the pinned copy."
            % (os.path.basename(path), repo, filename, revision[:12], size, sha256))


def check(source: str, filename: str, path: str) -> None:
    """Raise ``PinMismatch`` if the file at ``path`` is not the pinned bytes.

    A filename with no pin row is not checked: it is not one OTR fetched."""
    pin = pin_for(source, filename)
    if pin is not None and not file_matches(path, *pin):
        raise PinMismatch(mismatch_message(source, filename, path))


def cached_copy(source: str, filename: str, cache_dir: "str | None" = None) -> "str | None":
    """A copy already in the Hugging Face cache that holds the pinned bytes.

    Any snapshot counts, not only the pinned commit's: a cache filled by the
    old unpinned code under an earlier commit holds the same bytes for a file
    that did not change, and an offline box must keep working from it. The
    pinned commit's snapshot is tried first. Each candidate is hashed before
    it is trusted; one with other bytes is skipped, never deleted."""
    pin = pin_for(source, filename)
    if pin is None:
        return None
    if cache_dir is None:
        try:
            from huggingface_hub import constants
            cache_dir = constants.HF_HUB_CACHE
        except Exception:  # noqa: BLE001 -- no hub library: nothing cached by it
            return None
    repo, revision = repo_and_revision(source)
    snapshots = os.path.join(str(cache_dir), "models--" + repo.replace("/", "--"), "snapshots")
    try:
        commits = sorted(os.listdir(snapshots), key=lambda name: name != revision)
    except OSError:
        return None
    relative = os.path.join(*filename.split("/"))
    for commit in commits:
        candidate = os.path.join(snapshots, commit, relative)
        if os.path.isfile(candidate) and file_matches(candidate, *pin):
            return candidate
    return None


def fetch(source: str, filename: str, **hub_kwargs) -> str:
    """The pinned file: a verified cached copy, or ``hf_hub_download`` at the
    pinned commit with no token, then verified.

    Returns the local path. Without a ``local_dir`` a cached copy holding the
    pinned bytes is returned with no network call (``cached_copy``). A file
    with the wrong digest raises ``PinMismatch``; if this call wrote it into a
    caller's ``local_dir`` it is removed first. A file that was already there
    is never removed: offline, the hub hands back whatever sits in a
    ``local_dir`` without asking which commit it came from, and a refusal must
    not delete a user's file (Sonnet review, 2026-11-27). A hub-cache copy is
    left to the hub, which keys it by digest. Every other failure is the hub
    library's own exception."""
    pin = pin_for(source, filename)
    if pin is None:
        raise PinMismatch("%s has no pinned copy of %s; OTR fetches only pinned files"
                          % (repo_and_revision(source)[0], filename))
    local_dir = hub_kwargs.get("local_dir")
    if not local_dir:
        cached = cached_copy(source, filename, hub_kwargs.get("cache_dir"))
        if cached:
            return cached
    already_there = bool(local_dir) and os.path.exists(
        os.path.join(str(local_dir), *filename.split("/")))
    from huggingface_hub import hf_hub_download

    repo, revision = repo_and_revision(source)
    hub_kwargs = dict(hub_kwargs)
    hub_kwargs.pop("revision", None)
    hub_kwargs["token"] = False
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision, **hub_kwargs)
    if not file_matches(path, *pin):
        if local_dir and not already_there:
            try:
                os.remove(path)
            except OSError:
                pass
        raise PinMismatch(mismatch_message(source, filename, path))
    return path
