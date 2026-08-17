"""The CROSS-ENGINE Lemmy audition -- one voice, four local engines, frozen lines.

WHY THIS IS A NEW SCRIPT AND NOT A GENERALIZED G1. `scripts/otr_g1_lemmy_audition.py`
is a durable blinded three-arm instrument whose output manifest is referenced BY
SHA inside the one qualified route in `config/cast_pools.py`. Generalizing it
risks rewriting that manifest and invalidating the only real audition evidence
this project has. G1 stays frozen; this runs beside it.

WHAT IT PRODUCES. Two clips per engine -- the frozen neutral line and the frozen
emotional line from `LEMMY_AUDITION_LINES` -- plus a manifest naming every clip,
its sha256 and the identity that produced it. Those hashes are what the
provisional routes cite, which is why the routes may not be written until this
has finished: a route that cites a clip that does not exist yet is a receipt for
audio nobody rendered.

WHAT IT DOES NOT DO. It does not score, decide, or write a qualification receipt.
Only a human listening session can produce a verdict, and when it does, the
result is a QUALIFIED route written by hand.

FOUR ENGINES, THREE WAYS TO SAY WHO IS SPEAKING, AND THE ADAPTERS DISAGREE ABOUT
BOTH:

  * how the identity ARRIVES -- the clone engines take a WAV path, kokoro takes a
    bank voice id, bark takes a preset. This script reads that from the adapter's
    own metadata (`voice_ref_kind` / `voice_ref_field`) rather than a hand-written
    table, because a table is a second source of truth that drifts.
  * what shape the WAVEFORM comes back in -- bark and kokoro return [1, 1, T],
    chatterbox and dia return [C, T]. G1 writes `waveform.cpu().numpy().T`, which
    is correct for exactly one of those. A cross-engine harness that copied that
    line would write silently malformed clips for half the roster, and the failure
    mode is audio that PLAYS -- so the listener hears garbage and blames the
    voice. Every result here goes through canonical_audio/mono_safe first.

    python scripts/otr_lemmy_cross_engine_audition.py                 # preflight
    python scripts/otr_lemmy_cross_engine_audition.py --render
    python scripts/otr_lemmy_cross_engine_audition.py --render --engine dia

Background it and poll the log -- eight clips across four engines, two of which
spawn sidecar worker processes, exceeds the MCP call ceiling comfortably.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

#: Named output root. A rendered asset is a deliverable and goes to its canonical
#: path the FIRST time -- never staged in tmp to be moved later.
CAMPAIGN = "lemmy_cross_engine"
_OUT_DIR = os.path.join(
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes", CAMPAIGN)

#: One seed for every clip, so the arms differ only in the engine.
RENDER_SEED = 20260816

#: WHICH IDENTITY EACH ENGINE SPEAKS AS. This is a decision, not a lookup, and it
#: is the entire content of the sprint -- so it is written down explicitly rather
#: than inferred. What is NOT written down here is HOW each identity is passed to
#: its adapter: that comes from the adapter's own metadata below.
#:
#:   bark        his shipped preset. `lemmy_row()` already pins it at writer time;
#:               bark is here for a comparison clip, not because it has a defect.
#:   kokoro      the warmest British male in the bank. A friendly-broadcaster
#:               cousin, NOT Cockney, accepted under the operator's no-skip
#:               direction -- and the listen page says exactly that.
#:   chatterbox  the approved Cockney reference, mirrored onto the clone engine.
#:   dia         the same reference, same bytes, on the other clone engine.
TARGETS = {
    "bark": "v2/en_speaker_8",
    "kokoro": "bm_george",
    "chatterbox": "cb_lemmy_algenib_cockney_v1",
    "dia": "dia_lemmy_algenib_cockney_v1",
}
ENGINE_ORDER = ("bark", "kokoro", "chatterbox", "dia")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity_kind_for(adapter) -> str:
    """How this adapter is told WHO is speaking -- read from the adapter itself.

    The two declarations are not interchangeable and neither engine family
    carries both: the clone engines declare ``voice_ref_kind='wav_path'`` and no
    row field, while bark and kokoro declare ``voice_ref_field`` and no kind. So
    read whichever is present, and refuse an engine that declares neither rather
    than guessing -- a guess here renders the wrong voice and still sounds fine.
    """
    kind = getattr(adapter, "voice_ref_kind", None)
    if kind == "wav_path":
        return "local_wav"
    field = getattr(adapter, "voice_ref_field", None)
    if field == "voice_ref_id":
        return "bank_voice_id"
    if field == "voice_preset":
        return "preset"
    if field == "provider_voice_id" or kind == "provider_voice_id":
        return "provider_voice"
    raise SystemExit(
        "engine %r declares neither voice_ref_kind nor a known voice_ref_field "
        "-- refusing to guess how its identity arrives"
        % (getattr(adapter, "name", adapter),))


def resolve_identity(engine: str, adapter, bank) -> dict:
    """The concrete thing handed to ``generate_voice``, plus what it is.

    A local identity is resolved to an ABSOLUTE on-disk path through the same
    resolver the render path uses -- bank ref paths are relative to ComfyUI's
    MODELS root, not to this repo, so a repo-root join names a file that has
    never existed.
    """
    from nodes._otr_voice_node_common import _resolve_ref_to_disk

    kind = identity_kind_for(adapter)
    target = TARGETS[engine]
    if kind == "preset":
        return {"identity_kind": kind, "identity_id": target, "argument": target}

    row = next((e for e in bank
                if e.voice_ref_id == target and e.engine == engine), None)
    if row is None:
        raise SystemExit(
            "engine %s: %r is not a bank entry on that engine -- the bank rows "
            "must exist before the audition renders" % (engine, target))
    if kind == "bank_voice_id":
        # kokoro takes the ID and resolves its own `.pt`; prove the file is there
        # anyway, because a missing voice fails deep inside the pipeline.
        disk = _resolve_ref_to_disk(row.ref_path)
        if not (disk and os.path.isfile(disk)):
            raise SystemExit("engine %s: voice file missing on disk: %s"
                             % (engine, disk or row.ref_path))
        return {"identity_kind": kind, "identity_id": target, "argument": target,
                "resolved_path": disk}

    disk = _resolve_ref_to_disk(row.ref_path)
    if not (disk and os.path.isfile(disk)):
        raise SystemExit("engine %s: reference WAV missing on disk: %s"
                         % (engine, disk or row.ref_path))
    claimed = (row.ref_sha256 or "").strip().lower()
    if claimed and claimed not in ("pending", "cloud"):
        actual = sha256_of(disk)
        if actual.lower() != claimed:
            raise SystemExit(
                "engine %s: reference bytes do NOT match the bank\n  actual %s\n"
                "  bank   %s" % (engine, actual, claimed))
    return {"identity_kind": kind, "identity_id": target, "argument": disk,
            "resolved_path": disk}


def preflight(engines) -> dict:
    """Prove every engine can actually run BEFORE any of them renders.

    `assert_usable` proves REGISTRATION and does no IO at all, so it cannot tell a
    working install from a missing one. The honest check is the engine's own
    ``load()``: it spawns the sidecar worker, opens the venv, reads the weights
    and raises with a real message when any of that is not there. Doing it for
    every engine up front means a broken install costs seconds instead of being
    discovered forty minutes into a batch.
    """
    from nodes._otr_audio_engines import get_engine
    from nodes._otr_voice_bank import load_voice_bank

    bank, _sha = load_voice_bank()
    resolved = {}
    for engine in engines:
        adapter = get_engine(engine)
        for label, probe in (("isolated venv python", "_venv_python"),
                             ("worker script", "_worker_script")):
            fn = getattr(adapter, probe, None)
            if callable(fn):
                path = fn()
                if not os.path.exists(path):
                    raise SystemExit("engine %s: %s missing at %s"
                                     % (engine, label, path))
        identity = resolve_identity(engine, adapter, bank)
        try:
            adapter.load()
        except Exception as exc:                        # noqa: BLE001
            raise SystemExit("engine %s: load() failed -- %s" % (engine, exc))
        finally:
            unload = getattr(adapter, "unload", None)
            if callable(unload):
                unload()
        resolved[engine] = identity
        print("  %-11s %-14s %s  OK" % (engine, identity["identity_kind"],
                                        identity["identity_id"]))
    return resolved


def _write_clip(audio, path: str) -> dict:
    """Normalize, then write ONE clip atomically. Returns its manifest row.

    THE RANK NORMALIZATION IS THE POINT. Engines return [1, 1, T] or [C, T] and
    a single hand-written transpose is right for exactly one of them. Everything
    goes through canonical_audio -> mono_safe, the batch is asserted to be one,
    and the file is written from ``waveform[0].T`` -- which is now the same shape
    for every engine because it was made so.
    """
    import soundfile as sf

    from nodes._otr_audio_utils import mono_safe

    normalized = mono_safe(audio)
    wave = normalized["waveform"]
    if wave.shape[0] != 1:
        raise SystemExit("expected a batch of one, got %d" % (wave.shape[0],))
    data = wave[0].detach().cpu().numpy().T             # [T, C], always
    part = path + ".part"
    # FORMAT IS EXPLICIT because the temp name ends `.wav.part`, and soundfile
    # infers the container from the extension -- it raised TypeError on all four
    # engines the first time this ran, after every one of them had already spent
    # its GPU time generating the audio. Subtype is left at the WAV default
    # (PCM_16), which is what the G1 audition clips are, so the listen page
    # compares like with like.
    sf.write(part, data, int(normalized["sample_rate"]), format="WAV")
    os.replace(part, path)                              # atomic: no half clips
    return {
        "file": os.path.basename(path),
        "sha256": sha256_of(path),
        "sample_rate": int(normalized["sample_rate"]),
        "samples": int(data.shape[0]),
    }


def _load_manifest(lines) -> dict:
    path = os.path.join(_OUT_DIR, "MANIFEST.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("lines_version") == lines["lines_version"]:
                return existing
            print("  manifest is for a different lines_version -- starting fresh")
        except Exception:                               # noqa: BLE001
            print("  manifest could not be read -- starting fresh")
    return {
        "campaign": CAMPAIGN,
        "purpose": "Lemmy cross-engine provisional audition -- NOT a "
                   "qualification. No verdict is recorded here or anywhere else "
                   "by this script.",
        "lines_version": lines["lines_version"],
        "neutral_line": lines["neutral_line"],
        "emotional_line": lines["emotional_line"],
        "render_seed": RENDER_SEED,
        "engines": {},
    }


def _save_manifest(manifest) -> str:
    """Write the manifest and return its path. Called after EVERY clip, so a
    batch that dies at engine three keeps what engines one and two earned."""
    manifest["generated_utc"] = _now()
    path = os.path.join(_OUT_DIR, "MANIFEST.json")
    part = path + ".part"
    with open(part, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=True)
        fh.write("\n")
    os.replace(part, path)
    return path


def render(engines, resolved) -> int:
    from config.cast_pools import LEMMY_AUDITION_LINES as LINES
    from nodes._otr_audio_engines import get_engine

    os.makedirs(_OUT_DIR, exist_ok=True)
    manifest = _load_manifest(LINES)
    incomplete = []

    for engine in engines:
        adapter = get_engine(engine)
        identity = resolved[engine]
        row = {
            "identity_kind": identity["identity_kind"],
            "identity_id": identity["identity_id"],
            "engine_impl_version": str(getattr(adapter, "impl_version", "") or ""),
            "declared_sample_rate": int(getattr(adapter, "sample_rate", 0) or 0),
            "clips": {},
            "complete": False,
        }
        manifest["engines"][engine] = row
        print("\n%s -- %s %s" % (engine, identity["identity_kind"],
                                 identity["identity_id"]))
        # ONE ENGINE AT A TIME, and unloaded in a `finally`. Chatterbox and dia
        # own worker PROCESSES, so a leaked adapter is a leaked process holding
        # VRAM on a 16 GB card while the next engine tries to load.
        try:
            adapter.load()
            for kind, text in (("neutral", LINES["neutral_line"]),
                               ("emotional", LINES["emotional_line"])):
                name = "%s_%s.wav" % (engine, kind)
                out = os.path.join(_OUT_DIR, name)
                print("  rendering %s ..." % name)
                prepared = (adapter.prepare_text(text, None)
                            if hasattr(adapter, "prepare_text") else text)
                audio = adapter.generate_voice(
                    prepared, identity["argument"], None, RENDER_SEED)
                clip = _write_clip(audio, out)
                clip["line_kind"] = kind
                row["clips"][kind] = clip
                _save_manifest(manifest)
                print("      -> %s  %s  %.2fs" % (
                    out, clip["sha256"][:16],
                    clip["samples"] / float(clip["sample_rate"] or 1)))
            row["complete"] = len(row["clips"]) == 2
            row["rendered_utc"] = _now()
        except Exception as exc:                        # noqa: BLE001
            # ONE ENGINE FAILING MUST NOT COST THE OTHER THREE. Record it, keep
            # going, and exit non-zero at the end so nothing downstream reads a
            # partial batch as a finished one.
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            print("  FAILED: %s" % row["error"])
        finally:
            unload = getattr(adapter, "unload", None)
            if callable(unload):
                unload()
            _save_manifest(manifest)
        if not row.get("complete"):
            incomplete.append(engine)

    path = _save_manifest(manifest)
    digest = sha256_of(path)
    print("\nmanifest -> %s" % path)
    print("manifest sha256 = %s" % digest)
    print("clips    -> %s" % _OUT_DIR)
    if incomplete:
        print("\nINCOMPLETE: %s -- re-run those with --engine <name>"
              % ", ".join(incomplete))
        return 1
    print("\nEvery requested engine rendered both frozen lines. Nothing here is "
          "a verdict: the routes stay provisional until a human listens.")
    return 0


def _resident_server_warning() -> bool:
    """True when something is already listening on the ComfyUI port.

    A resident server holds around 9.9 GiB of a 16 GiB card, and loading four TTS
    adapters beside it is how a batch dies at engine three with an allocator
    error that looks like an engine bug.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", 8000)) == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="load the engines and render (default: preflight only)")
    ap.add_argument("--engine", action="append", choices=ENGINE_ORDER,
                    help="render only this engine; repeatable. Default: all four")
    ap.add_argument("--allow-resident-server", action="store_true",
                    help="render even though something is listening on :8000")
    args = ap.parse_args(argv)

    engines = [e for e in ENGINE_ORDER if not args.engine or e in args.engine]
    print("Lemmy cross-engine audition -- preflight (%s)" % ", ".join(engines))
    resolved = preflight(engines)

    if not args.render:
        print("\nPreflight OK. Re-run with --render to produce the clips.")
        return 0
    if _resident_server_warning() and not args.allow_resident_server:
        print("\nREFUSING: something is listening on 127.0.0.1:8000. A resident "
              "ComfyUI holds most of the card, and these adapters need it. Tear "
              "it down (CLAUDE.md section 4) or pass --allow-resident-server.")
        return 2
    return render(engines, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
