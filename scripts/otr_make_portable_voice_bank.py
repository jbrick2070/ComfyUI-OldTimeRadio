#!/usr/bin/env python
"""Build a complete portable voice bank around two authorized IndexTTS2 WAVs.

The shipped bank contains operator-local IndexTTS2 references that cannot be
redistributed. A two-row replacement is not sufficient: it would also remove
the Kokoro and other provider rows used by announcer/character casting. This
utility preserves every non-Index row, replaces all IndexTTS2 rows with one
male and one female authorized reference, copies those WAVs under the selected
models root, and writes a new bank atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SHIPPED_BANK = os.path.join(_REPO, "config", "voice_reference_bank.json")
_PRIVATE_ROUTES_UNAVAILABLE_IN_PORTABLE_BANK = (
    "lemmy-indextts2-algenib-cockney-v2",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pcm_contract():
    name = "_otr_portable_pcm_reference"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = os.path.join(_HERE, "otr_pcm_reference.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load PCM reference contract: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_wav(path: str) -> None:
    _pcm_contract().require_usable_wav(path)


def _canonical(path: str) -> str:
    return os.path.normcase(os.path.realpath(
        os.path.abspath(os.path.expanduser(path))))


def _stage_wav(source: str, destination_dir: str, label: str,
               expected_sha256: str) -> str:
    """Copy one source into a unique verified stage without touching finals."""
    os.makedirs(destination_dir, exist_ok=True)
    fd, part = tempfile.mkstemp(
        prefix=".otr_portable_%s_" % label, suffix=".wav.part",
        dir=destination_dir)
    os.close(fd)
    try:
        shutil.copyfile(source, part)
        _validate_wav(part)
        if _sha256(part) != expected_sha256:
            raise ValueError("copied reference WAV failed SHA-256 verification")
        return part
    except Exception:
        try:
            if os.path.exists(part):
                os.unlink(part)
        except OSError:
            pass
        raise


def _publish_asset(part: str, destination: str, expected_sha256: str) -> None:
    """Atomically publish a content-addressed WAV, reusing a valid existing one."""
    if os.path.isfile(destination):
        try:
            _validate_wav(destination)
            if _sha256(destination) == expected_sha256:
                os.unlink(part)
                return
        except ValueError:
            pass
    os.replace(part, destination)
    _validate_wav(destination)
    if _sha256(destination) != expected_sha256:
        raise ValueError("published reference WAV failed SHA-256 verification")


def _voice_bank_authority():
    name = "_otr_portable_voice_bank_authority"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = os.path.join(_REPO, "nodes", "_otr_voice_bank.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load voice-bank authority: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_generated_bank(path: str) -> None:
    authority = _voice_bank_authority()
    entries, _digest = authority.load_voice_bank(path)
    if authority.unavailable_qualified_route_ids(
            source_sha256=_digest) != frozenset(
            _PRIVATE_ROUTES_UNAVAILABLE_IN_PORTABLE_BANK):
        raise ValueError("generated bank lost its exact unavailable route ids")
    index = [row for row in entries if row.engine == "indextts2"]
    if {(row.voice_ref_id, row.gender) for row in index} != {
            ("idx_portable_male_v1", "male"),
            ("idx_portable_female_v1", "female")}:
        raise ValueError("generated bank lost exact portable IndexTTS2 rows")


def build_portable_bank(
        *, shipped_bank: str, models_root: str, male_wav: str, female_wav: str,
        output: str, commercial_clean: bool = False) -> dict:
    shipped_bank = os.path.abspath(os.path.expanduser(shipped_bank))
    output = os.path.abspath(os.path.expanduser(output))
    male_source = os.path.abspath(os.path.expanduser(male_wav))
    female_source = os.path.abspath(os.path.expanduser(female_wav))
    for protected in (shipped_bank, male_source, female_source):
        if _canonical(output) == _canonical(protected):
            raise ValueError("output must not alias an input: %s" % protected)

    with open(shipped_bank, encoding="utf-8") as handle:
        shipped = json.load(handle)
    voices = shipped.get("voices")
    if not isinstance(voices, list):
        raise ValueError("shipped voice bank has no voices list")

    refs_dir = os.path.join(os.path.abspath(os.path.expanduser(models_root)),
                            "TTS", "refs", "indextts2")
    for source in (male_source, female_source):
        if not os.path.isfile(source):
            raise ValueError("reference WAV is missing: %s" % source)
        _validate_wav(source)
    male_hash = _sha256(male_source)
    female_hash = _sha256(female_source)
    if male_hash == female_hash:
        raise ValueError("male and female references must be distinct WAVs")

    male_dest = os.path.join(
        refs_dir, "otr_portable_male_%s.wav" % male_hash)
    female_dest = os.path.join(
        refs_dir, "otr_portable_female_%s.wav" % female_hash)
    for asset in (male_dest, female_dest):
        if _canonical(output) == _canonical(asset):
            raise ValueError("output must not alias a generated asset: %s" % asset)

    male_part = _stage_wav(male_source, refs_dir, "male", male_hash)
    try:
        female_part = _stage_wav(
            female_source, refs_dir, "female", female_hash)
    except Exception:
        try:
            os.unlink(male_part)
        except OSError:
            pass
        raise
    try:
        _publish_asset(male_part, male_dest, male_hash)
        male_part = ""
        _publish_asset(female_part, female_dest, female_hash)
        female_part = ""
    finally:
        for part in (male_part, female_part):
            try:
                if part and os.path.exists(part):
                    os.unlink(part)
            except OSError:
                pass

    retained = [row for row in voices
                if isinstance(row, dict) and row.get("engine") != "indextts2"]

    def row(voice_id: str, gender: str, filename: str, digest: str) -> dict:
        return {
            "voice_ref_id": voice_id,
            "engine": "indextts2",
            "gender": gender,
            "timbre": ["authorized", "portable"],
            "roles": ["char_voice"],
            "age_band": "adult",
            "ref_path": "models/TTS/refs/indextts2/%s" % filename,
            "ref_sha256": digest,
            "commercial_clean": bool(commercial_clean),
        }

    result = {
        "voice_bank_id": "otr_portable_voice_reference_bank_v1",
        "schema_version": shipped.get("schema_version", "1"),
        "unavailable_qualified_route_ids": list(
            _PRIVATE_ROUTES_UNAVAILABLE_IN_PORTABLE_BANK),
        "notes": (
            "Generated by scripts/otr_make_portable_voice_bank.py. Every "
            "non-Index row is preserved; operator-local Index rows are replaced "
            "by the two authorized references below. The private Lemmy-specific "
            "Index route is intentionally unavailable in this portable bank."
        ),
        "voices": retained + [
            row("idx_portable_male_v1", "male", os.path.basename(male_dest),
                male_hash),
            row("idx_portable_female_v1", "female",
                os.path.basename(female_dest), female_hash),
        ],
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fd, part = tempfile.mkstemp(
        prefix=".%s." % os.path.basename(output), suffix=".part",
        dir=os.path.dirname(output))
    os.close(fd)
    try:
        with open(part, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _validate_generated_bank(part)
        os.replace(part, output)
    finally:
        try:
            if os.path.exists(part):
                os.unlink(part)
        except OSError:
            pass
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--male-wav", required=True)
    parser.add_argument("--female-wav", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--commercial-clean", action="store_true",
        help="mark only the supplied references clean after verifying their rights")
    args = parser.parse_args(argv)
    try:
        result = build_portable_bank(
            shipped_bank=_SHIPPED_BANK,
            models_root=args.models_root,
            male_wav=args.male_wav,
            female_wav=args.female_wav,
            output=args.output,
            commercial_clean=args.commercial_clean,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("[portable-bank] FAILED: %s" % exc, file=sys.stderr)
        return 1
    print("[portable-bank] wrote %s rows to %s" %
          (len(result["voices"]), os.path.abspath(args.output)))
    print("[portable-bank] export OTR_VOICE_REFERENCE_BANK=%s" %
          os.path.abspath(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
