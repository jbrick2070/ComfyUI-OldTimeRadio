"""Audition LEMMY through the PRODUCTION path, before vs after the voice fix.

WHY THIS EXISTS, AND WHY G1 COULD NOT DO IT. `otr_g1_lemmy_audition.py` renders
with `emo_vector=None` at a hardcoded RENDER_SEED handed straight to the
adapter. That sidesteps BOTH halves of the voice-identity fix -- with no vector
the emotion ceiling never binds, and with a fixed seed the character-stable seed
never applies -- so re-running it after the fix produced clips BYTE-IDENTICAL to
the 2026-08-10 ones. Byte-identical audio cannot qualify a route against new
code; it is evidence-shaped and empty, which is the exact thing the route
contract exists to refuse.

So this drives the REAL dispatch (`OTR_BatchCharacterVoices` ->
`_render_per_line`): the delivery vector is derived from the line text the way
production derives it, the alpha and the ceiling are read by the adapter the way
production reads them, and the engine seed comes from the seed policy the
profile actually declares. Only `generate_voice` is wrapped, and only to keep a
copy of each clip -- the call itself is the production call.

THREE ARMS, BLINDED, so the two variables separate. A two-arm shipped-vs-pre-fix
comparison confounds them: it varies the seed policy AND the emotion blend at
once, so it can only say the builds differ, never which change did it.

  * `shipped`         -- character seed, ADAPTER DEFAULTS (alpha 1.0, ceiling 0.56)
  * `ceiling_control` -- character seed, ceiling DISABLED. Isolates the ceiling:
                         same seed policy as shipped, so any difference is the
                         emotion budget and nothing else.
  * `pre_fix_control` -- per-line seed, ceiling disabled. The historical arm,
                         i.e. the build before the voice-identity fix.

THE SHIPPED ARM DECLARES NO EMOTION ENVIRONMENT AT ALL. It renders on whatever
`eng_indextts2.py` says today, so this instrument can never certify a build the
constants do not describe -- the failure mode where a qualification record cites
an audition of settings that were never shipped.

THE HEARABLE PREDICTION. On the NEUTRAL line the control arms spend the whole
emotion budget (`calm=1.0` sums to 1.0), leaving `1 - 1.0 == 0` of Lemmy's own
embedding; the shipped build leaves 0.44 of him. The neutral lines are the ones
to listen to hardest: 56 of 57 real character lines sampled from recent episodes
are calm-dominated, so they are what ordinary dialogue actually sounds like, and
the operator has twice described a neutral beat as "Indian rather than Cockney".

THREE LINES PER ARM, not two, so seed stability is audible: the same man should
sound like the same man across all three.

Usage:
    python scripts/otr_lemmy_production_audition.py --render
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nodes._otr_audio_engines.eng_indextts2 import (  # noqa: E402
    EFFECTIVE_EMOTION_MASS_CAP, EMO_ALPHA_DEFAULT,
)

EPISODES = pathlib.Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")

#: Blinding shuffle. Fixed so the mapping is reproducible from the KEY.
SHUFFLE_SEED = 18082026

#: Every environment variable an arm is allowed to control. `_render_arm` POPS
#: all of them before applying an arm's own overrides, so each arm's environment
#: is TOTAL rather than incremental.
#:
#: THIS LIST IS THE BUG FIX. The arms render sequentially in a SHUFFLED order
#: into one process, and the old loop only ever assigned the keys an arm named.
#: An arm that "inherits the defaults" by omitting a key would therefore inherit
#: whatever the previously-rendered arm left behind -- and because the shuffle
#: is what decides which arm ran first, it would not even fail the same way
#: twice. Silent cross-arm contamination is the one defect a blinded instrument
#: must not have, because every downstream conclusion is drawn from it.
ARM_ENV_KEYS = ("OTR_VOICE_CHARACTER_SEED",
                "OTR_INDEXTTS2_EMO_ALPHA",
                "OTR_INDEXTTS2_EMO_MASS_CAP")

#: The three builds under test. Env only -- the adapter reads the emotion knobs
#: per render and the seed policy per line, so no server restart is needed.
#:
#: `shipped` NAMES NO EMOTION VARIABLE ON PURPOSE: absent means "resolve from
#: the adapter", which is exactly what production does.
ARMS = {
    "shipped": {"OTR_VOICE_CHARACTER_SEED": "1"},
    "ceiling_control": {"OTR_VOICE_CHARACTER_SEED": "1",
                        "OTR_INDEXTTS2_EMO_MASS_CAP": "8"},
    "pre_fix_control": {"OTR_VOICE_CHARACTER_SEED": "0",
                        "OTR_INDEXTTS2_EMO_ALPHA": "1.0",
                        "OTR_INDEXTTS2_EMO_MASS_CAP": "8"},
}

#: What each arm is, in plain words, for the KEY the listener opens afterwards.
ARM_DESCRIPTIONS = {
    "shipped": "SHIPPED (character seed, adapter defaults: alpha %s, ceiling %s)",
    "ceiling_control": "CEILING DISABLED (character seed, no emotion ceiling)",
    "pre_fix_control": "PRE-FIX (per-line seed, alpha 1.0, no ceiling)",
}


def _lemmy_reference():
    """Lemmy's QUALIFIED bank reference on indextts2, resolved to disk.

    The route pins `idx_lemmy_algenib_cockney_v1` -- the Algenib Cockney clone,
    which is the arm the operator preferred blind on 2026-08-10 and NOT the
    incumbent he has now twice called Indian.
    """
    from nodes._otr_voice_bank import load_voice_bank
    from nodes._otr_voice_node_common import _resolve_ref_to_disk

    bank, _sha = load_voice_bank()
    entry = next((e for e in bank
                  if e.voice_ref_id == "idx_lemmy_algenib_cockney_v1"
                  and e.engine == "indextts2"), None)
    if entry is None:
        raise SystemExit("Lemmy's indextts2 bank entry is missing")
    path = _resolve_ref_to_disk(entry.ref_path)
    if not path or not os.path.exists(path):
        raise SystemExit("reference does not resolve on disk: %r" % entry.ref_path)
    return entry.voice_ref_id, path


def _ledger(ref_id, ref_path):
    """A minimal ledger casting LEMMY, speaking the FROZEN audition lines.

    A third line is added so seed stability is audible across his own dialogue;
    it is the neutral line's sibling in register, not a new frozen asset.
    """
    from config.cast_pools import LEMMY_AUDITION_LINES as LINES

    return {
        "meta": {"episode_seed": 20260818, "cast_lock_revision": "1"},
        "cast": [{
            "char_id": "c02", "name": "LEMMY", "gender": "male",
            "age_band": "adult", "timbre": ["gravelly", "warm"],
            "voice_ref_id": ref_id, "voice_ref_path": ref_path,
        }],
        "lines": [
            {"line_id": "L1", "char_id": "c02", "speaker_role": "character",
             "text": LINES["neutral_line"], "scene_tension": 0.0},
            {"line_id": "L2", "char_id": "c02", "speaker_role": "character",
             "text": LINES["emotional_line"], "scene_tension": 0.8},
            {"line_id": "L3", "char_id": "c02", "speaker_role": "character",
             "text": "Right you are, Captain. Relay's holding steady.",
             "scene_tension": 0.0},
        ],
    }


def _render_arm(env, ledger):
    """Run the REAL dispatch under `env`; return `(captured, log)`.

    THE ENVIRONMENT IS SET TOTALLY, NOT INCREMENTALLY. Every key in
    :data:`ARM_ENV_KEYS` is POPPED first, then this arm's own overrides are
    applied. Assigning only the keys an arm names would let it inherit the
    previous arm's values -- and since the arms render in a shuffled order in
    one process, an arm meaning "use the adapter defaults" would silently
    render under whichever control ran before it.
    """
    from nodes._otr_audio_engines import get_engine
    from nodes.batch_character_voices import BatchCharacterVoices

    for key in ARM_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in env.items():
        os.environ[key] = value

    adapter = get_engine("indextts2")
    real = adapter.generate_voice
    captured = []

    def _recording_generate(text, ref_clip_path, delivery_vector, seed):
        audio = real(text, ref_clip_path, delivery_vector, seed)
        captured.append({
            "audio": audio, "seed": int(seed),
            "emotion": adapter.emotion_payload(delivery_vector),
        })
        return audio

    adapter.generate_voice = _recording_generate
    try:
        _batch, log, _done = BatchCharacterVoices().generate(
            script_json=json.dumps(ledger), engine="indextts2")
    finally:
        adapter.generate_voice = real

    # A PARTIAL ARM IS NOT EVIDENCE. The manifest used to `zip` lines against
    # captures, which TRUNCATES silently: render two of three lines and the
    # manifest describes a complete arm that never happened, under a sha256
    # a qualification record then cites. Refuse instead.
    expected = len(ledger["lines"])
    if len(captured) != expected:
        raise SystemExit(
            "INCOMPLETE ARM: expected %d rendered line(s), captured %d. "
            "Refusing to write evidence for a render that did not finish."
            % (expected, len(captured)))
    return captured, log


def _write_wav(audio, path):
    import soundfile as sf

    wave = audio["waveform"]
    data = wave.detach().cpu().numpy()
    while data.ndim > 1:
        data = data[0] if data.shape[0] == 1 else data.mean(axis=0)
    sf.write(str(path), data, int(audio["sample_rate"]), subtype="PCM_16")
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", action="store_true",
                    help="load the engine and render (default: preflight only)")
    ap.add_argument("--out-dir",
                    default="lemmy_production_audition_ceiling_2026-08-18")
    args = ap.parse_args(argv)

    ref_id, ref_path = _lemmy_reference()
    print("reference: %s\n           %s" % (ref_id, ref_path))
    ledger = _ledger(ref_id, ref_path)
    if not args.render:
        print("Preflight OK. Re-run with --render to produce the clips.")
        return 0

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = EPISODES / args.out_dir
    key_dir = pathlib.Path(str(out_dir) + "_KEY")

    # EVIDENCE IS NEVER OVERWRITTEN, AND "EVIDENCE" IS NOT JUST THE MANIFEST.
    # An earlier version refused only when MANIFEST.json existed, which left
    # two holes: a directory holding WAVs but no manifest, and the whole _KEY
    # directory, which was created with `exist_ok=True` and then written
    # unconditionally. Both are cited or citable material -- the KEY is what
    # says which arm the operator actually approved -- so a re-render in place
    # destroys the record a qualification cites by sha256.
    for label, path in (("output", out_dir), ("key", key_dir)):
        if path.exists() and not path.is_dir():
            print("REFUSING: %s path %s exists and is not a directory."
                  % (label, path))
            return 2
        if path.is_dir() and any(path.iterdir()):
            print("REFUSING: %s directory %s is not empty -- name a new "
                  "directory rather than overwriting evidence." % (label, path))
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    # Blind the arms: the listener must not be told which build is which.
    labels = ["armX", "armY", "armZ"]
    order = list(ARMS)
    random.Random(SHUFFLE_SEED).shuffle(order)
    mapping = dict(zip(labels, order))
    if len(mapping) != len(ARMS):
        print("REFUSING: %d blinding label(s) for %d arm(s) -- an unlabelled "
              "arm would be dropped silently." % (len(labels), len(ARMS)))
        return 2

    # THE MANIFEST MUST DESCRIBE THE RUNTIME THAT MADE IT. A qualification
    # record cites this file BY SHA256, so anything the record claims about the
    # build has to be provable from inside it. Clip hashes and seeds alone
    # cannot say which alpha, which ceiling, or which adapter produced them.
    from nodes import _otr_voice_route as ROUTE
    ROUTE._LIVE_FINGERPRINT_CACHE.clear()
    fingerprint = ROUTE.live_engine_impl_version("indextts2")
    if not fingerprint:
        print("REFUSING: no live engine fingerprint -- a manifest that cannot "
              "say which build made it is not evidence.")
        return 2
    ref_digest = hashlib.sha256(pathlib.Path(ref_path).read_bytes()).hexdigest()

    manifest = {
        "test": "LEMMY through the production dispatch -- shipped vs two controls",
        "engine": "indextts2",
        "reference": ref_id,
        "reference_sha256": ref_digest,
        "engine_impl_version": fingerprint,
        "engine_profile_id": "char_indextts2_v1",
        "neutral_line": ledger["lines"][0]["text"],
        "emotional_line": ledger["lines"][1]["text"],
        "third_line": ledger["lines"][2]["text"],
        "arms": {},
    }

    for label in labels:
        build = mapping[label]
        print("\n=== rendering %s ===" % label)
        captured, log = _render_arm(ARMS[build], ledger)
        # Resolved INSIDE this arm's environment, so the manifest records what
        # the arm actually rendered under rather than what it asked for.
        from nodes._otr_audio_engines import get_engine
        adapter = get_engine("indextts2")
        resolved = {"emo_alpha": adapter.current_emo_alpha(),
                    "emo_mass_cap": adapter.current_emo_mass_cap()}
        rows = []
        for idx, (line, cap) in enumerate(zip(ledger["lines"], captured), 1):
            name = "%s_line%d.wav" % (label, idx)
            digest = _write_wav(cap["audio"], out_dir / name)
            mass = cap["emotion"]["effective_mass"]
            rows.append({"clip": name, "line_id": line["line_id"],
                         "seed": cap["seed"],
                         "effective_mass": mass,
                         # PER LINE, never a constant. The rescale FLOORS and
                         # the shave steps down, so a line lands at or just
                         # UNDER the ceiling: 0.5590 retains 0.4410, not 0.4400.
                         "speaker_retained": round(1.0 - mass, 4),
                         "sha256": digest})
            print("  %s  seed=%d  mass=%s  speaker=%.4f"
                  % (name, cap["seed"], mass, 1.0 - mass))
        seeds = {r["seed"] for r in rows}
        print("  resolved alpha=%s ceiling=%s | distinct seeds across his "
              "three lines: %d"
              % (resolved["emo_alpha"], resolved["emo_mass_cap"], len(seeds)))
        manifest["arms"][label] = {"resolved": resolved, "clips": rows,
                                   "distinct_seeds": len(seeds)}

    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    key_mapping = {}
    for label, build in mapping.items():
        text = ARM_DESCRIPTIONS[build]
        # The shipped arm names the constants it inherited, so the KEY cannot
        # describe a build the adapter has since moved away from.
        if "%s" in text:
            text = text % (EMO_ALPHA_DEFAULT, EFFECTIVE_EMOTION_MASS_CAP)
        key_mapping[label] = text
    (key_dir / "KEY.json").write_text(json.dumps({
        "WARNING": "Do not open until after listening.",
        "mapping": key_mapping,
        "shuffle_seed": SHUFFLE_SEED,
    }, indent=1), encoding="utf-8")

    print("\nclips -> %s\nKEY   -> %s  (do not open until after listening)"
          % (out_dir, key_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
