"""The FOLEY STEM format, and the one place that knows it.

A foley stem is the audio LTX 2.5 already computes while it renders a clip --
footsteps, room tone, a score written for the exact picture on screen -- kept
instead of discarded and written beside the clip as a 16-bit PCM WAV. It is
mixed UNDER the frozen episode master at ``OTR_MasterAudioMux``, at the fixed
0.20 foley / 0.80 master the operator ruled on 2026-08-26.

THIS IS NOT THE SFX BED. The SFX bed was separately GENERATED effects from a
dedicated model; it was ripped on 2026-08-06 and is staying dead, and
``tests/test_rip_sfx_bed_guard.py`` exists to keep it that way. Nothing here is
restored from it, no constant is inherited from it, and every field name in this
module is ``foley_`` for exactly that reason. Operator, 2026-08-26: *"sfx bed is
different than foley bed, i won't get the two confused."*

WHY THIS IS ITS OWN MODULE RATHER THAN A FEW HELPERS IN THE ADAPTER. Three
different stages touch a stem -- the ENGINE writes one per rendered segment, the
COVERAGE assembler cuts and concatenates them into one per beat, and the MUX
reads and mixes them -- and the format is the contract between all three. A
stem written by one reader's assumptions and read by another's is the defect
class this module removes: there is exactly one writer, one reader, and one
frames-to-samples conversion, and everything uses them.

Cold-import clean: stdlib only at module scope. ``numpy`` is resolved lazily
inside the two functions that need it, so importing this module costs nothing on
a box that is only listing nodes.
"""
from __future__ import annotations

import hashlib
import logging
import os
import wave

_LOG = logging.getLogger("OTR.foley")

#: THE FIXED MIX -- an operator ruling, not a knob (2026-08-26): *"ducking
#: fixed i would say .20 foley .80 voice."* Linear coefficients applied to the
#: FULL master (dialogue, procedural room tone, themes and music cues together),
#: because ``OTR_EpisodeAssembler`` folds all of it into ONE WAV and there is no
#: separate voice bus to duck against. No sidechain, no envelope following, no
#: per-beat loudness analysis: a static gain is deterministic, so the same
#: inputs give the same master every single time.
#:
#: THE MASTER IS NOT ATTENUATED BY THE FOLEY'S PRESENCE. Voice holds 0.80
#: whether or not a stem exists for a given beat, so a beat with no foley does
#: not get louder than its neighbours.
#:
#: NOT the retired SFX bed's 0.45, and the difference is the point rather than
#: an accident: this bed plays UNDER dialogue continuously, where that one
#: played in the gaps.
#:
#: Ledger-driven ducking -- 0.20 under a line and higher between lines, driven
#: from the frozen ledger rather than by detecting the voice -- is DEFERRED, not
#: rejected. The operator: *"let's start simple 80/20."* 0.20 remains the speech
#: floor in that later design, so these two constants survive it.
FOLEY_GAIN = 0.20
MASTER_GAIN_UNDER_FOLEY = 0.80

#: The receipt keys a foley row carries. ONE tuple, consumed by the adapter, the
#: coverage assembler, the manifest builder and the mux alike, so a key cannot
#: be added in one place and silently dropped in another.
FOLEY_RECEIPT_KEYS = (
    "foley_path", "foley_sha256", "foley_samples", "foley_sample_rate",
    "foley_channels", "foley_duration_s",
)


#: THE TWO LANES THAT KEEP THE MODEL'S AUDIO, and the gains each one mixes at:
#: ``internal engine id -> (foley gain, master gain)``.
#:
#: BOTH ARE THE SAME MECHANISM WITH DIFFERENT CONSTANTS -- that is the whole
#: reason they share this module. Operator, 2026-08-26: *"foley and mime, we
#: need this feature for both."*
#:
#: * ``ltx25_foley_plus`` -- 0.20 / 0.80. A bed UNDER the episode. The master
#:   gain is GLOBAL: it applies to the whole timeline, not only to the beats
#:   that carry a bed, because RULING 1 is explicit that *"voice holds 0.80
#:   whether or not a foley stem exists for that beat, so a beat without foley
#:   does not get louder"*.
#: * ``ltx25_mime`` -- 1.00 / 0.00. A silent performance carrying the video's
#:   own score. The master gain is PER-WINDOW: it zeroes only the mime beats'
#:   own samples, because engines are ROLE-WIDE and an episode with a mime role
#:   still has roles that speak -- all of them sharing ONE master WAV. Zeroing
#:   globally would silence the entire episode.
#:
#: THE TTS AND MUSIC FOR A MIME BEAT ARE STILL GENERATED, and then multiplied
#: by zero. That waste is deliberate (RULING 4, superseding the 2026-08-10
#: "mime generates no TTS" brief): nothing has to happen before the master
#: freezes, because nothing is being REPLACED -- the master is simply
#: attenuated to zero in that window at mux time, exactly as foley_plus
#: attenuates it to 0.80. Same pipeline, same code path, one different
#: constant. It deletes a whole node and an execution-order inversion.
FOLEY_LANE_GAINS = {
    "ltx25_foley_plus": (0.20, 0.80),
    "ltx25_mime": (1.00, 0.00),
}

#: The lane whose master gain applies GLOBALLY rather than per-window. Exactly
#: one lane does this and the distinction is a ruling, not an implementation
#: detail -- see the table above.
GLOBAL_MASTER_GAIN_LANES = frozenset({"ltx25_foley_plus"})

#: Kept as the name three call sites already read. It is the FOLEY-BED lane
#: specifically; :data:`FOLEY_LANE_GAINS` is the full roster.
FOLEY_ENGINE_ID = "ltx25_foley_plus"
MIME_ENGINE_ID = "ltx25_mime"


def is_foley_route(video_policy_json):
    """True when ANY role on this episode renders with the foley engine.

    ANY, not all, and there is only one right answer here: the episode has ONE
    master WAV. A mixed-role episode where a single role carries foley still
    needs the assembler to write a pre-loudness provisional master and the mux
    to do the delivery levelling, because the bed lands in that one file.

    IDS ARE RESOLVED BEFORE THEY ARE COMPARED. ``effective_video_models`` can
    hold a public menu string, an internal id, or a legacy alias depending on
    how the policy was frozen, so a bare ``== FOLEY_ENGINE_ID`` would answer
    False for ``'ltx25_high_foley_plus (16:9)'`` -- an episode that really is on
    the route. Both callers use THIS function for exactly that reason.

    Pure and total: unparseable, empty or absent policy means "not a foley
    route", which is the historical behaviour every existing episode expects.
    """
    import json

    if not video_policy_json:
        return False
    try:
        policy = json.loads(video_policy_json)
    except (TypeError, ValueError):
        return False
    if not isinstance(policy, dict):
        return False
    effective = policy.get("effective_video_models")
    if not isinstance(effective, dict):
        return False
    try:
        from .._otr_shared.public_engines import resolve_engine_id
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared.public_engines import resolve_engine_id  # type: ignore
    return any(resolve_engine_id(value) in FOLEY_LANE_GAINS
               for value in effective.values())


def route_lane_ids(video_policy_json):
    """Every audio-keeping lane this episode's roles resolve to, as a set.

    :func:`is_foley_route` answers yes/no; this answers WHICH, which the mix
    needs because the two lanes attenuate the master differently -- one
    globally, one per window.

    Pure and total, for the same reason and in the same way as its sibling: an
    unreadable policy yields an empty set, which is the historical no-bed path.
    """
    import json

    if not video_policy_json:
        return frozenset()
    try:
        policy = json.loads(video_policy_json)
    except (TypeError, ValueError):
        return frozenset()
    if not isinstance(policy, dict):
        return frozenset()
    effective = policy.get("effective_video_models")
    if not isinstance(effective, dict):
        return frozenset()
    try:
        from .._otr_shared.public_engines import resolve_engine_id
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared.public_engines import resolve_engine_id  # type: ignore
    return frozenset(
        resolved for resolved in
        (resolve_engine_id(v) for v in effective.values())
        if resolved in FOLEY_LANE_GAINS)


class FoleyStemError(RuntimeError):
    """A stem could not be written, read, or cut to its picture.

    Its own type so callers can name the failure without catching every
    ``RuntimeError`` in the render path. Every raise site says what disagreed
    with what -- a stem is picture-locked audio, and the only useful message
    about one is which clock it stopped agreeing with.
    """


def sha256_of_file(path):
    """SHA-256 of a file, streamed.

    The stem's identity, and the mux hashes it into ``IS_CHANGED`` -- so a
    rewritten stem at the same path is a different cache key rather than a
    stale mix served silently.
    """
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def samples_per_frame(sample_rate, fps):
    """Exact integer samples per video frame, or a NAMED refusal.

    THE GUARD IS NOT DECORATION. The decode's sample rate is read off the
    WEIGHTS at runtime (``audio_vae.first_stage_model.output_sample_rate``),
    never from a constant, so it can change under us with a re-quant. Both rates
    seen in practice divide evenly by 25 fps (44100 -> 1764, 48000 -> 1920); one
    that did not would make every frame-to-sample conversion here a rounding,
    and a rounding compounds across a chained beat until the foley slides
    audibly against the picture it was generated for. Fail rather than round.
    """
    rate, rate_fps = int(sample_rate), int(fps)
    if rate <= 0 or rate_fps <= 0 or rate % rate_fps != 0:
        raise FoleyStemError(
            "a foley stem decoded at %r Hz against a %r fps picture clock, "
            "which is not an exact number of samples per frame. NO ROUNDING "
            "-- a fractional conversion slides the foley a little further off "
            "its own picture with every segment" % (sample_rate, fps))
    return rate // rate_fps


def write_pcm16_wav(path, samples, sample_rate):
    """Write a float waveform ``(channels, n)`` as a 16-bit PCM WAV.

    stdlib ``wave`` only, and 16-bit PCM specifically, because that is exactly
    what ``OTR_EpisodeAssembler`` writes the episode master as
    (``scene_sequencer.py``). One format for every OTR audio artifact means the
    mux never has to ask what it is reading before it can mix.

    Returns ``(n_samples, n_channels)`` as WRITTEN, so a caller's receipt is the
    file's own shape rather than the array it hoped it wrote.
    """
    import numpy as np

    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise FoleyStemError(
            "a foley stem must be (channels, samples); got shape %r"
            % (tuple(arr.shape),))
    pcm = (arr * 32767.0).clip(-32768, 32767).astype(np.int16)
    n_channels, n_samples = int(pcm.shape[0]), int(pcm.shape[1])
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(n_channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        # WAV wants interleaved (samples, channels) in C order.
        handle.writeframes(pcm.T.copy(order="C").tobytes())
    return n_samples, n_channels


def read_pcm16_wav(path):
    """Read a stem back as ``(float array (channels, n), sample_rate)``.

    The exact inverse of :func:`write_pcm16_wav`, and the only reader the foley
    path uses -- so a stem is always read by the code that knows how it was
    written. A file of any other sample width is a NAMED refusal rather than a
    silent misread: reading 24-bit bytes as 16-bit produces plausible-looking
    noise, which is the worst possible failure for an audio path.
    """
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise FoleyStemError(
                "%r is %d-bit PCM; the foley path writes and reads 16-bit only"
                % (str(path), handle.getsampwidth() * 8))
        n_channels = int(handle.getnchannels())
        rate = int(handle.getframerate())
        raw = handle.readframes(handle.getnframes())
    flat = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return flat.reshape(-1, n_channels).T.copy(), rate


def durable_foley_dir():
    """``<episode>/audio/foley/`` -- where every foley stem is written, first try.

    THE LOCATION IS PART OF THE FORMAT, which is why it lives here beside the
    reader and the writer rather than in whichever caller happened to need it
    first. Two different stages write stems (the engine, per rendered segment;
    the coverage assembler, per beat) and a third reads them a whole render
    later, so a stem placed by one stage's idea of "durable" and looked for by
    another's is the same class of defect this module exists to remove.

    Resolved exactly the way ``OTR_EpisodeAssembler`` resolves the episode
    master WAV: the in-flight ledger lives at
    ``<ep_root>/audio/<id>_ledger.json``, so its parent IS the episode audio
    directory. ``in_flight_ledger_path()`` takes NO arguments -- it is a
    singleton lookup, not a path builder.

    NEVER a request-id ``pending_*`` directory: ``OTR_SignalLostVideo`` (node
    12, order 13) renames that tree before the mux reads anything out of it, so
    a stem addressed there is a stem the mux opens by a name that no longer
    exists.

    AND NEVER TMP. ``persist_episode_clips`` moves only ``clip['path']``, and
    ``OTR_MasterAudioMux.mux()`` sweeps ``episodes/_shared/tmp`` after muxing --
    so a stem staged in scratch is deleted somewhere between being written and
    being needed. A missing ledger is therefore a NAMED refusal rather than a
    fallback: a foley route that silently delivers no foley is exactly the false
    green this path must not produce.
    """
    import pathlib

    try:
        from .. import _otr_ledger as _OTRL       # type: ignore
    except ImportError:                            # pragma: no cover -- flat
        import _otr_ledger as _OTRL                # type: ignore
    ledger_path = _OTRL.in_flight_ledger_path()
    if ledger_path is None:
        raise FoleyStemError(
            "there is no in-flight ledger, so there is no durable "
            "episodes/<ep>/audio/ to write a foley stem into. NO TMP FALLBACK "
            "-- the mux sweeps _shared/tmp after muxing, so a stem staged "
            "there is deleted before it can be mixed")
    dest = pathlib.Path(ledger_path).parent / "foley"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def assemble_beat_foley_segments(segments, out_path, *, expect_frames, fps):
    """Cut and concatenate per-SEGMENT stems into ONE beat-level stem.

    The audio sibling of ``wan_shared.assemble_beat_segments``, and deliberately
    the same shape: ``segments`` is a sequence of ``(path, drop_head,
    keep_frames)`` in play order, exactly the tuples the video assembler is
    handed for the same beat. Returns the ``foley_*`` receipt dict.

    THE CUTS HAPPEN HERE AND ONLY HERE. The engine emits a stem exactly as long
    as the mp4 it wrote and applies no head or tail work of its own; the
    coverage plan's ``drop_head`` / ``keep_frames`` are this function's job, in
    SAMPLE space, at exact integer offsets. Doing it in both places would cut
    picture-locked audio twice -- and ``drop_head=1`` on every chained successor
    means it would be wrong on every beat longer than one rung, not just
    occasionally.

    SINGLE-SEGMENT BEATS COME THROUGH HERE TOO. A one-segment plan can still owe
    a tail trim (a beat shorter than the 97-frame rung is rounded up to it), and
    routing only multi-segment beats through the cutter is how the surplus
    survives on exactly the beats nobody thinks to check.

    TRANSACTIONAL, like its video sibling: the assembled length is proved
    against ``expect_frames`` BEFORE the file is handed back, and a failure
    removes the partial output rather than leaving a half-right stem for the mux
    to find. A stem that disagrees with its beat's frame count is not a receipt
    problem -- it is audio that will play under the wrong picture.
    """
    import numpy as np

    rows = [(str(path), int(drop), int(keep)) for path, drop, keep in segments]
    if not rows:
        raise FoleyStemError("beat foley assembly was handed NO segments")

    pieces = []
    rate = None
    channels = None
    for path, drop_head, keep_frames in rows:
        if not path or not os.path.isfile(path):
            raise FoleyStemError(
                "foley stem %r is missing, so this beat's bed cannot be "
                "assembled. NO FALLBACK -- silence here would be "
                "indistinguishable from a lane that worked" % path)
        arr, stem_rate = read_pcm16_wav(path)
        if rate is None:
            rate, channels = stem_rate, int(arr.shape[0])
        elif stem_rate != rate or int(arr.shape[0]) != channels:
            # A rate or channel-count change partway through a beat is Bug
            # Bible 12.29's silent mismatch: it concatenates without complaint
            # and plays at the wrong speed or on one side only.
            raise FoleyStemError(
                "foley stem %r is %d Hz x%dch but this beat started at "
                "%d Hz x%dch -- a beat's stems must share one format"
                % (path, stem_rate, int(arr.shape[0]), rate, channels))
        step = samples_per_frame(rate, fps)
        start = drop_head * step
        stop = start + keep_frames * step
        if keep_frames <= 0:
            raise FoleyStemError(
                "foley stem %r was asked to keep %d frame(s)"
                % (path, keep_frames))
        if arr.shape[-1] < stop:
            raise FoleyStemError(
                "foley stem %r holds %d sample(s) but its segment asks for "
                "frames [%d, %d) of it, which needs %d. The stem is shorter "
                "than the picture it belongs to"
                % (path, int(arr.shape[-1]), drop_head,
                   drop_head + keep_frames, stop))
        pieces.append(arr[:, start:stop])

    assembled = (pieces[0] if len(pieces) == 1
                 else np.concatenate(pieces, axis=-1))
    step = samples_per_frame(rate, fps)
    want = int(expect_frames) * step
    if int(assembled.shape[-1]) != want:
        raise FoleyStemError(
            "assembled foley is %d sample(s) for a beat of %d frame(s), which "
            "needs %d. The cut plan and the beat disagree about this beat's "
            "length" % (int(assembled.shape[-1]), int(expect_frames), want))

    try:
        n_samples, n_channels = write_pcm16_wav(out_path, assembled, rate)
    except Exception:
        # TRANSACTIONAL: never leave a partial stem where the mux will find it
        # and mix it. A missing stem fails loud one stage later; a truncated one
        # plays quietly under the wrong picture.
        try:
            os.remove(str(out_path))
        except OSError:
            pass
        raise

    _LOG.info(
        "[OTR foley] assembled %d segment(s) -> %d sample(s) x%dch @%d Hz "
        "(%.3f s, %d frame(s)) -> %s",
        len(rows), n_samples, n_channels, rate, n_samples / float(rate),
        int(expect_frames), os.path.basename(str(out_path)))
    return {
        "foley_path": str(out_path),
        "foley_sha256": sha256_of_file(out_path),
        "foley_samples": int(n_samples),
        "foley_sample_rate": int(rate),
        "foley_channels": int(n_channels),
        "foley_duration_s": float(n_samples) / float(rate),
    }


def conform_to_master(stem, stem_rate, master_rate, master_channels):
    """Bring one stem to the master's channel count and sample rate.

    EXPLICIT, BEFORE ANY MIXING (Bug Bible 12.29). A silent rate mismatch does
    not raise -- it plays the bed at the wrong speed and pitch under a picture
    that is still correct, which sounds like a bad model rather than a bug. A
    silent channel mismatch is worse: the bed lands on one side only, or vanishes
    when the file is folded to mono downstream.

    Channels are matched by TILING mono up and by AVERAGING extra channels down,
    both of which are energy-sane for a bed sitting 14 dB under the dialogue.
    Rate is matched by the sequencer's own resampler -- scipy polyphase, with a
    named linear fallback -- so the foley path and the dialogue path resample
    audio with ONE implementation rather than two that can diverge.

    Returns ``(array (master_channels, n), notes)`` where ``notes`` lists what
    had to be changed, for the mux's receipt. An empty list means the stem
    already matched, which is the expected case.
    """
    import numpy as np

    notes = []
    arr = np.asarray(stem, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]

    if int(stem_rate) != int(master_rate):
        try:
            from ..scene_sequencer import _resample_audio
        except ImportError:  # pragma: no cover -- flat test imports
            from scene_sequencer import _resample_audio  # type: ignore
        arr = np.stack(
            [_resample_audio(np.ascontiguousarray(row), int(stem_rate),
                             int(master_rate)) for row in arr])
        notes.append("resampled %d->%d Hz" % (int(stem_rate),
                                              int(master_rate)))

    have = int(arr.shape[0])
    want = int(master_channels)
    if have != want:
        if have == 1:
            arr = np.repeat(arr, want, axis=0)
        elif want == 1:
            arr = arr.mean(axis=0, keepdims=True)
        else:
            arr = np.repeat(arr.mean(axis=0, keepdims=True), want, axis=0)
        notes.append("channels %d->%d" % (have, want))
    return arr, notes


def mix_foley_under_master(master, master_rate, rows, *, fps,
                           lane_ids=frozenset()):
    """``master * envelope + bed`` -- the operator's fixed ratios, once.

    ``master`` is ``(channels, n)`` float, ``rows`` is the audio-keeping subset
    of the clip manifest (each needing ``foley_path``, ``start_s``,
    ``frame_count`` and ``engine_id``). ``lane_ids`` is the set of audio-keeping
    lanes the episode's ROLES resolve to, which is how a globally-attenuating
    lane can be known about even on a beat that carries no stem. Returns
    ``(mixed array, stats)``.

    WHY AN ENVELOPE AND NOT A SCALAR, and it is two rulings rather than a
    generalisation for its own sake:

    * ``ltx25_foley_plus`` attenuates the master GLOBALLY to 0.80. RULING 1 is
      explicit -- *"voice holds 0.80 whether or not a foley stem exists for
      that beat, so a beat without foley does not get louder"* -- so this one
      really is a single scale across the whole timeline.
    * ``ltx25_mime`` attenuates PER WINDOW, to 0.00. Engines are ROLE-WIDE, so
      an episode with a mime role still has roles that speak, and all of them
      share ONE master WAV. A global zero would silence the episode; what
      RULING 4 describes is zeroing *"a beat's window"*.

    One array expresses both without a special case: it starts at the global
    gain and mime rows punch their own windows down to zero.

    SPLAT ONTO ZEROS, NEVER CONCATENATE. The manifest legitimately contains
    OVERLAPPING positioned rows -- the opening/body and body/closing seams
    overlap by design, and the assembler crossfades across them -- so "each beat
    follows the last" is simply false about this timeline. Every stem is written
    at its own absolute offset into a silence bed the length of the master.

    ADDITIVE, NOT COPY-OVER. Where two beats overlap, both are audible for the
    overlap, which is what the picture does too. Copy-over would silence the
    outgoing beat's bed the instant the next one started.

    THE OFFSET IS INTEGER FRAMES, NEVER A FLOAT MULTIPLICATION.
    ``round(start_s * fps) * (master_rate // fps)`` lands every bed on a frame
    boundary; ``int(start_s * master_rate)`` accumulates a sub-sample error per
    beat with no defined rounding, which is exactly the drift this whole feature
    would be blamed for.

    A BEAT WITH NO STEM IS SILENCE, and it stays silence. Carrying a neighbour's
    bed forward would place audio generated for one picture under a different
    one -- the single worst thing a picture-conditioned bed can do.
    """
    import numpy as np

    arr = np.asarray(master, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    channels, length = int(arr.shape[0]), int(arr.shape[-1])
    step = samples_per_frame(master_rate, fps)

    bed = np.zeros_like(arr)
    # THE GLOBAL FLOOR OF THE ENVELOPE. Driven by the episode's ROLES rather
    # than by which beats produced a stem, because RULING 1's "whether or not a
    # foley stem exists for that beat" is precisely a statement about beats
    # that have none. A row's own engine_id is the fallback for a manifest
    # handed in without a policy.
    row_lanes = {str((r or {}).get("engine_id") or "") for r in rows}
    present = set(lane_ids) | (row_lanes & set(FOLEY_LANE_GAINS))
    global_master_gain = 1.0
    for lane in sorted(present & GLOBAL_MASTER_GAIN_LANES):
        global_master_gain = min(global_master_gain, FOLEY_LANE_GAINS[lane][1])
    envelope = np.full(length, global_master_gain, dtype=np.float32)

    placed, skipped, notes, overrun, unpositioned = 0, 0, [], 0, 0
    lanes_mixed = {}
    for row in rows:
        path = str((row or {}).get("foley_path") or "")
        if not path:
            continue
        space = str((row or {}).get("start_s_space") or "")
        if space and space != "master_mix":
            # PROVEN, NOT ASSUMED. A scene-audio offset in a master-mix
            # timeline is every bed early by the length of the opening theme.
            raise FoleyStemError(
                "foley row %r is positioned in %r space, not master_mix. Its "
                "bed would land under the wrong picture by however long the "
                "opening theme runs" % (os.path.basename(path), space))
        if not os.path.isfile(path):
            raise FoleyStemError(
                "foley stem %r is named by the manifest but is not on disk. NO "
                "SILENT SKIP -- a bed that quietly drops a beat is a lane "
                "reporting success for work it did not deliver" % path)
        stem, stem_rate = read_pcm16_wav(path)
        stem, stem_notes = conform_to_master(stem, stem_rate, master_rate,
                                             channels)
        notes.extend("%s: %s" % (os.path.basename(path), n)
                     for n in stem_notes)

        frames = int((row or {}).get("frame_count") or 0)
        if frames > 0 and int(stem.shape[-1]) > frames * step:
            raise FoleyStemError(
                "foley stem %r holds %d sample(s) for a %d-frame beat whose "
                "own slot is %d. A stem longer than its own picture has been "
                "cut against the wrong plan"
                % (os.path.basename(path), int(stem.shape[-1]), frames,
                   frames * step))

        # A ROW WITH NO POSITION IS SKIPPED, NOT GUESSED AT AND NOT FATAL.
        #
        # THIS WAS A HARD FAILURE UNTIL A LIVE LEG KILLED AN EPISODE WITH IT
        # (2026-08-26, 3h17m of render lost at the very last node). The guard's
        # reasoning was right and is kept: position zero would stack every
        # unplaced bed on top of the opening, so a position is never invented.
        # What was wrong was treating "no position" as impossible.
        #
        # IT IS ROUTINE. A `music_inter` beat is a video-only bridge -- ledger
        # b006 of that episode reads `start_s=None, dur_s=None, text=''`,
        # "Bridge to the next phase with music only" -- and it occupies NO time
        # in the master mix at all: its neighbours b005 (33.936 + 3.499) and
        # b007 (37.435) are exactly contiguous. Such a beat still renders a
        # picture, and on a foley lane it still produces a stem, but there is
        # no window in the master WAV to splat that stem into.
        #
        # So skipping is the only honest answer -- and skipping is NOT
        # guessing. It is LOUD and COUNTED (`unpositioned` rides the stats into
        # the mux receipt) so that a lane quietly dropping half its beds is
        # visible, which is the failure the hard raise was really guarding
        # against.
        raw_start = (row or {}).get("start_s")
        try:
            start_s = float(raw_start)
        except (TypeError, ValueError):
            unpositioned += 1
            _LOG.warning(
                "[OTR foley] stem %s has no master-mix position "
                "(start_s=%r) -- NOT mixed. A beat with no slot in the master "
                "cannot carry a bed; this is normal for a music_inter bridge "
                "and a fault for anything else",
                os.path.basename(path), raw_start)
            continue
        offset = int(round(start_s * float(fps))) * step
        if offset < 0 or offset >= length:
            skipped += 1
            _LOG.warning(
                "[OTR foley] stem %s starts at %.3f s, outside the %.3f s "
                "master -- not mixed", os.path.basename(path), start_s,
                length / float(master_rate))
            continue
        end = min(offset + int(stem.shape[-1]), length)
        if end - offset < int(stem.shape[-1]):
            overrun += int(stem.shape[-1]) - (end - offset)

        # WHICH LANE THIS ROW IS, AND THEREFORE AT WHAT GAINS. Read off the
        # row's own engine_id rather than assumed from the episode: a mixed
        # episode can legitimately carry both lanes, on different roles, and
        # they attenuate the master differently.
        lane = str((row or {}).get("engine_id") or "")
        if lane not in FOLEY_LANE_GAINS:
            raise FoleyStemError(
                "manifest row %r carries a foley stem but its engine_id is "
                "%r, which is not an audio-keeping lane. NO GUESS -- mixing a "
                "stem at gains chosen for a different lane is how a bed ends "
                "up over or under the dialogue it was balanced against"
                % (os.path.basename(path), lane))
        foley_gain, master_gain = FOLEY_LANE_GAINS[lane]
        bed[:, offset:end] += stem[:, :end - offset] * foley_gain
        # A PER-WINDOW LANE PUNCHES ITS OWN WINDOW DOWN. The global lane has
        # already set the floor everywhere and must not re-apply it here.
        if lane not in GLOBAL_MASTER_GAIN_LANES:
            envelope[offset:end] = np.minimum(envelope[offset:end],
                                              master_gain)
        lanes_mixed[lane] = lanes_mixed.get(lane, 0) + 1
        placed += 1

    if overrun:
        # Real and benign: the last beat's picture can run marginally past the
        # audio the composite was built to. Trimming the tail is right; doing
        # it silently is not.
        _LOG.info("[OTR foley] %d bed sample(s) past the end of the master "
                  "were trimmed", overrun)

    # The bed already carries each row's own foley gain, so this is the one
    # place the two sides meet and neither is scaled twice.
    mixed = arr * envelope + bed
    stats = {
        "placed": placed, "skipped": skipped, "trimmed_samples": overrun,
        # Beats that produced a stem but own no window in the master mix --
        # a music_inter bridge is the normal case. Counted so a lane silently
        # dropping beds is visible in the receipt rather than only in a log.
        "unpositioned": unpositioned,
        "conform_notes": notes,
        "lanes": dict(sorted(lanes_mixed.items())),
        "global_master_gain": float(global_master_gain),
        "muted_samples": int((envelope <= 0.0).sum()),
        "bed_peak": float(abs(bed).max()) if bed.size else 0.0,
    }
    return mixed, stats


__all__ = [
    "FOLEY_GAIN", "MASTER_GAIN_UNDER_FOLEY", "FOLEY_RECEIPT_KEYS",
    "FOLEY_ENGINE_ID", "is_foley_route",
    "FoleyStemError", "sha256_of_file", "samples_per_frame",
    "durable_foley_dir",
    "write_pcm16_wav", "read_pcm16_wav", "assemble_beat_foley_segments",
    "conform_to_master", "mix_foley_under_master",
]
