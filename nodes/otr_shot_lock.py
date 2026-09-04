"""OTR_ShotLock -- the video-phase lock authority (A-S1/W1).

The video analogue of ``OTR_CastLock``: it runs AFTER audio timing freezes
(gated on ``OTR_EpisodeAssembler``'s ``audio_done``, out3), reads the frozen
ledger, and stamps ONE ``ledger['video']`` section -- the audio-derived clip
budget, the DAG-validated ordered ``execution_groups``, and the per-shot rows
with their creative directives. It mirrors ``OTR_CastLock``'s I/O exactly,
including the load-bearing ``done`` STRING gate that downstream ordering needs.

It OWNS prompt generation (it supersedes ``OTR_VideoPlan``): for character-
bearing beats it runs the M4 per-beat derivation -- one batched LLM call on the
writer's slot (V-11, NO new model_id widget), mirroring the Meta-brief protocol
of ``nodes/_otr_music_prompt.py`` -- deriving ``{expression, motion, camera}``
and composing a rich ``text_prompt`` + a structured ``creative`` sidecar into
``ledger['video'].shots[].creative``. The derivation is fail-soft: empty /
unparseable / truncated model output reseeds (max 2) then falls back to a
DETERMINISTIC template (``{appearance}, {setting}, {beat_text}``); a consistency
check that the prompt carries the cast's core traits + the brief setting WARNs +
falls back on a miss. It NEVER aborts the episode or touches the frozen audio
(invariant V-1). Cheap families (abstract / still / station-card) get NO creative
LLM call.

Determinism (V-7): per-shot ``request_hash`` mixes brief + cast content hashes +
beat_id + char_id; the prompt hash is taken AFTER the call. 3D ``expression`` is
a DRIVER-channel directive, never part of any mesh/cache key.

Import-time is side-effect-free; module scope imports only stdlib + the dep-free
shared resolver/registry/schemas. The LLM is resolved lazily (and is injectable
for tests). UTF-8, no BOM, ASCII source.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re

log = logging.getLogger("OTR")

from ._otr_shared import resolver as _resolver
from ._otr_shared.role_compat import Role
from ._otr_shared import role_slots as _role_slots

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

# ---------------------------------------------------------------------------
# Role mapping + which roles are "character-bearing" (get the rich derivation)
# ---------------------------------------------------------------------------

#: ledger ``speaker_role`` -> video role.
#: BUG 1 (2026-06-20): ``"character"`` is the CANONICAL writer speaker_role for a
#: dialogue line (set in OTR_LedgerScriptWriter / _otr_outline, compared in
#: _otr_anti_loop / _otr_ledger_reviewer). "char_voice"/"dialogue" stay as aliases.
#: rip-sfx-broll (2026-07-01): the "sfx" entry + the _DEFAULT_VIDEO_ROLE
#: fallback (retired_role_b) were REMOVED with their roles -- an unmapped
#: speaker_role now FAILS LOUD in :func:`_video_role_for_line` (NO FALLBACKS).
SPEAKER_TO_VIDEO_ROLE = {
    "announcer": Role.ANNOUNCER_VISUAL.value,
    "music": Role.MUSIC_VISUAL.value,
    "music_open": Role.MUSIC_VISUAL.value,
    "music_close": Role.MUSIC_VISUAL.value,
    "music_inter": Role.MUSIC_VISUAL.value,
    "character": Role.CHARACTER_VIDEO.value,
    "char_voice": Role.CHARACTER_VIDEO.value,
    "dialogue": Role.CHARACTER_VIDEO.value,
}

#: Only these roles receive the M4 creative LLM derivation. Everything else is
#: a cheap family (radio floor / abstract) and gets NO creative LLM call.
CHARACTER_BEARING_ROLES = frozenset({Role.CHARACTER_VIDEO.value})

_FALLBACK_SETTING = "a vintage radio studio"


def _video_role_for_line(line: dict) -> str:
    role = str((line or {}).get("speaker_role") or "").strip().lower()
    mapped = SPEAKER_TO_VIDEO_ROLE.get(role)
    if mapped is None:
        raise ValueError(
            f"OTR_ShotLock: line "
            f"{str((line or {}).get('line_id') or '?')!r} carries unmapped "
            f"speaker_role {role!r} (known: {tuple(SPEAKER_TO_VIDEO_ROLE)}). "
            f"The 'sfx' role + the retired_role_b default were removed "
            f"2026-07-01 (rip-sfx-broll) -- NO FALLBACKS; regenerate the "
            f"episode with the current writer."
        )
    return mapped


# ---------------------------------------------------------------------------
# Brief / cast readers (Meta-brief protocol, never crash on absent brief)
# ---------------------------------------------------------------------------


def _read_setting(meta: dict) -> str:
    """Setting string from the Meta brief, via the brief-reader protocol when
    available; tolerant fallback otherwise.

    Terms are normalised through `spoken_term` because this string is joined
    into a model-facing prompt and the brief emits identifier case
    (PBUG-20260903-04). All four consumers of this field normalise the same way,
    or the same episode gets spelled two ways in two prompts.
    """
    from ._otr_brief_reader import spoken_term

    terms = (meta or {}).get("story_brief_terms") or {}
    setting = []
    if isinstance(terms, dict):
        raw = terms.get("setting") or []
        if isinstance(raw, list):
            setting = [spoken_term(t) for t in raw if spoken_term(t)]
    if not setting:
        try:
            from ._otr_brief_reader import _read_brief_field

            raw = _read_brief_field(meta, "setting", default=[])
            if isinstance(raw, list):
                setting = [spoken_term(t) for t in raw if spoken_term(t)]
            elif isinstance(raw, str) and spoken_term(raw):
                setting = [spoken_term(raw)]
        except Exception:  # noqa: BLE001
            pass
    return ", ".join(setting[:2]) if setting else _FALLBACK_SETTING


#: Clauses in a cast description that NO picture model can render.
#:
#: The writer authors one character description that serves several consumers,
#: and casting deliberately asks it for a `Voice:` clause -- TTS needs a timbre.
#: Every caller of `_appearance_for_char`, though, is a VIDEO or STILL prompt
#: composer (8 sites across otr_shot_lock.py and otr_meta_brief_image_prompt.py,
#: checked), and none of them can draw a voice.
#:
#: On the JOINT AUDIO-VIDEO lanes it is worse than dead weight. Those models
#: read one prompt for picture AND sound, and the standing ruling is that voice
#: and identity words never reach them because they get VOCALIZED. Measured on
#: shipped ledgers before this fix: `Voice:` appeared in 18 of 20 joint-AV
#: prompts (90%) -- e.g. ltx25_mime receiving "Voice: raspy, deliberate,
#: punctuated by heavy breaths." Every one of those was the banned case.
#:
#: `Face:` and `Presence:` are deliberately KEPT. Presence is behaviour --
#: "alert, scanning the room with restless, guarded intensity" -- which is
#: exactly the kind of thing a video model can act on.
_UNRENDERABLE_APPEARANCE_CLAUSE = re.compile(
    r"\bVoice:\s*[^.]*(?:\.|$)", re.IGNORECASE)


def _strip_unrenderable_appearance(text: str) -> str:
    """Drop the clauses of a cast description a picture model cannot draw."""
    if not text:
        return text
    cleaned = _UNRENDERABLE_APPEARANCE_CLAUSE.sub(" ", str(text))
    return " ".join(cleaned.split()).strip(" ,")


def _appearance_for_char(ledger: dict, char_id: str) -> str:
    """Appearance LOOKUP by char_id (alias-safe), never by display name.

    Non-renderable clauses (see :data:`_UNRENDERABLE_APPEARANCE_CLAUSE`) are
    removed: every consumer of this function draws a picture.
    """
    if not char_id:
        return ""
    try:
        from . import _otr_ledger_consumers as _OTRLC

        entry = _OTRLC.cast_lookup(ledger, char_id)
    except Exception:  # noqa: BLE001
        entry = {}
        for c in (ledger or {}).get("cast") or []:
            if isinstance(c, dict) and str(c.get("char_id") or "") == str(char_id):
                entry = c
                break
    # character_description added 2026-06-10 (operator look-QA): the writer's
    # RICH per-character physical description lives under that key on the
    # cast row; without it the M4 prompts lost the character grounding.
    base = ""
    for key in ("portrait_prompt", "appearance", "description",
                "character_description"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            base = val.strip()
            break
    if not base:
        name = entry.get("name")
        base = str(name) if name else ""
    base = _strip_unrenderable_appearance(base)
    # The opt-in outfit LOCK was ripped 2026-08-27 (operator: "outfits yeah i
    # didnt even know we had outfits"). It was `OTR_OUTFIT_LOCK`, default OFF
    # and set by no profile or launcher, so it never once ran -- and its call
    # site was already guarded to return this same `base` untouched whenever
    # the module was absent. Removing it is byte-identical by that guard's own
    # design; appearance has always been exactly what the writer described.
    return base


def _content_hash(obj) -> str:
    try:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:  # noqa: BLE001
        blob = repr(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Beat extraction + audio-derived clip budget (Smart Generation Limit)
# ---------------------------------------------------------------------------


def _same_frozen_episode(wire: dict, disk: dict) -> tuple[bool, str]:
    """Prove that a post-audio disk ledger belongs to ``wire``.

    ``meta.freeze_timestamp`` is minted once by Phase 10 and survives the normal
    pending-id -> final-title rename, so it is the strongest identity receipt at
    this join. If either side has one, both sides must have the same value.
    Older ledgers may fall back to an exact, non-empty episode id only when
    neither carries a receipt. A mismatch is never guessed through title/slug
    similarity: that would let a stale sibling episode donate audio truth.
    """
    wire_meta = wire.get("meta") if isinstance(wire.get("meta"), dict) else {}
    disk_meta = disk.get("meta") if isinstance(disk.get("meta"), dict) else {}
    # A replay workspace shares its source's freeze receipt on purpose; the
    # workspace id is what tells them apart (campaign item 0, 2026-09-02).
    wire_ws = str(wire_meta.get("replay_workspace_id") or "").strip()
    disk_ws = str(disk_meta.get("replay_workspace_id") or "").strip()
    if wire_ws or disk_ws:
        if not (wire_ws and disk_ws and wire_ws == disk_ws):
            return False, (
                "replay_workspace_id mismatch "
                f"wire={wire_ws!r} disk={disk_ws!r}"
            )
    wire_freeze = str(wire_meta.get("freeze_timestamp") or "").strip()
    disk_freeze = str(disk_meta.get("freeze_timestamp") or "").strip()
    if wire_freeze or disk_freeze:
        if wire_freeze and disk_freeze and wire_freeze == disk_freeze:
            return True, f"freeze_timestamp={wire_freeze}"
        return False, (
            "freeze_timestamp mismatch "
            f"wire={wire_freeze!r} disk={disk_freeze!r}"
        )

    wire_episode = str(wire.get("episode_id") or "").strip()
    disk_episode = str(disk.get("episode_id") or "").strip()
    if wire_episode and disk_episode and wire_episode == disk_episode:
        return True, f"episode_id={wire_episode}"
    return False, (
        "no matching immutable identity receipt "
        f"wire_episode={wire_episode!r} disk_episode={disk_episode!r}"
    )


class PostAudioJoinFailed(RuntimeError):
    """A ``strict`` post-audio overlay could not prove its join.

    Named, with a stable message prefix, so tests assert on a type rather than
    on incidental ``JSONDecodeError`` / ``OSError`` wording.
    """


def overlay_audio_timing(ledger: dict, strict: bool = False) -> dict:
    """Rehydrate the pre-audio wire ledger from the active post-audio ledger.

    ShotLock is the canonical graph join gated by ``audio_done``. The freeze
    cascade supplies authored content on the wire while SceneSequencer,
    AudioEnhance, and EpisodeAssembler persist their producer-owned truth to the
    in-flight disk ledger. Once episode identity is proven, disk owns the full
    ``audio`` section and the other post-audio top-level sections; metadata and
    row-local timing/WAV fields merge only into empty wire slots.

    ``strict`` IS CALLER-SCOPED ON PURPOSE, AND THAT IS THE WHOLE DESIGN.
    This free function has TWO live callers with opposite criticality, and a
    global contract is wrong for one of them whichever way it is written:

    * :meth:`OTRShotLock.lock` gates on the ``audio_done`` forceInput, and
      ``EpisodeAssembler`` ATTEMPTS the durable save in the same function body
      immediately before minting that string. So once ``audio_done`` is
      genuinely non-empty, a missing or unprovable join here is a real anomaly
      and must fail LOUD -- silently returning the PRE-AUDIO ledger restores the
      old beat-id space and makes the PBUG-20260811-02 repair inert for that
      run. Bug Bible 12.57: resolve the durable owner, prove same-run identity,
      REJECT mismatches. That caller passes ``strict=True``.

      ATTEMPTS, not guarantees, and the distinction is load-bearing: that save
      sits inside a blanket ``except Exception`` in ``scene_sequencer`` and
      ``save_ledger_safe`` RETURNS ``False`` rather than raising, with the
      result unchecked at the call site -- so ``audio_done`` can fire non-empty
      after a save that silently failed. That is precisely why raising here is
      an improvement rather than a formality: this gate is currently the only
      thing between a failed durable save and a render that quietly uses
      pre-audio ids. Making the save itself gate the signal is the real repair
      and is a separate item.
    * ``SignalLostVideoRenderer.render_video`` (``video_engine.py``) calls this
      with no ``audio_done`` input at all -- only an ``AUDIO`` data dependency
      -- and uses the overlay for title-card timing, not for an identity-
      critical join. It is a registered, sanctioned policy-floor node and the
      call has no try/except around it, so raising there would convert a
      slightly-worse title card into a hard crash of the floor renderer. It
      keeps the fail-soft default.

    Under ``strict`` the whole join is validated BEFORE anything is written, and
    the merge happens on a deep copy: a rejected join must leave the caller's
    ledger untouched rather than half-overlaid. Without ``strict`` the behaviour
    is exactly what it has always been -- warn and return the wire unchanged.
    """
    if otr_env.get("OTR_TEST_MODE") == "1":
        return ledger                       # CPU tests never read disk state
    lines = ledger.get("lines") or []
    try:
        from pathlib import Path
        from . import _otr_ledger as _OL
        p = _OL.in_flight_ledger_path()
        if not p:
            if strict:
                raise PostAudioJoinFailed(
                    "post-audio join failed: no durable ledger path resolved, "
                    "but audio_done has fired -- EpisodeAssembler saves the "
                    "durable ledger before minting that signal, so its absence "
                    "here is an anomaly, not a pre-audio run")
            return ledger
        p = Path(p)
        if strict and not p.exists():
            raise PostAudioJoinFailed(
                "post-audio join failed: durable ledger %s does not exist" % p)
        disk = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(disk, dict):
            raise ValueError(f"post-audio ledger is {type(disk).__name__}, expected object")
        same_episode, identity = _same_frozen_episode(ledger, disk)
        if not same_episode:
            if strict:
                # Two different failures wear this one flag, and conflating them
                # sends the next reader hunting a collision that never happened.
                # _same_frozen_episode returns False both when a STALE SIBLING
                # really did try to donate audio truth, and when NEITHER side
                # carries any identity receipt at all.
                #
                # DECIDED FROM THE LEDGERS, NOT FROM THE PROSE. Sniffing
                # _same_frozen_episode's message for "no matching immutable
                # identity receipt" got this wrong for one real case: a legacy
                # pair with no freeze_timestamp on either side but two DIFFERENT
                # non-empty episode_ids falls through to that same wording, and
                # that is a genuine stale-sibling collision being labelled as an
                # absence. Only a total absence of receipts is "no identity".
                def _receipts(led):
                    meta = led.get("meta") if isinstance(led.get("meta"), dict) else {}
                    return (str(meta.get("freeze_timestamp") or "").strip(),
                            str(led.get("episode_id") or "").strip())

                no_identity = not any(_receipts(ledger) + _receipts(disk))
                raise PostAudioJoinFailed(
                    "post-audio join failed: %s from %s -- %s"
                    % ("no identity available on either side" if no_identity
                       else "identity mismatch (stale sibling rejected)",
                       p.name, identity))
            log.warning(
                "[OTR_ShotLock] post-audio ledger overlay REJECTED from %s: %s",
                p.name, identity,
            )
            return ledger

        # VALIDATED -- only now may anything be written, and never to the
        # caller's own object. Everything below mutates `ledger` in place
        # (episode_id, audio, meta, lines, music), so under strict we work on a
        # deep copy and hand it back only if the whole merge completes. A
        # half-overlaid ledger in the caller's hands is worse than no overlay:
        # it looks joined.
        if strict:
            ledger = copy.deepcopy(ledger)
            # REBIND, or the copy is a lie. ``lines`` was bound from the
            # caller's ledger before this try block, so every row merge below
            # would still write through to the ORIGINAL list while the returned
            # copy came back empty of timing -- caller mutated, result useless,
            # both at once. Caught by the round-trip assertion in
            # tests/test_post_audio_join_strict.py.
            lines = ledger.get("lines") or []

        # The durable ledger owns the legitimate pending-id -> final-title
        # transition.  ``freeze_timestamp`` proves this is the same authored
        # run, so downstream image/video consumers must receive the final id
        # rather than the stale frozen-wire placeholder.
        disk_episode_id = str(disk.get("episode_id") or "").strip()
        if disk_episode_id:
            ledger["episode_id"] = disk_episode_id

        # EpisodeAssembler is the producer/owner of this entire section. Disk
        # wins even when the pre-audio wire contains a stale non-empty hash.
        if "audio" in disk:
            ledger["audio"] = disk["audio"]
        for key in (
            "audio_gates", "transitions", "radio_bookend_path",
            "final_audio_path", "final_video_path",
        ):
            if key in disk:
                ledger[key] = disk[key]

        disk_meta = disk.get("meta") if isinstance(disk.get("meta"), dict) else {}
        wire_meta = ledger.get("meta")
        if not isinstance(wire_meta, dict):
            wire_meta = {}
        for key, value in disk_meta.items():
            # ``meta.paths`` is rename-owned durable truth.  It must replace a
            # non-empty pending-root block on the frozen wire.  All other
            # writer metadata retains missing-only merge semantics.
            if key == "paths":
                wire_meta[key] = value
            elif key not in wire_meta or wire_meta.get(key) in (None, "", [], {}):
                wire_meta[key] = value
        ledger["meta"] = wire_meta

        dmap = {str(dl.get("line_id")): dl for dl in (disk.get("lines") or [])
                if isinstance(dl, dict) and dl.get("line_id")}
        tkeys = ("dur_s", "duration_s", "start_s", "samples", "audio_samples", "sample_rate")
        rows_overlaid = 0
        for ln in lines:
            if not isinstance(ln, dict):
                continue
            d = dmap.get(str(ln.get("line_id") or ""))
            if not d:
                continue
            changed = False
            for k in tkeys:
                if ln.get(k) in (None, "") and d.get(k) not in (None, ""):
                    ln[k] = d[k]
                    changed = True
            for k, v in d.items():
                if str(k).endswith("wav_path") and v and not ln.get(k):
                    ln[k] = v
                    changed = True
            rows_overlaid += int(changed)

        # Music is a separate keyed ownership surface.  Disk owns rendered
        # path/timing, but only while a fresh cue-spec identity matches.  This
        # is deliberately stricter than the empty-slot line overlay: a stale
        # pre-audio music row must never keep or receive audio for a re-authored
        # cue.  Legacy banks have no wire music rows, so validated disk rows are
        # appended after same-episode proof.
        from .production_ledger import music_cue_spec_sha256

        wire_music = ledger.get("music")
        if not isinstance(wire_music, list):
            wire_music = []
        disk_music = disk.get("music")
        if not isinstance(disk_music, list):
            disk_music = []

        disk_music_by_id = {}
        invalid_disk_cues = set()
        for row in disk_music:
            if not isinstance(row, dict):
                continue
            cue_id = str(row.get("cue_id") or "")
            if not cue_id:
                continue
            if cue_id in disk_music_by_id:
                invalid_disk_cues.add(cue_id)
                continue
            disk_music_by_id[cue_id] = row

        wire_music_by_id = {
            str(row.get("cue_id")): row for row in wire_music
            if isinstance(row, dict) and row.get("cue_id")
        }
        matched_cue_ids = set()
        music_rows_overlaid = 0
        music_render_fields = (
            "wav_path", "start_s", "dur_s", "start_s_space", "shot_id",
        )
        music_fill_fields = ("placement", "anchor_line_id")

        for cue_id, wire_row in list(wire_music_by_id.items()):
            disk_row = disk_music_by_id.get(cue_id)
            if disk_row is None or cue_id in invalid_disk_cues:
                continue
            wire_hash = music_cue_spec_sha256(wire_row)
            disk_hash = music_cue_spec_sha256(disk_row)
            disk_stored_hash = disk_row.get("cue_spec_sha256")
            if (
                wire_hash is None
                or disk_hash is None
                or disk_stored_hash != disk_hash
                or wire_hash != disk_hash
            ):
                log.warning(
                    "[OTR_ShotLock] music overlay REJECTED cue=%s "
                    "wire=%s disk=%s stored=%s",
                    cue_id,
                    str(wire_hash or "missing")[:12],
                    str(disk_hash or "missing")[:12],
                    str(disk_stored_hash or "missing")[:12],
                )
                continue
            changed = False
            for key in music_render_fields:
                value = disk_row.get(key)
                if value not in (None, "") and wire_row.get(key) != value:
                    wire_row[key] = value
                    changed = True
            for key in music_fill_fields:
                if wire_row.get(key) in (None, "") and disk_row.get(key) not in (None, ""):
                    wire_row[key] = disk_row[key]
                    changed = True
            wire_row["cue_spec_sha256"] = wire_hash
            matched_cue_ids.add(cue_id)
            music_rows_overlaid += int(changed)

        for cue_id, disk_row in disk_music_by_id.items():
            if cue_id in wire_music_by_id or cue_id in invalid_disk_cues:
                continue
            disk_hash = music_cue_spec_sha256(disk_row)
            if disk_hash is None or disk_row.get("cue_spec_sha256") != disk_hash:
                log.warning(
                    "[OTR_ShotLock] legacy music append REJECTED cue=%s: "
                    "missing/stale cue identity",
                    cue_id,
                )
                continue
            appended = dict(disk_row)
            appended["cue_spec_sha256"] = disk_hash
            wire_music.append(appended)
            wire_music_by_id[cue_id] = appended
            matched_cue_ids.add(cue_id)
            music_rows_overlaid += 1
        ledger["music"] = wire_music

        # EpisodeAssembler alone mints mirrored music lines.  Replace any stale
        # wire mirrors with same-episode disk mirrors whose cue successfully
        # passed the identity join above; ordinary authored lines remain owned
        # by the wire ledger.
        base_lines = [
            row for row in lines
            if not (
                isinstance(row, dict)
                and row.get("mirrored_from") == "music"
            )
        ]
        line_index = {
            str(row.get("line_id")): index
            for index, row in enumerate(base_lines)
            if isinstance(row, dict) and row.get("line_id")
        }
        mirrors_overlaid = 0
        for disk_line in disk.get("lines") or []:
            if not isinstance(disk_line, dict):
                continue
            if disk_line.get("mirrored_from") != "music":
                continue
            cue_id = str(disk_line.get("music_cue_id") or "")
            if cue_id not in matched_cue_ids:
                continue
            mirror = dict(disk_line)
            line_id = str(mirror.get("line_id") or "")
            if not line_id:
                continue
            if line_id in line_index:
                base_lines[line_index[line_id]] = mirror
            else:
                line_index[line_id] = len(base_lines)
                base_lines.append(mirror)
            mirrors_overlaid += 1
        if mirrors_overlaid or len(base_lines) != len(lines):
            base_lines.sort(key=lambda row: (
                float(row.get("start_s"))
                if isinstance(row, dict)
                and isinstance(row.get("start_s"), (int, float))
                else 1e18
            ))
            ledger["lines"] = base_lines
            lines = base_lines
        master_hash = str((ledger.get("audio") or {}).get("master_audio_sha256") or "")
        log.info(
            "[OTR_ShotLock] post-audio ledger overlay from %s "
            "(%s, rows=%d, music=%d, mirrors=%d, master_sha=%s)",
            p.name, identity, rows_overlaid, music_rows_overlaid,
            mirrors_overlaid, master_hash[:12] or "missing",
        )
    except PostAudioJoinFailed:
        # Already the considered verdict -- never demote it to a warning.
        raise
    except Exception as exc:                 # noqa: BLE001
        if strict:
            # Under strict every remaining failure here is a real one: an
            # unreadable or malformed durable ledger, a disk error, a bad row
            # shape. Swallowing it returns the PRE-AUDIO ledger, which silently
            # restores the old beat-id space -- exactly the class of quiet
            # fallback PBUG-20260811-02 was.
            raise PostAudioJoinFailed(
                "post-audio join failed: %s: %s" % (type(exc).__name__, exc)
            ) from exc
        log.warning("[OTR_ShotLock] post-audio ledger overlay skipped: %s", exc)
    return ledger


#: Seconds of picture for an act-break music bridge nothing timed.
#: PBUG-20260829-16.
#:
#: Commit 59286499 ("rip interstitial audio insertion", 2026-07-22) removed the
#: ``interstitial`` cue slot and with it the ONLY code that stamped ``start_s``
#: and ``dur_s`` onto ``music_inter`` rows::
#:
#:     -_CUE_SLOTS = ("opening", "closing", "interstitial")
#:     -    _mrow["start_s"] = float(_p["start_s"])
#:     -    _mrow["dur_s"]   = float(_p["dur_s"])
#:
#: The rip itself was deliberate and stands. What it did not do is re-home the
#: ledger fields it stopped writing, and the writer kept planning the beats
#: (``_otr_episode_budget``: ``music_inter_count = act_count - 1`` whenever
#: ``include_act_breaks``). So every act break since has minted a row with no
#: cue, no audio, no ``start_s`` and no ``dur_s`` -- it budgets to ZERO frames
#: and the video stage refuses it, killing the whole episode 40+ minutes in.
#: Measured across every ledger on this box: 742 carry ``music_inter`` rows and
#: NONE has ever published.
#:
#: The value is not a taste call. The last correctly-timed bridge this repo
#: produced was ``music_inter_01_001`` on 2026-07-21 -- the day before the rip --
#: at ``dur_s=4.087``. This rounds it. The bridge is silent by design (see
#: ``otr_master_audio_mux``: "it renders a picture and occupies no master-mix
#: time at all"), so nothing but the picture's length depends on the number.
MUSIC_BRIDGE_FALLBACK_DUR_S = 4.0

#: Video roles the assembler mints a deterministic MIRROR row for.
_MIRRORED_MUSIC_ROLES = ("music_open", "music_close")

#: Untimed music roles that get the fallback picture duration. Every music role
#: that can survive the sentinel filter belongs here: a row the filter cannot
#: recognise as a sentinel (because nothing mirrors it) and that gets no
#: fallback either is a row that budgets to zero frames and kills the leg.
_UNTIMED_MUSIC_FALLBACK_ROLES = ("music_inter", "music_open", "music_close")


def _untimed_music_sentinels(lines) -> set:
    """``line_id``s of pre-audio music SENTINELS, which own no timeline.

    A bank may author its own music row (``shot_000_music``) alongside the
    assembler's deterministic one (``music_opening_001``). These are NOT two
    competing copies of one beat -- they are one beat in two lifecycle stages:

    * the **sentinel** is authored pre-audio, carries the bank's own id, is
      untimed BY DESIGN, and is load-bearing -- it is what tells the assembler
      to mint the mirror and reserve that beat's still;
    * the **mirror** is minted post-audio under the assembler's deterministic
      id and carries the real ``start_s``/``dur_s``.

    Only the mirror is a timeline segment. The sentinel must never become one,
    **regardless of whether a cue could supply it a duration** -- and a cue can,
    because the cue's ``anchor_line_id`` points at the SENTINEL, not the mirror.
    An earlier fix here read that anchor and handed the sentinel the cue's
    duration, which put 10.0 s + 8.0 s onto the timeline a second time on top of
    the mirror that already carried it. The 4060 measured the result as an
    18.93 s overshoot at the master mux. The sentinel did not need a number; it
    needed to not be rendered.

    Detected by role rather than by cue id, because keying on the cue selects
    exactly the wrong row of the pair.

    ``music_inter`` is deliberately OUT of scope: an act break has no mirror to
    defer to, so it is a real beat missing its duration and gets
    :data:`MUSIC_BRIDGE_FALLBACK_DUR_S` instead of being dropped. And nothing
    here changes what a bank EMITS -- suppressing the sentinel upstream is what
    killed ``fastwan_8gb`` twice (PBUG-20260811-02, commit 3446af3f); this only
    decides which rows become video beats.
    """
    def _timed(row) -> bool:
        return (row.get("dur_s", row.get("duration_s")) is not None
                or row.get("samples", row.get("audio_samples")) is not None)

    mirrored_roles = {
        str(r.get("speaker_role") or "").strip().lower()
        for r in lines
        if isinstance(r, dict) and _timed(r)
    }

    sentinels = set()
    for row in lines:
        if not isinstance(row, dict) or _timed(row):
            continue
        role = str(row.get("speaker_role") or "").strip().lower()
        if role in _MIRRORED_MUSIC_ROLES and role in mirrored_roles:
            sentinels.add(str(row.get("line_id") or ""))
    return sentinels


def _music_bridge_dur_s(line: dict):
    """:data:`MUSIC_BRIDGE_FALLBACK_DUR_S` for ANY untimed music row.

    Returns ``None`` for every non-music row, so a missing duration anywhere
    else still surfaces as the loud zero-frame warning rather than being
    papered over with a number nobody measured.

    WIDENED 2026-08-31 FROM ``music_inter`` ALONE, which left a third case
    falling through both guards and killing the leg:

      * an untimed ``music_open`` / ``music_close`` WITH a timed mirror is
        dropped by :func:`_untimed_music_sentinels` -- the mirror owns the
        timeline, and that is correct;
      * an untimed ``music_inter`` gets the bridge duration below;
      * an untimed ``music_open`` / ``music_close`` with **NO** mirror got
        NEITHER. It is not a sentinel (nothing mirrors it, so the filter cannot
        see it as one) and it was not a bridge (wrong role), so it reached the
        frame budget with ``dur_s=None``, budgeted to ZERO frames, and the
        engine refused it at the VIDEO stage -- roughly seventy minutes in,
        after the writer, cast, voices and the full audio master were done.

    PROVEN ON REAL LEDGERS, not reasoned about. In
    ``signal_lost_the_apprentices_number`` the sentinel filter dropped 0 rows
    while ``shot_000_music`` (``music_open``) and ``shot_002_music``
    (``music_close``) both resolved to an effective duration of ``None``. Six
    recent ledgers carry no ``samples`` on ANY row, so the cumulative-samples
    path never runs at all and every beat depends on ``dur_s``.

    Still not a raise, deliberately: an OOM is the only acceptable killer, and
    a bank with a genuinely silent beat must not be blocked by a budget helper.
    """
    role = str((line or {}).get("speaker_role") or "").strip().lower()
    if role not in _UNTIMED_MUSIC_FALLBACK_ROLES:
        return None
    log.info(
        "[OTR_ShotLock] untimed music row %s (%s) carries no duration (its "
        "audio pass was removed in 59286499); rendering %.1fs of picture",
        str((line or {}).get("line_id") or "?"), role,
        MUSIC_BRIDGE_FALLBACK_DUR_S)
    return MUSIC_BRIDGE_FALLBACK_DUR_S


def extract_beats(ledger: dict) -> list:
    """Ordered, non-skipped beats from the frozen ledger.

    One beat per ledger line. Each beat carries its video role, char_id, text,
    and whatever audio timing the frozen ledger stamped (samples + sample_rate
    preferred; ``dur_s`` seconds as a fallback). Never raises on a sparse line.
    """
    try:
        from . import _otr_ledger_consumers as _OTRLC

        lines = list(_OTRLC.iter_lines(ledger))
    except Exception:  # noqa: BLE001
        lines = [
            ln for ln in (ledger or {}).get("lines") or []
            if isinstance(ln, dict) and not ln.get("skip")
        ]
    # Round 5 F5: lines may carry a NAME-scheme char_id (the announcer lines
    # stamp 'announcer') while the cast table + portrait index key by row id
    # (c01..). Normalize at the JOIN -- on the BEAT row only (frozen line rows
    # are never touched): an unknown char_id that case-insensitively matches a
    # cast row's name resolves to that row's char_id.
    cast_rows = [c for c in (ledger or {}).get("cast") or []
                 if isinstance(c, dict)]
    cast_ids = {str(c.get("char_id") or "") for c in cast_rows}
    name_to_id = {str(c.get("name") or "").strip().lower():
                  str(c.get("char_id") or "")
                  for c in cast_rows if c.get("name") and c.get("char_id")}

    def _normalize_char_id(cid: str) -> str:
        if not cid or cid in cast_ids:
            return cid
        return name_to_id.get(cid.strip().lower(), cid)

    id_to_first = {str(c.get("char_id") or ""):
                   (str(c.get("name") or "").split() or [""])[0]
                   for c in cast_rows}
    music_sentinels = _untimed_music_sentinels(lines)
    beats = []
    for i, ln in enumerate(lines):
        if not isinstance(ln, dict):
            continue
        if str(ln.get("line_id") or "") in music_sentinels:
            log.info(
                "[OTR_ShotLock] music row %s is a pre-audio sentinel; the timed %s mirror owns the timeline",
                ln.get("line_id"), ln.get("speaker_role"))
            continue
        cid = _normalize_char_id(str(ln.get("char_id") or ""))
        text = str(ln.get("text") or "").strip()
        # Round 5 F4 backstop (warn-only -- the ledger is FROZEN here): a
        # talking-head line that still opens with its own speaker's name as
        # a vocative means the writer's attribution repair missed it; the
        # beat renders with the stamped face, so make the miss LOUD.
        first = id_to_first.get(cid, "")
        if (len(first) > 1 and text.lower().startswith(first.lower())
                and re.match(r"^\s*" + re.escape(first) + r"\s*[,!?:;-]",
                             text, flags=re.IGNORECASE)):
            log.warning(
                "[OTR_ShotLock] line %s text opens with its OWN speaker's "
                "name (%s) -- probable mis-attribution shipped from the "
                "writer; the beat renders with char_id=%s's face",
                ln.get("line_id"), first, cid)
        beats.append({
            "beat_id": str(ln.get("line_id") or f"beat_{i:04d}"),
            "role": _video_role_for_line(ln),
            "char_id": cid,
            "text": text,
            "samples": ln.get("samples", ln.get("audio_samples")),
            "sample_rate": ln.get("sample_rate"),
            "dur_s": (ln.get("dur_s", ln.get("duration_s"))
                      or _music_bridge_dur_s(ln)),
            # THE STORY FIELDS RIDE THE BEAT (2026-08-29). The line row already
            # carries them (production_ledger.set_lines stamps every line with
            # shot_id / beat_intent / traits) and dropping them here forced the
            # nonverbal director to work from the bare line text alone --
            # motion with no stake in it, which is how a demand under threat
            # rendered as a fidget. Copied AT THE SOURCE rather than joined
            # back by beat_id later, because a synthesized beat id
            # (`beat_0007`, the music beats) has no line row to join against
            # and a silent empty join is exactly the miss this avoids.
            "shot_id": ln.get("shot_id"),
            "beat_intent": ln.get("beat_intent"),
            "traits": ln.get("traits"),
        })
    return beats


#: The synthetic OPENING-MUSIC beat id (operator look-QA 2026-06-10): the
#: opening theme plays over the episode head (audio starts at first-line
#: start_s, typically ~8-10s in) but no ledger LINE covers that span, so the
#: head fell to the procgen floor. A synthetic music_visual beat gives the
#: open a REAL rendered scene on the music engine (ltx_video in production).
OPENING_MUSIC_BEAT_ID = "b000_music_open"
_OPENING_MIN_S = 2.0


def derive_opening_music_beat(ledger: dict, fps: int):
    """``(beat, frames)`` for the head-gap opening-music scene, or ``(None, 0)``.

    Reads the FIRST non-skipped line's ``start_s`` from the frozen ledger
    (read-only); a head gap >= 2s earns the synthetic beat. Pure."""
    lines = [ln for ln in (ledger or {}).get("lines") or []
             if isinstance(ln, dict) and not ln.get("skip")]
    if not lines:
        return None, 0
    try:
        first_start = float(lines[0].get("start_s") or 0.0)
    except (TypeError, ValueError):
        return None, 0
    if first_start < _OPENING_MIN_S:
        return None, 0
    frames = int(round(first_start * int(fps or 25)))
    beat = {
        "beat_id": OPENING_MUSIC_BEAT_ID,
        "role": Role.MUSIC_VISUAL.value,
        "char_id": "",
        "text": "",
        "samples": None,
        "sample_rate": None,
        "dur_s": first_start,
        "_synthetic_open": True,
        "_start_s": 0.0,
    }
    return beat, frames


def compute_clip_budget(beats: list, fps: int) -> dict:
    """Audio-derived per-beat ``target_frame_count``.

    Frame counts come from CUMULATIVE audio SAMPLES -- ``frame_at(pos) =
    (pos*fps)//sample_rate`` -- so adjacent beats meet exactly (no double-count,
    no gap). When a beat carries only ``dur_s`` (no samples) it degrades to
    ``round(dur_s*fps)``. Returns ``{per_beat:{beat_id:frames}, total_frames}``.
    Pure; gated by the caller on ``audio_done``.

    NARROWED 2026-08-28. It used to accept a ``policy`` dict it never read, and
    to return a ``warnings`` list that was initialised empty and never appended
    to on any path -- while the docstring advertised it and the caller dutifully
    extended from it. An always-empty return key is a promise the function
    cannot keep.

    rip-sfx-broll (2026-07-01): the POOLING budget
    (clip_mode / pool_n / character_render_count) was removed with the
    retired_role_a / retired_role_b roles -- every beat renders per-beat.
    """
    fps = int(fps) if fps else 25
    sample_rate = 0
    for b in beats:
        sr = b.get("sample_rate")
        if sr:
            sample_rate = int(sr)
            break

    per_beat: dict = {}
    if sample_rate and all(b.get("samples") is not None for b in beats):
        cum = 0
        prev_frame = 0
        for b in beats:
            cum += int(b.get("samples") or 0)
            frame_at = (cum * fps) // sample_rate
            per_beat[b["beat_id"]] = max(0, frame_at - prev_frame)
            prev_frame = frame_at
    else:
        for b in beats:
            dur = b.get("dur_s")
            frames = int(round(float(dur) * fps)) if dur else 0
            if frames < 1:
                # PBUG-20260829-16: this used to be a silent 0. A beat with no
                # frames cannot render, but nothing said so until the engine
                # refused its own inputs at the VIDEO stage -- after the
                # writer, voices, music and the full audio master were done.
                # 72 minutes to learn that one beat had no duration. Say it
                # HERE, where it is cheap and the beat is named. Still not a
                # raise: an OOM-or-nothing operator directive governs the
                # render path, and a bank with a genuinely silent beat must
                # not be blocked by a budget helper.
                log.warning(
                    "[OTR_ShotLock] beat %r budgets to ZERO frames (dur_s=%r, "
                    "samples=%r). Any engine that needs a delivered target "
                    "will refuse this beat later, at the video stage. If this "
                    "is a music beat, its duration may live on a music cue "
                    "that never crossed anchor_line_id onto the line.",
                    b.get("beat_id"), dur, b.get("samples"))
            per_beat[b["beat_id"]] = frames

    total_frames = sum(per_beat.values())

    return {
        "per_beat": per_beat,
        "total_frames": total_frames,
    }


# ---------------------------------------------------------------------------
# M4 per-beat creative derivation (LLM with deterministic collapse guard)
# ---------------------------------------------------------------------------

_DIRECTIVE_KEYS = ("expression", "motion", "camera")


def _policy_engine_for_role(policy, role) -> str:
    """The engine that will actually render ``role``, read from a video policy.

    THE FROZEN ROUTE FIRST (chunk 1b): a nonblank
    ``effective_video_models[role]`` is what OTR_VideoDirector stamped after
    every redirect, so it is what renders. Absent -- a pre-1b policy or a
    hand-built fixture -- this falls back to the PICKED slot map, exactly as
    before.

    Extracted 2026-08-26 so ``derive_creative_directives`` and
    ``build_execution_plan`` cannot answer "which engine" differently for the
    same beat. The prompt policy and the shot row have to agree, or a lane
    renders a prompt written for a different engine.

    Deliberately NOT ``_effective_engine_for_role`` -- that re-enters the
    route-freeze authority, and a read must not freeze route state -- and
    deliberately NOT ``resolve_engine_id``: VideoDirector is the one
    public-alias normalisation boundary, and downstream policy reads frozen
    internal ids. Pure.
    """
    policy = policy or {}
    effective_video = policy.get("effective_video_models") or {}
    if effective_video:
        frozen = str(effective_video.get(role) or "")
        if frozen:
            return frozen
    return _role_slots.engine_id_for_role(policy.get("video_models") or {},
                                          role)


def _lane_preserves_dialogue(engine_id, role) -> bool:
    """True iff this lane's prompt may carry the beat's literal spoken line.

    TRUE for the AUDIO-IN families -- those engines are DRIVEN by the audio and
    the line is the thing they lip-sync to -- and for ``still_word``, the one
    non-audio lane that keeps the words on purpose, because it renders them
    into the picture as a word card.

    FALSE for every other character lane. An ordinary video engine handed a
    line of dialogue cannot speak it, so it does the only thing it can: it
    draws a mouth in motion. That is the H3 Caretaker defect -- a silent lane
    pantomiming speech it was never able to deliver.

    THIS IS NOT AN ADMISSION GATE. It refuses nothing an operator picked, and
    every registered engine passes it; models stay in the dropdown and a bad
    pick fails at render time on its own (operator, 2026-08-26). What it
    refuses is to CLASSIFY an engine it cannot identify, because that guess is
    silent in both directions: read an unknown lane as non-audio and a genuine
    lip-sync engine loses the line it needed; read it as audio-in and a silent
    lane goes on mouthing.
    """
    eid = str(engine_id or "")
    if not eid:
        raise ValueError(
            "OTR_ShotLock: the video policy names no engine for role %r, so "
            "the prompt policy cannot tell whether this lane is driven by the "
            "dialogue (audio-in, which needs the spoken line) or renders it "
            "silently (which must never receive it). Pick an engine for %r."
            % (role, role))
    try:
        from ._otr_video_engines import registry as _registry  # type: ignore
        from ._otr_video_engines import mouth_policy as _mouth  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import registry as _registry  # type: ignore
        from _otr_video_engines import mouth_policy as _mouth  # type: ignore
    lookup = eid
    if not _registry.is_registered(lookup):
        # A PUBLIC MENU ID resolves; a typo does not. VideoDirector normalises
        # the ids it stamps, but a hand-written `video_policy_json` can carry
        # the menu name (`ltx25_high_foley_plus`), and refusing that would fail
        # a policy the operator could legitimately write. Resolving HERE is a
        # read for CLASSIFICATION ONLY -- the resolved name never becomes the
        # engine, never reaches the shot row, and never re-freezes the route.
        try:
            from ._otr_shared.public_engines import resolve_engine_id
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared.public_engines import resolve_engine_id
        lookup = str(resolve_engine_id(eid) or "")
    if not lookup or not _registry.is_registered(lookup):
        raise ValueError(
            "OTR_ShotLock: role %r is routed to %r, which is not a registered "
            "video engine and does not resolve to one. The prompt policy reads "
            "the engine's FAMILY to decide whether this lane receives the "
            "spoken line, and an unknown engine would silently classify itself "
            "as non-audio and lose it." % (role, eid))
    # REGISTRATION BEFORE FAMILY, and the order is the whole point: the family
    # helpers answer "abstract" for an unknown id, and "abstract" is not in
    # AUDIO_IN_FAMILIES, so an unregistered engine would classify itself as a
    # silent lane without a word in the log.
    family = str(getattr(_registry.get_engine(lookup), "family", "") or "")
    return family in _mouth.AUDIO_IN_FAMILIES or lookup == "still_word"


#: How much of the spoken line M4 is shown as CONTEXT on a silent lane.
_M4_LINE_CONTEXT_CHARS = 240


def _line_context(text) -> str:
    """The capped slice of the spoken line a silent lane's writer is shown.

    ONE VALUE, TWO CONSUMERS, AND THAT IS THE WHOLE POINT (2026-08-27). The M4
    payload used to send ``b["text"][:240]`` while the literal-line filter
    tokenised the FULL line. On any line longer than the cap the model could
    quote back, verbatim, exactly what it had been shown -- and the filter,
    hunting the COMPLETE line's token run, would not match it. The quote sailed
    into the prompt with no warning at all.

    Not theoretical: measured over the real corpus, 295 of 7096 ledger lines
    are over the cap, across 111 episodes. Found by the Codex panel lane and
    reproduced before it was believed.

    Cut on a word boundary, so the last token is whole -- a half-word is a
    token the filter could never match anyway.
    """
    raw = str(text or "")
    if len(raw) <= _M4_LINE_CONTEXT_CHARS:
        return raw
    cut = raw[:_M4_LINE_CONTEXT_CHARS]
    head, sep, _tail = cut.rpartition(" ")
    return head if sep else cut


def _word_tokens(text) -> list:
    """Casefolded whole-word tokens. Punctuation and spacing are not evidence.

    UNICODE-AWARE, not ASCII (2026-08-27). ``[a-z0-9']`` produced NO tokens at
    all for a wholly non-Latin line, which made ``_repeats_the_line`` answer
    False for an exact quotation -- the filter was silently inert on exactly
    the material the adaptation lanes carry in the author's own language.
    ``casefold`` rather than ``lower`` for the same reason.
    """
    #: Typographic apostrophes fold to the ASCII one FIRST. Public-domain and
    #: Gutenberg source text is full of U+2019, and without this "I<U+2019>ll"
    #: tokenises as "i" + "ll" while an ASCII "I'll" stays one token -- so the
    #: same words spelled two ways would not compare equal. Same defect class
    #: as the ASCII-only pattern this replaced.
    folded = str(text or "").casefold()
    for _curly in ("’", "‘", "‚", "‛", "ʼ"):
        folded = folded.replace(_curly, "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", folded, re.UNICODE)


def _repeats_the_line(field, line_tokens) -> bool:
    """True iff ``field`` contains the beat's whole line as a CONTIGUOUS run.

    CONTIGUOUS, and by whole-word TOKEN rather than by substring, for two
    reasons that both bit earlier drafts of this filter: a substring test makes
    the line "Yes." match the word "eyes" inside a perfectly good expression,
    and an any-token test makes any field sharing two common words with the
    line look like a quotation. An empty line matches nothing.
    """
    if not line_tokens:
        return False
    tokens = _word_tokens(field)
    span = len(line_tokens)
    if span > len(tokens):
        return False
    return any(tokens[i:i + span] == line_tokens
               for i in range(len(tokens) - span + 1))


#: What a non-audio lane gets when the writer returns nothing usable for a
#: field. REWRITTEN 2026-08-27 (operator: *"no sense in rendering a video that
#: looks like a silly pan"*). The first cut of these floors said "restrained",
#: "subtle" and "stable" -- three damping words -- and 8 of 12 beats on the
#: first qualified foley episode rendered on them, so the floors WERE the
#: picture. A floor still must not invent a specific performance (it cannot
#: know the line), but it can demand real movement in the abstract and let
#: the model choose the verb; and the camera floor buys motion the subject
#: fallback cannot -- a push-in reads as life even on a hesitant performer.
_NONVERBAL_FALLBACKS = {
    "expression": "a vivid, readable reaction",
    "motion": "decisive full-body movement matched to the moment",
    "camera": "mid-shot, slow push-in",
}


def _deterministic_template(appearance: str, setting: str, beat_text: str) -> str:
    """The collapse-guard fallback prompt (BUG-046: never an empty/generic
    prompt into a render). Deterministic in its inputs."""
    parts = [p for p in (appearance, setting, beat_text) if p]
    return ", ".join(parts) if parts else setting


# Person-anchor DETECTOR removed (operator directive 2026-07-04): no automated
# person/face analyzer gates the talking-head prompt. The _subject_anchor below
# is prompt COMPOSITION (it always leads with face/framing tokens); face QUALITY
# is QA'd visually by the operator reviewing prompts, not grep-gated here.


def _subject_anchor(appearance: str) -> str:
    """The leading subject clause prepended to EVERY talking-head prompt path
    (round 5 F3): face/framing tokens lead (engines weigh leading tokens
    hardest), the appearance (bounded) follows. Pure."""
    base = "face visible, speaking to camera"
    app = (appearance or "").strip().rstrip(",.;: ")
    return f"{base}, {app[:120].rstrip(', ')}" if app else base



# --- Reply budget (PBUG-20260903-07) ----------------------------------------
# A directive row -- beat_id + expression + motion + camera, JSON-quoted --
# costs ~60 tokens on the live writer. The wrapper below spent five months
# handing EVERY caller a flat 300, while `derive_creative_directives` batches
# up to `batch_size=15` beats and asks for one row each: ~900 tokens of reply
# against a 300-token ceiling. The reply stopped mid-object, `_parse_directives`
# reported it as unparseable, and the reseed loop then retried the SAME prompt
# under the SAME ceiling -- so all three attempts failed identically and every
# character beat in every episode fell through to the deterministic template.
#
# Proven live on the 5080 (2026-09-03, gemma-4-12b-it, a FOUR-beat batch --
# well under the production 15):
#     max_new_tokens=300  -> reply ends '"slow push-in on the eyes."\n  ' -> 0/4 parsed
#     max_new_tokens=1200 -> reply ends '}\n]\n```'                       -> 4/4 parsed
# The model was never the problem; it was being cut off mid-sentence.
WRITER_REPLY_TOKENS_DEFAULT = 300      # historical flat budget: short single-shot asks
# 110, not the ~73/beat the probe measured: neither batch-prompt builder caps
# the length of expression/motion/camera (the nonverbal one actively asks for
# "two or three visible physical actions... name the concrete verbs, objects and
# materials"), so a verbose sample is normal rather than exceptional -- and the
# 300-budget probe run WAS the verbose one, which is why it got cut. Granting
# more than the model uses is free; granting less costs the whole batch.
DIRECTIVE_TOKENS_PER_BEAT = 110
DIRECTIVE_TOKEN_OVERHEAD = 120         # fences, brackets, the model's habitual preamble
DIRECTIVE_TOKEN_CEILING = 2400         # escalation stop: past this the prompt is at fault
#
# NOT TAKEN, deliberately, and worth knowing about: `_otr_generation_budget`
# already ships `_otr_fail_on_output_limit`, a payload flag every transport
# honours that turns "the reply hit its ceiling" into a raise instead of a
# truncated string -- authoritative where `_classify_unparsed_reply` below is a
# heuristic. It is not wired here because the loader signals that condition with
# a plain `ModelLoaderError` rather than one of the module's own CAPACITY_ERRORS,
# so consuming it means matching on message text across seven transports. That
# is a taxonomy decision, not a mechanical one. The heuristic needs no such
# taxonomy and also works for an injected `llm_fn`.


def _directive_token_budget(beat_count: int) -> int:
    """Reply budget for a derivation batch of ``beat_count`` beats.

    Sized to the ACTUAL request. Over-granting is free -- generation stops at
    the closing bracket -- whereas under-granting truncates the JSON and costs
    the whole batch, so this errs high on purpose.
    """
    beats = max(1, int(beat_count or 1))
    want = beats * DIRECTIVE_TOKENS_PER_BEAT + DIRECTIVE_TOKEN_OVERHEAD
    return max(WRITER_REPLY_TOKENS_DEFAULT, min(want, DIRECTIVE_TOKEN_CEILING))


def _classify_unparsed_reply(raw: str) -> str:
    """Name the SHAPE of a reply that did not yield a full batch.

    `_parse_directives` returns {} for three unrelated causes and the old
    warning called all of them "empty/unparseable", which is why this defect
    survived weeks of green logs: the one word that would have identified it --
    truncated -- was never printed. Truncation is the recoverable one (raise the
    budget); the other two are not, so they must be told apart.
    """
    txt = str(raw or "").strip()
    if not txt:
        return "empty"
    # JSON that opened and never closed is a reply the sampler cut off.
    if ("[" in txt or "{" in txt) and not txt.rstrip("`").rstrip().endswith(("]", "}")):
        return "truncated"
    return "malformed"


def _parse_directives(raw: str, expected_ids: list) -> dict:
    """Parse a batch LLM reply into ``{beat_id:{expression,motion,camera}}``.

    Returns ``{}`` on empty / unparseable / truncated output (the collapse
    guard's trigger); :func:`_classify_unparsed_reply` tells those three apart
    for the caller's warning. Accepts a JSON list or object; tolerant of extra
    keys.
    """
    if not raw or not str(raw).strip():
        return {}
    txt = str(raw).strip()
    # tolerate ```json fences / leading prose: slice to the first bracket.
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = txt.find(opener), txt.rfind(closer)
        if 0 <= i < j:
            try:
                data = json.loads(txt[i:j + 1])
                break
            except (ValueError, TypeError):
                continue
    else:
        return {}
    rows = data if isinstance(data, list) else data.get("beats") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bid = str(row.get("beat_id") or "")
        if bid and bid in expected_ids:
            parsed = {k: str(row.get(k) or "").strip() for k in _DIRECTIVE_KEYS}
            # An adapter MAY author the full rich prompt; captured here and
            # passed through the consistency gate (a hallucinated text_prompt
            # that drops the cast/setting falls back to the template).
            parsed["text_prompt"] = str(row.get("text_prompt") or "").strip()
            out[bid] = parsed
    return out


def _build_batch_prompt(batch: list, meta: dict, ledger: dict, setting: str) -> str:
    """Compose ONE batched derivation prompt (mirrors the Meta-brief protocol:
    derive from brief + beat + cast; instrumental wording kept model-agnostic)."""
    lines = [
        # PARITY WITH THE SILENT SIBLING (2026-09-03). This asked only for "a
        # concise expression, motion, and camera direction that fits the
        # character and the setting", which permits a pose and a framing --
        # both of which render as a photograph. The silent builder had already
        # been taught to ask for a kinetic chain and to scale it to the line;
        # this path stayed on the weaker wording and produced the flatter half
        # of the corpus. The character IS speaking here, so the body action is
        # what reads WHILE they speak rather than a full cross-room move.
        "You are a film director. For EACH beat below give the visible "
        "performance: a vivid facial expression, and body action that reads "
        "WHILE the character speaks -- a gesture that lands, a weight shift, a "
        "turn, a hand that makes real contact with something the setting "
        "supports. Name the concrete verbs and objects; scale it to the line, "
        "so an urgent or angry line earns real movement and a calm one earns "
        "less. Then ONE camera movement, and it must be STRATEGIC: it has to "
        "earn its place in THIS beat -- push in as a realisation lands, pull "
        "back as someone is left alone, tilt or crane to reveal what the beat "
        "turns on, track with a body that moves. Say what it moves toward or "
        "away from. A shot size or an angle on its own is a photograph, not a "
        "movement, so never answer with framing alone, and never repeat the "
        "same move twice in this batch. Reply ONLY with a JSON list of "
        'objects {"beat_id","expression","motion","camera"}.',
        # Gap-audit F3 (2026-06-10): the era/style tails are APPENDED later
        # by the prompt finisher -- the model must not duplicate them.
        "Do not include film-stock, film-grain, or lighting-style terms; "
        "they are appended automatically later.",
        # Round 5 F3 (the b002 no-person catch): any authored text_prompt must
        # keep the character on screen -- props/scenery alone lose the face.
        "If you author a text_prompt, describe the named character as the "
        "VISIBLE subject (face-forward, mid-shot or closer); never describe "
        "scenery or props without the character.",
        f"Setting: {setting}",
        "Beats:",
    ]
    for b in batch:
        appearance = _appearance_for_char(ledger, b["char_id"])
        lines.append(
            json.dumps({
                "beat_id": b["beat_id"],
                "character": appearance[:160],
                "line": b["text"][:240],
            }, ensure_ascii=True)
        )
    return "\n".join(lines)


def _build_nonverbal_batch_prompt(batch: list, meta: dict, ledger: dict,
                                  setting: str) -> str:
    """The batch derivation prompt for a lane that CANNOT speak.

    The literal line is still handed to the model, and that is deliberate: it
    is the only way to know what the moment IS, so the acting can be inferred
    from it. What changes is what is asked BACK -- the visible performance,
    never the words. Whatever comes back is filtered against the line anyway
    (``_repeats_the_line``), because an instruction is not an enforcement.
    """
    # THE ASK IS KINETIC AND LINE-DRIVEN (2026-08-27, operator: the action
    # part of the prompt is inspired by the dialogue/story). The first cut
    # asked for a "restrained facial expression", which told the writer to
    # keep the body still -- on lanes whose entire value is motion. The
    # scale-to-the-line language below is deliberately the same contract as
    # the ripped `_otr_motion_clause.build_clause_messages` (its kinetic
    # amendment), so the two derivation paths cannot drift apart in spirit.
    #
    # THE STORY CONTEXT IS PRIVATE EVIDENCE, NOT OUTPUT (2026-08-29). This
    # builder used to `del meta` and hand the model nothing but the bare
    # line, so the director staged motion with no stake in it -- a demand
    # under threat and a weather report earned the same fidget. It now reads
    # the episode logline and key objects from meta and each beat's
    # beat_intent/traits off the beat row (stamped by `extract_beats` from
    # the frozen line), all as INPUT-ONLY context: the response schema, the
    # quote filter, and every ban below are unchanged.
    logline = str(((meta or {}).get("produced_story") or {})
                  .get("logline") or "").strip()
    key_objects = [str(o).strip() for o in (meta or {}).get("key_objects")
                   or [] if str(o or "").strip()]
    lines = [
        "You are a film director working on a SILENT shot. For EACH beat "
        "below you are given the line the character speaks, as CONTEXT ONLY -- "
        "the camera records no sound and the actor must NOT be shown speaking "
        "it. Treat the character, the story context, the beat intent and the "
        "line as PRIVATE EVIDENCE: never quote them back. Give the visible "
        "PERFORMANCE the line implies instead: a vivid facial expression, and "
        "the KINETIC body action the moment demands -- convert the line's "
        "dramatic pressure into one coherent chain of two or three visible "
        "physical actions, what the body actually DOES, the motion vector, "
        "not a pose. SCALE THE MOVEMENT TO THE LINE: a calm line earns small "
        "motion; an urgent, angry or frightened line earns real movement "
        "(rises, strides, wheels around, slams, recoils, grabs, points), and "
        "at least one of its actions makes REAL CONTACT with an object the "
        "setting or the listed objects support -- name the concrete verbs, "
        "objects and materials (a brass dial rolled down, a ledger snapped "
        "shut, a chart slapped onto the wooden desk), because the shot's "
        "sound is derived from exactly those words. Do not cap yourself at "
        "fidgets. Then ONE camera movement, and it must be STRATEGIC: the move "
        "has to earn its place in THIS beat -- push in as a realisation lands, "
        "pull back as someone is left alone, tilt or crane to reveal the thing "
        "the beat turns on, track with a body that crosses the room. Say what "
        "it moves toward or away from. A shot size or an angle on its own is a "
        "photograph, not a movement, so never answer with framing alone, and "
        "never repeat the same move twice in this batch. Reply ONLY with a "
        'JSON list of objects {"beat_id","expression","motion","camera"}.',
        # Same finisher contract as the spoken path: the era/style tails are
        # APPENDED later and must not be duplicated here.
        "Do not include film-stock, film-grain, or lighting-style terms; "
        "they are appended automatically later.",
        "NEVER quote or paraphrase the line itself. Do NOT write a "
        "text_prompt field. Do NOT describe speaking, talking, dialogue, "
        "lip-sync, mouth movement, an open mouth, subtitles, captions, or any "
        "visible text or lettering.",
        "Describe the named character as the VISIBLE subject; never describe "
        "scenery or props without the character.",
        f"Setting: {setting}",
    ]
    if logline:
        lines.append("Story (private context): %s" % logline[:240])
    if key_objects:
        lines.append("Objects on hand: %s" % ", ".join(key_objects[:8]))
    lines.append("Beats:")
    for b in batch:
        appearance = _appearance_for_char(ledger, b["char_id"])
        payload = {
            "beat_id": b["beat_id"],
            "character": appearance[:160],
            # The SAME capped value the filter compares against; see
            # `_line_context`. Sending one slice and filtering on another
            # is how an exact quote of the shown context got through.
            "line_context_do_not_quote": _line_context(b["text"]),
        }
        # Present-key only: a beat that never carried these (a synthetic
        # music beat, an old ledger) does not acquire nulls. beat_intent is
        # capped through the same word-boundary helper as the line -- the
        # writer emits full SENTENCES there (test_look_qa_round5's live
        # catch) and an uncapped one would drown the payload.
        intent = str(b.get("beat_intent") or "").strip()
        if intent:
            payload["beat_intent"] = _line_context(intent)
        traits = b.get("traits")
        if traits:
            payload["traits"] = str(traits)[:160]
        lines.append(
            # ensure_ascii FALSE, unlike the spoken-path builder above, and
            # that is the point of "one value, two consumers": escaped as
            # \uXXXX the model is shown something the filter -- which
            # tokenises the DECODED string -- could never match, so the
            # shared-value invariant would hold in Python and be false in
            # the only place it matters.
            json.dumps(payload, ensure_ascii=False)
        )
    return "\n".join(lines)


def derive_creative_directives(
    beats: list,
    meta: dict,
    ledger: dict,
    *,
    llm_fn=None,
    batch_size: int = 15,
    max_reseed: int = 2,
    consistency_gate_warn_only: bool = False,
    video_policy=None,
):
    """Derive per-beat creative directives for character-bearing beats.

    Returns ``(creative_by_beat, warnings)`` where ``creative_by_beat[beat_id]``
    is ``{expression, motion, camera, text_prompt, source, prompt_hash}``.
    Cheap-family beats are skipped entirely (NO llm call). ``llm_fn`` is a
    ``callable(prompt:str) -> str`` (injectable for tests); when None it is
    resolved lazily from the writer's slot and, if unavailable (llm_fn stays
    None), the deterministic template carries every beat -- the legit local lane.
    If an attempted writer LLM yields no usable directive after ``max_reseed``
    reseeds, the deterministic template carries that beat with a loud warning.
    Authored non-empty visual vocabulary is preserved; this pass does not use
    Python token overlap as a story-world judge. It never touches frozen audio.
    """
    del consistency_gate_warn_only  # retained only for saved-workflow ABI
    warnings: list = []
    # The image dispatcher owns effective init-image capability (including
    # force-map and radio-host redirects).  Ask it lazily so every direct legacy
    # caller still behaves as before when it has no complete policy.  A proven
    # ``False`` for character_video lets an all-visualizer run return before
    # resolving the writer LLM; unknown remains author-all.
    still_capabilities = None
    if video_policy is not None:
        try:
            from ._otr_shared.route_freeze import RouteFreezeError
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared.route_freeze import RouteFreezeError
        try:
            from .otr_image_gen_dispatcher import still_consumer_capabilities
            still_capabilities = still_consumer_capabilities(video_policy)
        except RouteFreezeError:
            # A MALFORMED ROUTING ENVIRONMENT IS TERMINAL (2026-07-25 QA, third
            # swallow site found). "Uncertainty retains authoring" is the right
            # answer for an incomplete policy -- it is the wrong answer for a
            # typo'd OTR_FORCE_ENGINE_MAP, which would silently fall back to
            # authoring every character beat as though capability were unknown.
            raise
        except Exception:  # noqa: BLE001 -- uncertainty retains authoring
            still_capabilities = None
    char_beats = [
        b for b in beats
        if (b["role"] in CHARACTER_BEARING_ROLES
            and (still_capabilities is None
                 or still_capabilities.get(b["role"]) is not False))
    ]
    if not char_beats:
        return {}, warnings

    # THE PROMPT POLICY SEAM (2026-08-26). Whether a lane may receive the
    # spoken line is a property of the ENGINE, never of the beat's text, so it
    # is resolved ONCE per role from the same policy ``build_execution_plan``
    # reads -- one resolution, two consumers, no divergence.
    #
    # A caller with NO policy at all keeps the historical behaviour. There is
    # nothing to classify from, and defaulting a policy-free legacy call to
    # "strip the dialogue" would silently change every one of them. Production
    # always passes one (``lock()`` calls this with ``video_policy=policy``).
    preserves_dialogue_for_role = {}
    if video_policy is not None:
        for _role in sorted({b["role"] for b in char_beats}):
            preserves_dialogue_for_role[_role] = _lane_preserves_dialogue(
                _policy_engine_for_role(video_policy, _role), _role)

    # The budget is sized to the REQUEST, not to a constant: this pass asks for
    # one JSON row per beat, so the reply grows with the batch (PBUG-20260903-07).
    # Keep the raw generator too -- a truncated reply is recoverable by re-binding
    # a bigger budget, and that must not cost a model re-resolution.
    directive_budget = _directive_token_budget(batch_size)
    writer_gen = None
    if llm_fn is None:
        writer_gen, writer_model_id = _resolve_writer_llm_binding(meta, warnings)
        if writer_gen is not None:
            llm_fn = _writer_call_at(writer_gen, directive_budget)
            log.info(
                "[shot_lock] derivation writer %s: %d char beats, batch %d, "
                "reply budget %d tokens",
                writer_model_id, len(char_beats), batch_size, directive_budget)

    setting = _read_setting(meta)
    brief_hash = _content_hash(meta.get("story_brief_terms") or meta.get("story_brief") or {})
    cast_hash = _content_hash(ledger.get("cast") or [])

    creative: dict = {}
    for start in range(0, len(char_beats), max(1, int(batch_size))):
        batch = char_beats[start:start + max(1, int(batch_size))]
        expected = [b["beat_id"] for b in batch]
        directives: dict = {}
        # One role per batch in practice (CHARACTER_BEARING_ROLES has a single
        # member), so the batch's policy is the first beat's policy; the
        # per-beat composition below still consults each beat's own role.
        batch_preserves_dialogue = preserves_dialogue_for_role.get(
            batch[0]["role"], True)
        if llm_fn is not None:
            prompt = (_build_batch_prompt(batch, meta, ledger, setting)
                      if batch_preserves_dialogue
                      else _build_nonverbal_batch_prompt(
                          batch, meta, ledger, setting))
            attempt_fn = llm_fn
            budget = directive_budget
            for attempt in range(max_reseed + 1):
                try:
                    raw = attempt_fn(prompt)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"derivation llm_fn raised ({exc}); reseed {attempt}")
                    raw = ""
                directives = _parse_directives(raw, expected)
                # Break only on FULL batch coverage -- a partial reply (e.g. 14
                # of 15 beats) must spend its remaining reseed budget before
                # the missing beats drop to the deterministic collapse guard.
                if directives and all(bid in directives for bid in expected):
                    break
                if attempt < max_reseed:
                    # SAY WHICH FAILURE IT IS. The old wording -- "empty/
                    # unparseable" -- covered three unrelated causes, so a
                    # truncation read exactly like a refusal and the one number
                    # that mattered (the budget) never appeared in any log.
                    shape = _classify_unparsed_reply(raw)
                    detail = ("%s derivation for batch %s..; reseed %d/%d "
                              "(%d of %d beats parsed, reply %d chars, "
                              "budget %d tokens)"
                              % (shape, expected[:1], attempt + 1, max_reseed,
                                 len(directives), len(expected),
                                 len(str(raw or "")), budget))
                    warnings.append(detail)
                    log.warning("[shot_lock] %s", detail)
                    log.debug("[shot_lock] reply tail: %r", str(raw or "")[-240:])
                    # A retry under the SAME budget reproduces a truncation
                    # exactly -- same prompt, same ceiling, same cut. Give the
                    # reseed something to do: re-bind a bigger reply budget.
                    # Only possible when we resolved the generator ourselves;
                    # an injected llm_fn owns its own budget.
                    if (shape == "truncated" and writer_gen is not None
                            and budget < DIRECTIVE_TOKEN_CEILING):
                        budget = min(budget * 2, DIRECTIVE_TOKEN_CEILING)
                        attempt_fn = _writer_call_at(writer_gen, budget)
                        log.warning(
                            "[shot_lock] reply was cut off mid-JSON; "
                            "reseeding at %d tokens", budget)
        for b in batch:
            appearance = _appearance_for_char(ledger, b["char_id"])
            d = directives.get(b["beat_id"]) or {}
            # A lane that cannot speak never receives the words -- not the
            # line, and not an authored text_prompt that might have quoted it.
            nonverbal = not preserves_dialogue_for_role.get(b["role"], True)
            llm_text = "" if nonverbal else d.get("text_prompt", "")
            has_directives = any(d.get(k) for k in _DIRECTIVE_KEYS)
            if nonverbal:
                # THE SILENT CHARACTER LANE (2026-08-26). Expression, motion
                # and camera only: no beat text, and no `_subject_anchor` --
                # that anchor opens with "speaking to camera", which is the
                # instruction that made H3's Caretaker mouth a line it had no
                # way to voice.
                # The capped context, NOT the raw line -- the model can only
                # quote what it was shown, and comparing against a longer
                # sequence than it ever saw is a filter that cannot fire.
                #
                # THE STORY CONTEXT GETS THE SAME BACKSTOP (2026-08-29). The
                # prompt shows beat_intent and the logline as PRIVATE
                # EVIDENCE, and an instruction is not an enforcement -- the
                # docstring above says exactly that about the line. The same
                # contiguous whole-run test guards them: only a verbatim
                # quote can fire it, never shared vocabulary. `traits` is
                # deliberately NOT filtered -- two adjectives meant to color
                # the expression are not a quotation when they reach it.
                context_runs = [_word_tokens(_line_context(b["text"]))]
                _intent = str(b.get("beat_intent") or "").strip()
                if _intent:
                    context_runs.append(_word_tokens(_line_context(_intent)))
                _logline = str(((meta or {}).get("produced_story") or {})
                               .get("logline") or "").strip()[:240]
                if _logline:
                    context_runs.append(_word_tokens(_logline))
                kept = {}
                for key in _DIRECTIVE_KEYS:
                    value = str(d.get(key) or "").strip()
                    if value and any(_repeats_the_line(value, run)
                                     for run in context_runs if run):
                        warnings.append(
                            "beat %s: the writer put shown private context "
                            "(the line, the beat intent, or the logline) back "
                            "into %r on a silent lane; dropped (the lane "
                            "cannot deliver it)" % (b["beat_id"], key))
                        value = ""
                    kept[key] = value
                if any(kept.values()):
                    source = "llm"
                elif llm_fn is not None:
                    source = "template_after_llm_miss"
                    warnings.append(
                        "writer LLM produced no usable nonverbal directive for "
                        "beat %s after %d reseeds; using the deterministic "
                        "acting fallbacks" % (b["beat_id"], max_reseed))
                else:
                    # No writer LLM configured: the fallbacks ARE the primary
                    # local lane here, exactly as the template is on the
                    # spoken path.
                    source = "template"
                d = dict(kept)
                text_prompt = ", ".join(p for p in (
                    appearance, setting,
                    kept["expression"] or _NONVERBAL_FALLBACKS["expression"],
                    kept["motion"] or _NONVERBAL_FALLBACKS["motion"],
                    kept["camera"] or _NONVERBAL_FALLBACKS["camera"],
                ) if p)
            elif llm_text:
                text_prompt = llm_text
                source = "llm"
            elif has_directives:
                text_prompt = ", ".join(
                    p for p in (
                        appearance, setting, b["text"],
                        d.get("expression"), d.get("motion"), d.get("camera"),
                    ) if p
                )
                source = "llm"
            elif llm_fn is not None:
                text_prompt = _deterministic_template(
                    appearance, setting, b["text"])
                source = "template_after_llm_miss"
                d = {k: "" for k in _DIRECTIVE_KEYS}
                warnings.append(
                    "writer LLM produced no usable directive for beat %s after "
                    "%d reseeds; using deterministic template collapse guard"
                    % (b["beat_id"], max_reseed)
                )
            else:
                # No writer LLM configured (llm_fn None): the deterministic
                # template IS the primary local lane, not a fallback.
                text_prompt = _deterministic_template(appearance, setting, b["text"])
                source = "template"
                d = {k: "" for k in _DIRECTIVE_KEYS}
            # No Python vocabulary or token-overlap judge may replace an
            # authored non-empty visual prompt. The subject anchor remains
            # prompt composition: it leads every talking-head path (LLM,
            # composed, template) with face/framing context.
            # ...on the SPOKEN paths only. The anchor's first clause is
            # "face visible, speaking to camera", which is a speaking
            # instruction; a silent lane composed its own subject above and
            # must not be handed one.
            if not nonverbal:
                text_prompt = f"{_subject_anchor(appearance)}, {text_prompt}"
            # FINISH the prompt (gap-audit F3, 2026-06-10): era tail (brief
            # atmosphere/palette/lighting) + the film style tail, restored
            # from the deleted legacy composer. MUST run before
            # prompt_hash so the stored hash
            # matches the rendered prompt. Fail-soft.
            try:
                try:
                    from ._otr_story_brief_helpers import (  # type: ignore
                        finish_visual_prompt)
                except ImportError:  # pragma: no cover -- flat test imports
                    from _otr_story_brief_helpers import (  # type: ignore
                        finish_visual_prompt)
                text_prompt = finish_visual_prompt(meta, text_prompt)
            except Exception:  # noqa: BLE001
                pass
            creative[b["beat_id"]] = {
                "expression": d.get("expression", ""),
                "motion": d.get("motion", ""),
                "camera": d.get("camera", ""),
                "text_prompt": text_prompt,
                "source": source,
                "request_hash": _content_hash(
                    [brief_hash, cast_hash, b["beat_id"], b["char_id"]]
                ),
                "prompt_hash": _content_hash(text_prompt),
            }
    return creative, warnings


def writer_model_id_from_meta(meta) -> str:
    """The writer model the ledger already selected. No widget, no new pick.

    ONE spelling of this read (2026-08-22). Ghost Prompt v2 needs the same
    identity M4 uses so a normalized-id check can prove the two agree, and a
    second ``meta.get("technical_model") or ...`` expression is a second chance
    to disagree about which model an episode ran.
    """
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("technical_model")
               or meta.get("creative_writing_model") or "")


def _resolve_writer_llm_binding(meta: dict, warnings: list):
    """``(generate_fn, model_id)`` for the already-selected writer slot.

    The RAW message-based callable -- ``(messages, *, temperature,
    max_new_tokens) -> str`` -- plus the exact normalized model id the loader
    cached the entry under. Returns ``(None, "")`` for the two legitimate
    no-model paths (``OTR_TEST_MODE``, no model in meta) and RAISES for a
    configured model that will not load, which is the existing fail-loud policy
    and is not softened here.

    Extracted so Ghost Prompt v2 can send a real chat batch without going
    through :func:`_resolve_writer_llm`'s prompt-only wrapper, and without a
    second copy of the slot/policy/GGUF load contract that would drift.
    """

    if otr_env.get("OTR_TEST_MODE") == "1":
        return None, ""
    model_id = writer_model_id_from_meta(meta)
    if not model_id:
        warnings.append("no writer model in meta; creative derivation uses template")
        return None, ""
    try:  # lazy: never import the loader at module scope (V-12)
        # FIXED 2026-06-10 (operator look-QA root cause): this called
        # make_generate_fn(model_id, slot=...) -- a signature that never
        # existed -- so the LLM path failed on EVERY live run and the
        # deterministic template silently carried all creative/image
        # derivation. The real seam is request_slot(slot, model_id) ->
        # cache entry (a same-model call is a cache HIT, no reload) ->
        # make_generate_fn(entry) -> gen(messages, ...).
        from ._otr_model_loader import make_generate_fn, request_slot  # type: ignore
        # Post-ship audit fix (2026-07-10): same policy the writer ran
        # under (ledger stamp); None = pre-stamp backstop.
        from ._otr_shared.llm_policy import policy_from_meta
        from ._otr_gguf_backend import load_config_from_meta

        # GGUF row registry (2026-07-16): thread the writer's exact per-slot load
        # contract so a resident-cache MISS reloads the selected row under ITS
        # registry entry, never the gemma env-fallback. None for a non-GGUF run.
        entry = request_slot(  # LLM slot: technical
            "technical", model_id,
            policy=policy_from_meta(meta),
            load_config=load_config_from_meta(meta, "technical"))
        gen = make_generate_fn(entry)
        return gen, str(entry.get("model_id", model_id) or model_id)
    except Exception as exc:  # noqa: BLE001
        # Operator directive (2026-07-16): a REQUESTED writer LLM that fails must
        # FAIL LOUD -- never silently degrade shot/creative derivation to the
        # deterministic template (that is the local-LM fallback we ban). model_id
        # is non-empty here (the no-model path returned above), and TEST_MODE
        # already returned None up front, so this is always a live requested-model
        # failure.
        warnings.append(f"writer LLM failed ({exc}); no template fallback")
        raise


def _writer_call_at(gen, max_new_tokens: int):
    """Bind a raw message generator to ONE reply budget.

    Split out so a caller that knows how much reply it asked for can size the
    budget to the request instead of inheriting a flat constant, and so the
    reseed loop can re-bind a bigger budget without re-resolving the model.
    """
    def _call(prompt: str) -> str:
        # 0.1 not 0.0: the local HF lane hardcodes do_sample=True and
        # transformers rejects a non-positive temperature (live 30w4
        # catch); 0.1 is near-greedy for short derivation prompts.
        return gen([{"role": "user", "content": str(prompt)}],
                   temperature=0.1, max_new_tokens=int(max_new_tokens))

    return _call


def _resolve_writer_llm(meta: dict, warnings: list,
                        max_new_tokens: int = WRITER_REPLY_TOKENS_DEFAULT):
    """Best-effort writer-slot LLM resolver (V-11: no new model_id widget --
    the model name comes from the ledger meta the writer stamped). Returns a
    ``callable(prompt)->str`` or None. Fails soft to None in headless/test mode
    so the deterministic template carries the episode (the live wiring lands
    with the M4 GPU gate before CW-6).

    UNCHANGED CALL SURFACE. Every M4 caller still hands it a prompt STRING, and
    an omitted budget still yields the historical 300.

    ``max_new_tokens`` IS CURRENTLY UNUSED BY EVERY CALLER, and that is worth
    saying rather than implying otherwise. It was added for
    `derive_creative_directives`, whose reply length scales with the batch --
    but the same change taught that function to call
    :func:`_resolve_writer_llm_binding` + :func:`_writer_call_at` directly so it
    could re-bind a larger budget mid-reseed, which bypasses this wrapper
    entirely. The two remaining callers
    (`otr_meta_brief_image_prompt.py`, both sites) ask for ONE short line each
    and take the 300 default. The parameter stays because the next
    batch-shaped caller should not have to rediscover why 300 was wrong -- see
    :func:`_directive_token_budget`.
    """
    gen, _model_id = _resolve_writer_llm_binding(meta, warnings)
    if gen is None:
        return None
    return _writer_call_at(gen, max_new_tokens)


# ---------------------------------------------------------------------------
# Execution plan (groups + shots) -> ledger['video']
# ---------------------------------------------------------------------------


def _assert_family_inputs_satisfiable_cast_time(engine_name, beat, ledger,
                                               policy, subject_sigils=None,
                                               ghost_prompts=None):
    try:
        from ._otr_video_engines.registry import get_engine, is_registered, EngineNotRunnableError
    except ImportError:
        from _otr_video_engines.registry import get_engine, is_registered, EngineNotRunnableError

    try:
        from ._otr_video_engines import mouth_policy as _mouth  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import mouth_policy as _mouth  # type: ignore

    try:
        from ._otr_video_engines.render_driver import (
            DeferredImageGapError, FamilyInputGap, _is_character_face_beat,
            _is_never_humo_video_role, _line_index, _present_request_tokens,
            _radio_is_host_redirect_applies, _required_inputs_for_engine,
            _uses_ambient_master_audio, build_request, build_request_from_shot,
            engine_family, parse_engine_override)
    except ImportError:
        from _otr_video_engines.render_driver import (
            DeferredImageGapError, FamilyInputGap, _is_character_face_beat,
            _is_never_humo_video_role, _line_index, _present_request_tokens,
            _radio_is_host_redirect_applies, _required_inputs_for_engine,
            _uses_ambient_master_audio, build_request, build_request_from_shot,
            engine_family, parse_engine_override)

    def _effective_cast_time_engine(role: str, eng_id: str) -> str:
        """DELEGATES to the ONE route-freeze authority (2026-07-25, chunk 1a).

        This was the THIRD independent copy of force-map + radio-host redirect.
        It hard-coded the redirect target as the bare literal ``"ltx_audio_in"``
        instead of ``render_driver._NEVER_HUMO_REDIRECT_ENGINE``, and it
        SWALLOWED a malformed force map with only a warning while the render
        path treats the identical condition as terminal (``57f4983a``) -- so a
        typo'd map passed cast-time preflight against the UNFORCED plan and
        then died at render. It is now fail-closed with everything else."""
        try:
            from ._otr_shared import route_freeze as _rf  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_shared import route_freeze as _rf  # type: ignore
        return _rf.effective_engine_for_role(str(role or ""), str(eng_id or ""))

    def _cast_time_image_gap_request(shot):
        return build_request(
            shot,
            {"init_image": "__cast_time_image__"},
            cast_frame_count,
        )

    role = str(beat.get("role") or "")
    cast_frame_count = int(beat.get("target_frame_count") or 1)
    if cast_frame_count <= 0:
        cast_frame_count = 1
    effective_engine = _effective_cast_time_engine(role, engine_name)
    if is_registered(effective_engine):
        eng = get_engine(effective_engine)
        if not getattr(eng, "invocable", True):
            reason = getattr(eng, "invocability_reason", "not currently invocable")
            raise EngineNotRunnableError(
                f"engine {effective_engine!r} is not runnable: {reason} "
                f"-- pick a runnable engine for this role"
            )

    shot = {
        "shot_id": beat.get("beat_id", ""),
        "role": role,
        "engine_id": effective_engine,
        "char_id": beat.get("char_id", ""),
        "source_line_ids": [beat.get("beat_id", "")],
        "target_frame_count": cast_frame_count,
    }
    if beat.get("_synthetic_open"):
        shot["start_s"] = beat.get("_start_s", 0.0)
        shot["dur_s"] = beat.get("dur_s")
    # The temporary shot carries the SAME durable identity the row will (Ghost
    # Signal, 2026-08-22). Default None keeps every existing caller -- fixtures
    # and the flat-import tests among them -- working unchanged.
    _cast_sigil = (subject_sigils or {}).get(str(beat.get("char_id") or "").strip())
    if _cast_sigil:
        shot["subject_sigil"] = _cast_sigil
    # THE GHOST PROMPT v2 OBJECT, and its coverage is ASSERTED rather than
    # tolerated. A registered Ghost-profile beat with no authored object must
    # not fall through to the v1 composer here -- that fall-through would look
    # exactly like a healthy legacy replay while quietly reproducing the
    # name-leaking fragment this sprint exists to remove.
    if ghost_prompts is not None and _ghost_prompt_lane(effective_engine):
        _cast_ghost = ghost_prompts.get(str(beat.get("beat_id") or ""))
        if not _cast_ghost:
            raise ValueError(
                "[OTR_ShotLock] Ghost beat %s reached cast-time preflight with "
                "no authored ghost_prompt. Coverage is exact by contract; a "
                "silent v1 downgrade here is not an acceptable degradation."
                % (beat.get("beat_id"),))
        shot["ghost_prompt"] = copy.deepcopy(_cast_ghost)

    # ONE NARROW CATCH, AND NOTHING ELSE IS SWALLOWED (2026-07-29, WIRE-W2).
    #
    # This was three excepts deep and two of them were FAIL-OPEN: a
    # non-deferrable `ValueError` and then a bare `except Exception` both
    # logged a warning and `return`ed, so cast-time preflight silently passed
    # a beat it had failed to check at all. A preflight that answers "fine"
    # when it crashed is worse than no preflight, because the plan is then
    # built on its silence.
    #
    # Deferrability is now DECLARED BY THE RAISE SITE, not guessed here.
    # The old `_is_deferred_image_gap` substring-matched the message against
    # four needles, and the LTX-I2V gap's wording matched none of them -- so
    # ShotLock re-raised, plan-build died, and `ltx_video` came back NO_RENDER
    # in the 2026-07-28 engine-coverage campaign. Any raise site that means
    # "the image phase has not run yet" says so by TYPE.
    try:
        req = build_request_from_shot(shot, ledger, master_audio_path="")
    except DeferredImageGapError as exc:
        req = _cast_time_image_gap_request(shot)
        log.warning(
            "[OTR_ShotLock] cast-time image input deferred to "
            "ImageGenDispatcher/render gate for engine %r beat %s: %s",
            effective_engine, beat.get("beat_id", ""), exc)

    # build_request_from_shot may still apply an in-place structural redirect;
    # validate the engine that would actually receive this request.
    effective_engine = str(shot.get("engine_id") or effective_engine or "")
    fam = engine_family(effective_engine, "")

    # WHO OWNS THIS BEAT'S MOUTH (WIRE-W7, 2026-07-29).
    #
    # r3 MUST-FIX 11: no W1-W6 step enforced the operator's three rulings, and
    # an unowned ruling silently lapses. Asked HERE because this is the one
    # place that has the FROZEN route -- the effective engine AFTER the
    # route-freeze and the radio-is-host redirect -- and the ruling is decided
    # from the route, never from the beat's prose.
    #
    # It refuses only the case nobody owned: an audio-in beat that is neither a
    # character face nor a cabinet role. Every routed beat in the shipped
    # roster answers HUMAN or RADIO, so this is a gate rather than a change.
    _mouth.mouth_owner_for_beat(
        engine_id=effective_engine, family=fam, role=role,
        is_character_face=_is_character_face_beat(shot))
    required = _required_inputs_for_engine(effective_engine, fam)
    if not required and fam != "static_image_gen":
        return

    line = _line_index(ledger).get(beat.get("beat_id", ""), {})
    is_timed = (
        (line.get("start_s") is not None and line.get("dur_s") is not None)
        or (shot.get("start_s") is not None and shot.get("dur_s") is not None)
        or beat.get("dur_s") is not None
    )
    ambient_audio_deferred = _uses_ambient_master_audio(
        effective_engine, fam, _is_character_face_beat(shot), role=role)
    if "audio_ref" in required and (is_timed or ambient_audio_deferred):
        if req.get("audio_ref") is None:
            req["audio_ref"] = {"path": "__cast_time_master_slice__"}
            if ambient_audio_deferred and not is_timed:
                log.warning(
                    "[OTR_ShotLock] cast-time audio_ref for candidate %r beat "
                    "%s is deferred to VideoRenderBatch master-audio slicing",
                    effective_engine, beat.get("beat_id", ""))

    present = _present_request_tokens(req)

    if fam == "static_image_gen":
        if not ({"text_prompt", "init_image"} & present):
            raise FamilyInputGap(
                "candidate %r (family %s) needs text_prompt or init_image; "
                "the request carries neither -- LOUD skip down the chain"
                % (effective_engine, fam))
        return

    missing = [t for t in required if t not in present]
    if missing == ["init_image"]:
        # ShotLock runs BEFORE OTR_ImageGenDispatcher in the real workflow; the
        # dispatcher mints portraits, scene stills, radio-face stills, and mesh
        # fodder after this plan is stamped. Keep render-time strict (the driver
        # still refuses a wrong-shaped request), but do not freeze-halt a
        # renderable plan merely because the image-phase output is not in the
        # ledger yet.
        log.warning(
            "[OTR_ShotLock] cast-time init_image for candidate %r beat %s is "
            "deferred to ImageGenDispatcher/render-time validation",
            effective_engine, beat.get("beat_id", ""))
        return
    if missing:
        raise FamilyInputGap(
            "candidate %r (family %s) requires input(s) %s the request does "
            "not carry -- LOUD skip down the chain (never feed a wrong-shaped "
            "request to an engine)" % (effective_engine, fam, missing))


#: Re-exported so ``build_execution_plan``'s log line names the operator's
#: number rather than a literal that could drift from the policy's.
_MOUTH_MAX_FACES = 1


def _audit_episode_faces(shots):
    """Ask the mouth policy how many human faces this episode shows.

    Translates SHOT ROWS into the plain route facts the policy takes -- it is
    pure and dependency-free by design, so the translation lives here, next to
    the rows, rather than teaching the policy this module's schema.

    The two derived facts are taken from the SAME authorities the render
    dispatcher uses: ``engine_family`` for the family (a shot row's own
    ``family`` is still ``""`` at this point -- it is filled downstream), and
    ``_is_character_face_beat`` for the talking-head question. A second
    derivation of either is a second chance to disagree with the dispatcher
    about which beats show a face.
    """
    try:
        from ._otr_video_engines import mouth_policy as _mouth  # type: ignore
        from ._otr_video_engines.render_driver import (  # type: ignore
            _is_character_face_beat, engine_family)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import mouth_policy as _mouth  # type: ignore
        from _otr_video_engines.render_driver import (  # type: ignore
            _is_character_face_beat, engine_family)

    global _MOUTH_MAX_FACES
    _MOUTH_MAX_FACES = _mouth.MAX_HUMAN_FACES_PER_EPISODE

    beats = []
    for shot in shots or ():
        engine_id = str(shot.get("engine_id") or "")
        plan = shot.get("coverage_plan") or {}
        beats.append({
            "beat_id": str(shot.get("shot_id") or ""),
            "engine_id": engine_id,
            "family": engine_family(engine_id, ""),
            "role": str(shot.get("role") or ""),
            "char_id": str(shot.get("char_id") or ""),
            "is_character_face": _is_character_face_beat(shot),
            "is_multi_clip": len(plan.get("segments") or ()) > 1,
        })
    return _mouth.audit_episode_faces(beats)


def _lane_consumes_a_still(shot, engine_id):
    """Does this shot's lane take a scene still at all? (2026-07-25, chunk 4.)

    DELEGATES to ``render_driver._still_spine_requires_scene`` on purpose: that
    is the predicate the still spine uses to DEMAND a beat's still back, so
    asking it here means the mint and the demand are the same decision rather
    than two that can drift apart. An audio-reactive visualizer or a
    portrait-only face lane answers False and owes no per-segment stills, which
    is why its beats may jump cut without an image phase at all.

    THE UNCLASSIFIABLE CASE NEVER ARRIVES HERE (corrected 2026-07-26, QA4).
    An earlier draft of this docstring promised that an engine the driver
    cannot classify is treated as still-consuming. The code does not do that,
    and does not need to: :func:`_stamp_coverage_plan` returns before it ever
    asks, for any id the registry does not know, so an unimportable module or
    a stub gets NO plan and NO requests rather than a guessed still. What the
    delegated predicate answers for an unknown id is therefore not this mint's
    contract. The correction is the point: a docstring describing a
    fail-closed branch that does not exist is worse than no docstring, because
    the next reader trusts it and stops looking.
    """
    try:
        from ._otr_video_engines.render_driver import (  # type: ignore
            _still_spine_requires_scene, engine_family)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines.render_driver import (  # type: ignore
            _still_spine_requires_scene, engine_family)
    family = engine_family(engine_id, str(shot.get("family") or ""))
    return bool(_still_spine_requires_scene(shot, engine_id, family))


def _planning_ceiling(policy):
    """The tier's render-length ceiling off the policy, normalized ONCE (B3).

    ``lock()`` stamps this value onto the ledger and ``build_execution_plan``
    plans against it; the render boundary then re-derives from what was
    stamped. So the normalization has to be ONE rule rather than three literal
    copies of ``max(0, int(x or 0))`` that a later edit can desynchronize --
    which would make the stamped receipt and the re-derived one disagree for a
    reason that has nothing to do with the ceiling. It delegates to
    ``frame_contract.normalized_planning_ceiling``, the same function the
    render side calls, so there is exactly one definition in the tree.
    """
    try:
        from ._otr_video_engines import frame_contract as _fc  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import frame_contract as _fc  # type: ignore
    return _fc.normalized_planning_ceiling(
        (policy or {}).get("max_render_frames"))


def _stamp_frame_bounded(shot):
    """Stamp whether this shot's engine has a ceiling a beat can OVERFLOW.

    THE 7.3 DECISION of the no-mirror build, made 2026-08-06: the acceptance
    grader must know which engines are BOUNDED, because a bounded engine's beats
    are split into real segments and therefore owe native-count evidence, while
    an unbounded one can never be split and owes none. The grader cannot work
    that out for itself -- ``acceptance.py`` imports nothing (a test pins its
    import list to exactly ``["__future__"]``) and refuses to query live state on
    principle. So the fact has to reach it as DATA, stamped here, at the same
    freeze that stamps the route and the coverage plan.

    DERIVED FROM ``frame_contract.can_split``, never restated. That function IS
    the boundedness question -- ``bool(max_frames or discrete_frames)``, "does
    this engine have a ceiling to exceed" -- and it is the same predicate
    ``partition_beat`` effectively answers when it decides whether a beat needs
    more than one clip. One definition means the planner and the grader cannot
    disagree about what "bounded" means.

    AND IT IS CLOCK-DOMAIN CORRECT, which is the reason a stamp beats a lookup.
    ``acceptance.py``'s first stated refusal is that it never queries live
    routing state, because "the environment has moved on, so a disagreement
    would report the grader's clock rather than the episode's". An adapter's
    ``frame_contract`` is exactly such state: it can be re-declared between the
    render and the grade. Re-grading a January episode in June must judge it by
    the contract it was RENDERED under, and only a stamp can say what that was.

    ABSENCE MEANS "NOT STATED", NEVER "UNBOUNDED". An unregistered engine, an
    unbuildable one, or a legacy ledger written before this stamp existed simply
    carries no key, and the grader must then demand no native-count evidence.
    Only an explicit ``True`` obliges a shot to prove its frames. Defaulting a
    missing key to ``False`` would read "we checked and it is unbounded", which
    is a claim nobody made; defaulting it to ``True`` would indict every legacy
    episode for lacking a receipt that did not exist when it rendered.
    """
    engine_id = str(shot.get("engine_id") or "")
    if not engine_id:
        return
    try:
        from ._otr_video_engines import registry as _vreg_local  # type: ignore
        from ._otr_video_engines import frame_contract as _fc  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import registry as _vreg_local  # type: ignore
        from _otr_video_engines import frame_contract as _fc  # type: ignore
    if not _vreg_local.is_registered(engine_id):
        return
    try:
        engine = _vreg_local.get_engine(engine_id)
    except Exception:  # noqa: BLE001 -- an unbuildable engine states nothing
        return
    # ``can_split`` is TOTAL -- ``frame_contract_for`` already answers
    # SINGLE_ONLY for a missing or raising declaration -- so there is nothing
    # here to guard against. Wrapping it would only hide a future change that
    # made it partial.
    shot["frame_bounded"] = bool(_fc.can_split(engine))


def _apply_replay_engine_override(planned: dict, override: str, policy: dict) -> None:
    """Rewrite a FROZEN plan onto ``override`` -- whole plan, atomically, in place.

    The still-in lab peer's A/B needs one ledger rendered on two Ghost siblings.
    The frozen route wins at every reader (``build_request_from_shot`` verifies
    ``shot.engine_id`` against ``video.roles_effective`` and refuses on any
    disagreement), so the override must move four surfaces together: every
    role in ``roles_effective`` that the plan routed to the source engine, every
    shot's ``engine_id`` and ``family``, every ``execution_groups[*].engine_id``,
    and each shot's coverage contract re-derived through the same function the
    render boundary uses. Restricted to a registered Ghost SIBLING (equal family,
    roles, prompt_profile and frame_contract) so prompts, seeds, beats and the
    coverage plan stay exactly the frozen ones -- anything else would be a
    re-plan wearing a replay's receipt. Refuses NAMED on any leftover mismatch.
    """
    from ._otr_video_engines import registry as _vreg
    from ._otr_video_engines import frame_contract as _fc_mod

    shots = [s for s in (planned.get("shots") or []) if isinstance(s, dict)]
    if not shots:
        raise ValueError("OTR_ShotLock: replay engine override on a plan with no shots")
    sources = sorted({str(s.get("engine_id") or "") for s in shots} - {""})
    if len(sources) != 1:
        raise ValueError(
            "OTR_ShotLock: replay engine override needs a plan rendered on ONE "
            "engine; the frozen plan routes %r" % (sources,))
    source = sources[0]
    if override == source:
        return
    if not _vreg.is_registered(override):
        raise ValueError(
            "OTR_ShotLock: replay engine override %r is not a registered video "
            "engine" % override)
    if not _vreg.is_registered(source):
        raise ValueError(
            "OTR_ShotLock: the frozen plan's engine %r is not registered on this "
            "tree, so sibling equality cannot be checked" % source)
    src_eng, dst_eng = _vreg.get_engine(source), _vreg.get_engine(override)
    for attr in ("family", "roles", "prompt_profile", "frame_contract"):
        if getattr(src_eng, attr, None) != getattr(dst_eng, attr, None):
            raise ValueError(
                "OTR_ShotLock: replay engine override %r is not a sibling of %r: "
                "%s differs (%r vs %r). A replay never re-plans; use a fresh "
                "render for a different family."
                % (override, source, attr, getattr(src_eng, attr, None),
                   getattr(dst_eng, attr, None)))
    family = str(getattr(dst_eng, "family", "") or "")
    ceiling = _planning_ceiling(policy)
    # Through the same resolver the fresh-plan path and the render boundary
    # use (a callable-or-attribute contract, SINGLE_ONLY on anything else), so
    # the re-derived coverage receipt can never disagree with the boundary's.
    declared = _fc_mod.frame_contract_for(dst_eng)

    roles_eff = planned.get("roles_effective")
    if not isinstance(roles_eff, dict) or not roles_eff:
        # A frozen plan always carries its effective route; without it the
        # render boundary would skip its route check and a partial rewrite
        # would ship unverified. Refuse rather than move three surfaces of four.
        raise ValueError(
            "OTR_ShotLock: replay engine override needs the frozen plan's "
            "roles_effective map and it carries none -- freeze an episode "
            "planned on a tree that stamps the effective route")
    for role, eid in list(roles_eff.items()):
        if str(eid) == source:
            roles_eff[role] = override
    roles = planned.get("roles")
    if isinstance(roles, dict):
        for role, eid in list(roles.items()):
            if str(eid) == source:
                roles[role] = override
    for group in planned.get("execution_groups") or []:
        if isinstance(group, dict) and str(group.get("engine_id") or "") == source:
            group["engine_id"] = override
    for shot in shots:
        shot["engine_id"] = override
        if family:
            shot["family"] = family
        receipt = _fc_mod.coverage_contract_receipt(override, declared, ceiling)
        stamped = shot.get("coverage_contract")
        if receipt is None:
            shot.pop("coverage_contract", None)
        else:
            shot["coverage_contract"] = receipt
        if (stamped is None) != (receipt is None) or (
                stamped is not None and receipt is not None and stamped != receipt):
            raise ValueError(
                "OTR_ShotLock: replay engine override %r re-derives a coverage "
                "contract for shot %s that differs from the frozen one (%r vs %r); "
                "the siblings do not share a ceiling. NO FALLBACK."
                % (override, shot.get("shot_id"), stamped, receipt))
    leftover = [s.get("shot_id") for s in shots if str(s.get("engine_id")) != override]
    if leftover:
        raise ValueError("OTR_ShotLock: replay engine override left shots on the "
                         "source engine: %r" % leftover[:5])


def _stamp_coverage_plan(shot, beat_id, *, max_render_frames):
    """Attach this beat's durable ``coverage_plan`` to its shot row (chunk 3b).

    Resolves the shot's engine to its declared :class:`FrameContract` and
    partitions the beat's ``target_frame_count`` into legal render segments.
    Validated here, at the plan boundary, so a plan that cannot be executed
    never reaches the wire.

    ``max_render_frames`` IS REQUIRED, keyword-only, and has no default (B3,
    2026-07-26). It is the tier's render-length ceiling off the same ``policy``
    dict the ledger stamp reads, and for the engines in
    ``frame_contract.PLANNING_CAP_ENGINES`` it NARROWS the contract this beat
    is partitioned against -- see :func:`frame_contract.effective_frame_contract`
    for why membership is a per-engine decision with a live proof attached
    rather than a rollout. (This read "why that allowlist is one engine long
    and why WAN must stay out of it"; both halves went stale. It is three
    engines, and ``wan_ti2v`` joined on 2026-08-02 when the no-mirror ruling
    removed the adapter-side ping-pong that had made its ceiling harmless.)
    A default of 0 was proposed and rejected: it would let a caller that forgot
    the ceiling plan silently unpinned, which is the exact silent-fallback
    shape this build exists to remove. There is one caller; it passes the
    value. When the ceiling narrows anything, the resulting contract is stamped
    beside the plan as ``shot["coverage_contract"]`` and the render boundary
    re-derives it and requires exact equality.

    FAIL-CLOSED ON A REAL PARTITION FAILURE, tolerant of an ABSENT one. A
    :class:`CoveragePlanError` means the adapter genuinely cannot cover this
    beat -- for a ``single_only`` engine that means the beat exceeds its cap,
    which today is answered by ping-pong, loop-fill or a held frame, i.e. the
    three silent mechanisms this build removes. That must surface, so it
    propagates. An engine that is simply not registered yet (a custom slot, a
    test stub) gets NO plan rather than a guessed one: the row stays absent and
    the render path behaves exactly as it did before 3b.

    ALSO MINTS THE JUMP-STILL REQUESTS (2026-07-25, chunk 4). A jump segment is
    an independent render with nothing to begin from, so every segment after
    the first owes the image phase its own still. They are minted HERE, where
    ``beat_id`` is authoritative rather than re-derived, and stamped durably so
    the image dispatcher and the still spine can READ the ids.

    ONLY FOR A LANE THAT ACTUALLY CONSUMES A STILL (2026-07-25 QA fix). The
    question is asked with ``render_driver._still_spine_requires_scene`` -- the
    SAME predicate the spine will use to demand the stills back. That identity
    is the point. Minting unconditionally created a contradiction with a real
    failure: for an audio-reactive visualizer beat (no scene object, no
    required target) the image dispatcher correctly concluded the lane needs no
    still and skipped, while the spine still demanded every stamped request and
    raised ``RenderError`` before the first render. Two policies over one state,
    whichever fires first wins -- the defect shape this build exists to remove.
    One predicate, asked at the mint, and the disagreement cannot be
    constructed.
    """
    target = int(shot.get("target_frame_count") or 0)
    if target < 1:
        return                      # a zero-length beat has nothing to cover
    engine_id = str(shot.get("engine_id") or "")
    if not engine_id:
        return
    try:
        from ._otr_video_engines import registry as _vreg_local  # type: ignore
        from ._otr_video_engines import frame_contract as _fc  # type: ignore
        from ._otr_video_engines import coverage_plan as _cp  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import registry as _vreg_local  # type: ignore
        from _otr_video_engines import frame_contract as _fc  # type: ignore
        from _otr_video_engines import coverage_plan as _cp  # type: ignore
    if not _vreg_local.is_registered(engine_id):
        return
    try:
        declared = _fc.frame_contract_for(_vreg_local.get_engine(engine_id))
    except Exception:  # noqa: BLE001 -- an unbuildable engine gets no plan
        return
    # THE TIER CEILING PLANS (B3, 2026-07-26), and this call sits OUTSIDE the
    # except above ON PURPOSE. It raises PlanningCapError for a ceiling no
    # legal segment fits under, and that catch means "unbuildable engine, give
    # it no plan" -- absorbing a misconfigured ceiling into it would produce a
    # beat with NO coverage plan, indistinguishable in the log from an
    # unregistered engine, which is chunk 1a's swallowed-fail-closed shape.
    # Three independent reviewers named this line before it was written.
    #
    # ``declared`` STAYS BOUND to the static contract. Both derivations below
    # take the DECLARED one: feeding the already-narrowed contract back into
    # ``coverage_contract_receipt`` makes it compare equal to itself and return
    # None, so the receipt would silently never be stamped and the render
    # boundary would have nothing to check.
    contract = _fc.effective_frame_contract(engine_id, declared,
                                            max_render_frames)
    plan = _cp.partition_beat(target, contract)
    _cp.validate_coverage_plan(plan, contract)

    # HISTORY -- THE REFUSAL THAT USED TO LIVE HERE, kept because the reasoning
    # is what makes the lift below safe to read. It was written 2026-07-26 by
    # the chunk 7a QA panel and REMOVED 2026-07-29 once its prerequisite was
    # built; the note after it says by what.
    #
    # A lane that REQUIRES an audio_ref generates its
    # frames FROM that audio -- HuMo animates a mouth against speech. Splitting
    # such a beat into segments needs each segment to receive its own slice of
    # that speech, and nothing in this build slices it: ``_voice_audio_for_line``
    # takes a line and returns one path, with no segment index anywhere in its
    # signature or its callers.
    #
    # So a split HuMo beat would hand EVERY segment the whole line from its
    # start, and the assembled clip would speak the opening syllables three
    # times over while the audio ran on. That is a sync defect that ships as a
    # finished episode -- the exact failure class this build exists to remove,
    # and worse than the refusal because nothing in the log would say so.
    #
    # Refusing here is not a new gate; it is the SAME refusal moved earlier and
    # given a reason. ``humo_14B_169`` already raised ``MirrorExtensionForbidden``
    # at render time for any beat past its 49-frame cap -- after the GPU work,
    # with a message about mirroring. This one lands at plan time, names the
    # beat, and says what is actually missing.
    # THE REFUSAL ABOVE IS LIFTED (WIRE-W4e, 2026-07-29) BECAUSE ITS STATED
    # PREREQUISITE NOW EXISTS. It read, in full:
    #
    #     "beat %s needs %d clips on %s ... but %s renders frames FROM its
    #      audio_ref and nothing in this build slices that audio per segment
    #      -- every segment would receive the whole line from its start and
    #      the assembled beat would repeat the opening syllables. NO FALLBACK:
    #      per-segment audio is the prerequisite, not a workaround."
    #
    # It was right, and it was right to name the prerequisite instead of
    # inventing a workaround. That prerequisite is now built, for BOTH audio
    # sources a beat can have:
    #
    #   * the FROZEN MASTER slice -- WIRE-W4b narrowed it to the segment's own
    #     render window, and WIRE-W4c made the trimmed tail silence rather than
    #     the next beat's speech (r4/A4).
    #   * a PER-LINE VOICE WAV -- WIRE-W4e slices that too, from its own zero,
    #     through the same `coverage_plan.segment_render_window` authority.
    #
    # A HuMo beat past its cap therefore renders as real multi-clip coverage
    # now, each segment driven by its own slice of the line. This was the ONE
    # thing standing between the audio-driven lanes and the 45-word run: the
    # first campaign leg died here on `beat l001 needs 2 clips on humo (185
    # frames, cap 177)`.
    #
    # If per-segment slicing is ever removed, restore this refusal rather than
    # letting the beats through -- a split lip-synced beat with whole-line
    # audio is a sync defect that SHIPS, and nothing downstream would say so.

    shot["coverage_plan"] = plan.to_dict()
    # THE SIBLING RECEIPT (B3). Present only when the tier ceiling actually
    # narrowed something, and its ABSENCE is as load-bearing as its content:
    # ``render_driver.assert_coverage_plans`` re-derives this with the same
    # function and refuses on any difference, so a ceiling that appeared,
    # vanished or moved between plan time and render time is terminal rather
    # than silently re-planned. Stamped BEFORE the still-lane early return --
    # whether a lane owes stills has nothing to do with what it may render.
    receipt = _fc.coverage_contract_receipt(engine_id, declared,
                                            max_render_frames)
    if receipt is not None:
        shot["coverage_contract"] = receipt
    if not _lane_consumes_a_still(shot, engine_id):
        return
    requests = _cp.jump_still_requests(
        plan, beat_id,
        role=str(shot.get("role") or ""),
        engine_id=engine_id,
        char_id=str(shot.get("char_id") or ""),
    )
    if requests:
        shot["jump_still_requests"] = [dict(row) for row in requests]


def _prompt_owned_lane(engine_id):
    """True when ``engine_id`` is a registered lane that OWNS ITS OWN SUBJECT.

    A CAPABILITY TEST, NOT AN ENGINE-NAME TEST, and deliberately so: the coding
    plan's own law is that "engine-id string tests must not substitute for a
    declared capability at any downstream boundary" (section 3), and the same
    plan describes this filter loosely as "engine is exactly animatediff15_video"
    in section 6.1. The capability reading is the one that survives: it selects
    exactly the same single lane today, it satisfies the stated rationale
    verbatim -- a non-Ghost episode must not acquire a new seed/style
    requirement -- and a second prompt-owned lane is covered the day it declares
    itself rather than the day someone remembers this function exists.

    Answers False rather than raising for an unregistered id: a frozen ledger
    may name an engine this build no longer ships, and plan-building is not the
    seam that should die of it.
    """
    if not engine_id:
        return False
    try:
        from ._otr_video_engines import registry as _vreg  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import registry as _vreg  # type: ignore
    try:
        if not _vreg.is_registered(engine_id):
            return False
        eng = _vreg.get_engine(engine_id)
    except Exception:  # noqa: BLE001 -- a predicate answers, never raises
        return False
    return (str(getattr(eng, "subject_ownership", "") or "") == "prompt"
            and bool(getattr(eng, "prompt_profile", None)))


def _build_subject_sigils(beats, ledger, engine_for):
    """One durable heraldic identity per character on a prompt-owned lane.

    Built ONCE per episode, here, because a subject that changes between the
    cast-time preflight and the durable row is not an identity at all -- and
    because the composer must be able to refuse a character beat that has none.

    THE RAW CAST ROW, NEVER ``_appearance_for_char``. That helper may invoke the
    optional wardrobe writer, which would turn a deterministic identity READ
    into a hidden mutation and a credit spend on every episode that happens to
    contain a Ghost beat.

    ``ledger is None`` yields an empty map: that fixture path already skips the
    cast-time preflight and must stay valid. A missing cast row yields ``{}`` and
    the distiller falls to its checked-in neutral pools -- it never reaches for
    the wardrobe or any other author.
    """
    sigils = {}
    if ledger is None:
        return sigils

    wanted = []
    for b in beats:
        if str(b.get("role") or "") != "character_video":
            continue
        if not _prompt_owned_lane(engine_for(b["role"])):
            continue
        char_id = str(b.get("char_id") or "").strip()
        if char_id and char_id not in wanted:
            wanted.append(char_id)
    if not wanted:
        return sigils

    try:
        from ._otr_video_engines import ghost_signal_prompt as _gsp  # type: ignore
        from ._otr_ledger_consumers import cast_lookup as _cast_lookup  # type: ignore
        from ._otr_visual_styles import get_visual_style as _get_visual_style  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import ghost_signal_prompt as _gsp  # type: ignore
        from _otr_ledger_consumers import cast_lookup as _cast_lookup  # type: ignore
        from _otr_visual_styles import get_visual_style as _get_visual_style  # type: ignore

    meta = (ledger or {}).get("meta") or {}
    # FAIL LOUD BY NAME. Collapsing a missing seed to 0 or "" would silently give
    # every character in the episode a sigil drawn from the same degenerate hash
    # domain -- deterministic, reproducible, and wrong in a way no receipt would
    # ever show.
    if "episode_seed" not in meta or meta.get("episode_seed") in (None, ""):
        raise ValueError(
            "[OTR_ShotLock] a prompt-owned video lane has character beat(s) "
            "(%s) but ledger meta carries no episode_seed. The durable subject "
            "sigil is keyed on it; there is no safe default and it must not "
            "collapse to 0." % ", ".join(wanted))
    episode_seed = meta["episode_seed"]
    style = _get_visual_style(meta)
    style_id = str(getattr(style, "style_id", "") or "")

    for char_id in wanted:
        sigils[char_id] = _gsp.distill_subject_sigil(
            _cast_lookup(ledger, char_id) or {},
            episode_seed=episode_seed, char_id=char_id, style_id=style_id)
    log.info("[OTR_ShotLock] prompt-owned lane: %d durable subject sigil(s) "
             "stamped (%s)", len(sigils), ", ".join(sorted(sigils)))
    return sigils


# ---------------------------------------------------------------------------
# GHOST PROMPT v2 -- the one authoring transaction (2026-08-22)
#
# ShotLock is the authority because it is the only place that holds all four
# things at once: the effective route, the durable identities, the ledger lines
# and the already-selected writer model. Authoring anywhere later would mean
# either a second route resolution or a render-time LLM call, and this lane
# renders from stored strings on purpose.
# ---------------------------------------------------------------------------


def _ghost_modules():
    """The two Ghost text modules, under both import shapes."""
    try:
        from ._otr_video_engines import ghost_signal_author as _gsa  # type: ignore
        from ._otr_video_engines import ghost_signal_prompt as _gsp  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import ghost_signal_author as _gsa  # type: ignore
        from _otr_video_engines import ghost_signal_prompt as _gsp  # type: ignore
    return _gsa, _gsp


def _ghost_prompt_lane(engine_id) -> bool:
    """True when ``engine_id`` DECLARES the Ghost prompt capability.

    A capability read, never an engine-name comparison -- five peers ship this
    profile today (base, official v3, the haunted adapter sibling, hold-3 and
    hold-5) and a sixth must be picked up by declaring it, not by being added
    to a list here.
    """
    eid = str(engine_id or "")
    if not eid:
        return False
    try:
        from ._otr_video_engines.registry import get_engine, is_registered  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines.registry import get_engine, is_registered  # type: ignore
    if not is_registered(eid):
        return False
    _gsa, _gsp = _ghost_modules()
    return (getattr(get_engine(eid), "prompt_profile", None)
            == _gsp.GHOST_PROMPT_PROFILE)


def _ghost_cast_names(ledger) -> tuple:
    """Every known cast name, ordered, for removal at the model boundary."""
    names = []
    for entry in (ledger or {}).get("cast") or []:
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
    return tuple(names)


def _ghost_line_index(ledger) -> dict:
    """``{line_id: line}`` -- the same join the render driver uses."""
    out = {}
    for ln in (ledger or {}).get("lines") or []:
        if isinstance(ln, dict):
            lid = str(ln.get("line_id") or "")
            if lid:
                out.setdefault(lid, ln)
    return out


def _ghost_prior_objects(ledger) -> dict:
    """``{beat_id: ghost_prompt}`` already on the incoming ledger, for replay.

    Joined by ``source_line_ids[0]`` when present and otherwise by stripping
    the durable ``shot_`` prefix, which is what covers the synthetic opening.
    There is no hidden disk lookup: an object replays only if it is already
    IN the ledger this call was handed.
    """
    out = {}
    for shot in (((ledger or {}).get("video") or {}).get("shots") or []):
        if not isinstance(shot, dict):
            continue
        obj = shot.get("ghost_prompt")
        if not isinstance(obj, dict):
            continue
        sids = shot.get("source_line_ids")
        beat_id = ""
        if isinstance(sids, list) and sids:
            beat_id = str(sids[0])
        else:
            shot_id = str(shot.get("shot_id") or "")
            if shot_id.startswith("shot_"):
                beat_id = shot_id[len("shot_"):]
        if beat_id:
            out[beat_id] = obj
    return out


def _ghost_unload_writer(warnings):
    """Release the writer before any preflight / image / video work.

    The image and video phases follow immediately and this lane runs on a
    16 GB card; a writer still resident here is VRAM the render does not have.
    The assertion is the point -- an unload that silently did nothing would be
    indistinguishable from one that worked until the first OOM.
    """
    try:
        from ._otr_model_loader import (  # type: ignore
            has_local_resident_llm, unload_llm_if_local_resident)
    except ImportError:  # pragma: no cover -- flat test imports
        try:
            from _otr_model_loader import (  # type: ignore
                has_local_resident_llm, unload_llm_if_local_resident)
        except ImportError:
            return
    try:
        unload_llm_if_local_resident()
    except Exception as exc:  # noqa: BLE001 -- report, never mask the author
        # REPORTED, THEN STILL CHECKED. Returning here would skip the one line
        # that proves anything: an unload that raised is precisely the case
        # most likely to have left the weights resident, so it is the last
        # moment to skip asking.
        warnings.append("Ghost author: writer unload raised (%s)" % (exc,))
    if has_local_resident_llm():
        raise RuntimeError(
            "[OTR_ShotLock] Ghost author: a local writer LLM is STILL resident "
            "after unload -- refusing to enter the image/video phases holding "
            "writer weights")


def _ghost_validate_batch(leaves, specs, style, meta, names,
                          already_used=()):
    """Raise unless EVERY leaf in the batch is acceptable.

    Whole-batch, deliberately. Salvaging the good rows and re-asking for the
    bad ones would make an episode's prompts a function of how many attempts
    each row happened to take, which is neither reproducible nor auditable.
    """
    _gsa, _gsp = _ghost_modules()
    # SEEDED WITH THE REPLAYED LEAVES. Uniqueness is a property of the EPISODE
    # the viewer watches. The deterministic path was given `already_used` and
    # the authored path was not, so a freshly written leaf could duplicate a
    # replayed one and nothing would notice. Case-folded so both paths agree on
    # what "the same" means.
    seen = {str(leaf).casefold(): "a replayed row"
            for leaf in (already_used or ())}
    for spec in specs:
        leaf = leaves.get(spec["id"], "")
        ok, reason = _gsa.validate_drawable_beat(
            leaf, mode=spec["mode"], names=names)
        if not ok:
            raise _gsa.GhostAuthorValidationError(
                "leaf for %s rejected (%s): %r" % (spec["id"], reason, leaf))
        key = leaf.casefold()
        if key in seen:
            raise _gsa.GhostAuthorValidationError(
                "leaf for %s repeats the one written for %s: %r"
                % (spec["id"], seen[key], leaf))
        seen[key] = spec["id"]
        fits, why = _gsa.candidate_fits(
            role=spec["role"], style=style, mode=spec["mode"],
            motif_cue=spec["motif_cue"], drawable_beat=leaf, ledger_meta=meta)
        if not fits:
            raise _gsa.GhostAuthorValidationError(
                "leaf for %s does not fit (%s): %r" % (spec["id"], why, leaf))


def _ghost_generate_batch(gen, specs, *, style, meta, episode_seed, names,
                          warnings, already_used=()):
    """``(leaves, source, fallback_reason)`` for one whole batch.

    One call for a normal episode. An invalid batch gets ONE fresh whole-batch
    retry -- not a conversation -- and a second failure receives the complete
    deterministic batch with the reason recorded, so the disposition is visible
    in the ledger rather than inferred from prose in a log.
    """
    _gsa, _gsp = _ghost_modules()
    if gen is None:
        return (_gsa.deterministic_batch(specs, episode_seed=episode_seed,
                                         already_used=already_used),
                "deterministic_fallback", "no writer model configured")

    prompt = _gsa.build_batch_prompt(specs)
    budget = _gsa.batch_output_tokens(len(specs))
    ids = [spec["id"] for spec in specs]
    reason = ""
    for attempt in (1, 2):
        # THE RETRY IS A DIFFERENT QUESTION, NOT THE SAME ONE ASKED TWICE.
        # Attempt 2 used to re-send byte-identical text at temperature 0.1 --
        # near greedy -- so a model that wrote a four-word leaf wrote it again
        # and the batch fell to deterministic clauses having spent two
        # generations to learn nothing. That is exactly how the clock-hand
        # false positive cost two live episodes. Now the rejection reasons go
        # back with the request and the sampler runs warmer.
        message = prompt
        temperature = _gsa.GHOST_BATCH_TEMPERATURE
        if attempt > 1 and reason:
            message = "%s\n\nYOUR PREVIOUS ANSWER WAS REJECTED: %s\nFix ONLY that and answer again in the same JSON shape." % (prompt, reason)
            temperature = _gsa.GHOST_BATCH_RETRY_TEMPERATURE
        try:
            raw = gen([{"role": "user", "content": message}],
                      temperature=temperature, max_new_tokens=budget)
        except Exception as exc:  # noqa: BLE001 -- the GENERATION call only
            reason = "attempt %d generation failed: %s" % (attempt, exc)
            warnings.append("Ghost author %s" % reason)
            log.warning("[OTR_ShotLock] Ghost author: %s", reason)
            continue
        # NARROW ON PURPOSE. A broad `except Exception` around the parse and
        # the validators laundered a programming error in OUR OWN code into
        # "the model failed", and the episode quietly took deterministic
        # clauses instead of failing loud. Only a rejected CANDIDATE is caught
        # here; anything else is a bug and must surface.
        try:
            leaves = _gsa.parse_batch_response(raw, ids)
            _ghost_validate_batch(leaves, specs, style, meta, names,
                                  already_used=already_used)
        except _gsa.GhostAuthorError as exc:
            reason = "attempt %d rejected: %s" % (attempt, exc)
            warnings.append("Ghost author %s" % reason)
            log.warning("[OTR_ShotLock] Ghost author: %s", reason)
            continue
        if attempt > 1:
            log.warning("[OTR_ShotLock] Ghost author: batch accepted on the "
                        "informed retry")
        return leaves, "writer_llm", ""
    return (_gsa.deterministic_batch(specs, episode_seed=episode_seed,
                                     already_used=already_used),
            "deterministic_fallback", reason)


def _author_ghost_prompts(beats, ledger, engine_for, warnings=None):
    """``{beat_id: ghost_prompt}`` for every Ghost beat in this episode.

    Runs ONCE, after the effective route and the durable sigils exist and
    BEFORE the cast-time preflight -- because the preflight builds a request
    per beat through the same builder the render path uses, so a Ghost beat
    reaching it without its authored object would either refuse or silently
    fall through to the v1 composer.

    ``ledger is None`` yields an empty map: that fixture path already skips the
    preflight and must stay valid. A REAL ledger with Ghost beats authors every
    one of them.
    """

    warnings = warnings if isinstance(warnings, list) else []
    if ledger is None:
        return {}

    _gsa, _gsp = _ghost_modules()

    ghost_beats = []
    for b in beats:
        role = str(b.get("role") or "")
        picked = str(engine_for(role) or "")
        # BOTH resolutions, because the cast-time preflight validates against
        # the route-frozen engine while the durable row carries the picked one.
        # On a director-built ledger these are the same value; taking the union
        # means a divergence produces an unused map entry rather than an
        # unauthored beat that silently renders on the v1 composer.
        effective = _effective_engine_for_role(role, picked)
        if _ghost_prompt_lane(picked) or _ghost_prompt_lane(effective):
            ghost_beats.append(b)
    if not ghost_beats:
        return {}

    meta = (ledger or {}).get("meta") or {}
    # FAIL LOUD BY NAME, and for a BOOKEND-ONLY Ghost episode too: the mode
    # schedule is keyed on the seed, so collapsing a missing one to 0 would
    # give every episode the same representation rotation -- deterministic,
    # reproducible, and wrong in a way no receipt would show.
    if "episode_seed" not in meta or meta.get("episode_seed") in (None, ""):
        raise ValueError(
            "[OTR_ShotLock] a Ghost-profile video lane has beat(s) (%s) but "
            "ledger meta carries no episode_seed. The representation schedule "
            "and every deterministic clause are keyed on it; there is no safe "
            "default." % ", ".join(str(b.get("beat_id")) for b in ghost_beats))
    episode_seed = meta["episode_seed"]

    try:
        from ._otr_ledger_consumers import cast_lookup as _cast_lookup  # type: ignore
        from ._otr_visual_styles import get_visual_style as _get_visual_style  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_ledger_consumers import cast_lookup as _cast_lookup  # type: ignore
        from _otr_visual_styles import get_visual_style as _get_visual_style  # type: ignore

    style = _get_visual_style(meta)
    style_id = str(getattr(style, "style_id", "") or "")
    names = _ghost_cast_names(ledger)
    lines = _ghost_line_index(ledger)

    # BEFORE ANY MODEL CALL. A pack whose cue plus the longest motif plus a
    # mode law leaves no room for the SHORTEST checked-in clause is a composer
    # constant defect, and no number of retries can fix it -- discovering that
    # as a retry loop would burn a live generation to learn a static fact.
    _gsa.assert_shell_fits([style], ledger_meta=meta)

    modes = _gsa.schedule_ghost_modes(
        [(b["beat_id"], b.get("role")) for b in ghost_beats], episode_seed)

    components = {}
    rows = []
    for b in ghost_beats:
        beat_id = str(b["beat_id"])
        role = _gsa.normalize_role(b.get("role"))
        mode = modes[beat_id]
        if role == "character_video":
            char_id = str(b.get("char_id") or "").strip()
            if char_id not in components:
                components[char_id] = _gsp.distill_sigil_components(
                    _cast_lookup(ledger, char_id) or {},
                    episode_seed=episode_seed, char_id=char_id,
                    style_id=style_id)
            comp = components[char_id]
            motif = _gsa.motif_for_character(comp, mode,
                                             seed_int=comp["seed_int"])
        else:
            motif = _gsa.motif_for_bookend(role, mode)
        line = lines.get(beat_id, {})
        rows.append({
            "beat_id": beat_id,
            "role": role,
            "mode": mode,
            "motif_cue": motif,
            "sanitized_intent": _gsa.sanitize_intent(
                line.get("beat_intent"), names),
            "normalized_emotion": _gsa.normalize_emotion(line.get("traits")),
            "mapped_arc": _gsa.map_arc(line.get("arc_phase")),
        })

    # THE MODEL IDENTITY IS NORMALIZED BEFORE ANYTHING LOADS. `validate_model_id`
    # is pure -- it strips the display label, rejects a structurally unsafe id
    # and confirms an admit path -- so the request hash is computed against the
    # id the loader will actually cache under, and `request_slot` is not called
    # for an episode whose every row replays.
    requested_model = writer_model_id_from_meta(meta)
    model_id = _gsa.GHOST_DETERMINISTIC_MODEL_ID
    if requested_model and otr_env.get("OTR_TEST_MODE") != "1":
        try:
            from . import _otr_model_catalog as _catalog  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            import _otr_model_catalog as _catalog  # type: ignore
        model_id = _catalog.validate_model_id(requested_model)

    specs = _gsa.build_ghost_author_specs(rows, model_id=model_id)
    prior = _ghost_prior_objects(ledger)

    out = {}
    needs = []
    for spec in specs:
        stored = prior.get(spec["beat_id"])
        if isinstance(stored, dict) and \
                stored.get("request_sha256") == spec["request_sha256"]:
            # A SAME-HASH MALFORMED OBJECT FAILS CLOSED. The hash says the
            # inputs are unchanged, so a broken body is corruption rather than
            # a stale artifact to quietly rewrite.
            _gsa.validate_ghost_prompt_object(
                stored, expected_request_sha256=spec["request_sha256"])
            replayed = dict(stored)
            if replayed["source"] == "writer_llm":
                replayed["source"] = "replay"
            out[spec["beat_id"]] = replayed
            continue
        needs.append(spec)

    if needs:
        gen = None
        try:
            if model_id != _gsa.GHOST_DETERMINISTIC_MODEL_ID:
                gen, loaded_model_id = _resolve_writer_llm_binding(
                    meta, warnings)
                if gen is not None and loaded_model_id and \
                        loaded_model_id != model_id:
                    raise ValueError(
                        "[OTR_ShotLock] Ghost author: the loader cached %r "
                        "while the request hash was computed against %r -- a "
                        "stored leaf must name the model that wrote it"
                        % (loaded_model_id, model_id))
            leaves, source, reason = _ghost_generate_batch(
                gen, needs, style=style, meta=meta, episode_seed=episode_seed,
                names=names, warnings=warnings,
                # EVERY leaf already decided by replay, so a fallback cannot
                # collide with one. Uniqueness is a property of the EPISODE the
                # viewer watches, not of whichever subset this call authored.
                already_used=[obj["drawable_beat"] for obj in out.values()])
        finally:
            _ghost_unload_writer(warnings)
        for spec in needs:
            out[spec["beat_id"]] = _gsa.build_ghost_prompt_object(
                spec, leaves[spec["id"]], source=source,
                fallback_reason=(reason if source == "deterministic_fallback"
                                 else ""))

    dispositions = {}
    for obj in out.values():
        dispositions[obj["source"]] = dispositions.get(obj["source"], 0) + 1
    log.info("[OTR_ShotLock] Ghost Prompt v2: %d beat(s) authored (%s), "
             "model=%s, style=%s", len(out),
             ", ".join("%s=%d" % kv for kv in sorted(dispositions.items())),
             model_id, style_id)
    return out


def _effective_engine_for_role(role, engine_id) -> str:
    """One role's effective engine through the ONE route-freeze authority."""
    try:
        from ._otr_shared import route_freeze as _rf  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import route_freeze as _rf  # type: ignore
    return _rf.effective_engine_for_role(str(role or ""), str(engine_id or ""))


def build_execution_plan(beats, budget, creative, policy, ledger=None,
                         warnings=None):
    """Build DAG-validated ``execution_groups`` + per-shot rows.

    CW-1 emits one consumer group per role that has beats (no base-clip
    providers yet -> no edges). Each shot carries its engine_id (from the
    policy), audio-derived ``target_frame_count``, the creative sidecar, and
    cache_keys that deliberately EXCLUDE ``expression`` (3D expression is a
    driver-channel directive, never a cache/mesh key). Returns ``(groups,
    shots)`` after ``resolver.validate_execution_groups``.
    """
    #: THE TIER RENDER-LENGTH CEILING (B3, 2026-07-26). Read ONCE here, off the
    #: same ``policy`` object ``lock()`` stamps onto the ledger, and handed to
    #: every ``_stamp_coverage_plan`` call. For the engines in
    #: ``frame_contract.PLANNING_CAP_ENGINES`` it narrows what the partitioner
    #: may emit; for every other engine -- WAN above all -- it is inert here and
    #: stays an adapter-side native cap, which is what keeps the 8GB WAN tier's
    #: 17-frame contract from becoming 17-frame BEATS.
    max_render_frames = _planning_ceiling(policy)
    def engine_for(role):
        # Route-A: dedicated per-role video slot only (empty resolves empty /
        # fails loud) (ONE shared map; nodes/_otr_shared/role_slots.py).
        #
        # EFFECTIVE FROM BIRTH (chunk 1b). Every consumer of this function --
        # the execution GROUPS, the cast-time preflight, and the shot ROWS --
        # now mints the engine that will actually render. Previously the groups
        # and rows carried the PICKED engine while the preflight quietly
        # re-derived the effective one through its own private mirror, so a
        # redirected bookend was validated as one engine and stamped as
        # another. One resolution, three consumers, no divergence.
        #
        # THE RESOLUTION ITSELF MOVED OUT (2026-08-26) to
        # ``_policy_engine_for_role``, because the PROMPT policy needs the same
        # answer: ``derive_creative_directives`` decides whether a lane may
        # carry the spoken line, and deciding that against a different engine
        # than the one this function stamps is how a lane gets a prompt written
        # for somebody else's adapter. Four consumers now, still one resolution.
        return _policy_engine_for_role(policy, role)

    roles_present = []
    for b in beats:
        if b["role"] not in roles_present:
            roles_present.append(b["role"])

    groups = [{
        "group_id": f"grp_{role}",
        "kind": "consumer",
        "engine_id": engine_for(role),
        "profile_id": "",
        "depends_on": [],
        "produces_base_for": [],
    } for role in roles_present]
    groups = _resolver.validate_execution_groups(groups)

    # THE DURABLE SUBJECT SIGILS (2026-08-22), built AFTER engine_for exists and
    # BEFORE the cast-time preflight -- because the preflight builds a request
    # per beat through the same builder the render path uses, and a Ghost
    # character shot reaching that builder without its identity is refused BY
    # NAME. Deriving the sigil only inside the durable-row loop below would
    # leave the temporary shot without one and force that refusal at plan time.
    subject_sigils = _build_subject_sigils(beats, ledger, engine_for)

    # THE GHOST PROMPT v2 TRANSACTION (2026-08-22), in the same seam and for
    # the same reason as the sigils above: the cast-time preflight below builds
    # a request per beat through the render path's own builder, so the authored
    # object has to exist before it runs and has to be the IDENTICAL object the
    # durable row will carry.
    ghost_prompts = _author_ghost_prompts(
        beats, ledger, engine_for,
        warnings=warnings if isinstance(warnings, list) else None)

    # Preflight family compatibility gate (F2):
    if ledger is not None:
        for b in beats:
            engine_id = engine_for(b["role"])
            if engine_id:
                _assert_family_inputs_satisfiable_cast_time(
                    engine_id, b, ledger, policy, subject_sigils,
                    ghost_prompts)

    # rip-sfx-broll (2026-07-01): the pool_n_loop still/clip POOLING died with
    # the retired_role_a / retired_role_b roles -- every beat renders
    # per-beat with its own scene still (no still_pool_key stamping).
    shots = []
    for b in beats:
        cre = creative.get(b["beat_id"], {})
        _timing = ({"start_s": b.get("_start_s", 0.0), "dur_s": b.get("dur_s")}
                   if b.get("_synthetic_open") else None)
        shots.append({
            "shot_id": f"shot_{b['beat_id']}",
            # Synthetic beats have no ledger LINE, so the shot row itself
            # carries the timeline position (the render driver falls back to
            # it when the line lookup is empty).
            **({"start_s": _timing["start_s"], "dur_s": _timing["dur_s"]}
               if _timing else {}),
            "source_line_ids": [] if b.get("_synthetic_open")
            else [b["beat_id"]],
            "group_id": f"grp_{b['role']}",
            # The shot's video ROLE, stamped explicitly (2026-06-10): the
            # render driver's role-scoped behaviors (the LTX radio-open
            # prompt) read it; before this only group_id embedded the role.
            "role": b["role"],
            # Round 5 F5: the NORMALIZED char_id rides the shot row so the
            # render driver's portrait join never depends on the raw line
            # scheme (the announcer's 'announcer' -> cast row id case).
            "char_id": b.get("char_id", ""),
            "engine_id": engine_for(b["role"]),
            "profile_id": "",
            "family": "",
            # Schema-stable constant post-pooling-rip (every beat is per-beat).
            "strategy": {"mode": "unique_per_beat"},
            "request_seed": 0,
            "target_frame_count": int(budget["per_beat"].get(b["beat_id"], 0)),
            "render_request_hash": cre.get("request_hash", ""),
            "binding_hash": "",
            # cache_keys EXCLUDE expression on purpose (V-7 / PASS-M4): the
            # expression is a driver-channel directive, not part of identity.
            "cache_keys": {
                "prompt_hash": cre.get("prompt_hash", ""),
                "request_hash": cre.get("request_hash", ""),
            },
            "degradation_trail": [],
            "creative": {k: v for k, v in cre.items() if k != "request_hash"},
        })
        # THE DURABLE COVERAGE PLAN (2026-07-25, multi-clip chunk 3b).
        # How many REAL clips cover this beat, exactly how long each render is,
        # and how they join -- computed HERE, at plan time, from the adapter's
        # own static FrameContract, and stamped on the row so the render phase
        # executes a plan rather than inventing one. A wire-only plan would be
        # useless (r3): it must ride the durable ledger or it cannot support
        # replay, and the render boundary would have nothing to validate.
        #
        # NO LONGER INERT (chunk 7a, 2026-07-26). This comment used to end
        # "INERT TODAY BY CONSTRUCTION: every adapter still resolves to
        # frame_contract.SINGLE_ONLY, whose ladder accepts any length, so every
        # beat gets a one-segment plan that renders exactly as it does now."
        # That was true for chunks 3b through 6 and stopped being true in the
        # commit that gave all 31 adapters real ladders. A beat past its
        # engine's cap now genuinely partitions into multiple clips here, and a
        # beat off the quantum grid gets a render length its adapter accepts
        # rather than the one the beat asked for. Left in place as a correction
        # rather than deleted: a reader who remembers the old promise should
        # find out here that it expired, not by trusting it.
        # BOUNDEDNESS FIRST, and separately (no-mirror 7.3). It is stamped by
        # its own function with its own early returns rather than folded into
        # the call below, because ``_stamp_coverage_plan`` returns early for a
        # zero-length beat -- and a beat that renders nothing still has a
        # perfectly knowable engine. Sharing those early returns would leave the
        # stamp missing on shots whose engine is not in doubt at all.
        # THE IDENTICAL VALUE the cast-time preflight already validated against
        # (2026-08-22). Stamped BEFORE the two calls below so a durable row is
        # never observed mid-way through acquiring its identity. Absent for
        # every beat that is not a prompt-owned character shot, and absence is
        # the honest state there -- the field is Optional precisely so a
        # non-Ghost episode carries no trace of a decision it never made.
        _row_sigil = subject_sigils.get(str(b.get("char_id") or "").strip())
        if _row_sigil:
            shots[-1]["subject_sigil"] = _row_sigil
        # THE SAME OBJECT the cast-time preflight already validated against, by
        # value. Absent for every non-Ghost beat, and absence there is the
        # honest state -- the field is Optional precisely so an episode carries
        # no trace of a decision it never made.
        _row_ghost = ghost_prompts.get(str(b.get("beat_id") or ""))
        if _row_ghost:
            shots[-1]["ghost_prompt"] = copy.deepcopy(_row_ghost)
        _stamp_frame_bounded(shots[-1])
        _stamp_coverage_plan(shots[-1], b["beat_id"],
                             max_render_frames=max_render_frames)

    # HOW MANY FACES THIS EPISODE SHOWS (WIRE-W7, 2026-07-29).
    #
    # The per-beat half runs in the cast-time preflight above; this is the half
    # that can only be asked once, of the WHOLE episode: "One face per episode
    # at most, and only for a line the engine can hold in a single take."
    #
    # It runs HERE rather than earlier because it needs the coverage plans --
    # the single-take clause is a question about the STAMPED plan, and the
    # plans do not exist until the loop above has finished. It runs BEFORE the
    # rows reach the ledger, so an episode that breaks the look contract never
    # becomes a lock a downstream phase would honour.
    _faces, _long_takes, _demoted = _audit_episode_faces(shots)
    for _cid, _bids in _demoted:
        # THE CAP ROUTED RATHER THAN REFUSING. Say so LOUDLY and by name: the
        # episode asked for more overheard faces than the house rule allows,
        # and these characters were handed the cabinet instead. This is the
        # remedy the rule's own message used to demand of the operator by hand
        # ("give the other character(s) the cabinet"), now performed. It is a
        # LOOK decision the operator must be able to see and reverse, so it is
        # never silent.
        log.warning(
            "[OTR_ShotLock] LOOK: %r speaks from the CABINET on %s -- the "
            "episode asked for more than %d overheard human face(s), so the "
            "house rule kept the face the episode leans on hardest and gave "
            "this character the set. THE SET SPEAKS BY DEFAULT; A FACE MUST "
            "BE OVERHEARD.", _cid, ", ".join(_bids), _MOUTH_MAX_FACES)
    for _bid, _cid in _long_takes:
        # LOUD, because it is a look defect the operator can see and judge, and
        # the remedy is his: "shorten the line, or let the cabinet speak it".
        log.warning(
            "[OTR_ShotLock] LOOK: beat %s shows the human face of %r across "
            "MORE THAN ONE clip -- the engine cannot hold this line in a "
            "single take, so the face jump-cuts to a regenerated copy of "
            "itself mid-line. Shorten the line, or let the cabinet speak it.",
            _bid, _cid)
    if _faces:
        log.info("[OTR_ShotLock] the set speaks except for %s -- %d overheard "
                 "human face(s) of at most %d", ", ".join(_faces), len(_faces),
                 _MOUTH_MAX_FACES)
    else:
        log.info("[OTR_ShotLock] the set speaks throughout: no overheard human "
                 "face in this episode")
    return groups, shots


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


class OTRShotLock:
    """Registered as ``OTR_ShotLock``. Single ``ledger['video']`` authority."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "lock"
    # episode_id output is ADDITIVE (still-spine ST-6 / DS-3): ShotLock holds
    # the audio-overlaid ledger, so it is the in-graph episode_id authority;
    # the saved json wires it into OTR_ImageGenDispatcher.episode_id so every
    # still lands in episodes/<ep>/stills/. Existing slot indexes unchanged.
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("patched_ledger_json", "video_revision", "shot_report",
                    "done", "episode_id")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "forceInput": True,
                    "tooltip": (
                        "Frozen ledger JSON (OTR_LedgerFreezeCascade out1 "
                        "script_json). ShotLock stamps a video section into it."
                    ),
                }),
                "audio_done": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Audio-done gate (OTR_EpisodeAssembler out3). Wiring it "
                        "orders ShotLock AFTER audio timing freezes so the clip "
                        "budget is bound against the real timeline. Opaque STRING."
                    ),
                }),
                "video_policy_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "forceInput": True,
                    "tooltip": "Per-role selection policy from OTR_VideoDirector.",
                }),
            },
            "optional": {
                # `image_done` (an input socket) and `consistency_gate_warn_only`
                # (a widget) were REMOVED 2026-08-28, safety-gated first:
                #
                # * image_done was a SUPERSEDED UNWIRED FIX. It promised the
                #   image-before-video ordering gate, was unlinked in all 62
                #   graphs carrying this node, and its parameter was never read
                #   -- not even forwarded. The LIVE ordering mechanism is
                #   canonical link 267, ImageGenDispatcher.image_done ->
                #   VideoRenderBatch.image_done, which bypasses ShotLock
                #   entirely. Do NOT wire Dispatcher -> ShotLock instead:
                #   ShotLock already feeds the Dispatcher its locked ledger and
                #   episode_id, so any edge back closes a 90 -> 91 -> 90 cycle.
                #   Removing input index 3 moved gate_in (link 284) dst_slot
                #   4 -> 3; repaired by identity in canonical, variants
                #   regenerated.
                # * consistency_gate_warn_only was displayed, forwarded one
                #   hop, and deleted -- a knob controlling nothing. The helper
                #   `derive_creative_directives` keeps its own parameter (a
                #   direct-call test exercises both values); the NODE no longer
                #   advertises a choice it does not honour.
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Re-lock whenever the ROUTING ENVIRONMENT changes (2026-07-25, 1b).

        ShotLock validates the frozen route against the live environment and
        mints every shot from it, so a cached lock served across an env change
        would hand the dispatcher and the render batch a plan built for a route
        the operator has already replaced. The same fingerprint is used by
        OTR_VideoDirector, so both ends of the freeze invalidate together.
        """
        from ._otr_shared import route_freeze as _rf
        return _rf.snapshot_fingerprint(_rf.routing_env_snapshot())

    # ------------------------------------------------------------------ #
    def lock(self, script_json, audio_done="", video_policy_json="{}",
             gate_in=""):
        from . import _otr_ledger_consumers as _OTRLC
        try:
            from .production_ledger import stamp_durable as _stamp_durable
        except ImportError:  # pragma: no cover -- flat imports
            from production_ledger import stamp_durable as _stamp_durable  # type: ignore

        led = _OTRLC.load_ledger(script_json)
        # STRICT HERE, AND ONLY HERE. This is the canonical graph join: it is
        # gated on the audio_done forceInput, and EpisodeAssembler saves the
        # durable ledger immediately before minting that string. So a join that
        # cannot be proved at THIS call site is an anomaly, and quietly
        # returning the pre-audio ledger would restore the old beat-id space
        # and make the PBUG-20260811-02 repair inert for the run. The other
        # caller (SignalLostVideoRenderer, title-card timing only) keeps the
        # fail-soft default -- see overlay_audio_timing's docstring.
        led = overlay_audio_timing(led, strict=bool(str(audio_done or "").strip()))
        meta = led.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            led["meta"] = meta
        try:
            policy = json.loads(video_policy_json or "{}")
            if not isinstance(policy, dict):
                policy = {}
        except (ValueError, TypeError):
            policy = {}

        # S4 platform-portability (2026-07-10): a NON-EMPTY policy must be
        # version 2 (device_policy/dtype_policy present). A v1 policy here
        # means a stale OTR_VideoDirector emitted it -- fail LOUD before
        # any render work burns on it.
        if policy and int(policy.get("policy_version") or 0) != 2:
            raise ValueError(
                "OTR_ShotLock: video_policy_json carries policy_version="
                f"{policy.get('policy_version')!r}; expected 2. Re-run "
                "OTR_VideoDirector (stale/hand-crafted policy).")

        # THE ROUTE-FREEZE GUARD (2026-07-25, multi-clip coverage chunk 1b).
        # OTR_VideoDirector froze effective routing from a captured environment
        # and stamped that capture into the policy. If the environment has moved
        # since -- a cached upstream node, an operator exporting
        # OTR_FORCE_ENGINE_MAP between nodes, a headless leg submitted to a
        # server booted with a different env -- then the frozen map describes a
        # route that is no longer real, and every still, prompt and shot planned
        # from it is planned for engines that will not run. That is precisely
        # the defect class this build closes, so it is TERMINAL, before any GPU
        # time or image time is spent. A policy with NO snapshot is a pre-1b or
        # hand-built ledger and is left alone.
        _frozen_snapshot = policy.get("routing_env_snapshot")
        if isinstance(_frozen_snapshot, dict) and _frozen_snapshot:
            from ._otr_shared import route_freeze as _rf
            _live_snapshot = _rf.routing_env_snapshot()
            if not _rf.snapshots_agree(_frozen_snapshot, _live_snapshot):
                raise ValueError(
                    "OTR_ShotLock: the routing environment CHANGED after "
                    "OTR_VideoDirector froze it. Frozen %r, live %r. The "
                    "frozen route no longer describes what will render, so "
                    "every still and prompt planned from it is planned for the "
                    "wrong engine. Re-run the graph from OTR_VideoDirector with "
                    "the environment you actually want. NO FALLBACK."
                    % (_frozen_snapshot, _live_snapshot))

        canvas = (policy.get("canvas") or {})
        fps = int(canvas.get("fps") or 25)
        report: list = []
        warnings: list = []
        # CANONICAL REPLAY (campaign item 0): the PLANNED section is reused from
        # the imported ledger -- no LLM derivation, no re-plan, no revision bump.
        # The seeds are the same because the request hashes are the same rows.
        from .production_ledger import replay_descriptor as _replay_descriptor
        if _replay_descriptor(meta):
            planned = led.get("video")
            if not isinstance(planned, dict) or not planned.get("shots"):
                raise ValueError(
                    "OTR_ShotLock: REPLAY needs the frozen ledger's planned video "
                    "section (video.shots) and the bundle carries none -- freeze an "
                    "episode rendered after the durable planned stamp landed.")
            revision = int(planned.get("video_revision") or meta.get("video_revision") or 0)
            meta["video_revision"] = revision
            # REPLAY ENGINE OVERRIDE (still-in lab peer, campaign item 2,
            # 2026-09-02). A DERIVED bundle names the engine the replay must
            # render on; the ledger import stamped it raw. Validated HERE (the
            # only replay seat that imports the video registry) and applied to
            # the WHOLE plan atomically before the durable stamp: the render
            # boundary requires every shot to match roles_effective and its
            # execution group, so a partial re-stamp would be refused there.
            _override = str(meta.get("replay_engine_override") or "").strip()
            if _override:
                _apply_replay_engine_override(planned, _override, policy)
                log.warning("[OTR_ShotLock] REPLAY: engine override -> %r on every "
                            "shot, role and execution group", _override)
            _stamp_durable(sections={"video": planned},
                           meta_updates={"video_revision": revision,
                                         "replay_engine_override": _override},
                           source="OTR_ShotLock:replay")
            patched = json.dumps(led, ensure_ascii=True, separators=(",", ":"))
            log.warning("[OTR_ShotLock] REPLAY: planned section reused (revision %d, "
                        "%d shot(s)), no LLM", revision, len(planned.get("shots") or []))
            return (patched, revision,
                    "shot_lock: replay reuse of the planned section",
                    f"shot_lock:done:rev={revision}",
                    str(led.get("episode_id") or meta.get("episode_id") or ""))

        # Brief disposition, ONCE per run (gap-audit G4 restore).
        try:
            try:
                from ._otr_story_brief_helpers import (  # type: ignore
                    log_story_brief_disposition)
            except ImportError:  # pragma: no cover
                from _otr_story_brief_helpers import (  # type: ignore
                    log_story_brief_disposition)
            log_story_brief_disposition(meta, "shotlock_m4", log)
        except Exception:  # noqa: BLE001
            pass

        beats = extract_beats(led)
        budget = compute_clip_budget(beats, fps)

        # The OPENING-MUSIC scene (operator look-QA 2026-06-10): injected
        # AFTER the budget so the real beats keep their exact cumulative-
        # samples frame math; the synthetic head beat adds its own frames.
        _open_beat, _open_frames = derive_opening_music_beat(led, fps)
        if _open_beat is not None and _open_frames > 0:
            beats.insert(0, _open_beat)
            budget["per_beat"][OPENING_MUSIC_BEAT_ID] = _open_frames
            budget["total_frames"] = int(budget.get("total_frames") or 0) \
                + _open_frames
            report.append(
                "opening-music scene injected: %d frames (head 0..%.2fs) on "
                "the music_visual engine" % (_open_frames,
                                             _open_frames / max(1, fps)))

        creative, cre_warn = derive_creative_directives(
            beats, meta, led,
            # The widget is gone; the helper keeps its parameter and the
            # shipped value was always False.
            consistency_gate_warn_only=False,
            video_policy=policy,
        )
        warnings.extend(cre_warn)

        groups, shots = build_execution_plan(beats, budget, creative, policy,
                                             led, warnings=warnings)

        revision = int(meta.get("video_revision") or 0) + 1
        video_section = {
            "video_revision": revision,
            # S4: the v2 policy stamps ride the ledger so render_driver
            # (and every adapter's assert_usable) sees the SAME device/
            # dtype truth the directors emitted. Defaults = nv50 baseline
            # (tolerates an empty policy in unit fixtures).
            "policy_version": 2,
            "device_policy": str(policy.get("device_policy") or "cuda"),
            "dtype_policy": str(policy.get("dtype_policy") or "fp8_ok"),
            # WAN 8GB launch contract (2026-07-24): the tier's render-length
            # ceiling rides the ledger next to the device/dtype stamps, so the
            # engine sees it on a production leg regardless of how the server
            # was booted. 0 = unpinned. Beat frame targets are untouched.
            # ONE normalization (B3): the same helper build_execution_plan
            # planned against, so the stamped ceiling and the planned ceiling
            # cannot be two expressions that drift apart.
            "max_render_frames": _planning_ceiling(policy),
            "canonical_canvas": {
                "w": int(canvas.get("w") or 832),
                "h": int(canvas.get("h") or 480),
            },
            "fps": fps,
            "execution_groups": groups,
            "roles": policy.get("video_models") or {},
            # BOTH maps ride the ledger (2026-07-25, chunk 1b). ``roles`` stays
            # the PICKED map for continuity; ``roles_effective`` is what every
            # shot was actually minted from, and ``routing_env_snapshot`` is the
            # environment it was frozen under. Stamping only the picked map made
            # an equality failure at render unreplayable -- you could see that
            # the engines disagreed but not what the plan had believed or why.
            "roles_effective": policy.get("effective_video_models") or {},
            "routing_env_snapshot": policy.get("routing_env_snapshot") or {},
            "shots": shots,
            "clip_budget": {
                "total_frames": budget["total_frames"],
            },
            "warnings": warnings,
        }
        led["video"] = video_section
        meta["video_revision"] = revision
        # THE PLANNED SECTION IS DURABLE (campaign item 0, 2026-09-02). Until
        # today it lived only on the wire: every later disk write started from
        # the on-disk file and _merge_with_disk kept no "video" key, so 0 of the
        # 60 newest ledgers carried what the director asked for. Stamped through
        # the SINGLETON (a disk-only save would desync the state later saves
        # merge from), LOUD on failure, and never in test mode.
        _stamp_durable(sections={"video": video_section},
                       meta_updates={"video_revision": revision},
                       source="OTR_ShotLock")

        report.append(f"shot_lock_revision={revision} beats={len(beats)} shots={len(shots)}")
        report.append(
            f"clip_budget: total_frames={budget['total_frames']}"
        )
        report.append(f"execution_groups={[g['group_id'] for g in groups]}")
        for w in warnings:
            report.append(f"WARN: {w}")
            log.warning("[OTR_ShotLock] %s", w)

        patched = json.dumps(led, ensure_ascii=True, separators=(",", ":"))
        done = f"shot_lock:done:rev={revision}"
        episode_id = str(led.get("episode_id")
                         or (led.get("meta") or {}).get("episode_id") or "")
        return (patched, int(revision), "\n".join(report), done, episode_id)
