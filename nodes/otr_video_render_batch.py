"""OTR_VideoRenderBatch -- the in-process render entry that walks the registry
video engines via :mod:`nodes._otr_video_engines.render_driver` (A-S7.5).

Three modes: ``soak`` runs the full-episode A-S7.5 diagnostic soak (40 beats,
all roles, forced mid-episode fallback-trail testing with LOUD restamps, run
TWICE back-to-back for determinism, frozen audio untouched);
``single`` renders
ONE shot via one engine (the focused in-process forward validation); ``episode``
renders one REAL per-beat clip per shot from a ShotLock-planned ledger
(``run_real_episode``) and emits a beat-ordered clip manifest for the downstream
OTR_SilentComposite. Emits the structured render report as a JSON STRING.
Model-agnostic: no model is "primary".

Cold-import clean (V-12): heavy work + the driver import are LAZY inside the
FUNCTION; module scope imports only stdlib. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("OTR")


def _legacy_receipt_bypass_allowed(ledger):
    """May this ledger skip the still-spine validator entirely? Pure.

    The bypass exists for a small number of legacy CPU-only render-driver
    fixtures that use synthetic ``X:\\`` paths and predate the image-dispatch
    receipt. They do not exercise the production handoff, dedicated contract
    tests cover the validator, and production (``OTR_TEST_MODE`` unset) stays
    fail-closed.

    A SHOT THAT OWES PER-SEGMENT STILLS IS CATEGORICALLY INELIGIBLE
    (2026-07-25 QA fix, chunk 4). This bypass skips the WHOLE validator, so
    when chunk 4 added the per-segment jump-still checks inside that same
    validator, the bypass silently grew to cover them too -- nobody decided
    that. A fixture carrying ``jump_still_requests`` is by definition
    exercising the multi-clip handoff, which is exactly the thing that must not
    be waved through the one gate that proves it.

    Extracted from an inline condition so the decision has a name, and so a
    test can pin it directly rather than by inference.
    """
    if os.environ.get("OTR_TEST_MODE", "0") != "1":
        return False
    images = ledger.get("images") if isinstance(ledger, dict) else None
    if isinstance(images, dict) and isinstance(
            images.get("required_scene_targets"), list):
        return False
    shots = ((ledger or {}).get("video") or {}).get("shots") or ()
    return not any(isinstance(shot, dict) and shot.get("jump_still_requests")
                   for shot in shots)


# The by_engine roll-up's IDENTITY fields: each one is a statement about what
# the ENGINE did, so it may only be reported when every clip that engine
# delivered agrees. Ordered as they appear in the per-clip receipt so the two
# read side by side in the ledger. ``vram_peak_mb`` is deliberately NOT here --
# it is a measurement, not an identity (see _roll_up_engine_receipts).
#: ``delivery_scale_mode`` and ``cadence_mode`` join the identity set because
#: each is a statement about WHAT THE ENGINE DID that is constant across every
#: clip that engine delivered. The cadence COUNTS deliberately do not:
#: ``model_frame_count`` and the source/delivered counts and the tail trim vary
#: beat by beat with the beat's own length, so they are per-clip facts and
#: rolling them up could only produce a number no clip actually carries.
_ROLLUP_IDENTITY_FIELDS = ("recipe", "quant", "use_lora", "render_canvas",
                           "family", "delivery_scale_mode", "cadence_mode")


def _roll_up_engine_receipts(receipts):
    """Pure: collapse ONE engine's per-clip receipts into one engine-level row.

    THE RULE IS PER FIELD, NOT PER ROW. Until 2026-07-28 this was
    ``by_engine.setdefault(eng, receipt)`` -- the FIRST clip's receipt stood in
    for every later clip on the same engine. That lost nothing while every clip
    on an engine stamped the same string, and it stopped being true: since
    LANE 2 a prequalification cell NAMES its own departures, so two clips on
    one engine legitimately carry DIFFERENT recipe receipts; and per-shot
    ``render_canvas`` already differs today on any episode that mixes aspect
    ratios on one engine. ``per_clip`` was always lossless -- this stops the
    summary beside it from asserting things it cannot know.

    An identity field is reported ONLY when every clip agrees. When they
    disagree the field is ``None`` and its NAME is listed in ``varied``. So
    ``None`` means "this engine has no single value", which covers BOTH the
    engine that stamped nothing at all and the engine that stamped several --
    ``varied`` is what tells those apart. A consumer therefore never has to
    sniff a sentinel string out of a field whose type it depends on
    (``use_lora`` is a bool; a truthy "(varied)" marker parked there would have
    made the credits card print "lora" for an engine that mixed both).

    ``vram_peak_mb`` is a MEASUREMENT, so it has a correct engine-level answer:
    the WORST clip's peak. It is returned exactly as that clip stamped it (not
    coerced to float) and is never blanked merely because clips disagreed.
    """
    rolled = {}
    varied = []
    for field in _ROLLUP_IDENTITY_FIELDS:
        values = [r.get(field) for r in receipts]
        first = values[0] if values else None
        if all(v == first for v in values):
            rolled[field] = first
        else:
            rolled[field] = None
            varied.append(field)
    worst = None
    worst_key = None
    for r in receipts:
        raw = r.get("vram_peak_mb")
        try:
            key = float(raw)
        except (TypeError, ValueError):
            continue
        if key != key:
            # NaN compares False against everything, so once it became the
            # incumbent no later peak could ever beat it -- the worst-of would
            # DISCARD a real measurement, and the answer would depend on clip
            # order. Latent (gpu_residency stamps int(info.used) // _MB, so no
            # adapter produces a float peak), closed here because the failure
            # is silent LOSS. The identity comparison above is deliberately
            # NOT hardened the same way: a NaN there degrades to "no single
            # value", which is conservative and visible rather than lossy.
            continue
        if worst_key is None or key > worst_key:
            worst_key, worst = key, raw
    rolled["vram_peak_mb"] = worst
    rolled["clip_count"] = len(receipts)
    rolled["varied"] = varied
    return rolled


def _clip_delivered_motion(clip):
    """Pure: did this manifest row actually put rendered frames on the timeline?

    ONE PAYLOAD MAY NOT COUNT "WHICH ENGINE DELIVERED" TWO DIFFERENT WAYS. The
    ``histogram`` key that ships beside ``by_role`` is minted by
    :func:`render_driver.build_clip_manifest`, which counts a row only once its
    clip is real on disk (``if exists: hist[eid] += 1``); the composite reaches
    the same verdict one layer down, emitting a clip segment for a row with
    ``exists`` and a floor/black segment for one without
    (:mod:`nodes.otr_silent_composite`). So ``exists`` is already what
    "delivered" means on both sides of this payload, and the accounting below
    now says the same word.

    ``exists`` ALONE, deliberately -- not ``exists and path``. The manifest
    builder derives ``exists`` FROM the path (a video row is real when the file
    is on disk; a directory row when it holds its full frame count), so adding
    a second path test here would either be redundant or a quiet second opinion
    about a fact the builder already settled.
    """
    return bool((clip or {}).get("exists"))


def _build_render_engines_payload(manifest, vram_peak_mb):
    """Pure: the ``meta.render_engines`` payload. Preserves the existing keys
    (histogram / video_revision / by_role / vram_peak_mb) and ADDS the S-E5
    recipe receipt -- a per-beat ``delivered_engine`` + recipe/quant/LoRA/canvas/
    peak block + a PER-FIELD by-engine roll-up (:func:`_roll_up_engine_receipts`
    -- never one clip's receipt standing in for the rest) -- so every saved
    episode self-documents "what did I use?" (the durable twin of no-fallbacks
    + dropdown labels). An engine that emits no receipt stamps recipe=None
    (never drops the row). No I/O; unit-testable.

    A GAP ROW IS NOT DELIVERED VIDEO (2026-08-26). Until now every manifest row
    was projected into ``by_role`` / ``per_clip`` / the by-engine roll-up with
    no reference to ``exists`` at all. That is harmless only while every planned
    beat renders. This payload is the DURABLE MOTION RECEIPT
    ``OTR_CreditsRoll`` reads to state which engine rendered each beat, so
    billing a floored beat to an engine puts a claim on the credits card that no
    frame on screen supports. An all-refused episode -- which the operator ruled
    on 2026-08-26 must still publish, full audio and correct timing over a
    floored picture -- would advertise a complete slate of rendered video while
    containing none of it.

    **SANCTIONED GAPS REACH THIS FUNCTION, AND THE PATH IS LIVE.** This
    docstring used to say the opposite, in bold, and it was true when written:
    the accounting landed ahead of the control path that feeds it. That path
    shipped on 2026-08-28 (C0-C7). ``render_driver.build_clip_manifest`` now
    stamps gap rows with ``STATUS_SANCTIONED_GAP`` from
    ``sanctioned_gap_beat_ids(led)``, those rows travel in
    ``manifest["clips"]``, and ``is_sanctioned_gap(clip)`` sorts them here. A
    bold instruction to disbelieve a finished path is worse than no comment,
    which is why this paragraph was rewritten rather than deleted.

    THE GAPS ARE COUNTED, NEVER DROPPED. Silently omitting the refused beats
    would trade one untruth for another: a receipt that shows six delivered
    beats where the ledger planned fourteen, with nothing saying where the
    other eight went. ``sanctioned_gap_count`` and ``sanctioned_gap_shot_ids``
    carry them in manifest order (which is beat order), so the two halves add
    back up to the episode.

    THE PAYLOAD IS NEVER EMPTY, and that is load-bearing rather than tidy:
    ``OTR_CreditsRoll._require`` rejects ``{}``/``[]``/``None``/``""``, so an
    all-gap episode returning an empty payload would convert a publishable
    degraded episode into a hard failure at mux time -- the exact outcome the
    sanctioned gap exists to prevent. Every key below is present on every
    return, whatever the clips did."""
    by_role: dict[str, dict[str, int]] = {}
    per_clip: list = []
    receipts_by_engine: dict[str, list] = {}
    # Siblings are imported inside the function here, matching this module's
    # own import-isolation idiom.
    from ._otr_shared import still_receipt as _receipt
    sanctioned_gap_shot_ids: list = []
    # Undelivered WITHOUT a sanction: a real fault, kept apart so the success
    # predicate can tell a refused beat from a broken one.
    unsanctioned_gap_shot_ids: list = []
    for clip in (manifest or {}).get("clips") or []:
        if not _clip_delivered_motion(clip):
            # The beat is a real fact about the episode, so it leaves this loop
            # by the OTHER door rather than vanishing. It keeps its shot id and
            # its position; what it does not keep is a claim that an engine
            # rendered it. Its ``engine_id`` is read PAST deliberately -- the
            # gap row still names the engine that was PLANNED, and reading that
            # name here is the precise move that produced the wrong receipt.
            # The plan stays recoverable from the ledger's own video section,
            # which is where a plan belongs; a receipt records delivery.
            # C5 (2026-08-28): AN ABSENCE IS NOT A SANCTION.
            #
            # This branch used to sweep EVERY undelivered row into the
            # sanctioned-gap list, which was correct only while no gap row
            # could arrive -- nothing upstream could mint one, so the only way
            # to land here was this accounting's own path. Now that a refusal
            # mints an explicit status, three different things reach this
            # branch: a model refusal (sanctioned), a vanished clip, and a
            # render that never ran. Counting them alike would let a crashed
            # episode report itself publishable-degraded, which is precisely
            # the laundering this control path exists to prevent.
            if _receipt.is_sanctioned_gap(clip):
                sanctioned_gap_shot_ids.append(str(clip.get("shot_id") or "?"))
            else:
                unsanctioned_gap_shot_ids.append(
                    str(clip.get("shot_id") or "?"))
            continue
        role = str(clip.get("role") or "?")
        eng = str(clip.get("engine_id") or "?")
        by_role.setdefault(role, {})
        by_role[role][eng] = by_role[role].get(eng, 0) + 1
        receipt = {
            "recipe": clip.get("recipe"), "quant": clip.get("quant"),
            "use_lora": clip.get("use_lora"),
            "render_canvas": clip.get("render_canvas"),
            "vram_peak_mb": clip.get("vram_peak_mb"),
            # WHAT TEXT AND WHICH SEED (campaign item 0, 2026-09-02): carried from
            # the manifest row so a published clip can be tied to its prompt by
            # hash and to its seed by value. None when no receipt was built.
            "prompt_sha8": clip.get("prompt_sha8"),
            "request_seed": clip.get("request_seed"),
            "actual_request_sha": clip.get("actual_request_sha"),
            # family = the engine's render family (audio_driven_face / text_to_video /
            # image_to_video ...) -- the human-readable suffix the credits MODELS.VIDEO
            # rows show (e.g. "visualizer . camera"). Sourced from the manifest
            # clip row. build_clip_manifest writes
            # ``clip.get("family") or shot.get("family") or ""``, so an engine
            # that stamps nothing arrives here as "" -- NOT None (cite re-pinned
            # 2026-07-28 against render_driver.build_clip_manifest). Either way
            # the absence IS the receipt, never a fallback.
            "family": clip.get("family"),
            # THE CADENCE / DELIVERY RECEIPTS (Ghost Signal, 2026-08-22),
            # PRESENT-KEY-ONLY. ``per_clip`` is the LOSSLESS half of this
            # summary -- the roll-up beside it is allowed to say "no single
            # value", this is not allowed to drop one -- and it is an
            # independent hand-written projection, so schema-real data reaching
            # the manifest can still go missing here unless it is copied.
            **{k: clip[k] for k in (
                "model_frame_count", "cadence_mode",
                "cadence_source_frame_count", "cadence_delivered_frame_count",
                "cadence_tail_trim", "delivery_scale_mode") if k in clip},
        }
        per_clip.append({"shot_id": clip.get("shot_id"), "role": role,
                         "delivered_engine": eng, **receipt})
        receipts_by_engine.setdefault(eng, []).append(receipt)
    by_engine = {eng: _roll_up_engine_receipts(rows)
                 for eng, rows in receipts_by_engine.items()}
    return {
        "histogram": (manifest or {}).get("engine_histogram") or {},
        "video_revision": (manifest or {}).get("video_revision"),
        "by_role": by_role,
        "vram_peak_mb": vram_peak_mb,
        "per_clip": per_clip,        # E5
        "by_engine": by_engine,      # E5
        # THE OTHER HALF OF THE EPISODE (2026-08-26). Present on every payload,
        # zero on a clean render -- an absent key would make "no gaps" and "an
        # older ledger that never looked" read identically, and a reader
        # auditing a refusal needs to tell those apart.
        "sanctioned_gap_count": len(sanctioned_gap_shot_ids),
        "sanctioned_gap_shot_ids": sanctioned_gap_shot_ids,
        "unsanctioned_gap_count": len(unsanctioned_gap_shot_ids),
        "unsanctioned_gap_shot_ids": unsanctioned_gap_shot_ids,
    }


def _stamp_render_engines_meta(manifest, vram_peak_mb):
    """Stamp the per-role VIDEO engine map + histogram + VRAM peak + the E5
    recipe receipt into the production-ledger meta so the episode treatment /
    credits sheet reports exactly which engine + recipe rendered each beat.

    S2 durable-persistence contract (credits enrichment 2026-07-03): this
    stamp was previously best-effort (bare ``led.save()`` -- but ``save()``
    returns None on failure and never raises, so a miss was SILENT and the
    MOTION credit line would be missing at OTR_CreditsRoll time). Now routed
    through ``stamp_durable``: LOUD LedgerStampError on save failure in
    production; test-mode injects in-memory only.

    (This node already operates on the SINGLETON (not a wire-parsed local
    dict), so there is no local-to-singleton copy step here -- the payload is
    handed straight to the stamp helper.)"""
    payload = _build_render_engines_payload(manifest, vram_peak_mb)
    from .production_ledger import stamp_durable

    stamp_durable(
        meta_updates={"render_engines": payload},
        source="video_render_batch",
    )


def _stamp_render_trace(receipts, *, render_run_id=""):
    """Campaign item 0 (2026-09-02): the ACTUAL render trace -- one receipt per
    rendered segment, in play order, built by ``render_driver.render_shot`` from
    the request the engine consumed and the clip it returned -- stamped ONCE,
    after every segment succeeded, under ``meta.render_trace`` through
    ``stamp_durable`` (a shallow meta update, so the stamp REPLACES, never
    appends; a re-render of the same episode dir replaces its trace). LOUD on
    save failure like the render-engines receipt beside it: a trace that did not
    persist is a run nobody can cite."""
    rows = []
    for i, r in enumerate(receipts or []):
        if isinstance(r, dict):
            row = dict(r)
            row["render_run_id"] = str(render_run_id or "")
            row["order"] = i
            rows.append(row)
    from .production_ledger import stamp_durable

    stamp_durable(
        meta_updates={"render_trace": rows,
                      "render_trace_version": "render_trace_v1"},
        source="video_render_batch",
    )
    return len(rows)


def _stamp_audio_motion_profiles(amp_rows):
    """S-C C1: build + durably stamp the per-beat ``audio_motion_profiles``
    onto the production ledger from the shots' resolved conditioning WAVs.

    Read-only analysis (each WAV is a per-line voice clip or a read-only slice
    of the FROZEN master -- the master is NEVER touched, so audio byte-identity
    holds). FAIL-SOFT by design: unlike the render-engines receipt (which the
    credits roll requires, so it stamps LOUD), NOTHING consumes this profile yet
    (C2 deferred), so a load/save miss is logged and swallowed -- it must never
    fail the render. Uses the same in-flight-ledger discovery + save_ledger_safe
    seam as nodes 93/85; runs AFTER the render-engines durable stamp so the
    on-disk ledger is already current."""
    if not amp_rows:
        return
    try:
        from ._otr_audio_motion import (build_audio_motion_profiles,
                                        stamp_ledger_audio_motion)
        from . import _otr_ledger as _OTRL
        profiles = build_audio_motion_profiles(amp_rows, lambda r: r.get("_wav"))
        ledger_p = _OTRL.in_flight_ledger_path()
        if ledger_p is None:
            log.info("[OTR_VideoRenderBatch] audio_motion: no in-flight ledger "
                     "singleton; profile not persisted")
            return
        led = _OTRL.load_ledger_safe(ledger_p)
        if led is None:
            log.warning("[OTR_VideoRenderBatch] audio_motion: could not load "
                        "ledger %s; profile not persisted", ledger_p.name)
            return
        stamp_ledger_audio_motion(led, profiles)
        ok = _OTRL.save_ledger_safe(ledger_p, led)
        log.warning("[OTR_VideoRenderBatch] audio_motion: stamped %d profile(s) "
                    "(%d ok) -> %s saved=%s", len(profiles),
                    sum(1 for p in profiles if p.get("ok")), ledger_p.name, ok)
    except Exception as exc:  # noqa: BLE001 -- best-effort, never blocks the render
        log.warning("[OTR_VideoRenderBatch] audio_motion stamp skipped: %s", exc)


class OTRVideoRenderBatch:
    """Registered as ``OTR_VideoRenderBatch``. Walks the model-agnostic render
    loop in-process (NODE_CLASS_MAPPINGS populated). OUTPUT_NODE so it can be the
    terminal of a render-gate workflow."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "render"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("render_report_json", "clip_manifest_json")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["soak", "single", "episode"], {"default": "soak", "tooltip": (
                    "Run mode. 'episode' is the production path (renders one REAL "
                    "per-beat clip per shot from patched_ledger_json). 'soak' and "
                    "'single' are diagnostic harness modes -- some widgets below are "
                    "read ONLY in a specific mode (see engine / oom_index)."
                )}),
                "beats": ("INT", {"default": 40, "min": 1, "max": 400, "tooltip": (
                    "Diagnostic harness only (mode=soak): synthetic beat count. "
                    "Ignored in mode=episode (beats come from the planned ledger)."
                )}),
                "oom_index": ("INT", {"default": 20, "min": 0, "max": 399, "tooltip": (
                    "Diagnostic harness only -- read ONLY in mode=soak (the beat index "
                    "at which to inject a simulated OOM for fallback-trail testing). "
                    "Inert in mode=single and mode=episode."
                )}),
                "frame_count": ("INT", {"default": 25, "min": 1, "max": 240, "tooltip": (
                    "Diagnostic harness only: per-clip frame count, consumed by "
                    "mode=single AND mode=soak. Ignored in mode=episode (the "
                    "per-shot frame budget is planned upstream)."
                )}),
            },
            "optional": {
                "engine": ("STRING", {"default": "viz_camera", "tooltip": (
                    "Diagnostic harness only -- read ONLY in mode=single (forces the "
                    "single engine to render). Inert in mode=soak and mode=episode "
                    "(episode routes each beat by its planned per-role engine)."
                )}),
                "portrait_path": ("STRING", {
                    "default": "",
                    "tooltip": "Diagnostic harness only (mode=single and "
                               "mode=soak): conditioning still for the "
                               "harness render. Inert in mode=episode, where "
                               "each beat's still comes from the planned "
                               "ledger.",
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "Diagnostic harness only (mode=single and "
                               "mode=soak): driving audio for the harness "
                               "render. Inert in mode=episode, where per-beat "
                               "audio comes from the episode's own clips.",
                }),
                "patched_ledger_json": ("STRING", {
                    "default": "{}", "multiline": True, "forceInput": True,
                    "tooltip": (
                        "mode=episode: the OTR_ShotLock-planned (and image-genned) "
                        "ledger JSON. run_real_episode renders one REAL per-beat "
                        "clip per shot; the emitted clip manifest feeds "
                        "OTR_SilentComposite."
                    ),
                }),
                "master_audio_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": (
                        "mode=episode: path to the FROZEN master mix (MP4 or WAV). "
                        "Beats whose ledger line has no per-line *_wav_path get "
                        "audio_ref filled by slicing [start_s, start_s+dur_s] from "
                        "this file (read-only ffmpeg; master is NEVER mutated). "
                        "Wire from OTR_EpisodeAssembler.output_path (the frozen "
                        "master WAV). Leave unset to degrade "
                        "LOUD on missing per-line wavs."
                    ),
                }),
                "image_done": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": (
                        "Image-done gate (mirrors audio_done; ST-0.2 still-spine). "
                        "Wire from OTR_ImageGenDispatcher.image_done so the video "
                        "render cannot start before every episode still exists on "
                        "disk (W4). Opaque STRING; the token is never parsed."
                    ),
                }),
            },
        }

    def render(self, mode, beats, oom_index, frame_count,
               engine="viz_camera", portrait_path="", audio_path="",
               patched_ledger_json="{}", master_audio_path="",
               image_done=""):
        # ``image_done`` is the W4 ordering gate (opaque STRING token from
        # OTR_ImageGenDispatcher): consuming it forces ComfyUI to finish the
        # image phase -- every episode still on disk -- before this node runs.
        # NOTE: this MUST run inside ComfyUI's executor thread (i.e. submitted via
        # /prompt, not a background HTTP-route thread): only there does ComfyUI's
        # model_management evict the umt5/whisper encoders between encode and
        # sample, keeping the heavy in-process forward under the VRAM ceiling.
        # Single resident engine; any per-beat detach/reclaim behavior lives inside
        # the selected engine wrapper. The frozen audio section is read-only
        # throughout.
        import os
        from ._otr_video_engines import render_driver as _rd
        manifest_payload = ""
        if mode == "episode":
            report, manifest_payload, name = self._render_episode(
                _rd, patched_ledger_json,
                master_audio_path=str(master_audio_path or ""))
        elif mode == "single":
            assets = {"init_image": portrait_path or "", "audio_ref": audio_path or ""}
            report = _rd.render_single(engine, assets=assets,
                                       frame_count=int(frame_count))
            name = "node_single_%s.json" % engine
        else:
            assets = {"init_image": portrait_path or "", "audio_ref": audio_path or ""}
            report = _rd.run_gpu_soak(n_beats=int(beats), oom_index=int(oom_index),
                                      frame_count=int(frame_count), assets=assets)
            name = "node_soak.json"
        ok = bool(report.get("ok"))
        payload = json.dumps(report, ensure_ascii=True, default=str)
        # OUTPUT HYGIENE (output-tree contract OH-1, 2026-06-11): ALL JSON
        # run artifacts (episode reports AND the soak/single probe reports
        # that used to land in the now-retired top-level otr/aship) go to
        # the per-machine state tier episodes/_shared/state via the ONE
        # path authority -- never a hand-composed otr/<sub> join.
        try:                                  # durable artifacts the operator polls
            from ._otr_paths import otr_state_dir
            out_dir = str(otr_state_dir())
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                f.write(payload)
            if manifest_payload:
                with open(os.path.join(out_dir, "node_episode_manifest.json"),
                          "w", encoding="utf-8") as f:
                    f.write(manifest_payload)
        except Exception as exc:              # noqa: BLE001
            log.warning("[OTR_VideoRenderBatch] report write failed: %s", exc)
        log.warning("[OTR_VideoRenderBatch] mode=%s ok=%s -> %s", mode, ok, name)
        return {"ui": {"text": [payload[:6000]]},
                "result": (payload, manifest_payload)}

    @staticmethod
    def _render_episode(_rd, patched_ledger_json, master_audio_path=""):
        """Render one REAL episode from a ShotLock-planned ledger ->
        ``(report, clip_manifest_json, report_name)``. Fail-soft: a bad or empty
        ledger yields an error report + an empty manifest so the graph never
        crashes (the procgen floor still carries the visual elsewhere).

        ``master_audio_path``: path to the FROZEN master mix; forwarded to
        :func:`run_real_episode` for per-beat audio slicing.  Read-only."""
        try:
            ledger = json.loads(patched_ledger_json or "{}")
        except (ValueError, TypeError) as exc:
            return ({"ok": False, "mode": "episode",
                     "error": "patched_ledger_json is not valid JSON: %s" % exc},
                    "", "node_episode_report.json")
        if not isinstance(ledger, dict) or not (ledger.get("video") or {}).get("shots"):
            return ({"ok": False, "mode": "episode",
                     "error": "ledger has no video.shots (run OTR_ShotLock first)"},
                    "", "node_episode_report.json")
        # S-F VISUAL SMOKE CAPTURE: persist the exact node-92 INPUT (the
        # OTR_ShotLock-planned ledger + the master audio path) as a forensic
        # artifact -- a twin of the node_episode_report write below. This is the
        # ONLY place that ledger shape (video.shots + images.images) exists on
        # the wire; nothing else persists it, so scripts/otr_visual_smoke.py
        # bakes its bundle from this. Best-effort, never raises; no behavior
        # change to the render (read-only of the inputs).
        try:
            import os as _os
            from ._otr_paths import otr_state_dir as _sd
            _sdir = str(_sd())
            _os.makedirs(_sdir, exist_ok=True)
            with open(_os.path.join(_sdir, "node_episode_input.json"),
                      "w", encoding="utf-8") as _cf:
                json.dump({"ledger": ledger,
                           "master_audio_path": str(master_audio_path or "")},
                          _cf, ensure_ascii=True, default=str)
        except Exception as _cap_exc:  # noqa: BLE001 -- never break the render
            log.warning("[OTR_VideoRenderBatch] smoke-input capture skipped: %s",
                        _cap_exc)
        episode_id = str(ledger.get("episode_id")
                         or (ledger.get("meta") or {}).get("episode_id") or "")
        # Final image-to-video handoff: every effective still consumer must
        # carry an authoritative target receipt and a real file in the active
        # episode's stills directory before any engine request is built. This
        # may repair a stale pending path from its content-addressed pool, but
        # it never substitutes a different beat or a scene still for mesh
        # fodder.
        # THE ROUTE LOCK, BEFORE the still-spine check (2026-07-25, kibitz
        # per-beat-stills r1 -- codex gpt-5.6-sol high + agy, independently).
        # The spine used to be validated against the PICKED engine while the
        # force-map rewrite (run_real_episode) and the radio-is-host redirect
        # (build_request_from_shot, per shot) rewrote the engine AFTERWARDS --
        # so a forced-route or announcer/music-HuMo beat validated its stills
        # against one engine and rendered on another. Resolve first, validate
        # against the truth. Idempotent; run_real_episode calls it again.
        ledger = _rd.resolve_final_shot_engines(ledger)
        if _legacy_receipt_bypass_allowed(ledger):
            # A small number of legacy CPU-only render-driver fixtures use
            # synthetic X:/ paths and predate the image-dispatch receipt. They
            # do not exercise the production handoff; the dedicated contract
            # tests cover this validator. Production (OTR_TEST_MODE unset)
            # remains fail-closed.
            log.info(
                "[OTR_VideoRenderBatch] still-spine receipt check skipped for "
                "legacy OTR_TEST_MODE fixture"
            )
        else:
            _rd.validate_and_repair_still_spine(ledger)
        ep = _rd.run_real_episode(ledger,
                                  master_audio_path=str(master_audio_path or ""))
        # CLIP-FILL Piece 4: move each rendered per-beat clip out of the
        # janitor-swept _shared/tmp scratch tier into the durable
        # episodes/<ep>/clips/ workspace BEFORE building the manifest, so the
        # manifest (and the composite) reference stable paths the sweep won't
        # delete. Best-effort + LOUD; never aborts the render.
        manifest_episode_id = _rd.resolve_episode_id_for_clip_persistence(
            episode_id,
            freeze_timestamp=str(
                ((ledger.get("meta") or {}).get("freeze_timestamp") or "")
            ),
        )
        _rd.persist_episode_clips(ep, manifest_episode_id)
        manifest = _rd.build_clip_manifest(ep, episode_id=manifest_episode_id)
        # Round 5 F2 (warn-only): the per-beat brief-composed prompts must
        # actually DIFFER -- an all-equal sha set means the beat clauses never
        # landed (the 2026-06-10 "one terse prompt x3" eyeball failure).
        diversity = _rd.ltx_prompt_diversity_status(ep.get("trace"))
        if not diversity.get("ok"):
            log.warning("[OTR_VideoRenderBatch] LTX prompt diversity FAILED: "
                        "%s brief-composed prompts all identical (%s)",
                        diversity.get("n"), diversity.get("sha8s"))
        # C4 (2026-08-28): SUCCESS IS "EVERY BEAT IS ACCOUNTED FOR", not
        # "at least one clip exists".
        #
        # The old predicate could not express the operator's 2026-08-27 ruling
        # that an all-refused episode still PUBLISHES: with every beat
        # sanctioned, ``clip_count`` is legitimately 0 and the episode reported
        # FAILURE. Nor could it catch the opposite error -- a render that
        # delivered one clip and silently lost eight others passed, because one
        # was more than none.
        #
        # Both are the same missing idea: a beat is accounted for when it was
        # DELIVERED or when its absence was SANCTIONED. Anything else is an
        # unexplained hole and the episode is not ok. ``degraded`` then carries
        # the ruling's other half -- publishable, but never reported clean.
        from ._otr_shared import still_receipt as _receipt
        _clips = manifest.get("clips") or []
        _sanctioned = sum(1 for c in _clips if _receipt.is_sanctioned_gap(c))
        _delivered_n = sum(1 for c in _clips if (c or {}).get("exists"))
        _unaccounted = len(_clips) - _delivered_n - _sanctioned
        report = {
            "ok": bool(_clips) and _unaccounted == 0,
            "degraded": _sanctioned > 0,
            "sanctioned_gap_count": _sanctioned,
            "unaccounted_beat_count": _unaccounted,
            "mode": "episode",
            "episode_id": manifest_episode_id, "n_beats": manifest["n_beats"],
            "clip_count": manifest["clip_count"],
            "engine_histogram": manifest["engine_histogram"],
            "video_revision": manifest["video_revision"],
            "prompt_diversity": diversity,
            "vram_peak_mb": ep.get("vram_peak_mb"), "trace": ep.get("trace"),
        }
        # Stamp the per-role render-engine map + VRAM peak into the
        # production-ledger meta for the credits roll. S2 durable-persistence
        # contract (credits enrichment 2026-07-03): this stamp is REQUIRED --
        # OTR_CreditsRoll raises on a missing MOTION receipt, so a swallowed
        # save failure here would just move the crash to mux time with less
        # context. LedgerStampError propagates (no silent skip).
        _stamp_render_engines_meta(manifest, ep.get("vram_peak_mb"))
        # THE ACTUAL TRACE (campaign item 0): every segment's receipt, once.
        import uuid as _uuid
        _n_trace = _stamp_render_trace(ep.get("receipts") or [],
                                       render_run_id=_uuid.uuid4().hex[:12])
        report["render_trace_rows"] = _n_trace
        # S-C C1: stamp the per-beat audio_motion_profile onto the production
        # ledger from the shots' resolved conditioning WAVs. Read-only analysis
        # (the frozen master is never touched); fail-soft -- a profiling error
        # must NEVER fail the render (unlike the render-engines receipt above,
        # nothing consumes this yet -- C2 deferred).
        _stamp_audio_motion_profiles(ep.get("audio_motion_rows") or [])
        return (report, json.dumps(manifest, ensure_ascii=True, default=str),
                "node_episode_report.json")


__all__ = ["OTRVideoRenderBatch"]
