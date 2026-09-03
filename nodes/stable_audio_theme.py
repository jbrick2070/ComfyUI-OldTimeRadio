"""OTR_StableAudioTheme -- model-agnostic music-theme node (Wave 1 / 1c).

The generic theme-music surface for the opt-in v2 audio lane. It emits the three
fixed cues (opening / closing / interstitial) the EpisodeAssembler consumes, and
picks its engine from the shared audio-engine registry, dispatching FAIL-CLOSED:

Every music engine is a self-contained ``clip`` engine (the legacy
batch-delegation path was retired in the audio clean-break, 1c):

  * ``musicgen`` (default, ``clip``) -> a per-cue prompt from the Meta brief
    (``_otr_music_prompt.compose_music_prompt``), a per-cue external seed
    (``_seed_to_int64(music_rng_seed, slot)``, G1), the adapter ``generate_clip``
    call, and ``pack_audio_batch`` into the AUDIO-batch contract (C-4).
  * ``stable_audio_music`` (opt-in, flag-gated + HF-token-gated, ``clip``) -> the
    same clip path; native stereo downmixed to mono while the assembly chain is
    mono.

Unlike the voice nodes this node has THREE AUDIO outputs, so it is self-contained
rather than built on the voice-node base; it reuses the shared
``coerce_int_seed`` / ``build_engine_combo`` helpers plus the
``compose_music_prompt`` composer. Engine libraries are lazy-imported inside
``generate``. Teardown runs in ``finally`` BEFORE the ``done`` signal (I-7).
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import gc
import logging

from ._otr_voice_node_common import build_engine_combo, coerce_int_seed
from ._otr_music_prompt import compose_music_prompt

log = logging.getLogger("OTR")

ROLE = "music"
_LEGACY_FIRST_FALLBACK = ("musicgen", "stable_audio_music")

# The three fixed theme cues. Durations + the per-cue prompt are composed by the
# single-source Meta-brief composer (nodes/_otr_music_prompt.compose_music_prompt),
# so music pulls period/setting/mood from the same propagating creative brief as
# every other downstream creative call -- not a local template.
_CUE_SLOTS = ("opening", "closing")


class StableAudioTheme:
    """Generic theme-music node. Registered as ``OTR_StableAudioTheme``.

    Engine order: musicgen (legacy byte-identical default) > stable_audio_music.
    """

    CATEGORY = "OldTimeRadio/v2/audio"
    FUNCTION = "generate"
    # 720-bakeoff C3: ONE padded AUDIO batch of ALL cues + the manifest that
    # maps each batch row back to a cue (opening / closing / interstitial,
    # scifi_news_pro authored cues included). The old three-AUDIO opening/closing/
    # interstitial surface is retired -- consumers slice by manifest sample_count.
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cue_audio_clips", "cue_manifest_json", "render_log", "done")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        engines = build_engine_combo(ROLE, _LEGACY_FIRST_FALLBACK)
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "forceInput": True,
                    "tooltip": (
                        "Frozen v2 ledger JSON from OTR_LedgerFreezeCascade. "
                        "Passed VERBATIM to the legacy engine on the batch "
                        "path; read for cue mood on the clip path."
                    ),
                }),
                "engine": (engines, {
                    "default": engines[0],
                    "tooltip": (
                        "Theme-music engine. Legacy MusicGen is the "
                        "byte-identical default; stable_audio_music is opt-in "
                        "(flag + HF token) until the GPU pilot promotes it. "
                        "Unusable selections fail closed with a named error."
                    ),
                }),
            },
            "optional": {
                "ledger_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Cast-locked ledger from OTR_CastLock (carries "
                        "episode_seed). Read on the clip path for the per-cue "
                        "seed; the batch path delegates the raw script_json."
                    ),
                }),
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Optional ordering signal. Wire an upstream 'done' "
                        "here to force this node to run after it."
                    ),
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # D: local-disk only; engine/token/model checks are fail-closed on the
        # generate() path, not here, so a box-fresh graph validates clean.
        return True

    # ------------------------------------------------------------------ #
    def generate(self, script_json, engine, ledger_json="", gate_in="",
                 stereo_policy="mono_safe"):
        # CANONICAL REPLAY (campaign item 0): the cues are inside the frozen
        # master; no music model loads. The manifest is "" on purpose -- node 7
        # takes its replay branch BEFORE the cue-pair check.
        try:
            import json as _json
            from .production_ledger import replay_descriptor as _replay_descriptor
            from ._otr_resolved_request import empty_audio_batch as _empty_audio
            _rmeta = (_json.loads(ledger_json or script_json or "{}") or {}).get("meta") or {}
        except (ValueError, TypeError, ImportError, AttributeError):
            # AttributeError: a non-dict wire (the legacy parser LIST) is not a replay
            _rmeta = {}
        if _rmeta and _replay_descriptor(_rmeta):
            log.warning("[OTR_StableAudioTheme] REPLAY: pass-through, no music model")
            return (_empty_audio(), "", "replay: pass-through (frozen master)",
                    "replay:passthrough")
        from ._otr_audio_engines import (
            EngineUnusable, EngineUsabilityReason, assert_usable, get_engine,
            pack_audio_batch,
        )
        from ._otr_audio_engines.base import canonical_audio, mono_safe
        from . import _otr_cue_manifest as _CM

        render_log: list = []
        cue_rows = None   # list of per-cue dicts (clip + authored metadata)
        adapter = None
        sr = 44100
        try:
            engine = assert_usable(engine, ROLE)  # FAIL CLOSED (6-class)
            adapter = get_engine(engine)
            # S4 platform-portability (2026-07-10): theme music reads the
            # SAME CastLock ledger stamp (meta.voice_device) the voice nodes
            # do -- ONE explicit device truth per episode. Adapters without
            # a local device (sonilo/google/comfy-native SA3) ignore it.
            from ._otr_voice_node_common import _voice_device_from_ledger
            adapter.requested_device = _voice_device_from_ledger(
                ledger_json, script_json)
            interface = getattr(adapter, "interface", "clip")

            if interface == "clip":
                cue_rows, sr, lines = self._render_clips(
                    adapter, engine, script_json, ledger_json,
                )
                render_log.extend(lines)
            else:
                raise EngineUnusable(
                    engine, ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                    f"engine {engine!r} declares unsupported interface "
                    f"{interface!r} for role {ROLE!r}",
                )
        finally:
            # I-7: free single-engine residency in finally, BEFORE 'done'.
            self._teardown(adapter)

        mono = (stereo_policy == "mono_safe")
        canon = mono_safe if mono else canonical_audio

        if not cue_rows:
            # Engine produced nothing (fail-closed already raised for a bad
            # engine; this is the empty-ledger / no-cue floor). Emit the
            # canonical empty batch + an empty (but valid) manifest.
            srx = int(sr or getattr(adapter, "sample_rate", 44100) or 44100)
            cue_audio = pack_audio_batch([], sample_rate=srx, mono=mono)
            manifest_json = _CM.dumps(_CM.build_manifest(srx, []))
            n_cues = 0
        else:
            clips = [row["clip"] for row in cue_rows]
            # Exact per-cue length AFTER channel canonicalization but BEFORE
            # the batch right-pad, so a consumer's [bi, :, :sample_count]
            # slice recovers the cue with ZERO padding (MF-G: no silence trim).
            sample_counts = [
                int(canon(clip)["waveform"].shape[-1]) for clip in clips
            ]
            cue_audio = pack_audio_batch(clips, sample_rate=int(sr), mono=mono)
            packed = cue_audio["waveform"]  # [B, C, max_T]

            audio_dir = self._episode_audio_dir()
            manifest_rows = []
            for bi, row in enumerate(cue_rows):
                sample_count = sample_counts[bi]
                # Slice the exact (unpadded) cue out of the packed batch and
                # persist it to the episode audio dir (canonical path first,
                # never tmp -- CLAUDE.md section 6). Byte-identical to what the
                # consumer slices at [bi, :, :sample_count].
                cue_slice = packed[bi, :, :sample_count]
                output_path = ""
                if audio_dir is not None:
                    output_path = self._write_cue_wav(
                        audio_dir, row["cue_id"], cue_slice, int(sr))
                manifest_rows.append({
                    "cue_id": row["cue_id"],
                    "batch_index": bi,
                    "sample_count": sample_count,
                    "sample_rate": int(sr),
                    "prompt": row["prompt"],
                    "prompt_sha256": _CM.prompt_sha256(row["prompt"]),
                    "cue_spec_sha256": row.get("cue_spec_sha256"),
                    "placement": row["placement"],
                    "anchor_line_id": row.get("anchor_line_id"),
                    "seed": int(row["seed"]),
                    "requested_duration_s": float(row["requested_duration_s"]),
                    "actual_duration_s": float(sample_count) / float(sr),
                    "output_path": output_path,
                })
            manifest_json = _CM.dumps(_CM.build_manifest(int(sr), manifest_rows))
            n_cues = len(manifest_rows)

        # S2 durable persistence (credits enrichment 2026-07-03): the music
        # engine had NO durable provenance path -- node 83's ``done`` output
        # is unlinked in the workflow JSON, so ``music:done:engine=...`` died
        # on the wire and the credits roll had nothing to read. Stamp the
        # delivered engine into the singleton meta and save LOUDLY (raises
        # LedgerStampError on save failure; test-mode injects in-memory only).
        from .production_ledger import stamp_durable
        stamp_durable(
            meta_updates={"music_engine": str(engine)},
            source="music_theme",
        )

        done = f"music:done:engine={engine}:cues={int(n_cues)}"
        return (cue_audio, manifest_json, "\n".join(render_log), done)

    # ------------------------------------------------------------------ #
    def _render_clips(self, adapter, engine, script_json, ledger_json):
        """Render every cue for this episode into raw clips + metadata.

        scifi_news_pro lane: one clip per authored ``ledger.music[]`` row (each carries
        its own generation_prompt / placement / anchor_line_id). Legacy lane
        (no ledger.music[]): SYNTHESIZE the three fixed cues via
        compose_music_prompt with the SLOT seed key -- byte-identical to the
        pre-C3 opening/closing/interstitial render.
        """
        from ._otr_engine_profiles import (
            assert_model_available, assert_token_for_profile, require_resolver,
        )
        from ._otr_determinism import deterministic_inference
        from ._otr_resolved_request import _seed_to_int64

        resolver = require_resolver()
        profile = resolver.resolve_casting_plan(role=ROLE, engine=engine)
        assert_token_for_profile(profile)   # MISSING_HF_TOKEN if HF-gated
        assert_model_available(profile)
        sr = int(profile.sample_rate or getattr(adapter, "sample_rate", 44100))

        led = self._load_ledger(ledger_json, script_json)
        meta = (led.get("meta") or {}) if isinstance(led, dict) else {}
        music_rows = (led.get("music") or []) if isinstance(led, dict) else []
        ledger_lines = (led.get("lines") or []) if isinstance(led, dict) else []
        music_seed_base = _seed_to_int64(
            "music_rng_seed_v1", coerce_int_seed(meta.get("episode_seed")),
        )

        cue_specs = self._resolve_cue_specs(meta, music_rows, ledger_lines)
        cue_rows = []
        log_lines = [
            f"music: rendering {len(cue_specs)} cue(s) on '{engine}' "
            f"(profile {profile.profile_id})"
        ]
        for spec in cue_specs:
            prompt = spec["prompt"]
            duration_s = spec["requested_duration_s"]
            engine_seed = _seed_to_int64(music_seed_base, spec["seed_key"])  # G1
            # G1: scope determinism + seed/restore around the single forward
            # (non-strict; bit_exact is gated on the F pilot -- see voice path).
            with deterministic_inference(engine_seed, warn_only=True):
                clip = adapter.generate_clip(prompt, duration_s, engine_seed)
            cue_rows.append({
                "cue_id": spec["cue_id"],
                "placement": spec["placement"],
                "prompt": prompt,
                "seed": int(engine_seed),
                "requested_duration_s": float(duration_s),
                "anchor_line_id": spec.get("anchor_line_id"),
                "cue_spec_sha256": spec.get("cue_spec_sha256"),
                "clip": clip,
            })
            log_lines.append(
                f"  [{spec['cue_id']}/{spec['placement']}] "
                f"{duration_s:.0f}s at {sr} Hz")
        return cue_rows, sr, log_lines

    # ------------------------------------------------------------------ #
    def _resolve_cue_specs(self, meta, music_rows, ledger_lines=None):
        """Build the ordered cue-spec list. scifi_news_pro (authored music[]) vs legacy
        (synthesize the 3 fixed cues, byte-parity slot seed keys)."""
        from ._otr_music_prompt import CUE_DURATIONS, compose_music_prompt
        from .production_ledger import music_cue_spec_sha256

        specs: list = []
        if music_rows:
            for row in music_rows:
                cue_id = str(row.get("cue_id") or "").strip()
                if not cue_id:
                    continue
                placement = self._canonical_placement(row.get("placement"), cue_id)
                prompt = row.get("generation_prompt") or row.get("description") or ""
                if str(prompt).strip():
                    dur = row.get("target_duration_s")
                    duration_s = (
                        float(dur)
                        if isinstance(dur, (int, float)) and float(dur) > 0.0
                        else float(CUE_DURATIONS[placement])
                    )
                else:
                    # Authored row with no prompt: compose from the brief by
                    # placement (MF-B: map placement -> canonical cue so
                    # compose_music_prompt never KeyErrors on inter_NN).
                    prompt, duration_s = compose_music_prompt(meta, placement)
                specs.append({
                    "cue_id": cue_id,
                    "placement": placement,
                    "prompt": str(prompt),
                    "requested_duration_s": float(duration_s),
                    "seed_key": cue_id,
                    "anchor_line_id": row.get("anchor_line_id"),
                    "cue_spec_sha256": (
                        row.get("cue_spec_sha256")
                        or music_cue_spec_sha256(row)
                    ),
                })
        else:
            role_lines: dict[str, list[str]] = {
                "music_open": [], "music_inter": [], "music_close": [],
            }
            for line in ledger_lines or []:
                if not isinstance(line, dict):
                    continue
                role = str(line.get("speaker_role") or "")
                line_id = str(line.get("line_id") or "")
                if role in role_lines and line_id:
                    role_lines[role].append(line_id)
            anchor_by_slot = {
                "opening": (
                    role_lines["music_open"][0]
                    if role_lines["music_open"] else None
                ),
                "interstitial": (
                    role_lines["music_inter"][0]
                    if role_lines["music_inter"] else None
                ),
                "closing": (
                    role_lines["music_close"][-1]
                    if role_lines["music_close"] else None
                ),
            }
            for slot in _CUE_SLOTS:
                prompt, duration_s = compose_music_prompt(meta, slot)
                anchor_line_id = anchor_by_slot[slot]
                cue_spec_sha256 = music_cue_spec_sha256({
                    "generation_prompt": prompt,
                    "target_duration_s": float(duration_s),
                    "placement": slot,
                    "anchor_line_id": anchor_line_id,
                })
                specs.append({
                    "cue_id": slot,
                    "placement": slot,
                    "prompt": prompt,
                    "requested_duration_s": float(duration_s),
                    "seed_key": slot,
                    "anchor_line_id": anchor_line_id,
                    "cue_spec_sha256": cue_spec_sha256,
                })
        return specs

    @staticmethod
    def _canonical_placement(placement, cue_id):
        """Map an authored placement / cue_id to one of opening/closing/
        interstitial (MF-B: the composer + CUE_DURATIONS tables are keyed on
        these, so inter_NN must resolve to 'interstitial')."""
        from ._otr_cue_manifest import canonical_placement
        return canonical_placement(placement, cue_id)

    # ------------------------------------------------------------------ #
    def _load_ledger(self, *sources):
        """Full ledger dict from the first parseable source (empty on miss)."""
        from . import _otr_ledger_consumers as _OTRLC

        for src in sources:
            if not (src or "").strip():
                continue
            try:
                return _OTRLC.load_ledger(src)
            except Exception:  # noqa: BLE001 -- fall back to the neutral default
                continue
        return {}

    @staticmethod
    def _episode_audio_dir():
        """The in-flight episode audio dir (the ledger's parent), created if
        needed. None when headless / no ledger is in flight."""
        import pathlib

        try:
            from . import _otr_ledger as _OTRL  # type: ignore
            lp = _OTRL.in_flight_ledger_path()
            if lp is not None:
                d = pathlib.Path(lp).parent
                d.mkdir(parents=True, exist_ok=True)
                return d
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _write_cue_wav(audio_dir, cue_id, waveform, sample_rate):
        """Write one cue's (unpadded) waveform to a 16-bit PCM wav in the
        episode audio dir. Returns the path, or "" on failure (best-effort --
        the manifest still carries the in-memory batch)."""
        import pathlib
        import wave

        import numpy as np

        try:
            arr = (
                waveform.detach().cpu().numpy()
                if hasattr(waveform, "detach") else np.asarray(waveform)
            )
            arr = np.atleast_2d(arr)          # [C, T]
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[-2], arr.shape[-1])
            pcm = (arr * 32767.0).clip(-32768, 32767).astype(np.int16)
            n_ch = int(pcm.shape[0])
            interleaved = pcm.T.copy(order="C")   # (T, C) for WAV
            safe = "".join(
                c if (c.isalnum() or c in "._-") else "_" for c in str(cue_id)
            )
            path = str(pathlib.Path(audio_dir) / f"music_cue_{safe}.wav")
            with wave.open(path, "wb") as wav_f:
                wav_f.setnchannels(n_ch)
                wav_f.setsampwidth(2)             # 16-bit
                wav_f.setframerate(int(sample_rate))
                wav_f.writeframes(interleaved.tobytes())
            return path
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------ #
    @staticmethod
    def _teardown(adapter):
        """I-7 teardown: unload local engines + free VRAM. Best-effort.

        Cloud music adapters declare ``native=False`` and have no local model
        residency, so do not import torch / touch CUDA for those routes.
        """
        try:
            if adapter is not None and hasattr(adapter, "unload"):
                adapter.unload()
        except Exception:  # noqa: BLE001
            pass
        try:
            gc.collect()
            if adapter is not None and getattr(adapter, "native", True) is False:
                return
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
