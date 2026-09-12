"""Bark voice adapter -- self-contained per_line (clean-break 1a).

Serves both char_voice and announcer_voice (2026-08-24) -- the same v2/*
preset mechanism either way; the caller's ``role`` (threaded onto the
adapter instance by the dispatch core) only selects which curated profile
(``char_bark_v1`` / ``announcer_bark_v1``) supplies the per-stage temps.

Sources inference from _otr_bark_lib (relocated, delegation-free); no
construction of the heavy batch node. interface == "per_line". Library imports
are lazy so importing the registry package stays light (C-5).

Gate 3 (voice-path-cleanbreak): an empty / non-v2/* voice_preset is a writer
cast-lock contract violation -- generate_voice fails closed with a named
EngineUnusable(MALFORMED_CONFIG), the same renderability net the legacy batch
node enforced before it was retired.

The live per-stage temperatures come from the char_bark_v1 profile
(config/audio_engine_profiles.yaml, the curated-params SSOT, plan D5) via
_resolve_stage_temps, which honors the legacy text_temp/waveform_temp KEY
aliases inside the profile dict; tests/test_bark_voice_stage_temps.py pins
that resolution ladder. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import logging

from .registry import register

log = logging.getLogger("OTR")


class BarkSilentOutputError(RuntimeError):
    """Bark returned audio no listener could hear -- empty, nonfinite, or with
    a global peak below 1e-4 (~-80 dBFS, the sequencer's own silence
    threshold).

    A DEDICATED type, deliberately: the C2 verdict (2026-08-28) forbids
    remapping the preset or engine on this failure, and a dedicated class
    means no broad `except` an engine-fallback path might own can quietly
    absorb it into a remap. It is a RENDER error -- the line failed, loudly,
    before any downstream spend -- not a usability reason: the engine itself
    is fine and stays selectable.
    """


@register
class BarkEngine:
    name = "bark"
    roles = ("char_voice", "announcer_voice")
    default_roles = ()                   # DEMOTED 2026-06-04: indextts2 is now the char_voice default; bark stays selectable
    commercial_clean = False             # Suno Bark terms not confirmed commercial
    requires_flag = None
    interface = "per_line"
    sample_rate = 24000
    supports_external_generator = False  # Bark.generate binds no external Generator
    voice_ref_field = "voice_preset"     # dispatch routes cast.voice_preset to the ref slot
    # Per-stage temperatures (2026-06-17 whiny-voice fix). These class attrs are
    # the FALLBACK baseline when the char_bark_v1 profile is missing/malformed;
    # the live values come from the profile via _resolve_stage_temps (which honors
    # an explicit 0.0 and accepts the text_temp/waveform_temp aliases). semantic
    # stays warm (0.7) for content commitment; coarse/fine drop to 0.5 to firm up
    # the acoustic stages (the thin/whiny timbre came from over-hot acoustics).
    semantic_temp = 0.7
    coarse_temp = 0.5
    fine_temp = 0.5

    def __init__(self):
        self._loaded = False
        self._presets_started = set()    # first-line anti-hallucination guard tracking

    def load(self):
        if self._loaded:
            return
        from .._otr_bark_lib import _load_bark

        # Thread the CastLock ledger's device stamp, exactly as
        # eng_musicgen / eng_kokoro / eng_stable_audio already do
        # (`getattr(self, "requested_device", None) or "cuda"`, the S4
        # 2026-07-10 idiom stamped by _otr_voice_node_common.py).
        #
        # bark was the one adapter in this group that never joined that sweep:
        # it passed NO device, so `_load_bark` fell through to its own
        # auto-probe and the operator's `voice_device` selection was discarded
        # before it could be honoured. Found by the 2026-09-07 portability
        # sweep, and it is why bark ran on CPU on a Mac even after the probe
        # itself learned about mps.
        _load_bark("suno/bark", device=self._requested_device())
        self._loaded = True

    def _requested_device(self):
        """The device the ledger asked for, or ``None`` to let `_load_bark`
        auto-detect.

        DELIBERATELY NOT `or "cuda"`, which is what the sibling adapters
        (eng_musicgen, eng_kokoro, eng_stable_audio) use. Those hand their
        default straight to a `.to(device)`, so a literal "cuda" default is
        merely the nv50 baseline for them. Bark is different: `_load_bark`
        treats ``device=None`` as "auto-detect", and that probe is now
        cuda -> mps -> cpu. Passing a hard "cuda" here would FORCE cuda on a
        Mac and break the very platform this change exists to serve -- caught
        by tests/test_bark_silent_output_gate.py when it was written that way.

        So: an explicit ledger stamp is honoured on every platform, and the
        absence of one falls through to the portable probe rather than to a
        vendor guess."""
        return getattr(self, "requested_device", None)

    def unload(self):
        self._loaded = False
        self._presets_started = set()    # reset so the next episode re-guards first lines
        try:
            from .._otr_bark_lib import _unload_bark

            _unload_bark()
        except Exception:  # noqa: BLE001 -- teardown must never raise
            pass

    def prepare_text(self, text, delivery_vector=None):
        from .._otr_bark_lib import _clean_text_for_bark

        return _clean_text_for_bark(text)

    def _resolve_stage_temps(self):
        """(semantic, coarse, fine) temps from the char_bark_v1 /
        announcer_bark_v1 profile (role-dependent -- ``self.role`` is set by
        the dispatch core before ``generate_voice`` runs), with the
        BarkEngine class attrs as the fail-soft fallback.

        Alias ladder (precise -- a key is used only if PRESENT and NOT None, so an
        explicit ``0.0`` is honored, unlike ``or``):
          semantic <- semantic_temp, then text_temp, then class attr (0.7)
          coarse   <- coarse_temp,   then waveform_temp, then class attr (0.5)
          fine     <- fine_temp,     then waveform_temp, then class attr (0.5)
        A missing/malformed profile yields ``{}`` -> the class attrs stand in.
        """
        params: dict = {}
        try:
            from .._otr_engine_profiles import load_resolver

            resolver = load_resolver()
            if resolver is not None:
                prof = resolver.profile_for(
                    getattr(self, "role", "char_voice"), self.name)
                if prof is not None:
                    params = dict(prof.default_params or {})
        except Exception:  # noqa: BLE001 -- profile read must never break a render
            params = {}

        def pick(*keys, default):
            for k in keys:
                if k in params and params[k] is not None:
                    return float(params[k])
            return float(default)

        semantic = pick("semantic_temp", "text_temp", default=self.semantic_temp)
        coarse = pick("coarse_temp", "waveform_temp", default=self.coarse_temp)
        fine = pick("fine_temp", "waveform_temp", default=self.fine_temp)
        return semantic, coarse, fine

    def generate_voice(self, text, voice_preset, delivery_vector, seed):
        """One character line -> mono AUDIO {"waveform":[1,1,T], "sample_rate"}.

        voice_preset (e.g. "v2/en_speaker_3") arrives via voice_ref_field (the
        dispatch routes cast.voice_preset into the positional ref slot). Runs
        inside the caller's deterministic_inference wrap; Bark binds no external
        Generator. Preserves the [clears throat] first-line guard per preset so
        the per-line path keeps the same anti-hallucination behavior the grouped
        batch path had (guard fires on the first occurrence of each preset).
        """
        import numpy as np
        import torch

        from .._otr_bark_lib import (
            BARK_REROLLS_MAX,
            SPEECH_SHAPE_PASS,
            _generate_single_line,
            _load_bark,
            _resolve_bark_inject_anchor,
            _resolve_bark_speech_only,
            bark_reroll_seed,
            speech_shape_score,
        )
        from .registry import EngineUnusable, EngineUsabilityReason

        if not voice_preset or not str(voice_preset).startswith("v2/"):
            raise EngineUnusable(
                self.name, getattr(self, "role", "char_voice"),
                EngineUsabilityReason.MALFORMED_CONFIG,
                f"bark requires a v2/* voice_preset; got {voice_preset!r}",
            )
        model, processor = _load_bark("suno/bark", device=self._requested_device())
        self._loaded = True
        is_first = voice_preset not in self._presets_started
        self._presets_started.add(voice_preset)
        semantic_temp, coarse_temp, fine_temp = self._resolve_stage_temps()
        # B1 (2026-06-22): bark is the char_voice engine -> every line is
        # DIALOGUE, so render in speech-only mode (drop the squeal tokens) and
        # skip the first-line [clears throat] anchor by default. Both are
        # explicit kwargs resolved from env (OTR_BARK_SPEECH_ONLY=1 default,
        # OTR_BARK_DISABLE_THROAT_CLEAR=1 default) -- no implicit defaults.
        speech_only = _resolve_bark_speech_only()
        inject_first_line_anchor = _resolve_bark_inject_anchor()
        # THE OUTPUT GUARD (PBUG-20260902-03, 2026-09-12), ONE ORDERED CONTRACT
        # per take: (1) unusable audio -- empty, non-finite, or below the
        # sequencer's silence floor -- is not a candidate and is not scored;
        # (2) a usable take is scored for speech shape and returned the moment
        # it passes; (3) a usable take that does not pass is kept as a
        # candidate and the line is re-rolled on a domain-separated seed, at
        # most BARK_REROLLS_MAX times; (4) when every take is spent, the
        # best-scoring usable take ships (the ledger field is always filled --
        # a hole is never the answer); (5) only when NO take was usable does
        # the silent-output error below raise, exactly as it did before the
        # guard existed. Every re-roll logs at WARNING with the score. The
        # ladder is a pure function of the line seed, so a replay walks it to
        # the same winner.
        base_seed = seed
        best_wav = None
        best_score = -1.0
        best_seed = None
        best_sr = None
        last_unusable = None
        last_error = None
        for attempt in range(1 + BARK_REROLLS_MAX):
            seed = bark_reroll_seed(base_seed, attempt)
            try:
                audio_np, sr = _generate_single_line(
                    text, voice_preset, model, processor, is_first_line=is_first,
                    semantic_temp=semantic_temp, coarse_temp=coarse_temp,
                    fine_temp=fine_temp,
                    inject_first_line_anchor=inject_first_line_anchor,
                    speech_only=speech_only,
                    # B2: thread the EXISTING per-line seed (was dropped before)
                    # so the clip is reproducible (Bark.generate is unseeded
                    # otherwise). On a re-roll `seed` is the ladder's next rung.
                    seed=seed,
                )
            except Exception as exc:  # noqa: BLE001 -- a thrown attempt is an
                # UNUSABLE TAKE, NOT A LOST LINE (Sonnet QA, 2026-09-12). Before
                # the guard, one failed generation meant one failed line and
                # there was nothing banked to lose. Now a transient error on
                # attempt 2 must not discard a usable take from attempt 1 --
                # "the ledger field is always filled" has to survive it. The
                # error is re-raised below only if NOTHING was usable, so a
                # genuinely broken engine still fails loudly on the first line.
                last_error = exc
                log.warning(
                    "[OTR.bark] preset %s attempt %d/%d raised (%s: %s); "
                    "treating it as an unusable take",
                    voice_preset, attempt + 1, 1 + BARK_REROLLS_MAX,
                    type(exc).__name__, str(exc)[:200])
                continue
            wav = torch.from_numpy(
                np.asarray(audio_np, dtype=np.float32)
            ).reshape(1, 1, -1)
            # THE OUTPUT GATE (2026-08-28, C2 verdict after adversarial review).
            # Every downstream contract -- packing, sequencing, enhance,
            # mastering, mux, obs_publish -- checks shape, duration, rate and
            # hash, and NOT ONE checks that the audio is audible. 1e-4 is
            # ALIGNED WITH EXISTING SILENCE SEMANTICS: the exact threshold
            # scene_sequencer._trim_trailing_silence uses to call a sample
            # silent (~-80 dBFS). NO remap, NO generalisation to other
            # engines. A silent take is not a candidate for the guard below.
            peak = float(wav.abs().max().item()) if wav.numel() else 0.0
            finite = bool(torch.isfinite(wav).all()) if wav.numel() else False
            if wav.numel() == 0 or not finite or peak < 1e-4:
                last_unusable = (int(wav.numel()), peak, finite)
                log.warning(
                    "[OTR.bark] preset %s attempt %d/%d: unusable audio "
                    "(samples=%d, peak=%.2e, finite=%s); not a candidate",
                    voice_preset, attempt + 1, 1 + BARK_REROLLS_MAX,
                    int(wav.numel()), peak, finite)
                continue
            score = float(speech_shape_score(audio_np, sr))
            if score >= SPEECH_SHAPE_PASS:
                if attempt:
                    log.warning(
                        "[OTR.bark] preset %s: re-roll %d passed the speech-shape "
                        "guard (score %.2f, seed %d)",
                        voice_preset, attempt, score, seed)
                return {"waveform": wav, "sample_rate": int(sr)}
            if score > best_score:
                # `best_sr` rides with the take it belongs to; reading `sr`
                # from the last iteration would mislabel the shipped audio
                # the day anything reloads the model mid-ladder (Sonnet QA).
                best_wav, best_score, best_seed, best_sr = wav, score, seed, sr
            log.warning(
                "[OTR.bark] preset %s attempt %d/%d: take is not shaped like "
                "speech (score %.2f < %.2f, seed %d) -- %s",
                voice_preset, attempt + 1, 1 + BARK_REROLLS_MAX, score,
                SPEECH_SHAPE_PASS, seed,
                "re-rolling" if attempt < BARK_REROLLS_MAX else "keeping the best take")
        if best_wav is not None:
            log.warning(
                "[OTR.bark] preset %s: every take failed the speech-shape guard; "
                "shipping the best of %d (score %.2f, seed %d). The ledger field "
                "is filled; listen to this line.",
                voice_preset, 1 + BARK_REROLLS_MAX, best_score, best_seed)
            return {"waveform": best_wav, "sample_rate": int(best_sr)}
        if last_error is not None and last_unusable is None:
            # Every attempt threw and none produced audio at all: that is the
            # engine failing, not the guard rejecting, so the real error is
            # what the caller must see.
            raise last_error
        samples, peak, finite = last_unusable or (0, 0.0, False)
        raise BarkSilentOutputError(
            "bark returned unusable audio for preset %r (samples=%d, "
            "peak=%.2e, finite=%s): a silent-but-nonempty clip would "
            "pack, sequence and PUBLISH as a structurally valid line, "
            "so it is rejected at the engine instead. Re-run the line; "
            "never remap the preset." % (
                voice_preset, int(samples), peak, finite))
