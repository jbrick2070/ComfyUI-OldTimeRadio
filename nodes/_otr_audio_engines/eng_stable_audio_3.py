"""Stable Audio 3 music adapter -- ComfyUI-NATIVE (no stable_audio_tools).

Drives ComfyUI's own audio node classes (CheckpointLoaderSimple + optional
t5gemma CLIPLoader + CLIPTextEncode + ConditioningStableAudio + EmptyLatentAudio
+ KSampler + VAEDecodeAudio) so SA3 uses ComfyUI's `comfy.model_management` --
no PyPI dependency, no torch/numpy conflict on the Blackwell stack. Weights:
`Comfy-Org/stable-audio-3` (ungated). `interface == "clip"`.

Fail-closed: absent ComfyUI runtime or absent checkpoint raise a clear named
error (the 6-class taxonomy's MISSING_MODEL), never a silent fetch or crash.
SA3 community license = commercial-OK -> commercial_clean = True. PROMOTED
2026-06-03 to the DEFAULT music engine (default_roles=("music",), no flag);
prompt/conditioning/sampler defaults tuned 2026-06-14 (roundtable) for the show's
short instrumental sci-fi cues -- see _SA3_* constants + generate_clip().
"""
from __future__ import annotations

import hashlib
import logging

from .registry import EngineUnusable, EngineUsabilityReason, register

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore
try:
    from .._otr_music_prompt import NEGATIVE_PROMPT_DEFAULT
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_music_prompt import NEGATIVE_PROMPT_DEFAULT  # type: ignore

log = logging.getLogger("OTR")

# Default model files in ComfyUI's models/ tree (Comfy-Org/stable-audio-3).
#
#: THE BASE CHECKPOINT IS PREFERRED, AND THE REASON IS THE NEGATIVE PROMPT
#: (measured 2026-09-12, 65 renders). Stability ships each model in a BASE
#: and a POST-TRAINED form and documents that `cfg_scale` and
#: `negative_prompt` "have no effect on post-trained checkpoints". Every
#: anti-loop word this pack spent a campaign writing lives in the negative
#: prompt, so on a post-trained checkpoint that work is inert -- and worse,
#: ComfyUI still applies the guidance (it drops the uncond branch only at
#: cfg 1.0), extrapolating off a branch the post-training collapsed. The
#: result is the broadband burst the operator hears as "an odd tape loop
#: scratch": 74 bursts in 17 renders of the shipped combination against 2
#: in 48 renders of every other one.
#:
#: So: take the base file when it is on disk, and fall back to whatever is
#: there otherwise. An existing install downloads nothing and still gets a
#: clean cue, because the cfg below follows the checkpoint.
_CKPT_PREFERENCE = ("stable_audio_3_small_music_base.safetensors",
                    "stable_audio_3_small_music.safetensors")
#: A base checkpoint can be steered, so it gets real guidance: 4.0 measured
#: clean across a 1.0-6.0 sweep with the healthiest peak (-2.2 dBFS). A
#: post-trained one cannot, so it gets the value its own publisher
#: documents, which is also the only value ComfyUI turns the extrapolation
#: off at.
_CFG_FOR_BASE = 4.0
_CFG_FOR_POST_TRAINED = 1.0
_CKPT = otr_env.get("OTR_SA3_CKPT", "")
_TENC = otr_env.get("OTR_SA3_TEXT_ENCODER", "t5gemma_b_b_ul2.safetensors")
_CLIP_TYPE = otr_env.get("OTR_SA3_CLIP_TYPE", "stable_audio")

# BUG-408 (2026-06): SA3 is a different model than the old MusicGen default
# and wants genre + instrumentation + a production anchor, a real negative
# prompt, and a multi-second STRUCTURAL context (seconds_total) to sound like
# music rather than a 4-12 s texture. The structural window is still here.
# The prompt anchor is NOT (2026-09-11): every branch of it said "analog tape
# warmth" and the negative pushed "AWAY from a clean modern sound" -- the
# radio-hiss texture the operator has now withdrawn ("make them more
# musical"). The instruments come from the STORY through the shared composer
# (`_otr_music_prompt.compose_engine_prompt`) for every engine; this adapter
# sends what it is handed and reads the composer's negative unless the
# operator overrides it with OTR_SA3_NEG_PROMPT, the one escape hatch.



def _env_float(name, default):
    """Parse an env override as float; fall back to ``default`` (LOUD) on a bad
    value instead of crashing the render."""
    raw = otr_env.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning("[OTR.sa3] %s=%r is not a float -- using default %s",
                    name, raw, default)
        return float(default)


def _env_int(name, default):
    """Parse an env override as int; fall back to ``default`` (LOUD) on a bad value."""
    raw = otr_env.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        log.warning("[OTR.sa3] %s=%r is not an int -- using default %s",
                    name, raw, default)
        return int(default)


#: The conditioning window must be LONGER than the cue AT THE DEFAULT RATIO,
#: or the model is
#: asked for a complete self-contained piece that begins and ends inside its
#: own world -- which is a loop-shaped request, and is what BUG-408 set out
#: to remove. It stayed live for the longest cue: at the shipped 12 s context
#: the 12 s opening cue got seconds_total == dur exactly. Measured
#: 2026-09-12. The ratio is the floor, not the value; OTR_SA3_CONTEXT_S
#: still wins whenever it asks for more.
#: Env-tunable BECAUSE it must be measurable (Fable, 2026-09-12): the 3x
#: floor is the change most likely to have over-corrected the music into
#: formlessness -- the opening cue now renders the first third of a 36 s arc
#: and may never reach the theme it is told to settle into -- and a constant
#: cannot be A/B'd by `scripts/otr_music_ab.py`.
#:
#: A RATIO BELOW THE DEFAULT RESTORES THE DEFECT, not merely the old
#: behaviour (codex, 2026-09-12): at 1.0 the 12 s opening cue is asked for
#: as a complete self-contained piece again, which is the loop-shaped
#: request described above. That is exactly what makes it a useful control
#: arm and exactly what makes it wrong to ship, so it warns every time it
#: is used and the invariant below is claimed only at the default.
_SA3_MIN_CONTEXT_RATIO_DEFAULT = 3.0


#: SA3 NEVER READS `seconds_start`. VERIFIED IN THE RUNTIME, 2026-09-12:
#: `StableAudio3.extra_conds` (comfy/model_base.py) embeds `seconds_total` and
#: nothing else, and the checkpoint carries no seconds_start embedder at all --
#: only the older `StableAudio1` class has one (model_base.py:805-835). So the
#: placement half of `_sa3_clip_window` -- "an outro sits at the TAIL, an intro
#: at the HEAD" -- has never reached this model. It is computed, sent, logged
#: and written into the render receipt, describing something that did not
#: happen.
#:
#: It is still SENT, because ComfyUI's ConditioningStableAudio node takes the
#: argument and a future checkpoint may read it. What changes is that nothing
#: CLAIMS it worked: the log says it is ignored and the receipt records it as
#: ignored, so a reader chasing the music never again reasons from a field that
#: describes nothing. The lever that IS live is `seconds_total`, the other half
#: of the same function.
_SA3_READS_SECONDS_START = False


def _sa3_guidance(is_base: bool) -> float:
    """The guidance this checkpoint can actually answer.

    A BASE checkpoint is trained with classifier-free guidance and a negative
    prompt, so it gets real guidance. A POST-TRAINED one is documented by its
    publisher as ignoring both -- and ComfyUI only stops extrapolating off the
    unconditional branch at exactly 1.0, so anything else there is a 7x pull
    off a branch the post-training collapsed. That is the broadband burst.
    """
    return _env_float("OTR_SA3_CFG",
                      _CFG_FOR_BASE if is_base else _CFG_FOR_POST_TRAINED)


def _sa3_context_ratio() -> float:
    """The conditioning-window floor multiplier this render is using.

    ONE READER FOR TWO CALLERS (cursor r3, 2026-09-12). The window maths
    below needs it, and the render receipt has to RECORD it -- a receipt
    that carries only `seconds_total` cannot tell a 1.0 control arm from
    the shipped default, because a 36 s window satisfies "at least 1x a
    12 s cue" perfectly well. An A/B arm that cannot be told apart from
    the thing it controls for measures nothing.
    """
    return max(1.0, _env_float("OTR_SA3_CONTEXT_RATIO",
                               _SA3_MIN_CONTEXT_RATIO_DEFAULT))


def _sa3_clip_window(prompt: str, dur: float, context_s: float):
    """Place the ``dur``-second clip within a ``context_s`` structural window per
    cue so SA3 renders a real slice of a longer piece: an ``outro`` sits at the
    TAIL (resolving), an ``intro`` at the HEAD (build), anything else in the
    MIDDLE (unresolved bridge). Returns ``(seconds_start, seconds_total)``. The
    latent length stays ``dur`` -- only the CONDITIONING window changes, so clip
    length + seed determinism are unchanged."""
    dur = float(dur)
    ratio = _sa3_context_ratio()
    if ratio < _SA3_MIN_CONTEXT_RATIO_DEFAULT:
        log.warning("[OTR.sa3] OTR_SA3_CONTEXT_RATIO=%.2f is below the %.1fx "
                    "floor, so a cue may be asked for as a self-contained "
                    "piece -- the loop-shaped request the floor exists to "
                    "prevent. This is a control arm; never ship on it.",
                    ratio, _SA3_MIN_CONTEXT_RATIO_DEFAULT)
    floor = dur * ratio
    ctx = max(float(context_s), floor)
    if float(context_s) < floor:
        log.info("[OTR.sa3] context %.1fs is not longer than the %.1fs cue by "
                 "the %.1fx floor -- widening to %.1fs so the cue is a SLICE "
                 "of a longer piece rather than a self-contained loop",
                 float(context_s), dur, ratio, ctx)
    low = (prompt or "").lower()
    if "outro" in low or "closing" in low:
        start = max(0.0, ctx - dur)          # resolving tail
    elif "intro" in low or "opening" in low:
        start = 0.0                          # build / head
    else:
        start = max(0.0, (ctx - dur) / 2.0)  # unresolved middle
    return start, ctx


@register
class StableAudio3Engine:
    name = "stable_audio_3"
    roles = ("music",)
    default_roles = ("music",)  # PROMOTED 2026-06-03: shipped music default
    commercial_clean = True     # SA3 community license: commercial use OK
    requires_flag = None        # default engine -> always usable; weights checked in load()
    native = True               # drives ComfyUI's own nodes -> no external dep pilot
    interface = "clip"
    sample_rate = 44100

    def __init__(self):
        self._bundle = None     # (model, clip, vae)

    # -- ComfyUI handles for the native pipeline (lazy; only at execute time) --
    def _native(self):
        try:
            import nodes as comfy_nodes
            import comfy_extras.nodes_audio as audio_nodes
        except Exception as exc:  # noqa: BLE001 -- absent ComfyUI runtime
            raise EngineUnusable(
                self.name, "music", EngineUsabilityReason.MALFORMED_CONFIG,
                "ComfyUI runtime not importable -- stable_audio_3 drives native "
                "nodes and only runs inside ComfyUI",
            ) from exc
        return comfy_nodes, audio_nodes

    @staticmethod
    def resolve_ckpt():
        """The checkpoint this render will use, and whether it is a BASE one.

        An explicit ``OTR_SA3_CKPT`` always wins -- that is how the A/B
        harness moves the variable. Otherwise the base file is taken when it
        is on disk and the post-trained one when it is not, so an install
        that never downloads anything keeps rendering.
        """
        if _CKPT:
            return _CKPT, _CKPT.endswith("_base.safetensors")
        try:
            import folder_paths
            for name in _CKPT_PREFERENCE:
                if folder_paths.get_full_path("checkpoints", name):
                    return name, name.endswith("_base.safetensors")
        except Exception:  # noqa: BLE001 -- resolution never costs a render
            pass
        return _CKPT_PREFERENCE[-1], False

    def _ckpt_present(self):
        try:
            import folder_paths
            return folder_paths.get_full_path("checkpoints", self.resolve_ckpt()[0])
        except Exception:
            return None

    def load(self):
        if self._bundle is not None:
            return
        if self._ckpt_present() is None:
            raise EngineUnusable(
                self.name, "music", EngineUsabilityReason.MISSING_MODEL,
                "no SA3 checkpoint found in ComfyUI/models/checkpoints -- "
                "fetch Comfy-Org/stable-audio-3 (ungated) first. Either %s "
                "works; the first is preferred because its negative prompt "
                "is live." % " or ".join(_CKPT_PREFERENCE),
            )
        comfy_nodes, _ = self._native()
        name, is_base = self.resolve_ckpt()
        log.info("[OTR.sa3] checkpoint %s (%s; guidance %s)", name,
                 "base -- negative prompt is live" if is_base
                 else "post-trained -- negative prompt is inert, per Stability",
                 _CFG_FOR_BASE if is_base else _CFG_FOR_POST_TRAINED)
        model, clip, vae = comfy_nodes.CheckpointLoaderSimple().load_checkpoint(name)
        # SA3 ships t5gemma separately; if the checkpoint did not carry a usable
        # conditioner, load it explicitly. Defensive: prefer the bundled clip.
        if clip is None:
            clip = comfy_nodes.CLIPLoader().load_clip(_TENC, _CLIP_TYPE)[0]
        self._bundle = (model, clip, vae)

    def unload(self):
        import gc
        self._bundle = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def generate_clip(self, prompt, duration_s, seed, *,
                      placement="", negative_prompt=""):
        """Text prompt -> stereo AUDIO clip ``{"waveform", "sample_rate"}`` via
        ComfyUI's native SA3 graph. Determinism: KSampler takes the int seed and
        builds its generator internally (a bound torch.Generator cannot cross a
        node boundary), so the seed-int is the determinism carrier."""
        self.load()
        comfy_nodes, audio_nodes = self._native()
        model, clip, vae = self._bundle
        seed = int(seed)
        dur = float(duration_s)

        # BUG-408: SA3-shaped prompt + real negative + a structural seconds_total
        # context with a per-cue seconds_start. The LATENT stays exactly dur, so
        # clip length + seed determinism are unchanged; only the conditioning
        # window + prompt change. All knobs env-overridable for A/B tuning.
        # Defaults TUNED 2026-06-14 (roundtable). OTR_SA3_CONTEXT_S=12 was
        # chosen then so each short cue would be a coherent SLICE of a tight
        # phrase rather than an aimless fragment of a 30 s piece. It is now a
        # FLOOR-ADJUSTED request, not the final window: `_sa3_clip_window`
        # widens it to at least 3x the cue (2026-09-12), because at 12 s the
        # 12 s opening cue got seconds_total == dur -- a self-contained,
        # loop-shaped request, and the exact condition BUG-408 existed to
        # remove. Live windows today: opening 36 s, closing 24 s,
        # interstitial 12 s. cfg=7.0 -- NOT, as this comment long claimed, the "SA3 native
        # default": Comfy-Org ships its checkpoints in matched base/non-base
        # pairs and gives each its own recipe, verified against the two
        # templates in comfyui_workflow_templates 0.11.55 --
        #   audio_stable_audio_3_medium.json       steps=8  cfg=1 lcm simple
        #   audio_stable_audio_3_medium_base.json  steps=50 cfg=7 lcm simple
        # We load stable_audio_3_small_music, the NON-base (distilled) member,
        # so Comfy-Org's own default for it is cfg=1 / 8 steps. Ours is the
        # base recipe and then double the steps. MEASURED A/B on a Mac mini M4
        # (same checkpoint, seed and prompt, one 12 s cue): ours 22.3 s, theirs
        # 2.1 s -- 10.6x -- with both producing valid non-silent audio at an
        # identical 0.0003 dBFS peak (RMS -14.76 vs -16.17, noise floor -37.4
        # vs -32.8). The numbers do not separate them on quality, and 2 cues an
        # episode makes the saving ~40 s of a ~24 min run, so the values are
        # LEFT AS THEY ARE pending a listening test; changing them also forces
        # a CUDA golden re-baseline. Only the false provenance claim is fixed
        # for stronger prompt adherence; sampler/steps stay Stable Audio's
        # reference dpmpp_3m_sde_gpu @ 100 (determinism proven by the byte-
        # identical golden). All env-overridable; the operator never NEEDS to set them.
        context_s = _env_float("OTR_SA3_CONTEXT_S", 12.0)
        # The cue's placement names the window; the prompt-word fallback stays
        # for a caller that hands none over (the bug408 window test pins it).
        seconds_start, seconds_total = _sa3_clip_window(
            placement or prompt, dur, context_s)
        # The prompt arrives COMPOSED -- instruments, production anchor, row
        # text (`compose_engine_prompt`) -- and is sent as handed. The negative
        # is the composer's unless the operator overrides it.
        pos_text = str(prompt or "").strip()
        neg_text = (otr_env.get("OTR_SA3_NEG_PROMPT") or negative_prompt
                    or NEGATIVE_PROMPT_DEFAULT)
        steps = _env_int("OTR_SA3_STEPS", 100)
        # THE GUIDANCE FOLLOWS THE CHECKPOINT. A constant here is what put
        # a 7x extrapolation on a model that cannot answer one.
        ckpt_name, is_base = self.resolve_ckpt()
        cfg = _sa3_guidance(is_base)
        sampler = otr_env.get("OTR_SA3_SAMPLER", "dpmpp_3m_sde_gpu")
        scheduler = otr_env.get("OTR_SA3_SCHEDULER", "exponential")
        denoise = _env_float("OTR_SA3_DENOISE", 1.0)

        pos = comfy_nodes.CLIPTextEncode().encode(clip, pos_text)[0]
        neg = comfy_nodes.CLIPTextEncode().encode(clip, neg_text)[0]
        pos, neg = audio_nodes.ConditioningStableAudio().append(
            pos, neg, seconds_start, seconds_total)
        latent = audio_nodes.EmptyLatentAudio().generate(dur, 1)[0]
        sampled = comfy_nodes.KSampler().sample(
            model, seed, steps, cfg, sampler, scheduler,
            pos, neg, latent, denoise)[0]
        audio = audio_nodes.VAEDecodeAudio().decode(vae, sampled)[0]
        # traceability for A/B listens (no determinism impact)
        _phash = hashlib.blake2s((pos_text + "||" + neg_text).encode()).hexdigest()[:8]
        log.info("[OTR.sa3] cue_window start=%.1fs%s total=%.1fs dur=%.1fs seed=%d "
                 "steps=%d cfg=%.1f sampler=%s/%s prompt_hash=%s",
                 seconds_start, "" if _SA3_READS_SECONDS_START else " (IGNORED by SA3)",
                 seconds_total, dur, seed, steps, cfg,
                 sampler, scheduler, _phash)
        # native AUDIO dict already carries {"waveform","sample_rate"}.
        sr = int(audio.get("sample_rate", self.sample_rate))
        # The receipt rides with the clip so the ledger can say what this
        # cue actually heard and did (prompt hash, window, sampler).
        return {"waveform": audio["waveform"], "sample_rate": sr,
                "receipt": {"engine": self.name, "steps": int(steps),
                            "cfg": float(cfg), "sampler": str(sampler),
                            "scheduler": str(scheduler),
                            # The text the ENGINE used, which is not always
                            # the composer's: OTR_SA3_NEG_PROMPT overrides it
                            # above, and a receipt that recorded the spec's
                            # negative would describe a cue nobody heard.
                            "negative_prompt": str(neg_text),
                            "denoise": float(denoise),
                            "context_ratio": _sa3_context_ratio(),
                            # WHICH MODEL MADE THIS CUE. The family ships a
                            # small variant its own publisher describes as
                            # "on-device-friendly loops" and a medium one
                            # built for "stronger structure and musicality";
                            # a receipt that does not say which one played
                            # cannot answer the operator's question about
                            # the music, and the A/B harness refuses to
                            # measure an arm it cannot attribute.
                            "ckpt": str(ckpt_name),
                            "seconds_start": float(seconds_start),
                            # Recorded so the receipt cannot be read as proof
                            # that the placement reached the model. It did not.
                            "seconds_start_read_by_model": _SA3_READS_SECONDS_START,
                            "seconds_total": float(seconds_total),
                            "duration_s": float(dur), "seed": int(seed),
                            "prompt_hash": _phash}}
