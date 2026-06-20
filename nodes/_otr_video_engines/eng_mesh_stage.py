"""mesh_stage -- the traditional local 3D chain (0-E tickets E-1..E-6).

The first REAL 3D OBJECT in the workflow: portrait still -> Hunyuan3D-2mv
NATIVE ComfyUI-core mesh generation (in-process, compile-free -- the core
nodes are verified PRESENT in the running install) -> GLB cached per
character (canonical-portrait content-hash key + manifest JSON sidecar,
E-2) -> pinned PORTABLE Blender headless stage (WORKBENCH matcap v1,
turntable-orbit camera preset, exact ledger frame count, E-3) -> straight-
alpha PNG frame DIRECTORY, atomically published (E-4), composited by
OTR_SilentComposite's directory-clip read path (Track-3, 80ce175).

LICENSE (E-7 gate): Hunyuan3D-2mv ships under the **Tencent Hunyuan
community license** (threshold + territory clauses) -- ``commercial_clean``
stays False and the engine stays SELECTABLE-NOT-DEFAULT until the operator-
visible license record lands (docs/2026-06-11-comfy-native-3d-options/).
TripoSR (MIT) is the recorded license-hedge mesher; swapping meshers only
changes the E-1 slot (the cache key carries mesher id/version).

GPU DISCIPLINE (E-1): the mesher runs in-process via wrapper_bridge with
``free_after_use``; after mesh-gen (or a cache hit skips it entirely) the
adapter runs the BUG-291 reclaim + ``torch.cuda.empty_cache()`` BARRIER
**before** the Blender spawn -- the torch mesher and Blender's GPU never
run concurrently (the GPU-lease seam).

ANIMATION TIER-1 (0-E): camera/object motion ONLY -- rigged/talking 3D
remains the parked toolkit lane (triposg_talk et al.); this engine NEVER
claims lip-sync (no audio_ref input; family ``image_to_video``, NOT
``character_3d`` -- the family token is verified against schemas.FAMILIES
at build, never overloading triposg_talk, E-5).

The hy3d graph TOPOLOGY below is the ASSUMED native chain and is
VERIFY-ON-GPU (probe-first discipline, the LTX probe_f3 pattern): the
operator confirms node candidates + widget names against the installed
core's INPUT_TYPES before enabling. EEVEE is BANNED v1 headless; Cycles is
the v1.5 tier (stage script carries the determinism pins).

Cold-import clean (V-12): module scope imports only stdlib + the dep-free
registry / cheap-family base / directory-clip helpers. torch / comfy / PIL
are imported LAZILY inside load / render_clip.

Config (env): ``OTR_ENABLE_MESH_STAGE`` opt-in flag; ``OTR_BLENDER_EXE``
pinned portable Blender (fail-closed LOUD, no dev-path hardcode);
``OTR_HY3D_CKPT`` / ``OTR_HY3D_CKPT_NAME`` model overrides (the all-in-one
checkpoint EMBEDS the DINO image encoder ``conditioner.main_image_encoder.*``
and the ShapeVAE ``vae.*`` -- no separate CLIP-vision file, A4 audit
2026-06-11); ``OTR_MESH_CACHE_DIR`` the GLB cache root (MUST live under the
ComfyUI output dir -- core SaveGLB refuses prefixes outside it; the default
``<comfy>/output/otr/mesh_cache`` complies); ``OTR_MESH_STAGE_*``
stage knobs (frames cap, timeout, selftest skip).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time

from .cheap_families import _CheapFamilyBase
from .directory_clip import validate_directory_clip
from .registry import EngineUnusable, EngineUsabilityReason, register

_LOG = logging.getLogger("OTR.video.eng_mesh_stage")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

#: The Blender-side stage script (tracked, ASCII; runs in Blender's python).
STAGE_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "otr_mesh_stage_blender.py")

#: Mesher identity baked into the E-2 cache key (mesher swappable: TripoSR
#: hedge would register its own id/version; old cache rows never collide).
MESHER_ID = "hy3d2mv"
MESHER_VERSION = "1"

#: E-6 canvas contract: mesh_stage sets the landscape canvas EXPLICITLY when
#: the request leaves it unset (the cheap-family default is 832x480 -- the
#: proven look-QA landscape canvas is 1472x832).
DEFAULT_W = 1472
DEFAULT_H = 832

_LICENSE_NOTE = ("Tencent Hunyuan community license (thresholds + territory "
                 "clauses) -- operator license record required BEFORE any "
                 "default-on (0-E E-7)")


# --------------------------------------------------------------------------- #
# E-2 -- mesh cache key + manifest sidecar (pure; CPU-tested)
# --------------------------------------------------------------------------- #
def portrait_content_hash(path):
    """sha256 hex of the portrait FILE BYTES (the canonical-reference-portrait
    content key, E-2). Content, never path/mtime -- moving the file never
    regenerates the mesh; editing it always does."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mesh_cache_key(character_id, portrait_sha256, mesher_id=MESHER_ID,
                   mesher_version=MESHER_VERSION):
    """The E-2 cache key: character_id + CANONICAL portrait content-hash +
    mesher id/version. Deliberately NOT the per-episode portrait hash (the
    regen trap): episode casts reuse within the episode via their canonical
    reference portrait; a stable cast (announcer) reuses across episodes.
    Filesystem-safe, deterministic, pure."""
    char = re.sub(r"[^A-Za-z0-9_.-]", "_", str(character_id) or "uncast")[:48]
    sha = str(portrait_sha256 or "")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError(
            "mesh_cache_key: portrait_sha256 must be a sha256 hex digest, "
            "got %r" % (portrait_sha256,))
    return "%s__%s__%s_v%s" % (char, sha[:16], mesher_id, mesher_version)


def build_mesh_manifest(character_id, portrait_sha256, seed,
                        mesher_id=MESHER_ID, mesher_version=MESHER_VERSION):
    """The manifest JSON sidecar (E-2): provenance for later ARKit-lane
    bridging. Pure dict build; written next to the cached GLB."""
    return {
        "schema_version": 1,
        "character_id": str(character_id) or "uncast",
        "source_portrait_sha256": str(portrait_sha256),
        "mesher_id": str(mesher_id),
        "mesher_version": str(mesher_version),
        "seed": int(seed or 0),
        "license": _LICENSE_NOTE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_manifest(manifest, path):
    """Atomic manifest write (tmp + replace; UTF-8, no BOM)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# E-3 -- Blender stage kit (command + sanitized env; pure builders)
# --------------------------------------------------------------------------- #
def build_blender_cmd(blender_exe, glb_path, out_dir, frames, width, height,
                      seed, *, mode="render", engine="WORKBENCH",
                      stage_script=None, portrait="", start_angle=None,
                      arc_degrees=None):
    """The headless Blender invocation (E-3): ``--background
    --factory-startup --python-exit-code 1 --python stage.py -- <args>``.
    ``--python-exit-code 1`` makes ANY unhandled stage exception a nonzero
    Blender exit (fail-closed: a partial render never publishes). Pure.

    ``portrait`` (optional): the resolved hero still projected onto the
    front-facing vertices (the textured-hero PoC). Appended only when set, so
    the legacy flat-matcap invocation is byte-identical. ``start_angle`` /
    ``arc_degrees`` (optional): the bounded hero arc, appended only when set."""
    cmd = [
        str(blender_exe), "--background", "--factory-startup",
        "--python-exit-code", "1",
        "--python", str(stage_script or STAGE_SCRIPT), "--",
        "--mode", str(mode),
        "--glb", str(glb_path or ""),
        "--out", str(out_dir),
        "--frames", str(int(frames)),
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--seed", str(int(seed or 0)),
        "--render-engine", str(engine),
    ]
    if portrait:
        cmd += ["--portrait", str(portrait)]
    if start_angle is not None:
        cmd += ["--start-angle", str(float(start_angle))]
    if arc_degrees is not None:
        cmd += ["--arc-degrees", str(float(arc_degrees))]
    return cmd


#: Env keys that must never leak from the ComfyUI venv into Blender's bundled
#: python (the sanitized-env pattern, E-4 / _b_harness lineage).
BLENDER_ENV_LEAK_KEYS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                         "PYTHONEXECUTABLE", "VIRTUAL_ENV")


def build_blender_env(base_env):
    """Sanitized env for the Blender spawn: strip the venv's python leaks +
    force PYTHONNOUSERSITE (Blender ships its OWN python; a leaked
    PYTHONPATH into it is the classic silent-import-poison). Pure."""
    env = dict(base_env)
    for k in BLENDER_ENV_LEAK_KEYS:
        env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


# --------------------------------------------------------------------------- #
# E-4 -- frame-dir validation + ATOMIC publish (+ stale-tmp cleanup)
# --------------------------------------------------------------------------- #
def _long_path(p):
    """Defensive Windows long-path prefix (\\\\?\\) for rename/exists calls."""
    p = os.path.abspath(p)
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + p
    return p


def validate_frame_dir(path, expect_frames, expect_w, expect_h):
    """E-4/E-6 gate before publish: exactly N sorted nonzero frames (the ONE
    directory-clip rule) AND first/last frame dims == the explicit canvas.
    Raises ValueError naming the violation; returns the frame list."""
    from .directory_clip import list_directory_frames
    frames = list_directory_frames(path)
    if len(frames) != int(expect_frames):
        raise ValueError(
            "mesh_stage: rendered %d frames, ledger target is %d in %r (the "
            "delivered clip keeps EXACTLY the audio-derived frame count)"
            % (len(frames), int(expect_frames), path))
    from PIL import Image
    for probe in (frames[0], frames[-1]):
        with Image.open(probe) as im:
            if im.size != (int(expect_w), int(expect_h)):
                raise ValueError(
                    "mesh_stage: frame %r is %r, the explicit canvas contract "
                    "is (%d, %d)" % (probe, im.size, expect_w, expect_h))
            if probe.lower().endswith(".png") and im.mode != "RGBA":
                raise ValueError(
                    "mesh_stage: frame %r mode %r, straight-alpha RGBA "
                    "required (the ONE alpha handoff)" % (probe, im.mode))
    return frames


def publish_frame_dir(tmp_dir, final_dir):
    """ATOMIC publish: rename the validated tmp render dir to its final name.
    The final path must NOT exist (short-ID dirs are single-writer); a
    half-rendered dir can never be observed at the final path. Returns
    final_dir; raises on collision instead of merging."""
    if os.path.exists(final_dir):
        raise FileExistsError(
            "mesh_stage publish: final frame dir already exists: %r (short-ID "
            "dirs are single-writer; refusing to merge)" % (final_dir,))
    parent = os.path.dirname(os.path.abspath(final_dir))
    os.makedirs(parent, exist_ok=True)
    try:
        os.rename(tmp_dir, final_dir)
    except OSError:
        # Long-path retry (Windows MAX_PATH defense, E-4).
        os.rename(_long_path(tmp_dir), _long_path(final_dir))
    return final_dir


def cleanup_stale_tmp(parent, prefix="otr_mesh_tmp_", max_age_hours=24.0):
    """Sweep abandoned tmp render dirs older than ``max_age_hours`` (E-4
    stale-tmp cleanup; a crashed render never leaks disk forever). Returns
    the removed paths; never raises on a racing delete."""
    removed = []
    if not parent or not os.path.isdir(parent):
        return removed
    cutoff = time.time() - float(max_age_hours) * 3600.0
    for name in os.listdir(parent):
        if not name.startswith(prefix):
            continue
        p = os.path.join(parent, name)
        try:
            if os.path.isdir(p) and os.path.getmtime(p) < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p)
        except OSError:  # pragma: no cover - racing delete is fine
            continue
    if removed:
        _LOG.warning("[eng_mesh_stage] swept %d stale tmp render dir(s) "
                     "under %r", len(removed), parent)
    return removed


# --------------------------------------------------------------------------- #
# The engine (E-1 mesher + E-5 registration)
# --------------------------------------------------------------------------- #
@register
class MeshStageEngine(_CheapFamilyBase):
    """``mesh_stage`` -- portrait -> hy3d-2mv GLB -> Blender orbit stage."""

    name = "mesh_stage"
    honest_label = ("real 3D object on a turntable stage (hy3d-2mv mesh + "
                    "Blender matcap orbit; camera motion only, no rig, no "
                    "lip-sync)")
    family = "image_to_video"           # verified against schemas.FAMILIES in
    #                                     tests; NEVER character_3d (that family
    #                                     requires audio_ref = the talking lane)
    roles = ("music_visual", "announcer_visual", "character_video")
    #: NEVER a default until the operator's look-QA + the E-7 license record.
    default_roles = ()
    requires_flag = "OTR_ENABLE_MESH_STAGE"
    required_inputs = ("init_image",)
    commercial_clean = False            # Tencent community license (E-7 gate)
    engine_version = "1"
    uses_still = True
    #: E-5 LOUD fallback chain: mesh_stage -> still_parallax -> still_kenburns
    #: (each hop restamps the ledger; the chain composes via fallback_engine).
    fallback_engine = "still_parallax"

    _selftest_passed = False            # E-6: cube probe gates the first use

    def __init__(self):
        self._classes = None

    # ---- config resolution (env override -> box default) ----
    @staticmethod
    def _blender_exe():
        return os.environ.get("OTR_BLENDER_EXE", "")

    def _ckpt_path(self):
        explicit = os.environ.get("OTR_HY3D_CKPT")
        if explicit:
            return explicit
        candidate_dirs = [os.path.join(_COMFY_ROOT, "models", "checkpoints")]
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            candidate_dirs.append(
                os.path.join(os.path.dirname(hf_home), "checkpoints"))
        names = ("hunyuan3d-dit-v2-mv.safetensors",
                 "hunyuan3d-dit-v2-mv-fp16.safetensors",
                 "hunyuan3d-dit-v2-0.safetensors")
        for d in candidate_dirs:
            for name in names:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    return p
        fallback_dir = (os.path.join(os.path.dirname(hf_home), "checkpoints")
                        if hf_home else candidate_dirs[0])
        return os.path.join(fallback_dir, names[0])

    def _ckpt_name(self):
        return os.environ.get("OTR_HY3D_CKPT_NAME") or os.path.basename(
            self._ckpt_path())

    def _installed(self):
        """True iff the hy3d checkpoint exists on disk (cheap, no import).
        The full node-class check is load()/the GPU probe."""
        return os.path.exists(self._ckpt_path())

    def _cache_root(self):
        if os.environ.get("OTR_MESH_CACHE_DIR"):
            return os.environ["OTR_MESH_CACHE_DIR"]
        # The mesh cache MUST live under ComfyUI's CONFIGURED output dir -- core
        # SaveGLB resolves filename_prefix through folder_paths.get_save_image_path
        # and REFUSES any path outside the active output folder. The headless
        # launcher pins --output-directory to the Documents tree, so the old
        # hardcoded <comfy_install>/output was a DIFFERENT tree -> SaveGLB raised
        # "Saving image outside the output folder is not allowed" -> mesh_stage
        # fell back to still_parallax (2026-06-12 catch). Honor the configured
        # output (folder_paths), then OTR_OUTPUT_DIR, then the install default.
        out = ""
        try:
            import folder_paths  # type: ignore  # ComfyUI runtime
            out = folder_paths.get_output_directory()
        except Exception:  # noqa: BLE001 -- off-box / pre-Comfy import
            out = ""
        if not out:
            out = os.environ.get("OTR_OUTPUT_DIR") or os.path.join(
                _COMFY_ROOT, "output")
        # MUST be under the configured output (SaveGLB) AND under the otr tree's
        # RESERVED system tier (the OUTPUT-TREE CONTRACT: otr/ top level is
        # EXACTLY episodes + obs; everything else lives under
        # episodes/_shared). otr/mesh_cache was a stray top-level dir -> hygiene
        # gate fail (2026-06-12 catch, after the mesh actually rendered). Nest
        # it under episodes/_shared so SaveGLB + the output-tree contract both
        # pass; _shared (not _shared/tmp) so the janitor does not sweep the cache.
        return os.path.join(out, "otr", "episodes", "_shared", "mesh_cache")

    # ---- usability ladder (fail-closed; ORDER MATTERS; no heavy import) ----
    # flag -> Blender exe -> hy3d checkpoint. NEVER the cache/mesh dir (the
    # 3D-plan 7.1 deadlock lesson: meshes are GENERATED at render time).
    def assert_usable(self, host_caps, profile, request_template=None):
        if os.getenv(self.requires_flag, "0") != "1":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "%s is opt-in (0-E easy on-ramp; selectable-not-default until "
                "operator look-QA + the Tencent license record E-7); set %s=1 "
                "to enable it" % (self.name, self.requires_flag), kind="video")
        blender = self._blender_exe()
        if not blender or not os.path.exists(blender):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s: pinned portable Blender missing (set OTR_BLENDER_EXE to "
                "the portable blender.exe under the env-pointed tools dir; "
                "got %r -- no dev-path hardcode, fail-closed LOUD)"
                % (self.name, blender), kind="video")
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s not installed: hy3d all-in-one checkpoint=%s "
                "(OTR_HY3D_CKPT; embeds the DINO image encoder + ShapeVAE) "
                "-- install and verify the core hy3d nodes on the GPU box "
                "(probe-first)" % (self.name, self._ckpt_path()),
                kind="video")
        return self.name

    # ---- E-1: the native hy3d graph (A4-audited vs the install) ----
    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node. A4 audit
        2026-06-11 verified the topology against the INSTALLED core source
        AND the shipped official blueprint ('Image to Model (Hunyuan3d
        2.1)' -- same core nodes as 2.0-mv): the loader is
        **ImageOnlyCheckpointLoader** (returns MODEL / CLIP_VISION / VAE;
        the all-in-one hy3d checkpoint embeds the DINO image encoder and
        ShapeVAE, so there is NO separate CLIPVisionLoader file), and the
        MODEL is patched by **ModelSamplingAuraFlow** before the KSampler
        (blueprint shift=1). Runtime behavior remains VERIFY-ON-GPU."""
        return {
            "checkpoint": ("ImageOnlyCheckpointLoader",),
            "cv_encode": ("CLIPVisionEncode",),
            "loadimage": ("LoadImage",),
            "modelsampling": ("ModelSamplingAuraFlow",),
            "cond": ("Hunyuan3Dv2Conditioning",),
            "latent": ("EmptyLatentHunyuan3Dv2",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecodeHunyuan3D",),
            "voxeltomesh": ("VoxelToMesh",),
            "saveglb": ("SaveGLB",),
        }

    def _build_mesh_graph(self, image_name, seed, glb_prefix):
        """The declarative portrait->GLB graph (wrapper_bridge.run_graph
        format). Topology + widget sets VERIFIED against the INSTALLED core
        source + official blueprint 2026-06-11 (comfy_extras/
        nodes_hunyuan3d.py + nodes_save_3d.py + nodes_video_model.py +
        nodes_model_advanced.py + nodes.py CLIPVisionEncode); runtime
        behavior still VERIFY-ON-GPU:
        ImageOnlyCheckpointLoader(MODEL/CLIP_VISION/VAE) ->
        ModelSamplingAuraFlow(shift=1) + CLIPVisionEncode(embedded encoder)
        -> Hunyuan3Dv2Conditioning -> KSampler(EmptyLatentHunyuan3Dv2) ->
        VAEDecodeHunyuan3D -> VoxelToMesh -> SaveGLB.

        The hy3d nodes are V3 ``IO.ComfyNode`` classes: ``EXECUTE_NORMALIZED``
        calls ``execute(**inputs)`` with NO schema-default backfill, so every
        required widget must be passed EXPLICITLY (``crop``, ``num_chunks``,
        ``octree_resolution``, ``algorithm`` were the 2026-06-11 A4 audit
        gaps). SaveGLB resolves ``filename_prefix`` through
        ``folder_paths.get_save_image_path`` -> ``<prefix>_<counter:05>_.glb``
        and REFUSES prefixes outside the ComfyUI output dir -- the cache root
        (OTR_MESH_CACHE_DIR) must stay under it (the default does)."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        steps = int(os.environ.get("OTR_HY3D_STEPS", "30"))
        cfg = float(os.environ.get("OTR_HY3D_CFG", "5.0"))
        return {
            "checkpoint": {"class": "checkpoint",
                           "inputs": {"ckpt_name": self._ckpt_name()}},
            "loadimage": {"class": "loadimage",
                          "inputs": {"image": image_name}},
            "cv_encode": {"class": "cv_encode",
                          # slot 1 = the checkpoint's EMBEDDED DINO encoder
                          "inputs": {"clip_vision": W("checkpoint", 1),
                                     "image": W("loadimage", 0),
                                     # required combo (installed nodes.py)
                                     "crop": "center"}},
            "modelsampling": {"class": "modelsampling",
                              # blueprint pins shift=1 (node default 1.73)
                              "inputs": {"model": W("checkpoint", 0),
                                         "shift": 1.0}},
            "cond": {"class": "cond",
                     "inputs": {"clip_vision_output": W("cv_encode", 0)}},
            "latent": {"class": "latent", "inputs": {"resolution": 3072,
                                                     "batch_size": 1}},
            "ksampler": {"class": "ksampler",
                         "inputs": {"seed": int(seed or 0), "steps": steps,
                                    "cfg": cfg, "sampler_name": "euler",
                                    "scheduler": "normal", "denoise": 1.0,
                                    "model": W("modelsampling", 0),
                                    "positive": W("cond", 0),
                                    "negative": W("cond", 1),
                                    "latent_image": W("latent", 0)}},
            "vaedecode": {"class": "vaedecode",
                          "inputs": {"samples": W("ksampler", 0),
                                     "vae": W("checkpoint", 2),
                                     # V3 schema defaults, passed EXPLICITLY
                                     # (EXECUTE_NORMALIZED backfills nothing)
                                     "num_chunks": 8000,
                                     "octree_resolution": 256}},
            "voxeltomesh": {"class": "voxeltomesh",
                            "inputs": {"voxel": W("vaedecode", 0),
                                       # required combo; "surface net" is the
                                       # schema's first/default option
                                       "algorithm": "surface net",
                                       "threshold": 0.6}},
            "saveglb": {"class": "saveglb",
                        "inputs": {"mesh": W("voxeltomesh", 0),
                                   "filename_prefix": glb_prefix}},
        }

    def load(self):
        """Fail CLOSED until installed, then RESOLVE the core hy3d node
        classes (fail-closed NAMED if absent). Weights load when the loader
        nodes execute inside the mesher graph (V-5: adapter-internal, lazy)."""
        if not self._installed():
            raise RuntimeError(
                "mesh_stage not installed: hy3d checkpoint missing at %s -- "
                "install it, set OTR_ENABLE_MESH_STAGE=1, and run the GPU "
                "probe" % self._ckpt_path())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())

    def unload(self):
        self._classes = None

    # ---- the GPU-lease BARRIER (E-1): torch mesher NEVER co-runs Blender ----
    @staticmethod
    def _vram_barrier():
        """BUG-291 reclaim + empty_cache BEFORE any Blender spawn. A no-op
        when torch never loaded (cache-hit path on a CPU box)."""
        import sys as _sys
        if "torch" not in _sys.modules:
            return
        try:
            from . import wrapper_bridge as _wb
            _wb.reclaim_idle_models("mesh_stage pre-Blender barrier")
        except Exception as exc:  # noqa: BLE001 -- reclaim is best-effort
            _LOG.warning("[eng_mesh_stage] reclaim_idle_models skipped: %s",
                         exc)
        torch = _sys.modules["torch"]
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            _LOG.info("[eng_mesh_stage] VRAM barrier: empty_cache before "
                      "Blender spawn (torch mesher and Blender-GPU never "
                      "run concurrently)")

    # ---- E-2: cache lookup / fill ----
    def _request_character_id(self, request):
        get = self._get(request)
        cond = get("conditioning_refs") or {}
        c_get = cond.get if isinstance(cond, dict) else (
            lambda k, d=None: getattr(cond, k, d))
        return str(c_get("char_id", "") or get("char_id") or "uncast")

    def _cached_glb(self, key):
        root = self._cache_root()
        glb = os.path.join(root, key + ".glb")
        manifest = os.path.join(root, key + ".manifest.json")
        return glb, manifest

    def _generate_mesh(self, still_path, seed, glb_path, manifest):
        """E-1: run the in-process hy3d graph -> the cached GLB + manifest.
        free_after_use drops every intermediate once consumed; the SaveGLB
        write lands under the cache root (atomic: tmp prefix + rename)."""
        from . import wrapper_bridge as _wb
        if self._classes is None:
            self.load()
        image_name = _wb.stage_into_comfy_input(still_path)
        os.makedirs(os.path.dirname(glb_path), exist_ok=True)
        prefix = os.path.splitext(glb_path)[0] + ".tmp"
        graph = self._build_mesh_graph(image_name, seed, prefix)
        _wb.run_graph(graph, self._classes, free_after_use=True,
                      keep={"saveglb"})
        produced = self._resolve_saveglb_output(prefix)
        os.replace(produced, glb_path)
        write_manifest(manifest, glb_path[:-4] + ".manifest.json")
        return glb_path

    @staticmethod
    def _resolve_saveglb_output(prefix):
        """SaveGLB writes ``<prefix>_<counter>.glb`` (core save-node pattern;
        VERIFY-ON-GPU). Resolve the newest match LOUDLY."""
        import glob as _glob
        matches = sorted(_glob.glob(prefix + "*.glb"), key=os.path.getmtime)
        if not matches:
            raise RuntimeError(
                "mesh_stage: SaveGLB produced no GLB at prefix %r -- confirm "
                "the installed core SaveGLB filename pattern (GPU probe)"
                % (prefix,))
        return matches[-1]

    # ---- E-6: the cube self-test probe (gates the first real use) ----
    def _run_selftest(self, blender):
        """3-frame cube probe through the REAL stage script + validators
        (mode=selftest builds its own cube GLB inside Blender, exports +
        re-imports it, renders 3 RGBA frames). Once per process; skippable
        via OTR_MESH_STAGE_SKIP_SELFTEST=1 (the operator's call)."""
        if MeshStageEngine._selftest_passed:
            return
        if os.environ.get("OTR_MESH_STAGE_SKIP_SELFTEST") == "1":
            _LOG.warning("[eng_mesh_stage] selftest SKIPPED by env (operator)")
            MeshStageEngine._selftest_passed = True
            return
        tmp = tempfile.mkdtemp(prefix="otr_mesh_selftest_")
        cmd = build_blender_cmd(blender, "", tmp, 3, 256, 256, 0,
                                mode="selftest")
        self._spawn_blender(cmd)
        validate_frame_dir(tmp, 3, 256, 256)
        shutil.rmtree(tmp, ignore_errors=True)
        MeshStageEngine._selftest_passed = True
        _LOG.info("[eng_mesh_stage] cube self-test probe PASSED (3 frames)")

    @staticmethod
    def _spawn_blender(cmd):
        """Run headless Blender with the sanitized env + timeout; fail-closed
        NAMED on a nonzero exit (the stage script exits nonzero on any
        exception -- a partial render never publishes)."""
        timeout = int(os.environ.get("OTR_MESH_STAGE_TIMEOUT_S", "1800"))
        env = build_blender_env(os.environ)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                "mesh_stage: Blender stage failed (exit %d):\n%s"
                % (proc.returncode, (proc.stderr or proc.stdout or "")[-2000:]))

    # ---- render (the full E-1..E-4 chain) ----
    def render_clip(self, request, prepared=None):
        """portrait -> (cache | hy3d mesh-gen) -> VRAM BARRIER -> Blender
        turntable stage -> validate -> ATOMIC publish -> directory clip."""
        get = self._get(request)
        w, h, fps = self._canvas_dims(request)
        if w == 832 and h == 480:
            canvas = get("canvas") or {}
            c_get = canvas.get if isinstance(canvas, dict) else (
                lambda k, d=None: getattr(canvas, k, d))
            if not int(c_get("w", 0) or 0):
                # E-6: the canvas contract is EXPLICIT -- never inherit the
                # cheap-family 832x480 default silently.
                w, h = DEFAULT_W, DEFAULT_H
        n = self._frame_count(request, fps)
        still = self._still_path(request)
        if not still or not os.path.exists(still):
            raise FileNotFoundError(
                "mesh_stage requires an on-disk portrait/still (asset_refs "
                "still/init_image/image); got %r" % (still,))
        blender = self._blender_exe()
        if not blender or not os.path.exists(blender):
            raise RuntimeError(
                "mesh_stage: OTR_BLENDER_EXE missing at render time (%r) -- "
                "assert_usable should have failed closed" % (blender,))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        seed = int(s_get("request_seed", 0) or 0)
        # E-2: cache by character + CANONICAL portrait content hash.
        char_id = self._request_character_id(request)
        key = mesh_cache_key(char_id, portrait_content_hash(still))
        glb_path, manifest_path = self._cached_glb(key)
        if os.path.exists(glb_path):
            _LOG.info("[eng_mesh_stage] mesh cache HIT %s (no mesher run)",
                      key)
        else:
            _LOG.info("[eng_mesh_stage] mesh cache MISS %s -- running the "
                      "hy3d-2mv mesher", key)
            manifest = build_mesh_manifest(char_id,
                                           portrait_content_hash(still), seed)
            self._generate_mesh(still, seed, glb_path, manifest)
        # E-1: the lease seam -- torch is FULLY reclaimed before Blender.
        self._vram_barrier()
        # E-6: the cube probe gates the first real Blender use.
        self._run_selftest(blender)
        # E-3/E-4: render into a tmp dir, validate, ATOMIC publish.
        parent = os.path.join(self._cache_root(), "stages")
        os.makedirs(parent, exist_ok=True)
        cleanup_stale_tmp(parent)
        tmp_dir = tempfile.mkdtemp(prefix="otr_mesh_tmp_", dir=parent)
        # Textured-hero PoC: project the RESOLVED portrait/still onto the front
        # vertices (the GLB stays geometry-only). The bounded hero arc keeps the
        # unpainted back off-camera; knobs are env-tunable, defaults in-script.
        sa = os.environ.get("OTR_MESH_STAGE_START_ANGLE")
        ad = os.environ.get("OTR_MESH_STAGE_ARC_DEGREES")
        cmd = build_blender_cmd(
            blender, glb_path, tmp_dir, n, w, h, seed, portrait=still,
            start_angle=(float(sa) if sa else None),
            arc_degrees=(float(ad) if ad else None))
        self._spawn_blender(cmd)
        validate_frame_dir(tmp_dir, n, w, h)
        final_dir = os.path.join(
            parent, "stage_%s_%s" % (key[:32], os.path.basename(tmp_dir)[-8:]))
        publish_frame_dir(tmp_dir, final_dir)
        return self._directory_clip(request, final_dir, fps, n)

    def _directory_clip(self, request, frame_dir, fps, n):
        """Pure: the directory CanonicalClip dict (Track-3 contract: rgba /
        straight / silent; frame_count is the integer timing authority)."""
        get = self._get(request)
        return {
            "clip_id": get("shot_id") or get("request_id") or "mesh_stage_clip",
            "type": "directory", "path": frame_dir,
            "pixel_format": "rgba", "alpha": "straight",
            "fps": int(fps), "frame_count": int(n),
            "has_audio": False,        # V-1: only OTR_MasterAudioMux emits audio
            "engine_id": self.name, "family": self.family,
        }

    def canonicalize(self, raw, request, profile):
        """Canonicalize-time enforcement: the Track-3 directory-clip validator
        runs HERE (type/rgba/straight/silent/count==disk==ledger target)."""
        get = self._get(request)
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        expect = int(t_get("target_frame_count", 0) or 0) or None
        validate_directory_clip(raw, expect_frames=expect)
        return raw


__all__ = [
    "MeshStageEngine", "MESHER_ID", "MESHER_VERSION", "DEFAULT_W", "DEFAULT_H",
    "STAGE_SCRIPT", "BLENDER_ENV_LEAK_KEYS",
    "portrait_content_hash", "mesh_cache_key", "build_mesh_manifest",
    "write_manifest", "build_blender_cmd", "build_blender_env",
    "validate_frame_dir", "publish_frame_dir", "cleanup_stale_tmp",
]
