"""THE LANE PREFLIGHT MATRIX -- `docs/VIDEO_LANE_PREFLIGHT.md` as executable law.

Spec S8c (`docs/2026-08-09-SPEC-lab-findings-into-otr.md`): the transplant
plan's five "working right" criteria and the lessons ledger
(`docs/LANE_BUILD_LESSONS.md`) become ONE parametrized suite that runs over
`registry.all_engine_names()` and asserts, per lane, everything the matrix page
claims. A lane is IN the matrix when its row is green here; a future lane -- or
a future card in the 5070-Ti-and-down ladder -- inherits the whole checklist for
free.

Three mechanisms make this suite usable DURING a build rather than only after:

* :data:`EXEMPTIONS` -- a row that legitimately cannot satisfy a gate declares a
  NAMED exemption. Never a `pytest.skip`: a skip is invisible in the summary
  line and reads as coverage. An exemption is data, it is printed in the matrix
  report, and the preflight doc's per-row claims and this table must agree.
* :data:`EXPECTED_RED` -- the progressive ledger. Every entry is bound to a
  DEFECT ID from the spec's S8b audit and names the lane packet that owns it.
  The suite stays green while the roster is being repaired one lane at a time,
  WITHOUT ever claiming a defect is fixed.
* STRICT UNEXPECTED-PASS. When an `EXPECTED_RED` row starts passing, this suite
  FAILS and tells you to delete the entry. A stale expected-red is how a ledger
  rots into a rubber stamp -- the gate would keep saying "known bad" about code
  that is now good, and the next real regression would hide behind it.

CPU-safe and pure: no renders, no CUDA, no model loads. Every check is a read of
a declaration, a source text, or the on-disk evidence manifest.
"""

from __future__ import annotations

import inspect
import json
import os
import re

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes import otr_video_director as vd
from nodes._otr_shared import public_engines as pub
from nodes._otr_shared import still_plan_helpers as sph
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import registry as vreg


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(
    REPO_ROOT, "docs", "evidence", "video_evidence_manifest.json")
ENGINE_MATRIX_PATH = os.path.join(REPO_ROOT, "docs", "ENGINE_MATRIX.md")

#: The canvas rate the whole timeline is expressed at. A 24 fps model declares
#: 25 and CONVERTS at delivery (the Veo/H3 pattern); it never relabels.
CANVAS_FPS = 25

#: Lanes whose output Sage silently corrupts, so they must REFUSE rather than
#: render noise. LTX 0.9.8/2.3 is the family BUG-070 was written for; MiniMax H3
#: joins this set with its adapters (Comfy-Org/ComfyUI#15263, and the per-model
#: KJ probe FAILED on sm_120).
SAGE_SENSITIVE = frozenset({
    "ltx_video", "ltx_audio_in", "ltx_8gb",
    "minimax_h3_video", "minimax_h3_audio_in",
})

#: Method names that resolve a lane's PRIMARY weight. G1 reads these, not the
#: whole module: a module can mention `folder_paths` in an unrelated helper
#: while its checkpoint resolver is still a bare hardcoded path -- which is
#: exactly how wan_i2v shipped dead (lesson L1).
WEIGHT_RESOLVER_METHODS = (
    "_installed", "_ckpt_path", "_weight_paths", "_unet_path",
    "_primary_weight_path", "_model_paths",
    # `_resolve_unet` is the ONE HuMo resolver every tier delegates to (lane 2).
    # A lane that factors its resolution into a shared helper must not read as
    # unresolved just because the delegation moved the tokens one call away --
    # the gate follows the names it is given, so the name goes here.
    "_resolve_unet",
)

#: Any of these in a resolver proves it goes THROUGH ComfyUI's configured model
#: roots rather than around them. `_resolve_model_file` (wan_shared),
#: `_resolve` (the LTX module helper) and `_loader_token_path` (ltx_8gb) are all
#: folder_paths-backed and honour `extra_model_paths.yaml`; the gate also
#: requires the defining module itself to name `folder_paths`, so a helper
#: cannot launder a hardcoded path past this check by having the right name.
FOLDER_PATHS_TOKENS = (
    "folder_paths", "_resolve_model_file", "_resolve(", "_loader_token_path",
)

GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")

GATE_TITLES = {
    "G1": "weights resolve",
    "G2": "canvas truth",
    "G3": "contract matches runtime",
    "G4": "admission honesty",
    "G5": "audio law (V-1) self-probe",
    "G6": "guards fire early and by name",
    "G7": "public surface",
}


# ---------------------------------------------------------------------------
# NAMED EXEMPTIONS -- a row that cannot satisfy a gate for a real, permanent
# reason. Each value is the sentence that appears in the matrix report and in
# the lane's receipt. Adding one is a deliberate act, not a way past a red row:
# an exemption says "this gate does not apply here and here is why", never
# "this gate applies and we have not done it" (that is EXPECTED_RED).
# ---------------------------------------------------------------------------
_CLOUD_LANES = (
    "cloud_kling_avatar", "cloud_seedance_2", "cloud_vidu_q2_pro_fast_720p",
    "cloud_wan_i2v", "cloud_wan_i2v_audio", "google_omni_video",
    "google_veo_video", "word_razzle",
)
#: CPU/ffmpeg still lanes -- zero VRAM, no diffusion model, no frame ladder.
_STILL_LANES = ("still_flat", "still_motion", "still_pan", "still_word")
_PROCEDURAL_LANES = ("viz_camera", "viz_green", "viz_mxc_cpu", "viz_mxc_mandala")
#: Lanes whose contract is legitimately unbounded (they render whatever length
#: the beat asks for, at the canvas rate). mesh_stage belongs here for G3 only:
#: it IS a GPU lane that loads a checkpoint, so it earns NO VRAM exemption.
_UNBOUNDED_CONTRACT_LANES = _STILL_LANES + ("mesh_stage",)

EXEMPTIONS: dict = {}

for _lane in _CLOUD_LANES:
    EXEMPTIONS[(_lane, "G1")] = (
        "provider-side render: no local weight resolves on this box, so there "
        "is no folder_paths path to check; missing credentials fail LOUD at "
        "invoke-time auth resolution instead")
    EXEMPTIONS[(_lane, "G2")] = (
        "the provider owns the render size; the canonicalizer conforms the "
        "returned asset, so a local render_canvas declaration would be fiction")
    EXEMPTIONS[(_lane, "G5")] = (
        "audio is stripped and RE-PROBED provider-side by "
        "cloud_media_canonical (must_strip_audio + a post-strip zero-audio-"
        "stream proof) -- the same law as validate_silent_clip_contract, "
        "enforced by a different function on the same emitted file")
    EXEMPTIONS[(_lane, "G6")] = (
        "no local model runs, so Sage residency and boot contracts cannot "
        "affect this lane's output")
    EXEMPTIONS[(_lane, "G4")] = (
        "the render happens provider-side at zero local VRAM, so there is no "
        "local envelope to qualify; the real refusal on this lane is "
        "invoke-time auth/credit failure, which is already loud")

for _lane in _STILL_LANES + _PROCEDURAL_LANES:
    EXEMPTIONS[(_lane, "G6")] = (
        "no diffusion model loads on this lane, so Sage residency cannot "
        "corrupt its output and it declares no boot contract")
    EXEMPTIONS[(_lane, "G4")] = (
        "procedural / CPU lane: it holds no model in VRAM, so there is no "
        "envelope to qualify and no over-budget render to refuse")

    # NOTE: there is deliberately NO blanket G3 exemption. G3.1 (the frame
    # rate) is scoped inside the gate by CANVAS_RATE_LANES / PROVIDER_RATE_LANES
    # so that G3.3 -- continuity declared explicitly, never defaulted -- still
    # runs on every single lane. Greying out the whole G3 row for a cheap lane
    # is how a defaulted CONTINUITY_NONE gets to look like a decision.


# ---------------------------------------------------------------------------
# THE PROGRESSIVE EXPECTED-RED LEDGER.
#
# Every entry names the S8b defect id (or the spec section) and the LANE PACKET
# that owns it, per the 21-lane order in
# `docs/2026-08-10-FINAL-QA-video-build-corpus.md`. An entry leaves this table
# in the same commit that closes its lane -- and if it starts passing before
# then, the suite fails and says so.
# ---------------------------------------------------------------------------
EXPECTED_RED: dict = {
    # LANE 1 CLOSED 2026-08-11 -- wan_i2v's G1 and G2 rows left this table when
    # the lane went green. The defects were: a hardcoded
    # models/checkpoints/wan2.2-i2v.safetensors default with a bare
    # os.path.exists that never consulted folder_paths (S8b-1 / lesson L1), and
    # an undeclared render_canvas that let the lane fall through to 1472x832
    # (S1 / lesson L2). Both are now pinned by tests/test_wan_i2v.py.
    # LANE 2 CLOSED 2026-08-11 -- humo_14B_169's G2 row left this table. The
    # defect was S8b-4: the request was rewritten to 1472x832 while the graph
    # rendered 832x480 (3.07x), and OTR_HUMO_WIDTH/HEIGHT could move it again,
    # so 832x480 was a default rather than a runtime guarantee. The tier now
    # declares its measured canvas and a contradicting override is a named
    # refusal. Pinned by tests/test_boot_contracts.py.
    # LANE 3 CLOSED 2026-08-11 -- humo_1.7B and humo_1.7B_169 left this table.
    # Both now declare their canvas, and the portrait tier's profile stopped
    # claiming 832x480 on a lane whose whole identity is the pillarbox.
    # LANE 4 CLOSED 2026-08-11 -- the last HuMo tier declares its canvas,
    # and BOTH its profiles stopped claiming landscape on the pillarbox.
    # LANE 7 CLOSED 2026-08-11 -- ltx_audio_in's G2 and G6 rows left this
    # table. G6 was S8b-9: OTR_LTX_AV_RESERVE_VRAM_GB was the one module-scope
    # numeric env read this adapter's own _env_num guard was never applied to,
    # so a typo raised at import, the guarded import swallowed it, and the lane
    # vanished from the dropdown (registry 27 -> 26) while frame_contract_for
    # silently answered SINGLE_ONLY. G2 was S3 + S8b-10 together, and they
    # turned out to be the same defect: the canvas was computed by an inline
    # RECIPE-DEPENDENT branch in the driver (832x480 under ia2v, 512x288
    # otherwise) that declared_render_canvas would have overruled anyway, and
    # 832x480 halves to the ia2v stage-A latent 416x240 -- 240 % 32 == 16.
    # Because LTXVLatentUpsampler doubles with NO target size, the delivered
    # canvas IS 2x the stage-A base, so only a /64 canvas has a legal stage A.
    # The lane declares (1024, 576), the halving is validated at its root, and
    # an env canvas that disagrees is a named refusal. Pinned by
    # tests/test_ltx_av_ia2v_canonical.py and tests/test_ltx_av_driver_wiring.py.
    # LANE 8 CLOSED 2026-08-11 -- ltx_8gb's G2 and G6 rows left this table. G6
    # was S8b-13: the only one of the three LTX lanes with NO
    # assert_sage_not_patched gate, on the exact family BUG-070 was written for
    # (int8-PV Sage process-aborts LTX with no traceback, so "no gate" means a
    # dead process instead of a named refusal), plus no node-class gate, so a
    # missing LTXV class surfaced at load() after the checkpoint was paid for.
    # Both now fire in assert_usable, Sage first. G2 was S8b-11: two profiles
    # claimed 832x480 on a lane that DECLARES 512x288 -- the declaration
    # overruled them, so the config an operator reads disagreed with the render
    # by 2.7x the pixels on the tier that exists because it cannot afford them.
    # 512x288 stands: it is the 2026-07-26 arc judgment's ruled canvas (exact
    # 16:9, /32-clean, zero pad area) and the profiles moved to it.
    # LANE 10 CLOSED 2026-08-11 -- mesh_stage's G1, G2, G3 and G5 rows all left
    # this table, and the G3 fix took the four still lanes' rows with it.
    # G1 was two defects in one row. S8b-16: `_node_candidates()` named ten core
    # hy3d classes that resolved only inside `load()`, so preflight passed and
    # the render died mid-beat after the checkpoint was paid for -- there is now
    # a node gate in `assert_usable`, ordered before weight resolution, reading
    # the ACTIVE candidate set and collecting every miss. And lesson L1, the
    # wan_i2v killer: `_ckpt_path` walked a hardcoded models/checkpoints plus an
    # HF_HOME sibling and never consulted `folder_paths`, so a checkpoint
    # registered through extra_model_paths.yaml was invisible on the runtime and
    # one installed under this box's real models root was invisible off it. It
    # now probes the env pin, `folder_paths`, the historical dirs, then lane 1's
    # `wan_shared.configured_models_root()` LAST -- additive, so it can only turn
    # a false negative into the truth.
    # G2 was S8b-11 plus an inline branch: the lane declared no render_canvas and
    # picked its size with a magic-number sniff inside render_clip (832x480 and
    # no explicit canvas -> 1472x832), the same shape lane 7 deleted. It declares
    # (1472, 832) now -- what Blender is actually told and what
    # `validate_frame_dir` refuses to publish anything else at -- the branch is
    # gone, and its one profile carries the same numbers as a drift guard.
    # G3 was the shared `_CheapFamilyBase.frame_contract`, whose comment had
    # reasoned about continuity for months while never passing `continuity=`.
    # Per lesson L13 that is a defect in every adapter sharing the mechanism, so
    # it was fixed at the base and the four still lanes' G3 rows came out in the
    # same commit. The viz lanes and google_omni_video have the identical defect
    # through their OWN contracts and stay red below -- a shared fix that did not
    # reach them must not be reported as though it had.
    # G5 was L4 exactly: `canonicalize` called `validate_directory_clip`, whose
    # audio check read `has_audio is not False` off the dict this adapter wrote
    # -- a declaration checking a declaration -- while frames were accepted by
    # FILENAME EXTENSION, so a file named .png containing anything at all
    # counted. The directory contract now PROVES each frame from its magic
    # bytes, which is what makes "no audio stream" a fact about the bytes, and
    # G5 is taught that named function for directory-clip lanes only.
    ("google_omni_video", "G3"): (
        "L3 -- CONTINUITY_NONE is inherited rather than declared on this "
        "text-to-video cloud lane. NONE is near-certainly the right answer "
        "(no reference input exists to lock a first frame), but it is "
        "currently a default and not a decision. OWNER: none tonight -- "
        "cloud lanes are outside the 21-lane transplant scope; queued in "
        "docs/GO_FORWARD_PLAN.md item 5 as a follow-up row."),
}

# The four procedural visualizers and the four CPU still lanes carry the same
# two defects each, from the same two audits, and each is owned by its own lane
# packet in the 21-lane order. Generated rather than typed eight times over, so
# the owner mapping stays visibly one-to-one.
_CHEAP_LANE_OWNERS = {
    "viz_green": "lane 11 (viz_green)",
    "viz_camera": "lane 12 (viz_camera)",
    "viz_mxc_cpu": "lane 13 (viz_mxc_cpu)",
    "viz_mxc_mandala": "lane 14 (viz_mxc_mandala)",
    "still_motion": "lane 15 (still_motion)",
    "still_pan": "lane 16 (still_pan)",
    "still_flat": "lane 17 (still_flat)",
    "still_word": "lane 18 (still_word)",
}
#: The four still lanes reach their FrameContract through `_CheapFamilyBase`,
#: and lane 10 fixed it there (lesson L13: a defect in a shared mechanism is a
#: defect in every adapter sharing it, and it gets swept before the lane that
#: found it closes). So their G3 rows left this table with mesh_stage's, in the
#: same commit -- the strict unexpected-pass gate insists on exactly that.
#: The four visualizers are NOT reached by that fix: each declares its own
#: contract and inherits the default independently, so each stays red until its
#: own packet runs. Their G2 rows are untouched either way -- a shared contract
#: fix says nothing about a lane's canvas.
#: `viz_green` joins them in lane 11 -- but for the opposite reason, and the
#: difference matters. The still lanes came green for FREE off a shared base
#: they did not own. `viz_green` is NOT in this tuple even though its G3 row
#: also left the table in lane 11 -- it leaves via the `continue` below, which
#: drops BOTH of its rows, and listing it here as well would be a membership
#: nothing ever reads. The distinction is the point: the still lanes came green
#: off a shared base, while `viz_green` declares its OWN contract in its own
#: module and lane 11 passed `continuity=` there by hand. The other three
#: visualizers each declare their own too, so nothing lane 11 did reaches them
#: and lanes 12-14 each still owe their own one-line declaration.
_G3_STILL_DEFAULTED_LANES = ("still_motion", "still_pan", "still_flat",
                             "still_word")
for _lane, _owner in _CHEAP_LANE_OWNERS.items():
    if _lane in ("viz_green", "viz_camera", "viz_mxc_cpu", "viz_mxc_mandala",
                 "still_motion"):
        # LANE 11 CLOSED 2026-08-11 -- viz_green's G2 row left this table by
        # declaring its profile channel INERT (see
        # PROFILE_CANVAS_DOCUMENTED_DEAD above) rather than by declaring a
        # canvas. The first draft of this lane DID declare (1472, 832), on the
        # measured argument that build_request_from_shot already hands it
        # exactly that while render_single hands it 832x480. A Codex consult
        # broke that framing and was right: the 1472x832 is the default of
        # OTR_VIDEO_LANDSCAPE_CANVAS, an operator lever, and a declaration is
        # applied LAST -- so declaring would have made this the one visualizer
        # that silently ignores the lever, on a lane with no native canvas to
        # pin. Lesson L2's own override-path check is what catches it, and the
        # draft had walked past that check. Its G3 row left in the same commit.
        #
        # LANE 12 CLOSED 2026-08-11 -- viz_camera's two rows left the same way,
        # after re-checking the PREMISE on its own render path rather than
        # assuming the family shares one (L19's runnable check). It does: every
        # painter, table and encoder call is built from the request's w/h.
        # LANE 13 CLOSED 2026-08-11 -- viz_mxc_cpu's two rows left the same way,
        # premise re-checked again on its own painter (ring_geom(w, h) and the
        # scanline/vignette/font tables are all request-derived). Declaring
        # would have been worst on THIS lane: its stated purpose is running on
        # ANY box, and a pinned canvas is the opposite of portable.
        # LANE 14 CLOSED 2026-08-11 -- viz_mxc_mandala's two rows left last, and
        # it was the one genuinely worth suspecting: the only visualizer with a
        # NAMED external dependency, painting through a graphics library rather
        # than numpy. Re-checked anyway and the premise held (an ImageSurface is
        # whatever size you ask for). ALL FOUR VISUALIZERS ARE NOW CLOSED, and
        # every one of them closed by declaring the profile canvas channel INERT
        # rather than by declaring a canvas -- if a future procedural lane finds
        # itself wanting to declare one, that is the anomaly, not the default.
        #
        # LANE 15 CLOSED 2026-08-11 -- still_motion's G2 row left the same way,
        # its G3 row having already gone green off lane 10's shared-base fix.
        # It is the FIRST non-visualizer to take the INERT answer, and it was
        # re-checked against a different ffmpeg path (wrapper_bridge's still
        # builders, not scope_draw's encoder). still_pan / still_flat /
        # still_word keep their G2 rows -- lanes 16-18 each decide their own.
        continue
    EXPECTED_RED[(_lane, "G2")] = (
        "S8b-11 -- the lane declares no render_canvas while its profiles set "
        "render.canvas_w/h, so the configured number reaches the node-87 "
        "director widgets and is then OVERWRITTEN by the 1472x832 landscape "
        "default before the render. Its packet decides whether to declare the "
        "canvas or document the channel inert. OWNER: %s." % _owner)
    if _lane not in _G3_STILL_DEFAULTED_LANES:
        EXPECTED_RED[(_lane, "G3")] = (
            "L3 -- the FrameContract never names continuity, so "
            "CONTINUITY_NONE is inherited rather than decided. OWNER: %s."
            % _owner)


# ---------------------------------------------------------------------------
# Shared readers
# ---------------------------------------------------------------------------
def _engines():
    out = []
    for name in sorted(vreg.all_engine_names()):
        try:
            out.append((name, vreg.get_engine(name)))
        except Exception as exc:  # noqa: BLE001
            pytest.fail("engine %r cannot be built: %r" % (name, exc))
    return out


ENGINE_NAMES = [name for name, _ in _engines()]


def _mro_source(cls) -> str:
    chunks = []
    for base in cls.__mro__:
        if base is object:
            continue
        try:
            chunks.append(inspect.getsource(base))
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(chunks)


def _method_source(eng, attr: str) -> str:
    out = []
    for base in type(eng).__mro__:
        fn = base.__dict__.get(attr)
        if fn is None:
            continue
        try:
            out.append(inspect.getsource(fn))
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(out)


def _defining_module_source(eng, attr: str) -> str:
    """Source of the MODULE that defines ``attr`` for this engine.

    The V-1 probe is called from a canonicalize helper that may live beside the
    method rather than inside it, so the honest unit is the defining module.
    """
    for base in type(eng).__mro__:
        if attr in base.__dict__:
            try:
                return inspect.getsource(inspect.getmodule(base))
            except Exception:  # noqa: BLE001
                return ""
    return ""


def _capability_row(name: str) -> dict:
    return vreg.CAPABILITIES.get(name, {})


def _is_local_gpu_lane(name: str) -> bool:
    row = _capability_row(name)
    return (list(row.get("device_backends") or []) == ["cuda"]
            and not row.get("practical_without_gpu", False))


def _has_weights(name: str) -> bool:
    return bool(_capability_row(name).get("model_requirements"))


def _manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# The seven gates. Each returns a list of failure sentences for ONE lane; an
# empty list is a green row. Gates are pure reads -- never a render.
# ---------------------------------------------------------------------------
def gate_g1_weights(name, eng):
    """G1.1 weights resolve via folder_paths / a documented env pin, no bare
    os.path.exists on a hardcoded default. G1.2 a missing weight produces a
    NAMED EngineUnusable from assert_usable, never a swallowed import."""
    if not (_is_local_gpu_lane(name) and _has_weights(name)):
        return []
    bad = []
    resolver_src = "\n".join(
        _method_source(eng, m) for m in WEIGHT_RESOLVER_METHODS)
    if not resolver_src.strip():
        bad.append(
            "declares model_requirements %r but exposes no weight-resolution "
            "method (%s) for preflight to check"
            % (list(_capability_row(name)["model_requirements"]),
               ", ".join(WEIGHT_RESOLVER_METHODS)))
    elif not (any(tok in resolver_src for tok in FOLDER_PATHS_TOKENS)
              and "folder_paths" in _defining_module_source(eng, "_installed")
              + _defining_module_source(eng, "_ckpt_path")
              + _defining_module_source(eng, "_weight_paths")):
        bad.append(
            "its weight resolver never reaches folder_paths: it resolves a "
            "hardcoded default directly, which is the wan_i2v killer "
            "(lesson L1)")
    au = _method_source(eng, "assert_usable")
    if not au.strip():
        bad.append("declares no assert_usable, so a missing weight cannot "
                   "produce a NAMED refusal before the first forward")
    elif "EngineUnusable" not in au:
        bad.append("assert_usable never raises EngineUnusable, so a missing "
                   "weight cannot fail closed by name")
    return bad


def _is_32_legal(canvas):
    w, h = int(canvas[0]), int(canvas[1])
    return w > 0 and h > 0 and w % 32 == 0 and h % 32 == 0


#: Lanes whose PROFILE canvas channel is knowingly, documentedly INERT -- the
#: number an operator sets in `render.canvas_w/h` cannot decide what this lane
#: renders at. An entry here is a claim the lane's row on the matrix page
#: repeats in words; it is not a way to silence G2.3.
#:
#: "INERT", NOT "DEAD", and the distinction was measured (lane 11's opening
#: check, 2026-08-11). The corpus calls this channel "read by nothing"; it is
#: nothing of the kind. `_otr_workflow_apply` flattens `render.canvas_w/h` into
#: the node-87 `OTR_VideoDirector` widgets -- regenerating a variant after
#: editing a profile visibly moves them -- and the director turns those widgets
#: into `request["canvas"]`. The number is then OVERWRITTEN by
#: `build_request_from_shot`'s landscape default for every non-face family, and
#: a `render_canvas` declaration overrules that in turn.
#:
#: Why the distinction earns its keep: a channel nobody reads is a tidiness
#: problem, while a channel that is read, carried into the request, and then
#: silently overruled is a TRAP -- the operator edits the profile, watches the
#: widget change, and concludes it took effect. Any lane documenting itself
#: here must say the second thing.
PROFILE_CANVAS_DOCUMENTED_DEAD: dict = {
    "viz_green": (
        "INERT, and this lane cannot honestly declare a canvas instead. "
        "`eng_visualizer.render_clip` paints and encodes at exactly the size "
        "the request carries -- no latent grid, no fixed model input, no "
        "canvas-dependent constant -- so it has no native canvas to declare. "
        "The 1472x832 an episode hands it is not a property of the lane: it is "
        "the default of OTR_VIDEO_LANDSCAPE_CANVAS, an operator lever, applied "
        "by build_request_from_shot to every non-face family. Declaring would "
        "overrule that lever (declared_render_canvas is applied LAST), making "
        "this the one visualizer that ignores it -- a behaviour change wearing "
        "a documentation label, which lesson L2's own override-path check "
        "exists to catch. What the profiles set IS carried (profile -> applier "
        "-> node-87 director widgets -> request canvas) and then overwritten, "
        "so the field cannot decide this lane's size and is declared inert "
        "here rather than reconciled to a number that would be equally unable "
        "to decide it. Lane 11, 2026-08-11."),
    "viz_camera": (
        "INERT, same mechanism and same reason as viz_green, and the premise "
        "was RE-CHECKED on this engine rather than inherited (L19 says copy "
        "the reasoning, not the shape). `eng_viz_camera.render_clip` builds "
        "paint_golden_camera_frame, the scanline table, the vignette and the "
        "encoder from the request's own w/h -- no latent grid, no trained "
        "input size, no canvas-dependent constant -- so the 1472x832 an "
        "episode hands it is OTR_VIDEO_LANDSCAPE_CANVAS's default, an operator "
        "lever, not a property of the lane. Declaring would overrule that "
        "lever for this lane alone. Lane 12, 2026-08-11."),
    "viz_mxc_cpu": (
        "INERT, same mechanism as viz_green and viz_camera, premise RE-CHECKED "
        "on this engine's own path (L19). `eng_viz_rainbow.render_clip` hands "
        "paint_rainbow_frame the request's w/h and lays the dial out through "
        "ring_geom(w, h); the scanline table, the vignette, the small font and "
        "the encoder are all built from the same pair. No latent grid, no "
        "trained input size, no canvas-dependent constant. Declaring would "
        "overrule OTR_VIDEO_LANDSCAPE_CANVAS for this lane alone -- and this "
        "is the tier whose stated purpose is running on ANY box, so pinning a "
        "canvas here would be the opposite of what it is for. "
        "Lane 13, 2026-08-11."),
    "viz_mxc_mandala": (
        "INERT, and this was the visualizer most likely to have needed a real "
        "declaration -- the only one with a NAMED external dependency, painting "
        "through a graphics library rather than numpy -- so the premise was "
        "re-checked with more suspicion, not less. It holds: render_clip "
        "allocates cairo.ImageSurface(FORMAT_ARGB32, w, h) from the request's "
        "own dimensions, paint_mandala(ctx, w, h, ...) and "
        "mandala_surface_to_rgb take the same pair, and so do the scanline and "
        "vignette tables and the encoder. Cairo imposes no canvas of its own -- "
        "an ImageSurface is whatever size you ask for. Lane 14, 2026-08-11."),
    "still_motion": (
        "INERT, and the premise was re-checked on a path NONE of the four "
        "visualizers touch: these lanes reach ffmpeg through "
        "wrapper_bridge.ffmpeg_still_motion_cmd, not scope_draw's encoder. It "
        "still holds -- the builder takes the caller's width/height and scales "
        "the still to COVER them, so the canvas is whatever the request "
        "carried. The one difference found in the whole family sweep is that "
        "these builders pass the dims through even_dim() first; that is a "
        "yuv420p mod-2 CODEC requirement applied to whatever it is given (and "
        "a no-op at every canvas in play, since 1472x832 and 832x480 are "
        "already even), not a native canvas. So the 1472x832 an episode hands "
        "this lane is OTR_VIDEO_LANDSCAPE_CANVAS's default -- an operator "
        "lever -- and declaring would overrule it for this lane alone. "
        "Lane 15, 2026-08-11."),
}


def gate_g2_canvas(name, eng):
    """G2.1 GPU lanes with a fixed render size declare render_canvas, both axes
    /32-legal. G2.2 the declaration equals what the graph emits -- enforced by
    the lane's own drift test, which G2 requires to exist. G2.3 every profile
    canvas either matches the declaration or the dead channel is documented."""
    bad = []
    canvas = getattr(eng, "render_canvas", None)
    if _is_local_gpu_lane(name) and not canvas:
        bad.append(
            "declares no render_canvas, so build_request_from_shot falls "
            "through to the 1472x832 landscape default no matter what its "
            "profile says (lesson L2)")
    elif canvas:
        if len(tuple(canvas)) != 2:
            bad.append("render_canvas %r is not a (w, h) pair" % (canvas,))
            return bad
        if not _is_32_legal(canvas):
            bad.append("render_canvas %dx%d is not /32-legal on both axes"
                       % (int(canvas[0]), int(canvas[1])))
        # G2.2's enforcement is a per-lane drift test naming BOTH the engine
        # and its canvas. Requiring the pin here is what stops a declaration
        # from drifting away from the graph silently.
        if not _canvas_pin_exists(name, canvas):
            bad.append(
                "declares render_canvas %dx%d with no test pinning it: every "
                "other declaring lane has one (tests/test_fastwan_8gb.py, "
                "tests/test_ltx_8gb_canonical_canvas.py), and without a pin "
                "the declaration can drift away from the graph in silence"
                % (int(canvas[0]), int(canvas[1])))
    # --- G2.3: what the PROFILES say this lane renders at ---
    assigned = profile_canvases_for(name)
    if assigned and name not in PROFILE_CANVAS_DOCUMENTED_DEAD:
        if canvas:
            wrong = sorted(
                "%s says %dx%d" % (pid, w, h)
                for pid, (w, h) in assigned.items()
                if (w, h) != (int(canvas[0]), int(canvas[1])))
            if wrong:
                bad.append(
                    "declares render_canvas %dx%d, which the declaration "
                    "applies LAST and overrules -- but these profiles "
                    "disagree in the config an operator reads: %s"
                    % (int(canvas[0]), int(canvas[1]), "; ".join(wrong)))
        else:
            sizes = sorted({"%dx%d" % v for v in assigned.values()})
            bad.append(
                "%d profile(s) set render.canvas_w/h (%s) on a lane that "
                "declares no render_canvas, so the number an operator "
                "configures is OVERWRITTEN by the 1472x832 landscape default "
                "and never reaches the render (S8b item 11)"
                % (len(assigned), ", ".join(sizes)))
    return bad


_PROFILES_DIR = os.path.join(REPO_ROOT, "config", "profiles")


def _load_profiles():
    out = {}
    if not os.path.isdir(_PROFILES_DIR):
        return out
    for fname in sorted(os.listdir(_PROFILES_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_PROFILES_DIR, fname), "r",
                      encoding="utf-8") as fh:
                out[fname[:-5]] = json.load(fh)
        except Exception:  # noqa: BLE001 -- a malformed profile is another
            continue      # test's business; this gate reads what parses
    return out


_PROFILES = _load_profiles()


def profile_canvases_for(name) -> dict:
    """``{profile_id: (w, h)}`` for every profile that SELECTS this engine and
    also declares a render canvas. Selection is read from the same two places
    production reads: the per-role visual overrides and the video render slot,
    both resolved through ``resolve_engine_id`` so a public or legacy id in a
    profile still lands on the right lane."""
    out = {}
    for pid, prof in _PROFILES.items():
        picks = set()
        for key, value in (prof.get("role_overrides") or {}).items():
            if key.endswith("_visual"):
                picks.add(pub.resolve_engine_id(value))
        slot = (prof.get("slot_overrides") or {}).get("video_render_engine")
        if slot:
            picks.add(pub.resolve_engine_id(slot))
        if name not in picks:
            continue
        render = prof.get("render") or {}
        w, h = render.get("canvas_w"), render.get("canvas_h")
        if w and h:
            out[pid] = (int(w), int(h))
    return out


_TESTS_DIR = os.path.join(REPO_ROOT, "tests")


def _canvas_pin_exists(name, canvas) -> bool:
    """True iff some test file mentions this engine id AND its exact canvas."""
    needle_a = repr(str(name))
    needle_b = '"%s"' % name
    dims = ("(%d, %d)" % (int(canvas[0]), int(canvas[1])),
            "%dx%d" % (int(canvas[0]), int(canvas[1])),
            "%d, %d" % (int(canvas[0]), int(canvas[1])))
    for fname in os.listdir(_TESTS_DIR):
        if not fname.startswith("test_") or not fname.endswith(".py"):
            continue
        try:
            with open(os.path.join(_TESTS_DIR, fname), "r",
                      encoding="utf-8") as fh:
                src = fh.read()
        except Exception:  # noqa: BLE001
            continue
        if needle_a not in src and needle_b not in src:
            continue
        if any(d in src for d in dims):
            return True
    return False


#: Lanes that legitimately declare no model-native frame rate: they render
#: whatever length the beat asks for, at whatever rate the canvas is. G3.1 is
#: not applicable to them -- but G3.3 (continuity declared explicitly) still is,
#: which is why this is a constant inside the gate rather than a table entry
#: that would grey out the whole row.
CANVAS_RATE_LANES = frozenset(_UNBOUNDED_CONTRACT_LANES)

#: Lanes whose rate is the PROVIDER's and is reconciled duration-preserving by
#: `_otr_shared/cloud_media_canonical`, so `target_fps` is not the adapter's to
#: declare.
PROVIDER_RATE_LANES = frozenset(_CLOUD_LANES)


#: G3.3's reader, and it lives in the ENGINE module rather than here on purpose
#: (lane 12, 2026-08-11): this gate and every lane's own continuity test must
#: ask the SAME question, and two readers of one invariant is how they drift.
#: It is AST-based because the substring check it replaced was satisfiable by
#: the COMMENT explaining the declaration -- see its docstring.
declares_continuity_kwarg = fc.declares_continuity_kwarg


def gate_g3_contract(name, eng):
    """G3.1 native_fps == target_fps == 25 (a 24 fps model declares 25 and
    converts at delivery). G3.2 discrete menus in FRAMES. G3.3 continuity
    declared explicitly, never defaulted -- which applies to EVERY lane."""
    contract = fc.frame_contract_for(eng)
    bad = []
    if contract is fc.SINGLE_ONLY:
        bad.append("declares no frame_contract and resolves to SINGLE_ONLY")
        return bad
    rate_applies = (name not in CANVAS_RATE_LANES
                    and name not in PROVIDER_RATE_LANES)
    if rate_applies:
        native = int(contract.native_fps or 0)
        target = getattr(eng, "target_fps", None)
        if native != CANVAS_FPS:
            bad.append(
                "declares native_fps=%r; the canvas rate is %d and a "
                "model-rate lane must declare the canvas rate and CONVERT at "
                "delivery (lesson L3)" % (contract.native_fps, CANVAS_FPS))
        if target != CANVAS_FPS:
            bad.append("declares target_fps=%r; must be %d"
                       % (target, CANVAS_FPS))
    if contract.discrete_frames:
        vals = [int(v) for v in contract.discrete_frames]
        if sorted(vals) != vals or len(set(vals)) != len(vals):
            bad.append("discrete_frames %r is not strictly ascending" % (vals,))
        # A menu authored in SECONDS is the trap the field rename exists for:
        # every legal clip is at least ~1 s, so any rung below the canvas rate
        # is a seconds-for-frames substitution.
        if vals and min(vals) < CANVAS_FPS:
            bad.append(
                "discrete_frames %r contains a rung below the canvas rate -- "
                "THE UNIT IS FRAMES, NOT SECONDS (frame_contract.py:104-112)"
                % (vals,))
    if not declares_continuity_kwarg(eng):
        bad.append("never passes continuity= at its FrameContract declaration, "
                   "so it inherits the CONTINUITY_NONE default silently and "
                   "refuses chaining without saying so (lesson L3)")
    return bad


def gate_g4_admission(name, eng):
    """G4.1 the lane has a QUALIFIED cost row / envelope key, OR its receipts
    say "admission NOT enforced" IN WORDS, on disk, reachable in the manifest.
    A silently unguarded lane that looks guarded is the failure this forbids."""
    if mc.cost_row_may_refuse(name):
        return _receipt_lora_agrees_with_graph(name, eng)
    unenforced = _manifest().get("admission_unenforced") or {}
    reason = str(unenforced.get(name) or "").strip()
    if not reason:
        return ["is not in QUALIFIED_COST_ROWS and the evidence manifest does "
                "not record it as admission-unenforced, so its receipts would "
                "imply a guard that does not run"]
    return _receipt_lora_agrees_with_graph(name, eng)


def _receipt_lora_agrees_with_graph(name, eng):
    """G4.2 -- a lane's LoRA receipt must equal what its GRAPH does.

    THE DEFECT THIS CLOSES (retro bug hunt on lanes 0-6, 2026-08-11). The two
    HuMo 1.7B tiers switch the distill LoRA off by setting its token to the
    STRING "none", and the graph honours that through `_lora_is_skipped`. Both
    receipts instead used raw truthiness -- and `bool("none")` is True -- so
    those lanes rendered LoRA-free while stamping `_lora` and `use_lora=True`,
    and `otr_credits_roll` printed "lora" into PUBLISHED credits.

    It survived six lanes and a green G4 row because G4 only asked whether
    admission was honestly declared; nothing compared a receipt against the
    render it describes. That is a hole in the GATE, not just in one lane, so
    the check lives here where every later lane inherits it.

    Generic on purpose: any lane exposing `_lora_is_skipped` is checked, so a
    future engine that adopts the skip-token idiom is covered on arrival
    without anyone remembering this file exists.
    """
    skipped = getattr(eng, "_lora_is_skipped", None)
    names = getattr(eng, "_loader_names", None)
    telem = getattr(eng, "_clip_telemetry", None)
    if not (callable(skipped) and callable(names) and callable(telem)):
        return []
    try:
        token = (names() or {}).get("lora")
        graph_loads = bool(token) and not skipped(token)
        declared = bool(telem(832, 480).get("use_lora"))
    except Exception:  # noqa: BLE001 -- a lane that cannot answer is not a lie
        return []
    if declared != graph_loads:
        return ["its receipt says use_lora=%s while its GRAPH %s the LoRA "
                "(token %r) -- a receipt that records a falsehood, and credits "
                "publish this field" % (declared,
                                        "loads" if graph_loads else "skips",
                                        token)]
    recipe = getattr(eng, "_recipe_receipt", None)
    if callable(recipe):
        try:
            stamped = recipe().endswith("_lora")
        except Exception:  # noqa: BLE001
            return []
        if stamped != graph_loads:
            return ["its recipe receipt %s a _lora suffix while its GRAPH %s "
                    "the LoRA" % ("carries" if stamped else "omits",
                                  "loads" if graph_loads else "skips")]
    return []


#: Lanes that deliver a frame DIRECTORY instead of an mp4, and the named
#: function that carries the audio law for them.
#:
#: G5 is a LEXICAL gate: it greps the canonicalize path for the name of the
#: function that proves silence. `validate_silent_clip_contract` ffprobes an
#: mp4 for audio streams, and `mesh_stage` -- the only directory-clip lane in
#: the roster (`"type": "directory"`, straight-alpha PNG frames) -- has no
#: container to probe. `validate_directory_clip` is its twin: it reads every
#: frame's MAGIC BYTES through `list_directory_frames` and refuses anything
#: that is not really a PNG/EXR still, which makes "carries no audio stream" a
#: structural fact about the bytes rather than a naming convention.
#:
#: TEACHING THE GATE A NEW NAME IS THE SANCTIONED MOVE, and widening it is not
#: (lesson L9: G1 was taught `_resolve_unet` rather than made to accept any
#: resolver). A gate that accepted "some validator, whatever it is called"
#: would let a future lane launder a missing proof past it. The mapping is
#: per-lane and explicit, and `test_the_directory_clip_audio_law_really_proves_the_frames`
#: below asserts the named function actually refuses a mis-named non-image --
#: so the name cannot go on meaning something after the proof behind it rots.
DIRECTORY_CLIP_AUDIO_LAW = {
    "mesh_stage": "validate_directory_clip",
}


def gate_g5_audio_law(name, eng):
    """G5.1 the adapter's canonicalize path runs validate_silent_clip_contract
    -- or, for a directory-clip lane, its named twin -- on its OWN emitted
    artifact. A has_audio: False literal is not evidence."""
    canon = getattr(eng, "canonicalize", None)
    if canon is None:
        return []
    src = _defining_module_source(eng, "canonicalize")
    wanted = DIRECTORY_CLIP_AUDIO_LAW.get(name, "validate_silent_clip_contract")
    if wanted not in src:
        return ["its canonicalize path never calls %s, so silence is DECLARED "
                "and never proved on the emitted artifact (lesson L4)"
                % wanted]
    return []


_MODULE_SCOPE_ENV_NUM = re.compile(
    r"^_?[A-Za-z][A-Za-z0-9_]*\s*=\s*(?:float|int)\s*\(\s*os\.environ",
    re.MULTILINE)


def gate_g6_guards(name, eng):
    """G6.1 Sage-sensitive lanes call assert_sage_not_patched inside
    assert_usable. G6.3 module-scope env reads go through the guarded numeric
    parser -- a malformed env var must not delete the lane from the registry."""
    bad = []
    if name in SAGE_SENSITIVE:
        au = _method_source(eng, "assert_usable")
        if "assert_sage_not_patched" not in au:
            bad.append(
                "is Sage-sensitive (Sage silently turns its output to noise) "
                "but assert_usable never calls assert_sage_not_patched")
    for base in type(eng).__mro__:
        if base is object:
            continue
        mod = inspect.getmodule(base)
        if mod is None or not getattr(mod, "__file__", None):
            continue
        if "_otr_video_engines" not in str(mod.__file__):
            continue
        try:
            src = inspect.getsource(mod)
        except Exception:  # noqa: BLE001
            continue
        for m in _MODULE_SCOPE_ENV_NUM.finditer(src):
            bad.append(
                "%s has an UNGUARDED module-scope numeric env read (%r): a "
                "malformed value raises at import, the guarded import in "
                "_otr_video_engines/__init__.py swallows it, and the lane "
                "vanishes from the dropdown with nothing in the log "
                "(lesson L5)"
                % (os.path.basename(mod.__file__),
                   m.group(0).strip().splitlines()[0]))
    return bad


def gate_g7_surface(name, eng):
    """G7.1 exactly one live menu id per internal engine. G7.4 still_plan
    declared and audit-clean. Plus the ENGINE_MATRIX.md row, which is a
    generated drift gate and must name every registered lane."""
    bad = []
    try:
        vd.exact_menu_option_for(name)
    except Exception as exc:  # noqa: BLE001
        bad.append("exact_menu_option_for failed: %s" % (exc,))
    if not sph.engine_has_still_plan(eng):
        bad.append("declares no still_plan (missing is UNKNOWN and fails "
                   "closed; an explicit () means 'needs no images')")
    elif not sph.engine_has_valid_still_plan(eng):
        try:
            sph.validate_still_plan(getattr(eng, "still_plan"))
        except ValueError as exc:
            bad.append("still_plan does not validate: %s" % (exc,))
    if os.path.isfile(ENGINE_MATRIX_PATH):
        with open(ENGINE_MATRIX_PATH, "r", encoding="utf-8") as fh:
            if name not in fh.read():
                bad.append("has no row in docs/ENGINE_MATRIX.md, the "
                           "generated drift gate")
    else:
        bad.append("docs/ENGINE_MATRIX.md is missing")
    return bad


GATE_FUNCS = {
    "G1": gate_g1_weights,
    "G2": gate_g2_canvas,
    "G3": gate_g3_contract,
    "G4": gate_g4_admission,
    "G5": gate_g5_audio_law,
    "G6": gate_g6_guards,
    "G7": gate_g7_surface,
}


def evaluate(gate: str, name: str) -> tuple:
    """``(state, detail)`` for one cell. State is one of
    ``pass`` / ``exempt`` / ``expected_red`` / ``RED``."""
    if (name, gate) in EXEMPTIONS:
        return ("exempt", EXEMPTIONS[(name, gate)])
    eng = vreg.get_engine(name)
    failures = GATE_FUNCS[gate](name, eng)
    if not failures:
        if (name, gate) in EXPECTED_RED:
            return ("unexpected_pass", EXPECTED_RED[(name, gate)])
        return ("pass", "")
    detail = "; ".join(failures)
    if (name, gate) in EXPECTED_RED:
        return ("expected_red", "%s || live failure: %s"
                % (EXPECTED_RED[(name, gate)], detail))
    return ("RED", detail)


# ---------------------------------------------------------------------------
# The seven gate tests. One per gate, over the whole live roster, so a failure
# names every lane the gate caught rather than stopping at the first.
# ---------------------------------------------------------------------------
def _run_gate(gate: str):
    reds, unexpected_passes = [], []
    for name in ENGINE_NAMES:
        state, detail = evaluate(gate, name)
        if state == "RED":
            reds.append("  %-24s %s" % (name, detail))
        elif state == "unexpected_pass":
            unexpected_passes.append("  %-24s %s" % (name, detail))
    msgs = []
    if reds:
        msgs.append(
            "%s (%s) -- these lanes FAIL the gate and are not in "
            "EXPECTED_RED:\n%s\nFix the lane, or -- if this is a known defect "
            "owned by a later lane packet -- add it to EXPECTED_RED with its "
            "S8b defect id and owning lane."
            % (gate, GATE_TITLES[gate], "\n".join(reds)))
    if unexpected_passes:
        msgs.append(
            "%s (%s) -- these lanes now PASS but are still listed in "
            "EXPECTED_RED:\n%s\nDelete those entries. A stale expected-red "
            "rots the ledger into a rubber stamp and hides the next real "
            "regression behind it."
            % (gate, GATE_TITLES[gate], "\n".join(unexpected_passes)))
    assert not msgs, "\n\n".join(msgs)


def test_g1_weights_resolve():
    _run_gate("G1")


def test_g2_canvas_truth():
    _run_gate("G2")


def test_g3_contract_matches_runtime():
    _run_gate("G3")


def test_g4_admission_honesty():
    _run_gate("G4")


def test_g5_audio_law_self_probe():
    _run_gate("G5")


def test_g6_guards_fire_early():
    _run_gate("G6")


def test_g7_public_surface():
    _run_gate("G7")


# ---------------------------------------------------------------------------
# Guards on the guard.
# ---------------------------------------------------------------------------
def test_registry_roster_is_intact():
    """A vanished lane is invisible from inside the registry (lesson L5), so
    the independent CAPABILITIES table is what proves the roster whole."""
    audit = vreg.audit_engine_roster()
    assert audit == {"missing": (), "unexpected": ()}, (
        "registry roster drift -- missing means a swallowed adapter import "
        "(the lane silently left the dropdown): %r" % (audit,))
    assert len(ENGINE_NAMES) >= 25, (
        "only %d engines registered; the matrix would be near-vacuous"
        % len(ENGINE_NAMES))


def test_every_exemption_and_expected_red_names_a_live_lane():
    """A table entry for a lane that no longer exists is dead weight that
    silently stops asserting anything."""
    live = set(ENGINE_NAMES)
    stale = sorted(
        "%s/%s" % (n, g) for (n, g) in list(EXEMPTIONS) + list(EXPECTED_RED)
        if n not in live)
    # H3's adapters are declared in SAGE_SENSITIVE before they register; that
    # is a forward declaration, not a table entry, so it is not checked here.
    assert not stale, (
        "EXEMPTIONS / EXPECTED_RED name lanes that are not registered: %s"
        % ", ".join(stale))
    bad_gates = sorted(
        "%s/%s" % (n, g) for (n, g) in list(EXEMPTIONS) + list(EXPECTED_RED)
        if g not in GATES)
    assert not bad_gates, "unknown gate ids: %s" % ", ".join(bad_gates)


def test_no_lane_is_both_exempt_and_expected_red():
    """The two tables answer different questions ('does not apply' versus
    'applies and is not done'). A cell in both is an unreadable claim."""
    both = sorted("%s/%s" % (n, g)
                  for (n, g) in EXPECTED_RED if (n, g) in EXEMPTIONS)
    assert not both, (
        "these cells are declared BOTH exempt and expected-red: %s"
        % ", ".join(both))


def test_evidence_manifest_is_well_formed():
    """Lesson L7: a number without its evidence key is not evidence, and a
    digest of a file nobody ships proves nothing to a reader without it."""
    assert os.path.isfile(MANIFEST_PATH), (
        "the evidence manifest is missing: %s" % MANIFEST_PATH)
    m = _manifest()
    assert m.get("schema") == "otr.video_evidence_manifest/1"
    assert int(m.get("manifest_version", 0)) >= 1
    assert m.get("lab_evidence_commit"), "no lab evidence commit recorded"
    for entry in m.get("entries") or []:
        for key in ("lane", "envelope_key", "qa_verdict", "note", "receipts"):
            assert entry.get(key), (
                "manifest entry %r is missing %r" % (entry.get("envelope_key"),
                                                     key))
        for receipt in entry["receipts"]:
            for key in ("path", "sha256", "present_on_disk",
                        "contained_in_evidence_commit"):
                assert key in receipt, (
                    "receipt %r in %r lacks %r"
                    % (receipt.get("path"), entry.get("envelope_key"), key))


def test_admission_unenforced_reasons_are_sentences_not_placeholders():
    """G4's escape hatch is 'say so IN WORDS'. An empty or one-word reason
    would satisfy the lookup while telling a reader nothing."""
    unenforced = _manifest().get("admission_unenforced") or {}
    thin = sorted(k for k, v in unenforced.items()
                  if len(str(v or "").split()) < 6)
    assert not thin, (
        "these admission-unenforced entries do not say anything a receipt "
        "reader could act on: %s" % ", ".join(thin))
    unknown = sorted(k for k in unenforced if k not in set(ENGINE_NAMES))
    assert not unknown, (
        "admission_unenforced names lanes that are not registered: %s"
        % ", ".join(unknown))


def test_legacy_aliases_resolve_and_never_appear_in_the_menu():
    """G7.1's other half: an old public id must still resolve a saved graph,
    and must NOT render as a second live menu option (two public ids on one
    internal id collapses the bijection at IMPORT time -- lesson L5)."""
    menu = [o for o in vd._video_model_combo() if o != vd.ADD_CUSTOM]
    live = set(ENGINE_NAMES)
    for legacy, internal in pub._LEGACY_ENGINE_ALIASES.items():
        assert pub.resolve_engine_id(legacy) == internal, (
            "legacy alias %r does not resolve to %r" % (legacy, internal))
        assert internal in live, (
            "legacy alias %r points at %r, which is not registered"
            % (legacy, internal))
        assert legacy not in menu, (
            "legacy alias %r appears as a live menu option" % (legacy,))


def test_the_matrix_report_renders():
    """The matrix page's per-row claims and this suite must agree exactly, so
    the suite is able to PRINT the matrix it enforces."""
    lines = ["lane" .ljust(28) + " ".join(g.ljust(4) for g in GATES)]
    for name in ENGINE_NAMES:
        cells = []
        for gate in GATES:
            state, _ = evaluate(gate, name)
            cells.append({"pass": "ok", "exempt": "n/a", "expected_red": "RED*",
                          "unexpected_pass": "??", "RED": "RED"}[state].ljust(4))
        lines.append(name.ljust(28) + " ".join(cells))
    report = "\n".join(lines)
    assert "lane" in report and len(lines) == len(ENGINE_NAMES) + 1
    print("\n" + report)


def test_g3_cannot_be_satisfied_by_a_COMMENT_about_continuity():
    """The guard on G3.3's own reading (lane 12, 2026-08-11).

    G3.3 used to be a substring search for `"continuity="` over the class's
    SOURCE TEXT. That is exactly satisfiable by a COMMENT -- and lanes 10, 11
    and 12 each added a comment explaining why their value is NONE, every one
    of which contains that literal. From that point the gate would have gone
    green for a lane whose real declaration had been deleted, satisfied by the
    paragraph explaining the declaration it no longer had.

    Caught by the post-coding QA pass on lane 12, on a test written minutes
    earlier. The reader is now `declares_continuity_kwarg`, which parses the
    AST -- comments are not nodes.

    Asserted with a class that TALKS about `continuity=` in a comment and a
    docstring while passing nothing, so this test fails the moment the gate
    goes back to reading text.
    """
    class _TalksButDoesNotDeclare:
        """A lane whose docstring mentions continuity=CONTINUITY_NONE."""
        #: continuity=CONTINUITY_NONE -- discussed at length, never passed.
        frame_contract = fc.FrameContract(min_frames=1, max_frames=0, quantum=1,
                                          allow_tail_trim=True)

    class _ActuallyDeclares:
        frame_contract = fc.FrameContract(min_frames=1, max_frames=0, quantum=1,
                                          allow_tail_trim=True,
                                          continuity=fc.CONTINUITY_NONE)

    assert not declares_continuity_kwarg(_TalksButDoesNotDeclare())
    assert declares_continuity_kwarg(_ActuallyDeclares())
    # And the resolved VALUE is identical for both, which is why a value check
    # could never have caught this either -- the default is the same constant.
    assert (_TalksButDoesNotDeclare.frame_contract.continuity
            == _ActuallyDeclares.frame_contract.continuity == fc.CONTINUITY_NONE)


def test_the_directory_clip_audio_law_really_proves_the_frames(tmp_path):
    """The guard on G5's exemption-by-name (lane 10, 2026-08-11).

    G5 is lexical, so teaching it that `validate_directory_clip` carries the
    audio law for a directory-clip lane buys a green row for a STRING. If the
    proof behind that string ever rots -- someone drops the magic-byte read
    back to an extension check, say -- the gate would keep saying "silence
    proved" about a directory that could hold anything. That is precisely how a
    ledger becomes a rubber stamp, so the named function is exercised here:
    a file called `0001.png` whose bytes are not a PNG must be REFUSED.

    Mutation-checked by construction: revert `prove_frame_is_a_silent_image` to
    trusting the extension and this test goes red immediately.
    """
    from nodes._otr_video_engines import directory_clip as dc

    for lane, fn_name in DIRECTORY_CLIP_AUDIO_LAW.items():
        assert lane in ENGINE_NAMES, lane
        fn = getattr(dc, fn_name, None)
        assert callable(fn), (
            "G5 accepts %r as %s's audio law, but "
            "nodes/_otr_video_engines/directory_clip.py defines no such "
            "function -- the gate would be matching a string that means "
            "nothing" % (fn_name, lane))

    d = tmp_path / "frames"
    d.mkdir()
    # A REAL frame: the PNG signature, so the honest case still passes.
    (d / "0001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    clip = {"type": "directory", "path": str(d), "pixel_format": "rgba",
            "alpha": "straight", "has_audio": False, "frame_count": 1}
    assert len(dc.validate_directory_clip(clip, expect_frames=1)) == 1

    # The same directory with an IMPOSTOR: named .png, containing an mp4's
    # ftyp box. Nothing about the name changed; everything about the proof did.
    (d / "0002.png").write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)
    clip["frame_count"] = 2
    with pytest.raises(ValueError) as excinfo:
        dc.validate_directory_clip(clip, expect_frames=2)
    assert "0002.png" in str(excinfo.value)


def test_a_halving_two_stage_lane_declares_a_64_legal_canvas():
    """L13 -- S8b-10 is a SHARED-MECHANISM defect, not one lane's.

    Any adapter whose graph halves the canvas for a first stage and upsamples
    with `LTXVLatentUpsampler` inherits it, because that node takes `samples` /
    `upscale_model` / `vae` and NO target size -- its whole contract is "x2".
    So the delivered canvas IS 2x the stage-A base, and stage A is /32-legal
    only when the full canvas is /64 on BOTH axes.

    It was recorded against `ltx_audio_in` in lane 7 and was live in
    `eng_ltx_video`'s HQ two-stage path the whole time (416x240 at its declared
    832x480). It hid because 1472x832 -- the old landscape default -- is also
    /64, so the path was legal until the lane moved to 832x480 and nobody
    rechecked the geometry against the new canvas.

    Generic over the registry on purpose: a future adapter that adopts the
    halve-then-x2 idiom is covered when it lands, not when someone remembers
    this file exists.
    """
    offenders = []
    for name in ENGINE_NAMES:
        eng = vreg.get_engine(name)
        src = _mro_source(type(eng))
        halves = ("// 2" in src or "//2" in src)
        fixed_x2 = "LTXVLatentUpsampler" in src
        if not (halves and fixed_x2):
            continue
        canvas = getattr(eng, "render_canvas", None)
        if not canvas:
            offenders.append(
                "%s halves its canvas and upsamples x2 but DECLARES NO CANVAS, "
                "so its stage-A legality depends on whatever the driver hands "
                "it" % name)
            continue
        w, h = int(canvas[0]), int(canvas[1])
        if w % 64 or h % 64:
            offenders.append(
                "%s declares %dx%d, which is not /64 on both axes, so its "
                "stage-A latent (%dx%d) is not /32-legal -- and snapping the "
                "base would deliver a different canvas than the declaration"
                % (name, w, h, w // 2, h // 2))
    assert not offenders, (
        "halve-then-fixed-x2 lanes must declare a /64 canvas (lesson L13):\n  "
        + "\n  ".join(offenders))
