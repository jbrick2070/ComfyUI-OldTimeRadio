"""otr_mesh_stage_blender.py -- the headless Blender stage (0-E ticket E-3).

Runs INSIDE the pinned portable Blender's bundled python:

    blender --background --factory-startup --python-exit-code 1
            --python otr_mesh_stage_blender.py -- <args>

ONE v1 camera preset (a BOUNDED hero arc: fixed radius/elevation, a
start angle + a small arc in degrees, LINEAR interpolation), ONE material
mode (WORKBENCH matcap; Color Attribute when the mesh carries vertex
colors), film transparent -> straight-alpha RGBA PNGs, EXACT ledger frame
count (frame_start=1, frame_end=N inclusive, step=1), fixed thread count.
EEVEE is BANNED v1 headless (fail-closed). CYCLES is the v1.5 tier and
carries the determinism pins (fixed seed = request seed,
use_animated_seed=False, fixed samples, adaptive + denoise OFF).

SINGLE-VIEW VERTEX-COLOR PROJECTION (the textured-hero PoC, 2026-06-20):
when a ``--portrait`` image is supplied the stage projects it onto a
per-VERTEX (point-domain) color attribute ``otr_proj`` -- the hero front
(the +X-facing camera-zero view) is painted from the still and rendered by
WORKBENCH ``color_type='VERTEX'``. The GLB stays GEOMETRY-ONLY on disk (the
mesher never writes color; MESHER_VERSION is NOT bumped) -- the projection
is a per-render Blender step. The hero arc is bounded so the unpainted back
never faces camera.

Modes: ``render`` imports the --glb mesh; ``selftest`` (E-6 cube probe)
builds a cube, exports it to GLB, re-imports that GLB (proving the GLB IO
seam), projects a deterministic in-memory portrait, and renders 3 frames --
the SAME stage path as production. The selftest FAILS nonzero unless a
NON-UNIFORM ``otr_proj`` color attribute exists after projection.

ADAPTIVE CAMERA RADIUS (mesh-improve item 2, 2026-06-30): the v1 camera used
a FLAT ``--radius``/``--elevation`` (2.5/0.35) regardless of mesh shape --
for a longest-dim-1.0 tall mesh this filled ~84% of the frame height (a full
top-to-bottom body, no headroom -- the operator's look-QA complaint). The
camera's vertical FOV is now PINNED explicitly (``sensor_fit='VERTICAL'``,
fixed lens/sensor-height -- Blender's own factory numbers, so this is a
determinism pin, not a behavior change) so the FOV is a fixed, known
quantity independent of render aspect. ``--radius``/``--elevation`` now
default to None (omitted): when unset, :func:`adaptive_camera_radius`
computes a distance from the mesh's ACTUAL post-normalize height so the
subject fills a target fraction of the frame (headroom on every mesh shape,
tall or squat), and elevation scales with it to preserve the v1 viewing
angle. An explicit ``--radius``/``--elevation`` still overrides (operator
escape hatch). The camera fix lives HERE, never a compositor crop (a crop
only re-frames pixels already rendered too tight -- it adds no resolution
the render never captured).

The host (eng_mesh_stage) validates frame count + dims + RGBA AFTER this
exits and publishes atomically; this script renders into the tmp dir it
was given and exits nonzero on ANY exception (a partial render never
publishes). Module scope imports NO bpy so the OTR test suite can import
the pure arg parser + the pure projection math. ASCII, UTF-8 no BOM.
"""
from __future__ import annotations

import argparse
import math
import sys


def parse_stage_args(argv):
    """Pure: parse the post-``--`` stage args (CPU-tested in the OTR suite)."""
    p = argparse.ArgumentParser(prog="otr_mesh_stage_blender")
    p.add_argument("--mode", choices=("render", "selftest"), default="render")
    p.add_argument("--glb", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-engine", dest="render_engine",
                   choices=("WORKBENCH", "CYCLES"), default="WORKBENCH")
    #: None (omitted) = ADAPTIVE (mesh-improve item 2): main() computes the
    #: distance/height from the mesh's actual post-normalize size. An
    #: explicit value is the operator escape hatch and is used verbatim.
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--elevation", type=float, default=None)
    #: Fraction of the vertical frame the mesh bbox height should fill when
    #: --radius is adaptive (headroom knob). None = DEFAULT_TARGET_HEIGHT_FRAC.
    p.add_argument("--target-height-frac", dest="target_height_frac",
                   type=float, default=None)
    p.add_argument("--threads", type=int, default=4)
    #: The hero still projected onto the front-facing vertices (optional; the
    #: GLB stays geometry-only, projection is per-render). Empty = flat matcap.
    p.add_argument("--portrait", default="")
    #: v1.1 surface look. DEFAULT "flat" so an OMITTED --surface preserves the
    #: legacy render behavior (flat gray matcap, or project when --portrait is
    #: also passed -- the legacy decal path). "gradient" = the sculpt gradient;
    #: "portrait" = the front-projected still (requires --portrait).
    p.add_argument("--surface", choices=("flat", "gradient", "portrait"),
                   default="flat")
    #: Bounded hero arc (degrees). start-angle is the camera azimuth at frame 1
    #: (0 = the +X front the projection paints); arc-degrees is the sweep. The
    #: arc is CLAMPED <= MAX_ARC_DEGREES so the unpainted back never shows.
    p.add_argument("--start-angle", dest="start_angle", type=float, default=0.0)
    p.add_argument("--arc-degrees", dest="arc_degrees", type=float, default=45.0)
    args = p.parse_args(argv)
    if args.frames < 1:
        p.error("--frames must be >= 1")
    if args.width < 8 or args.height < 8:
        p.error("--width/--height must be >= 8")
    if args.mode == "render" and not args.glb:
        p.error("--glb is required in render mode")
    if args.mode == "render" and args.surface == "portrait" and not args.portrait:
        p.error("--surface portrait requires --portrait in render mode")
    if (args.target_height_frac is not None
            and not (0.0 < args.target_height_frac <= 1.0)):
        p.error("--target-height-frac must be in (0, 1]")
    if args.radius is not None and args.radius <= 0.0:
        p.error("--radius must be > 0")
    return args


def _script_argv():
    """The args after Blender's ``--`` separator (empty when absent)."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


# --------------------------------------------------------------------------- #
# Pure projection + arc math (CPU-tested; NO bpy).
# --------------------------------------------------------------------------- #
#: The hero arc never exceeds this (degrees) -- single-view projection only
#: paints the +X front, so the camera must stay near it (no unpainted back).
MAX_ARC_DEGREES = 45.0

#: The color attribute the projection writes (per-vertex, point domain).
PROJ_ATTR_NAME = "otr_proj"


def clamp_arc_degrees(arc_degrees):
    """Clamp the requested sweep into [-MAX_ARC_DEGREES, +MAX_ARC_DEGREES]
    (pure). The single-view projection only covers the front hemisphere."""
    a = float(arc_degrees)
    if a > MAX_ARC_DEGREES:
        return MAX_ARC_DEGREES
    if a < -MAX_ARC_DEGREES:
        return -MAX_ARC_DEGREES
    return a


def arc_keyframes(frames, start_angle, arc_degrees):
    """Pure: the bounded hero-arc keyframes as ``[(frame, angle_radians), ...]``.
    frames==1 -> ONE keyframe at frame 1 (a still hero shot, no fcurve).
    frames>=2 -> a LINEAR sweep from start_angle to start_angle+arc over
    frames 1..N inclusive. The arc is CLAMPED to +/-MAX_ARC_DEGREES first."""
    n = int(frames)
    if n < 1:
        raise ValueError("arc_keyframes: frames must be >= 1, got %r" % (frames,))
    start_rad = math.radians(float(start_angle))
    sweep_rad = math.radians(clamp_arc_degrees(arc_degrees))
    if n == 1:
        return [(1, start_rad)]
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append((i + 1, start_rad + sweep_rad * t))
    return out


def project_uv(co_y, co_z):
    """Pure single-view orthographic UV for a vertex at normalized object
    coords (longest dim 1.0, centered at origin so coords ~[-0.5, 0.5]). The
    front view looks down -X (camera at +X), so the image plane is (Y, Z):
    u (left->right) from Y, t (top->bottom) from Z. Returns (u, t) clamped to
    [0, 1]."""
    u = 0.5 + float(co_y)
    t = 0.5 - float(co_z)
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return u, t


def sample_image(pixels, width, height, u, t):
    """Pure nearest-neighbour sample of a Blender-style flat RGBA pixel buffer
    (bottom-origin, row-major, length width*height*4). ``u`` is left->right and
    ``t`` is top->bottom in [0, 1]. Returns (r, g, b) floats in [0, 1]."""
    w = int(width)
    h = int(height)
    if w < 1 or h < 1:
        return (0.5, 0.5, 0.5)
    px = int(round(u * (w - 1)))
    # t is top->bottom; Blender buffers are bottom-origin -> flip.
    row_from_bottom = int(round((1.0 - t) * (h - 1)))
    if px < 0:
        px = 0
    elif px > w - 1:
        px = w - 1
    if row_from_bottom < 0:
        row_from_bottom = 0
    elif row_from_bottom > h - 1:
        row_from_bottom = h - 1
    idx = (row_from_bottom * w + px) * 4
    return (float(pixels[idx]), float(pixels[idx + 1]), float(pixels[idx + 2]))


def alpha_bbox_stats(width, height, alpha_top_origin_row_major):
    """Pure: the tightest bounding box of non-transparent (alpha > 0) pixels
    in a TOP-origin, row-major alpha channel of length width*height -- the
    convention a proof script gets from ``PIL.Image.getchannel("A")``
    (opposite of :func:`sample_image`'s Blender-native bottom-origin buffer;
    this function is for MEASURING a rendered PNG frame, not sampling a
    Blender in-memory image).

    Mesh-improve item 2's MEASURABLE headroom contract (kibitz r1 point 2):
    ``height_frac`` is the bbox height as a fraction of the frame height (the
    "how much of the frame does the subject fill" number the camera's
    ``target_frac`` knob controls); ``top_margin_px`` is the empty space
    above the bbox (the headroom the operator asked for). Returns None when
    every pixel is transparent (nothing rendered -- nothing to measure)."""
    w, h = int(width), int(height)
    a = alpha_top_origin_row_major
    top = bottom = left = right = None
    for y in range(h):
        base = y * w
        row_left = row_right = None
        for x in range(w):
            if a[base + x] > 0:
                if row_left is None:
                    row_left = x
                row_right = x
        if row_left is not None:
            if top is None:
                top = y
            bottom = y
            if left is None or row_left < left:
                left = row_left
            if right is None or row_right > right:
                right = row_right
    if top is None:
        return None
    bbox_h = bottom - top + 1
    bbox_w = right - left + 1
    return {
        "bbox": (left, top, right, bottom),
        "bbox_height_px": bbox_h,
        "bbox_width_px": bbox_w,
        "height_frac": bbox_h / float(h),
        "top_margin_px": top,
        "bottom_margin_px": h - 1 - bottom,
    }


#: Vertical sculpt-gradient endpoints (cool light over shadow). WORKBENCH VERTEX
#: shading draws these per-vertex colours; per-poly smooth normals interpolate
#: them into a soft ramp (the v1.1 "basic gradient" mesh texture).
#: v1.3 SATURATED duotone (teal -> navy). v1.2's blue-GREY (0.58,0.64,0.80)/(0.09,0.11,0.20) read as
#: near-white under STUDIO -- the colour was the other half of the "still plaster" problem. Saturated
#: endpoints read as actual COLOUR. Swap these two tuples for amber/green/bronze etc. (operator's call).
GRADIENT_TOP = (0.20, 0.70, 0.78)      # lit top: teal
GRADIENT_BOTTOM = (0.03, 0.08, 0.20)   # base: deep navy


def gradient_color(co_z, top=GRADIENT_TOP, bottom=GRADIENT_BOTTOM):
    """Pure vertical sculpt gradient for a vertex at normalized, CENTERED world
    z (the ``_normalize_meshes`` longest-dim-1.0, origin-centered space, so
    ~[-0.5, 0.5]). ``co_z`` is CLAMPED to [-0.5, 0.5] first (the range is an
    assumption, not a guarantee for every mesh), then lerped bottom->top so the
    top of the form is lighter than the base. Returns (r, g, b) in [0, 1]."""
    z = float(co_z)
    if z < -0.5:
        z = -0.5
    elif z > 0.5:
        z = 0.5
    s = z + 0.5                                   # [0,1]: 0=bottom, 1=top
    return (
        bottom[0] + (top[0] - bottom[0]) * s,
        bottom[1] + (top[1] - bottom[1]) * s,
        bottom[2] + (top[2] - bottom[2]) * s,
    )


# --------------------------------------------------------------------------- #
# Adaptive camera framing (mesh-improve item 2, 2026-06-30; pure, CPU-tested).
# --------------------------------------------------------------------------- #
#: The camera's PINNED lens/sensor (Blender's own factory-default numbers --
#: pinning them just makes the vertical FOV a fixed, known quantity instead of
#: an implicit one that varies with render aspect via sensor_fit='AUTO').
CAMERA_LENS_MM = 50.0
CAMERA_SENSOR_HEIGHT_MM = 24.0
#: Vertical FOV (degrees) at the pinned lens/sensor: 2*atan((sensor/2)/lens).
CAMERA_VERTICAL_FOV_DEG = math.degrees(
    2.0 * math.atan((CAMERA_SENSOR_HEIGHT_MM / 2.0) / CAMERA_LENS_MM))

#: The OLD flat v1 defaults (kept as the ratio that preserves the v1 viewing
#: angle: new_elevation = adaptive_radius * ELEVATION_TO_RADIUS_RATIO).
LEGACY_DEFAULT_RADIUS = 2.5
LEGACY_DEFAULT_ELEVATION = 0.35
ELEVATION_TO_RADIUS_RATIO = LEGACY_DEFAULT_ELEVATION / LEGACY_DEFAULT_RADIUS

#: Target fraction of the vertical frame the mesh bbox height should fill.
#: At the OLD flat radius=2.5 a longest-dim-1.0 (tall) mesh filled ~84% of the
#: frame (a full top-to-bottom body, no headroom) -- the look-QA complaint
#: this item fixes.
DEFAULT_TARGET_HEIGHT_FRAC = 0.62

#: A camera can never sit inside (or touching) the mesh.
MIN_CAMERA_RADIUS = 0.1


def adaptive_camera_radius(mesh_height, *, target_frac=DEFAULT_TARGET_HEIGHT_FRAC,
                           vertical_fov_deg=CAMERA_VERTICAL_FOV_DEG,
                           min_radius=MIN_CAMERA_RADIUS):
    """Pure: the camera distance that frames a mesh of the given (normalized,
    post-``_normalize_meshes``, world-space) height so its bbox fills
    ``target_frac`` of the vertical frame, given the camera's PINNED vertical
    FOV. Mesh-improve item 2 ("MORE HEADROOM"): the v1 stage used a FLAT
    radius regardless of mesh shape, which over-framed TALL meshes (a
    character/announcer, whose longest dim IS the height -> full top-to-
    bottom body). Using the mesh's ACTUAL post-normalize height (not an
    assumed 1.0) makes this shape-aware: a squat/wide mesh (the radio, whose
    longest dim is width) needs a DIFFERENT distance than a tall one. The fix
    lives in the camera distance, never a compositor crop (a crop only
    re-frames pixels already rendered too tight -- it adds no resolution the
    render never captured). Never returns below ``min_radius``."""
    h = float(mesh_height)
    if h <= 0.0:
        h = 1.0
    half_angle_rad = math.radians(float(vertical_fov_deg) * float(target_frac) / 2.0)
    if half_angle_rad <= 0.0:
        return max(float(min_radius), LEGACY_DEFAULT_RADIUS)
    radius = (h / 2.0) / math.tan(half_angle_rad)
    return max(float(min_radius), radius)


# --------------------------------------------------------------------------- #
# Everything below runs inside Blender only (bpy imported lazily).
# --------------------------------------------------------------------------- #
def _clear_scene(bpy):
    """Remove every object the factory startup ships (cube/camera/light) so
    the stage is built from nothing, deterministically."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _import_glb(bpy, path):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("GLB import produced no mesh objects: %r" % path)
    return meshes


def _normalize_meshes(bpy, meshes):
    """bbox-normalize the imported mesh set: center the combined bounds at
    the origin and scale the longest dimension to 1.0 (the camera preset's
    framing assumption). Returns ``(scale, height)`` -- ``height`` is the
    POST-scale Z-extent (mesh-improve item 2: the adaptive camera radius
    needs the mesh's ACTUAL height, not an assumed 1.0, since only a mesh
    whose longest dim IS its height reaches 1.0; a wide/squat mesh is
    shorter)."""
    from mathutils import Vector
    mins = Vector((1e30, 1e30, 1e30))
    maxs = Vector((-1e30, -1e30, -1e30))
    for obj in meshes:
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            mins = Vector(map(min, mins, wc))
            maxs = Vector(map(max, maxs, wc))
    center = (mins + maxs) * 0.5
    extent = max(maxs - mins)
    scale = 1.0 / extent if extent > 1e-9 else 1.0
    for obj in meshes:
        obj.location = (obj.location - center) * scale
        obj.scale = obj.scale * scale
    height = (maxs.z - mins.z) * scale
    return scale, height


def _has_vertex_colors(meshes):
    for obj in meshes:
        data = getattr(obj, "data", None)
        if data is not None and len(getattr(data, "color_attributes", ())):
            return True
    return False


def _new_proj_attr(mesh):
    """Create the per-VERTEX (POINT-domain) ``otr_proj`` FLOAT_COLOR attribute,
    REMOVING any pre-existing one of the same name first (re-entrancy guard: a
    re-imported GLB is fresh per process today, but never collide on the name)."""
    for ca0 in list(getattr(mesh, "color_attributes", ())):
        if getattr(ca0, "name", "") == PROJ_ATTR_NAME:
            try:
                mesh.color_attributes.remove(ca0)
            except Exception:                      # noqa: BLE001 - API variance
                pass
    return mesh.color_attributes.new(name=PROJ_ATTR_NAME, type="FLOAT_COLOR",
                                     domain="POINT")


def _activate_render_color(mesh, ca):
    """Make ``ca`` the ACTIVE + RENDER colour attribute (WORKBENCH VERTEX shading
    draws the render colour attribute). Tolerant of Blender API variance."""
    try:
        mesh.color_attributes.active_color = ca
    except Exception:                              # noqa: BLE001 - API variance
        pass
    for attr in ("active_color_index", "render_color_index"):
        try:
            idx = list(mesh.color_attributes).index(ca)
            setattr(mesh.color_attributes, attr, idx)
        except Exception:                          # noqa: BLE001 - API variance
            pass


def _project_portrait_onto_meshes(bpy, meshes, image):
    """Single-view vertex-color projection: create a per-VERTEX (point-domain)
    ``otr_proj`` color attribute on EVERY imported mesh, set it active+render,
    and paint each vertex from the portrait via the front (Y, Z) projection.
    Returns the count of DISTINCT colors written across all meshes (the
    selftest's non-uniformity gate). Guards zero-vert/zero-poly meshes."""
    pixels = list(image.pixels)
    w, h = int(image.size[0]), int(image.size[1])
    seen = set()
    for obj in meshes:
        mesh = getattr(obj, "data", None)
        if mesh is None:
            continue
        verts = getattr(mesh, "vertices", ())
        if not len(verts):
            continue                              # guard zero-vert meshes
        ca = _new_proj_attr(mesh)
        mw = obj.matrix_world
        for i, v in enumerate(verts):
            co = mw @ v.co                         # normalized, centered coords
            u, t = project_uv(co.y, co.z)
            r, g, b = sample_image(pixels, w, h, u, t)
            ca.data[i].color = (r, g, b, 1.0)
            seen.add((round(r, 4), round(g, 4), round(b, 4)))
        _activate_render_color(mesh, ca)
    return len(seen)


def _paint_gradient_onto_meshes(bpy, meshes):
    """The v1.1 SCULPT GRADIENT: paint a per-VERTEX ``otr_proj`` colour from
    :func:`gradient_color` over the WORLD-space centered height (``matrix_world
    @ v.co`` -- ``_normalize_meshes`` rewrote object loc/scale, so local
    ``v.co.z`` is NOT the centered height). Same vertex-colour mechanism the
    projection uses, so WORKBENCH VERTEX draws it. Returns the distinct-colour
    count. Replaces the flat photo decal as the default surface."""
    seen = set()
    for obj in meshes:
        mesh = getattr(obj, "data", None)
        if mesh is None:
            continue
        verts = getattr(mesh, "vertices", ())
        if not len(verts):
            continue
        ca = _new_proj_attr(mesh)
        mw = obj.matrix_world
        for i, v in enumerate(verts):
            co = mw @ v.co
            r, g, b = gradient_color(co.z)
            ca.data[i].color = (r, g, b, 1.0)
            seen.add((round(r, 4), round(g, 4), round(b, 4)))
        _activate_render_color(mesh, ca)
    return len(seen)


def _smooth_mesh_normals(meshes):
    """Smooth shading via mesh DATA (``poly.use_smooth = True``) -- NO
    ``bpy.ops`` (so it needs no active-object/selection context; headless
    ``--background`` safe). Melts the faceted matcap look with ZERO geometry
    change. ``mesh.update()`` makes headless WORKBENCH pick up the smoothing."""
    for obj in meshes:
        mesh = getattr(obj, "data", None)
        if mesh is None:
            continue
        for poly in getattr(mesh, "polygons", ()):
            poly.use_smooth = True
        try:
            mesh.update()
        except Exception:                          # noqa: BLE001 - API variance
            pass


def _make_selftest_portrait(bpy):
    """A deterministic NON-UNIFORM in-memory portrait for the E-6 selftest (no
    file IO, no PIL): an 8x8 RGBA image with a Y/Z gradient."""
    w = h = 8
    img = bpy.data.images.new("otr_selftest_portrait", width=w, height=h,
                              alpha=True)
    px = []
    for row in range(h):
        for col in range(w):
            px.extend((col / (w - 1), row / (h - 1), 0.5, 1.0))
    img.pixels = px
    return img


def _build_turntable(bpy, radius, elevation, frames, start_angle, arc_degrees):
    """The ONE v1 camera preset: a pivot empty at the origin, the camera
    parented at (radius, elevation), tracking the origin; the pivot sweeps a
    BOUNDED hero arc (start_angle .. start_angle+arc) over frames 1..N with
    LINEAR interpolation. frames==1 -> a single keyframe (no fcurve).

    Mesh-improve item 2 (2026-06-30): the camera's vertical FOV is PINNED
    (sensor_fit='VERTICAL' + fixed lens/sensor-height, matching
    CAMERA_LENS_MM/CAMERA_SENSOR_HEIGHT_MM) so it is a fixed, known quantity
    -- previously Blender's implicit AUTO sensor-fit made the vertical FOV
    vary with the render's aspect ratio, which made the adaptive-radius trig
    ill-defined."""
    scene = bpy.context.scene
    pivot = bpy.data.objects.new("otr_pivot", None)
    scene.collection.objects.link(pivot)
    cam_data = bpy.data.cameras.new("otr_cam")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.lens = CAMERA_LENS_MM
    cam_data.sensor_height = CAMERA_SENSOR_HEIGHT_MM
    cam = bpy.data.objects.new("otr_cam", cam_data)
    scene.collection.objects.link(cam)
    cam.parent = pivot
    cam.location = (float(radius), 0.0, float(elevation))
    track = cam.constraints.new(type="TRACK_TO")
    track.target = pivot
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    scene.camera = cam
    keys = arc_keyframes(frames, start_angle, arc_degrees)
    for frame, angle in keys:
        pivot.rotation_euler = (0.0, 0.0, angle)
        pivot.keyframe_insert(data_path="rotation_euler", frame=frame)
    if pivot.animation_data and pivot.animation_data.action:
        for fcu in pivot.animation_data.action.fcurves:
            for kp in fcu.keyframe_points:
                kp.interpolation = "LINEAR"
    return cam


def _configure_render(bpy, args, vertex_colors):
    scene = bpy.context.scene
    engine = args.render_engine
    if engine == "WORKBENCH":
        scene.render.engine = "BLENDER_WORKBENCH"
        shading = scene.display.shading
        # v1.2 fix: STUDIO, not MATCAP. A MATCAP renders its own baked clay
        # material and IGNORES the per-vertex albedo -> every mesh came out a
        # uniform white-marble blob regardless of the painted gradient. STUDIO
        # lighting shades the ACTUAL albedo (the vertex-colour gradient / single
        # colour), so the gradient reads AND the smooth 3D form gets real
        # light-to-shadow depth. (Deterministic: the factory-startup studio light
        # is fixed; no per-render variance.)
        shading.light = "STUDIO"
        shading.color_type = "VERTEX" if vertex_colors else "SINGLE"
        shading.single_color = (0.62, 0.64, 0.72)
        # v1.3 fix: STUDIO keeps the DIFFUSE 3D form, but its SPECULAR highlight blew the lit side
        # to white plaster (v1.2). Disable specular -> the colour ramp reads, the form still shades.
        try:
            shading.show_specular_highlight = False
        except Exception:                          # noqa: BLE001 - API variance
            pass
        # a touch of cavity shading sharpens the sculpted form (deterministic).
        try:
            shading.show_cavity = True
        except Exception:                          # noqa: BLE001 - API variance
            pass
        print("[otr_mesh_stage_blender] WORKBENCH light=STUDIO specular=off cavity=on "
              "color_type=%s gradient top=%s base=%s"
              % (shading.color_type, GRADIENT_TOP, GRADIENT_BOTTOM))
    elif engine == "CYCLES":
        # v1.5 tier -- determinism pins (0-E spec).
        scene.render.engine = "CYCLES"
        scene.cycles.seed = int(args.seed)
        scene.cycles.use_animated_seed = False
        scene.cycles.samples = 64
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
    else:  # pragma: no cover - argparse already constrains; EEVEE banned v1
        raise RuntimeError("render engine %r is banned for the v1 headless "
                           "stage (EEVEE excluded by spec)" % engine)
    scene.render.resolution_x = int(args.width)
    scene.render.resolution_y = int(args.height)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True            # straight-alpha handoff
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.threads_mode = "FIXED"             # determinism pin
    scene.render.threads = max(1, int(args.threads))
    scene.frame_start = 1
    scene.frame_end = int(args.frames)              # inclusive; EXACT count
    scene.frame_step = 1
    out = args.out.rstrip("/\\")
    scene.render.filepath = out + ("\\" if "\\" in out else "/") + "frame_"


def _selftest_glb(bpy, out_dir):
    """E-6 cube probe: build a cube, export GLB, wipe, hand back the path --
    the re-import below exercises the SAME GLB IO seam production uses."""
    import os
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    glb = os.path.join(out_dir, "_selftest_cube.glb")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB",
                              use_selection=False)
    _clear_scene(bpy)
    return glb


def main():
    import traceback
    try:
        import bpy                                   # Blender-only import
        args = parse_stage_args(_script_argv())
        _clear_scene(bpy)
        glb = args.glb
        if args.mode == "selftest":
            glb = _selftest_glb(bpy, args.out)
        meshes = _import_glb(bpy, glb)
        _scale, mesh_height = _normalize_meshes(bpy, meshes)
        # Surface look. selftest FIRST (unchanged: projects its in-memory
        # portrait + the non-uniformity gate). Render mode is a state machine:
        #   gradient -> sculpt gradient (ignore a stray --portrait, LOUD);
        #   portrait OR (flat + --portrait) -> project the still (the second arm
        #     preserves the legacy omitted-`--surface` + `--portrait` decal);
        #   flat with no portrait -> paint nothing (gray matcap).
        if args.mode == "selftest":
            image = _make_selftest_portrait(bpy)
            distinct = _project_portrait_onto_meshes(bpy, meshes, image)
            if distinct < 2:
                raise RuntimeError(
                    "mesh_stage selftest: projection produced a UNIFORM color "
                    "attribute (%d distinct) -- vertex-color projection is "
                    "broken" % distinct)
        elif args.surface == "gradient":
            if args.portrait:
                print("[otr_mesh_stage_blender] WARN: --portrait ignored under "
                      "--surface gradient")
            _paint_gradient_onto_meshes(bpy, meshes)
        elif args.surface == "portrait" or args.portrait:
            image = bpy.data.images.load(args.portrait)
            _project_portrait_onto_meshes(bpy, meshes, image)
        # else: surface=flat with no portrait -> gray matcap (no vertex colors)
        if args.mode != "selftest":
            _smooth_mesh_normals(meshes)
        # Mesh-improve item 2: ADAPTIVE radius/elevation unless the operator
        # passed an explicit override. Elevation scales with the computed
        # radius by the v1 ratio, preserving today's viewing angle.
        radius = args.radius
        if radius is None:
            target_frac = (args.target_height_frac
                          if args.target_height_frac is not None
                          else DEFAULT_TARGET_HEIGHT_FRAC)
            radius = adaptive_camera_radius(mesh_height, target_frac=target_frac)
        elevation = args.elevation
        if elevation is None:
            elevation = radius * ELEVATION_TO_RADIUS_RATIO
        print("[otr_mesh_stage_blender] camera radius=%.3f elevation=%.3f "
              "mesh_height=%.3f (adaptive=%s)"
              % (radius, elevation, mesh_height, args.radius is None))
        _build_turntable(bpy, radius, elevation, args.frames,
                         args.start_angle, args.arc_degrees)
        _configure_render(bpy, args, _has_vertex_colors(meshes))
        bpy.ops.render.render(animation=True)
        if args.mode == "selftest":
            import os
            try:
                os.remove(glb)                       # frames only in the dir
            except OSError:
                pass
        print("[otr_mesh_stage_blender] OK mode=%s frames=%d %dx%d portrait=%s"
              % (args.mode, args.frames, args.width, args.height,
                 "yes" if (args.mode == "selftest" or args.portrait) else "no"))
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)                                  # partial renders never publish


if __name__ == "__main__":
    main()
