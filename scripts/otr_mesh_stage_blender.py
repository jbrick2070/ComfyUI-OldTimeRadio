"""otr_mesh_stage_blender.py -- the headless Blender stage (0-E ticket E-3).

Runs INSIDE the pinned portable Blender's bundled python:

    blender --background --factory-startup --python-exit-code 1
            --python otr_mesh_stage_blender.py -- <args>

ONE v1 camera preset (turntable orbit: fixed radius/elevation, one full
revolution, LINEAR interpolation), ONE material mode (WORKBENCH matcap;
Color Attribute when the mesh carries vertex colors), film transparent ->
straight-alpha RGBA PNGs, EXACT ledger frame count (frame_start=1,
frame_end=N inclusive, step=1), fixed thread count. EEVEE is BANNED v1
headless (fail-closed). CYCLES is the v1.5 tier and carries the
determinism pins (fixed seed = request seed, use_animated_seed=False,
fixed samples, adaptive + denoise OFF).

Modes: ``render`` imports the --glb mesh; ``selftest`` (E-6 cube probe)
builds a cube, exports it to GLB, re-imports that GLB (proving the GLB IO
seam), and renders 3 frames -- the SAME stage path as production.

The host (eng_mesh_stage) validates frame count + dims + RGBA AFTER this
exits and publishes atomically; this script renders into the tmp dir it
was given and exits nonzero on ANY exception (a partial render never
publishes). Module scope imports NO bpy so the OTR test suite can import
the pure arg parser. ASCII, UTF-8 no BOM.
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
    p.add_argument("--radius", type=float, default=2.5)
    p.add_argument("--elevation", type=float, default=0.35)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    if args.frames < 1:
        p.error("--frames must be >= 1")
    if args.width < 8 or args.height < 8:
        p.error("--width/--height must be >= 8")
    if args.mode == "render" and not args.glb:
        p.error("--glb is required in render mode")
    return args


def _script_argv():
    """The args after Blender's ``--`` separator (empty when absent)."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


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
    framing assumption)."""
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
    return scale


def _has_vertex_colors(meshes):
    for obj in meshes:
        data = getattr(obj, "data", None)
        if data is not None and len(getattr(data, "color_attributes", ())):
            return True
    return False


def _build_turntable(bpy, radius, elevation, frames):
    """The ONE v1 camera preset: a pivot empty at the origin, the camera
    parented at (radius, elevation), tracking the origin; the pivot turns a
    single full revolution over frames 1..N with LINEAR interpolation."""
    scene = bpy.context.scene
    pivot = bpy.data.objects.new("otr_pivot", None)
    scene.collection.objects.link(pivot)
    cam_data = bpy.data.cameras.new("otr_cam")
    cam = bpy.data.objects.new("otr_cam", cam_data)
    scene.collection.objects.link(cam)
    cam.parent = pivot
    cam.location = (float(radius), 0.0, float(elevation))
    track = cam.constraints.new(type="TRACK_TO")
    track.target = pivot
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    scene.camera = cam
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    pivot.keyframe_insert(data_path="rotation_euler", frame=1)
    pivot.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    pivot.keyframe_insert(data_path="rotation_euler", frame=max(2, frames))
    action = pivot.animation_data.action
    for fcu in action.fcurves:
        for kp in fcu.keyframe_points:
            kp.interpolation = "LINEAR"
    return cam


def _configure_render(bpy, args, vertex_colors):
    scene = bpy.context.scene
    engine = args.render_engine
    if engine == "WORKBENCH":
        scene.render.engine = "BLENDER_WORKBENCH"
        shading = scene.display.shading
        shading.light = "MATCAP"
        shading.color_type = "VERTEX" if vertex_colors else "SINGLE"
        shading.single_color = (0.78, 0.78, 0.78)
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
        _normalize_meshes(bpy, meshes)
        _build_turntable(bpy, args.radius, args.elevation, args.frames)
        _configure_render(bpy, args, _has_vertex_colors(meshes))
        bpy.ops.render.render(animation=True)
        if args.mode == "selftest":
            import os
            try:
                os.remove(glb)                       # frames only in the dir
            except OSError:
                pass
        print("[otr_mesh_stage_blender] OK mode=%s frames=%d %dx%d"
              % (args.mode, args.frames, args.width, args.height))
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)                                  # partial renders never publish


if __name__ == "__main__":
    main()
