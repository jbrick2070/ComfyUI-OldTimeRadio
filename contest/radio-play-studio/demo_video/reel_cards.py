"""Radio Play Studio range reel -- the classic bold, blocky 3D lettering.

Run headless, one card per call:
  blender -b --factory-startup --python reel_cards.py -- section "LANGUAGES" OUT_DIR
  blender -b --factory-startup --python reel_cards.py -- badge 1 "ENGLISH" "TWELFTH NIGHT" OUT_DIR
  blender -b --factory-startup --python reel_cards.py -- open OUT_DIR
  blender -b --factory-startup --python reel_cards.py -- end OUT_DIR

``section`` and ``end`` render opaque 1920x1080 frames on black and purple.
``badge`` renders a transparent overlay (PNG with alpha) that slams a big number
and a label in over the top-left of an episode clip, holds, and drops away.
Impact, extruded and bevelled, lit on every face (operator 2026-10-21: the
"black rendered 3D letters" were hard to read), in the reel palette: yellow
type, aquamarine labels, purple plates.
"""
import math
import sys
from pathlib import Path

import bpy

FPS = 30
YELLOW = (1.0, 0.850, 0.120)
AQUA = (0.300, 1.0, 0.800)
NIGHT = (0.035, 0.020, 0.060)
PLATE = (0.160, 0.060, 0.260)
FONT_BLOCK = r"C:\Windows\Fonts\impact.ttf"

argv = sys.argv[sys.argv.index("--") + 1:]
MODE = argv[0]


def srgb(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


def reset(seconds, transparent=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
        e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, int(round(seconds * FPS))
    sc.render.film_transparent = transparent
    sc.view_settings.view_transform = "Standard"   # AgX greys out the yellow
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    world = bpy.data.worlds.new("bg")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (*srgb(NIGHT), 1)
    return sc


def mat(name, color, emit=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*srgb(color), 1)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Emission Color"].default_value = (*srgb(color), 1)
    bsdf.inputs["Emission Strength"].default_value = emit
    return m


def text(body, size, material, font=FONT_BLOCK, extrude=0.12, loc=(0, 0, 0), align="CENTER",
         bevel=0.008):
    cu = bpy.data.curves.new("t", type="FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(font, check_existing=True)
    cu.size = size
    cu.extrude = extrude
    cu.bevel_depth = bevel
    cu.align_x, cu.align_y = align, "CENTER"
    ob = bpy.data.objects.new("t", cu)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    ob.data.materials.append(material)
    return ob


def plane(w, h, material, loc):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
    ob = bpy.context.active_object
    ob.scale = (w, h, 1)
    ob.data.materials.append(material)
    return ob


def camera(loc=(0, 0, 12), lens=45):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.collection.objects.link(cam)
    cam.location = loc
    cam.data.lens = lens
    bpy.context.scene.camera = cam
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 4.5
    sun.data.use_shadow = False      # no ghost of the letters on the plate
    sun.rotation_euler = (math.radians(35), math.radians(-25), 0)
    bpy.context.collection.objects.link(sun)
    return cam


def key(ob, path, frame, value):
    setattr(ob, path, value)
    ob.keyframe_insert(path, frame=frame)


#: Where blocky type comes to rest: tipped toward the light so the extruded
#: sides read as a slab, a shade deeper than the faces -- never black.
REST = (math.radians(14), math.radians(-9), 0)


def slam(ob, start, settle=10, from_rot=(0, math.radians(-95), 0), overshoot=1.12, rest=REST):
    """Rotate in on Y and punch through a little past full size."""
    key(ob, "scale", 1, (0.0, 0.0, 0.0))
    key(ob, "scale", start, (0.0, 0.0, 0.0))
    key(ob, "rotation_euler", start, from_rot)
    key(ob, "scale", start + settle, (overshoot,) * 3)
    key(ob, "rotation_euler", start + settle, (0, math.radians(6), 0))
    key(ob, "scale", start + settle + 5, (1.0,) * 3)
    key(ob, "rotation_euler", start + settle + 5, rest)


def render(sc, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc.render.filepath = str(out / "####")
    bpy.ops.render.render(animation=True)


if MODE == "section":
    word, out_dir = argv[1], argv[2]
    sc = reset(2.2)
    camera()
    big = text(word, 1.5, mat("y", YELLOW, 0.45), extrude=0.22, loc=(0, 0.2, 0))
    slam(big, 2)
    key(big, "rotation_euler", sc.frame_end, (math.radians(3), math.radians(-10), 0))
    bar = plane(0.01, 0.07, mat("a", AQUA, 2.0), loc=(0, -1.2, 0))
    key(bar, "scale", 8, (0.01, 0.07, 1))
    key(bar, "scale", 22, (5.5, 0.07, 1))
    render(sc, out_dir)
elif MODE == "badge":
    num, label, sub, out_dir = argv[1], argv[2], argv[3], argv[4]
    sc = reset(3.0, transparent=True)
    camera()
    # The frame at the badge's depth is about 9.4 x 5.3 units, so this is the
    # top-left corner with a margin.
    x, y = -3.95, 2.0
    n = text(num, 1.35, mat("y", YELLOW, 0.45), extrude=0.18, loc=(x, y - 0.05, 0.2))
    slam(n, 1, settle=7)
    t = text(label, 0.5, mat("y2", YELLOW, 1.0), extrude=0.0, bevel=0.0,
             loc=(x + 0.62, y + 0.14, 0.2), align="LEFT")
    s = text(sub, 0.3, mat("a", AQUA, 1.3), extrude=0.0, bevel=0.0,          # flat: crisp when small
             loc=(x + 0.63, y - 0.33, 0.2), align="LEFT")
    for k, ob in enumerate((t, s)):
        slam(ob, 5 + k * 3, settle=7, from_rot=(math.radians(-80), 0, 0), rest=(0, 0, 0))
    width = max(len(label) * 0.26, len(sub) * 0.125) + 1.35
    back = plane(width, 1.05, mat("p", PLATE, 0.6), loc=(x - 0.45 + width / 2, y - 0.05, 0.0))
    key(back, "scale", 1, (0.0, 1.05, 1))
    key(back, "scale", 6, (width, 1.05, 1))
    for ob in (n, t, s, back):                 # drop away at the end
        held = tuple(ob.scale)
        key(ob, "scale", sc.frame_end - 7, held)
        key(ob, "scale", sc.frame_end, (0.0,) * 3)
    render(sc, out_dir)
elif MODE == "open":
    out_dir = argv[1]
    sc = reset(5.5)
    camera()
    head = text("100% OPEN WEIGHTS", 1.05, mat("y", YELLOW, 0.45), extrude=0.22, loc=(0, 1.85, 0))
    slam(head, 2)
    models = ["GEMMA 4 12B", "FLUX 3 KLEIN", "LTX 2.5", "KOKORO", "STABLE AUDIO 3"]
    for k, name in enumerate(models):
        col, row = k % 3, k // 3
        x = (col - 1) * 3.2 if row == 0 else (col - 0.5) * 3.2
        t = text(name, 0.58, mat("y%d" % k, YELLOW, 0.5), extrude=0.06,
                 loc=(x, 0.45 - row * 0.85, 0))
        slam(t, 20 + k * 6, settle=8)
    tail = text("ON YOUR OWN GPU.  NO CLOUD.  NOTHING TO PAY PER EPISODE.", 0.3,
                mat("a", AQUA, 1.3), extrude=0.0, bevel=0.0, loc=(0, -1.55, 0))
    slam(tail, 60, from_rot=(math.radians(-80), 0, 0), rest=(0, 0, 0))
    render(sc, out_dir)
elif MODE == "end":
    out_dir = argv[1]
    sc = reset(4.0)
    camera()
    a = text("RADIO PLAY STUDIO", 0.95, mat("y", YELLOW, 0.45), extrude=0.22, loc=(0, 1.0, 0))
    slam(a, 2)
    b = text("EVERY LANGUAGE. EVERY STORY. EVERY LOOK.", 0.36, mat("a", AQUA, 1.3), extrude=0.0,
             bevel=0.0, loc=(0, -0.4, 0))
    slam(b, 16, from_rot=(math.radians(-80), 0, 0), rest=(0, 0, 0))
    c = text("#OPENSOURCEHIGGSFIELD", 0.5, mat("y2", YELLOW, 1.0), extrude=0.0, bevel=0.0, loc=(0, -1.5, 0))
    slam(c, 30, from_rot=(math.radians(-80), 0, 0), rest=(0, 0, 0))
    render(sc, out_dir)
print("CARD DONE", MODE)
