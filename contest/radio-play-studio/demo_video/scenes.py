"""Radio Play Studio demo -- the motion-graphics scenes, built in Blender.

Run headless, one scene per call:
  blender -b --factory-startup --python scenes.py -- <scene> [--test] [--frames a-b]

Scenes: type, title, cards, nodes, ring, end. Output: renders/<scene>/####.png at
1920x1080, 30 fps. --test renders 3 frames at 480x270 on the CPU (Cycles) so the
GPU stays free for an episode render; a full render uses EEVEE on the GPU.

Motion after Comfy's launch video for the challenge (typewriter code, 3D rotating
type, app UI on panels), in the operator's own palette: black and purple backdrops,
yellow headlines, aquamarine secondary type. Fonts are Windows system fonts
(Montserrat Bold, Consolas, Bahnschrift); no Comfy logo or font is used.
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
REPO = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")
WORKFLOW = REPO / "workflows" / "otr_16gb_foley_open_flux3_ref.json"
FPS = 30
# Palette (operator 10-21): "purple and aquamarine and black and yellow"; no orange.
NIGHT = (0.035, 0.020, 0.060)     # black with a purple cast
YELLOW = (1.0, 0.850, 0.120)      # headlines, card edges, sound bars
AQUA = (0.300, 1.0, 0.800)        # secondary type, links
PURPLE = (0.100, 0.050, 0.180)    # the app and node scenes' backdrop
PANEL = (0.220, 0.120, 0.360)
FONT_HEAVY = r"C:\Windows\Fonts\Montserrat-Bold.ttf"    # Impact read as blocky
FONT_MONO = r"C:\Windows\Fonts\consola.ttf"
FONT_UI = r"C:\Windows\Fonts\bahnschrift.ttf"

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SCENE = argv[0] if argv else "title"
TEST = "--test" in argv
FRAMES = None
if "--frames" in argv:
    a, b = argv[argv.index("--frames") + 1].split("-")
    FRAMES = (int(a), int(b))
CUES = json.loads((HERE / "narration" / "lengths.json").read_text(encoding="utf-8"))


def srgb(c):
    """sRGB 0-1 to linear, so the brand colours render as themselves."""
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


# --------------------------------------------------------------------------- setup
def reset(seconds, bg):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, int(round(seconds * FPS))
    sc.render.resolution_x, sc.render.resolution_y = (960, 540) if TEST else (1920, 1080)
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"
    if TEST:
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = 8
    else:
        for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            try:
                sc.render.engine = eng
                break
            except TypeError:
                continue
        try:
            sc.eevee.taa_render_samples = 32
        except AttributeError:
            pass
    world = bpy.data.worlds.new("bg")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (*srgb(bg), 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    return sc


def mat(name, color, emit=0.0, rough=0.45, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*srgb(color), 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*srgb(color), 1)
        bsdf.inputs["Emission Strength"].default_value = emit
    return m


def image_mat(name, path, emit=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(path))
    if str(path).lower().endswith((".mp4", ".mov", ".mkv")):
        img.source = "MOVIE"
        tex.image_user.use_auto_refresh = True
        tex.image_user.frame_duration = 100000
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = emit
    bsdf.inputs["Roughness"].default_value = 0.6
    return m, img


def text(body, size=1.0, font=FONT_HEAVY, extrude=0.0, bevel=0.0, material=None,
         align="CENTER", loc=(0, 0, 0), rot=(0, 0, 0), name=None):
    cu = bpy.data.curves.new(name or "txt", type="FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(font, check_existing=True)
    cu.size = size
    cu.extrude = extrude
    cu.bevel_depth = min(bevel, 0.008)
    cu.space_character = 1.06 if font == FONT_HEAVY else 1.0
    cu.align_x = align
    cu.align_y = "CENTER"
    ob = bpy.data.objects.new(name or "txt", cu)
    bpy.context.collection.objects.link(ob)
    ob.location, ob.rotation_euler = loc, rot
    if material:
        ob.data.materials.append(material)
    return ob


def plane(w, h, material, loc=(0, 0, 0), rot=(0, 0, 0), name="plane"):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (w, h, 1)
    ob.data.materials.append(material)
    return ob


def box(w, h, d, material, loc=(0, 0, 0), name="box"):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (w, h, d)
    ob.data.materials.append(material)
    return ob


def camera(loc, target, lens=50):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.collection.objects.link(cam)
    cam.data.lens = lens
    cam.location = loc
    tgt = bpy.data.objects.new("target", None)
    bpy.context.collection.objects.link(tgt)
    tgt.location = target
    c = cam.constraints.new("TRACK_TO")
    c.target, c.track_axis, c.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
    bpy.context.scene.camera = cam
    return cam, tgt


def lights(strength=4.0):
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = strength
    sun.rotation_euler = (math.radians(50), math.radians(-20), math.radians(30))
    bpy.context.collection.objects.link(sun)
    fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "SUN"))
    fill.data.energy = strength * 0.35
    fill.rotation_euler = (math.radians(60), math.radians(40), math.radians(-120))
    bpy.context.collection.objects.link(fill)


def key(ob, path, frame, value):
    """Set ``ob.path`` and keyframe it (Blender's default Bezier easing)."""
    setattr(ob, path, value)
    ob.keyframe_insert(path, frame=frame)


def pop(ob, frame, dur=8, final=(1, 1, 1)):
    """Scale from nothing to full with a small overshoot."""
    key(ob, "scale", frame - 1, (0.0, 0.0, 0.0))
    key(ob, "scale", frame, (0.001, 0.001, 0.001))
    key(ob, "scale", frame + dur, tuple(v * 1.12 for v in final))
    key(ob, "scale", frame + dur + 5, final)


def typewriter(ob, full, start, cps=22, cursor="_"):
    """Reveal ``full`` one character at a time from ``start`` (frame handler)."""
    def handler(scene, *_):
        n = max(0, int((scene.frame_current - start) * cps / FPS))
        blink = cursor if (scene.frame_current // 15) % 2 == 0 else " "
        ob.data.body = full[:n] + (blink if cursor else "")
    bpy.app.handlers.frame_change_pre.append(handler)


def render(sc):
    out = HERE / "renders" / (SCENE + ("_test" if TEST else ""))
    out.mkdir(parents=True, exist_ok=True)
    sc.render.filepath = str(out) + os.sep
    if TEST:
        n = sc.frame_end
        for f in (max(1, n // 8), n // 2, n - 2):
            sc.frame_set(f)
            sc.render.filepath = str(out / f"{f:04d}.png")
            bpy.ops.render.render(write_still=True)
        return
    if FRAMES:
        sc.frame_start, sc.frame_end = FRAMES
    bpy.ops.render.render(animation=True)


# --------------------------------------------------------------------------- scenes
def scene_type():
    """1 -- black, a typewriter command line, a blinking cursor."""
    sc = reset(3.0, (0, 0, 0))
    camera((0, 0, 10), (0, 0, 0), lens=50)
    t = text("", size=0.42, font=FONT_MONO, material=mat("yellow", YELLOW, emit=3.0),
             align="LEFT", loc=(-3.6, 0.0, 0))
    typewriter(t, "> radio_play_studio --one-press", start=4, cps=24)
    return sc


def scene_title():
    """2 -- RADIO PLAY STUDIO rotates in, lightly extruded; the promise types on."""
    sc = reset(max(7.5, CUES["lengths"]["s2"] + 0.4), NIGHT)
    camera((0, -0.6, 11), (0, 0, 0), lens=45)
    lights(3.5)
    rake = bpy.data.objects.new("rake", bpy.data.lights.new("rake", "SUN"))
    rake.data.energy = 6.0
    rake.rotation_euler = (math.radians(15), math.radians(-75), 0)
    bpy.context.collection.objects.link(rake)
    yellow = mat("yellow", YELLOW, emit=0.9, rough=0.35)    # lit on every face, never a dark side
    title = text("RADIO PLAY STUDIO", size=0.85, extrude=0.06, bevel=0.03, material=yellow,
                 loc=(0, 0.55, 0), name="title")
    for ob in (title,):
        key(ob, "rotation_euler", 1, (0, math.radians(-100), 0))
        key(ob, "rotation_euler", 22, (0, math.radians(9), 0))
        key(ob, "rotation_euler", 34, (0, 0, 0))
        key(ob, "rotation_euler", sc.frame_end, (math.radians(4), math.radians(-14), 0))
    sub = text("", size=0.5, font=FONT_HEAVY, material=mat("aqua", AQUA, emit=1.5),
               loc=(0, -1.05, 0.2))
    typewriter(sub, "ONE PRESS. A WHOLE STORY.", start=40, cps=26, cursor="")
    return sc


def scene_cards():
    """3 -- dark purple; the real app pages arrive as tilted 3D panels."""
    seconds = CUES["lengths"]["s3"] + 0.7
    sc = reset(seconds, PURPLE)
    camera((0, -1.0, 12), (0, 0, 0), lens=40)
    lights(3.0)
    shots = sorted((HERE / "shots").glob("0[1-9]_*.png"))
    calls = {"01_cards": "PICK A STORY", "02_cast_cards": "YOUR OWN CAST",
             "03_progress": "LEARNS YOUR GPU"}
    span = sc.frame_end / max(1, len(shots))
    yellow = mat("yellow", YELLOW, emit=2.5)
    for i, shot in enumerate(shots):
        m, _ = image_mat(f"shot{i}", shot, emit=1.0)
        f0 = int(i * span) + 1
        f1 = int((i + 1) * span)
        p = plane(9.6, 5.4, m, name=f"panel{i}")
        side = -1 if i % 2 else 1
        key(p, "location", f0, (side * 14, 0.2, -2))
        key(p, "rotation_euler", f0, (math.radians(10), math.radians(side * -40), 0))
        key(p, "location", f0 + 18, (side * -1.1, 0.2, 0))
        key(p, "rotation_euler", f0 + 18, (math.radians(6), math.radians(side * 14), 0))
        key(p, "location", f1 - 10, (side * -1.6, 0.2, 0.4))
        key(p, "rotation_euler", f1 - 10, (math.radians(4), math.radians(side * 8), 0))
        key(p, "location", f1, (side * -16, 0.2, -2))
        call = calls.get(shot.stem, "RADIO PLAY STUDIO")
        word = text(call, size=0.55, extrude=0.04, material=yellow,
                    loc=(side * 1.0, -1.9, 3.0))
        word.rotation_euler = (math.radians(12), 0, math.radians(side * 3))
        # A dark plate behind the call, so it reads over the screenshot.
        plate = plane(len(call) * 0.42 + 0.6, 0.9, mat(f"call_plate{i}", PANEL, emit=0.2),
                      loc=(0, 0, -0.06))
        plate.parent = word
        for ob in (word, plate):
            ob.visible_shadow = False         # no wedge of shadow across the screenshot
        pop(word, f0 + 20)
        key(word, "scale", f1 - 6, (1, 1, 1))
        key(word, "scale", f1, (0, 0, 0))
    return sc


def node_cards(px=1 / 230.0):
    """The real workflow, laid out as 3D cards; returns {node id: (x, y, w, h)}."""
    g = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    card_m = mat("card", PANEL, emit=0.15, rough=0.7)
    edge_m = mat("edge", YELLOW, emit=4.0)
    label_m = mat("label", AQUA, emit=1.2)
    where = {}
    for n in g["nodes"]:
        if n["type"] in ("Note", "MarkdownNote"):
            continue
        w = max(1.8, n["size"][0] * px)
        h = max(0.9, min(n["size"][1] * px, 3.2))
        x = n["pos"][0] * px + w / 2
        y = -n["pos"][1] * px - h / 2
        c = box(w, h, 0.12, card_m, loc=(x, y, 0), name=f"node{n['id']}")
        e = box(w + 0.06, h + 0.06, 0.13, edge_m, loc=(x, y, -0.01), name=f"edge{n['id']}")
        mod = e.modifiers.new("wire", "WIREFRAME")
        mod.thickness = 0.04
        title = n.get("title") or n["type"]
        text(title.upper(), size=0.28, font=FONT_UI, material=label_m, align="LEFT",
             loc=(x - w / 2 + 0.15, y + h / 2 - 0.3, 0.1))
        where[n["id"]] = (x, y, w, h)
    link_m = mat("link", AQUA, emit=2.5)
    order = []
    for _lid, src, _ss, dst, _ds, _typ in g["links"]:
        if src not in where or dst not in where:
            continue
        sx, sy, sw, _ = where[src]
        dx, dy, dw, _ = where[dst]
        a = (sx + sw / 2, sy, 0.05)
        b = (dx - dw / 2, dy, 0.05)
        cu = bpy.data.curves.new("link", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.025
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(1)
        p0, p1 = sp.bezier_points
        p0.co, p1.co = a, b
        reach = max(1.0, abs(b[0] - a[0]) * 0.5)
        p0.handle_right = (a[0] + reach, a[1], a[2])
        p0.handle_left = (a[0] - reach, a[1], a[2])
        p1.handle_left = (b[0] - reach, b[1], b[2])
        p1.handle_right = (b[0] + reach, b[1], b[2])
        ob = bpy.data.objects.new("link", cu)
        bpy.context.collection.objects.link(ob)
        ob.data.materials.append(link_m)
        order.append((a[0], cu))
    return where, order


def scene_nodes():
    """5 -- the 3D walk through the real graph; each model named on cue."""
    starts = CUES["sentence_starts"]["s5"]
    lead = 1.5
    seconds = lead + CUES["lengths"]["s5"] + 1.5
    sc = reset(seconds, PURPLE)
    lights(2.5)
    floor = plane(200, 200, mat("floor", (0.05, 0.025, 0.09), rough=0.9), loc=(10, -6, -0.2))
    where, links = node_cards()
    # Links draw themselves in, left to right, over the opening overview.
    for x, cu in sorted(links, key=lambda t: t[0]):
        f = 1 + int(x * 2.2)
        cu.bevel_factor_end = 0.0
        cu.keyframe_insert("bevel_factor_end", frame=f)
        cu.bevel_factor_end = 1.0
        cu.keyframe_insert("bevel_factor_end", frame=f + 18)
    xs = [v[0] for v in where.values()]
    ys = [v[1] for v in where.values()]
    mid = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    cam, tgt = camera((mid[0], mid[1] - 16, 26), (mid[0], mid[1], 0), lens=32)
    f = lambda s: int(round((lead + s) * FPS))
    # (node id, model label, sentence index, what flies out)
    stops = [(1, "GEMMA 4 12B", 1, "page"), (81, "KOKORO", 2, "bars"),
             (83, "STABLE AUDIO 3", 3, "bars"), (91, "FLUX 3 KLEIN", 4, "stills"),
             (92, "LTX 2.5", 5, "video"), (84, "1080P \u00b7 FFMPEG", 6, "frame")]
    key(cam, "location", 1, (mid[0], mid[1] - 16, 26))
    key(tgt, "location", 1, (mid[0], mid[1], 0))
    key(cam, "location", f(starts[1]) - 40, (mid[0] - 4, mid[1] - 14, 20))
    key(tgt, "location", f(starts[1]) - 40, (mid[0] - 4, mid[1], 0))
    yellow = mat("yellow", YELLOW, emit=1.0, rough=0.35)
    stills = sorted(HERE.joinpath("footage", "stills").glob("*.png"))[:6]
    video = next(iter(sorted(HERE.joinpath("footage").glob("*.mp4"))), None)
    for i, (nid, label, sent, fly) in enumerate(stops):
        if nid not in where:
            continue
        x, y, w, h = where[nid]
        at = f(starts[sent])
        key(cam, "location", at - 6, (x - 1.5, y - 9.5, 8.5))
        key(tgt, "location", at - 6, (x + 1.0, y, 0.8))
        if fly == "video":              # orbit the hero slab while the clip plays
            key(cam, "location", at + 40, (x + 4.5, y - 6.5, 5.0))
            key(tgt, "location", at + 40, (x + 1.5, y - 0.8, 3.0))
        end = f(starts[sent + 1]) - 8 if sent + 1 < len(starts) else sc.frame_end - 10
        key(cam, "location", end, (x + 0.5, y - 9.0, 8.0))
        key(tgt, "location", end, (x + 2.0, y, 0.8))
        tag = text(label, size=0.7, extrude=0.04, material=yellow,
                   loc=(x, y + h / 2 + 0.9, 1.4), rot=(math.radians(70), 0, 0))
        plate = plane(len(label) * 0.6 + 0.8, 1.15, mat(f"plate{i}", (0.06, 0.04, 0.08)),
                      loc=(0, 0, -0.06))
        plate.parent = tag
        pop(tag, at)
        for ob in [tag] + (_fly(fly, x, y, at, end, stills, video, i) or []):
            held = tuple(ob.scale)
            key(ob, "scale", end - 4, held)
            key(ob, "scale", end + 4, (0.0, 0.0, 0.0))
    key(cam, "location", sc.frame_end, (x + 3.0, y - 8.0, 8.0))
    return sc


def _fly(kind, x, y, at, end, stills, video, i):
    """What comes out of a node: a script page, sound bars, stills, the clip.
    Returns the objects made, so the stop can clear them when its line ends."""
    made = []
    if kind == "page":
        page = plane(1.6, 2.1, mat("paper", (0.96, 0.94, 0.88), emit=0.8), loc=(x, y, 0.2),
                     rot=(math.radians(70), 0, 0))
        ink = mat("ink", (0.1, 0.1, 0.1))
        for r, line in enumerate(("ACT ONE", "JEFFREY: I can't believe", "MAGNOLIAREX: Today",
                                  "belongs to you.", "(The festival roars.)")):
            t = text(line, size=0.13, font=FONT_UI, material=ink, align="LEFT",
                     loc=(-0.65, 0.75 - r * 0.28, 0.01))
            t.parent = page
        key(page, "location", at, (x, y, 0.2))
        key(page, "location", at + 18, (x + 1.8, y - 0.8, 2.4))
        pop(page, at, final=(1.6, 2.1, 1))
        made.append(page)
    elif kind == "bars":
        rnd = random.Random(i)
        for k in range(16):
            b = box(0.12, 0.12, 1.0, mat(f"bar{i}{k}", YELLOW, emit=3.0),
                    loc=(x - 0.9 + k * 0.13, y + 0.5, 0.8))
            for s in range(0, max(6, end - 4 - at), 6):     # stop before the clear
                hgt = 0.2 + rnd.random() * 1.0
                key(b, "scale", at + s, (0.12, 0.12, hgt if s else 0.01))
            made.append(b)
    elif kind == "stills" and stills:
        for k, s in enumerate(stills[:3]):
            m, _ = image_mat(f"still{k}", s)
            p = plane(1.92, 1.08, m, loc=(x, y, 0.2), rot=(math.radians(70), 0, 0))
            key(p, "location", at + k * 8, (x, y, 0.2))
            # Forward and low, so the stills sit under the model's name, not over it.
            key(p, "location", at + k * 8 + 20, (x - 2.2 + k * 2.2, y - 2.0, 0.9 + k * 0.1))
            key(p, "rotation_euler", at + k * 8 + 20, (math.radians(72), math.radians(-12 + k * 12), 0))
            pop(p, at + k * 8, final=(1.92, 1.08, 1))
            made.append(p)
    elif kind in ("video", "frame"):
        src = video if (kind == "video" and video) else (stills[-1] if stills else None)
        if src is None:
            return
        hero = kind == "video"          # agy r1: LTX is the hero beat -- bigger, higher
        w, h = (5.76, 3.24) if hero else (3.84, 2.16)
        m, _ = image_mat(f"{kind}{i}", src)
        p = plane(w, h, m, loc=(x, y, 0.2), rot=(math.radians(72), 0, 0))
        key(p, "location", at, (x, y, 0.2))
        key(p, "location", at + 20, (x + 1.5, y - 0.8, 3.2 if hero else 2.6))
        pop(p, at, final=(w, h, 1))
        made.append(p)
    return made


def scene_ring():
    """6 -- the claim, then every open model and tool on a turning 3D ring."""
    seconds = max(13.0, CUES["lengths"]["s6"] + 1.8)
    sc = reset(seconds, NIGHT)
    camera((0, -2.0, 13), (0, 0, 0), lens=40)
    lights(3.5)
    yellow = mat("yellow", YELLOW, emit=0.9, rough=0.35)
    aqua = mat("aqua", AQUA, emit=1.2)
    a = text("OTHER TOOLS MAKE CLIPS", size=0.6, extrude=0.05, material=aqua, loc=(0, 1.0, 0))
    b = text("THIS MAKES A WHOLE STORY", size=0.68, extrude=0.06, bevel=0.02, material=yellow,
             loc=(0, -0.4, 0))
    pop(a, 5)
    pop(b, 50)
    third = int(sc.frame_end * 0.45)
    for ob in (a, b):
        key(ob, "location", third - 1, tuple(ob.location))
        key(ob, "location", third + 12, (ob.location.x, ob.location.y + 9, ob.location.z))
    head = text("OPEN WEIGHTS ONLY", size=0.9, extrude=0.06, bevel=0.02, material=yellow,
                loc=(0, 2.6, 0.5))
    pop(head, third + 10)
    models = ["GEMMA 4 12B", "FLUX 3 KLEIN", "LTX 2.5", "KOKORO", "STABLE AUDIO 3"]
    tools = ["FFMPEG", "COMFYUI", "OLD-TIME RADIO PACK", "COMFY SDK", "COMFY-API-PROXY"]
    for col, (names, m, x0) in enumerate(((models, yellow, -2.9), (tools, aqua, 2.9))):
        for k, n in enumerate(names):
            t = text(n, size=0.5, extrude=0.03, material=m, loc=(x0, 1.15 - k * 0.8, 0.2))
            pop(t, third + 16 + col * 4 + k * 3)
    return sc


def scene_end():
    """7 -- the title settles; link, hashtag, byline."""
    sc = reset(7.0, NIGHT)
    camera((0, -0.4, 11), (0, 0, 0), lens=45)
    lights(3.5)
    yellow = mat("yellow", YELLOW, emit=0.9, rough=0.35)
    aqua = mat("aqua", AQUA, emit=1.3)
    t = text("RADIO PLAY STUDIO", size=0.8, extrude=0.06, bevel=0.03, material=yellow,
             loc=(0, 1.0, 0))
    key(t, "rotation_euler", 1, (0, math.radians(30), 0))
    key(t, "rotation_euler", 30, (0, 0, 0))
    text("github.com/jbrick2070/ComfyUI-OldTimeRadio", size=0.36, font=FONT_UI,
         material=aqua, loc=(0, -0.35, 0.1))
    h = text("#OPENSOURCEHIGGSFIELD", size=0.55, extrude=0.03, material=aqua, loc=(0, -1.2, 0.1))
    pop(h, 24)
    by = text("BY JEFFREY A. BRICK", size=0.42, material=aqua, loc=(0, -2.05, 0.1))
    pop(by, 40)
    return sc


BUILD = {"type": scene_type, "title": scene_title, "cards": scene_cards,
         "nodes": scene_nodes, "ring": scene_ring, "end": scene_end}
render(BUILD[SCENE]())
print("SCENE DONE", SCENE)
