"""Build the Radio Play Studio range reel: languages, stories, looks.

Each section opens on a blocky 3D title card; each clip is a few seconds of a
published episode, untouched apart from the trim and a 3D badge (number,
language or kind, title) over its first two seconds, with the episode's own
sound. Cards come from reel_cards.py (Blender). Run: python build_reel.py
Output: <demo_cuts>/radio_play_studio_range.mp4 and a phone copy under 30 MB.
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
OUT_DIR = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\demo_cuts")
WORK = OUT_DIR / "reel"
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
CLIP_S = 6.0

# (section, [(number, badge line, sub line, episode name stem, start seconds)])
SECTIONS = [
    ("LANGUAGES", [
        ("1", "ENGLISH", "TWELFTH NIGHT", "bird_bolts_20261022_034014", 30.0),
        ("2", "ESPAÑOL", "MUCH ADO ABOUT NOTHING", "balthasar_song_es_20261022_033941", 26.5),
        ("3", "FRANÇAIS", "THE FILM ARCHIVE", "red_emotion_fr_20261022_034355", 27.5),
        ("4", "HINDI", "HAMLET", "mist_fear_hi_20261022_034113", 19.5),
        ("5", "JAPANESE", "SIGNAL LOST", "lost_signal_ja_20261022_034530", 55.5),
        ("6", "MANDARIN", "RUDYARD KIPLING", "electric_roar_zh_20261022_055624", 35.5),
    ]),
    ("STORIES", [
        ("1", "SHAKESPEARE", "TWELFTH NIGHT", "bird_bolts_20261022_034014", 51.5),
        ("2", "YOUR OWN STORY", "A LAST DAY IN SEOUL", "seoul_duet_20261021_091839", 300.5),
        ("3", "SCI-FI FROM THE NEWS", "THE VOID IN THE COMPOSITE", "void_composite_20261018_212722", 23.5),
    ]),
    ("LOOKS", [
        ("1", "STORYBOOK ENGRAVING", "FLATLAND", "three_dimensions_20261021_171919", 227.0),
        ("2", "VIDEO ART", "FLATLAND", "three_dimensions_20261021_150516", 79.0),
        ("3", "ARCHIVAL DOCUMENTARY", "FLATLAND", "three_dimensions_20261022_034357", 89.5),
        ("4", "RECURSIVE FRACTAL", "THE ECHO OF THE DRAGON", "dragon_echo_20261009_014109", 43.0),
        ("5", "CARTOON", "THE LAST DUET", "final_duet_20261022_034335", 94.5),
        ("6", "ANIMATEDIFF ON 8 GB", "GUSTO: SIDE B", "side_b_20261021_215830", 103.5),
    ]),
]
EXTRA = WORK / "extra_clips.json"   # late episodes (Mandarin, Japanese) land here


def run(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def blender(*args):
    subprocess.run([BLENDER, "-b", "--factory-startup", "--python",
                    str(HERE / "reel_cards.py"), "--", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def episode(stem):
    found = sorted(OBS.glob(stem + "__*_final.mp4"))
    if not found:
        raise FileNotFoundError(stem)
    return found[0]


def card_part(frames: Path, path: Path):
    n = len(list(frames.glob("*.png")))
    run("-framerate", "30", "-i", str(frames / "%04d.png"), "-f", "lavfi", "-t", f"{n / 30:.3f}",
        "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{n / 30:.3f}", str(path))


def clip_part(video: Path, start: float, badge: Path, path: Path):
    run("-ss", f"{start:.2f}", "-t", f"{CLIP_S:.2f}", "-i", str(video),
        "-framerate", "30", "-i", str(badge / "%04d.png"),
        "-filter_complex",
        "[0:v]fps=30,scale=1920:1080,setsar=1[v0];[v0][1:v]overlay=0:0:eof_action=pass,format=yuv420p[v];"
        f"[0:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:d=0.08,"
        f"afade=t=out:st={CLIP_S - 0.15:.2f}:d=0.15[a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "15", "-c:a", "aac",
        "-b:a", "192k", "-t", f"{CLIP_S:.2f}", str(path))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    sections = [(name, list(clips)) for name, clips in SECTIONS]
    if EXTRA.is_file():
        for row in json.loads(EXTRA.read_text(encoding="utf-8")):
            sections[0][1].append((str(len(sections[0][1]) + 1), *row))
    parts = []
    for s_i, (name, clips) in enumerate(sections):
        frames = WORK / "cards" / f"sec_{s_i}"
        if not frames.is_dir():
            blender("section", name, str(frames))
        part = WORK / f"{s_i:02d}_00_section.mp4"
        card_part(frames, part)
        parts.append(part)
        for c_i, (num, line, sub, stem, start) in enumerate(clips, 1):
            badge = WORK / "cards" / f"badge_{s_i}_{c_i}"
            if not badge.is_dir():
                blender("badge", num, line, sub, str(badge))
            part = WORK / f"{s_i:02d}_{c_i:02d}_clip.mp4"
            clip_part(episode(stem), start, badge, part)
            parts.append(part)
            print(part.name, stem, start)
    end = WORK / "cards" / "end"
    if not end.is_dir():
        blender("end", str(end))
    part = WORK / "99_end.mp4"
    card_part(end, part)
    parts.append(part)
    (WORK / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    out = OUT_DIR / "radio_play_studio_range.mp4"
    run("-f", "concat", "-safe", "0", "-i", str(WORK / "list.txt"), "-c:v", "libx264", "-crf", "17",
        "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out))
    seconds = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
    rate = int(27 * 8 * 1024 / seconds - 128)
    run("-i", str(out), "-vf", "scale=1280:720", "-c:v", "libx264", "-b:v", f"{rate}k",
        "-maxrate", f"{rate}k", "-bufsize", f"{rate * 2}k", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(OUT_DIR / "radio_play_studio_range_phone.mp4"))
    print(json.dumps({"out": str(out), "seconds": round(seconds, 1), "parts": len(parts)}))


if __name__ == "__main__":
    main()
