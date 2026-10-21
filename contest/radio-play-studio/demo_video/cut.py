"""Cut the demo: the Blender scenes with their Kokoro narration, the episode
excerpt untouched apart from a corner tag, then one 1920x1080 30 fps MP4.

Run after scenes.py has rendered every scene: python cut.py
Output: ../radio_play_studio_demo.mp4 and a phone copy under 30 MB."""
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "radio_play_studio_demo.mp4"
PHONE = HERE.parent / "radio_play_studio_demo_phone.mp4"
WORK = HERE / "cut_parts"
WORK.mkdir(exist_ok=True)
EXCERPT = next((HERE / "footage" / "excerpt").glob("*.mp4"))
TAG = "UNEDITED OUTPUT \u00b7 3-ACT EPISODE \u00b7 RTX 5080 LAPTOP 16 GB"
LIME = (228, 255, 58, 255)

# (scene, narration line, seconds into the scene the voice starts)
PLAN = [("type", None, 0), ("title", "s2", 0.3), ("cards", "s3", 0.35),
        ("EPISODE", None, 0), ("nodes", "s5", 1.5), ("ring", "s6", 0.4), ("end", "s7", 0.5)]


def run(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def scene_part(name, line, delay, path):
    frames = HERE / "renders" / name / "%04d.png"
    n = len(list((HERE / "renders" / name).glob("*.png")))
    assert n, name
    seconds = n / 30
    if line:
        wav = HERE / "narration" / f"{line}.wav"
        run("-framerate", "30", "-i", str(frames), "-i", str(wav),
            "-filter_complex", f"[1:a]loudnorm=I=-16:TP=-1.5,adelay={int(delay * 1000)}:all=1,aresample=48000,"
            f"apad,atrim=0:{seconds:.3f},aformat=channel_layouts=stereo[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-t", f"{seconds:.3f}", str(path))
    else:
        run("-framerate", "30", "-i", str(frames), "-f", "lavfi", "-t", f"{seconds:.3f}",
            "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k", "-t", f"{seconds:.3f}", str(path))
    return seconds


def tag_png():
    font = ImageFont.truetype(r"C:\Windows\Fonts\impact.ttf", 30)
    img = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = d.textlength(TAG, font=font)
    d.rounded_rectangle((36, 30, 36 + w + 36, 84), radius=10, fill=(33, 25, 39, 210))
    d.text((54, 38), TAG, font=font, fill=LIME)
    path = WORK / "tag.png"
    img.save(path)
    return path


def episode_part(path):
    run("-i", str(EXCERPT), "-i", str(tag_png()), "-filter_complex",
        "[0:v][1:v]overlay=0:0,format=yuv420p[v];[0:a]aresample=48000,"
        "aformat=channel_layouts=stereo[a]", "-map", "[v]", "-map", "[a]", "-r", "30",
        "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-c:a", "aac", "-b:a", "256k",
        str(path))


parts, total = [], 0.0
for i, (name, line, delay) in enumerate(PLAN):
    path = WORK / f"{i:02d}_{name.lower()}.mp4"
    if name == "EPISODE":
        episode_part(path)
    else:
        scene_part(name, line, delay, path)
    total += float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "csv=p=0", str(path)], capture_output=True,
                                  text=True).stdout)
    parts.append(path)
    print(f"{path.name}  running total {total:.1f} s")
(WORK / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
run("-f", "concat", "-safe", "0", "-i", str(WORK / "list.txt"), "-c:v", "libx264", "-crf", "16",
    "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
    "-movflags", "+faststart", str(OUT))            # the episode's own sound stays as rendered
bitrate = int(28 * 8 * 1024 / max(total, 1) - 128)        # kbit/s for ~28 MB
run("-i", str(OUT), "-c:v", "libx264", "-b:v", f"{bitrate}k", "-maxrate", f"{bitrate}k",
    "-bufsize", f"{bitrate * 2}k", "-vf", "scale=1280:720", "-c:a", "aac", "-b:a", "128k",
    "-movflags", "+faststart", str(PHONE))
print(json.dumps({"out": str(OUT), "phone": str(PHONE), "seconds": round(total, 2)}))
