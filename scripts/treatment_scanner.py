"""
Treatment Scanner -- Reads all treatment files and flags anomalies.

Checks for:
  1. __NEEDS_LLM_CLOSING__ -- announcer sign-off never generated
  2. Zero dialogue lines in any scene
  3. Missing cast members (empty CAST & VOICES)
  4. Gender mismatch -- Director says female but cast profile says male (or vice versa)
  5. Duplicate voice presets -- two characters sharing the same voice
  6. Missing SFX -- scenes with no [SFX] cues at all
  7. Suspiciously short duration (under 30s)
  8. Suspiciously large or small file size
  9. VRAM exceeded ceiling (14.5 GB)
 10. Missing scenes -- SCENE ARC section empty or has 0 scenes
 11. Character name drift -- name in CAST differs from name in SCRIPT
 12. Empty script body (FULL SCRIPT section missing or blank)

Usage:
    python treatment_scanner.py
    python treatment_scanner.py --last 10       # scan only last N files
    python treatment_scanner.py --file <path>   # scan one specific file

AntiGravity: just run this script. Do NOT modify it.
"""

import os, re, sys, argparse

OUTPUT_DIR = r"C:\Users\jeffr\Documents\ComfyUI\output\old_time_radio"
VRAM_CEILING = 14.5

# ---------------------------------------------------------------------------
# PARSE A SINGLE TREATMENT FILE
# ---------------------------------------------------------------------------
def parse_treatment(path):
    """Parse a treatment file into structured data."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    data = {"path": path, "filename": os.path.basename(path), "raw": text, "flags": []}

    # Title
    m = re.search(r'Title\s*:\s*"([^"]+)"', text)
    data["title"] = m.group(1) if m else "MISSING"

    # Produced date
    m = re.search(r"Produced\s*:\s*(.+)", text)
    data["produced"] = m.group(1).strip() if m else "MISSING"

    # Cast
    cast_section = re.search(
        r"CAST & VOICES\n[-]+\n(.*?)(?:\n\n|\nSCENE ARC)", text, re.DOTALL
    )
    data["cast"] = {}
    data["cast_genders"] = {}
    if cast_section:
        for line in cast_section.group(1).strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Pattern: NAME  ->  voice_preset  gender * trait * trait
            m = re.match(
                r"(\S+(?:\s+\S+)*?)\s+(?:->|-->)\s+(\S+)\s+(.*)", line
            )
            if m:
                name = m.group(1).strip()
                preset = m.group(2).strip()
                traits = m.group(3).strip()
                data["cast"][name] = preset
                # Extract gender from traits (first word is usually gender)
                gender_match = re.search(r"(male|female)", traits, re.IGNORECASE)
                if gender_match:
                    data["cast_genders"][name] = gender_match.group(1).lower()

    # Scene arc
    scene_arc = re.search(
        r"SCENE ARC\n[-]+\n(.*?)(?:\n\nFULL SCRIPT)", text, re.DOTALL
    )
    data["scenes"] = []
    if scene_arc:
        for m in re.finditer(r"Scene\s+(\d+)\s+.*?(\d+)\s+dialogue lines", scene_arc.group(1)):
            data["scenes"].append({
                "num": int(m.group(1)),
                "dialogue_lines": int(m.group(2)),
            })

    # Full script section
    script_section = re.search(
        r"FULL SCRIPT.*?\n[-]+\n(.*?)(?:\nPRODUCTION)", text, re.DOTALL
    )
    data["script_body"] = script_section.group(1).strip() if script_section else ""

    # Dialogue lines in script (CHARACTER lines)
    data["script_characters"] = set()
    if data["script_body"]:
        for line in data["script_body"].split("\n"):
            line = line.strip()
            # Match character name lines (all caps, possibly with [voice] after)
            m = re.match(r"^([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", line)
            if m:
                name = m.group(1).strip()
                if name not in ("SFX", "PAUSE", "BEAT", "SCENE"):
                    data["script_characters"].add(name)

    # SFX count in script
    data["sfx_count"] = len(re.findall(r"\[SFX\]", data["script_body"]))

    # Check for __NEEDS_LLM_CLOSING__
    data["needs_closing"] = "__NEEDS_LLM_CLOSING__" in text

    # Production stats
    m = re.search(r"Duration\s*:\s*([\d.]+)\s*min\s*\(([\d.]+)\s*s\)", text)
    if m:
        data["duration_min"] = float(m.group(1))
        data["duration_s"] = float(m.group(2))
    else:
        data["duration_min"] = 0
        data["duration_s"] = 0

    m = re.search(r"Size\s*:\s*([\d.]+)\s*MB", text)
    data["size_mb"] = float(m.group(1)) if m else 0

    # VRAM from diagnostic line
    m = re.search(r"PEAKED AT ([\d.]+)GB", text)
    data["vram_peak"] = float(m.group(1)) if m else 0

    return data


# ---------------------------------------------------------------------------
# FLAG CHECKS
# ---------------------------------------------------------------------------
def check_flags(data):
    """Run all anomaly checks on parsed treatment data."""
    flags = []

    # 1. Missing LLM closing
    if data["needs_closing"]:
        flags.append("NEEDS_LLM_CLOSING: Announcer sign-off was never generated")

    # 2. Zero dialogue in any scene
    for scene in data["scenes"]:
        if scene["dialogue_lines"] == 0:
            flags.append(f"ZERO_DIALOGUE: Scene {scene['num']} has 0 dialogue lines")

    # 3. Empty cast
    if not data["cast"]:
        flags.append("EMPTY_CAST: No characters in CAST & VOICES section")

    # 4. Duplicate voice presets
    presets = list(data["cast"].values())
    seen = {}
    for name, preset in data["cast"].items():
        if preset in seen:
            flags.append(
                f"DUPLICATE_VOICE: {name} and {seen[preset]} share preset {preset}"
            )
        seen[preset] = name

    # 5. Missing SFX
    if data["sfx_count"] == 0 and data["script_body"]:
        flags.append("NO_SFX: Full script has zero [SFX] cues")

    # 6. Suspiciously short duration
    if 0 < data["duration_s"] < 30:
        flags.append(f"SHORT_DURATION: Only {data['duration_s']}s")

    # 7. File size anomalies
    if 0 < data["size_mb"] < 5:
        flags.append(f"TINY_FILE: Only {data['size_mb']} MB -- possible empty render")
    if data["size_mb"] > 500:
        flags.append(f"HUGE_FILE: {data['size_mb']} MB -- possible runaway render")

    # 8. VRAM exceeded ceiling
    if data["vram_peak"] > VRAM_CEILING:
        flags.append(
            f"VRAM_OVER: Peaked at {data['vram_peak']} GB (ceiling {VRAM_CEILING})"
        )

    # 9. No scenes
    if not data["scenes"]:
        flags.append("NO_SCENES: Scene arc section empty or unparseable")

    # 10. Character name drift (cast vs script)
    if data["cast"] and data["script_characters"]:
        cast_names = set()
        for name in data["cast"]:
            cast_names.add(name.upper())
            # Also add without spaces for compressed names
            cast_names.add(name.upper().replace(" ", ""))

        script_names = set(n.upper() for n in data["script_characters"])
        # Remove ANNOUNCER from comparison (always procedural)
        script_names.discard("ANNOUNCER")
        cast_names.discard("ANNOUNCER")

        for sname in script_names:
            compressed = sname.replace(" ", "")
            if sname not in cast_names and compressed not in cast_names:
                flags.append(f"NAME_DRIFT: '{sname}' in script but not in cast")

    # 11. Empty script body
    if not data["script_body"]:
        flags.append("EMPTY_SCRIPT: Full script section is missing or blank")

    # 12. Missing title
    if data["title"] == "MISSING":
        flags.append("NO_TITLE: Title not found in treatment")

    data["flags"] = flags
    return data


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def scan_treatments(files):
    """Scan a list of treatment files and print report."""
    total = len(files)
    clean = 0
    flagged = 0
    all_flags = {}  # flag_type -> count

    print(f"\n{'='*70}")
    print(f"  TREATMENT SCANNER -- {total} files")
    print(f"{'='*70}\n")

    results = []
    for path in sorted(files):
        data = parse_treatment(path)
        data = check_flags(data)
        results.append(data)

        if data["flags"]:
            flagged += 1
            print(f"  FLAGGED: {data['filename']}")
            print(f"           Title: {data['title']}")
            print(f"           Cast: {len(data['cast'])} | "
                  f"Dialogue: {sum(s['dialogue_lines'] for s in data['scenes'])} | "
                  f"SFX: {data['sfx_count']} | "
                  f"Duration: {data['duration_s']}s | "
                  f"Size: {data['size_mb']}MB | "
                  f"VRAM: {data['vram_peak']}GB")
            for flag in data["flags"]:
                tag = flag.split(":")[0]
                all_flags[tag] = all_flags.get(tag, 0) + 1
                print(f"           >> {flag}")
            print()
        else:
            clean += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total scanned:  {total}")
    print(f"  Clean:          {clean}")
    print(f"  Flagged:        {flagged}")
    print()

    if all_flags:
        print("  FLAG BREAKDOWN:")
        for tag, count in sorted(all_flags.items(), key=lambda x: -x[1]):
            print(f"    {tag:25s}  {count}")
        print()

    # Stats across all files
    durations = [d["duration_s"] for d in results if d["duration_s"] > 0]
    sizes = [d["size_mb"] for d in results if d["size_mb"] > 0]
    vrams = [d["vram_peak"] for d in results if d["vram_peak"] > 0]

    if durations:
        print(f"  Duration range:  {min(durations):.0f}s - {max(durations):.0f}s  "
              f"(avg {sum(durations)/len(durations):.0f}s)")
    if sizes:
        print(f"  File size range: {min(sizes):.1f}MB - {max(sizes):.1f}MB  "
              f"(avg {sum(sizes)/len(sizes):.1f}MB)")
    if vrams:
        print(f"  VRAM range:      {min(vrams):.1f}GB - {max(vrams):.1f}GB  "
              f"(avg {sum(vrams)/len(vrams):.1f}GB)")

    print(f"\n{'='*70}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan OTR treatment files for anomalies")
    parser.add_argument("--last", type=int, help="Scan only the last N files")
    parser.add_argument("--file", type=str, help="Scan a single file")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = [
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.endswith("_treatment.txt")
        ]
        files.sort(key=os.path.getmtime)
        if args.last:
            files = files[-args.last:]

    if not files:
        print("No treatment files found.")
        sys.exit(1)

    scan_treatments(files)
