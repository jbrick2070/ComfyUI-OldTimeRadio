"""
SIGNAL LOST -- Soak Test Operator
Runs episodes continuously via ComfyUI HTTP API with controlled
parameter randomization. Logs every run. Reboots ComfyUI if stalled.

Usage:
    python soak_operator.py

AntiGravity: just run this script. Do NOT modify it.
"""

import json, requests, time, random, uuid, os, subprocess, re, sys, textwrap

# ---------------------------------------------------------------------------
# PATHS (Windows, Jeffrey's machine)
# ---------------------------------------------------------------------------
COMFYUI = "http://127.0.0.1:8000"
WORKFLOW_PATH = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json"
SOAK_LOG = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\logs\soak_log.md"
RUNTIME_LOG = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\otr_runtime.log"
OUTPUT_DIR = r"C:\Users\jeffr\Documents\ComfyUI\output\old_time_radio"
COMFYUI_EXE = r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
COMFYUI_MAIN = r"C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\main.py"
PROMPT_OUTPUT_PATH = r"C:\Users\jeffr\Documents\ComfyUI\otr_prompt_built.json"

TIMEOUT_S = 1800   # 30 min max per episode
POLL_S = 10         # seconds between completion checks
COOLDOWN_S = 30     # seconds between episodes

# ---------------------------------------------------------------------------
# ALLOWED PARAMETER VALUES (closed sets -- no invention allowed)
# ---------------------------------------------------------------------------
GENRES = [
    "hard_sci_fi", "space_opera", "dystopian", "time_travel",
    "first_contact", "cosmic_horror", "cyberpunk", "post_apocalyptic",
]
TARGET_WORDS = [350, 700, 1050, 1400, 2100]
TARGET_LENGTHS = ["short (3 acts)", "medium (5 acts)", "long (7-8 acts)"]
STYLES = [
    "tense claustrophobic", "space opera epic",
    "psychological slow-burn", "hard-sci-fi procedural",
    "noir mystery", "chaotic black-mirror",
]
CREATIVITIES = ["safe & tight", "balanced", "wild & rough", "maximum chaos"]
OPT_PROFILES = ["Pro (Ultra Quality)", "Standard"]
# NOTE: "Obsidian (UNSTABLE/4GB)" excluded -- not valid on 16 GB hardware

# Node 1 (OTR_Gemma4ScriptWriter) widgets_values index map:
#   [0] episode_title       [1] genre_flavor       [2] target_words
#   [3] num_characters      [4] model_id           [5] custom_premise
#   [6] include_act_breaks  [7] self_critique       [8] open_close
#   [9] target_length       [10] style_variant      [11] creativity
#   [12] arc_enhancer       [13] optimization_profile
WV_GENRE = 1
WV_TARGET_WORDS = 2
WV_TARGET_LENGTH = 9
WV_STYLE = 10
WV_CREATIVITY = 11
WV_OPT_PROFILE = 13


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def print_f(msg):
    """Print with immediate flush, handling Unicode encoding on Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Fallback for Windows consoles that don't support UTF-8
        print(msg.encode('ascii', 'replace').decode('ascii'))
    sys.stdout.flush()


def comfyui_alive():
    try:
        r = requests.get(f"{COMFYUI}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _find_comfyui_pid():
    """Find the PID of the ComfyUI server by checking port 8000."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if ":8000" in line and "LISTENING" in line:
                parts = line.strip().split()
                if parts:
                    return parts[-1]
    except Exception:
        pass
    return None


def reboot_comfyui():
    """Kill and restart ComfyUI Desktop. Wait for startup.

    Uses targeted PID kill (port 8000 only) instead of blanket
    taskkill /IM python.exe, which would kill the soak operator too.
    Launches ComfyUI in a NEW_CONSOLE so its lifecycle is fully
    decoupled from the operator process.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print_f(f"REBOOT: ComfyUI unresponsive -- initiating restart at {ts}")
    append_to_log(f"REBOOT: ComfyUI unresponsive -- initiating restart at {ts}")

    # --- Kill only the ComfyUI process (port 8000) ----------------------
    comfy_pid = _find_comfyui_pid()
    if comfy_pid:
        print_f(f"REBOOT: Killing ComfyUI PID {comfy_pid}")
        subprocess.run(
            ["taskkill", "/F", "/PID", comfy_pid],
            capture_output=True
        )
    else:
        print_f("REBOOT: No ComfyUI PID found on port 8000 -- may already be dead")

    time.sleep(10)

    # --- Restart ComfyUI in a decoupled console -------------------------
    comfy_args = [
        COMFYUI_EXE, COMFYUI_MAIN,
        "--port", "8000",
        "--highvram", "--force-fp16", "--cuda-malloc",
        "--user-directory", r"C:\Users\jeffr\Documents\ComfyUI",
    ]
    comfy_env = os.environ.copy()
    comfy_env["PYTHONIOENCODING"] = "utf-8"

    print_f("REBOOT: Launching ComfyUI in new console window...")
    # Fire-and-forget: CREATE_NEW_CONSOLE decouples ComfyUI from this operator
    # so it survives when the soak loop exits. We still wrap the spawn in
    # try/except + proc.kill() so a half-launched process never zombies if
    # Popen itself raises after forking. (Bug Bible BUG-09.02 compliance.)
    proc = None
    try:
        proc = subprocess.Popen(
            comfy_args,
            env=comfy_env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    except Exception:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        raise

    # --- Wait for ComfyUI to become responsive --------------------------
    print_f("REBOOT: Waiting up to 120s for ComfyUI startup...")
    for attempt in range(24):  # 24 x 5s = 120s max
        time.sleep(5)
        if comfyui_alive():
            print_f(f"REBOOT: ComfyUI responding after {(attempt + 1) * 5}s")
            return
        if attempt % 4 == 3:
            print_f(f"REBOOT: Still waiting... ({(attempt + 1) * 5}s)")
    print_f("REBOOT: ComfyUI did not respond within 120s")


def append_to_log(text):
    os.makedirs(os.path.dirname(SOAK_LOG), exist_ok=True)
    with open(SOAK_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def get_runtime_details():
    """Parse otr_runtime.log for the most recent episode title, dialogue
    line count, and VRAM peak from the current run."""
    if not os.path.exists(RUNTIME_LOG):
        return "Unknown", "Unknown", "Unknown"
    try:
        with open(RUNTIME_LOG, "r", encoding="utf-8") as f:
            # Read last ~500 lines (enough for one generation cycle)
            f.seek(0, 2)
            end = f.tell()
            f.seek(max(0, end - 40000))
            lines = f.readlines()

        title = "Unknown"
        dialogue = "Unknown"
        vram = "Unknown"

        for line in reversed(lines):
            if title == "Unknown":
                m = re.search(r"ScriptWriter: FINGERPRINT .*? \| (.*?) \|", line)
                if m:
                    title = m.group(1).strip()
            if dialogue == "Unknown":
                m = re.search(r"ScriptWriter: CAST_MAP .*? \| (\d+) lines", line)
                if m:
                    dialogue = m.group(1).strip()
            if vram == "Unknown":
                m = re.search(r"VRAM_SNAPSHOT .*? peak_gb=([\d.]+)", line)
                if m:
                    vram = m.group(1).strip()
            if title != "Unknown" and dialogue != "Unknown" and vram != "Unknown":
                break

        return title, dialogue, vram
    except Exception as e:
        print_f(f"Error parsing runtime log: {e}")
        return "Unknown", "Unknown", "Unknown"


def check_treatment(before_count):
    """Check if a new treatment file appeared since the run started."""
    try:
        after = set(os.listdir(OUTPUT_DIR))
        treatments = [f for f in after if f.endswith("_treatment.txt")]
        return len(treatments) > before_count, len(treatments)
    except Exception:
        return False, 0


def count_treatments():
    """Count existing treatment files in output dir."""
    try:
        return len([f for f in os.listdir(OUTPUT_DIR)
                     if f.endswith("_treatment.txt")])
    except Exception:
        return 0


def find_newest_treatment():
    """Return path to the most recently modified treatment file, or None."""
    try:
        treatments = [
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.endswith("_treatment.txt")
        ]
        if not treatments:
            return None
        return max(treatments, key=os.path.getmtime)
    except Exception:
        return None


def scan_treatment(path):
    """Read-only scan of a treatment file. Returns list of flag strings.
    Never modifies files. Never tries to fix anything."""
    flags = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as exc:
        return [f"READ_FAIL: Could not read treatment: {exc}"]

    filename = os.path.basename(path)

    # 1. Missing LLM closing
    if "__NEEDS_LLM_CLOSING__" in text:
        flags.append("NEEDS_LLM_CLOSING: Announcer sign-off never generated")

    # 2. Parse cast
    cast = {}
    cast_section = re.search(
        r"CAST & VOICES\n[-]+\n(.*?)(?:\n\n|\nSCENE ARC)", text, re.DOTALL)
    if cast_section:
        for line in cast_section.group(1).strip().split("\n"):
            m = re.match(r"\s*(\S+(?:\s+\S+)*?)\s+(?:->|-->)\s+(\S+)\s+(.*)", line)
            if m:
                cast[m.group(1).strip()] = {
                    "preset": m.group(2).strip(),
                    "traits": m.group(3).strip(),
                }
    if not cast:
        flags.append("EMPTY_CAST: No characters found in CAST section")

    # 3. Duplicate voice presets
    presets_seen = {}
    for name, info in cast.items():
        p = info["preset"]
        if p in presets_seen:
            flags.append(f"DUPLICATE_VOICE: {name} and {presets_seen[p]} share {p}")
        presets_seen[p] = name

    # 4. Gender mismatch (trait says female but name profile says male, etc.)
    for name, info in cast.items():
        traits = info["traits"].lower()
        # Check for contradictions like "male * anxious" on a female-hinted name
        has_male = "male" in traits and "female" not in traits
        has_female = "female" in traits
        # Flag if both appear (shouldn't happen but check)
        if has_male and has_female:
            flags.append(f"GENDER_CONFLICT: {name} traits say both male and female")

    # 5. Scene arc dialogue counts
    scene_arc = re.search(
        r"SCENE ARC\n[-]+\n(.*?)(?:\n\nFULL SCRIPT)", text, re.DOTALL)
    total_dialogue = 0
    if scene_arc:
        for m in re.finditer(r"Scene\s+(\d+).*?(\d+)\s+dialogue lines", scene_arc.group(1)):
            dl = int(m.group(2))
            total_dialogue += dl
            if dl == 0:
                flags.append(f"ZERO_DIALOGUE: Scene {m.group(1)} has 0 lines")
    else:
        flags.append("NO_SCENE_ARC: Scene arc section missing or unparseable")

    # 6. Full script checks
    script_section = re.search(
        r"FULL SCRIPT.*?\n[-]+\n(.*?)(?:\nPRODUCTION)", text, re.DOTALL)
    script_body = script_section.group(1).strip() if script_section else ""

    if not script_body:
        flags.append("EMPTY_SCRIPT: Full script section is blank")

    sfx_count = len(re.findall(r"\[SFX\]", script_body))
    if sfx_count == 0 and script_body:
        flags.append("NO_SFX: Script has zero [SFX] cues")

    # 7. Production stats
    m = re.search(r"Duration\s*:\s*([\d.]+)\s*min\s*\(([\d.]+)\s*s\)", text)
    duration_s = float(m.group(2)) if m else 0
    if 0 < duration_s < 30:
        flags.append(f"SHORT_DURATION: Only {duration_s}s")

    m = re.search(r"Size\s*:\s*([\d.]+)\s*MB", text)
    size_mb = float(m.group(1)) if m else 0
    if 0 < size_mb < 5:
        flags.append(f"TINY_FILE: Only {size_mb}MB -- possible empty render")
    if size_mb > 500:
        flags.append(f"HUGE_FILE: {size_mb}MB -- possible runaway render")

    # 8. VRAM
    m = re.search(r"PEAKED AT ([\d.]+)GB", text)
    vram = float(m.group(1)) if m else 0
    if vram > 14.5:
        flags.append(f"VRAM_OVER: {vram}GB exceeds 14.5GB ceiling")

    # 9. Character name drift (cast vs script body)
    if cast and script_body:
        cast_upper = set()
        for name in cast:
            cast_upper.add(name.upper())
            cast_upper.add(name.upper().replace(" ", ""))
        script_chars = set()
        for line in script_body.split("\n"):
            m = re.match(r"^\s+([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", line)
            if m:
                cname = m.group(1).strip()
                if cname not in ("SFX", "PAUSE", "BEAT", "SCENE"):
                    script_chars.add(cname)
        script_chars.discard("ANNOUNCER")
        cast_upper.discard("ANNOUNCER")
        for sc in script_chars:
            compressed = sc.replace(" ", "")
            if sc not in cast_upper and compressed not in cast_upper:
                flags.append(f"NAME_DRIFT: '{sc}' in script but not in cast list")

    # 10. All-same-gender cast (known bug pattern)
    if cast:
        genders = set()
        for name, info in cast.items():
            if name.upper() == "ANNOUNCER":
                continue
            traits = info["traits"].lower()
            if "female" in traits:
                genders.add("female")
            elif "male" in traits:
                genders.add("male")
        if len(genders) == 1 and len(cast) > 2:
            flags.append(f"ALL_SAME_GENDER: All non-announcer cast is {genders.pop()}")

    # 11. Title stuck on default
    m_title = re.search(r'Title\s*:\s*"([^"]+)"', text)
    if m_title:
        title_text = m_title.group(1)
        if title_text.lower() in ("the last frequency", "untitled", "episode"):
            flags.append(f"TITLE_STUCK: Title is '{title_text}' -- may be default/stuck")

    # 12. Dialogue count mismatch (arc total vs actual script lines)
    if script_body and total_dialogue > 0:
        actual_lines = 0
        for line in script_body.split("\n"):
            stripped = line.strip()
            m_char = re.match(r"^([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", stripped)
            if m_char and m_char.group(1).strip() not in ("SFX", "PAUSE", "BEAT", "SCENE"):
                actual_lines += 1
        # Each character name line precedes dialogue, so count those
        if abs(actual_lines - total_dialogue) > 2:
            flags.append(f"DIALOGUE_MISMATCH: Arc says {total_dialogue} lines but script has ~{actual_lines}")

    # 13. Name squish (compressed names without spaces, like NIRANACKELS)
    if cast:
        for name in cast:
            if name.upper() == "ANNOUNCER":
                continue
            # Flag names longer than 10 chars with no spaces (likely squished)
            if len(name) > 10 and " " not in name and name.upper() == name:
                flags.append(f"NAME_SQUISH: '{name}' looks like a compressed name (missing spaces)")

    # 14. Single-line character (wasted voice load)
    if script_body and cast:
        char_line_counts = {}
        for line in script_body.split("\n"):
            stripped = line.strip()
            m_char = re.match(r"^([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", stripped)
            if m_char:
                cname = m_char.group(1).strip()
                if cname not in ("SFX", "PAUSE", "BEAT", "SCENE", "ANNOUNCER"):
                    char_line_counts[cname] = char_line_counts.get(cname, 0) + 1
        for cname, count in char_line_counts.items():
            if count == 1 and len(char_line_counts) > 1:
                flags.append(f"SINGLE_LINE_CHAR: {cname} only speaks once (wasted voice load)")

    # 15. Announcer dominates (more lines than any character)
    if script_body:
        ann_count = 0
        max_char_count = 0
        for line in script_body.split("\n"):
            stripped = line.strip()
            if re.match(r"^ANNOUNCER\b", stripped):
                ann_count += 1
        if char_line_counts:
            max_char_count = max(char_line_counts.values()) if char_line_counts else 0
        if ann_count > max_char_count and ann_count > 2:
            flags.append(f"ANNOUNCER_DOMINATES: Announcer has {ann_count} lines vs max character {max_char_count}")

    # 16. News seed missing or blank
    news_m = re.search(r"NEWS SEED\n[-]+\n(.*?)(?:\n\n|\nCAST)", text, re.DOTALL)
    if not news_m or not news_m.group(1).strip():
        flags.append("NEWS_SEED_MISSING: No news seed found -- RSS feed may not be injecting")

    # 17. Duplicate dialogue (same line appears twice -- LLM hallucination loop)
    if script_body:
        dialogue_lines = []
        capture_next = False
        for line in script_body.split("\n"):
            stripped = line.strip()
            if capture_next and stripped and not stripped.startswith("["):
                dialogue_lines.append(stripped)
                capture_next = False
            m_char = re.match(r"^([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", stripped)
            if m_char and m_char.group(1).strip() not in ("SFX", "PAUSE", "BEAT", "SCENE"):
                capture_next = True
        seen_lines = set()
        for dl in dialogue_lines:
            if dl in seen_lines and len(dl) > 20:
                flags.append(f"DUPLICATE_DIALOGUE: Line repeated: '{dl[:60]}...'")
                break
            seen_lines.add(dl)

    # 18. Non-ASCII / encoding corruption (mojibake)
    mojibake_patterns = ["a]]]", "A(c)", "a]!", "a]\"", "A<<", "A>>"]
    for pattern in mojibake_patterns:
        if pattern in text:
            flags.append(f"MOJIBAKE: Encoding corruption detected ('{pattern}' found)")
            break

    # 19. Scene count mismatch (arc vs script body)
    if scene_arc:
        arc_scene_count = len(re.findall(r"Scene\s+\d+", scene_arc.group(1)))
        script_scene_count = len(re.findall(r"SCENE\s+\d+", script_body, re.IGNORECASE))
        if arc_scene_count > 0 and script_scene_count > 0:
            if abs(arc_scene_count - script_scene_count) > 1:
                flags.append(f"SCENE_MISMATCH: Arc has {arc_scene_count} scenes but script has {script_scene_count}")

    return flags


# ---------------------------------------------------------------------------
# BABA & BOOEY -- Balcony Preview (before each run)
# ---------------------------------------------------------------------------
HECKLES = {
    "hard_sci_fi": [
        "Baba: Hard sci-fi? The only hard part is staying awake!",
        "Booey: I once read hard sci-fi. Took me three naps to finish!",
        "Baba: They say hard sci-fi is scientifically accurate. Accurately boring!",
    ],
    "space_opera": [
        "Booey: Space opera? I prefer the regular opera -- at least I can sleep in a nice chair!",
        "Baba: The last space opera I saw, the best performance was the vacuum of space!",
        "Booey: Space opera -- where no one can hear you yawn!",
    ],
    "dystopian": [
        "Baba: A dystopian story? Just describe this theater!",
        "Booey: Dystopian? I have been living one since they put us in this balcony!",
        "Baba: Another dystopia. As if watching this show was not punishment enough!",
    ],
    "time_travel": [
        "Booey: Time travel? I wish I could travel to before I agreed to watch this!",
        "Baba: If I had a time machine, I would skip to the end credits!",
        "Booey: Time travel -- the only way to get those 30 minutes back!",
    ],
    "first_contact": [
        "Baba: First contact? The aliens took one look and left!",
        "Booey: First contact with this show? More like last contact!",
        "Baba: Even the aliens have better things to watch!",
    ],
    "cosmic_horror": [
        "Booey: Cosmic horror? The real horror is we are still up here watching!",
        "Baba: They say cosmic horror is unknowable. Just like why we keep coming back!",
        "Booey: Lovecraft never imagined anything as terrifying as this balcony seat!",
    ],
    "cyberpunk": [
        "Baba: Cyberpunk? I can barely work the remote!",
        "Booey: High tech, low life -- just like this theater!",
        "Baba: In cyberpunk, everyone has implants. I just need new knees!",
    ],
    "post_apocalyptic": [
        "Booey: Post-apocalyptic? Looks like the budget already survived one!",
        "Baba: After the apocalypse, only two things survive -- cockroaches and us!",
        "Booey: Post-apocalyptic -- finally a setting that matches this theater!",
    ],
}

WORD_HECKLES = {
    350: "Baba: 350 words? That is shorter than my grocery list!",
    700: "Booey: 700 words? Just enough rope to hang itself!",
    1050: "Baba: 1050 words? Getting ambitious, are we?",
    1400: "Booey: 1400 words? I will need a second nap for that!",
    2100: "Baba: 2100 words? They are writing a novel up there! Booey: A novel way to bore us!",
}

LENGTH_HECKLES = {
    "short (3 acts)": "Booey: Three acts? Even Shakespeare needed five! Baba: Shakespeare also had talent!",
    "medium (5 acts)": "Baba: Five acts. The perfect number -- of chances to walk out!",
    "long (7-8 acts)": "Booey: Seven acts? Baba: Wake me for the curtain call!",
}


def balcony_preview(config):
    """Baba and Booey heckle the upcoming episode from the balcony."""
    genre = config["genre"]
    words = config["words"]
    length = config["length"]

    lines = []
    lines.append("")
    lines.append("*" * 50)
    lines.append("  FROM THE BALCONY -- Baba & Booey")
    lines.append("*" * 50)
    lines.append("")

    # Genre heckle
    genre_lines = HECKLES.get(genre, ["Baba: What is this? Booey: I have no idea!"])
    lines.append(random.choice(genre_lines))

    # Word count heckle
    wh = WORD_HECKLES.get(words, f"Booey: {words} words? That is a number, all right!")
    lines.append(wh)

    # Length heckle
    lh = LENGTH_HECKLES.get(length, "Baba: However many acts, it is too many!")
    lines.append(lh)

    # Closing zinger
    closers = [
        "Both: Hehehehe!",
        "Booey: Why do we keep coming back? Baba: Beats me!",
        "Baba: Well, here we go again! Booey: Same time, same suffering!",
        "Both: Bravo! ...Just kidding!",
        "Booey: Ready? Baba: I was born ready. Ready to leave!",
    ]
    lines.append(random.choice(closers))
    lines.append("")

    text = "\n".join(lines)
    print_f(text)
    return text


# ---------------------------------------------------------------------------
# CRITIC REVIEW -- Haiku + Rotten Tomatoes score (after success)
# ---------------------------------------------------------------------------
POSITIVE_HAIKU = [
    ("Voices fill the dark", "static hums then fades to song", "signal found at last"),
    ("Echoes in the void", "characters breathe and stumble", "a world comes alive"),
    ("Dials turn slowly", "the frequency locks on tight", "pure storytelling"),
    ("Old radio glows", "words crackle through dust and time", "the arc lands just right"),
    ("Wavering signal", "then the narrative locks in", "a clean transmission"),
]

MIXED_HAIKU = [
    ("The premise was bold", "but the middle lost its way", "still worth a listen"),
    ("Ambitious in scope", "some static in the middle", "but the end came through"),
    ("A slow-burning start", "then the signal flickered twice", "decent reception"),
    ("Not their finest hour", "but the voices carried weight", "above average"),
]

NEGATIVE_HAIKU = [
    ("The signal was weak", "static drowned the dialogue", "tune in next time please"),
    ("Too short for its reach", "ambition outran the words", "needs more frequency"),
    ("Lost in transmission", "the arc never quite resolved", "adjust the antenna"),
]

TOMATO_LABELS = [
    (90, 100, "Certified Fresh"),
    (75, 89, "Fresh"),
    (60, 74, "Fresh (barely)"),
    (40, 59, "Rotten"),
    (0, 39, "Splat"),
]


def critic_review(config, title, dialogue, vram, has_treatment, duration):
    """Post-episode critic review: haiku + Rotten Tomatoes score."""
    lines = []
    lines.append("")
    lines.append("-" * 50)
    lines.append("  CRITIC REVIEW")
    lines.append("-" * 50)
    lines.append(f'  "{title}"')
    lines.append("")

    # Score based on signals: treatment present, dialogue count, duration
    score = 50  # base
    if has_treatment:
        score += 15
    try:
        dl = int(dialogue)
        if dl >= 20:
            score += 10
        elif dl >= 10:
            score += 5
    except (ValueError, TypeError):
        pass
    try:
        v = float(vram)
        if v <= 14.5:
            score += 10  # stayed under VRAM ceiling
    except (ValueError, TypeError):
        pass
    if duration < 600:
        score += 5  # fast run
    elif duration > 1500:
        score -= 10  # suspiciously slow

    # Add genre/length flavor variance
    score += random.randint(-8, 8)
    score = max(0, min(100, score))

    # Pick haiku pool
    if score >= 75:
        haiku = random.choice(POSITIVE_HAIKU)
    elif score >= 50:
        haiku = random.choice(MIXED_HAIKU)
    else:
        haiku = random.choice(NEGATIVE_HAIKU)

    lines.append(f"  {haiku[0]}")
    lines.append(f"  {haiku[1]}")
    lines.append(f"  {haiku[2]}")
    lines.append("")

    # Tomato label
    label = "Unknown"
    for lo, hi, lbl in TOMATO_LABELS:
        if lo <= score <= hi:
            label = lbl
            break

    tomato = "[FRESH]" if score >= 60 else "[ROTTEN]"
    lines.append(f"  Rotten Tomatoes: {score}% -- {label} {tomato}")
    lines.append(f"  Config: {config.get('genre', '?')} / {config.get('words', '?')}w / {config.get('length', '?')}")
    lines.append(f"  Dialogue: {dialogue} lines | VRAM: {vram} GB | Duration: {duration}s")
    lines.append("-" * 50)
    lines.append("")

    text = "\n".join(lines)
    print_f(text)
    return text


# ---------------------------------------------------------------------------
# PERFORMANCE REVIEW -- Lead Developer suggestions (no code, just ideas)
# ---------------------------------------------------------------------------
ENTERTAINMENT_IDEAS = [
    "Add recurring characters across episodes -- a grizzled captain who keeps showing up in different genres would build listener loyalty.",
    "Introduce cold opens before the announcer -- 10 seconds of raw scene audio to hook the listener before the title card.",
    "Add episode-to-episode continuity Easter eggs -- subtle references to previous episodes that reward repeat listeners.",
    "Create signature SFX for each genre -- a distinct radio static pattern for cyberpunk vs cosmic horror vs space opera.",
    "Add a post-credits stinger scene -- a short 5-second teaser that hints at a larger universe.",
    "Experiment with unreliable narrators -- have the announcer occasionally contradict what the characters say.",
    "Try a two-part cliffhanger format -- end one episode mid-crisis, resolve in the next.",
    "Add ambient soundscapes between dialogue -- breathing room makes the drama land harder.",
    "Introduce a mystery box format -- drop a cryptic audio artifact that only makes sense 3 episodes later.",
    "Use silence as a dramatic tool -- 2 seconds of dead air before a revelation hits different on radio.",
    "Create genre mashup episodes -- what if first_contact meets noir_mystery?",
    "Add a listener mail segment voiced by the announcer -- fake letters that deepen the world.",
]

ROBUSTNESS_IDEAS = [
    "Track dialogue-to-SFX ratio per genre -- some genres might systematically underuse sound design.",
    "Add a voice diversity score -- measure how many unique presets get used vs total available.",
    "Build a character name dictionary to catch LLM name-squishing bugs automatically.",
    "Log token counts per scene to detect when the LLM is running out of context budget.",
    "Add a prompt replay mode -- re-run a specific config that produced a flagged treatment to see if it reproduces.",
    "Track VRAM over time to detect slow memory leaks across 100+ episodes.",
    "Add a regression baseline -- save 5 known-good treatments and diff new ones against structure.",
    "Monitor MusicGen cue durations -- flag if opening theme is shorter than 8s or longer than 15s.",
    "Track Bark generation time per line -- sudden spikes indicate hallucination retries.",
    "Add a word-count audit -- compare target_words in config vs actual word count in script body.",
    "Log the news seed used and check if it actually influenced the premise.",
    "Build a cast diversity tracker -- how often does the same voice preset appear across episodes?",
]

VIDEO_IDEAS = [
    "Start with static scene cards -- genre-appropriate still images with Ken Burns zoom during dialogue.",
    "Use LTX-2.3 for environment establishing shots only -- 5-second clips at scene transitions, not character shots.",
    "Generate character portrait cards with IP-Adapter -- show once at first dialogue, then cut to environment.",
    "Add a retro TV static transition between scenes -- fits the old-time radio aesthetic perfectly.",
    "Create a visual radio dial animation as a recurring motif -- the frequency tuning ties into Signal Lost.",
    "Try split-screen during two-character dialogue -- left portrait, right portrait, environment behind.",
    "Generate genre-specific title cards -- cyberpunk neon, cosmic horror fog, space opera starfield.",
    "Add waveform visualization during pure audio moments -- visual representation of the signal.",
    "Use crossfade transitions timed to music cues -- sync visual cuts to the MusicGen interstitials.",
    "Start with audio-only for first 30 seconds, then fade in visuals -- the radio-to-TV transition IS the concept.",
    "Generate one hero shot per scene in subprocess, composite with FFmpeg -- keeps VRAM under control.",
    "Build a visual template system -- 3-4 shot types per genre that get randomized per scene.",
]


def performance_review(config, title, dialogue, vram, has_treatment, duration,
                        scan_flags):
    """Post-run lead developer review. Picks contextual suggestions based on
    what happened this run. No code changes -- just ideas logged to soak_log."""
    lines = []
    lines.append("")
    lines.append("=" * 50)
    lines.append("  PERFORMANCE REVIEW -- Lead Developer Notes")
    lines.append("=" * 50)
    lines.append(f'  Episode: "{title}"')
    lines.append(f"  Config: {config.get('genre', '?')} / {config.get('words', '?')}w / {config.get('length', '?')}")
    lines.append("")

    # Pick 1 entertainment idea
    lines.append("  ENTERTAINMENT:")
    lines.append(f"    {random.choice(ENTERTAINMENT_IDEAS)}")
    lines.append("")

    # Pick 1 robustness idea, weighted toward relevant ones if flags exist
    lines.append("  ROBUSTNESS:")
    if scan_flags:
        # If there were flags, mention them and pick a relevant idea
        lines.append(f"    (This run had {len(scan_flags)} flag(s) -- worth investigating)")
    lines.append(f"    {random.choice(ROBUSTNESS_IDEAS)}")
    lines.append("")

    # Pick 1 video idea
    lines.append("  VIDEO INTEGRATION:")
    lines.append(f"    {random.choice(VIDEO_IDEAS)}")
    lines.append("")

    # Contextual observation based on this specific run
    lines.append("  RUN-SPECIFIC OBSERVATION:")
    obs = []
    try:
        dl = int(dialogue)
        if dl <= 3:
            obs.append(f"    Only {dl} dialogue lines -- this is barely a vignette. Could we set a minimum floor per act?")
        elif dl >= 20:
            obs.append(f"    {dl} dialogue lines is substantial. Does the pacing hold or does it drag in the middle?")
    except (ValueError, TypeError):
        obs.append("    Dialogue count unknown -- runtime log parsing might need attention.")

    try:
        v = float(vram)
        if v > 12:
            obs.append(f"    VRAM at {v}GB -- getting warm. Long episodes with more voices could push past ceiling.")
        elif v < 6:
            obs.append(f"    VRAM only {v}GB -- lots of headroom. Could we use it for higher quality TTS settings?")
    except (ValueError, TypeError):
        pass

    if not has_treatment:
        obs.append("    No treatment file -- the audience never sees the script. This is a blind listen.")

    words = config.get("words", 0)
    length = config.get("length", "")
    if words == 2100 and "short" in length:
        obs.append("    2100 words in 3 acts -- that is dense. Does the script feel rushed?")
    elif words == 350 and "long" in length:
        obs.append("    350 words across 7-8 acts -- each act is ~50 words. Is there enough meat per scene?")

    if config.get("creativity") == "wild & rough":
        obs.append("    Wild and rough creativity -- check if the LLM went off the rails or produced something genuinely fresh.")

    if not obs:
        obs.append("    Solid run. No unusual patterns detected.")

    for o in obs:
        lines.append(o)

    lines.append("")
    lines.append("  (Ideas only -- no code changes. Log reviewed by Jeffrey.)")
    lines.append("=" * 50)
    lines.append("")

    text = "\n".join(lines)
    print_f(text)
    return text


# ---------------------------------------------------------------------------
# WEB-FORMAT TO API-FORMAT CONVERTER
# Uses /object_info schema to correctly map widgets_values to named inputs.
# ---------------------------------------------------------------------------
def _workflow_to_api_prompt(workflow, schemas):
    """Convert ComfyUI web-format workflow JSON to API prompt format."""
    # Build link map: link_id -> [source_node_id, source_slot]
    link_map = {}
    for lnk in workflow.get("links", []):
        link_id, src_node, src_slot = lnk[0], lnk[1], lnk[2]
        link_map[link_id] = [str(src_node), src_slot]

    prompt = {}
    for node in workflow.get("nodes", []):
        nid = str(node["id"])
        ntype = node["type"]

        # Start with linked inputs
        inputs = {}
        linked_names = set()
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                inputs[inp["name"]] = link_map.get(inp["link"])
                linked_names.add(inp["name"])

        # Map widgets_values to named params via schema
        if ntype in schemas:
            schema = schemas[ntype].get("input", {})
            required = list(schema.get("required", {}).keys())
            optional = list(schema.get("optional", {}).keys())
            all_params = required + optional

            wv = node.get("widgets_values", [])
            wv_idx = 0
            for param in all_params:
                if param in linked_names:
                    # This param is wired as a link -- check if it also has
                    # a widget value to consume (converted widget)
                    for inp in node.get("inputs", []):
                        if inp["name"] == param and inp.get("widget"):
                            if wv_idx < len(wv):
                                wv_idx += 1  # consume but don't override link
                    continue
                if wv_idx < len(wv):
                    inputs[param] = wv[wv_idx]
                    wv_idx += 1

        prompt[nid] = {"class_type": ntype, "inputs": inputs}
    return prompt


# ---------------------------------------------------------------------------
# SINGLE SOAK ITERATION
# ---------------------------------------------------------------------------
def run_iteration(run_num):
    print_f(f"\n{'='*60}")
    print_f(f"  RUN {run_num} -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_f(f"{'='*60}")

    start_time = time.time()
    result = "FAIL"
    error_msg = ""
    config = {}

    try:
        # 1. Health check
        if not comfyui_alive():
            print_f("ComfyUI not responding -- attempting reboot...")
            reboot_comfyui()
        if not comfyui_alive():
            reboot_comfyui()  # second attempt
        if not comfyui_alive():
            print_f("CRITICAL: ComfyUI unresponsive after 2 reboots.")
            append_to_log(f"### RUN {run_num:03d} -- {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                          f"- **Result:** CRITICAL\n- **Error:** ComfyUI dead after 2 reboots\n")
            return False

        # 2. Load workflow fresh from disk
        print_f("Loading workflow from disk...")
        with open(WORKFLOW_PATH, encoding="utf-8") as f:
            workflow = json.load(f)

        # 3. Controlled randomization (ONLY these 5 fields)
        config = {
            "genre": random.choice(GENRES),
            "words": random.choice(TARGET_WORDS),
            "length": random.choice(TARGET_LENGTHS),
            "style": random.choice(STYLES),
            "creativity": random.choice(CREATIVITIES),
            "profile": random.choice(OPT_PROFILES),
        }
        for node in workflow.get("nodes", []):
            if node.get("id") == 1 and node.get("type") == "OTR_Gemma4ScriptWriter":
                wv = node["widgets_values"]
                wv[WV_GENRE] = config["genre"]
                wv[WV_TARGET_WORDS] = config["words"]
                wv[WV_TARGET_LENGTH] = config["length"]
                wv[WV_STYLE] = config["style"]
                wv[WV_CREATIVITY] = config["creativity"]
                wv[WV_OPT_PROFILE] = config["profile"]
                break

        config_str = (f"genre={config['genre']} | words={config['words']} | "
                      f"length={config['length']} | style={config['style']} | "
                      f"creativity={config['creativity']} | profile={config['profile']}")
        print_f(f"CONFIG: {config_str}")

        # 3b. Balcony preview (Baba & Booey heckle)
        heckle_text = balcony_preview(config)
        append_to_log(heckle_text)

        # 4. Convert web format to API format
        print_f("Fetching node schemas...")
        schemas = requests.get(f"{COMFYUI}/object_info", timeout=30).json()
        print_f("Converting workflow to API format...")
        api_prompt = _workflow_to_api_prompt(workflow, schemas)
        
        # [THE PUSH] Save the API JSON to disk for the user/inspection
        try:
            with open(PROMPT_OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"prompt": api_prompt}, f, indent=2)
            print_f(f"API JSON pushed to: {PROMPT_OUTPUT_PATH}")
        except Exception as e:
            print_f(f"WARNING: Failed to push API JSON: {e}")

        # 5. Count treatments before submission
        treatments_before = count_treatments()

        # 6. Wait for clear queue before submitting
        for _qwait in range(60):  # up to 10 min (60 x 10s)
            try:
                q = requests.get(f"{COMFYUI}/queue", timeout=10).json()
                running = len(q.get("queue_running", []))
                pending = len(q.get("queue_pending", []))
                if running == 0 and pending == 0:
                    break
                print_f(f"Queue busy (running={running}, pending={pending}) -- waiting 10s...")
                time.sleep(POLL_S)
            except Exception:
                break  # if queue endpoint fails, just submit anyway

        client_id = str(uuid.uuid4())
        print_f("Submitting prompt...")

        # Try to serialize before sending to catch malformed JSON early
        try:
            payload = json.dumps({"prompt": api_prompt, "client_id": client_id})
            payload_size_mb = len(payload.encode('utf-8')) / (1024*1024)
            print_f(f"Payload size: {payload_size_mb:.2f} MB")
        except Exception as e:
            error_msg = f"JSON serialization failed: {e}"
            print_f(f"SUBMIT FAILED: {error_msg}")
            raise RuntimeError(error_msg)

        try:
            resp = requests.post(
                f"{COMFYUI}/prompt",
                json={"prompt": api_prompt, "client_id": client_id},
                timeout=30,
            )
        except requests.exceptions.Timeout:
            error_msg = "POST /prompt timed out after 30s -- ComfyUI may be hung"
            print_f(f"SUBMIT FAILED: {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"POST /prompt failed: {e}"
            print_f(f"SUBMIT FAILED: {error_msg}")
            raise RuntimeError(error_msg)

        if resp.status_code != 200:
            error_msg = f"HTTP {resp.status_code}: {resp.text[:300]}"
            print_f(f"SUBMIT FAILED: {error_msg}")
            raise RuntimeError(error_msg)

        prompt_id = resp.json().get("prompt_id")
        print_f(f"Submitted. prompt_id={prompt_id}")

        # 7. Poll for completion
        result = "TIMEOUT"
        poll_start = time.time()
        print_f(f"Polling (timeout {TIMEOUT_S}s)...")

        while time.time() - poll_start < TIMEOUT_S:
            try:
                hist = requests.get(
                    f"{COMFYUI}/history/{prompt_id}", timeout=10
                ).json()
                if prompt_id in hist:
                    status = hist[prompt_id].get("status", {})
                    if status.get("completed", False):
                        result = "SUCCESS"
                        break
                    if status.get("status_str") == "error":
                        result = "FAIL"
                        error_msg = str(status.get("messages",
                                                    "Workflow execution error"))[:300]
                        break
            except Exception:
                pass
            time.sleep(POLL_S)

        # 8. Check treatment file
        has_treatment, treatment_count = check_treatment(treatments_before)
        if result == "SUCCESS" and not has_treatment:
            print_f("WARNING: Episode completed but no treatment file written!")

    except Exception as e:
        result = "FAIL"
        error_msg = str(e)[:300]
        print_f(f"EXCEPTION: {error_msg}")
        has_treatment = False

    # 9. Gather runtime details
    duration = int(time.time() - start_time)
    title, dialogue, vram = get_runtime_details()

    # 10. Log
    log_entry = (
        f"### RUN {run_num:03d} -- {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- **Config:** {config_str if config else 'N/A'}\n"
        f"- **Result:** {result}\n"
        f"- **Duration:** {duration}s\n"
        f"- **Episode:** {title}\n"
        f"- **Treatment:** {'YES' if has_treatment else 'NO'}\n"
        f"- **Dialogue lines:** {dialogue}\n"
        f"- **VRAM peak:** {vram} GB\n"
        f"- **Error:** {error_msg if result != 'SUCCESS' else 'None'}\n"
        f"- **Notes:** Soak run {result.lower()}.\n"
    )
    append_to_log(log_entry)
    print_f(log_entry)

    # 10b. Critic review (haiku + Rotten Tomatoes) on success
    if result == "SUCCESS":
        review_text = critic_review(
            config, title, dialogue, vram, has_treatment, duration)
        append_to_log(review_text)

    # 10c. Treatment scan (read-only, flags only, never fixes)
    scan_flags = []
    if result == "SUCCESS":
        newest = find_newest_treatment()
        if newest:
            scan_flags = scan_treatment(newest)
            if scan_flags:
                scan_text = (
                    "\n" + "!" * 50 + "\n"
                    "  TREATMENT SCAN -- FLAGS DETECTED\n"
                    + "!" * 50 + "\n"
                    f"  File: {os.path.basename(newest)}\n"
                )
                for flag in scan_flags:
                    scan_text += f"  >> {flag}\n"
                scan_text += "!" * 50 + "\n"
                print_f(scan_text)
                append_to_log(scan_text)
            else:
                clean_msg = "  TREATMENT SCAN: Clean -- no flags.\n"
                print_f(clean_msg)
                append_to_log(clean_msg)

    # 10d. Performance review (lead developer suggestions, logged to soak_log)
    if result == "SUCCESS":
        review = performance_review(config, title, dialogue, vram,
                                    has_treatment, duration, scan_flags)
        append_to_log(review)

    # 11. Cooldown
    print_f(f"Cooldown: {COOLDOWN_S}s")
    time.sleep(COOLDOWN_S)
    return True


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(SOAK_LOG), exist_ok=True)
    append_to_log(f"\n--- NEW SOAK SESSION {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")