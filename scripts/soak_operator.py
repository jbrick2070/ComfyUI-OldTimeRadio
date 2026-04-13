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
COMFYUI_EXE = r"C:\Users\jeffr\AppData\Local\Programs\comfyui-electron\ComfyUI.exe"

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
CREATIVITIES = ["safe & tight", "balanced", "wild & rough"]
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
    """Print with immediate flush so AntiGravity sees output in real time."""
    print(msg)
    sys.stdout.flush()


def comfyui_alive():
    try:
        r = requests.get(f"{COMFYUI}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def reboot_comfyui():
    """Kill and restart ComfyUI Desktop. Wait for startup."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print_f(f"REBOOT: ComfyUI unresponsive -- initiating restart at {ts}")
    append_to_log(f"REBOOT: ComfyUI unresponsive -- initiating restart at {ts}")
    subprocess.run(["taskkill", "/F", "/IM", "ComfyUI.exe"], capture_output=True)
    time.sleep(15)
    subprocess.run(
        ["powershell", "-Command", f'Start-Process "{COMFYUI_EXE}"'],
        capture_output=True,
    )
    print_f("Waiting 90s for full startup...")
    time.sleep(90)


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

    return flags


# ---------------------------------------------------------------------------
# STATLER & WALDORF -- Balcony Preview (before each run)
# ---------------------------------------------------------------------------
HECKLES = {
    "hard_sci_fi": [
        "Statler: Hard sci-fi? The only hard part is staying awake!",
        "Waldorf: I once read hard sci-fi. Took me three naps to finish!",
        "Statler: They say hard sci-fi is scientifically accurate. Accurately boring!",
    ],
    "space_opera": [
        "Waldorf: Space opera? I prefer the regular opera -- at least I can sleep in a nice chair!",
        "Statler: The last space opera I saw, the best performance was the vacuum of space!",
        "Waldorf: Space opera -- where no one can hear you yawn!",
    ],
    "dystopian": [
        "Statler: A dystopian story? Just describe this theater!",
        "Waldorf: Dystopian? I have been living one since they put us in this balcony!",
        "Statler: Another dystopia. As if watching this show was not punishment enough!",
    ],
    "time_travel": [
        "Waldorf: Time travel? I wish I could travel to before I agreed to watch this!",
        "Statler: If I had a time machine, I would skip to the end credits!",
        "Waldorf: Time travel -- the only way to get those 30 minutes back!",
    ],
    "first_contact": [
        "Statler: First contact? The aliens took one look and left!",
        "Waldorf: First contact with this show? More like last contact!",
        "Statler: Even the aliens have better things to watch!",
    ],
    "cosmic_horror": [
        "Waldorf: Cosmic horror? The real horror is we are still up here watching!",
        "Statler: They say cosmic horror is unknowable. Just like why we keep coming back!",
        "Waldorf: Lovecraft never imagined anything as terrifying as this balcony seat!",
    ],
    "cyberpunk": [
        "Statler: Cyberpunk? I can barely work the remote!",
        "Waldorf: High tech, low life -- just like this theater!",
        "Statler: In cyberpunk, everyone has implants. I just need new knees!",
    ],
    "post_apocalyptic": [
        "Waldorf: Post-apocalyptic? Looks like the budget already survived one!",
        "Statler: After the apocalypse, only two things survive -- cockroaches and us!",
        "Waldorf: Post-apocalyptic -- finally a setting that matches this theater!",
    ],
}

WORD_HECKLES = {
    350: "Statler: 350 words? That is shorter than my grocery list!",
    700: "Waldorf: 700 words? Just enough rope to hang itself!",
    1050: "Statler: 1050 words? Getting ambitious, are we?",
    1400: "Waldorf: 1400 words? I will need a second nap for that!",
    2100: "Statler: 2100 words? They are writing a novel up there! Waldorf: A novel way to bore us!",
}

LENGTH_HECKLES = {
    "short (3 acts)": "Waldorf: Three acts? Even Shakespeare needed five! Statler: Shakespeare also had talent!",
    "medium (5 acts)": "Statler: Five acts. The perfect number -- of chances to walk out!",
    "long (7-8 acts)": "Waldorf: Seven acts? Statler: Wake me for the curtain call!",
}


def balcony_preview(config):
    """Statler and Waldorf heckle the upcoming episode from the balcony."""
    genre = config["genre"]
    words = config["words"]
    length = config["length"]

    lines = []
    lines.append("")
    lines.append("*" * 50)
    lines.append("  FROM THE BALCONY  --  Statler & Waldorf")
    lines.append("*" * 50)
    lines.append("")

    # Genre heckle
    genre_lines = HECKLES.get(genre, ["Statler: What is this? Waldorf: I have no idea!"])
    lines.append(random.choice(genre_lines))

    # Word count heckle
    wh = WORD_HECKLES.get(words, f"Waldorf: {words} words? That is a number, all right!")
    lines.append(wh)

    # Length heckle
    lh = LENGTH_HECKLES.get(length, "Statler: However many acts, it is too many!")
    lines.append(lh)

    # Closing zinger
    closers = [
        "Both: Dohohoho!",
        "Waldorf: Why do we keep coming back? Statler: Beats me!",
        "Statler: Well, here we go again! Waldorf: Same time, same suffering!",
        "Both: Bravo! ...Just kidding!",
        "Waldorf: Ready? Statler: I was born ready. Ready to leave!",
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

    tomato = "🍅" if score >= 60 else "🤢"
    lines.append(f"  Rotten Tomatoes: {score}% -- {label} {tomato}")
    lines.append(f"  Config: {config.get('genre', '?')} / {config.get('words', '?')}w / {config.get('length', '?')}")
    lines.append(f"  Dialogue: {dialogue} lines | VRAM: {vram} GB | Duration: {duration}s")
    lines.append("-" * 50)
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

        # 3b. Balcony preview (Statler & Waldorf heckle)
        heckle_text = balcony_preview(config)
        append_to_log(heckle_text)

        # 4. Convert web format to API format
        print_f("Fetching node schemas...")
        schemas = requests.get(f"{COMFYUI}/object_info", timeout=30).json()
        print_f("Converting workflow to API format...")
        api_prompt = _workflow_to_api_prompt(workflow, schemas)

        # 5. Count treatments before submission
        treatments_before = count_treatments()

        # 6. Submit
        client_id = str(uuid.uuid4())
        print_f("Submitting prompt...")
        resp = requests.post(
            f"{COMFYUI}/prompt",
            json={"prompt": api_prompt, "client_id": client_id},
            timeout=30,
        )

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

    # Resume run numbering from existing log
    run_num = 1
    if os.path.exists(SOAK_LOG):
        with open(SOAK_LOG, "r", encoding="utf-8") as f:
            runs = re.findall(r"### RUN (\d+)", f.read())
            if runs:
                run_num = int(runs[-1]) + 1

    print_f(f"SIGNAL LOST Soak Operator starting at run {run_num}")
    print_f(f"Workflow: {WORKFLOW_PATH}")
    print_f(f"Soak log: {SOAK_LOG}")
    print_f(f"Randomizing: genre, target_words, target_length, style, creativity")
    print_f("")

    while True:
        try:
            if not run_iteration(run_num):
                print_f("CRITICAL: Stopping soak after unrecoverable failure.")
                break
            run_num += 1
        except KeyboardInterrupt:
            print_f(f"\nSoak stopped by user at run {run_num}.")
            append_to_log(f"\n--- SOAK STOPPED BY USER at run {run_num} "
                          f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            break
