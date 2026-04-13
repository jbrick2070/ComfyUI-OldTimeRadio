"""
SIGNAL LOST -- Soak Test Operator
Runs episodes continuously via ComfyUI HTTP API with controlled
parameter randomization. Logs every run. Reboots ComfyUI if stalled.

Usage:
    python soak_operator.py

AntiGravity: just run this script. Do NOT modify it.
"""

import json, requests, time, random, uuid, os, subprocess, re, sys

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
