r"""
OTR Monitor — Live Dashboard for SIGNAL LOST Renders
=====================================================

Standalone watchdog you run in a separate terminal during a render.
Provides real-time visibility into what's happening inside ComfyUI
without touching the UI.

Three daemon threads run in parallel:
  1. LOG TAILER    — tails comfyui_8000.log for state transitions
                     (prompt received → executing → complete/crashed)
  2. WS LISTENER  — connects to ComfyUI's WebSocket for structured
                     node-execution events and progress percentages
  3. HEARTBEAT    — tails otr_runtime.log written by GemmaHeartbeatStreamer
                     and BatchBark for live script/TTS progress

All three update a shared status dict, which is flushed to
otr_dashboard.json after every change so external tools (or you)
can poll a single file.

Usage:
    python otr_monitor.py              # auto-detects port from log filename
    python otr_monitor.py --port 8188  # explicit port override

v1.0  2026-04-04  Jeffrey Brick
"""

import argparse
import json
import os
import re
import time
import threading
import logging
from datetime import datetime

log = logging.getLogger("OTR.Monitor")

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# Auto-detect the log file — try port 8000 first (user's config), then 8188
_LOG_CANDIDATES = [
    os.path.join(COMFY_ROOT, "user", "comfyui_8000.log"),
    os.path.join(COMFY_ROOT, "user", "comfyui_8188.log"),
    os.path.join(COMFY_ROOT, "user", "comfyui.log"),
]
LOG_PATH = next((p for p in _LOG_CANDIDATES if os.path.exists(p)), _LOG_CANDIDATES[0])

DASHBOARD_PATH = os.path.join(BASE_DIR, "otr_dashboard.json")
RUNTIME_LOG_PATH = os.path.join(BASE_DIR, "otr_runtime.log")


# ─────────────────────────────────────────────────────────────────────────────
# OTR NODE ID → FRIENDLY NAME MAPPING
# Matches the node IDs in otr_lite_prompt.json / otr_prompt_final.json
# ─────────────────────────────────────────────────────────────────────────────

_NODE_NAMES = {
    "1":  "Gemma4 ScriptWriter",
    "2":  "Gemma4 Director",
    "3":  "Scene Sequencer",
    "4":  "Audio Enhance",
    "5":  "SFX: Opening Theme",
    "6":  "SFX: Closing Theme",
    "7":  "Episode Assembler",
    "10": "Preview Audio",
    "11": "Batch Bark Generator",
}


class OTRMonitor:
    """Live dashboard for SIGNAL LOST renders.

    Watches three data sources simultaneously and writes a combined
    status JSON that reflects the real-time state of the render.
    """

    def __init__(self, ws_port=8000):
        self.ws_port = ws_port
        self.status = {
            "last_update": "",
            "state": "IDLE",
            "current_node": "None",
            "current_node_name": "None",
            "progress": 0,
            "elapsed_sec": 0,
            "last_heartbeat": "None",
            "bark_progress": "—",
            "news_headline": "—",
            "last_error": "None",
            "error_count": 0,
            "active_run": False,
        }
        self.running = True
        self._run_start = None  # timestamp when render started
        self._lock = threading.Lock()

    # ── Dashboard writer ─────────────────────────────────────────────────

    def _update_dashboard(self):
        """Flush current status to otr_dashboard.json (atomic-ish write)."""
        with self._lock:
            self.status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update elapsed time if a run is active
            if self._run_start and self.status["active_run"]:
                self.status["elapsed_sec"] = int(time.time() - self._run_start)

            tmp = DASHBOARD_PATH + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.status, f, indent=2)
                os.replace(tmp, DASHBOARD_PATH)  # atomic on Windows + POSIX
            except OSError:
                pass  # non-fatal: dashboard is advisory

    # ── Thread 1: ComfyUI log tailer ─────────────────────────────────────

    def watch_logs(self):
        """Tail the ComfyUI log for state transitions and errors.

        Waits for the log file to appear (ComfyUI may start after this
        script). Once found, seeks to end and tails new lines.
        """
        # Wait for log file to exist
        while self.running and not os.path.exists(LOG_PATH):
            time.sleep(2)

        if not self.running:
            return

        log.info("Tailing: %s", LOG_PATH)

        with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, 2)  # jump to end

            while self.running:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue

                self._process_log_line(line)

    def _process_log_line(self, line):
        """Parse a single ComfyUI log line and update state."""
        # ── State transitions ────────────────────────────────────────
        if "got prompt" in line:
            self.status["state"] = "STARTING"
            self.status["active_run"] = True
            self.status["progress"] = 0
            self._run_start = time.time()

        elif "Prompt executed" in line:
            self.status["state"] = "COMPLETE"
            self.status["active_run"] = False

        elif "Interrupting prompt" in line:
            self.status["state"] = "INTERRUPTED"
            self.status["active_run"] = False

        # ── Error detection (precise: avoid false-positives) ─────────
        # Only match lines that look like real errors, not field names
        elif re.search(r'(?:^|\s)(?:Traceback|RuntimeError|TypeError|ValueError'
                       r'|KeyError|AttributeError|FileNotFoundError'
                       r'|ImportError|OSError|CUDA out of memory)', line):
            self.status["state"] = "CRASHED"
            self.status["last_error"] = line.strip()[:200]
            self.status["error_count"] += 1

        # ── OTR-specific log parsing ─────────────────────────────────
        # News headline
        m = re.search(r'\[Gemma4ScriptWriter\] News seed:\s*(.+?)(?:\s*\|)', line)
        if m:
            self.status["news_headline"] = m.group(1).strip()[:120]

        # BatchBark progress: "BatchBark: 5/12 lines complete"
        m = re.search(r'BatchBark.*?(\d+)/(\d+)\s*lines', line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            self.status["bark_progress"] = f"{done}/{total} lines"
            if total > 0:
                self.status["progress"] = int(100 * done / total)

        # BatchBark final: "Generated 12/12 lines"
        m = re.search(r'\[BatchBark\] Generated (\d+)/(\d+) lines', line)
        if m:
            self.status["bark_progress"] = f"{m.group(1)}/{m.group(2)} lines (done)"

        self._update_dashboard()

    # ── Thread 2: WebSocket listener ─────────────────────────────────────

    def watch_ws(self):
        """Connect to ComfyUI's WebSocket for node-execution events.

        Auto-reconnects on disconnect with exponential backoff (1s → 30s).
        Gracefully degrades if websocket-client is not installed.
        """
        try:
            import websocket
        except ImportError:
            log.warning(
                "websocket-client not installed — WS monitoring disabled. "
                "Install with: pip install websocket-client"
            )
            return

        backoff = 1
        ws_url = f"ws://127.0.0.1:{self.ws_port}/ws"

        while self.running:
            try:
                log.info("Connecting to WebSocket: %s", ws_url)

                def on_message(ws, message):
                    try:
                        data = json.loads(message)
                    except (json.JSONDecodeError, TypeError):
                        return

                    msg_type = data.get("type", "")

                    if msg_type == "executing":
                        node_id = data.get("data", {}).get("node")
                        if node_id:
                            self.status["current_node"] = node_id
                            self.status["current_node_name"] = _NODE_NAMES.get(
                                str(node_id), f"Node {node_id}"
                            )
                            self.status["state"] = "EXECUTING"
                        else:
                            self.status["current_node"] = "Idle"
                            self.status["current_node_name"] = "Idle"
                        self._update_dashboard()

                    elif msg_type == "progress":
                        d = data.get("data", {})
                        val = d.get("value", 0)
                        max_val = d.get("max", 1)
                        if max_val > 0:
                            self.status["progress"] = int(100 * val / max_val)
                        self._update_dashboard()

                def on_error(ws, error):
                    log.debug("WS error: %s", error)

                def on_close(ws, close_status, close_msg):
                    log.debug("WS closed (status=%s)", close_status)

                def on_open(ws):
                    log.info("WebSocket connected to port %d", self.ws_port)
                    nonlocal backoff
                    backoff = 1  # reset on successful connect

                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open,
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)

            except Exception as e:
                log.debug("WS connection failed: %s", e)

            if not self.running:
                break

            # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s max
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    # ── Thread 3: Runtime heartbeat log tailer ───────────────────────────

    def watch_heartbeat(self):
        """Tail otr_runtime.log for GemmaHeartbeatStreamer + BatchBark pulses.

        This file is written by the nodes themselves during generation.
        Each line is timestamped: [HH:MM:SS] ScriptWriter: Voice: HAYES, male...
        """
        # Wait for the file to appear
        while self.running and not os.path.exists(RUNTIME_LOG_PATH):
            time.sleep(2)

        if not self.running:
            return

        log.info("Tailing heartbeat: %s", RUNTIME_LOG_PATH)

        with open(RUNTIME_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, 2)  # jump to end

            while self.running:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue

                line = line.strip()
                if not line:
                    continue

                self.status["last_heartbeat"] = line[:150]

                # Parse structured heartbeats
                # BatchBark progress: "BatchBark: 8/20 lines complete"
                m = re.search(r'BatchBark.*?(\d+)/(\d+)\s*lines', line)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    self.status["bark_progress"] = f"{done}/{total} lines"

                # ScriptWriter act progress: "Generating Act 2/3"
                m = re.search(r'Generating Act (\d+)/(\d+)', line)
                if m:
                    self.status["state"] = f"WRITING ACT {m.group(1)}/{m.group(2)}"

                self._update_dashboard()

    # ── Main entry point ─────────────────────────────────────────────────

    def start(self):
        """Launch all three watcher threads and block until Ctrl+C."""
        print(f"╔══════════════════════════════════════════════════════════╗")
        print(f"║  OTR Monitor — SIGNAL LOST Render Dashboard             ║")
        print(f"╠══════════════════════════════════════════════════════════╣")
        print(f"║  Log:       {os.path.basename(LOG_PATH):42} ║")
        print(f"║  Heartbeat: {os.path.basename(RUNTIME_LOG_PATH):42} ║")
        print(f"║  Dashboard: {os.path.basename(DASHBOARD_PATH):42} ║")
        print(f"║  WebSocket: ws://127.0.0.1:{self.ws_port}/ws{' ' * (23 - len(str(self.ws_port)))}║")
        print(f"╚══════════════════════════════════════════════════════════╝")
        print()

        threads = [
            threading.Thread(target=self.watch_logs,      name="LogTailer",  daemon=True),
            threading.Thread(target=self.watch_ws,         name="WSListener", daemon=True),
            threading.Thread(target=self.watch_heartbeat,  name="Heartbeat",  daemon=True),
        ]
        for t in threads:
            t.start()

        try:
            # Print live status to terminal every 5 seconds
            while True:
                time.sleep(5)
                self._print_status()
        except KeyboardInterrupt:
            self.running = False
            print("\nMonitor stopped.")

    def _print_status(self):
        """Print a compact one-line status to the terminal."""
        s = self.status
        elapsed = ""
        if s["elapsed_sec"] > 0:
            m, sec = divmod(s["elapsed_sec"], 60)
            elapsed = f" [{m}m{sec:02d}s]"

        node = s["current_node_name"]
        if node == "None":
            node = "—"

        bark = s["bark_progress"]
        state = s["state"]
        pct = s["progress"]

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {state:14} | {node:24} | {pct:3d}% | Bark: {bark}{elapsed}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _detect_port():
    """Try to detect ComfyUI port from the log filename."""
    for candidate in _LOG_CANDIDATES:
        if os.path.exists(candidate):
            m = re.search(r'comfyui_(\d+)\.log', candidate)
            if m:
                return int(m.group(1))
    return 8000  # default


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTR Monitor — SIGNAL LOST render dashboard")
    parser.add_argument("--port", type=int, default=None,
                        help="ComfyUI WebSocket port (auto-detected from log filename if omitted)")
    args = parser.parse_args()

    port = args.port or _detect_port()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    monitor = OTRMonitor(ws_port=port)
    monitor.start()
