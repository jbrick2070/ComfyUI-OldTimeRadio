import os
import sys
import time
import re

LOG_FILE = "otr_runtime.log"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Waiting for {LOG_FILE} to be created...")
        while not os.path.exists(LOG_FILE):
            time.sleep(1)

    # Force UTF-8 for Windows Console to prevent Emoji crashes
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    # Enable ANSI escape codes on Windows 10+
    if os.name == 'nt':
        os.system("")

    print(f"\033[92m[VRAM Telemetry Dashboard Initializing...]\033[0m")
    
    current_phase = "IDLE"
    vram_current = "0.00"
    vram_peak = "0.00"
    speed_tok_s = "0.00"
    last_action = "Waiting for generation to begin..."
    
    # Regex patterns
    re_vram = re.compile(r"VRAM_SNAPSHOT phase=(\S+)\s+current_gb=([0-9.]+)\s+peak_gb=([0-9.]+)")
    re_speed = re.compile(r"DONE:\s+.*?([0-9.]+)\s+tok/s")
    re_zero_prime = re.compile(r"\[StoryOrchestrator\] Zero-Prime VRAM State: ([0-9.]+)GB Free. Capacity: ([0-9.]+)GB")
    
    zero_prime_free = "N/A"
    zero_prime_cap = "N/A"
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        # Pre-process history so we immediately show the last known state
        lines = f.readlines()
        for line in lines:
            line_str = str(line)
            if "VRAM_SNAPSHOT" in line_str:
                m = re_vram.search(line_str)
                if m:
                    current_phase = m.group(1)
                    vram_current = m.group(2)
                    vram_peak = m.group(3)
            elif "tok/s" in line_str and "DONE:" in line_str:
                m = re_speed.search(line_str)
                if m:
                    speed_tok_s = m.group(1)
            elif "Zero-Prime VRAM State:" in line_str:
                m = re_zero_prime.search(line_str)
                if m:
                    zero_prime_free = m.group(1)
                    zero_prime_cap = m.group(2)
            else:
                clean_line = line_str.strip()
                if clean_line and any(k in clean_line for k in ("ScriptWriter", "Director", "VOICE_", "LLM", "Audio")):
                    parts = clean_line.split("]", 1)
                    if len(parts) == 2:
                        last_action = parts[1].strip()[:90]
                    else:
                        last_action = clean_line[:90]
                    
        f.seek(0, 2) # Move cursor to the end
        
        # Render initial screen
        render_dashboard(current_phase, speed_tok_s, last_action, vram_current, vram_peak, zero_prime_free, zero_prime_cap)

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
                
            line = str(line).strip()
            updated = False
            
            if "VRAM_SNAPSHOT" in line:
                m = re_vram.search(line)
                if m:
                    current_phase = m.group(1)
                    vram_current = m.group(2)
                    vram_peak = m.group(3)
                    updated = True
            elif "tok/s" in line and "DONE:" in line:
                m = re_speed.search(line)
                if m:
                    speed_tok_s = m.group(1)
                    updated = True
            elif "Zero-Prime VRAM State:" in line:
                m = re_zero_prime.search(line)
                if m:
                    zero_prime_free = m.group(1)
                    zero_prime_cap = m.group(2)
                    updated = True
            elif line and any(k in line for k in ("ScriptWriter", "Director", "VOICE_", "LLM", "Audio", "ARC_")):
                parts = line.split("]", 1)
                if len(parts) == 2:
                    last_action = parts[1].strip()[:90]
                else:
                    last_action = line[:90]
                updated = True
                
            if updated:
                render_dashboard(current_phase, speed_tok_s, last_action, vram_current, vram_peak, zero_prime_free, zero_prime_cap)

def render_dashboard(current_phase, speed_tok_s, last_action, vram_current, vram_peak, zero_prime_free, zero_prime_cap):
    clear_screen()
    print(f"\033[1;36m=================================================================\033[0m")
    print(f"\033[1;33m       📻 OLD TIME RADIO - LIVE TELEMETRY DASHBOARD 📻\033[0m")
    print(f"\033[1;36m=================================================================\033[0m")
    
    print(f"\n⚡ \033[1mPIPELINE STATE\033[0m")
    print(f"   Current Phase       : \033[92m{current_phase}\033[0m")
    print(f"   Last LLM Speed      : \033[95m{speed_tok_s} tokens/sec\033[0m")
    print(f"   Latest Action       : \033[37m{last_action}\033[0m")
    
    print(f"\n🧠 \033[1mVRAM HARDENING METRICS\033[0m")
    
    try:
        peak_f = float(vram_peak)
    except:
        peak_f = 0.0
        
    peak_color = "\033[92m" # Green (Safe)
    if peak_f > 13.0: peak_color = "\033[93m" # Yellow (Sovereignty Buffer hit)
    if peak_f > 15.0: peak_color = "\033[91m" # Red (Danger Close)
    
    print(f"   Snapshot Current    : \033[96m{vram_current} GB\033[0m")
    print(f"   Lifecycle Peak      : {peak_color}{vram_peak} GB\033[0m")
    print(f"   Z-Prime Baseline    : {zero_prime_free} GB Free (Max: {zero_prime_cap} GB)")
    
    print(f"\n\033[90m(Monitoring otr_runtime.log... Press Ctrl+C to exit)\033[0m")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\033[91mExiting Telemetry Dashboard.\033[0m")
