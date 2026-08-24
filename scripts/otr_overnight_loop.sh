#!/usr/bin/env bash
# Overnight supervisor: re-run the writer bank gate across all banks,
# forever, until stopped. Each pass proves the canonical workflow end to
# end and publishes every successful leg to otr/obs (the success signal --
# CLAUDE.md, "a test is not complete unless published to obs").
#
# Started 2026-08-24 to prove the PBUG-20260802-02 fix
# (nodes/_otr_cast_coverage_repair.py) live: the writer bank gate's
# selection_mode=random redraws a fresh shakespeare scene every pass, so
# repeated runs are what eventually re-hits the Malvolio's-letter
# scene/budget combination that failed once already tonight.
#
# Server-death recovery follows CLAUDE.md section 4 exactly: selective kill
# by CommandLine (never a blanket python kill -- it would sever this
# session's own MCP tooling), confirm port 8000 and VRAM are back to
# baseline, then reboot via the documented launcher.
set -u
REPO="/c/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio"
PY="/c/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe"
LOG="$REPO/tmp/otr_overnight_loop.log"
OBS="/c/Users/jeffr/Documents/ComfyUI/output/otr/obs"
LAUNCH="C:\\Users\\jeffr\\Documents\\ComfyUI\\custom_nodes\\ComfyUI-OldTimeRadio\\scripts\\_otr_soak_server_launch.cmd"
BOOT_LOG="$REPO/tmp/otr_overnight_server_boot.log"

mkdir -p "$REPO/tmp"
echo "=== overnight loop started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"

pass_n=0
while true; do
  pass_n=$((pass_n + 1))

  # Health check before every pass -- reboot only if genuinely dead.
  if ! curl -s -m 8 http://127.0.0.1:8000/queue > /dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%SZ)] pass $pass_n: server unresponsive, resetting per section 4" >> "$LOG"
    powershell -NoProfile -Command '
      Get-CimInstance Win32_Process -Filter "Name='"'"'python.exe'"'"'" |
        Where-Object { $_.CommandLine -match "ComfyUI-OldTimeRadio|main\.py.*--port 8000|otr_writer_bank_gate|otr_canonical_api_run" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    ' >> "$LOG" 2>&1
    sleep 5
    if [ -z "$(netstat -ano 2>/dev/null | grep ':8000' | grep -i listen)" ]; then
      powershell -NoProfile -Command "Start-Process -FilePath \"$LAUNCH\" -ArgumentList \"\`\"$BOOT_LOG\`\"\" -WindowStyle Hidden" >> "$LOG" 2>&1
      for i in $(seq 1 30); do
        curl -s -m 3 http://127.0.0.1:8000/queue > /dev/null 2>&1 && break
        sleep 2
      done
    fi
    if ! curl -s -m 8 http://127.0.0.1:8000/queue > /dev/null 2>&1; then
      echo "[$(date -u +%H:%M:%SZ)] pass $pass_n: reboot FAILED, sleeping 5m before retry" >> "$LOG"
      sleep 300
      continue
    fi
    echo "[$(date -u +%H:%M:%SZ)] pass $pass_n: server back up" >> "$LOG"
  fi

  obs_before=$(ls "$OBS" 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%SZ)] pass $pass_n: launching bank gate (obs=$obs_before)" >> "$LOG"

  cd "$REPO"
  export PYTHONUTF8=1
  "$PY" scripts/otr_writer_bank_gate.py --acts 1 >> "$LOG" 2>&1
  gate_exit=$?

  obs_after=$(ls "$OBS" 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%SZ)] pass $pass_n: gate exit=$gate_exit obs=$obs_before->$obs_after" >> "$LOG"

  # PRESERVE THE EVIDENCE. The bank gate writes tmp/_bankgate_<bank>.log and
  # OVERWRITES it every pass, so a failure's actual reason survives only until
  # the next pass touches that bank. The 2026-08-24 overnight run lost the
  # reason for two of six scifi_news_pro failures exactly this way -- the
  # durations were still readable, the errors were not. Archive per pass so a
  # 60%-failure lane can be diagnosed from ONE loop instead of re-run.
  archive="$REPO/tmp/legs/pass$(printf '%03d' "$pass_n")"
  mkdir -p "$archive"
  for leg in "$REPO"/tmp/_bankgate_*.log; do
    [ -e "$leg" ] && cp -p "$leg" "$archive/" 2>/dev/null
  done

  # LOCK is self-clearing on a clean gate exit; a stale one from a killed
  # pass would refuse every future pass forever, so clear it defensively.
  rm -f "$REPO/tmp/_writer_bank_gate.lock" 2>/dev/null

  sleep 30
done
