#!/usr/bin/env bash
# Run shipped graphs through a live ComfyUI, one act each -- the POSIX twin of
# scripts/otr_shipping_set_legs.ps1, for the Mac and Linux boxes.
#
#   scripts/otr_shipping_set_legs.sh <comfyui-url> <obs-dir> <python> <graph> [<graph> ...]
#
#   OTR_ACT_COUNT    acts per leg (default 1 -- a smoke of every graph fits a night)
#   OTR_LEG_TIMEOUT  seconds the runner watches one leg (default 9000)
#
# One log per leg under otr/legs/shipping_set_<stamp>/ and a SUMMARY.txt whose
# lines read RESULT, minutes, and the episodes that landed in <obs-dir> during
# the leg -- otr/obs is the success signal, a green log is not. After a leg
# that did not reach SUCCESS the server queue is cleared so the next leg does
# not queue behind a wedged render. Never passes --title: the harness label
# would become the on-screen title card.
set -u
if [ "$#" -lt 4 ]; then
  echo "usage: $0 <comfyui-url> <obs-dir> <python> <graph> [<graph> ...]" >&2
  exit 64
fi
URL="${1%/}"; OBS="$2"; PY="$3"; shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
ACTS="${OTR_ACT_COUNT:-1}"
TIMEOUT="${OTR_LEG_TIMEOUT:-9000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/otr/legs/shipping_set_$STAMP"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.txt"
note() { echo "$*" | tee -a "$SUMMARY"; }
note "shipping-set legs  $STAMP  url=$URL  act_count=$ACTS  obs=$OBS"
note "graphs: $*"

clear_server() {
  # Drop every pending prompt and interrupt the running one.
  "$PY" - "$URL" <<'PYEOF'
import json, sys, urllib.request
url = sys.argv[1]
def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method="POST" if data is not None else "GET",
                                 headers={"User-Agent": "otr-legs/1.0", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()
q = json.loads(call("/queue"))
pending = [item[1] for item in q.get("queue_pending", [])]
if pending:
    call("/queue", {"delete": pending})
if q.get("queue_running"):
    call("/interrupt", {})
print("cleared the server (was running=%d pending=%d)" % (len(q.get("queue_running", [])), len(pending)))
PYEOF
}

for g in "$@"; do
  wf="$ROOT/workflows/variants/$g.json"
  if [ ! -f "$wf" ]; then
    note "$g  SKIP  no such variant"
    continue
  fi
  log="$LOGDIR/$g.log"
  t0=$(date +%s)
  marker="$LOGDIR/.$g.started"; touch "$marker"
  note "$g  START  $(date +%H:%M:%S)"
  "$PY" scripts/otr_canonical_api_run.py --workflow "$wf" --act-count "$ACTS" --comfyui-url "$URL" --timeout "$TIMEOUT" > "$log" 2>&1
  rc=$?
  mins=$(( ( $(date +%s) - t0 ) / 60 ))
  result=$(grep -aoE "RESULT (SUCCESS|FAIL[A-Z_]*|TIMEOUT|ERROR)" "$log" | tail -n 1)
  [ -n "$result" ] || result="NO-RESULT-LINE"
  if [ -d "$OBS" ]; then
    landed=$(find "$OBS" -maxdepth 1 -name '*.mp4' -newer "$marker" -exec basename {} \; | sort | tr '\n' ' ')
    count=$(printf '%s' "$landed" | wc -w | tr -d ' ')
    note "$g  $result  rc=$rc  ${mins}min  obs=$count  $landed"
  else
    note "$g  $result  rc=$rc  ${mins}min  obs=n/a (no such folder: $OBS)"
  fi
  if [ "$result" != "RESULT SUCCESS" ]; then
    note "$g  $(clear_server 2>&1 | tail -n 1)"
    sleep 20
  fi
done
note "DONE  $(date +%H:%M:%S)"
