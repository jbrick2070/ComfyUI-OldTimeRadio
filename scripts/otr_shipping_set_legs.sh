#!/usr/bin/env bash
# Run shipped graphs through a live ComfyUI, one act each -- the POSIX twin of
# scripts/otr_shipping_set_legs.ps1, for the Mac and Linux boxes.
#
#   scripts/otr_shipping_set_legs.sh <comfyui-url> <obs-dir> <python> <graph> [<graph> ...]
#
#   OTR_ACT_COUNT    acts per leg (default 1 -- a smoke of every graph fits a night)
#   OTR_LEG_TIMEOUT  seconds the runner watches one leg (default 9000; 0 waits
#                    until the render reaches a terminal result, which is the
#                    right setting for a long lane -- see the LAST LEG note below)
#   OTR_SOURCE_BANK  pin every leg to ONE bank instead of letting each roll.
#                    Unset (the default) leaves the graph's saved value alone,
#                    which for all 17 shipped graphs is the roll sentinel.
#                    Added 2026-09-14: proving a single lane meant abandoning
#                    this harness, because six legs rolling uniformly over six
#                    banks is not a test of any one of them.
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
# The final graph in this run -- the cleanup below must never interrupt it.
# Counted by POSITION, not matched by NAME: a caller may legitimately list the
# same graph twice (re-running one lane back to back), and a name match would
# call the FIRST occurrence the last leg and skip the queue cleanup that the
# second occurrence still needs.
# Count only the legs that will ACTUALLY RUN. Counting the requested list
# instead is a trap: a graph whose variant file is missing is skipped below, so
# if the LAST name in the list does not exist, no leg ever matches "last", and
# the real final render gets /interrupt'd by the cleanup -- which is the exact
# 2.5-hour loss the LAST LEG rule was written to stop.
TOTAL_LEGS=0
for _g in "$@"; do
  [ -f "$ROOT/workflows/variants/$_g.json" ] && TOTAL_LEGS=$((TOTAL_LEGS + 1))
done
LEG_INDEX=0
TIMEOUT="${OTR_LEG_TIMEOUT:-9000}"
BANK="${OTR_SOURCE_BANK:-}"
# Collected as an ARRAY so an unset bank contributes no argument at all.
# Passing `--source-bank ""` would pin the bank to the empty string, which is a
# different and much worse thing than not pinning it.
BANK_ARGS=()
[ -n "$BANK" ] && BANK_ARGS=(--source-bank "$BANK")
# Expanded below as ${BANK_ARGS[@]+"${BANK_ARGS[@]}"}, NOT as a bare
# "${BANK_ARGS[@]}". `set -u` is on (line 16) and bash before 4.4 treats the
# expansion of an EMPTY array as an unbound variable and aborts the script.
# macOS still ships bash 3.2.57 as /bin/bash and this file's whole reason to
# exist is the Mac and Linux boxes, so the unguarded form would abort exactly
# where the POSIX twin is needed. `${x[@]+...}` is the portable idiom;
# "${x[@]:-}" is NOT a substitute -- it yields one EMPTY ARGUMENT instead of
# no argument, which the runner would read as an empty --workflow value.
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/otr/legs/shipping_set_$STAMP"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.txt"
note() { echo "$*" | tee -a "$SUMMARY"; }
if [ -n "$BANK" ]; then bank_note="source_bank=$BANK (PINNED)"; else bank_note="source_bank=per-graph (roll)"; fi
note "shipping-set legs  $STAMP  url=$URL  act_count=$ACTS  obs=$OBS  $bank_note"
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
  # AFTER the skip, so the index counts legs that run, not names requested.
  LEG_INDEX=$((LEG_INDEX + 1))
  log="$LOGDIR/$g.log"
  t0=$(date +%s)
  marker="$LOGDIR/.$g.started"; touch "$marker"
  note "$g  START  $(date +%H:%M:%S)"
  "$PY" scripts/otr_canonical_api_run.py --workflow "$wf" --act-count "$ACTS" \
        --comfyui-url "$URL" --timeout "$TIMEOUT" \
        ${BANK_ARGS[@]+"${BANK_ARGS[@]}"} > "$log" 2>&1
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
    # NEVER CLEAR ON THE LAST LEG. On 2026-09-14 the PowerShell twin killed a
    # live LTX 2.5 render this way: it hit the observation window as the FINAL
    # graph, the runner said plainly "BUT THE RENDER IS STILL ALIVE ... the
    # episode should still publish to otr/obs on its own", and the cleanup
    # POSTed /interrupt anyway. Two and a half hours, no episode, to protect a
    # next leg that did not exist. A timeout means "I stopped watching", not
    # "it is wedged".
    if [ "$LEG_INDEX" -eq "$TOTAL_LEGS" ]; then
      note "$g  LAST LEG -- leaving the render alone. It may still be running and may still publish to $OBS on its own; check there before calling this a failure. Re-run with a larger OTR_LEG_TIMEOUT (or the runner's --timeout 0) to watch it to a terminal result."
    else
      note "$g  $(clear_server 2>&1 | tail -n 1)"
      sleep 20
    fi
  fi
done
note "DONE  $(date +%H:%M:%S)"
