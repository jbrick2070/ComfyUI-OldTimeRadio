#!/usr/bin/env bash
# Continuously run canonical one-act qualifications for every public w45
# profile. A failed lane is recorded and retried on the next round.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=otr_pod_runtime.sh
source "$SCRIPT_DIR/otr_pod_runtime.sh"
otr_load_runtime || exit 1

RESULTS="${OTR_SOAK_RESULTS:-/root/soak_results.txt}"
ACTS="${SOAK_ACTS:-1}"
MAX_ROUNDS="${SOAK_MAX_ROUNDS:-0}"
[[ "$ACTS" =~ ^[1-5]$ ]] || { echo "SOAK_ACTS must be 1..5" >&2; exit 2; }
[[ "$MAX_ROUNDS" =~ ^[0-9]+$ ]] \
  || { echo "SOAK_MAX_ROUNDS must be zero or a positive integer" >&2; exit 2; }

ROSTER_FILE=$(mktemp) || exit 1
if ! otr_profile_roster > "$ROSTER_FILE"; then
  rm -f "$ROSTER_FILE"
  exit 1
fi
otr_roster_preflight "$ROSTER_FILE" || { rm -f "$ROSTER_FILE"; exit 1; }
mapfile -t PROFILES < "$ROSTER_FILE"
rm -f "$ROSTER_FILE"

echo "=== OTR POD SOAK START $(date -u '+%F %H:%M:%SZ') acts=$ACTS profiles=${#PROFILES[@]} ===" \
  | tee -a "$RESULTS"
round=0
while true; do
  round=$((round + 1))
  echo "--- ROUND $round $(date -u '+%H:%M:%SZ') ---" | tee -a "$RESULTS"
  for profile in "${PROFILES[@]}"; do
    log="/root/soak_r${round}_${profile}.log"
    if ! otr_ensure_profile_server "$profile"; then
      printf "  r%-3s %-38s BOOT_FAILED\n" "$round" "$profile" | tee -a "$RESULTS"
      continue
    fi
    t0=$(date +%s)
    (
      cd "$OTR_REPO_ROOT" || exit 1
      "$COMFY_PY" scripts/otr_canonical_api_run.py \
        --profile "$profile" --act-count "$ACTS" \
        --source-bank original --visual-style sci_fi_radio --timeout 0
    ) > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    result=$(grep -oE 'RESULT (SUCCESS|FAIL)' "$log" | tail -1 || true)
    reason=$(grep -oE \
      'ERROR node [0-9]+ \([A-Za-z_]+\)|is not usable for role [a-z_]+: [a-z_]+' \
      "$log" | head -1 | cut -c1-80 || true)
    printf "  r%-3s %-38s %-14s %5ds rc=%s %s\n" \
      "$round" "$profile" "${result:-NO_RESULT}" "$((t1-t0))" "$rc" "$reason" \
      | tee -a "$RESULTS"
    if ! otr_ready; then
      echo "  server stopped responding; recovering with the same profile" | tee -a "$RESULTS"
      otr_boot_profile "$profile" || true
    fi
    # Keep only the last three logs for this exact profile.
    mapfile -t old_logs < <(
      find /root -maxdepth 1 -type f -name "soak_r*_${profile}.log" \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | tail -n +4 | cut -d' ' -f2-
    )
    for old_log in "${old_logs[@]:-}"; do
      [[ -n "$old_log" ]] && rm -f -- "$old_log"
    done
  done
  episode_count=$(find "$OTR_OBS_DIR" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)
  echo "  round $round done; episodes in pod obs: $episode_count" | tee -a "$RESULTS"
  [[ "$MAX_ROUNDS" -gt 0 && "$round" -ge "$MAX_ROUNDS" ]] && break
done
