#!/usr/bin/env bash
# Run the canonical one-act qualification for every public w45 profile, then
# run three acts only for the profiles that passed. This script runs on the pod.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=otr_pod_runtime.sh
source "$SCRIPT_DIR/otr_pod_runtime.sh"
otr_load_runtime || exit 1
otr_acquire_campaign_lock "overnight sweep" || exit 1

RESULTS="${OTR_SWEEP_RESULTS:-$OTR_POD_LOG_DIR/overnight_results.txt}"
ROSTER_FILE=$(mktemp) || exit 1
if ! otr_profile_roster > "$ROSTER_FILE"; then
  rm -f "$ROSTER_FILE"
  exit 1
fi
otr_roster_preflight "$ROSTER_FILE" || { rm -f "$ROSTER_FILE"; exit 1; }
mapfile -t PROFILES < "$ROSTER_FILE"
rm -f "$ROSTER_FILE"

echo "=== OTR POD SWEEP START $(date -u '+%F %H:%M:%SZ') ===" | tee "$RESULTS"
echo "  profiles: ${#PROFILES[@]}" | tee -a "$RESULTS"

run_leg() {
  local profile="$1" acts="$2" log t0 t1 rc result reason
  log="$OTR_POD_LOG_DIR/leg_${profile}_${acts}act.log"
  if ! otr_ensure_profile_server "$profile"; then
    printf "  %-38s %s-act BOOT_FAILED\n" "$profile" "$acts" | tee -a "$RESULTS"
    return 1
  fi
  t0=$(date +%s)
  (
    cd "$OTR_REPO_ROOT" || exit 1
    "$COMFY_PY" scripts/otr_canonical_api_run.py \
      --profile "$profile" --act-count "$acts" \
      --source-bank original --visual-style sci_fi_radio --timeout 0
  ) > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  result=$(grep -oE 'RESULT (SUCCESS|FAIL)' "$log" | tail -1 || true)
  reason=$(grep -oE \
    'is not usable for role[^.]*|MISSING[_A-Z]*|ERROR node [0-9]+ \([A-Za-z_]+\)' \
    "$log" | head -1 | cut -c1-80 || true)
  printf "  %-38s %s-act %-14s %5ds rc=%s %s\n" \
    "$profile" "$acts" "${result:-NO_RESULT}" "$((t1-t0))" "$rc" "$reason" \
    | tee -a "$RESULTS"
  if ! otr_ready; then
    echo "  server stopped responding; recovering with the same profile" | tee -a "$RESULTS"
    otr_boot_profile "$profile" || true
  fi
  [[ "$rc" -eq 0 && "$result" == "RESULT SUCCESS" ]]
}

declare -A PASSED=()
echo "=== PASS 1: one act $(date -u '+%H:%M:%SZ') ===" | tee -a "$RESULTS"
for profile in "${PROFILES[@]}"; do
  if run_leg "$profile" 1; then
    PASSED["$profile"]=1
  fi
done

echo "=== PASS 2: three acts for passers $(date -u '+%H:%M:%SZ') ===" | tee -a "$RESULTS"
for profile in "${PROFILES[@]}"; do
  [[ -n "${PASSED[$profile]:-}" ]] && run_leg "$profile" 3 || true
done

episode_count=$(find "$OTR_OBS_DIR" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)
echo "=== OTR POD SWEEP DONE $(date -u '+%F %H:%M:%SZ') ===" | tee -a "$RESULTS"
echo "  episodes in pod obs: $episode_count" | tee -a "$RESULTS"
