#!/usr/bin/env bash
# One-act live coverage sweep across the per-engine video profiles.
#
# WHY THIS EXISTS. Ghost Signal shipped with a real live leg behind it; most of
# the other lanes have not had one recently, and two are named as unexercised in
# the carried queue item F. This runs ONE act per lane through the REAL canonical
# workflow -- the same wrapper the Ghost smoke used, which resets the box and
# boots the UTF-8 launcher per leg, so a lane that OOMs cannot poison the next.
#
# ORDER IS BY WHAT IS MOST OWED, so stopping the sweep early still buys the most:
#   1. wan_ti2v   -- named unexercised in queue item F
#   2. ltx_video  -- G4 records its only datapoint as a FAILED chained diagnostic
#   3. ltx_8gb    -- G4 records NO measurement of any kind on this box
#   4. ltx25      -- the newest lane
#   5. fastwan / humo / ltx_audio_in
#
# The bank ROLLS per leg and the visual style rolls with it (production leaves
# OTR_CAST_SEED / OTR_STYLE_SEED unset), so the sweep also samples content
# variety rather than testing one story shape seven times.
#
# A FAILING LEG DOES NOT STOP THE SWEEP. Each result is recorded and the next
# lane runs -- the point is a coverage map, not a first-failure abort.
set -u

REPO="C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio"
OUT="$REPO/otr_soak_receipts/lane_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.md"

PROFILES=(
  otr_g4_wan_ti2v
  otr_g4_ltx_video
  otr_g4_ltx_8gb
  otr_ltx25_high_video
  otr_g4_fastwan
  otr_g4_humo
  otr_g4_ltx_audio_in
)

{
  echo "# One-act video lane sweep"
  echo
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Writer: gemma-4-12b (the g4 profile family). Bank: rolled per leg."
  echo "Each leg loads the REAL workflows/otr_canonical.json via"
  echo "scripts/otr_headless_canonical.ps1, which resets and reboots per leg."
  echo
  echo "| # | profile | result | wall | published |"
  echo "|---:|---|---|---|---|"
} > "$SUMMARY"

i=0
for prof in "${PROFILES[@]}"; do
  i=$((i + 1))
  leg="$OUT/${i}_${prof}.log"
  srv="$OUT/${i}_${prof}_server.log"
  echo "[sweep] ($i/${#PROFILES[@]}) $prof -> $leg"
  start=$(date +%s)

  powershell -NoProfile -ExecutionPolicy Bypass \
    -File "$REPO/scripts/otr_headless_canonical.ps1" \
    -Profile "$prof" -Acts 1 -Timeout 5400 \
    -ServerLog "$srv" \
    -Set "OTR_LedgerScriptWriter.source_bank=roll (any eligible bank)" \
    > "$leg" 2>&1
  rc=$?

  end=$(date +%s)
  wall=$(( end - start ))
  mins=$(( wall / 60 ))
  secs=$(( wall % 60 ))

  if grep -q "RESULT SUCCESS" "$leg" 2>/dev/null; then
    result="PASS"
  else
    result="FAIL(rc=$rc)"
  fi

  pub=$(grep -oE "obs_publish OK -> .*" "$srv" 2>/dev/null | tail -1 \
        | sed 's/.*obs\\//' | cut -c1-58)
  [ -z "$pub" ] && pub="(none)"

  printf "| %d | \`%s\` | %s | %dm%02ds | %s |\n" \
    "$i" "$prof" "$result" "$mins" "$secs" "$pub" >> "$SUMMARY"
  echo "[sweep] ($i/${#PROFILES[@]}) $prof = $result in ${mins}m${secs}s"
done

{
  echo
  echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$SUMMARY"

echo "[sweep] DONE -- summary at $SUMMARY"
cat "$SUMMARY"
