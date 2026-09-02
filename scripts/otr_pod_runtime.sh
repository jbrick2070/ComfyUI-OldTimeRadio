#!/usr/bin/env bash
# Shared RunPod runtime owner. Source this file from pod sweep/soak scripts.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source scripts/otr_pod_runtime.sh from an OTR pod runner." >&2
  exit 2
fi

OTR_ACTIVE_LAUNCH_ENV_KEYS=()

otr_runtime_error() {
  echo "ERROR: $*" >&2
  return 1
}

otr_load_runtime() {
  local runtime_file="${OTR_RUNTIME_ENV:-/workspace/otr-config/otr-runtime.env}"
  local key
  local -a keys=(
    OTR_COMFY_ROOT OTR_REPO_ROOT COMFY_PY OTR_COMFYUI_MODELS_ROOT HF_HOME
    OTR_INDEXTTS2_ROOT OTR_INDEXTTS2_DIR OTR_INDEXTTS2_WORKER
    OTR_VOICE_REFERENCE_BANK OTR_PROVISION_PROFILE OTR_HEADLESS_PORT
  )
  local -A was_set=() prior=()

  [[ -s "$runtime_file" ]] \
    || { otr_runtime_error "runtime receipt is missing: $runtime_file; rerun scripts/otr_pod_provision.sh"; return 1; }
  for key in "${keys[@]}"; do
    if [[ -v "$key" ]]; then
      was_set["$key"]=1
      prior["$key"]="${!key}"
    fi
  done
  # The receipt is written by the provisioner, mode 0600, using Bash %q.
  # shellcheck disable=SC1090
  source "$runtime_file" \
    || { otr_runtime_error "could not source runtime receipt: $runtime_file"; return 1; }
  for key in "${keys[@]}"; do
    if [[ -n "${was_set[$key]:-}" ]]; then
      printf -v "$key" '%s' "${prior[$key]}"
      export "$key"
    fi
  done

  [[ -f "$OTR_COMFY_ROOT/folder_paths.py" ]] \
    || { otr_runtime_error "OTR_COMFY_ROOT is not a ComfyUI tree: $OTR_COMFY_ROOT"; return 1; }
  [[ -d "$OTR_REPO_ROOT/config/profiles" ]] \
    || { otr_runtime_error "OTR_REPO_ROOT is not an OTR checkout: $OTR_REPO_ROOT"; return 1; }
  [[ -x "$COMFY_PY" ]] \
    || { otr_runtime_error "COMFY_PY is not executable: $COMFY_PY"; return 1; }
  [[ -n "${OTR_COMFYUI_MODELS_ROOT:-}" && -n "${HF_HOME:-}" ]] \
    || { otr_runtime_error "model/cache roots are absent from the runtime receipt"; return 1; }
  [[ "${OTR_HEADLESS_PORT:-}" =~ ^[0-9]+$ ]] \
    || { otr_runtime_error "OTR_HEADLESS_PORT is not numeric: ${OTR_HEADLESS_PORT:-<unset>}"; return 1; }

  export COMFYUI_URL="http://127.0.0.1:$OTR_HEADLESS_PORT"
  export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
  if [[ -z "${HF_TOKEN:-}" && -s /root/.hf_token ]]; then
    HF_TOKEN=$(tr -d ' \t\r\n' < /root/.hf_token)
    export HF_TOKEN
  fi

  OTR_OUTPUT_ROOT="${OTR_OUTPUT_ROOT:-$OTR_COMFY_ROOT/output}"
  OTR_SERVER_LOG="${OTR_SERVER_LOG:-/workspace/otr-config/comfyui.log}"
  OTR_SERVER_FINGERPRINT="${OTR_SERVER_FINGERPRINT:-/workspace/otr-config/otr-server.fingerprint}"
  export OTR_OUTPUT_ROOT OTR_SERVER_LOG OTR_SERVER_FINGERPRINT
  export OTR_OUTPUT_DIR="$OTR_OUTPUT_ROOT"
  export OTR_OBS_DIR="$OTR_OUTPUT_ROOT/otr/obs"
  export OTR_TMP="$OTR_OUTPUT_ROOT/otr/episodes/_shared/tmp"
  export TMP="$OTR_TMP" TEMP="$OTR_TMP" OTR_GPU_LEASE_DIR="$OTR_TMP"
  mkdir -p "$OTR_OBS_DIR" "$OTR_TMP" "$(dirname "$OTR_SERVER_LOG")" \
    "$(dirname "$OTR_SERVER_FINGERPRINT")" \
    || { otr_runtime_error "could not create persistent output/runtime directories"; return 1; }
}

# Read ss -lptnH or netstat -lptn text on stdin and emit exact listener PIDs.
otr_listener_pids_from_stream() {
  local port="$1"
  awk -v port="$port" '
    $1 == "LISTEN" && $4 ~ (":" port "$") {
      line = $0
      while (match(line, /pid=[0-9]+/)) {
        token = substr(line, RSTART + 4, RLENGTH - 4)
        print token
        line = substr(line, RSTART + RLENGTH)
      }
    }
    $6 == "LISTEN" && $4 ~ (":" port "$") && $7 ~ /^[0-9]+\// {
      split($7, owner, "/")
      print owner[1]
    }
  ' | sort -nu
}

otr_port_listener_from_stream() {
  local port="$1"
  awk -v port="$port" '
    ($1 == "LISTEN" && $4 ~ (":" port "$")) ||
    ($6 == "LISTEN" && $4 ~ (":" port "$")) { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

otr_listener_table() {
  if command -v ss >/dev/null 2>&1; then
    ss -lptnH
  elif command -v netstat >/dev/null 2>&1; then
    netstat -lptn
  else
    otr_runtime_error "neither ss nor netstat is installed; listener ownership cannot be proved"
    return 1
  fi
}

otr_stop_template_listeners() {
  local table_file pid_file port pid attempt
  local -a ports=(8188)
  local -A targets=()
  [[ "$OTR_HEADLESS_PORT" == "8188" ]] || ports+=("$OTR_HEADLESS_PORT")
  table_file=$(mktemp) || return 1
  pid_file=$(mktemp) || { rm -f "$table_file"; return 1; }

  if ! otr_listener_table > "$table_file"; then
    rm -f "$table_file" "$pid_file"
    return 1
  fi
  for port in "${ports[@]}"; do
    : > "$pid_file"
    if ! otr_listener_pids_from_stream "$port" < "$table_file" > "$pid_file"; then
      rm -f "$table_file" "$pid_file"
      return 1
    fi
    if otr_port_listener_from_stream "$port" < "$table_file" && [[ ! -s "$pid_file" ]]; then
      rm -f "$table_file" "$pid_file"
      otr_runtime_error "port $port is listening but its PID is unavailable; refusing a broad process kill"
      return 1
    fi
    while IFS= read -r pid; do
      [[ "$pid" =~ ^[0-9]+$ ]] && targets["$pid"]=1
    done < "$pid_file"
  done

  for pid in "${!targets[@]}"; do
    echo "  stopping listener PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
  done
  for attempt in $(seq 1 20); do
    local alive=0
    for pid in "${!targets[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    [[ "$alive" -eq 0 ]] && break
    sleep 0.5
  done
  for pid in "${!targets[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  escalating listener PID $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done

  sleep 1
  if ! otr_listener_table > "$table_file"; then
    rm -f "$table_file" "$pid_file"
    return 1
  fi
  for port in "${ports[@]}"; do
    if otr_port_listener_from_stream "$port" < "$table_file"; then
      rm -f "$table_file" "$pid_file"
      otr_runtime_error "port $port is still listening after targeted shutdown"
      return 1
    fi
  done
  rm -f "$table_file" "$pid_file"
}

otr_ready() {
  "$COMFY_PY" - <<'PY' >/dev/null 2>&1
import json
import os
import urllib.request

base = os.environ["COMFYUI_URL"].rstrip("/")
with urllib.request.urlopen(base + "/object_info", timeout=90) as response:
    objects = json.loads(response.read().decode("utf-8"))
with urllib.request.urlopen(base + "/queue", timeout=30) as response:
    queue = json.loads(response.read().decode("utf-8"))
count = sum(1 for key in objects if key.startswith("OTR_"))
busy = len(queue.get("queue_running", [])) + len(queue.get("queue_pending", []))
if count <= 0 or busy:
    raise SystemExit(1)
PY
}

otr_wait_ready() {
  local tries="${1:-90}" attempt
  for attempt in $(seq 1 "$tries"); do
    otr_ready && return 0
    sleep 10
  done
  return 1
}

otr_profile_output() {
  local profile="$1" mode="$2" output="$3"
  local helper="$OTR_REPO_ROOT/scripts/otr_profile_launch_args.py"
  [[ -f "$helper" ]] \
    || { otr_runtime_error "profile launch helper is missing: $helper"; return 1; }
  if ! "$COMFY_PY" "$helper" "$profile" --mode "$mode" > "$output"; then
    otr_runtime_error "could not resolve $mode for profile $profile"
    return 1
  fi
}

otr_profile_fingerprint() {
  local profile="$1" output="$2"
  otr_profile_output "$profile" fingerprint "$output" || return 1
  grep -Eq '^[0-9a-f]{64}$' "$output" \
    || { otr_runtime_error "invalid launch fingerprint for $profile"; return 1; }
}

otr_apply_profile_env() {
  local profile="$1" output="$2" key value line
  for key in "${OTR_ACTIVE_LAUNCH_ENV_KEYS[@]:-}"; do
    [[ -n "$key" ]] && unset "$key"
  done
  OTR_ACTIVE_LAUNCH_ENV_KEYS=()
  otr_profile_output "$profile" env "$output" || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "$line" == *=* ]] \
      || { otr_runtime_error "malformed launch environment row for $profile"; return 1; }
    key="${line%%=*}"
    value="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
      || { otr_runtime_error "invalid launch environment key for $profile: $key"; return 1; }
    export "$key=$value"
    OTR_ACTIVE_LAUNCH_ENV_KEYS+=("$key")
  done < "$output"
}

otr_boot_profile() {
  local profile="$1" env_file args_file fingerprint_file fingerprint_tmp
  local server_pid
  local -a launch_args=()
  env_file=$(mktemp) || return 1
  args_file=$(mktemp) || { rm -f "$env_file"; return 1; }
  fingerprint_file=$(mktemp) || { rm -f "$env_file" "$args_file"; return 1; }

  otr_apply_profile_env "$profile" "$env_file" \
    || { rm -f "$env_file" "$args_file" "$fingerprint_file"; return 1; }
  otr_profile_output "$profile" args "$args_file" \
    || { rm -f "$env_file" "$args_file" "$fingerprint_file"; return 1; }
  otr_profile_fingerprint "$profile" "$fingerprint_file" \
    || { rm -f "$env_file" "$args_file" "$fingerprint_file"; return 1; }
  mapfile -t launch_args < "$args_file"
  otr_stop_template_listeners \
    || { rm -f "$env_file" "$args_file" "$fingerprint_file"; return 1; }

  export OTR_ACTIVE_PROFILE="$profile"
  echo "  booting $profile on $COMFYUI_URL"
  (
    cd "$OTR_COMFY_ROOT" || exit 1
    nohup "$COMFY_PY" main.py \
      --listen 0.0.0.0 --port "$OTR_HEADLESS_PORT" \
      --output-directory "$OTR_OUTPUT_ROOT" --enable-cors-header \
      "${launch_args[@]}" > "$OTR_SERVER_LOG" 2>&1 &
    echo $!
  ) > "$args_file"
  server_pid=$(tail -n 1 "$args_file")
  if ! [[ "$server_pid" =~ ^[0-9]+$ ]] || ! otr_wait_ready; then
    echo "  server boot failed for $profile (PID ${server_pid:-unknown})" >&2
    tail -n 40 "$OTR_SERVER_LOG" >&2 2>/dev/null || true
    otr_stop_template_listeners || true
    rm -f "$env_file" "$args_file" "$fingerprint_file"
    return 1
  fi
  fingerprint_tmp="${OTR_SERVER_FINGERPRINT}.tmp.$$"
  if ! cp "$fingerprint_file" "$fingerprint_tmp" || ! mv -f "$fingerprint_tmp" "$OTR_SERVER_FINGERPRINT"; then
    rm -f "$fingerprint_tmp" "$env_file" "$args_file" "$fingerprint_file"
    otr_runtime_error "could not publish server fingerprint"
    return 1
  fi
  echo "  server ready for $profile (PID $server_pid)"
  rm -f "$env_file" "$args_file" "$fingerprint_file"
}

otr_ensure_profile_server() {
  local profile="$1" wanted current
  wanted=$(mktemp) || return 1
  otr_profile_fingerprint "$profile" "$wanted" || { rm -f "$wanted"; return 1; }
  current=$(cat "$OTR_SERVER_FINGERPRINT" 2>/dev/null || true)
  if otr_ready && [[ "$current" == "$(cat "$wanted")" ]]; then
    rm -f "$wanted"
    return 0
  fi
  rm -f "$wanted"
  otr_boot_profile "$profile"
}

otr_profile_roster() {
  local explicit="${OTR_POD_PROFILES:-}" profile path contract_file fingerprint_file
  local discovered=0
  local raw_file sorted_file
  local -a candidates=()
  raw_file=$(mktemp) || return 1
  sorted_file=$(mktemp) || { rm -f "$raw_file"; return 1; }
  contract_file=$(mktemp) || { rm -f "$raw_file" "$sorted_file"; return 1; }
  fingerprint_file=$(mktemp) \
    || { rm -f "$raw_file" "$sorted_file" "$contract_file"; return 1; }

  if [[ -n "$explicit" ]]; then
    read -r -a candidates <<< "$explicit"
  else
    discovered=1
    for path in "$OTR_REPO_ROOT"/config/profiles/otr_w45_*.json; do
      [[ -f "$path" ]] || continue
      candidates+=("$(basename "$path" .json)")
    done
  fi
  [[ "${#candidates[@]}" -gt 0 ]] \
    || { rm -f "$raw_file" "$sorted_file" "$contract_file" "$fingerprint_file"; otr_runtime_error "no pod profiles were selected"; return 1; }

  : > "$raw_file"
  for profile in "${candidates[@]}"; do
    otr_profile_output "$profile" contract "$contract_file" \
      || { rm -f "$raw_file" "$sorted_file" "$contract_file" "$fingerprint_file"; return 1; }
    if [[ "$(cat "$contract_file")" == "h3" ]]; then
      if [[ "$discovered" -eq 1 ]]; then
        echo "  excluding operator-local H3 profile: $profile" >&2
        continue
      fi
      rm -f "$raw_file" "$sorted_file" "$contract_file" "$fingerprint_file"
      otr_runtime_error "H3 profile $profile is operator-local and cannot run in the cloud roster"
      return 1
    fi
    otr_profile_fingerprint "$profile" "$fingerprint_file" \
      || { rm -f "$raw_file" "$sorted_file" "$contract_file" "$fingerprint_file"; return 1; }
    printf '%s\t%s\n' "$(cat "$fingerprint_file")" "$profile" >> "$raw_file"
  done
  if ! sort -k1,1 -k2,2 "$raw_file" > "$sorted_file"; then
    rm -f "$raw_file" "$sorted_file" "$contract_file" "$fingerprint_file"
    return 1
  fi
  awk -F '\t' '{print $2}' "$sorted_file"
  local rc=$?
  rm -f "$raw_file" "$sorted_file" "$contract_file" "$fingerprint_file"
  return "$rc"
}

otr_roster_preflight() {
  local roster_file="$1" profile required_file needs_index=0
  required_file=$(mktemp) || return 1
  while IFS= read -r profile; do
    [[ -z "$profile" ]] && continue
    otr_profile_output "$profile" requires-indextts2 "$required_file" \
      || { rm -f "$required_file"; return 1; }
    [[ "$(cat "$required_file")" == "1" ]] && needs_index=1
  done < "$roster_file"
  rm -f "$required_file"
  if [[ "$needs_index" -eq 1 && ! -s "$OTR_VOICE_REFERENCE_BANK" ]]; then
    otr_runtime_error "selected roster needs IndexTTS2, but the portable bank is missing: $OTR_VOICE_REFERENCE_BANK"
    return 1
  fi
}
