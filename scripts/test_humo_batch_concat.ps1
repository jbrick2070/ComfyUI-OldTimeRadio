# scripts\test_humo_batch_concat.ps1
# End-to-end batch HuMo + concat test against the latest non-pending ledger.
# Three sequential phases:
#   1. PASS1 portraits  via render_flux_batch.py     (~2 min)
#   2. HuMo batch       via render_humo_batch.py     (~30-45 min depending on line count)
#   3. Episode concat   via render_episode_concat.py (~30s)
#
# Usage:
#   cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
#   .\scripts\test_humo_batch_concat.ps1
#   .\scripts\test_humo_batch_concat.ps1 -Ledger "path\to\ledger.json"
#   .\scripts\test_humo_batch_concat.ps1 -SkipPass1   # if portraits already exist
#
# Progress writes to scripts\test_humo_batch_concat.log so Claude can tail it.

[CmdletBinding()]
param(
    [string]$Ledger,
    [string]$EpisodeMp4,
    [string]$OutDir,
    [switch]$SkipPass1,
    [switch]$UseExistingStillsAsPortraits,  # smoke-test: copy N env stills as fake portraits (auto-pick newest)
    [string]$PortraitMap,                   # explicit map: "ANNOUNCER=17,ANTON SHAW=10,KAEL PIERCE=14"
    [string]$Scope = "all",                 # all / first-per-scene / cold-open / custom:l001,l005
                                            # NOTE: first-per-scene needs lines[*].shot_id populated; ledger
                                            # currently nulls those out, so default to 'all' for safety
    [int]$MaxClips = 4,                     # smoke default: 4 clips. set 0 for unlimited (full episode)
    [double]$AutoSlice = 7.0,               # auto-slice master WAV when lines lack timing.
                                            # Default bumped from 3.88 to 7.0 per BUG-LOCAL-076 /
                                            # FULL-workflow spec: 7.0s clips give twice the line
                                            # density per clip and match HuMo's 7.08s ceiling.
                                            # Must match render_humo_batch.py --clip-length to
                                            # avoid stride/window overlap or gaps.
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot   = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$Python     = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$AudioDir   = "C:\Users\jeffr\Documents\ComfyUI\output\otr\audio"
$LegacyDir  = "C:\Users\jeffr\Documents\ComfyUI\output\old_time_radio"
$LogPath    = Join-Path $RepoRoot "scripts\test_humo_batch_concat.log"

function Log {
    param([string]$msg, [string]$level = "INFO")
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$ts] [$level] $msg"
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding utf8
}

# Reset log
Set-Content -Path $LogPath -Value "" -Encoding utf8
Log "test_humo_batch_concat starting"

# ---------- 1. Pick ledger ----------

if (-not $Ledger) {
    $candidates = @()
    foreach ($d in @($AudioDir, $LegacyDir)) {
        if (Test-Path $d) {
            $candidates += Get-ChildItem $d -Filter "*_ledger.json" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notlike "pending_*" }
        }
    }
    $latest = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        Log "no non-pending ledger found in $AudioDir or $LegacyDir" "ERROR"
        exit 1
    }
    $Ledger = $latest.FullName
}
Log "ledger: $Ledger"

# Read episode_id and audio path from ledger
$led = Get-Content $Ledger -Raw | ConvertFrom-Json
$episodeId = $led.episode_id
Log "episode_id: $episodeId"
Log "  cast=$($led.cast.Count) scenes=$($led.scenes.Count) shots=$($led.shots.Count) lines=$($led.lines.Count) dur=$($led.total_episode_dur_s)s"

# ---------- 2. Resolve episode audio ----------

if (-not $EpisodeMp4) {
    $candidates = @(
        (Join-Path $AudioDir   "$episodeId.mp4"),
        (Join-Path $LegacyDir  "$episodeId.mp4")
    )
    $EpisodeMp4 = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $EpisodeMp4) {
        Log "could not auto-resolve episode mp4. tried: $($candidates -join '; ')" "ERROR"
        exit 1
    }
}
Log "episode mp4: $EpisodeMp4"

# ---------- 3. Output dirs ----------

# render_episode_concat.py auto-resolves clips at <comfy-out>/otr/videos/<episode_id>/
# so we MUST land HuMo clips there, not at a custom test path.
$ComfyOut    = "C:\Users\jeffr\Documents\ComfyUI\output"
$ClipsDir    = Join-Path $ComfyOut "otr\videos\$episodeId"
$PortraitsDir = Join-Path $ComfyOut "otr\portraits\$episodeId"
$ConcatDir    = Join-Path $ComfyOut "otr\episodes\$episodeId"
foreach ($d in @($ClipsDir, $PortraitsDir, $ConcatDir)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Log "portraits_dir: $PortraitsDir"
Log "clips_dir:     $ClipsDir"
Log "concat_dir:    $ConcatDir"

if ($DryRun) {
    Log "DRY RUN -- would now run Pass1, HuMo batch, concat. exiting."
    exit 0
}

# ---------- 4. PASS1 portraits ----------

if ($PortraitMap -or $UseExistingStillsAsPortraits) {
    $stillsSrc = "C:\Users\jeffr\Documents\ComfyUI\output\otr\stills"
    if (-not (Test-Path $stillsSrc)) {
        Log "stills folder not found: $stillsSrc" "ERROR"
        exit 1
    }

    # Parse portrait map (if provided) into a hashtable: cast_name -> still_index
    $mapTable = @{}
    if ($PortraitMap) {
        foreach ($pair in $PortraitMap -split ',') {
            $parts = $pair -split '=', 2
            if ($parts.Count -eq 2) {
                $key = $parts[0].Trim().ToUpper()
                $val = $parts[1].Trim()
                $mapTable[$key] = $val
            }
        }
        Log "PHASE 1: -PortraitMap parsed: $(($mapTable.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ', ')"
    } else {
        Log "PHASE 1: -UseExistingStillsAsPortraits set; auto-picking newest stills"
    }

    # Build the picks: one per cast member, in cast order
    $picks = @()
    $newest = Get-ChildItem $stillsSrc -Filter "full_env_*.png" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    $autoIdx = 0
    foreach ($c in $led.cast) {
        $castName = ($c.name | Out-String).Trim().ToUpper()
        $pickedFile = $null
        if ($mapTable.ContainsKey($castName)) {
            $idx = $mapTable[$castName]
            # Pad to width 5 because filenames are full_env_NNNNN_.png
            $candidate = Join-Path $stillsSrc ("full_env_{0:D5}_.png" -f [int]$idx)
            if (Test-Path $candidate) {
                $pickedFile = Get-Item $candidate
                Log "  $castName -> $($pickedFile.Name) (mapped)"
            } else {
                Log "  $castName -> mapped index $idx not found at $candidate; falling back" "WARN"
            }
        }
        if (-not $pickedFile) {
            if ($autoIdx -lt $newest.Count) {
                $pickedFile = $newest[$autoIdx]
                $autoIdx++
                Log "  $castName -> $($pickedFile.Name) (auto-picked newest)"
            } else {
                Log "  $castName -> NO PNG available; portraits dir will be incomplete" "ERROR"
                continue
            }
        }
        $picks += [PSCustomObject]@{ CastName = $castName; SourcePng = $pickedFile }
    }

    # Copy with cast-index-ordered names so render_humo_batch's modulo lookup
    # lands the right portrait on the right cast row.
    $i = 1
    foreach ($p in $picks) {
        $dst = Join-Path $PortraitsDir ("otr_humo_pass1_portrait_{0:D3}.png" -f $i)
        Copy-Item $p.SourcePng.FullName $dst -Force
        $i++
    }
    Log "PHASE 1: $($picks.Count) portraits placed (visually wrong characters per scene but face-real; HuMo lip-sync will produce honest-looking video)"
} elseif ($SkipPass1) {
    Log "PHASE 1: Pass1 portraits SKIPPED (--SkipPass1)"
} else {
    $existing = Get-ChildItem $PortraitsDir -Filter "otr_humo_pass1_portrait_*.png" -ErrorAction SilentlyContinue
    if ($existing.Count -ge $led.cast.Count) {
        Log "PHASE 1: $($existing.Count) portraits already in $PortraitsDir, skipping Pass1"
    } else {
        Log "PHASE 1: rendering Pass1 portraits to $PortraitsDir"
        $t0 = Get-Date
        & $Python (Join-Path $RepoRoot "scripts\render_flux_batch.py") `
            --ledger $Ledger `
            --scope portraits-only `
            --out-dir $PortraitsDir `
            --comfyui-url "http://127.0.0.1:8000" 2>&1 | Tee-Object -FilePath $LogPath -Append
        if ($LASTEXITCODE -ne 0) {
            Log "PHASE 1 FAILED: render_flux_batch exit=$LASTEXITCODE" "ERROR"
            exit 1
        }
        Log "PHASE 1 complete in $(((Get-Date) - $t0).TotalSeconds.ToString('0'))s"
    }
}

# ---------- 5. HuMo batch ----------

Log "PHASE 2: HuMo batch -> $ClipsDir (scope=$Scope auto-slice=$AutoSlice max-clips=$MaxClips)"
$t0 = Get-Date
$humoArgs = @(
    "--ledger", $Ledger,
    "--portraits-dir", $PortraitsDir,
    "--out-dir", $ClipsDir,
    "--comfyui-url", "http://127.0.0.1:8000",
    "--scope", $Scope,
    "--auto-slice", $AutoSlice
)
if ($MaxClips -gt 0) { $humoArgs += @("--max-clips", $MaxClips) }
& $Python (Join-Path $RepoRoot "scripts\render_humo_batch.py") @humoArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    Log "PHASE 2 FAILED: render_humo_batch exit=$LASTEXITCODE" "ERROR"
    exit 1
}
Log "PHASE 2 complete in $(((Get-Date) - $t0).TotalSeconds.ToString('0'))s"

# ---------- 6. Concat ----------

Log "PHASE 3: episode concat -> $ConcatDir"
$t0 = Get-Date
& $Python (Join-Path $RepoRoot "scripts\render_episode_concat.py") `
    --ledger $Ledger `
    --comfy-output-dir $ComfyOut `
    --audio-mp4 $EpisodeMp4 `
    --out-dir $ConcatDir `
    --trim `
    --embed-subs 2>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    Log "PHASE 3 FAILED: render_episode_concat exit=$LASTEXITCODE" "ERROR"
    exit 1
}
Log "PHASE 3 complete in $(((Get-Date) - $t0).TotalSeconds.ToString('0'))s"

# ---------- 7. Summary ----------

Log "ALL PHASES COMPLETE"
Log "  portraits: $PortraitsDir  ($((Get-ChildItem $PortraitsDir -Filter '*.png' -ErrorAction SilentlyContinue).Count) files)"
Log "  clips:     $ClipsDir       ($((Get-ChildItem $ClipsDir -Filter '*.mp4' -ErrorAction SilentlyContinue).Count) files)"
$finalCandidates = Get-ChildItem $ConcatDir -Filter "*.mp4" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($finalCandidates) {
    $f = $finalCandidates[0]
    Log "  final:     $($f.FullName)  ($([Math]::Round($f.Length / 1MB, 1)) MB)"
}
Log "tail this log at: $LogPath"
