# Overnight native-Shakespeare legs.
#
# Boots ONE server and reuses it for every leg (the harness does not tear down,
# and a warm server is the whole point of a sequence). Each leg pins the
# shakespeare bank and one of the vendored languages, so the episode performs a
# real translator's words rather than a machine translation.
#
# Quoting lives inside this file on purpose: every variable here is literal and
# safe, which is what the launcher-script rule exists for.

$ErrorActionPreference = "Continue"
$repo   = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$py     = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$runner = Join-Path $repo "scripts\otr_canonical_api_run.py"
$launch = Join-Path $repo "scripts\_otr_soak_server_launch.cmd"
$logdir = Join-Path $repo "otr\legs"
$url    = "http://127.0.0.1:8000"

New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$srvlog = Join-Path $logdir "shx_overnight_$stamp.server.log"
$runlog = Join-Path $logdir "shx_overnight_$stamp.runner.log"

function Note($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $runlog -Value $line -Encoding utf8
}

Note "overnight shakespeare legs starting"

# --- boot the server if it is not already up -------------------------------
$listening = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    Note "server already listening on 8000, reusing it"
} else {
    Note "booting server -> $srvlog"
    Start-Process -FilePath $launch -ArgumentList "`"$srvlog`"" -WindowStyle Hidden
    $up = $false
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 5
        $c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
        if ($c) { $up = $true; break }
    }
    if (-not $up) { Note "SERVER DID NOT COME UP - read $srvlog"; exit 2 }
    Note "server up"
}

# --- the legs --------------------------------------------------------------
# One per vendored language, cycled. `episode_language` is what sends the
# writer to the vendored corpus; `--source-bank shakespeare` pins the bank so
# nothing falls back to media_archive.
# NO --title: the run label already becomes the on-screen title card, and
# naming a canonical leg is what put harness scratch on screen before.
# PIN THE SCENE, not just the language. Left to pick for itself the bank chose
# Romeo and Juliet in French -- a scene with no French vendored text -- so the
# writer MODEL-TRANSLATED the Folger English instead of performing Hugo. That is
# a valid episode and it is not what "native Shakespeare" means here.
#
# These four pairs are exactly the ones with a vendored translation on disk, so
# every leg performs a real translator's words, sha256-verified at load.
$pairs = @(
    @{ lang = "French";  ref = "folger-hamlet:act1-scene1-platform-watch";        who = "hugo_hamlet" },
    @{ lang = "French";  ref = "folger-king-lear:act1-scene1-love-test";          who = "hugo_lear" },
    @{ lang = "Italian"; ref = "folger-macbeth:act1-scene3-witches";              who = "rusconi_macbeth" },
    @{ lang = "Spanish"; ref = "folger-as-you-like-it:act3-scene2-rosalind-orlando"; who = "marquez_ayli" }
)

$leg = 0
$deadline = (Get-Date).AddHours(9)
while ((Get-Date) -lt $deadline) {
    foreach ($p in $pairs) {
        if ((Get-Date) -ge $deadline) { break }
        $leg++
        $lang  = $p.lang
        $label = "shx_{0}_{1:d2}" -f $p.who, $leg
        Note "LEG $leg  language=$lang  ref=$($p.ref)  label=$label"
        & $py $runner `
            --comfyui-url $url `
            --source-bank "shakespeare" `
            --run-label $label `
            --act-count 1 `
            --num-characters 3 `
            --set "OTR_LedgerScriptWriter.episode_language=$lang" `
            --set "OTR_LedgerScriptWriter.source_ref=$($p.ref)" `
            --timeout 0 2>&1 | Tee-Object -FilePath $runlog -Append | Out-Null
        $rc = $LASTEXITCODE
        Note "LEG $leg  rc=$rc"
        # obs is under ComfyUI's OUTPUT root, not the repo. A repo-relative
        # path here reports zero published episodes on a run that published
        # fine, which is the shape of a false failure this project has hit
        # before.
        $obs = "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"
        if (Test-Path $obs) {
            $n = (Get-ChildItem $obs -Filter *.mp4 -ErrorAction SilentlyContinue |
                  Measure-Object).Count
            Note "LEG $leg  obs mp4 count now: $n"
        }
    }
}
Note "overnight window closed after $leg legs"
