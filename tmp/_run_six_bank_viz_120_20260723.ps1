# Six-bank 120-word viz campaign against the real canonical workflow.
# The ComfyUI server is intentionally reused between legs; each leg gets its
# own canonical API prompt, prompt dump, stdout, stderr, and summary receipt.
param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$RunDir
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$Python = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$BaseUrl = "http://127.0.0.1:$Port"
$Banks = @("media_archive", "original", "public_domain", "shakespeare", "scifi_news", "scifi_news_pro")
$Model = "google/gemma-4-12b-it"

New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:COMFYUI_URL = $BaseUrl
$env:OTR_HEADLESS_PORT = "$Port"
. (Join-Path $Repo "scripts\otr_campaign_receipt.ps1")

function Write-RunLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath (Join-Path $RunDir "campaign.log") -Value $line -Encoding utf8
}

function Get-Queue {
    Invoke-RestMethod -Uri "$BaseUrl/queue" -TimeoutSec 10
}

function Assert-Ready {
    $null = Invoke-RestMethod -Uri "$BaseUrl/system_stats" -TimeoutSec 10
    $null = Invoke-RestMethod -Uri "$BaseUrl/object_info/OTR_WorkflowValidator" -TimeoutSec 10
    $q = Get-Queue
    if (@($q.queue_running).Count -ne 0 -or @($q.queue_pending).Count -ne 0) {
        throw "queue is not empty before leg"
    }
}

Set-Location -LiteralPath $Repo
Write-RunLog "campaign start: six banks, 120 words, Gemma 4, viz canonical media"
Write-RunLog "server=$BaseUrl model=$Model workflow=$Repo\workflows\otr_canonical.json"

$receipts = @()
for ($i = 0; $i -lt $Banks.Count; $i++) {
    $bank = $Banks[$i]
    $tag = "leg{0:D2}_{1}_120w" -f ($i + 1), $bank
    $stdout = Join-Path $RunDir "$tag.stdout.log"
    $stderr = Join-Path $RunDir "$tag.stderr.log"
    $prompt = Join-Path $RunDir "$tag.prompt.json"
    Write-RunLog "LEG $($i + 1)/$($Banks.Count) START bank=$bank words=120"
    Assert-Ready

    $runnerArgs = @(
        "-u", "scripts\otr_canonical_api_run.py",
        "--profile", "none",
        "--words", "120",
        "--source-bank", $bank,
        "--creative-model", $Model,
        "--technical-model", $Model,
        "--dump-prompt", $prompt,
        "--timeout", "5400",
        "--poll-s", "10"
    )
    $started = Get-Date
    $proc = Start-Process -FilePath $Python -ArgumentList $runnerArgs -WorkingDirectory $Repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 15
        $proc.Refresh()
        $null = Invoke-RestMethod -Uri "$BaseUrl/system_stats" -TimeoutSec 10
    }
    $proc.Refresh()
    $exit = [int]$proc.ExitCode
    $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    $q = Get-Queue
    $queueEmpty = (@($q.queue_running).Count -eq 0 -and @($q.queue_pending).Count -eq 0)
    $verdict = Get-OtrReceiptVerdict -StdoutPath $stdout -ExitCode $exit -QueueEmpty $queueEmpty
    $status = $verdict.status
    $receipt = [pscustomobject][ordered]@{
        leg = $i + 1; bank = $bank; words = 120; model = $Model; media = "viz";
        status = $status; exit_code = $exit; elapsed_seconds = $elapsed;
        queue_empty = $queueEmpty; terminal_result = $verdict.terminal_result;
        terminal_line = $verdict.terminal_line; receipt_reason = $verdict.reason;
        stdout = $stdout; stderr = $stderr; prompt = $prompt
    }
    $receipts += $receipt
    $receipts | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $RunDir "receipts.json") -Encoding utf8
    Write-RunLog "LEG $($i + 1)/$($Banks.Count) $status bank=$bank exit=$exit terminal=$($verdict.terminal_result) reason=$($verdict.reason) elapsed=${elapsed}s queue_empty=$queueEmpty"
}

$passed = @($receipts | Where-Object status -eq "PASS").Count
Write-RunLog "campaign complete: $passed/$($Banks.Count) PASS"
exit $(if ($passed -eq $Banks.Count) { 0 } else { 1 })
