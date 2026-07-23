param(
    [switch]$Probe,
    [string]$StdoutPath = "",
    [int]$ExitCode = 0,
    [string]$QueueEmpty = "0"
)

function Get-OtrTerminalResult([string]$Path) {
    $matches = @(Select-String -LiteralPath $Path -Pattern '^\[canonical-api\] RESULT (?<status>[A-Z_]+)(?:\s|$)' -ErrorAction SilentlyContinue)
    if ($matches.Count -eq 0) {
        return [pscustomobject]@{ status = ""; line = "" }
    }
    $last = $matches[-1]
    $match = [regex]::Match([string]$last.Line, '^\[canonical-api\] RESULT (?<status>[A-Z_]+)(?:\s|$)')
    return [pscustomobject]@{
        status = if ($match.Success) { $match.Groups["status"].Value } else { "" }
        line = [string]$last.Line
    }
}

function Get-OtrReceiptVerdict {
    param(
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][int]$ExitCode,
        [Parameter(Mandatory = $true)][bool]$QueueEmpty
    )

    $terminal = Get-OtrTerminalResult -Path $StdoutPath
    $reasons = @()
    if ($ExitCode -ne 0) { $reasons += "child_exit_code=$ExitCode" }
    if (-not $QueueEmpty) { $reasons += "queue_not_empty" }
    if ($terminal.status -ne "SUCCESS") {
        $seen = if ($terminal.status) { $terminal.status } else { "MISSING" }
        $reasons += "terminal_result=$seen"
    }
    return [pscustomobject][ordered]@{
        status = if ($reasons.Count -eq 0) { "PASS" } else { "FAIL" }
        terminal_result = $terminal.status
        terminal_line = $terminal.line
        reason = ($reasons -join ";")
    }
}

if ($Probe) {
    if (-not $StdoutPath) { throw "-StdoutPath is required with -Probe" }
    $probeQueueEmpty = $QueueEmpty -in @("1", "true", "True")
    $verdict = Get-OtrReceiptVerdict -StdoutPath $StdoutPath -ExitCode $ExitCode -QueueEmpty $probeQueueEmpty
    $verdict | ConvertTo-Json -Compress
    exit $(if ($verdict.status -eq "PASS") { 0 } else { 1 })
}
