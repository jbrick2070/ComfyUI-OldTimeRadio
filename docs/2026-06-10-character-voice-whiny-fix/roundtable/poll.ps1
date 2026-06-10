# Poll the whiny-fix roundtable pass. Usage: poll.ps1 -Pass pass01
param([string]$Pass = 'pass01')
$ErrorActionPreference = 'Continue'
$camp = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-06-10-character-voice-whiny-fix\roundtable'
$out  = Join-Path $camp $Pass
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*roundtable_pass*' }
if ($procs) {
  $procs | ForEach-Object { Write-Output ("RUNNING pid=" + $_.ProcessId + " since " + $_.CreationDate) }
} else {
  Write-Output "NO roundtable python process running"
}
Write-Output "--- files in $Pass ---"
Get-ChildItem $out -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ($_.Name + "  " + $_.Length + " bytes") }
$log = Join-Path $out 'run.log'
if (Test-Path $log) {
  Write-Output "--- run.log tail ---"
  Get-Content $log -Tail 15 -ErrorAction SilentlyContinue
}
