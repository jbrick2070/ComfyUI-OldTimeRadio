# Character-voice whiny-fix roundtable WORKER. Usage: wf_pass_run.ps1 -Pass pass01 [-DryRun]
param([string]$Pass = 'pass01', [switch]$DryRun)
$ErrorActionPreference = 'Continue'
$py     = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$skill  = 'C:\Users\jeffr\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\cff774c2-bb97-4b73-b186-daa560353c46\6503cb4a-a0ec-4e89-b422-c889f09084d6\skills\roundtable'
$rt     = Join-Path $skill 'scripts\roundtable_pass.py'
$config = Join-Path $skill 'assets\panel.config.json'
$repo   = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
$camp   = "$repo\docs\2026-06-10-character-voice-whiny-fix\roundtable"
$out    = "$camp\$Pass"
$log    = "$out\run.log"
$doc    = "$repo\docs\2026-06-10-character-voice-whiny-fix__problem-statement.md"
$prompt = "$camp\open_prompt.md"
$models = "~openai/gpt-latest,~google/gemini-pro-latest,x-ai/grok-4.3,deepseek/deepseek-v4-pro"
$g = @(
  "$repo\nodes\_otr_delivery_vector.py",
  "$repo\nodes\_otr_audio_engines\eng_indextts2.py",
  "$repo\scripts\_otr_indextts2_worker.py",
  "$repo\nodes\_otr_voice_node_common.py",
  "$repo\nodes\_otr_voice_bank.py",
  "$repo\scripts\otr_dl_indextts2_refs.py",
  "$repo\nodes\_otr_script_prep.py"
)
New-Item -ItemType Directory -Force -Path $out | Out-Null
if ($DryRun) {
  & $py $rt --doc $doc --out $out --config $config --prompt $prompt --grounding $g --models $models --max-tokens 12000 --reasoning-effort none --budget 8.00 --dry-run *> $log
} else {
  & $py $rt --doc $doc --out $out --config $config --prompt $prompt --grounding $g --models $models --max-tokens 12000 --reasoning-effort none --budget 8.00 *> $log
}
"EXITCODE=$LASTEXITCODE" | Add-Content $log
