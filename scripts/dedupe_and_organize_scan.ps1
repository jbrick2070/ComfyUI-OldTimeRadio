# ComfyUI model dedupe + organization scan
# Read-only: produces a report at scripts\dedupe_report.txt. No files are modified.
# 2026-04-21

$ErrorActionPreference = 'SilentlyContinue'
$out     = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\dedupe_report.txt'
$jsonOut = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\dedupe_report.json'

$roots = @(
  'C:\Users\jeffr\Documents\ComfyUI\models',
  'C:\Users\jeffr\.cache\huggingface\hub',
  'C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\models'
)
$exts = @('*.safetensors','*.ckpt','*.gguf','*.bin','*.pth','*.pt','*.onnx')

# Canonical location preference order (first match wins as the "keeper")
$canonicalOrder = @(
  'C:\Users\jeffr\Documents\ComfyUI\models\',        # ComfyUI canonical
  'C:\Users\jeffr\.cache\huggingface\hub\',          # HF cache (leave alone, system)
  'C:\Users\jeffr\AppData\Local\Programs\ComfyUI\'   # Stale bundled install (delete first)
)

function Get-CanonicalRank($path) {
  for ($i = 0; $i -lt $canonicalOrder.Count; $i++) {
    if ($path.StartsWith($canonicalOrder[$i], [System.StringComparison]::OrdinalIgnoreCase)) { return $i }
  }
  return 999
}

'# ComfyUI Dedupe + Organization Report' | Out-File -FilePath $out -Encoding utf8
('Generated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) | Out-File -FilePath $out -Append -Encoding utf8
'' | Out-File -FilePath $out -Append -Encoding utf8

# ---- Phase 1 : collect ----
$files = @()
foreach($r in $roots){
  if(Test-Path $r){
    $files += Get-ChildItem -Path $r -Recurse -File -Include $exts -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.Length -gt 100MB }
  }
}
('## Phase 1 -- files collected: ' + $files.Count) | Out-File -FilePath $out -Append -Encoding utf8
('Total bytes: ' + [math]::Round(($files | Measure-Object Length -Sum).Sum / 1GB, 2) + ' GB') | Out-File -FilePath $out -Append -Encoding utf8
'' | Out-File -FilePath $out -Append -Encoding utf8

# ---- Phase 2 : hash ----
$hashed = New-Object System.Collections.ArrayList
$i = 0
$progressFile = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\dedupe_progress.txt'
foreach($f in $files){
  $i++
  ("$i/$($files.Count) $($f.FullName) size=$([math]::Round($f.Length/1GB,2))GB") | Out-File -FilePath $progressFile -Encoding utf8
  try {
    $h = (Get-FileHash -Algorithm SHA256 -Path $f.FullName -ErrorAction Stop).Hash
    $null = $hashed.Add([pscustomobject]@{
      Path = $f.FullName
      Size = $f.Length
      SHA256 = $h
      Rank = Get-CanonicalRank $f.FullName
    })
  } catch {
    ('HASH_ERROR ' + $f.FullName + ' : ' + $_.Exception.Message) | Out-File -FilePath $out -Append -Encoding utf8
  }
}

# ---- Phase 3 : group ----
$groups = $hashed | Group-Object SHA256 | Where-Object { $_.Count -gt 1 }

'## Phase 2 -- duplicate hash groups' | Out-File -FilePath $out -Append -Encoding utf8
('Duplicate groups found: ' + $groups.Count) | Out-File -FilePath $out -Append -Encoding utf8
'' | Out-File -FilePath $out -Append -Encoding utf8

$totalReclaim = 0
$deleteList = New-Object System.Collections.ArrayList
foreach($g in ($groups | Sort-Object { -($_.Group[0].Size) })){
  ('### Hash ' + $g.Name.Substring(0,16) + '...  copies=' + $g.Count + '  size/copy=' + [math]::Round($g.Group[0].Size/1GB,2) + 'GB') | Out-File -FilePath $out -Append -Encoding utf8
  # Keeper: lowest Rank (canonical). Tiebreak: shortest path.
  $sorted = $g.Group | Sort-Object Rank, { $_.Path.Length }
  $keeper = $sorted[0]
  ('  KEEP   ' + $keeper.Path) | Out-File -FilePath $out -Append -Encoding utf8
  for ($j = 1; $j -lt $sorted.Count; $j++) {
    $dup = $sorted[$j]
    ('  DELETE ' + $dup.Path + '  (reclaims ' + [math]::Round($dup.Size/1GB,2) + ' GB)') | Out-File -FilePath $out -Append -Encoding utf8
    $totalReclaim += $dup.Size
    $null = $deleteList.Add($dup.Path)
  }
  '' | Out-File -FilePath $out -Append -Encoding utf8
}

('## Total reclaimable from dedupes: ' + [math]::Round($totalReclaim/1GB,2) + ' GB') | Out-File -FilePath $out -Append -Encoding utf8
'' | Out-File -FilePath $out -Append -Encoding utf8

# ---- Phase 4 : organization check ----
# Flag files that live OUTSIDE the canonical ComfyUI models tree.
# HF hub + AppData ComfyUI are OK locations; they just shouldn't be duplicated.
# Inside Documents\ComfyUI\models\, check for "orphans" -- files in unusual subdirs.
'## Phase 3 -- folder organization audit (Documents\ComfyUI\models\ only)' | Out-File -FilePath $out -Append -Encoding utf8

$comfyModelsRoot = 'C:\Users\jeffr\Documents\ComfyUI\models'
$knownDirs = @(
  'checkpoints','clip','clip_vision','configs','controlnet','diffusers',
  'diffusion_models','embeddings','florence2','gligen','huggingface',
  'hypernetworks','latent_upscale_models','loras','photomaker','pulid',
  'style_models','text_encoders','TTS','unet','upscale_models','vae',
  'vae_approx','WorldMirror-V2','xlabs'
)
$orphans = @()
foreach($f in $files){
  if($f.FullName.StartsWith($comfyModelsRoot, [System.StringComparison]::OrdinalIgnoreCase)){
    $rel = $f.FullName.Substring($comfyModelsRoot.Length + 1)
    $topDir = ($rel -split '\\')[0]
    if($knownDirs -notcontains $topDir){
      $orphans += [pscustomobject]@{ Dir = $topDir; Path = $f.FullName; Size = $f.Length }
    }
  }
}
if($orphans.Count -eq 0){
  '  (no orphan files -- all live under canonical subdirs)' | Out-File -FilePath $out -Append -Encoding utf8
} else {
  ('  Found ' + $orphans.Count + ' files in unexpected subdirs:') | Out-File -FilePath $out -Append -Encoding utf8
  foreach($o in $orphans){
    ('    ' + [math]::Round($o.Size/1GB,2) + ' GB  [' + $o.Dir + ']  ' + $o.Path) | Out-File -FilePath $out -Append -Encoding utf8
  }
}

'' | Out-File -FilePath $out -Append -Encoding utf8

# ---- Emit JSON for a follow-up execute script ----
$json = @{
  generated = (Get-Date -Format 's')
  total_files = $files.Count
  duplicate_groups = $groups.Count
  reclaim_bytes = $totalReclaim
  deletes = $deleteList
} | ConvertTo-Json -Depth 5
$json | Out-File -FilePath $jsonOut -Encoding utf8

'DONE' | Out-File -FilePath $out -Append -Encoding utf8
