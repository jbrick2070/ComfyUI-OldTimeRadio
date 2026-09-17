$ErrorActionPreference = "Continue"
$roots = @(
    "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes",
    "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\custom_nodes",
    "C:\Users\jeffr\Documents\ComfyUI\custom_nodes"
)
foreach ($r in $roots) {
    Write-Host "=== $r ==="
    if (Test-Path $r) {
        Get-ChildItem $r -Directory | Where-Object { $_.Name -match "OldTime|otr" } | ForEach-Object {
            Write-Host $_.FullName
        }
        Get-Item $r | Format-List FullName
    } else {
        Write-Host "MISSING"
    }
}
Write-Host "=== yaml ==="
Get-Content "C:\Users\jeffr\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1789257855597.yaml" -ErrorAction SilentlyContinue
Write-Host "=== object_info writer widgets ==="
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
@'
import json, urllib.request
url = "http://127.0.0.1:8188/object_info/OTR_LedgerScriptWriter"
try:
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
except Exception as e:
    print("FETCH FAIL", type(e).__name__, e)
    raise SystemExit(1)
info = data.get("OTR_LedgerScriptWriter") or next(iter(data.values()))
inp = info.get("input") or {}
names = []
for section in ("required", "optional"):
    block = inp.get(section) or {}
    names.extend(list(block.keys()))
print("REQUIRED:", list((inp.get("required") or {}).keys()))
print("COUNT", len(names))
print("HAS perfect_run_spacesaver", "perfect_run_spacesaver" in names)
print("FIRST8", names[:8])
print("creative default", (inp.get("optional") or {}).get("creative_writing_model"))
'@ | Set-Content -Path "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\_tmp_object_info.py" -Encoding utf8
& $py "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\_tmp_object_info.py"
