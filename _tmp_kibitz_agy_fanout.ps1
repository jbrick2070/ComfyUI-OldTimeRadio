$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:KIBITZ_AGY_PRINT_TIMEOUT = "15m"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$kibitz = "C:\Users\jeffr\.codex\skills\kibitz\scripts\kibitz.py"
$doc = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\kibitz-runs\2026-09-15-cloud-video-fanout\r4\input.md"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
& $py $kibitz --check-pins
# ComfyUI local profile is ~1600 lines. agy -p= embeds the whole prompt on
# argv and Windows then raises WinError 206. --no-profiles keeps the lane;
# the finished-diff brief already names the files to read.
& $py $kibitz --doc $doc --round r4 --topic cloud-video-fanout --repo $repo --driver cursor --only agy --no-profiles --timeout 900
exit $LASTEXITCODE
