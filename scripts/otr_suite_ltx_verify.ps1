Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$env:HF_HOME = "C:\ComfyUI-Models\huggingface"
& "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe" -m pytest tests/ -x -q --tb=short 2>&1
