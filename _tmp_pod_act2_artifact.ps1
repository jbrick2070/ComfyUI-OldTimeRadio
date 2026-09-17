$ssh = @(
    "-i", "C:\Users\jeffr\.ssh\runpod_otr",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=20",
    "-p", "43125",
    "root@213.173.109.173"
)
$remote = @'
echo "=== pending ==="
ls -lt /workspace/runpod-slim/ComfyUI/output/otr/episodes | head
echo "=== find attempt receipts ==="
find /workspace/runpod-slim/ComfyUI/output/otr -name "*20260914_2031*" 2>/dev/null | head
find /workspace/runpod-slim/ComfyUI/output/otr/episodes/_shared/state/story_drafts -name "*af7641*" 2>/dev/null | head
ls /workspace/runpod-slim/ComfyUI/output/otr/episodes/pending_20260914_203120 2>/dev/null | head
'@
& ssh @ssh $remote
