$ssh = @(
    "-i", "C:\Users\jeffr\.ssh\runpod_otr",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=20",
    "-p", "43125",
    "root@213.173.109.173"
)
$remote = @'
sed -n "346,371p" /workspace/otr-config/comfy_8188.log
echo ====
sed -n "370,400p" /workspace/otr-config/comfy_8188.log
'@
& ssh @ssh $remote
