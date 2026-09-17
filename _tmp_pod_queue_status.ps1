$ErrorActionPreference = "Stop"
$key = "C:\Users\jeffr\.ssh\runpod_otr"
$hostName = "213.173.105.175"
$port = 13317
$ssh = @("-i", $key, "-p", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25")
ssh @ssh "root@${hostName}" "echo ===queue===; curl -fsS http://127.0.0.1:8188/queue; echo; echo ===runner_tail===; tail -n 40 /workspace/otr-config/foley_mystory_3act_chatgpt.log"
