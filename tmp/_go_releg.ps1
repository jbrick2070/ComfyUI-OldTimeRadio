# RE-LEG the three banks that died in the 420 sweep, against the fixes shipped
# during it. Sweep name bake420b -- the 720 gate reads bake420 + bake420b together
# and a green re-leg supersedes the original FAIL.
#
#   scifi_codex          P4 audit died on its own 240-char rationale      -> 80e30c2e
#   public_domain_story  reasoning model burned the token floor thinking  -> 707d4725
#   scifi_fable2         critic required a `speaker` key that may be empty-> 707d4725
#
# CALL _sweep.ps1 IN-PROCESS with a REAL array. Do NOT shell out via
# `powershell -File ... -Banks $array`: -File marshals every argument as a STRING,
# the array does not survive the process boundary, the Mandatory -Banks bind fails,
# and the sweep exits in ~1 second having run nothing (bit me at 10:57 today --
# the same class as the -Set array gotcha in the handoff).
$banks = @("scifi_codex", "public_domain_story", "scifi_fable2")

& "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\_sweep.ps1" `
    -Banks $banks -Words 420 -SweepName "bake420b"
