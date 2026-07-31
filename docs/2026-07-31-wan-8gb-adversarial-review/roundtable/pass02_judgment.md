# Pass 02 judgment: experiment design

The local review found four material omissions in the Codex anchor:

1. Canonical inputs were not sufficiently replayable to make the mechanism A/B
   causal.
2. Current single-mode receipts discard fields required to audit calibration.
3. A bundle comparison alone cannot isolate Dynamic versus legacy patching.
4. A generic multi-clip soak would not reproduce Wan's capped-clip/ping-pong
   production behavior.

All four are accepted. The final campaign therefore starts with a fixed-input
canonical replay and receipt contract; compares the same native artifacts under
Dynamic and legacy patching; then compares native-legacy with GGUF-legacy and
the actual stock server configuration. DisTorch remains a later optional arm.

The final length/pixel design reuses 832x480/17/tiled, then adds 512x288 at 17
and 129, 832x480 at 65 and 129, and 832x480/129 untiled. Physical 8 GB remains a
release gate regardless of clamped results.
