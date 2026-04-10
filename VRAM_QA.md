# 16GB Flagship Hardening Q&A (v1.4.9)
### Q: Why did I see "Wing Dings" in the log when testing 2-bit?
**A:** Model Collapse. At 2-bit (especially on Gemma 9B/27B), the weights are too crushed to maintain a coherent signal. The output becomes mathematical noise.
**FIX:** v1.4.9 now includes a **Wing Ding Shield** that forces an upgrade to 4-bit NF4 if unstable bit-depths are detected.

### Q: Why was my Voice Health check taking 90+ seconds?
**A:** CPU Ghosting. The Bark model was defaulting to CPU during the early startup phase.
**FIX:** Explicitly forced device='cuda' in the health check path.

### Q: How do I verify the 137W "Rumble" is working?
**A:** Check your hardware monitor during inference. You should see peak power draw and ~11.5 tok/s. If you see < 6 tok/s, the GPU is likely "waiting" on the CPU due to VRAM fragmentation.
