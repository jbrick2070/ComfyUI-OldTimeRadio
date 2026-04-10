# 16GB Flagship Hardening Q&A (v1.4)
### Q: Why did I see "Wing Dings" in the log when testing 2-bit?
**A:** Model Collapse. At 2-bit (especially on Gemma 9B/27B), the weights are too crushed to maintain a coherent signal. The output becomes mathematical noise.
**FIX:** v1.4 includes a **Wing Ding Shield** that forces an upgrade to 4-bit NF4 if unstable bit-depths are detected.

### Q: What is the "First-Name Shield"?
**A:** A roster guard that prevents two characters from sharing the same first name (e.g., preventing "Zuri vs. Zuri"). It ensures the LLM stays narratively focused.

### Q: Is 4GB or 2GB better for the Sovereignty Buffer?
**A:** On a 16GB card (RTX 5080), **2GB** is the sweet spot. We recalibrated down from 4GB to allow models like **Mistral Nemo 12B** more VRAM runway. Our "Zero-Prime" policy ensures the audio engine still has room to breathe.

### Q: How do I verify the 137W "Rumble" is working?
**A:** Check your hardware monitor during inference. You should see peak power draw and ~11.5 to 13.1 tok/s. If you see < 6 tok/s, the GPU is likely "waiting" on the CPU due to VRAM fragmentation.
