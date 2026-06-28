CLAUDE ANCHOR -- HuMo r4 (residual defects on the standalone bakeoff)

VERDICT: near-converged; ONE real residual = the mid-graph reclaim must be SCOPED + ordered, not a
blanket idle reclaim. Two minor (IS_CHANGED, sentinel residency).

RESIDUAL MUST-FIX (anchor):
1. OTR_BakeoffReclaim ORDER + SCOPE (the load-bearing one). For the eviction to (a) run after
   conditioning and (b) NOT evict the 14B unet/VAE the KSampler still needs:
   - WIRE it on the CONDITIONING edge: pos/neg CONDITIONING -> OTR_BakeoffReclaim -> KSampler
     positive/negative, so it executes only once conditioning is done (umt5/whisper are spent).
   - SCOPE the reclaim: a blanket `reclaim_idle_models()` mid-graph may evict the UNETLoader/LoRA/
     ModelSamplingSD3 model if ComfyUI loaded it early and it reads as "idle" before KSampler runs.
     The node should evict ONLY the text+audio encoders (umt5 CLIP + whisper) -- pass a keep/allow set
     or call the encoder-targeted path, NOT a full idle sweep. Verify what reclaim_idle_models touches
     and whether it already honors a keep-list; if not, the bakeoff node needs a narrow encoder-only evict.
2. IS_CHANGED: the node must force re-execution every run (IS_CHANGED returns a unique value) so the
   executor never cache-skips the reclaim across legs / repeats.

MINOR / VERIFY:
3. SENTINEL leg: loading LTX-AV + Whisper then HuMo in one session is what production already does, so
   no class/import conflict expected -- but the sentinel must actually LOAD (a real LTX-AV render or at
   least a model-load prompt), not just import, to create representative residency. Measure the HuMo
   two-stage peak AFTER that load.
4. Confirm the standalone graph's HuMo audio-conditioning path (loadaudio/audioenc) is fully wired in
   the bakeoff JSON -- a missing audio edge would make the face render but mis-measure lip-sync quality.

If the panel raises nothing beyond (1)-(2) grounded, this is CONVERGED and coder-ready.
[ASSUMPTION] reclaim_idle_models is idle-scoped; the risk is timing (unet idle pre-KSampler), which the
encoder-only scope removes.
