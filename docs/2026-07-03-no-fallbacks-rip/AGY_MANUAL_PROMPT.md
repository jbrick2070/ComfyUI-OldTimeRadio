# Manual Antigravity (agy) prompt — no-fallbacks rip, r2 implementability

Run this from the repo root in your terminal (Antigravity reads the repo itself).
Paste its output back to me and I'll ground every claim against the real code and
fold only the survivors into the hardened plan.

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
agy -p "You are a code-grounded reviewer. Read the REAL repo you are sitting in. Review the plan in docs/2026-07-03-no-fallbacks-rip/PLAN.md — a stack-wide 'no fallbacks, fail hard' rip of the LOCAL audio + LLM lanes (cloud voice/music already ship no-fallback; the video lane was already ripped 2026-07-02). This is the r2 IMPLEMENTABILITY round. Ground every claim against the actual files (nodes/_otr_voice_node_common.py, nodes/cast_lock.py, nodes/_otr_engine_profiles.py, nodes/stable_audio_theme.py, nodes/_otr_audio_engines/*, nodes/OTR_LedgerScriptWriter.py, story_orchestrator.py, and the tests/ that pin the fallback behavior). Output VERDICT + MUST-FIX + SHOULD-FIX. Focus on: (1) exactly which tests will break when each fallback is ripped and whether the plan retires/inverts them in the same commit; (2) whether ripping cast_lock._resolve_character_voices_fail_soft (called unconditionally at cast_lock.py:187) needs a producer-side writer cast-contract fix so episodes don't just fail earlier; (3) whether R1a (bark net) alone leaves the suite green or is entangled with R1b (cast_lock) and should be one commit; (4) any place the rip risks changing the BYTE-IDENTICAL happy-path render (indextts2+valid-ref, kokoro+valid-voice, SA3 music) rather than only removing the fallback branch; (5) missing_ref_fallback metadata sites that must all be removed together. Cite file:line for every claim. Do NOT edit anything — review only."
```

If `agy` complains about permissions on the read, add `--dangerously-skip-permissions`
(it is git-committed, so any stray write shows in `git status`).
