<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Direct contradictions with the locked sprint plan regarding frame math, canonicalize contracts, and JSON edits will cause immediate test failures.

MUST-FIX BEFORE BUILD:
1. [BUGS/RISKS] Frame math contradiction. The plan explicitly mandates the "SAME 8n+1 snap-DOWN (`((n-1)//8)*8+1`)" and says "do NOT diverge". The grounded `LTX_AV_SPRINT_PLAN.md` explicitly forbids this: "next_8n1(n) = ((n+6)//8)*8 + 1 (snap UP -- the legacy eng_ltx_video :281 formula snaps DOWN; never copy it)." Fix: Change BUGS/RISKS to mandate the snap UP formula and remove the instruction to mirror `eng_ltx_video`'s math.
2. [WIRING] JSON edit contradiction. The plan instructs auditing `otr_scifi_16gb_full.json` and adding the option if it's a static array. The grounded sprint plan explicitly states: "NO Director edits (V-6 auto-dropdown)." Fix: Remove the JSON audit/edit step entirely; rely on the V-6 auto-dropdown logic.
3. [WIRING] Canonicalize contract violation. The plan states the slice is padded/trimmed "BEFORE generation (not post-hoc in canonicalize)". The grounded sprint plan states: "canonicalize TRIMS to exactly T, or PADS-BY-LAST-FRAME to T (cap case)... AND stamps pad_tail_frames/padded_s". Fix: The audio slice can be padded before generation, but `canonicalize` MUST still trim/pad the resulting video to exactly `T` to satisfy the integer timing authority contract.

SHOULD-FIX:
4. [TICKETS] M1 ticket scope mismatch. The plan puts "schemas.py family + CAPABILITIES row" in M1. The sprint plan places the "schemas.py FAMILIES += ... FAMILY_REQUIRED_INPUTS" and "role_compat.py" edits in the driver wiring/goldens phase (which is M1/M3 in the sprint plan, but specifically grouped with driver deltas). Fix: Ensure the schema/role_compat edits are tested against the semantic-projection goldens before the graph is built.

OPTIONAL / NICE-TO-HAVE:
- [ARCHITECTURE] The `assert_usable` node gate mentions "NODE_CLASS_MAPPINGS via LAZY read". Ensure this lazy read is mocked properly in `test_video_ltx_av.py` as specified in the sprint plan's testing section.

CUT THESE (over-engineering):
1. [INVARIANTS] "JSON changes in otr_scifi_16gb_full.json same commit + re-validate." Safe to cut because the sprint plan explicitly forbids JSON/Director edits for this lane.