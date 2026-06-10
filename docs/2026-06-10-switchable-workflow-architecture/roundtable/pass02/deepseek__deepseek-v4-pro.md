<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? **no** — design gaps around stamp handling, VRAM ceiling fallback, and missing widget-name mappings would block implementation.

MUST-FIX BEFORE BUILD:
1. [VRAM safety / wrapper_bridge] The plan states wrapper_bridge “resolves the in‑graph profile stamp -> committed profile vram_budget_mb” as a fallback. wrapper_bridge is a module‑level constant with no access to the workflow graph or stamp. Fix: drop the stamp‑based fallback; make wrapper_bridge read OTR_VRAM_CEILING_MB env only (fall back to 14500). The launcher must set OTR_VRAM_CEILING_MB from the profile budget; the startup assertion (e.g. in the validator node) handles hardware‑vs‑profile mismatch, not wrapper_bridge.

2. [Headless parity test] `to_api_prompt(snapshot(P)) == to_api_prompt(apply(master,P))` will fail if the generator adds a profile stamp that apply(master,P) does not produce (the plan says the generator writes the stamp). Fix: either do not embed the stamp in the snapshot (e.g. store it only on disk as metadata), or adjust the parity test to strip the stamp fields before comparison.

3. [Profile stamp location] The document says the stamp goes into “node‑63 widget + node properties” but never names the widget. Node 63 currently has three widgets; a new stamp widget would change the class and break the widget‑vector contract unless carefully added. Fix: specify the exact widget name (e.g. `profile_stamp`), add it to OTR_WorkflowValidator INPUT_TYPES, and ensure the generator writes the stamp there. Without this the startup assertion cannot read the stamp.

4. [Engine widget mapping] The profile‑managed widget map lists expected values (“indextts2”, “kokoro”, etc.) but never states the widget names required by the patcher. `apply()` must patch by name; without a document mapping profile keys → widget names (e.g. `char_voice` → `character_voice_engine`) the applier cannot function. Fix: complete the mapping using actual node‑class INPUT_TYPES and commit it to the design. (The “verify at build” note for node 92 is one instance; the same gap exists for nodes 81, 82, 83.)

5. [Byte‑identical check] CI asserts `apply(master, 16gb_full) == master` byte‑identical. If the master is hand‑edited and lacks the profile stamp, while apply(master) would add it (per the stamp design), the check fails. Fix: clarify that apply() never adds the stamp; the stamp is a separate generator step. Self‑check must compare the widget‑patched master (unstamped) with the original master.

SHOULD-FIX:
6. [Node 92 widget name] Explicitly flagged as “verify at build”; the current plan cannot be actioned without that name. Resolve before coding.
7. [Over‑engineering: creative whitelist enforcement] Raising on post‑apply widget patching requires state tracking; regression tests (grepping for banned patches) are adequate. Cut the runtime guard to simplify the applier.
8. [Startup assertion placement] No node is designated to resolve the stamp. The natural home is an extended OTR_WorkflowValidator, but the master (unstamped) must still run without the assertion. Clarify that the assertion is only active when a stamp is present, or that generated snapshots contain a dedicated stamp‑check node.
9. [Master as source of truth] If the master is never stamped, the startup assertion must tolerate its absence; document that the master is NOT directly runnable with profile validation (users run snapshots).

OPTIONAL / NICE‑TO‑HAVE:
- Keep apply() a pure widget‑patching function free of stamp logic.
- Provide a clear mapping document of profile key → widget name with references to the node class.
- Include a pre‑flight check that all required widget names are discoverable via NODE_CLASS_MAPPINGS.

CUT THESE (over‑engineering):
- The runtime guard that raises when a profile‑managed widget is patched outside apply(). Safe to replace with a regression script that greps for such patches; the guard adds fragile state and false‑positive risk without proven benefit.

Mark [ASSUMPTION]: widget names (char_voice, announcer_voice, etc.) can be extracted from node classes; this is unverified in the provided grounding. All such names are assumptions until validated.