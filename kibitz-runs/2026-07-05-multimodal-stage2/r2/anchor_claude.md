# ANCHOR REVIEW (Claude, code-grounded) -- STAGE2_SUBPLAN v2, r2 (coding plan)

VERDICT: implementable with 1 MUST-FIX I found in my own v2 design, plus 2
SHOULD-FIX. Grounded against nodes/_otr_story_pack.py (working tree),
tests/test_story_pack_stage1.py, _otr_creative_prompt_router.py.

## MUST-FIX

M1 (CONFIRMED, self-caught). **`_validate(extra_seams=...)` poisons the shared
`_PACK_CACHE`.** _otr_story_pack._PACK_CACHE (line 91) keys by resolved path
only. If the routing layer loads the simple_4 pack with
extra_seams={pass_1..pass_4} and caches it, a later STRICT `load_pack(path)`
returns the cached object without re-validation -- the strict check silently
never runs (an admission-control hole, the exact class of bug the allowlist
exists to prevent). Conversely a strict-first load raises and never caches, so
behavior depends on call order. Fix (pick in 2A): keep `load_pack` strict-only
and UNCHANGED; add `_otr_story_pack.load_pack_with_seams(path, extra_seams:
frozenset)` with its OWN cache keyed `(resolved_path, extra_seams)`; the
routing layer is its only sanctioned caller (extend the consumer-guard test to
pin that: no production file besides _otr_story_routing.py may call
load_pack_with_seams). Stage-1 semantics of load_pack stay byte-for-byte.

## SHOULD-FIX

S1. Registry sweep cost/boot: the sweep validates every pack under
nodes/story_packs/ at first registry access. With 4 packs this is trivial, but
pin WHERE first access happens in 2A -- module import of _otr_story_routing
must NOT trigger I/O (lazy: sweep on first get_bank/resolve call, cached).
Import-time I/O in a ComfyUI custom-node tree slows/booby-traps boot
(antigravity r1 M2's kernel of truth, answered without the fallback).

S2. Error taxonomy: mirror Stage 1 exactly -- StoryRoutingError(Exception) base
+ UnknownBankError / UnknownPipelineError / RegistryValidationError /
StoryBankNotRunnableError. Tests assert the TYPED class, not message text
(message text pins make future wording edits churn tests).

## CONFIRMED-GOOD (v2 decisions re-checked)

- resolve_story_pack("science_news") in the router: the existing equivalence
  test (test_story_pack_stage1.py:225-238) pins VALUE identity to L._SYSTEM_PROMPT,
  so a routing regression fails the suite -- no new byte-identity scaffolding needed.
- Exact-equality allowlist test survives (no global growth).
- Caller-count pin (EXACTLY 4 resolve_creative_system_prompt production callers)
  is unaffected: _otr_story_routing calls load_pack*, not the resolver.
