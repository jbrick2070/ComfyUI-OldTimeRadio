# r3 Anchor Review (Claude driver, pre-fan-out)

Target: the LANDED v2 code (commit fe0a33f) + r2/final.md.
Focus: wiring / integration / sequencing.

VERDICT: wired and green (41 tests + smoke on the ComfyUI venv). Four
self-identified wiring risks for the panel to pressure:

MUST-FIX:
1. Validator cache staleness: OTR_UpstreamStoryLabValidator reads
   production_mirror for drift checks, but nodes._lab_state_digest() walks
   only src/ + fixtures/ + nodes.py + __init__.py - a refreshed mirror will
   NOT invalidate the ComfyUI cache, so a stale OK can render after a
   re-mirror. Fix: include production_mirror/ files in the digest walk (30
   files, cheap).
2. Bridge artifact handoff location: OTR_BridgeArtifactEmit writes
   bridge_out/<name>.json. The production adapter must take an EXPLICIT
   path input (forceInput STRING socket), never scan a conventional folder -
   convention scanning is a hidden default. State this in PATCH_PLAN item 1.

SHOULD-FIX:
3. Widget default rule for tomorrow: when writer widgets land, defaults MUST
   be source_bank=science_news / story_model=auto / story_pipeline=auto /
   visual_style=auto so an untouched canonical workflow behaves
   byte-identically. Say it once, in PATCH_PLAN, as an append-only gate.
4. nodes.py INPUT_TYPES raises RegistryError at ComfyUI registration when
   fixtures are broken (fail-loud, correct) - but the error surfaces as an
   import failure of the whole pack. Acceptable; consider wrapping with a
   message that names the offending fixture file first (RegistryError
   already does).

UNVERIFIABLE: none; all claims grounded in the landed files.
