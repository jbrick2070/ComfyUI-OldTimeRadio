# ANCHOR REVIEW (Claude, code-grounded) -- STAGE2_SUBPLAN.md r1

VERDICT: SOUND DIRECTION, 2 MUST-FIX, 3 SHOULD-FIX. All claims grounded against
the real files (nodes/_otr_story_pack.py, nodes/_otr_creative_prompt_router.py,
nodes/_otr_outline.py:1832-1857, tests/test_story_pack_stage1.py,
docs/multimodal-story-schema/schema-examples/{banks,pipelines}.json + story_packs).

## MUST-FIX

M1 (CONFIRMED). **simple_4 pass seams entering PRODUCTION_SEAM_ALLOWLIST in 2B
violates the Stage-1 allowlist rationale.** _otr_story_pack.py:26-27 states the
allowlist is "the EXACT set of seam keys Stage 1 authors + validates. Not a
superset: a reserved-but-unpinned name would let an unauthored seam pass
silently." The byte-identity pin (test_story_pack_stage1.py _expected_live)
maps every allowlist seam to a LIVE production constant. pass_1..pass_4 have NO
production constant and NO consumer -- adding them to the same global allowlist
makes them exactly the "reserved-but-unpinned" names the design forbids.
Resolution: the sub-plan's own open question should be answered PER-PIPELINE
seam declaration -- pipelines.json declares its seam_refs, and the validator
checks a pack's prompt_stages keys == (production allowlist INTERSECT consumed)
UNION (seams declared by the pack's own story_pipeline_id). Keys are still
strictly policed; nothing global grows.

M2 (CONFIRMED). **2A cross-ref "header triple matches" is under-specified for
multi-pack banks.** banks.json only names default_story_model, but 2B puts one
pack per lane while schema-examples carry 5 media_archive + 5 public_domain
packs. The validator must define: is every on-disk pack under
story_packs/<bank>/ validated (directory sweep), or only referenced defaults?
Draft answer: SWEEP the directory -- any pack file that exists must validate
and its header triple must match its path coordinates; an orphan/misfiled pack
is a hard error. Otherwise a typo'd pack sits invisible until selected.

## SHOULD-FIX

S1. The router's caller-count test (test_creative_prompt_router.py:173-188 pins
EXACTLY 4 production callers) and the sanctioned-consumer guard
(test_story_pack_stage1.py:208-220, allowed={_otr_story_pack.py,
_otr_creative_prompt_router.py}) BOTH need same-commit updates in 2A when
_otr_story_routing.py starts importing the loader. The sub-plan mentions the
consumer guard but not the caller-count pin -- name both.

S2. `executable_in_lab` rename to `executable`: good, but also validate
executable=false pipelines are REJECTED at resolve-time with a message naming
the pipeline id AND that this is expected until the pass runner ships --
otherwise the raise reads as a defect.

S3. Cache invalidation: _PACK_CACHE keys by resolved path; the new registry
cache must also be clear()-able for tests (Stage 1 tests clear _PACK_CACHE
directly). Give _otr_story_routing.py an explicit _clear_caches() test hook so
tests do not reach into module privates.

## CONFIRMED-GOOD

- Precondition already shipped: the outline swallow at _otr_outline.py is
  removed and pinned by test_outline_resolver_call_not_swallowed (AST no-Try
  ancestor check). Verified in the working tree.
- Keeping default_visual_style as an opaque validated string is right (Stage 3
  resolves it); rejecting it would force a schema bump.
- Chunk 2C (widget) deferred + gated is correct per BUG-LOCAL-097 positional
  widgets_values discipline.
- Seam-name law (production names, not lab names) is correct; the lab pack
  content (e.g. faithful_radio_adaptation.json) uses outline_system /
  pitch_room_system etc. which the Stage-1 validator would hard-reject today.
