<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan is missing the concrete ending_tag/domain assignments for the 100 styles and the ENDING_TEMPLATES definitions, making the catalog unbuildable.

MUST-FIX BEFORE BUILD:
1. [§A] The plan does not specify the `ending_tag` and `domain` for each of the 100 style entries. Without this, the catalog cannot be built. Provide a complete mapping (e.g., updated catalog entries) assigning an ending_tag from the 8 listed and a domain (matching the domains used by `select_domain`) to every style.
2. [§A] The plan does not define the `ENDING_TEMPLATES` dictionary (the concrete final-beat instructions for each of the 8 ending tags). Provide the 8 template strings so the line composer has the injection content.
3. [§A] The plan says "keep prose as `ending_flavor`" but does not clarify whether the existing `ending_mode` field is renamed to `ending_flavor` or kept as `ending_mode`. This ambiguity could lead to inconsistent catalog structure. Specify the exact field names: e.g., rename `ending_mode` to `ending_flavor` and add `ending_tag` and `domain`.

SHOULD-FIX:
1. [§A] The `ENDING_TAG_BY_SLUG` external map is redundant because each catalog entry will have an `ending_tag` field. Remove it to simplify; the writer can look up the tag via `get_style(slug)["ending_tag"]`.
2. [§E] The plan says "The flag must reach `_assemble_outline` (via the request or a module-level read consistent with the writer gate)." It would be clearer to explicitly add a boolean parameter to `OutlineRequest` or `generate_outline` to control the announcer intent gating, rather than relying on an implicit module-level read. This avoids coupling to environment variable reading inside the outline module.

OPTIONAL / NICE-TO-HAVE: none.

CUT THESE:
1. [§A] `ENDING_TAG_BY_SLUG` external map — safe to cut because the catalog entry already contains the ending_tag.

VERIFY-AT-BUILD checklist:
- Flag OFF: byte-identical line prompt, announcer intent, and writer output (golden fixture).
- `validate_catalog()` passes (every entry has valid ending_tag, domain, and a template exists for its tag).
- `select_style` determinism: same inputs => same slug; emergency styles only selected when disaster keywords present; default pool excludes emergency.
- Flag ON: final-beat line prompt includes the ending template; announcer close intent is the non-outcome string.
- C7 full writer run with flag OFF produces byte-identical output (audio gate holds).
- Validation soak: crisis-noun density at final beat near 0; ending_tag distribution >= 80% non-doomsday; critic arc_verdict mix; A/B test on ~6 episodes.