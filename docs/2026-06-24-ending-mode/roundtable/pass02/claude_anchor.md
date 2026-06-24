# R2 anchor review (Claude, code-grounded) — coding plan / implementability

Focus: is the R1-hardened plan buildable as written? Claims CONFIRMED / MISREAD /
UNVERIFIABLE.

## VERDICT

Buildable, but three implementation specifics must be pinned or the build drifts.

## MUST-FIX

1. **OutlineRequest threading — use a new FROZEN optional field, mirror the
   precedent.** CONFIRMED: `OutlineRequest` is `@dataclass(frozen=True)` with
   optional defaulted fields added before (script_brief, diversity_hint,
   prior_macro, prior_critique). Add `ending_tag: str = ""` and
   `ending_flavor: str = ""` (empty => byte-identical, asserted) the SAME way.
   The selector fills them via `dataclasses.replace`, exactly like the pitch
   room fills `script_brief`. No new mutable state.

2. **Final-beat detection must REUSE l12, not re-derive.** CONFIRMED:
   `assign_beat_roles(ordered_char_beat_ids)` already tags the last voiced
   CHARACTER beat `irreversible_choice`. The injection must locate that beat via
   the same helper (not "last beat in the list" — the announcer outro follows
   it). Render the ending_tag template into THAT beat's prompt only.

3. **Selector home + determinism.** Put the deterministic selector in
   `_otr_style_catalog.py` (it already has `get_style`/`non_emergency_slugs`/
   `EMERGENCY_TAG`) as `select_style(premise, meta, seed) -> slug`, reusing the
   keyword approach of `select_domain` (l12) — do NOT duplicate the keyword map;
   import/extend it. Deterministic tie-break off the existing style/cast seed so
   the C7 path holds. UNVERIFIABLE until wired — assert byte-identity in a test.

## SHOULD-FIX

4. **Catalog data migration.** Every one of the 100 entries needs an `ending_tag`
   from the 8-enum. That is a bulk authoring pass; add a startup/test assertion
   that every entry has a valid tag (no silent gaps) — mirrors the project's
   "fail loud" discipline.

5. **Prompt injection must be additive + gated.** Render the ending block ONLY
   when `req.ending_tag` is non-empty (empty => byte-identical prompt, asserted),
   exactly like the diversity_hint overlay precedent in `_build_beat_user_prompt`.

6. **Where the selector runs.** It needs the premise, which exists only AFTER the
   macro/outline stage. So select style AFTER the outline premise is known but
   BEFORE the per-beat composition — and thread the tag into the beat prompt.
   The current style string is chosen much earlier (pre-outline). Decide: keep the
   early style label for back-compat surfaces, add the ending_tag as a separate
   late-bound signal. UNVERIFIABLE ordering — pin in R3 wiring.

## Test plan (concrete)

- `ending_tag=""` => beat prompt byte-identical (golden).
- selector determinism: same (premise, seed) => same slug; emergency slug only on
  disaster keywords; default pool excludes emergency.
- every catalog entry has a valid `ending_tag` (coverage assert).
- C7: flag OFF => writer output + beat roles byte-identical.
