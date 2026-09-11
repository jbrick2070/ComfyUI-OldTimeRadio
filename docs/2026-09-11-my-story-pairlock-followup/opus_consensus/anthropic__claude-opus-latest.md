<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

## Verdict

Not clean. I find **three must‑fix defects** (one wrong output that defeats the stated purpose of change 5, one changed existing ledger contract in change 4, one crash/silent‑route risk in changes 1–2) plus several evidence gaps the added tests cannot catch. Details, each with a file/function and a reproducer.

---

### MUST‑FIX 1 — `otr_meta_brief_image_prompt._scene_source_context.character_context`: `"n/a"` masks the age you just exposed

```python
"age_band": row.get("age_band") or planned.get("age_band") or "",
```

`CastMember.age_band` (`_otr_my_story.py`, `_norm_age`) *defaults to the literal string `"n/a"`*, and normalizes every unknown to `"n/a"`. `"n/a"` is truthy. Two consequences:

1. A ledger cast row carrying `age_band: "n/a"` (the normal case for any row the writer did not age) **shorts the treatment join**, so the correction call receives `age_band: "n/a"` for an adult son whose treatment row says `"40s"`. That is exactly the pixel failure in the observed run ("a woman with two children"), and change 5's whole justification.
2. Where the treatment itself is the source, `planned.get("age_band")` returns `"n/a"` and the field is emitted as `"n/a"` rather than `""`, i.e. noise in the prompt instead of absence.

Reproducer: take `tests/test_my_story_visual_source.py::test_scene_owner_receives_current_ages_and_shared_action_without_literalizing_memory` and add `ledger["cast"][0]["age_band"] = "n/a"`, plus `"n/a"` on the companion row. Asserted `"30s"` / `"50s"` become `"n/a"` and the test fails. The suite passes today only because the fixture's ledger rows carry no `age_band` key at all — the tests do not exercise the field's real default. Same hazard for `gender`, where `canonical_bank_gender` may canonicalize a non‑empty junk value.

Fix in place: treat `"n/a"`/blank/whitespace as unset on both sides before choosing, and emit `""` when both are unset.

---

### MUST‑FIX 2 — `cast_lock._auto_registry`: removing the genderless `continue` silently changes three existing ledger contracts

The deleted branch did more than skip casting. Its `continue` ran **before** `_stamp_row`, so a genderless row:

* never entered `stamped_this_lock`, therefore the route‑tier sweep at lines 1275‑1289 never touched it. Now it does: a genderless row matching `tier_character_key` will be stamped `ROUTE_TIER_UNROUTED` with `provisional_route_id`/`provisional_reason`. That is a new assertion ("the ordinary seeded draw chose this voice") about a row that took a *fallback* draw, on a field the plan says is read downstream as an enumeration. Reproducer: any `_lemmy_voice_policy()` character key matching a blank‑gender row under `auto_registry`; previously no tier field, now `unrouted`.
* never reached `_stamp`, so it had no `presentation_gender`. `_stamp` (line 1651) now writes the *reference's* gender onto a row whose own `gender` is `""`. Any downstream consistency reader comparing `gender` to `presentation_gender` now sees a row declaring `""` and presenting `male`. The plan claims "retaining the row's actual gender unchanged"; that is true of `gender` only.
* never contributed to `gated`, so the "known‑gated" non‑blocking warning count and report text change for existing episodes.

Also: `voice_cast_fallback` now conflates two distinct causes — "bank has no rows for `other`" and "source never stated a gender" — under one token `gender_unservable`, and the report line renders as `gender '' unservable`. You deliberately deleted the only signal that distinguished them (`"no gender -- preserved"`). Source qualification in the next run cannot tell blank‑gender casting from unservable‑gender casting from the ledger. Use a distinct fallback token (e.g. `gender_unspecified`) — it is a string stamp, not a new writer or gate.

Stale comments now lie about live behaviour and must be corrected in the same change: lines 1036‑1042 ("a row with no usable gender ... fall through one of the loop's `continue`s") and 1267 ("no usable gender ... took no draw at all"). Those are the receipts a future reader will trust.

Minor, same function: with the early `not gender and target_engine == "google_tts"` raise at 1176, the `if target_engine == "google_tts": raise` at 1221 is now unreachable for the blank‑gender case. Harmless but dead; the control‑flow trick of raising `VoiceCastingError(f"{char_id}: source gender unspecified")` at 1202 purely to land in an `except` block is a readability regression in the one function you documented as historically mis‑ordered.

---

### MUST‑FIX 3 — `_otr_story_source.rewrite_story_source`: unguarded `accepted.model_dump()` and un‑registered list fields

Two distinct holes in the same function:

**(a) `accepted` can be `None` at line 263.** The invariant "structured_call always invokes `post_validator` on the attempt it returns" is assumed, never asserted. Any success path in `structured_call` that returns without calling the post‑validator (cache hit, zero‑attempt short‑circuit, an internal `model_construct`) yields `AttributeError: 'NoneType' object has no attribute 'model_dump'` *after* a real model call, with the receipt frozen at `status="preparing"` — a stale receipt on a spent budget. Note the previously returned `corrected` is now unused (dead local; also drops any post‑processing `structured_call` applies to its return value). Add an explicit check: if `accepted is None`, record `status="owner_contract"` and raise a named error.

**(b) A list field with no entry in `preserve_omitted` gets no conservation at all.** `_retain_omitted` only recurses into list items when `field_path in identities`; otherwise items are `model_dump`ed wholesale, so **omitted sub‑fields of those items silently take Pydantic defaults** — precisely the P0 defect being repaired. Today `_call` registers five paths by hand. Nothing fails loudly if a schema gains a list, or if a path is mistyped (e.g. `("named_cast",)` vs a renamed field): the conservation just quietly stops. Reproducer: rename `conflicts` in `StoryInterpretation` and the omission conservation for it disappears with 362 tests still green. Fix: at entry, assert every list‑of‑BaseModel field reachable in `schema` has a registered identity, and raise on an unknown/unused `preserve_omitted` key.

**(c) Related, and untested:** `_retain_omitted` returns data keyed by *field names*, then `schema.model_validate(...)` re‑validates it. That only works while every aliased model sets `populate_by_name=True` (`CastMember` does — line 209; the others are unverified here). If any nested model has an alias without it, `model_validate` raises `ValidationError` **inside `post_validator`**, which `rewrite_story_source`'s own except clause (line 249) classifies as a normal structural failure. The result is `status="unresolved"`, a burned budget and a receipt that blames the model for a bug in the conservation code. The same misattribution applies to