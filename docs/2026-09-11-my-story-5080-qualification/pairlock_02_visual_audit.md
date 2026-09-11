# pairlock_02: published episode, failed source visuals

Read-only evidence audit, 2026-09-11. Runtime/source HEAD inspected:
`bd8141487f2bbc093451cd7573cb3b338874b8e4`. No production changes, tests,
model calls, or hardware operations were performed for this audit.

Episode: `C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes/signal_lost_the_la_table_20260911_022847`.
The successful terminal publication does not qualify the images against the source.

## Observed failure and authoritative source

Both actual PNGs were independently viewed:

| File under `stills/` | Visible content | File SHA-256 |
| --- | --- | --- |
| `still_shot_001_b1_fbb588dee05a.png` | One adult man alone at a bare wooden kitchen table, without food or tableware. | `088179db51c874593a6874c6e65dc0b86edf0ce5fa3fb0c50f0cccf31bbba010` |
| `still_shot_001_b2_069b14658760.png` | An adult woman and two young boys at a bare wooden table, without food or tableware. | `0174f055a9d606322f98179a026fe7a694d3a597027d061b01e62378058e5f31` |

The filenames' suffixes are pipeline content/cache identities, not file SHA-256.

The saved ledger's `meta.source_meta.story_input.fields` explicitly requires:

- Jeffrey is an adult. His living mother is physically present and eating with him.
- Both sit across from each other at the same Los Angeles dinner table with plates
  of food, throughout one continuous present-day scene.
- Bay Area childhood is spoken memory; the physical setting stays at this table.
- The girlfriend is a separate person and does not attend.

These are supplied story facts, not an auditor's preferred composition.

## Complete source reached the correction, but the correction did not apply it

Inspect the two matching rows in `stills/stills_manifest.json`, their
`source_rewrite` objects, and the saved ledger
`audio/signal_lost_the_la_table_20260911_022847_ledger.json`.

For both stills:

- `source_scope=whole`; source digest
  `1c88360b742fa193dba710619d5e90a083af3e7e99412505d7a78107ff36cb05`.
- The receipt includes the current target, the other person as a candidate
  companion with `speaks_in_scene=true`, the correct shared scene/shot join,
  all eight dramatic dialogue lines, and the full working treatment.
- The treatment identifies Jeffrey as `age_band=30s` and Mother as `age_band=50s`.
  Its first turn describes them sitting across from each other and enjoying a
  warm meal. These ages are model-authored supporting context; raw source's adult
  Jeffrey and living mother remain the authority.
- One successful model attempt returned the original candidate prompt verbatim.
  `input_sha256=output_sha256`, `applied=false`, `status=unchanged`,
  `qualified=false`, `attempt_limit=2`. No second attempt was triggered because
  the response was structurally usable.
- The configured and binding model are `Qwen/Qwen3.5-4B`;
  `executed_model_id=null`. This audit does not turn the binding field into
  provider response identity evidence.

For b1, context hash is
`98d5fac659c6e04d0028e0e360874ad41653c46a3129ae280ede7d180bc055f2`;
for b2 it is
`9d1ccb344d1e0fe7ffb7732ee84bf937f8288927cdd38a1618e845665e0563ac`.

The retained b1 subject describes an adult man who speaks warmly of his mother.
The retained b2 subject describes a woman responding to her son's stories.
Both then say `dining room, kitchen table` and request a medium shot of
`the character` with a visible face. Neither actually instructs the renderer to
show the adult son and older mother together, plates of food, or eating. Both
also retain the generated phrase `a family reunited after years apart`, which
the source did not supply.

The complete final strings remain in each receipt's `final_prompt`. Their hashes:

| Beat | Final prompt hash |
| --- | --- |
| `shot_001_b1` | `0eab20bb1bafe92c58841e80a7c90f036c55da8618365a04b643249552a0c0d5` |
| `shot_001_b2` | `202f49b8b46ae0aec1de85a5b46cb4ae0251426c653fb2d929b6131531a83717` |

## Code boundary and last-mile conditioning

References below are source lines at the inspected HEAD.

1. `nodes/_otr_story_brief_helpers.py:533-590`, `compose_still_prompt`, supplies
   the candidate used by the source scene operation. For `scene_character`,
   subject is only that character's portrait/appearance/description, followed
   by the first two setting terms, singular-character framing, era, and style.
   It does not compose the scene's participants or shared visible action.

2. `nodes/otr_meta_brief_image_prompt.py:1597-1642`,
   `_build_char_scene_request`, asks for `this character` / `CHARACTER THEMSELVES`
   and tells the model to translate the beat into visible action. It supplies
   `they_are_saying` with no explicit instruction distinguishing a present
   action from a recollection. Its snippets are shortened, but the complete
   raw source and ordered dialogue are separately present in authoring context;
   this failure was not loss of the source at a snippet boundary.

3. `nodes/otr_meta_brief_image_prompt.py:1651-1709`, `_scene_source_context`,
   correctly joins the scene and includes the complete treatment. The direct
   target/companion projection has name and appearance, but no `age_band`;
   ages remain nested in the treatment. This is a salience opportunity, not
   evidence that the correction model was deprived of age information.

4. `nodes/otr_meta_brief_image_prompt.py:1712-1787`,
   `_rewrite_char_scene_from_source`, starts with that solitary-character
   candidate and passes it through the existing combined source correction.
   Its instruction already says to include required companions, but retains
   the target-face framing and the singular visual request. The response
   contract (`:1646-1648`) is only a nonempty string; the local validator
   (`:1729-1731`) checks that same structural necessity. This call accepted
   the unchanged candidate. Source schema validity is expressly not semantic
   proof (`nodes/_otr_story_source.py:129-134`).

5. The source branch returns directly from `_compose_char_scene_prompt`
   (`:1802-1807`); the ordinary legacy visual composer does not run afterward.
   `:2501-2545` carries the actual prompt and receipt to the image payload.
   Finishing deliberately does not re-prepend appearance after a correction,
   which avoids reinstating a fact the source model corrected.

6. Dispatcher normalization preserved that base (`base_matches_rewrite=true`).
   Its sole text change on these rows was the additive prefix
   `warm-toned oil. `. Safety and banana changes were false, with zero banana
   substitutions. `dispatch_disposition=transformed_after_source_operation`
   therefore does not mean a good scene description was removed downstream.
   See `nodes/otr_image_gen_dispatcher.py:1062`, `:1346-1358`, `:1746-1789`.

7. Z-Image Turbo v2 received the complete final positive string.
   `nodes/_otr_image_engines/z_image_turbo.py:348` forwards request text;
   `:426-443` binds that text to positive conditioning and the sampler.
   The adapter does not author participants or truncate the prompt here.
   Its production reference-image capability is deliberately false (`:285-293`)
   after the separate live corruption A/B, and `:364` keeps the base txt2img
   route. The rows have `portrait_anchor_mode=seed`, an identity seed basis,
   and empty `derived_from_portrait_hash`; a seed cannot communicate adult age
   or another person's physical presence.

8. The live server log proves fresh draws, 1472x832, eight steps, cfg 1.00,
   shift 3.00, euler/normal. b1 seed is 2953401791 and b2 seed is 2955717723
   (`tmp/my_story_5080_eos_server.log:823-876`). At this cfg the negative is
   inert; adding a child-negation clause there would not repair this route.
   `:1189` and `:1192` prove `still_pan` consumed these exact scene PNGs.
   It did not receive a portrait or generate a corrective scene afterward.

## Smallest existing-owner repair boundary

The demonstrated failure is in scene-prompt authoring: a source-rich correction
accepted a candidate that did not express required visible facts. It is not a
missing source wire, an image cache hit, an overwritten correction, or a portrait
reference accidentally substituted for the scene. The exact stochastic reason
the image model drew two boys cannot be proved from a prompt alone.

Use the existing `_rewrite_char_scene_from_source` operation and scene request
builder. Make its visual contract explicitly author the **current physical scene**:
resolve visible participants and their current ages from source/scene context;
keep the active speaker as the focus without treating that speaker as the entire
cast; describe required shared actions and props; keep spoken memories out of
the visible time/place unless the source actually calls for a flashback. Require
those facts in the returned positive prompt, rather than leaving them only in
the correction request. Preserve compatible style after the scene facts.

The direct target/companion projection can expose already-known ages from the
uniquely matched working-treatment cast, without inventing ages or overriding
raw source. That is supporting work at the same owner, not a new model pass.
The existing source operation must treat an incomplete solitary-speaker template
as a draft to compose for the moment, not merely check for explicit contradictions.

Do not force every scene speaker into every shot: source may legitimately make a
person absent, on a phone, or present only in a memory. Do not change the neutral
portrait owner, image identity seed, disabled Z-Image reference path, sampling,
workflow wiring, or add a presence/age/food keyword gate. Keep the existing
bounded correction/repair budget and honest unresolved/unchanged receipts.

## Meaningful regression and qualification followup

Extend `tests/test_my_story_visual_source.py`, which already exercises this owner:

- `test_complete_source_scene_dialogue_and_target_reach_applied_correction`
  currently checks source transport with a canned good reply. Add the real
  adult/older-mother present-day dinner and spoken-childhood context, checking
  the request's scene-versus-focus contract and lossless identity/age projection.
- Include a separate absent/phone/memory-only companion fixture to preserve the
  existing rule against forcing every act speaker into a frame. No presence
  classifier or semantic keyword rejection should appear in production.
- Extend `test_corrected_narrative_appearance_is_not_prepended_back_into_prompt`
  and `test_fresh_cache_and_durable_manifest_use_current_receipt` so a corrected
  multi-person adult dinner prompt survives finishing, dispatcher normalization,
  actual engine request, hashes, and durable manifest without old singular
  appearance being restored. Existing identity/reference invariants remain.
- Preserve tests for unchanged valid results, bounded malformed repair, and
  provider/OOM/cancellation propagation. An unchanged result remains possible;
  it must never be automatically called source-qualified.

Mock tests can prove the request contract and preservation, not that Qwen or
Z-Image obeys it. A fresh full canonical qualification must inspect the actual
final prompts and pixels for the supplied shared adult dinner before claiming
this production visual failure fixed.
