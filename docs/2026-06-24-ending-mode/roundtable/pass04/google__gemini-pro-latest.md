<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The sequencing circular dependency is resolved, but a data-key rename breaks an existing helper, and the flag-routing to the outline assembler is ambiguous.

MUST-FIX BEFORE BUILD:
1. [Section A] Data-key rename breaks existing helper. The plan says "keep prose as `ending_flavor`", but `render_style_grammar()` in `_otr_style_catalog.py` explicitly reads `s['ending_mode']`. If the key is renamed, the helper crashes or drops data, breaking the byte-identical baseline. 
   **Fix:** Do not rename the key. Keep the prose under the existing `ending_mode` key. Just add the new `ending_tag` and `domain` keys to the 100 entries.
2. [Section E] Flag routing contradiction. Section E says "The flag must reach `_assemble_outline` (via the request or a module-level read)", but the Sequencing section strictly forbids adding fields to `OutlineRequest`. 
   **Fix:** Remove the ambiguity. Explicitly specify reading the environment variable (`os.environ.get("OTR_ENABLE_STYLE_GRAMMAR") == "1"`) directly inside `_assemble_outline` to gate the announcer intent string. This honors the strict `OutlineRequest` freeze.

SHOULD-FIX:
1. [Section E] Stray text. "Metric language: 'last voiced CHARACTER beat'" appears at the end of Section E (Announcer-outro gating). The announcer is not a character beat. 
   **Fix:** Remove this sentence from Section E. Telemetry for the final character beat is already correctly covered in Section F.

OPTIONAL / NICE-TO-HAVE:
- In Section C, when looking up the `final_char_beat_id` from `roles_by_beat`, explicitly state to find the key where the value equals `BEAT_ROLE_IRREVERSIBLE_CHOICE` (imported from `_otr_story_quality_l12`).

CUT THESE:
1. [Section C] "Optional (flag ON): override meta.style with the selected slug...". Cut this. Mutating `meta.style` mid-flight risks breaking downstream visualizer/LTX assumptions that expect the original picker value. Keep the scope strictly constrained to outline/dialogue generation for this bundle.

VERIFY-AT-BUILD:
- Verify that `_assemble_outline` in `_otr_outline.py` successfully falls back to the exact existing string `("Close on a concrete final image showing what changed...")` when the env flag is missing/off, ensuring the C7 byte-identical baseline holds.
- Verify that `validate_catalog()` asserts every entry has `ending_tag`, `domain`, and `ending_mode` (prose), and that `ENDING_TAG_BY_SLUG` maps all 100 slugs.