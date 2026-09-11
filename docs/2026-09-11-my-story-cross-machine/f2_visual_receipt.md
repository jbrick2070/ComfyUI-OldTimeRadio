# F2 visual source correction component

Integrated on top of F1 bdeca482. Component checks: 489 passed, one skipped
in 13.97 seconds, including 26 new visual-source regression cases. Combined
campaign full regression, Bug Bible, final Sonnet QA and live qualification
remain outstanding. This is not a semantic or image-pixel qualification.

My Story character-scene prompts use the existing technical model owner and
the full raw source. Structural beat/shot/scene joins supply ordered scene
dialogue, the target appearance, and possible companions with their scene
participation. The treatment remains advisory. Neutral portraits, announcer
and music templates retain their existing paths.

The existing scene-author budget now produces the corrected prompt: at most
two calls including syntax/schema repairs, further capped by max_reseed + 1.
No report-only checker or recursive correction is added. An unusable optional
correction keeps the fully composed initial prompt and an unresolved receipt.
Provider, OOM and cancellation errors propagate with their original type.

The model returns the finished scene description. Re-prepending appearance
after correction could restore a contradiction, so only the existing no-text
instruction follows it. The dispatcher records subsequent safety, whitespace,
style and banana transformations and the exact final prompt hash. Application
does not assert fidelity. Configured and normalized binding identities are
distinct from executed model identity, which is unknown here.

Source/context identity participates in the actual image cache key, including
a fresh structural join against the current frozen ledger. Fresh and cached
rows receive the current receipt; stale historical receipts cannot silently
survive. The existing still manifest and durable stamp carry returned receipts.
An operation that raises before returning has no persisted image row: its
completed primitive failure receipt is emitted through the existing server
logger. Logging failures cannot replace the original provider/cancel error.
Frozen-ledger ownership is unchanged.

No node inputs, widgets or links changed. Canonical MetaBrief node 89 and image
dispatcher 91 both consume ShotLock node 90's same frozen script_json, via
links 255 and 256; link 258 carries the image prompt payload between them.

Independent read-only Einstein review found two concrete omissions in this
initial integrated snapshot: resolved story setting was absent from context
identity, and derived jump-segment stills lost the scene's source identity.
Both were corrected by the sole coder before final qualification. Resolved
setting is in the shared scene projection, and the real jump merge preserves
explicit scene scope for descendant context hashing. The combined B/F2 follow-up
set passed 226 tests; the final campaign focused set passed 287 with one live
Metal skip. The hashes below identify the initial component; final reviewed
hashes are in kibitz-runs/2026-09-11-my-story-finishing-qa/r4/snapshot.json.

Integrated SHA256:

- nodes/_otr_story_source.py: fef71c1f2f7ce79fb20fd9f40b30e78a759dd9842ae8b943cc54cef60695927d
- nodes/otr_meta_brief_image_prompt.py: 6861c0dbfc9fb4636adc6d86901be1ef3db9f09db7359d5c5b758f1192285c7b
- nodes/otr_image_gen_dispatcher.py: a1ff7251b41eae6f59c250c8d0a21713d8a71f7fcb0eed901be9bdc4a077e36b
- tests/test_my_story_visual_source.py: 0790d2080a0bca4cd6220e8573986a58ce964e6cdbcad68ba018100c2a0b8611
