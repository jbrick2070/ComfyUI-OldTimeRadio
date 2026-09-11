# Independent adaptive-reading coverage review

Read-only reviewer: Einstein, /root/my_story_audit. Windows repo and R2 final.
Root remains sole production editor. These are review findings, not live bugs.

1. Separate source and candidate coverage does not prove all comparisons were
   made. Reviewing (source A, candidate A) and (source B, candidate B) misses
   a contradiction between source A and candidate B while both axes appear
   complete. Record reviewed source-span/candidate-part pairs. A whole-candidate
   qualification requires full pair coverage for its declared scope; skipped
   pairs remain unresolved. Summary-derived relevance is not proof that a pair
   needs no review. Do not force a hidden quadratic spend to obtain a PASS;
   report the scope actually checked when complete review is unavailable.
2. Evidence must be grounded in the particular call's delivered raw spans and
   candidate projection, not merely found somewhere in the full saved source.
   Out-of-view evidence must request its real context or remain unresolved.

Regression examples: a living-mother assertion in source A versus a death claim
in candidate B; a correct source quotation that was never in the review packet.
Neither may produce a complete qualified receipt or authorize an unsupported
edit. Byte coverage is delivery evidence, not semantic proof.

Strictly smaller nonempty splits, bounded non-progress repair and reducing merge
groups already handle progress. No additional story-length, minimum-character
window, recursion-depth, or extraction-ontology rule is justified.
