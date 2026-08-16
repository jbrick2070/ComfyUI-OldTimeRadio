# scifi_news_pro fixtures

`legacy_reference_ledger.json` -- captured from the scifi_fable2 S1a
legacy `science_news` 30-word live smoke (episode
`signal_lost_etnas_secret_20260710_072427`, RESULT SUCCESS, published to
obs; tail extraction commit `948c5a0a`), then scrubbed: machine paths
anonymized (`<OTR_OUTPUT>` / `<MODELS>` / `<HOME>`), bulky source-article
text truncated. Every key and row is kept.

Contract role (architecture doc 2026-07-10-scifi-fable2, s13 S1b +
r4/CUT2): the ROW-ROLE-ORDERING and TAIL-OUTPUT-CONTRACT reference for
`test_scifi_news_pro_assembly.py` -- NEVER a byte-match target. Line role order
in this capture: announcer, character x3, announcer. Note: the capture
is a POST-RUN ledger, so cast/line rows carry downstream render-stage
stamps (`voice_engine`, `commercial_clean`, `voice_ref_id`,
`start_s_space`) that the writer-stage contract excludes
(`_POST_WRITER_*_STAMPS` in the test).

`golden_s1b_assembly.json` -- the S1b golden happy-path fixture: the
deterministic output of `F2._assemble` over the hand-authored inputs in
`test_scifi_news_pro_assembly.py` (no LLM anywhere). Pins the lane's OWN
five-hierarchy assembly contract (sentinel rows, cue ids, merged runs,
boundaries, beat/line 1:1). Regenerate DELIBERATELY -- only when the
assembly contract changes by design, never to paper over a drift -- by
re-running the `_assembled()` helper and dumping
`_hierarchies(led)` (see the module docstring).
