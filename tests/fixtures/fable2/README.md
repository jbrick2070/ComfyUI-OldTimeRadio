# fable2 fixtures

`legacy_reference_ledger.json` -- captured from the scifi_fable2 S1a
legacy `science_news` 30-word live smoke (episode
`signal_lost_etnas_secret_20260710_072427`, RESULT SUCCESS, published to
obs; tail extraction commit `948c5a0a`), then scrubbed: machine paths
anonymized (`<OTR_OUTPUT>` / `<MODELS>` / `<HOME>`), bulky source-article
text truncated. Every key and row is kept.

Contract role (architecture doc 2026-07-10-scifi-fable2, s13 S1b +
r4/CUT2): the ROW-ROLE-ORDERING and TAIL-OUTPUT-CONTRACT reference for
`test_fable2_assembly.py` -- NEVER a byte-match target. Line role order
in this capture: announcer, character x3, announcer.
