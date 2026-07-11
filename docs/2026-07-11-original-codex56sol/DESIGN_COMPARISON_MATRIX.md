# Original Codex 56SOL -- Post-Lock Design Comparison

**Compared:** 2026-07-11, after the independent design lock  
**Design fingerprint:** `DESIGN_FINGERPRINT_V1.md`  
**Fingerprint SHA-256:**
`B2A0C800583868FC85063FA595F0FBC973C88235976AA38A9289ACF5A9684BF9`  
**Live code:** `d39b134c41a5cbca97c578b583dd2f7075c80f79`

Comparison inputs:

- `nodes/story_packs/banks.json` --
  `ABEE40A60A9866417EE64C71ABE19ED386E3C3FADCCA2E5E2C6E8AF8C881B230`
- `nodes/story_packs/pipelines.json` --
  `7363A12D80B6BFB2BF85B240D220DBC3E17D17B061953A9376EC0B78625E59B9`
- `nodes/OTR_LedgerScriptWriter.py` --
  `9D7091AE4BEDC665843BD018C7BD3FF093DEF6A109D1F294CFECCFF02BCA9202`

The comparison normalizes cosmetic IDs and asks whether a material design
choice is the same. `partial` means the lanes share a generic activity such as
"write a script" or "audit a draft" but differ in why it exists, its authority,
its artifact, or its outgoing edge. Partial similarity is not counted as a
dimension match.

## Six-dimension matrix

| Registered lane | Source strategy | Pass DAG | Role / authority graph | Artifact handoffs | Retry / audit topology | Ledger-write strategy | Material matches |
|---|---|---|---|---|---|---|---:|
| `science_news`, `media_archive`, `public_domain_story`, `shakespeare` / `legacy_many_pass` | different: external source or adaptation vs no-source synthetic fiction | different: shared source interpretation and legacy composition tail | different: source interpreter plus shared outline/cast/composer authorities | different: compatibility briefs and legacy outline/canon artifacts | different: legacy writer/reviewer cascade | different: `legacy_full` vs lane-owned content | 0 |
| `original_radio` / `original_multi_pass` | **match:** no-source original fiction with an entropy-backed packaged seed deck | different: concept/select/compatibility-brief/whole-script QA feeds the legacy writer body | different: concept editor and compatibility-brief roles; shared writer owns lines | different: concept slate and interpreter-shaped briefs, not an audible truth map and closed performance manifest | different: post-composition QA with shared line composition, not clue-order proof plus lane-authored script repair | different: declares `line_composer_system`, so `legacy_full` | 1 |
| `scifi_fable2` / `fable2_multipass` | different: science RSS evidence vs no-source fiction | partial: both have divergent invention and a later draft, but Fable2 is dossier/pitch/treatment/markup; this lane is possibility/causal-knot/fair-play/score/manifest/performance | different: factual dossier and treatment authorities vs clue-causality and blind-listener authorities | partial: both eventually hold a full play, but the typed truth map, clue/reveal proof, and exact performance manifest are unique | different: factual critic/revision and markup ladders vs deterministic clue-order proof and evidence-linked creative repair | **match:** `content_owned_readonly` | 1 |
| `scifi_codex` / `scifi_codex_circuit` | different: indexed science payload and factual coda vs no-source fiction and no factual coda | partial: both separate structure from final lines; this lane has no fact index, dramatic-question stage, factual coda, or mandatory full-retake circuit, and adds caller-thread weaving plus listener-information replay | different: source-evidence auditor and pressure-cast roles vs causal architect, fair-play examiner, and desk-listener model | partial: a generic score/script/audit family is shared; `AudibleTruthMap`, typed clue order, benign resolution links, and redacted listener packet are not | different: Codex always sends P5 to a full retake and may loop closing rewrites; this lane repairs only corroborated contract defects while listener taste remains notes | **match:** `content_owned_readonly` | 1 |
| `scifi_gemini` / `scifi_gemini_multipass` | different: science RSS vs no-source fiction | partial: generic pitch/draft/critique/rewrite verbs only; no truth-map or fair-play proof exists in Gemini | different: source fact extraction and per-scene critique vs causal-clue authorities | different: fact index, outline, and scene drafts vs caller threads, clues, reveal links, and one closed performance artifact | different: per-scene rewrite vs whole-artifact repair after graph and listener-state evidence | **match:** `content_owned_readonly` | 1 |
| `scifi_sonnet` / `sonnet_archive_multipass` | different: science RSS vs no-source fiction | different: archive-session frame and fixed diegetic readings vs community call desk and converging caller threads | different: Registrar/Reliquarian/Warden authority play is not reused | different: dossier/session/readings/attestation vs truth map/score/performance/receipt | different: diegetic challenge/reopen loop vs non-diegetic contract repair and warning-only ear test | **match:** `content_owned_readonly` | 1 |
| `custom_source_bank` / `simple_4_prompt_experimental` | unresolved and non-runnable vs explicit no-source initializer | different: four generic prompts vs the target-derived causal-audio DAG | different: no declared listener/authority system vs explicit separated authorities | different: generic story/ledger vs typed clue and performance artifacts | different: schema cleanup/audit vs bounded per-artifact ladders and corroborated creative repair | partial: both target the shared ledger, but the experimental lane has no runnable ownership contract | 0 |

## Result

- No registered lane has an exact pass-DAG match.
- No registered lane matches four or more fingerprint dimensions.
- The only full discretionary match is the entropy-backed no-source strategy with
  `original_radio`; its dramatic form, roles, artifacts, retry topology, and ledger
  ownership are materially different.
- All runnable custom runners necessarily share `content_owned_readonly`; that is a
  shared integration contract, not a creative design choice.
- Generic activities such as structured generation, script drafting, validation,
  and bounded repair are shared infrastructure. No role names, prompt language,
  story frame, implementation block, typed artifact definition, or validator logic
  is to be copied from another lane.

Verdict: **PASS for independent design**, subject to a second hash-backed comparison
against the implementation at preflight. If implementation drifts into an existing
lane's discretionary structure, it must be redesigned back toward the locked
listener promise rather than justified after the fact.
