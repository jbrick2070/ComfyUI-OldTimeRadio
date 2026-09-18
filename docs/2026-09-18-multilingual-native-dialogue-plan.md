# Multilingual native dialogue correction plan

## Decision to make

The new episode-language row reaches the announcer, voices, captions, fonts and
credits, but character composition can remain English. The fix must decide
whether to translate captions after the fact or make the character-line author
write in the selected language at the actual generation seam.

## Live evidence

Portuguese episode `signal_lost_o_mapa_proibido_20260918_093115`:

- `meta.episode_language = "pt"`;
- announcer rows are Portuguese;
- character rows `b002` through `b005` are English;
- each character row's `text_for_tts` equals its English `text`;
- captions display `text`, so the English captions are accurate;
- Kokoro performs those English strings with Portuguese voices.

Mandarin episode
`signal_lost_项链之争_contest_for_the_locket_20260918_101411` has the same
shape: native announcer framing and English character rows. User screenshots
show the two caption languages in the same finished episode.

The announcer path calls `_otr_line_composer._announcer_system`, which prefixes
the language row's `writer_instruction` onto the chat system message. The
character path places that instruction only inside `LineRequest.canon_header`,
under `EPISODE CONTEXT` in the user message. The character system message is
the English pack seam. Mistral-Nemo follows that higher-priority English seam
and returns English dialogue.

## Non-negotiable behavior

- One `episode_language` switch. No caption-language widget.
- Captions remain the exact canonical spoken text. They do not claim a
  translation the voice did not perform.
- Author natively; never translate an English draft.
- English and `Off` prompts remain byte-identical.
- `ANNOUNCER` remains the internal identity key.
- No profanity/violence filtering and no prose-quality gate.
- No hard word/count/CPS rejection.
- Shakespeare/Public Domain keep their exclusion in THIS change. Operator
  ruling 2026-09-18: every lane is eligible, and for those two the verbatim
  passage is translated -- a separate row in GO_FORWARD.
- A helper is wired at every production authoring seam in the same change.
- The canonical workflow needs no widget or wiring change.

## Dispatched lanes (found in review, 2026-09-18)

The inline writer is not the only author. `_otr_lane_specs.LANE_SPECS`
dispatches `my_story` and `scifi_news_pro` to their own runner modules,
which build their own prompts and never read `canon_header`. Both round-1
reviewers found them still authoring English. The same row-owned instruction
now leads every spoken or displayed pass there:

- My Story: treatment (title, cast), each act, the announcer frame, and the
  source-rewrite fidelity pass (so a correction against the English typed
  source cannot pull a native line back). Interpretation stays as it was.
- SciFi News Pro: pitch, treatment (title), whole-play script, closing news
  read, spoken cast labels. Dossier extraction from the English source and
  voice casting stay as they were -- the news is not translated, the new
  story is written natively.

## Rejected approaches

### Caption-only translation

Reject. It would paint words different from the words stored and spoken, add a
new ledger owner, and conceal the actual English character dialogue.

### Deterministic Python translation

Reject. Python has no authority to author dialogue and cannot translate the
seven language rows without another model.

### A new translate-every-line model pass

Reject for the first correction. It adds cost and a new authorship stage,
rewrites accepted dialogue, needs language detection for four closely related
Latin-script rows, and violates the native-authoring design.

### A hard language detector

Reject as the first fix. Script-range checks can identify Hindi/CJK, but they
cannot reliably distinguish English, Spanish, Portuguese, Italian and French.
A weak detector would either pass English leakage or reject valid names and
short lines.

## Proposed root fix

Use the existing row-owned `writer_instruction`; add no registry field and no
new pass.

1. Add an optional `language_instruction` to
   `_otr_line_composer.LineRequest`.
2. In `compose_line_draft`, prefix that instruction onto the resolved
   character **system** prompt before dialogue policy is appended. Empty means
   byte-identical English/Off behavior.
3. Populate it from `_native_authoring_instruction(meta)` in the writer's
   `_build_line_request_for_beat`.
4. Populate it in `_otr_cast_coverage_repair` from the ledger language so a
   late zero-coverage line cannot reintroduce English.
5. Prefix the same instruction onto the grouped-exchange system prompt before
   `run_exchange_prepass`; the default widget is off, but the alternate
   production path must not remain a trapdoor.
6. Keep the existing copy in `canon_header`. The system copy establishes
   response language; the user copy keeps the episode context self-describing.
7. Do not change caption code. The corrected canonical line naturally becomes
   both TTS delivery and caption text.

## Repair-pass audit

Before code, enumerate every post-composition call that can write or replace a
spoken row:

- per-line composer;
- grouped exchange;
- cast-coverage repair;
- ledger-clean F2 attribution repair;
- ledger-clean stage-business repair.

Judges and act summaries may answer in English because they are internal
receipts. Any call that returns replacement dialogue must receive the same
language instruction in its system message. If a repair path cannot be wired
without widening this change, it must retain the original native line rather
than silently author English.

## Tests

- Non-English `compose_line_draft` sends a system message beginning with the
  row instruction.
- English/Off character messages are byte-identical to the pre-change fixture.
- Writer builds every character `LineRequest` with the ledger-owned
  instruction.
- Grouped exchange and cast-coverage repair receive it.
- Every ledger-clean prompt that can return replacement speech receives it;
  judge-only prompts need not.
- Captions still use `line.text`; no translation/display field appears.
- Focused multilingual, line-composer, clean-stage and workflow tests.
- Full regression compared with same-HEAD baseline, then Bug Bible regression.

## Live proof

After review and push:

1. Reset the GPU and boot the canonical server.
2. Run a Portuguese one-act with a randomly chosen eligible source bank.
3. Require all character and announcer ledger rows to be Portuguese, then
   require `obs_publish OK`.
4. Run Japanese and Mandarin one-acts for CJK dependency/font proof.
5. Run the remaining admitted languages with randomized eligible banks and
   audit source-bank music diversity.
6. Finish with an English one-act and English prompt-regression tests.

The first live failure stops qualification and becomes evidence; it does not
get hidden by a translated caption.
