# S3: Radio Play Studio proposals, for the owner of the design anchor

To: whoever owns `planning/studio-ui/DESIGN_ANCHOR.md` (v2, 2026-10-23) and `contest/radio-play-studio/`.
From: the CreatorOS-concepts work (`planning/creatoros-concepts/PLAN.md`, sprint S3). Written 2026-10-24.

These are proposals to fold into the anchor, not code. Nothing under `contest/radio-play-studio/`
was touched. Each one is checked against the files as of `main` today. The anchor's rules govern
all five: nine languages, contrast tokens, keyboard access, and "the lamp is the only thing on
the page that moves" (`DESIGN_ANCHOR.md:17`). PLAN scope rule 8 applies as well: none of the five
adds a stop, a confirm or a wait for a person.

| Row | Proposal | Grounding (read 2026-10-24) | Cost |
|---|---|---|---|
| 3.1 | **The tab title shows the run.** While a stage runs: `(42%) Radio Play Studio`. When a stage finishes with the tab hidden: `(1 ready) Radio Play Studio` until the tab is looked at, then the plain title. Every string goes through the i18n files. | `static/index.html` sets no `document.title` and has no `visibilitychange` handler. | Small: one title setter fed by the existing bar value, plus one listener. Text only, so the lamp rule holds. |
| 3.2 | **The running step in plain words** in the status line ("voicing the cast", "filming shot 12"), taken from the label table the headless runner now uses. | `runner.py:153` `stage_note` already carries `node` for progress frames. The table is `config/node_phase_labels.json` (23 OTR classes plus core patterns), read by `scripts/otr_api.py` `label_for_class`. | Medium. The labels are English, so the Studio needs them as i18n keys: either key the JSON's values or add a key column. Deciding where the shared table lives is part of folding this in. |
| 3.3 | **Say when a step runs long.** When the running node goes past its calibrated time and no pace can be read, say "this step is taking longer than last time" instead of a minutes-left figure that stops moving. | `runner.py:1022` `seconds_left`: when the node reports a fraction (past `PACE_MIN_FRACTION` and `PACE_MIN_SECONDS`), the time left comes from its pace. Otherwise it is `max(0, history - elapsed)`, which reaches 0 on an overrun without saying why. The proposal covers only that second branch. | Small: one flag on the estimate (`overrun = elapsed > history`) and one line of text. |
| 3.4 | **Before and after on a repainted card.** On a card with a new still take, flip or slide between the kept still and the new one, so keep vs new is a choice you make by looking. Only inside the Scene Producer's existing edit step; it adds no stop of its own. | The anchor already plans "New take" badges. Takes live in `draft.takes` (`static/index.html:1576`, `:1812`). **Open:** the previous take's file has to still exist. Confirm in the Scene Producer store that a re-take keeps the old still before building this. | Medium. If the old file is not kept, this row needs a store change first and is no longer just a page change. |
| 3.5 | **Phone ping when a stage finishes**, reusing the headless runner's `notify(title, body)`. | `scripts/otr_notify.py`: stdlib only, off unless `OTR_NOTIFY_URL` is set, never prints the topic, never follows a redirect, 15 s deadline. The Studio cannot import from `scripts/` today. Moving the module to a place both can import is part of folding this in. | Small once it is importable. Off by default; says on or off at start, as the runner does. |

**Not proposed:** a live preview on the cards. The runner can now keep the newest preview frame
(`--preview-file`), but on the page it would be a second moving thing, which the lamp concept rules
out. The anchor's owner can revisit it.

**Suggested order:** 3.1 and 3.3 first (small, text only), then 3.5 (needs the import home), then
3.2 (needs the i18n keys), then 3.4 (needs the store check).
