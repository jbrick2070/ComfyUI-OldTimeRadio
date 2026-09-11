# My Story

Choose `my_story` in the writer's Source bank control. Enter your idea in
`custom_premise` (Story input), or fill Character names and notes, Plot ideas,
or Setting. Story by is optional; a blank byline credits a listener.

The typed fields supply the story. No RSS fetch or random premise draw runs
on this bank. The established model-call/retry, voice, ledger and downstream
production machinery is reused. The My Story front interprets your request,
plans it, writes one act at a time and adds the announcer's frame.

Select one to six acts and request one to ten speaking characters, then queue
the normal graph. The act count is binding. Character count is flexible, as on
adaptation banks: the supplied story guides the cast, and the ledger records
requested and actual counts separately. The announcer is excluded. The model
repairs a mismatched act plan using its complete draft, preserving the people,
events and ending. Each character needs a distinct available voice. The house
cameo roll does not apply. There is no word-count or duration rejection.

Your input is saved before generation under the shared state directory:
`otr/episodes/_shared/state/story_drafts/<digest>/input.json`. Identical input
and controls reuse that draft; a fresh run generates a new episode. Cancellation
or a later failure leaves the admitted input in place. Linked fields are saved
when their evaluated values reach the writer.

Generation uses the existing provider-capacity path, including repairs. Input
and failed parsed drafts are not silently truncated; actual provider capacity,
storage failures, cancellation and out-of-memory outcomes remain explicit.
Optional descriptions, monologue acts and missing frame text do not reject the
story. The existing ledger cleanup and freeze still verify a usable result.

Clear `source_ref`, `replay_from`, and any source-snapshot manifest when using
My Story. Other banks do not read the four dedicated Story fields; clear those
fields before selecting another bank or an automatic bank roll.

The graph's delivery wire requires an actual published file in `otr/obs/` and
its recorded path in the matching episode ledger. An archived render alone is
not successful My Story delivery.

Qualification status: the initial local one-act trial stopped at a treatment
count mismatch before a ledger could freeze. The revised repair/count behavior
is under regression and component verification; no new live PASS is claimed
until its receipt exists. Native App, browser playback/history and full-media
publication qualification remain pending in GO_FORWARD_PLAN.md.
