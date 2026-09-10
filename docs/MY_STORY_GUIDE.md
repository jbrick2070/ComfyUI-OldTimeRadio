# My Story

Choose `my_story` in the writer's Source bank control. Enter your idea in
`custom_premise` (Story input), or fill Character names and notes, Plot ideas,
or Setting. Story by is optional; a blank byline credits a listener.

The typed fields supply the story. No RSS fetch or random premise draw runs
on this bank. The established model-call/retry, voice, ledger and downstream
production machinery is reused. The My Story front interprets your request,
plans it, writes one act at a time and adds the announcer's frame.

Pick one to six acts and a visual style, then queue the normal graph. Named
speaking characters take precedence over the numeric character request. If
you specify only those people, the cast stays exclusive. Each character needs
a distinct available voice. The house cameo roll does not apply.

Your input is saved before generation under the shared state directory:
`otr/episodes/_shared/state/story_drafts/<digest>/input.json`. Identical input
and controls reuse that draft; a fresh run generates a new episode. Cancellation
or a later failure leaves the admitted input in place. Linked fields are saved
when their evaluated values reach the writer.

The model's context must fit the input, instructions and response. If it cannot,
the error identifies the longest input field and asks you to shorten it or
choose a model with more context. Input is not silently truncated.

Clear `source_ref`, `replay_from`, and any source-snapshot manifest when using
My Story. Other banks do not read the four dedicated Story fields; clear those
fields before selecting another bank or an automatic bank roll.

The graph's delivery wire requires an actual published file in `otr/obs/` and
its recorded path in the matching episode ledger. An archived render alone is
not successful My Story delivery.

Qualification status: offline creator-path checks are in Sprint 4. The native
App view, browser playback/history and one-, three-, and six-act live model
publication receipts belong to Sprint 5 and are not yet claimed here.
