# Pass 07 plan -- scifi_news quality-context starvation

Reviewer lane: exact `Gemini 3.5 Flash (High)` through Antigravity, review-only.
Driver/coder/judge: Sol only.
Audit workspace: clean detached throwaway worktree at commit
`a9f51cab4ab9dbfd149fec08d660d694ff60788e`.

Scope: the live `scifi_news` P7 failure plus sibling implications for
`media_archive`, `original`, `public_domain`, `shakespeare`, and
`scifi_news_pro`. Review completion/liveness, ledger ownership, spoken-hygiene
bypasses, stale hashes/seals, row-local failure behavior, readiness mutation,
and image/video/audio/OBS consumers.

Proposed repair under review:

- Keep P5 as the only complete script artifact call.
- Convert P7/P9 to a strict line-text patch over a closed target-ID set.
- Include all voiced rows as read-only context and compact source facts needed
  by targets/findings; omit the complete score/artifact/schema envelope.
- Merge only `line.text`, require exact coverage/change, and run the complete
  post-validator.
- Try creative, then a colder technical slot; rejudge every successful merge.
  If both fail, retain the best valid script and stop without rejudging
  unchanged input.
- Add an opt-in full-output capacity marker to every transport so an impossible
  patch makes no generation/network call. Unmarked calls retain normal clamp
  behavior.
- Build authorship receipts, hashes, ledger rows, readiness, media assets, and
  OBS pointers only after the final accepted/floored script.

No canonical workflow node, widget, socket, or link change was proposed.
