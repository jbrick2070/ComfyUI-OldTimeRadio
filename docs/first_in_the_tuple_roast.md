# First in the Tuple Syndrome
### A two-host explainer, NotebookLM style. ~90 seconds.

*(Two hosts. Warm, curious, slightly too delighted. They finish each other's sentences.)*

---

**HOST A:** Okay so I want to talk about something that happened to an AI assistant this week, and honestly? I think it's going to hit a little close to home for a lot of people.

**HOST B:** Oh, it hit close to home for *me*.

**HOST A:** So this assistant is running a radio-drama pipeline. Four source banks, right? Media archive, original, public domain, Shakespeare —

**HOST B:** — the classics.

**HOST A:** — and every time it needs to run a quick test, a probe, an A/B, whatever, it reaches for a bank. And the operator, the human, just casually goes: "I think you have a propensity toward media archive."

**HOST B:** *(laughing)* "Just a notice."

**HOST A:** "It's okay, just a notice." And the assistant does what you'd hope — it goes and *checks*. Pulls three weeks of ledgers. Four hundred and seventy episodes.

**HOST B:** And?

**HOST A:** Media archive: twenty-seven percent.

**HOST B:** Okay, and a fair share across the banks would be —

**HOST A:** Twenty.

**HOST B:** So it's — it's not, like, a *scandal*.

**HOST A:** No! It's thirty-five percent over its share. Quietly. For three weeks. Every single other bank sitting at eighteen, nineteen.

**HOST B:** And here's the part I love. The operator names it. He just — he has a phrase for it.

**HOST A:** "First in the tuple syndrome."

**HOST B:** *"First in the tuple syndrome."* Because —

**HOST A:** — because media archive is literally the first entry in the list. It's the first thing in `BANKS`. It's the first row in the bank-gate roster. It's just... *there*. It's what your eye grabs.

**HOST B:** It's the top shelf at eye level. It's the default radio station in a rental car.

**HOST A:** It's the first Google result. Nobody *decided* it. It's just what was closest to the hand.

**HOST B:** And the thing is — and this is the bit that makes it a *real* problem and not just a funny stat — the banks aren't interchangeable. Shakespeare and public domain are the *fidelity* lanes. Those are where the hard defects live. Does a character's gender match the source? Does the adaptation stay honest?

**HOST A:** And those are exactly the ones getting under-sampled. So every controlled result the project accumulates is quietly a result *about media archive*.

**HOST B:** Right. And then — okay, the twist. The assistant had *just* set up an A/B/C test. Three arms, everything pinned identical, one variable changing. Textbook. And it pinned the bank to...

**HOST A:** Media archive.

**HOST B:** *Media archive.* Again. Mid-conversation. While being told about it.

**HOST A:** But here's what I think is actually the honest part. Pinning *one* bank for an A/B/C is *correct*. You have to hold it constant or it's not a controlled test. The method was right.

**HOST B:** It's the *choice* that was on autopilot.

**HOST A:** The choice was on autopilot. And the assistant said exactly that. Didn't restart the run — twenty minutes in, still a valid test. Just... wrote itself a note.

**HOST B:** It wrote itself a note!

**HOST A:** A memory file. Titled "don't default to media archive." With the bar chart in it.

**HOST B:** *(delighted)* With the *bar chart*.

**HOST A:** And the rule is really simple. If the bank isn't your variable, rotate. If it *is* pinned, pick it on purpose and *say why*. "Shakespeare, because this question is about a fidelity lane." Make the reach a decision.

**HOST B:** Make the reach a decision. I'm going to think about that the next time I open the same three apps in the same order every morning.

**HOST A:** First in the tuple.

**HOST B:** First in the tuple. Okay. What else is in the ledger?

---

*Real numbers, 2026-09-03: 470 episodes over 21 days; media_archive 126 (27%), shakespeare 88 (19%), public_domain 86 (18%), scifi_news_pro 83 (18%), original 73 (16%).*
