Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 76f88d8c.

THIS IS THE BETWEEN-DOCUMENTS PASS, AND IT IS OVERDUE. The standing
directive says to run it after each batch of pushes because a plan rots at
the speed you push to it; today's batch is fourteen commits, seven of them
code or corpus. The last time this pass ran it found a standing ruling
invalidated by a commit made the SAME DAY. Expect the same.

YOUR ONLY JOB IS TO FIND WHAT IS NOW FALSE. Not to review code -- other
lanes have that. Read documents against each other and against `git log
--oneline cc62b2c1..HEAD`. Read-only; no git writes; no GPU.

READ, IN THIS ORDER
  docs/GO_FORWARD_PLAN.md            -- section 2 row 1 especially
  docs/OTR_STANDING_RULINGS.md       -- the two 2026-09-19 entries at the top
  CLAUDE.md                          -- the first two bullets under operator
                                        directives, both dated today
  docs/2026-09-19-shakespeare-vendoring/*.md  -- the briefs; each states
                                        facts that were true when written
  git log --oneline cc62b2c1..HEAD   -- what actually happened

WHAT TO TEST, EACH WITH THE LINE THAT IS WRONG AND THE COMMIT THAT MADE IT SO
  1. GO_FORWARD_PLAN section 2 row 1 says "12 of them are ready to run
     TODAY" and describes the scanned lane. Every sentence in that row:
     still true, or made false by a commit today? It still says the two
     Portuguese Tempestade cells are ready. It still says `--end-label` is
     the answer to running heads. It does not mention `--pages`, the
     coordinate reader, or the standing ruling that inverted the gates.
  2. The plan's own rule is "ONLY UNFINISHED WORK BELONGS HERE" and "a
     finished prerequisite earns ONE CLAUSE inside the row that still needs
     it." Today shipped: the hyphen/label fix, the split-heading widening,
     the coordinate reader, the page window, and two corpus repairs. Which
     of those earned a clause, and which are absent? Write the clause.
  3. CLAUDE.md now carries two bullets dated 2026-09-19 above the 09-17
     Composer law. Does anything ELSE in CLAUDE.md or the standing rulings
     still say "do not push until QA holds" and now contradicts them? Grep
     for it; there are several review-routing sections.
  4. The briefs in docs/2026-09-19-shakespeare-vendoring/ each open with
     facts. PANEL_PROMPTS_spanish_abbreviated_labels.md still says the two
     Portuguese Tempestade cells "ARE ready" in its section 1 and corrects
     itself only in an addendum. Which brief statements would mislead a
     reader who arrived tomorrow, and should the folder carry a one-line
     README saying which are superseded?
  5. THE ONE THAT MATTERS MOST: the standing ruling says an unresolvable
     cue becomes an unbound speaker. GO_FORWARD_PLAN still describes the
     scanned lane as anchoring on the roster and REFUSING what does not
     resolve. Those two documents now disagree about the lane's core
     behaviour. Quote both.

FINISH WITH:
    NOW FALSE:   one line per stale statement -- file, line, the commit or
                 ruling that made it false, and the replacement sentence
    MISSING:     finished work that earned a clause in the plan and has none
    REFUTED:     every claim above you could not confirm
