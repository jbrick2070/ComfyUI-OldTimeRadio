you cn run 5080 gpu test comfy on yrou own just spin up ahealdess api thing i think it runs inn 8000 dont adk me you do it and ask roubndatbel if youc ant

PRIME DIRECTIVE: never ask the operator to run scripts, commands, or anything. Use Desktop Commander to run everything. If Desktop Commander can't do it, use Windows MCP. Never hand the operator a bat/cmd/PowerShell block and say "run this" -- YOU run it.

when it doubt yuuse dekstop commander to do anyting don ask me tio push scripts or cmd or pwoertsheell you di  t

i desktop commander doesnt do it use windows mcpo

be sure handoff fiels and try not to creatre new fiels try to kleep to te file ou are using t omake tinsg esier

dont use word dummy use placehodler as duymmy makels me feel bad

when coding try to keep coding into all sprints are done unles yo absoyult need me you can roubndtabel for a ocupel opions gpt and gemni and otehr if needed before aksing mne but only 2-3 panels

## GIT POLICY (operator directive 2026-06-10 -- never lose work)

- ONE branch: v2.0-alpha. COMMIT AND PUSH TOGETHER: every green commit gets
  pushed to origin immediately, same session, no exceptions. Local-only
  commits are the failure mode we guard against (fear of losing work).
- The operator eyeball gates TAGS and PROMOTIONS (v2.0-alpha-stable, prod,
  main, v2 release) -- NEVER pushes. Pushing to v2.0-alpha is always safe,
  expected, and required.
- This SUPERSEDES any "do not push until the eyeball passes" line in any
  handoff, doc, or memory written before 2026-06-10 evening.
- A stable branch only exists if the operator explicitly declares one.
- After every push verify: HEAD == origin, no 0-byte files, no BOM, AST
  parse on touched .py files.
