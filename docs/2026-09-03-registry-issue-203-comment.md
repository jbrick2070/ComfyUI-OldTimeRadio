# Ready-to-paste comment for Comfy-Org/registry-backend#203

Jeffrey posts this; Claude does not. Issue #203 was opened 2026-08-24 by
jbrick2070 and has no comments, labels or assignee as of 2026-09-03.

**Why comment rather than open a new issue:** #203 already names the right
mechanism (the leap-day cron). What it lacks is evidence that the problem is
registry-wide rather than one publisher's pack, and its stated date window is
wrong in a way that would send a maintainer looking in the wrong month. Both are
fixed below. Reproduce with `scripts/registry_extract_survey.py` in this repo.

---

## STEP 1 -- edit the issue title in place (GitHub allows this)

The current title says "published after ~Feb 2026". The data says 2026-04-28,
and the title is what a maintainer scans. Replace it with:

    Node extraction has produced no successful extract for any pack since 2026-04-28 (NODES panel empty registry-wide)

## STEP 2 -- post everything below as a comment

Following up with measurements across the whole registry, because I could only
speak to my own pack before, and one detail in my original report was wrong.

**The date window in my first post is wrong. Extraction did not stop in
February; it stopped at the end of April 2026.**

I sampled 480 node packs via `GET /nodes/{id}/versions?include_status_reason=true`
and read `comfy_node_extract_status`, which carries terminal values (I see both
`success` and `invalid_format`). Of the 480, 81 have ever recorded a `success`.
Grouping those packs by their most recent successful extract:

| month of newest success | packs |
|---|---|
| 2026-04 | 19 |
| 2026-03 | 13 |
| 2026-02 | 5 |
| 2026-01 | 4 |
| 2025-12 | 9 |

**The most recent successful extract anywhere in the sample is
`2026-04-28T03:21:26` (`comfyui-google-genmedia-custom-nodes`). Nothing in 480
packs has extracted since.** So the job was healthy through April and has
produced nothing for four months.

**It is not publisher-specific, and not related to a version's review status.**
These are established packs, several of them `Active`, whose newest versions sit
at `comfy_node_extract_status: "pending"` indefinitely while earlier versions of
the same pack extracted fine:

| pack | version status | prior successes | newest success | pending since |
|---|---|---|---|---|
| `yogurtnodes` | Active | 50 | 2026-02-28 | 2026-06-09 |
| `rgthree-comfy` | Flagged | 47 | 2026-04-07 | 2026-05-09 |
| `basic_data_handling` | Active | 37 | 2026-03-26 | 2026-05-07 |
| `comfyui-rmbg` | Active | 32 | 2026-01-18 | 2026-07-21 |
| `comfyui-inpaint-cropandstitch` | Active | 23 | 2026-03-17 | 2026-05-01 |

`rgthree-comfy` is the clearest case: 47 versions extracted successfully, and
every version published since 2026-05-09 is stuck `pending`.

**Consequence for the API:** `/nodes/{id}/versions/{version}/comfy-nodes` returns
`{"comfy_nodes": null, "totalNumberOfPages": 0}` for every affected version, so
the NODES panel is empty for anything published in the last four months.

Happy to share the survey script if it is useful. I am not asking for my own pack
to be prioritized -- I only care that publishers are not spending time debugging
their `requirements.txt` for a job that has not run since April. I nearly did
exactly that.
