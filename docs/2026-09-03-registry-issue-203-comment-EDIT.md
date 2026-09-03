# Corrected comment body for Comfy-Org/registry-backend#203

The posted version lost both markdown tables on paste: the histogram collapsed
onto one line and the named-packs table vanished entirely -- which was the
strongest evidence, since it is what shows the problem is not one publisher's.

Edit the existing comment (the "..." menu on your own comment -> Edit) and
replace its body with everything below the line. No pipe tables, so nothing can
collapse.

---

Following up with measurements across the whole registry, because I could only
speak to my own pack before, and one detail in my original report was wrong.

**The date window in my first post is wrong. Extraction did not stop in
February; it stopped at the end of April 2026.**

I sampled 480 node packs via `GET /nodes/{id}/versions?include_status_reason=true`
and read `comfy_node_extract_status`, which carries terminal values (I see both
`success` and `invalid_format`). Of those 480, only 81 have ever recorded a
`success`.

Grouping those 81 packs by the date of their most recent successful extract:

- 2026-04: 19 packs
- 2026-03: 13 packs
- 2026-02: 5 packs
- 2026-01: 4 packs
- 2025-12: 9 packs

**The most recent successful extract anywhere in the sample is
`2026-04-28T03:21:26` (`comfyui-google-genmedia-custom-nodes`). Nothing in 480
packs has extracted since.** The job was healthy through April and has produced
nothing in the four months since.

**It is not publisher-specific, and it is unrelated to a version's review
status.** These are established packs -- most of them `Active` -- whose newest
versions sit at `comfy_node_extract_status: "pending"` indefinitely, while
earlier versions of the same pack extracted fine:

- `yogurtnodes` (Active): 50 prior successes, newest success 2026-02-28, pending since 2026-06-09
- `rgthree-comfy` (Flagged): 47 prior successes, newest success 2026-04-07, pending since 2026-05-09
- `basic_data_handling` (Active): 37 prior successes, newest success 2026-03-26, pending since 2026-05-07
- `comfyui-rmbg` (Active): 32 prior successes, newest success 2026-01-18, pending since 2026-07-21
- `comfyui-inpaint-cropandstitch` (Active): 23 prior successes, newest success 2026-03-17, pending since 2026-05-01

`rgthree-comfy` is the clearest case: 47 versions extracted successfully, and
every version published since 2026-05-09 is stuck `pending`.

**Consequence for the API:** `/nodes/{id}/versions/{version}/comfy-nodes` returns
`{"comfy_nodes": null, "totalNumberOfPages": 0}` for every affected version, so
the NODES panel is empty for anything published in the last four months.

Happy to share the survey script if it is useful. I am not asking for my own pack
to be prioritized -- I only care that publishers are not spending time debugging
their `requirements.txt` for a job that has not run since April. I nearly did
exactly that.
