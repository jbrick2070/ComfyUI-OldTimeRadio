# Skill pointer updates -- re-point all handoff skills at docs/GO_FORWARD_PLAN.md (2026-06-12)

Canonical go-forward plan is now `docs/GO_FORWARD_PLAN.md` (single source of truth). The handoff
skills still name the old split docs. Apply the edits below via **Settings > Capabilities** (skill
editor) -- they cannot be persisted from a Cowork session (the in-session skill copy is a read-only
cache). After editing, the change takes effect in the NEXT session.

Three skills are affected: `otr-build-handoff` (ACTIVE resume skill -- most important),
`otr-video-handoff` (SUPERSEDED -- make it a redirect), and `session-handoff` (generic -- light
touch only).

---

## 1. otr-build-handoff  (ACTIVE -- update first)

This is the skill a fresh window runs to resume. Re-point its three source-of-truth references to
the canonical doc.

**a) In the intro `**Why:**` paragraph, replace:**
> The DURABLE roadmap is the `otr-build-tracker` artifact (persists across windows: the
> distance-to-100% gauge + lanes + the sprint table). The current-step detail is
> `docs/VIDEO_BUILD_HANDOFF.md`. The forward order is `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`
> **section 0**.

**with:**
> The SINGLE SOURCE OF TRUTH is `docs/GO_FORWARD_PLAN.md` (git-tracked: forward order + runway +
> open tickets + current step + hard rules). The `otr-build-tracker` artifact is the visual
> DASHBOARD that mirrors it. `docs/VIDEO_BUILD_HANDOFF.md` and `3D_TOOLKIT_PLAN.md` section 0 are
> thin pointers to `GO_FORWARD_PLAN.md`.

**b) In the RESUME section, replace step 1-2 file list:**
> 1. Read `docs/VIDEO_BUILD_HANDOFF.md` (current step + WHERE WE ARE); skim the `otr-build-tracker`
>    artifact ...; run `git log --oneline -12` + `git status` on `v2.0-alpha`.
> 2. Read the forward order in `3D_TOOLKIT_PLAN.md` **section 0**.

**with:**
> 1. Read `docs/GO_FORWARD_PLAN.md` IN FULL (current step + hard rules + forward order + runway +
>    open tickets); skim the `otr-build-tracker` dashboard; run `git log --oneline -12` +
>    `git status` on `v2.0-alpha`.
> 2. The forward order + current step are in `GO_FORWARD_PLAN.md` (sections 1 + 3). The 3D detail
>    spec is `3D_TOOLKIT_PLAN.md` (forward-order item 5).

**c) In the HAND OFF section, replace "Refresh `docs/VIDEO_BUILD_HANDOFF.md`" with**
> Refresh `docs/GO_FORWARD_PLAN.md` (and the `otr-build-tracker` dashboard to match)

and in the printed kickoff block, replace `read 3D_TOOLKIT_PLAN.md SECTION 0 + docs/VIDEO_BUILD_HANDOFF.md`
with `read docs/GO_FORWARD_PLAN.md`.

---

## 2. otr-video-handoff  (SUPERSEDED -- make it a redirect)

This skill is replaced by `otr-build-handoff` and still points at the retired
`OTR_VIDEO_ENGINE__EXECUTION-PLAN.md`. Replace the whole body under the frontmatter with:

```markdown
# OTR Video-Build Handoff -- SUPERSEDED

> Use the **otr-build-handoff** skill instead (bidirectional resume + hand off, anti-drift).
> The single source of truth is **docs/GO_FORWARD_PLAN.md** -- forward order, runway, open
> tickets, current step, and hard rules all live there. `VIDEO_BUILD_HANDOFF.md` and
> `3D_TOOLKIT_PLAN.md` section 0 are thin pointers to it. Do NOT write handoffs from this skill;
> run otr-build-handoff.
```

(Or delete the skill entirely, since otr-build-handoff covers both directions.)

---

## 3. session-handoff  (GENERIC -- light touch, keep it general)

This is a general-purpose skill; do not make it OTR-specific. One small addition keeps it
consistent. In **Generate mode**, the paragraph that begins "Do not duplicate what already lives
in persistent project files ... CLAUDE.md, README, ROADMAP, BUG_LOG", add `GO_FORWARD_PLAN.md` to
that list:

> ... If the project has a CLAUDE.md, README, ROADMAP, BUG_LOG, **GO_FORWARD_PLAN.md**, or similar,
> those load automatically in the next session -- repeating them wastes the very tokens this skill
> exists to save.

That's the only change -- session-handoff stays generic for non-OTR projects.

---

## Apply order
1. otr-build-handoff (active -- do this one for sure).
2. otr-video-handoff (redirect or delete).
3. session-handoff (optional one-line addition).
