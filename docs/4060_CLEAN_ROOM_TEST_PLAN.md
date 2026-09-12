# The Clean-Room Test: install OTR like a stranger, by hand, and report every lie

**Written 2026-09-12 on the operator's instruction:** *"set up in the docs a test
plan for the 4060 to nuke all local traces of OTR and use the mouse and get it
going from scratch like a human to test the most common plans for shipping, and
to phone home on any bug and change needed to make it work."*

**THIS IS THE ONLY TEST THAT CAN ANSWER THE ONE QUESTION THE DEV BOX CANNOT ASK
ABOUT ITSELF.** The 5080 wrote this code; of course it runs there. The 4060 owns
the portability surface precisely because it can be made ignorant again. A
fresh-install failure is not a 4060 bug -- it is a SHIPPING bug that every
stranger will hit on their first evening, and it is invisible everywhere else.

---

## The rules, and they are the whole point

1. **USE THE MOUSE.** ComfyUI Manager, the node menu, the template browser, the
   dropdowns. Do NOT `git clone`, do NOT `pip install -r requirements.txt` by
   hand, do NOT copy a workflow JSON into place, and do NOT set an environment
   variable, unless the README told a reader to. A step you take from muscle
   memory is a step the stranger never takes.
2. **THE README IS THE SCRIPT.** Follow it literally, in order. If you find
   yourself doing something it does not say, STOP: that is a finding, and the
   finding is that the README is wrong.
3. **DO NOT FIX ANYTHING IN CODE.** Not a path, not a pin, not a typo. Record
   it. A fix from inside the test destroys the measurement, and the 5080 owns
   `nodes/` and the shipping surface anyway.
4. **WRITE DOWN EVERY DEVIATION, INCLUDING THE ONES THAT FEEL TOO SMALL.**
   "I had to restart ComfyUI twice" is a finding. "The download bar sat at 0%
   for four minutes" is a finding. "I guessed which dropdown" is the most
   important kind of finding.
5. **TIME EVERY PHASE.** Wall clock, honestly, including downloads. "How long
   before a stranger sees an episode" is a shipping number nobody has.

---

## Phase 0 -- inventory BEFORE you destroy anything

Do this first and paste the output into the report. It is what makes the
teardown reversible-in-principle and the result interpretable.

```powershell
Get-ComputerInfo | Select-Object OsName, OsVersion, CsTotalPhysicalMemory
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version
Get-ChildItem C:\ComfyUI-Models -Directory | Select-Object Name, @{n='GB';e={0}}
```

Record: GPU, driver, RAM, free disk, Python version, and whether ComfyUI is the
Desktop app, the portable build, or a manual install. **Say which.** The three
behave differently and the README's advice splits on it.

---

## Phase 1 -- the teardown, in tiers. Pick one and SAY which you picked.

> **Read this before running anything.** Deleting the model tree costs a very
> large re-download on a laptop connection. Tier A is the honest stranger test;
> Tier B is the one you can actually finish in an evening. Both are useful and
> they answer different questions. **Tier B is the DEFAULT** unless the operator
> asks for A.

### Tier A -- total ignorance (the true clean room)

Everything goes: the pack, its models, its outputs, its caches.

```powershell
# The pack, wherever it landed
Remove-Item -Recurse -Force "<ComfyUI>\custom_nodes\ComfyUI-OldTimeRadio" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "<ComfyUI>\custom_nodes\comfyui-old-time-radio"  -ErrorAction SilentlyContinue
# Its outputs
Remove-Item -Recurse -Force "<ComfyUI>\output\otr" -ErrorAction SilentlyContinue
# Its weights. THIS IS THE EXPENSIVE ONE.
Remove-Item -Recurse -Force "C:\ComfyUI-Models\LLM\converted" -ErrorAction SilentlyContinue
# Hugging Face cache, which will otherwise silently make a download instant
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface" -ErrorAction SilentlyContinue
```

### Tier B -- pack-only (the realistic re-test)

The pack and its Python state go; the weights stay. This still catches the
majority of shipping bugs -- dependency resolution, node registration, template
discovery, workflow wiring, dropdown defaults -- because those are what break.

```powershell
Remove-Item -Recurse -Force "<ComfyUI>\custom_nodes\ComfyUI-OldTimeRadio" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "<ComfyUI>\custom_nodes\comfyui-old-time-radio"  -ErrorAction SilentlyContinue
```

### Either tier -- then prove it is GONE

```powershell
Get-ChildItem "<ComfyUI>\custom_nodes" | Select-Object Name
```

Restart ComfyUI and confirm **no `[OldTimeRadio]` lines** in the console and
**no OTR nodes** in the node menu. If either survives, the uninstall is
incomplete and THAT IS A FINDING -- a stranger who "removed" the pack still has
it.

**Also check for a junction.** On the dev box
`ComfyUI-Installs\...\OldTimeRadio` is a junction to the real repo; if anything
similar exists here, deleting the visible folder may not delete the target, or
may delete more than you meant:

```powershell
Get-ChildItem "<ComfyUI>\custom_nodes" -Force | Where-Object { $_.LinkType }
```

---

## Phase 2 -- install like a human

**By the mouse, through ComfyUI Manager.**

1. Open ComfyUI. **Manager → Custom Nodes Manager**.
2. Search `old time radio`. Record **exactly what you see**: the name, the
   version offered, the description, whether it is findable by the obvious
   search term. If you have to search something clever to find it, write down
   what you tried first and what failed.
3. Install it. **Watch the console while it installs** and keep the text.
4. Restart ComfyUI when Manager asks.

**Record, per the README's own warnings, whether any of these happened:**
* Manager says **"not a CNR node"** or **"Cannot resolve install target"**. The
  README says this means the newest version is still Pending the registry scan,
  and is NOT a local fault. Note the version it offered you.
* The pack installs but **some nodes are missing**. The loader imports each node
  in its own try/except and prints `[OldTimeRadio] Skipped '<name>': <reason>`.
  **Copy every Skipped line.** Those name a missing dependency.
* **ComfyUI does not start at all.** That is the worst class and the README
  documents a real instance of it. Capture the full traceback before anything
  else.

**Then count what you got:**

Console should carry `[OldTimeRadio] OK - All NN nodes loaded successfully`.
Write down NN. If it is lower than the previous drill, that is a regression.

---

## Phase 3 -- load the show, by the mouse

**Workflow → Browse Templates → EXTENSIONS → comfyui-old-time-radio →
`otr_canonical`.**

There is exactly one entry. Findings to record:
* Is the template browser path actually what the README says?
* Does the graph load with **no red nodes** and no missing-node dialog?
* Does anything show a **red or empty dropdown**? Name the node and the widget.
  An empty dropdown means a model the pack expects is not present, and the
  stranger has no way to know which.

---

## Phase 4 -- the most common shipping plans, in order of who will run them

Run each as its own attempt. **After each, say whether an episode reached
`output/otr/obs/`** -- that is the only definition of success. A green log with
nothing in obs is a failure.

### Plan 1 -- press Run and see what happens (the true default)

Load `otr_canonical`, change **nothing**, press **Queue Prompt**.

This is what most people will actually do. It is the highest-value single
measurement in this document. Record: does it run to a published episode on an
8 GB card with no edits at all? If it OOMs, at which node? If it stalls, where?

### Plan 2 -- the README's 8 GB row, by the mouse

Per the README: set the three **OTR_VideoDirector** video roles to
`animatediff15_v3_haunted_video (16:9)`, and `llm_device` → `cuda` in
**OTR_LedgerScriptWriter**. Queue.

Record: were those dropdowns findable from the README's wording alone? Did the
values it names actually exist in the dropdown, spelled that way? **A dropdown
label that does not match the README is a shipping bug**, and it is the single
most common one this pack has had.

### Plan 3 -- a shipping 4060 profile

The profiles marked `shipping` for this class are
`otr_4060_12b_gguf_offload`, `otr_nvidia_8gb_haunted`, `otr_w45_ltx_8gb` and
`otr_g4_ltx_8gb`. Run **at least** `otr_4060_12b_gguf_offload`.

Do NOT use a `draft` profile for the legs that must land. **Afterwards, try the
drafts anyway** -- the operator struck the blanket forbid on 2026-09-11 with
*"you kept telling me this won't work"* -- and record what actually happens, good
or bad. A draft that works is worth knowing.

### Plan 4 -- one act, every bank

One-act legs across `scifi_news_pro`, `public_domain`, `media_archive`,
`original` and `shakespeare`. The repo ships
`scripts/otr_oneact_regression.py` for exactly this, and it checks the artifacts
on disk rather than trusting the log. Using it is fine here -- it is not a
mouse-path test, it is a coverage sweep, and you have already done the mouse
path in plans 1 and 2.

### Plan 5 -- the opt-in extra, only if time allows

The `anime` visual style can use its own SD 1.5 checkpoint, documented in the
README beside the `sd15` row. It is deliberately NOT auto-fetched. Follow the
README's fetch instructions **exactly as written** and report whether they work
verbatim on a clean box.

---

## Phase 5 -- phone home

**Write findings into `docs/4060_DRILL_LOG.md`**, which is this box's own
chronological log and is append-only and shared; both boxes push it and
`.gitattributes` marks it `merge=union` so a tail collision keeps both sides.
Push it. Do not edit the 5080's files.

**For every finding, record all five of these.** A finding missing the last two
cannot be acted on:

| field | why |
|---|---|
| **What you did** | the click or command, verbatim |
| **What you expected** | from the README, quoted |
| **What happened** | the error, the console text, the screenshot |
| **What you had to change to get past it** | the actual workaround |
| **Whether a stranger could have worked that out** | yes/no, and be harsh |

That last row is the one that matters. A bug you solved in ten seconds from
knowing the codebase is a wall for somebody who does not.

**Phone home immediately, without waiting for the end, if:**
* ComfyUI does not start after install
* the node count is lower than the previous drill
* a README instruction names a dropdown value that does not exist
* nothing reaches `output/otr/obs/` on Plan 1

**Report the HEAD you ran**, and say whether the pack came from the registry or
from git. They are not the same artifact and they fail differently.

---

## What this test may NOT conclude

* **Nothing about whether the audio sounds good.** That is the operator's ear
  alone. Do not record a verdict on music or voices; record only whether they
  were PRODUCED.
* **Nothing about the 5080's behaviour.** A change that fixes an 8 GB card can
  silently degrade the 16 GB card that renders the real episodes. Findings here
  are input to a fix, never the fix itself.
* **Nothing from a green log alone.** Every claim in the report is backed by a
  file on disk or a console line you pasted.
