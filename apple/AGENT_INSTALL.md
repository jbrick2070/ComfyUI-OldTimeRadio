# Installing OTR with an AI coding agent

**You do not need this file.** The install path in the README is the normal one
and most people should just follow it: install from Extensions -> Node Manager, put
ffmpeg on PATH, restart, open the template, press Queue. This page exists for
people who would rather hand the job to Claude Code, Codex, Cursor, Gemini CLI,
Copilot or any other agent with a terminal -- and for the agent itself, which is
the real audience below.

Hand your agent this file, or paste the prompt in the next section. Everything
after that is written **to the agent**.

---

## The prompt to paste

> Install the ComfyUI-OldTimeRadio custom node pack on this machine and prove it
> loaded. Read `apple/AGENT_INSTALL.md` in the repo
> (https://github.com/jbrick2070/ComfyUI-OldTimeRadio) and follow it exactly --
> it names the verification command for every step and the traps that waste
> time. Do not modify any file in the pack. Report which ComfyUI install,
> which Python, and the output folder you verified.

---

## Agent instructions

You are installing a ComfyUI custom node pack. The user wants a working install
and a first episode, not a tour. Work in this order; each step has a command
that PROVES it rather than a claim that it worked.

### Find the real ComfyUI, and its real Python

Everything downstream depends on getting these two right, and on most machines
there is more than one candidate. **The Python that matters is the one ComfyUI
itself runs on** -- not `python3` on PATH, not a conda env you activated.

| Install kind | Where ComfyUI lives | The Python to use |
|---|---|---|
| ComfyUI Desktop (Win/Mac) | `%APPDATA%\ComfyUI`, or on a Mac `~/Library/Application Support/ComfyUI` **or** `~/Library/Application Support/Comfy Desktop` -- check both; models live elsewhere again | the bundled venv under the install dir |
| Portable (Windows) | the unzipped folder | `python_embeded\python.exe` |
| git clone | wherever they cloned it | that tree's `.venv` or the env they launch with |

Verify before continuing:

```
<ComfyUI Python> -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__, torch.cuda.is_available())"
```

Record each printed line. If `torch` does not import, you have the wrong Python --
find the right one before doing anything else. **On an Apple Silicon machine
`torch.cuda.is_available()` is correctly `False`; check `torch.backends.mps.is_available()`
instead.**

The Python version decides the voice backend, and this is not a preference:

* **3.10, 3.11, 3.12** -> the torch `kokoro` package and every admitted
  language (`requirements.txt` pins it to
  `python_version < "3.13"`).
* **3.13** (what Desktop and the portable build ship) -> `kokoro-onnx`, on the
  CPU, about six times faster than realtime. This is the English path;
  non-English rows require the torch backend above.
* **3.14** -> **no kokoro backend is packaged yet.** The pack still installs;
  an English user can switch the voice controls to Bark. Non-English episodes
  are unavailable because day-one multilingual rows admit Kokoro only. Say so
  plainly rather than letting them hit it at render time.

For a multilingual install, also read [MULTILINGUAL.md](MULTILINGUAL.md);
Japanese and Mandarin have language-specific readiness extras that are not an
English-install tax.

### Install the pack

Prefer the registry unless the user asked for the git tree:

```
comfy node install comfyui-old-time-radio
```

Or in the GUI: **Manager -> search "Old Time Radio"** (registry id
`comfyui-old-time-radio`, publisher `fluxus`).

The git route, which is the one to use if they want `scripts/` (the headless
runners and provisioners are **not** in a registry install):

```
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio
<ComfyUI Python> -m pip install -r ComfyUI-OldTimeRadio/requirements.txt
```

Clone it **into the ComfyUI `custom_nodes/` directory**, and use the ComfyUI
Python for the pip install. A system pip installs the libraries somewhere
ComfyUI will never look, and the failure appears much later as skipped nodes.

`main` is the default branch and the only one that moves. A clone made before
2026-09-13 wants re-cloning rather than pulling.

### ffmpeg AND ffprobe, and a current build

Both binaries, on PATH. Every episode is mixed, captioned and muxed through
them, and the final mux copies the master audio into the MP4 losslessly, which
older builds cannot write.

```
winget install Gyan.FFmpeg          # Windows
brew install ffmpeg                 # macOS
```

On Debian/Ubuntu, **check the version first**: 22.04's apt build is 4.4 and
fails. Take a static build. The floor is 6.1, where FFmpeg's MP4 muxer gained
PCM; measured on 2026-09-13 by running this pack's own probe on four builds
across three machines: 4.4.2 fails; 7.0.2-static, 8.0.1 and 9.0 pass. No 6.x
build has been run, so the floor is documented rather than measured.

```
ffmpeg -version; ffprobe -version
```

Use `;` rather than `&&`: on Windows PowerShell -- the platform this pack is
native to -- `&&` is a parser error.

Nothing in the pack parses that version number -- it muxes a fifth of a second
of real silence at the start of a run and refuses, in about a second, if the
build cannot write it. Your job is only to make both binaries reachable.

On Linux also install one monospace font for the burned captions
(`fonts-dejavu-core` is enough).

### Restart ComfyUI fully, then read the console

A reload is not enough; the process must restart. Then verify:

```
grep -i "OldTimeRadio" <comfyui console log>              # macOS / Linux
Select-String -Pattern OldTimeRadio <comfyui console log>    # Windows PowerShell
```

**What each outcome means -- do not guess between them:**

* `[OldTimeRadio]` lines and an **OldTimeRadio** category in the node menu ->
  installed. Move on.
* `[OldTimeRadio] Skipped '<name>': <reason>` for a FEW nodes -> one dependency
  is missing. That is by design: `__init__.py` loads each node in its own
  try/except, so a partial install still works and the skip names its own
  cause. Install the named dependency with the ComfyUI Python. A clean install
  prints `[OldTimeRadio] OK - All <n> nodes loaded successfully` (n is the pack's node count); a few skips
  below that number is a missing library, not a broken pack.
* **Zero nodes and no `[OldTimeRadio]` lines at all** -> the pack is not being
  loaded. It is in the wrong directory, ComfyUI is not scanning it, or
  `prestartup_script.py` died. **Do not chase missing libraries for this
  outcome** -- a dependency problem cannot produce a total zero.

### Run one episode

**Workflow -> Browse Templates -> EXTENSIONS -> comfyui-old-time-radio.** The
entry is named after the pack's folder, so on a git clone it reads
`ComfyUI-OldTimeRadio`; the console banner prints the exact name at boot. There
is exactly one entry, `otr_canonical`. Open it and press **Queue**. Change
nothing: every dropdown already holds a working value.

`act_count` on **OTR_LedgerScriptWriter** already ships at `1` in
`otr_canonical`, which is the shortest proof. Leave it alone.

The first run downloads about 12 GB -- the writer, the music model and its text
encoder, and the Kokoro voices. Later runs skip it.

### Prove it -- and this is the only proof that counts

The finish line is a finished `.mp4` in:

```
<ComfyUI output folder>/otr/obs/
```

Not the console, not a green node, not the absence of errors. Read the
`obs_publish` line before concluding anything: `obs_publish OK -> <path>` names
the real destination, and `obs_publish BLOCKED -- ...` means the run SUCCEEDED
with the episode in `otr/episodes/<episode>/` and only the published copy
withheld. **No `obs_publish` line at all means the run did not finish**, however
clean the log looked.

Report the absolute path you verified and the filename you found.

---

## Things that waste an agent's time here

Each of these cost a real session before it was written down.

* **Do not diagnose from the registry web page.** "No nodes found" there is
  normal for nearly every pack; it is a separate extraction service, not a
  health signal. Install locally and look at the node menu.
* **Do not chase torch or dependency ghosts** when Manager says "not a CNR
  node" or cannot resolve an install target. That is a registry-side state, not
  a local fault.
* **Do not edit files in the pack to make an install work.** If something needs
  editing, that is a bug worth reporting, not a local patch. The saved graphs in
  `workflows/variants/` are GENERATED -- editing one is silently undone the next
  time they are rebuilt.
* **Do not assume the output folder from the launch flags.** The pack honours
  `OTR_OUTPUT_DIR`, so a machine can publish somewhere other than ComfyUI's
  `--output-directory`. Take the truth from the server log's pinned line or from
  `obs_publish OK ->`.
* **Do not pick a per-machine graph for the user.** The graphs under
  `workflows/variants/` are presets, and the canonical template in the menu is
  the intended first run. Shipping rows have a published episode behind them.
  `otr_amd_still` reads `draft` in its profile but DOES have one -- an outside
  tester published an episode from it on a Radeon AI PRO R9700 under ROCm 7.2
  on 2026-09-14. The `status` field records promotion, not proof; for proof
  read the machine table in [MACHINES.md](MACHINES.md), which is generated.
* **On Apple Silicon, read [MAC.md](MAC.md) before choosing anything heavier
  than the defaults.** An out-of-memory on unified memory can reboot the
  machine, not just the render. Some Desktop installs also need
  `--extra-model-paths-config` and `--output-directory` on the launch line, or
  the model folders come up empty and episodes land somewhere nobody is looking.

## Where to read more

[INSTALL.md](INSTALL.md) is the long form of all of this for humans.
[RUN.md](RUN.md) covers what to do once it runs. [MACHINES.md](MACHINES.md) is
the engine-by-machine table. [PREFLIGHT.md](PREFLIGHT.md) is the checks that say
whether what you built will actually work.
