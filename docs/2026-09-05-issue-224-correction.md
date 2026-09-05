# Correction comment for Comfy-Org/registry-backend#224

**PREPARED, NOT POSTED.** The original post is already public; this corrects two
claims in it that our own audit found to be wrong. Posting is yours.

**Why post at all rather than quietly fix it:** the post asks a reviewer to
trust our account of what the pack executes. One of its sentences is false and
is checkable in thirty seconds by grepping the zip. A reviewer who finds it
themselves discounts everything else we wrote; a publisher who corrects it
first is worth listening to. That asymmetry is the whole argument.

---

Correcting two claims in my own post above, both found by a follow-up audit of
the published `2.0.0-alpha.20` archive rather than the source tree.

**1. "Every `subprocess` call goes through one gateway" is wrong.** It is very
nearly true and I should have written what was actually measured. Every ffmpeg
and ffprobe spawn does go through `nodes/_otr_shared/proc.py`, which refuses
`shell=True`, refuses a string argv, and allowlists `argv[0]`. But
`nodes/_otr_audio_engines/eng_indextts2.py:214` calls `subprocess.Popen`
directly and bypasses that gateway. Its argv is a fixed
`[interpreter, worker.py, --model-dir, ...]` shape with no widget-supplied
element, and the file is content-hashed by the pack's own voice-route
fingerprint so it cannot be edited without demoting the route -- which is also
precisely why it was not migrated. That is a reason, not an exemption, and the
sentence as published overstated the guarantee.

**2. "no longer returns an absolute path" is overstated** for the one
unauthenticated route we still register, `GET /otr/latest_ledger`. The outer
`fullpath` field and the raw exception text were removed in alpha.20, and that
part is accurate. But the response still returns the ledger document itself,
and that document's `meta.paths` carries resolved `ledger_path`, `episode_root`
and `audio_dir`. So the route still discloses absolute paths to anyone who can
reach it. The route takes no request parameter, has no side effect, and reads a
server-chosen file -- but "no absolute paths" was not a correct description and
I would rather say so than have it read as one.

**Two packaging defects the same audit turned up, both fixed for the next
version:**

- `scripts/otr_mesh_stage_blender.py` is required at runtime by
  `nodes/_otr_video_engines/eng_mesh_stage.py` and was excluded from the
  published archive, so the mesh/3D lane could not start on a registry install.
- `scripts/_otr_indextts2_install.ps1` shipped in the archive and bootstrapped a
  dependency with `irm https://astral.sh/uv/install.ps1 | iex`. Fetching and
  executing a remote script has no business in a package under security review,
  regardless of how well-known the host is. It now refuses and prints the manual
  install command instead.

None of the four changes the scan record, which stands at 12 `info` findings.
I am flagging them because they are the kind of thing a reviewer should not
have to find, and because the first two mean my summary above was more
confident than the code justified.
