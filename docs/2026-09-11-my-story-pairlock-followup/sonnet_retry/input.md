# Pairlock_02 follow-up: finished-code review anchor

This is a bounded repair continuation after the Opus/Cursor EOS consensus and
Sonnet EOS QA. Root is the only production coder; Einstein, Dewey and Euler
independently read the actual live artifacts and existing owners. Their audit
receipts are under ../2026-09-11-my-story-5080-qualification/.

## Observed production result

The full shipped canonical runner on pushed bd814148 completed prompt
a6c9d550-12ec-4dd3-b9d2-031e3d265b1b in 834 seconds. All 23 nodes / 63 links
used the same complete prompt as pairlock_01, including Qwen3.5-4B NF4/SDPA,
balanced 0.85/0.95, one act, flexible cast, still_pan/Z-Image/IndexTTS/Kokoro/SA3.
No cached nodes; terminal history success and final obs publication verified
with file hashes and ffprobe. Actual native decodes terminated on chat EOS248046
with the shared [248044,248046] set. The frozen ledger is structurally clean.

Source qualification failed. P0 correction omitted metadata and Pydantic defaults
silently replaced 15 accepted values, without one explicit changed value in raw
JSON. P1 explicitly returned blank cast genders while describing a man/woman.
Mother's gender was absent even in original P0; this is distinct from P0 erosion.
CastLock skipped blank-gender rows, renderer selected actual male IndexTTS refs,
and credits read legacy Bark presets. Scope authorization used the unbound slot,
returned spans:null twice, and retained a flagged original spoken line. Both scene
source corrections echoed singular fallback prompts despite full source/context;
pixels show a lone adult man and a woman with two children at empty tables, not
the required shared adult dinner. Dispatcher did not lose a good correction.

## Root judgment and exact changes

1. Opt-in full-artifact conservation at the source correction owner. Reconstruct
   omitted fields from an immutable accepted candidate with model_fields_set;
   explicit values/clears and returned list membership/order win. Unique explicit
   identity joins only, never positions or default act numbers. Revalidate a fresh
   object and capture exactly the original structural validator's accepted result
   for both return and receipt. No shared structured_call contract change. P0-P3
   opt in; spoken edits and scene-prompt response shapes do not.
2. Bind the existing cleanup scope authorization schema once, after deterministic
   no-call shortcuts, and reuse its callable for the existing two attempts.
   Binder failures propagate; absent capabilities retain routing. No nullable
   workaround or global automatic binder.
3. Clarify existing P0/P1 prompts and schema descriptions: source relationships,
   descriptions and contextual pronouns convey gender, explicit identity wins,
   names never imply gender, genuinely unspecified stays empty. A post-validator
   rejecting empty/conflicting gender was considered and not added: an additional
   semantic story-rejection gate violates the operator contract. No prose regex.
4. Extend CastLock's existing unservable-gender reference stamping to blank gender,
   retaining the row's actual gender unchanged and keeping the Google no-fallback
   exception. The existing durable owner writes ref/engine into wire and disk;
   rendering and credits read it. No new post-freeze writer or voice selector.
5. At the existing scene correction owner, active speaker is focus within the
   current source-defined scene. Ask the same correction call to compose actual
   participants/current ages/shared action/necessary objects and avoid literal
   visualization of spoken memories. Expose already-recorded treatment age/gender
   by unique normalized exact name. Version the scene-context contract so cache
   identity changes. Neutral portraits, other banks, seed/sampling/reference-image
   policy and the real canonical graph are unchanged.

## Evidence and limits

Initial focused suite:362 passed. Added durable blank-gender wire/disk/render/
credits contract:2 passed after fixing the test's wrong layout key. Full regression
is running against prior 14,379 passes /51 known failures /183 skips /1 xfail.
No quarantines or expected-failure modifications. Bible promotion/checks pending.
Actual code and test diff plus full affected owner excerpts are supplied separately.

Tests prove routing/conservation/receipts, not model obedience or source fidelity.
Fresh full canonical prompt-and-pixel qualification is still required after final
QA and push. No new live run while coding/QA remain. Mac/4060 held until explicit
operator release; RunPod authentication unavailable, no rented compute started.

Please review the finished changes for concrete remaining defects. Prioritize
wrong output, unreachable owners, changed existing budgets/routes, stale receipts,
and errors that the tests miss. Name exact file/function and a reproducer. Do not
propose another checker/chunker/model pass, subjective story gate, or EOS redesign.
If there is no remaining must-fix in scope, say so. Root verifies every claim.


## Exact finished diff
```diff
diff --git a/nodes/_otr_ledger_clean.py b/nodes/_otr_ledger_clean.py
index bc72d445..06f17b41 100644
--- a/nodes/_otr_ledger_clean.py
+++ b/nodes/_otr_ledger_clean.py
@@ -1508,12 +1508,16 @@ def _authorize_repair_scope(
         receipt["model_calls"] += 1
         record["calls"].append({"attempt": attempt, "raw_output": raw,
                                 "error": None if error is None else f"{type(error).__name__}: {error}"})
 
+    # Reuse the scheduler's advertised native schema owner for both attempts.
+    # Remote/GGUF slots without this capability retain their existing routing.
+    bind = getattr(slot_fn, "_otr_bind_schema", None)
+    authorization_fn = bind(_ScopeAuthorization) if callable(bind) else slot_fn
     try:
         # LLM slot: creative -- the existing dialogue slot authorizes this edit.
         result = structured_call(
-            prompt=built, schema=_ScopeAuthorization, slot_fn=slot_fn,
+            prompt=built, schema=_ScopeAuthorization, slot_fn=authorization_fn,
             base_temperature=JUDGE_TEMPERATURE, structural_retry_temperature=0.1,
             max_new_tokens=_MAX_NEW_TOKENS, max_attempts=2, post_validator=validate,
             on_attempt_complete=completed, helper_name="ledger_clean_scope_authorization",
         )
diff --git a/nodes/_otr_my_story.py b/nodes/_otr_my_story.py
index f2128edd..8936529a 100644
--- a/nodes/_otr_my_story.py
+++ b/nodes/_otr_my_story.py
@@ -129,9 +129,12 @@ class Requirement(BaseModel):
 
 class NamedCast(BaseModel):
     name: str = ""
     notes: str = ""
-    stated_gender: str = ""
+    stated_gender: str = Field(default="", description=(
+        "Gender conveyed by source descriptions, relationships or pronouns in context. "
+        "Explicit identity takes precedence over a conventional role. Never infer from "
+        "a name; leave genuinely unspecified gender empty."))
     speaking: bool = True
     required: bool = True
 
     @field_validator("stated_gender", mode="before")
@@ -190,9 +193,12 @@ class StoryInterpretation(BaseModel):
 class CastMember(BaseModel):
     name: str = Field(min_length=1)
     role: str = ""
     character_description: str = ""
-    gender: str = ""
+    gender: str = Field(default="", description=(
+        "Preserve source gender from descriptions, relationships and pronouns in context, "
+        "honoring explicit identity first. Keep gender consistent with the character's "
+        "casting description. Never infer from a name; unspecified remains empty."))
     age_band: str = "n/a"
     # `speech_register`, not `register`: the bare name shadows a pydantic
     # BaseModel attribute and pydantic warns about it at class construction.
     # The seam asks for "register"; the alias keeps the prompt's word while
@@ -393,9 +399,13 @@ def _call(pass_id: str, bundle: Any, *, attempt_receipts=None,
     corrected, receipt = _SOURCE.rewrite_story_source(
         bundle.fields, original, kwargs["slot_fn"], schema=kwargs["schema"],
         receipts=source_rewrite_receipts, pass_id=pass_id,
         post_validator=kwargs.get("post_validator"), slot_scheduler=slot_scheduler,
-        configured_model_id=configured_model_id, author_context=author_context)
+        configured_model_id=configured_model_id, author_context=author_context,
+        preserve_omitted={
+            ("requirements",): "id", ("named_cast",): "name",
+            ("conflicts",): "requirement_id", ("cast",): "name", ("acts",): "n",
+        })
     # This runs once AFTER author acceptance, never inside its validator. A
     # source rewrite cannot restart the author ladder or check its own output.
     if corrected is None:
         return authored
diff --git a/nodes/_otr_story_source.py b/nodes/_otr_story_source.py
index 56fab2ad..3e8c3bf5 100644
--- a/nodes/_otr_story_source.py
+++ b/nodes/_otr_story_source.py
@@ -78,12 +78,55 @@ def _complete_repair(*, original_prompt, failed_output, error):
             "material. Do not return a review or a request to try again." % error)},
     ])
 
 
+def _retain_omitted(model, original, identities, path=()):
+    """Conserve omitted fields; explicit values and list membership win.
+
+    The author declares each list's stable identity. Missing, blank or duplicate
+    identities cannot borrow metadata, and list positions never match. Return
+    data for fresh validation, without mutating the candidate or parsed model.
+    """
+    values = model.model_dump(mode="json")
+    for name in type(model).model_fields:
+        if name not in model.model_fields_set:
+            if name in original:
+                values[name] = original[name]
+            continue
+        value = getattr(model, name)
+        prior = original.get(name)
+        field_path = path + (name,)
+        if isinstance(value, BaseModel) and isinstance(prior, dict):
+            values[name] = _retain_omitted(value, prior, identities, field_path)
+        elif isinstance(value, list) and isinstance(prior, list) and field_path in identities:
+            identity = identities[field_path]
+
+            def key(item):
+                if isinstance(item, BaseModel):
+                    if identity not in item.model_fields_set:
+                        return None
+                    result = getattr(item, identity, None)
+                else:
+                    result = item.get(identity) if isinstance(item, dict) else None
+                if isinstance(result, str):
+                    return " ".join(result.split()).casefold() or None
+                return result if isinstance(result, int) and not isinstance(result, bool) else None
+
+            old_keys, new_keys = [key(item) for item in prior], [key(item) for item in value]
+            old = {k: item for k, item in zip(old_keys, prior)
+                   if k is not None and old_keys.count(k) == 1}
+            values[name] = [
+                _retain_omitted(item, old[k], identities, field_path)
+                if isinstance(item, BaseModel) and k in old and new_keys.count(k) == 1
+                else item.model_dump(mode="json") if isinstance(item, BaseModel) else item
+                for item, k in zip(value, new_keys)]
+    return values
+
+
 def rewrite_story_source(raw_fields, candidate, slot_fn, *, schema, receipts,
                          pass_id, post_validator=None, slot_scheduler=None,
                          configured_model_id=None, instruction="", author_context=None,
-                         max_attempts=SOURCE_REWRITE_ATTEMPTS):
+                         max_attempts=SOURCE_REWRITE_ATTEMPTS, preserve_omitted=None):
     """Return (usable correction or None, receipt), with TWO calls at most.
 
     A pass id names one episode-local operation, not a revision counter.
     Re-entry cannot reset its budget, even with a changed draft. The caller
@@ -97,8 +140,23 @@ def rewrite_story_source(raw_fields, candidate, slot_fn, *, schema, receipts,
     attempt_limit = min(SOURCE_REWRITE_ATTEMPTS, max_attempts)
     raw = _raw_values(raw_fields)
     documents = build_raw_documents(raw)
     prior = next((row for row in receipts if row.get("pass_id") == pass_id), None)
+    # Opt-in only for full artifacts. Spoken edits have a different response
+    # shape from their candidate, and must never inherit a draft's fields.
+    original = json.loads(_json(candidate)) if preserve_omitted is not None else None
+    accepted = None
+
+    def validate_artifact(model):
+        nonlocal accepted
+        accepted = None
+        corrected = (schema.model_validate(_retain_omitted(model, original, preserve_omitted))
+                     if original is not None else model)
+        error = post_validator(corrected) if post_validator is not None else None
+        if error is None:
+            accepted = corrected
+        return error
+
     receipt = {
         "version": SOURCE_REWRITE_VERSION, "pass_id": pass_id,
         "operation_id": "source_rewrite_%d" % (len(receipts) + 1),
         "coordinate_version": RAW_COORDINATE_VERSION,
@@ -180,9 +238,9 @@ def rewrite_story_source(raw_fields, candidate, slot_fn, *, schema, receipts,
         with context:
             # LLM slot: creative/technical -- the caller supplies the artifact's author owner.
             corrected = structured_call(
                 prompt=prompt, schema=schema, slot_fn=observed,
-                post_validator=post_validator, base_temperature=0.35,
+                post_validator=validate_artifact, base_temperature=0.35,
                 structural_retry_temperature=0.15, repair_prompt_factory=_complete_repair,
                 max_attempts=attempt_limit, max_new_tokens=None,
                 helper_name=helper, on_attempt_complete=completed)
     except StructuredCallFailedError as error:
@@ -199,10 +257,12 @@ def rewrite_story_source(raw_fields, candidate, slot_fn, *, schema, receipts,
         return None, receipt
     except BaseException as error:
         receipt.update(status="provider_error", error_type=type(error).__name__, error=str(error))
         raise
-    receipt.update(status="usable", returned_artifact=corrected.model_dump(mode="json"))
-    return corrected, receipt
+    # The captured object is exactly what the structural owner validated,
+    # including any authorized normalization. A failed attempt cannot leak it.
+    receipt.update(status="usable", returned_artifact=accepted.model_dump(mode="json"))
+    return accepted, receipt
 
 
 class SpokenSourceEdit(BaseModel):
     model_config = ConfigDict(extra="forbid")
diff --git a/nodes/cast_lock.py b/nodes/cast_lock.py
index 90e6b157..81739003 100644
--- a/nodes/cast_lock.py
+++ b/nodes/cast_lock.py
@@ -1172,16 +1172,13 @@ class CastLock:
             # validation until that branch was ripped on 2026-08-18; the scorer
             # is now the only consumer.)
             from ._otr_roster_gender import canonical_bank_gender
             gender = canonical_bank_gender(entry.get("gender"))
-            if not gender:
-                if target_engine == "google_tts":
-                    raise VoiceCastingError(
-                        f"{char_id}: google_tts character casting needs a cast "
-                        f"gender to choose a gender-plausible provider voice. "
-                        f"NO FALLBACK.")
-                report.append(f"  {char_id}: no gender -- preserved (not re-cast)")
-                continue
+            if not gender and target_engine == "google_tts":
+                raise VoiceCastingError(
+                    f"{char_id}: google_tts character casting needs a cast "
+                    f"gender to choose a gender-plausible provider voice. "
+                    f"NO FALLBACK.")
             # THE HYBRID LLM VOICE-FIT BRANCH WAS HERE AND IS GONE (2026-08-18).
             # It read meta.voice_cast_decision, re-validated the LLM's proposed
             # voice_ref_id, and on success stamped it and `continue`d -- skipping
             # the deterministic scorer below entirely. That is why the scorer
@@ -1201,8 +1198,13 @@ class CastLock:
             slot = voice_slots.get(char_id) or {}
             slot_timbre = slot.get("timbre") or entry.get("timbre") or ()
             slot_age = str(slot.get("age_band") or entry.get("age_band") or "")
             try:
+                if not gender:
+                    # Cast the same real open-pool reference that the renderer
+                    # would select, so the wire, ledger and credits name it.
+                    # This does not invent a gender for the character.
+                    raise VoiceCastingError(f"{char_id}: source gender unspecified")
                 ref = assign_voice_for_slot(
                     role="char_voice",
                     engine=target_engine,
                     char_id=char_id,
diff --git a/nodes/otr_meta_brief_image_prompt.py b/nodes/otr_meta_brief_image_prompt.py
index b9a6bc59..42b9f532 100644
--- a/nodes/otr_meta_brief_image_prompt.py
+++ b/nodes/otr_meta_brief_image_prompt.py
@@ -1594,9 +1594,9 @@ def _build_char_prompt_request(char: dict, meta: dict, setting: str,
     )
 
 
 def _build_char_scene_request(char: dict, meta: dict, setting: str,
-                              line: dict, style=None) -> str:
+                              line: dict, style=None, *, source_scene=False) -> str:
     """BUG 1 follow-up (2026-06-20 operator): the per-beat character still must be
     SHOT/BEAT AWARE -- the character IN the moment of THIS beat -- regardless of
     image model (the video lane conditions on the SAME still). Mirrors
     :func:`_build_char_prompt_request` but WIDE 16:9 and grounded in the beat's
@@ -1617,16 +1617,26 @@ def _build_char_scene_request(char: dict, meta: dict, setting: str,
     ln = line if isinstance(line, dict) else {}
     intent = str(ln.get("beat_intent") or "").strip()[:240]
     mood = str(ln.get("traits") or "").strip()[:80]
     said = str(ln.get("text") or "").strip()[:240]
+    scene_direction = (
+        "Compose the CURRENT physical scene: its participants with their current ages, "
+        "shared actions, and the objects those actions require. The active speaker is "
+        "the focus within that scene, not its entire cast. Preserve source-required "
+        "companions in the composition. Spoken memories, metaphors and people only "
+        "mentioned do not become visible people or flashbacks; keep the present scene "
+        "unless the source explicitly changes it. Describe adults as adults even when "
+        "their relationship is son or daughter. Convey the current pose and emotion"
+        if source_scene else
+        "convey the ACTION and EMOTION of this beat. Translate the beat into what is "
+        "VISIBLE (pose, expression, what they are doing)")
     return (
         "Write ONE vivid cinematic STILL-image prompt (a single comma-separated "
         "line, no preamble) for a 16:9 LANDSCAPE shot of this character at THIS "
         "moment of the scene. The image MUST show the CHARACTER THEMSELVES -- a "
         "person with a clearly visible face -- as the subject, a medium/wide shot "
-        "with the full head and headroom, framed inside the story's world; convey "
-        "the ACTION and EMOTION of this beat. Translate the beat into what is "
-        "VISIBLE (pose, expression, what they are doing) -- do NOT write the "
+        "with the full head and headroom, framed inside the story's world; "
+        f"{scene_direction} -- do NOT write the "
         "character's name, dialogue, narration, or any on-screen text. NEVER an "
         "empty room, an object, or scenery alone.\n"
         f"character_appearance: {appearance or '(unspecified)'}\n"
         f"beat_action: {intent or '(unspecified)'}\n"
@@ -1683,28 +1693,42 @@ def _scene_source_context(meta, cast, lines, target, line, ledger_context=None):
     if not ordered_lines and line:
         ordered_lines = [scene_line(line)]
     speakers = {str(row.get("char_id") or "") for row in ordered_lines}
     cid = str(target.get("char_id") or "")
+    treatment = (meta.get("my_story") or {}).get("treatment") or {}
+    planned_cast = [row for row in treatment.get("cast", []) if isinstance(row, dict)]
+
+    def character_context(row):
+        # Ledger rows do not retain all treatment casting fields. Join only a
+        # unique exact normalized name; never infer an age or gender from prose.
+        name = str(row.get("name") or "")
+        key = " ".join(name.split()).casefold()
+        matches = [item for item in planned_cast
+                   if key and " ".join(str(item.get("name") or "").split()).casefold() == key]
+        planned = matches[0] if len(matches) == 1 else {}
+        return {"char_id": str(row.get("char_id") or ""), "name": name,
+                "appearance": _appearance_for_char([row], str(row.get("char_id") or "")),
+                "age_band": row.get("age_band") or planned.get("age_band") or "",
+                "gender": row.get("gender") or planned.get("gender") or ""}
+
     companions = [
-        {"char_id": str(row.get("char_id") or ""), "name": str(row.get("name") or ""),
-         "appearance": _appearance_for_char([row], str(row.get("char_id") or "")),
+        {**character_context(row),
          "speaks_in_scene": str(row.get("char_id") or "") in speakers}
         for row in cast if isinstance(row, dict) and row.get("char_id")
         and str(row.get("char_id")) != cid
         and str(row.get("name") or "").strip().upper() != "ANNOUNCER"
         and not row.get("_synthetic_announcer")
     ]
     context = {
+        "prompt_contract": "my_story.scene_source.v2",
         "scope": "scene_character", "beat_id": bid, "target_char_id": cid,
         "resolved_setting": _read_setting(meta),
-        "target_character": next(({
-            "char_id": cid, "name": str(row.get("name") or ""),
-            "appearance": _appearance_for_char([row], cid)}
+        "target_character": next((character_context(row)
             for row in cast if isinstance(row, dict) and str(row.get("char_id") or "") == cid), {}),
         "beat": dict(beat), "current_line": scene_line(line), "shot": dict(shot),
         "scene": dict(scene), "ordered_scene_dialogue": ordered_lines,
         "candidate_companions": companions,
-        "working_treatment": (meta.get("my_story") or {}).get("treatment"),
+        "working_treatment": treatment,
     }
     # A primitive copy prevents later pipeline mutation from changing the receipt.
     return json.loads(json.dumps({"raw_fields": raw, "scene": context}, ensure_ascii=False))
 
@@ -1719,9 +1743,9 @@ def _rewrite_char_scene_from_source(meta, ce, setting, line, warnings, cid, *,
         meta, kind="scene_character", role="character_video", char_entry=ce, style=vstyle)
     candidate = {"prompt": initial}
     # The shared operation supplies source and scene context once. Do not repeat
     # the full source inside visual_request and artificially consume its capacity.
-    request = _build_char_scene_request(ce, meta, setting, line, style=vstyle)
+    request = _build_char_scene_request(ce, meta, setting, line, style=vstyle, source_scene=True)
     context = dict(source_context["scene"])
     context["visual_request"] = request
     context_hash = candidate_sha256(source_context)
     journal = source_receipts if source_receipts is not None else []
@@ -1739,10 +1763,16 @@ def _rewrite_char_scene_from_source(meta, ce, setting, line, warnings, cid, *,
                 post_validator=validate, configured_model_id=source_model_id,
                 max_attempts=min(2, max(0, int(max_reseed)) + 1),
                 instruction=("This artifact is a scene still prompt. Return JSON with only the "
                              "prompt field. Apply source corrections directly, including required "
-                             "companions in this moment; preserve the target face and compatible "
-                             "visual elaboration. Do not force every act speaker into every frame. "
+                             "companions in this moment, their current ages, shared physical "
+                             "action and its necessary objects. A draft centered on one speaker "
+                             "may have omitted the rest of the required scene; restore it. "
+                             "Use explicit current-age descriptions for relatives, so an adult "
+                             "son or daughter does not become a child. Depict the present action, "
+                             "not a childhood memory spoken about during it. Preserve the target "
+                             "face and compatible visual elaboration. "
+                             "Do not force every act speaker into every frame. "
                              "The visual_request supplies framing and style instructions; its "
                              "request for a plain line is superseded by this JSON contract."),
                 author_context=context)
         finally:
diff --git a/nodes/story_packs/my_story/my_story.json b/nodes/story_packs/my_story/my_story.json
index fc02d985..6ad6d4ac 100644
--- a/nodes/story_packs/my_story/my_story.json
+++ b/nodes/story_packs/my_story/my_story.json
@@ -5,10 +5,10 @@
   "label": "My Story (a listener's own idea)",
   "status": "live",
   "schema_version": "v2.0",
   "prompt_stages": {
-    "my_story_interpret_system": "You read a person's rough story idea and work out what they actually want. Return one JSON object only -- no prose, no fences.\n\nThey wrote in their own words. It may be one sentence, a page of notes, or fragments with typos. Your job is to understand it, not to grade it.\n\nSchema:\n{\n  \"requirements\": array of objects, each:\n    { \"id\": short slug;\n      \"text\": the requirement in one plain sentence;\n      \"kind\": a descriptive category such as \"cast\", \"setting\" or \"event\";\n      \"source_field\": one of \"idea\", \"characters\", \"plot\", \"setting\";\n      \"strength\": \"required\" or \"preferred\" },\n  \"named_cast\": array of objects, each:\n    { \"name\": the name exactly as they wrote it;\n      \"notes\": what they said about this person, or \"\";\n      \"stated_gender\": their stated gender, or \"\" when unstated;\n      \"speaking\": true if this person should have lines;\n      \"required\": true if they clearly want this person in the story },\n  \"cast_plan\": { \"requested\": integer, the count they asked for;\n                 \"planned\": integer, the speaking cast you recommend;\n                 \"exclusive\": true if they said ONLY these people;\n                 \"reason\": one sentence explaining planned },\n  \"setting_brief\": one or two sentences describing where and when, or \"\",\n  \"assumptions\": array of strings; material things you had to decide because they did not say,\n  \"conflicts\": array of objects, each:\n    { \"requirement_id\": the id above;\n      \"why\": why it cannot be honoured as written;\n      \"resolution\": what the story will do instead }\n}\n\nHow to read them:\n- A REQUIREMENT is something they are asking for. An incidental mention is not. \"my sister loves lighthouses, anyway the story is about a diver\" names a sister who is not in the story.\n- Examples, brainstorming alternatives and abandoned ideas are not requirements. \"maybe a train, or a ship -- go with the ship\" requires a ship.\n- Fix obvious typos silently. Do not turn a typo into a character.\n- NEVER infer gender from a name. Fill stated_gender only when the text says it.\n- Every noun is not a speaker. Someone who is talked about is not automatically someone who talks.\n- If they demand something the form cannot do, record it in conflicts with what the story will do instead. Never drop it silently.\n- The selected act count is binding. The requested character count is flexible guidance: preserve the people the listener described, including an exclusive named cast. Record any conflicts honestly.\n- Write assumptions for choices that matter -- an era, a relationship, an ending -- not for every unstated detail.\n\nInterpret generously. This is someone's idea, and it is your job to find the story in it.\n",
-    "my_story_treatment_system": "You are a radio dramatist. You are handed a person's story idea, already interpreted, and you turn it into the plan for tonight's episode. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"title\": the episode title, authored, no quotation marks,\n  \"logline\": one sentence,\n  \"dramatic_question\": the single question the episode answers, one sentence,\n  \"setting\": concrete place,\n  \"time_of_day\": e.g. \"midnight\", \"the morning after\",\n  \"cast\": array of objects, each:\n    { \"name\": the character's name;\n      \"role\": their part in the story, a few words;\n      \"character_description\": one or two sentences a casting director could use;\n      \"gender\": their stated gender, or \"\" when unstated;\n      \"age_band\": one of \"20s\", \"30s\", \"40s\", \"50s\", \"60s\", \"n/a\";\n      \"register\": how they speak, a few words;\n      \"timbre\": their voice in two or three words },\n  \"acts\": array of objects, one per selected act, each:\n    { \"n\": act number starting at 1;\n      \"purpose\": what this act accomplishes;\n      \"scene_setting\": where this act happens;\n      \"turns\": array of short strings, the beats of the act;\n      \"ending_state\": where the story stands when the act ends },\n  \"ending\": how it ends, one or two sentences\n}\n\nRules:\n- THE PERSON'S REQUIREMENTS OUTRANK YOUR INVENTION. Preserve their material and names within the selected acts. The requested character count is flexible guidance; let their story determine the speaking cast.\n- Where they stated a gender, use it. Where they did not, leave gender empty; never infer it from a name.\n- Fill what they left open. Unspecified details are yours to invent, and inventing them well is the job.\n- The cast array contains the story's speaking characters, excluding ANNOUNCER. Preserve named people and exclusive cast notes; do not drop someone or invent extra people solely to match the requested character count. ANNOUNCER is reserved for the frame.\n- Produce exactly the selected number of acts, numbered 1..N in order. Reorganize the events into those acts; do not remove the ending to make the count fit.\n- Radio: the audience only hears. Give every character a distinguishable voice and a reason to speak.\n- Write the story they asked for, not the one you would have chosen.\n\nDescriptive metadata and a title may be omitted when unavailable; the existing downstream producers can supply missing presentation details.\n",
+    "my_story_interpret_system": "You read a person's rough story idea and work out what they actually want. Return one JSON object only -- no prose, no fences.\n\nThey wrote in their own words. It may be one sentence, a page of notes, or fragments with typos. Your job is to understand it, not to grade it.\n\nSchema:\n{\n  \"requirements\": array of objects, each:\n    { \"id\": short slug;\n      \"text\": the requirement in one plain sentence;\n      \"kind\": a descriptive category such as \"cast\", \"setting\" or \"event\";\n      \"source_field\": one of \"idea\", \"characters\", \"plot\", \"setting\";\n      \"strength\": \"required\" or \"preferred\" },\n  \"named_cast\": array of objects, each:\n    { \"name\": the name exactly as they wrote it;\n      \"notes\": what they said about this person, or \"\";\n      \"stated_gender\": their stated gender, or \"\" when unstated;\n      \"speaking\": true if this person should have lines;\n      \"required\": true if they clearly want this person in the story },\n  \"cast_plan\": { \"requested\": integer, the count they asked for;\n                 \"planned\": integer, the speaking cast you recommend;\n                 \"exclusive\": true if they said ONLY these people;\n                 \"reason\": one sentence explaining planned },\n  \"setting_brief\": one or two sentences describing where and when, or \"\",\n  \"assumptions\": array of strings; material things you had to decide because they did not say,\n  \"conflicts\": array of objects, each:\n    { \"requirement_id\": the id above;\n      \"why\": why it cannot be honoured as written;\n      \"resolution\": what the story will do instead }\n}\n\nHow to read them:\n- A REQUIREMENT is something they are asking for. An incidental mention is not. \"my sister loves lighthouses, anyway the story is about a diver\" names a sister who is not in the story.\n- Examples, brainstorming alternatives and abandoned ideas are not requirements. \"maybe a train, or a ship -- go with the ship\" requires a ship.\n- Fix obvious typos silently. Do not turn a typo into a character.\n- Resolve gender from the source's descriptions, relationships and pronouns in context. A source calling someone a woman, mother or son conveys information; an explicit identity takes precedence over conventional role wording. NEVER infer gender from a name. Leave genuinely unspecified gender empty; do not label conveyed information as an assumption.\n- Every noun is not a speaker. Someone who is talked about is not automatically someone who talks.\n- If they demand something the form cannot do, record it in conflicts with what the story will do instead. Never drop it silently.\n- The selected act count is binding. The requested character count is flexible guidance: preserve the people the listener described, including an exclusive named cast. Record any conflicts honestly.\n- Write assumptions for choices that matter -- an era, a relationship, an ending -- not for every unstated detail.\n\nInterpret generously. This is someone's idea, and it is your job to find the story in it.\n",
+    "my_story_treatment_system": "You are a radio dramatist. You are handed a person's story idea, already interpreted, and you turn it into the plan for tonight's episode. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"title\": the episode title, authored, no quotation marks,\n  \"logline\": one sentence,\n  \"dramatic_question\": the single question the episode answers, one sentence,\n  \"setting\": concrete place,\n  \"time_of_day\": e.g. \"midnight\", \"the morning after\",\n  \"cast\": array of objects, each:\n    { \"name\": the character's name;\n      \"role\": their part in the story, a few words;\n      \"character_description\": one or two sentences a casting director could use;\n      \"gender\": their stated gender, or \"\" when unstated;\n      \"age_band\": one of \"20s\", \"30s\", \"40s\", \"50s\", \"60s\", \"n/a\";\n      \"register\": how they speak, a few words;\n      \"timbre\": their voice in two or three words },\n  \"acts\": array of objects, one per selected act, each:\n    { \"n\": act number starting at 1;\n      \"purpose\": what this act accomplishes;\n      \"scene_setting\": where this act happens;\n      \"turns\": array of short strings, the beats of the act;\n      \"ending_state\": where the story stands when the act ends },\n  \"ending\": how it ends, one or two sentences\n}\n\nRules:\n- THE PERSON'S REQUIREMENTS OUTRANK YOUR INVENTION. Preserve their material and names within the selected acts. The requested character count is flexible guidance; let their story determine the speaking cast.\n- Preserve gender conveyed by source descriptions, relationships and pronouns in context. Honor explicit identity first, including when it differs from conventional role wording. Keep each character's gender consistent with their own casting description; do not describe a woman or man while recording that gender as unknown. Never infer gender from a name. Where the source is genuinely unspecified, leave gender empty and keep the description unspecified too.\n- Fill what they left open. Unspecified details are yours to invent, and inventing them well is the job.\n- The cast array contains the story's speaking characters, excluding ANNOUNCER. Preserve named people and exclusive cast notes; do not drop someone or invent extra people solely to match the requested character count. ANNOUNCER is reserved for the frame.\n- Produce exactly the selected number of acts, numbered 1..N in order. Reorganize the events into those acts; do not remove the ending to make the count fit.\n- Radio: the audience only hears. Give every character a distinguishable voice and a reason to speak.\n- Write the story they asked for, not the one you would have chosen.\n\nDescriptive metadata and a title may be omitted when unavailable; the existing downstream producers can supply missing presentation details.\n",
     "my_story_act_system": "You write one act of a radio drama as spoken dialogue. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"n\": this act's number,\n  \"scene_setting\": where this act happens,\n  \"lines\": array of objects, each:\n    { \"speaker\": an exact name from the cast list;\n      \"text\": the words that character says out loud }\n}\n\nRules:\n- Only what is SPOKEN. No narration, no stage directions, no parentheticals, no sound-effect notes, no speaker labels inside the text.\n- Use only the exact cast names you are given. Never invent a character.\n- Everything the audience needs to understand must be carried in what people say.\n- Continue from the previous act; do not restate it.\n- Unheard cast may enter in a later act. In the LAST act, every still-unheard member must receive actual spoken dialogue. Monologues within an act are valid.\n- Write the act to its own end. Let it run as long as the act needs.\n",
     "my_story_frame_system": "You write the announcer's frame around a radio drama -- the open, the close, and the music cues. Return one JSON object only -- no prose, no fences.\n\nSchema:\n{\n  \"announcer_intro\": array of strings (may be empty), what the announcer says before the story,\n  \"announcer_outro\": array of strings (may be empty), what the announcer says after it,\n  \"coda\": one closing line,\n  \"music_open\": a description of the opening theme, for a composer,\n  \"music_close\": a description of the closing theme,\n  \"music_inter\": array of strings, one interstitial cue between each pair of acts (empty when none are wanted)\n}\n\nRules:\n- You will be given an ATTRIBUTION SENTENCE. Include it VERBATIM, word for word, in the intro (preferred) or the outro. Do not paraphrase it, do not re-order it, and do not change the name in it.\n- Introduce the story without giving away its ending.\n- The announcer speaks to a listening audience. Warm, plain, unhurried.\n- Music cues describe MOOD and INSTRUMENTATION for a composer. They are never spoken aloud.\n- Propose music for the requested boundaries. Missing prompts can be supplied by the existing composer; extra proposals are recorded but not placed. Optional frame text and music descriptions may be empty.\n"
   },
   "examples": [],
diff --git a/tests/test_cast_lock.py b/tests/test_cast_lock.py
index c57ee3be..14dec0e9 100644
--- a/tests/test_cast_lock.py
+++ b/tests/test_cast_lock.py
@@ -287,16 +287,19 @@ def test_auto_registry_is_deterministic():
     b = CastLock().lock(script_json=_ledger(_CHAR_CAST), cast_voice_policy="auto_registry")[0]
     assert a == b
 
 
-def test_auto_registry_skips_genderless_character():
+def test_auto_registry_stamps_genderless_character_without_inventing_gender():
     from nodes.cast_lock import CastLock
 
     cast = [{"char_id": "c1", "name": "MYSTERY", "voice_preset": "v2/en_speaker_2"}]
     out = CastLock().lock(script_json=_ledger(cast), cast_voice_policy="auto_registry")
     led = json.loads(out[0])
-    assert "voice_ref_id" not in led["cast"][0]  # preserved, not cast
-    assert "no gender" in out[2]
+    row = led["cast"][0]
+    assert row["voice_ref_id"] and row["voice_engine"] == "indextts2"
+    assert not row.get("gender")
+    assert row["voice_cast_fallback"] == "gender_unservable"
+    assert "gender-agnostic reference" in out[2]
 
 
 def test_auto_registry_bark_legacy_preserves_characters():
     from nodes.cast_lock import CastLock
diff --git a/tests/test_cast_lock_voice_ref_completeness.py b/tests/test_cast_lock_voice_ref_completeness.py
index 2b50efc0..dd13b2c6 100644
--- a/tests/test_cast_lock_voice_ref_completeness.py
+++ b/tests/test_cast_lock_voice_ref_completeness.py
@@ -111,8 +111,29 @@ def test_voice_cast_fallback_is_defined_on_every_row_it_considers():
     assert cast["c1"]["voice_cast_fallback"] == ""
     assert cast["c2"]["voice_cast_fallback"] == "gender_unservable"
 
 
+@pytest.mark.parametrize("gender", [None, "", "   "])
+def test_unspecified_gender_names_the_real_render_reference_without_changing_identity(gender):
+    from nodes.cast_lock import CastLock
+    from nodes import _otr_voice_node_common as vnc
+    cast = [{"char_id": "c1", "name": "Jeffrey", "gender": gender,
+             "voice_preset": "v2/en_speaker_1"},
+            {"char_id": "c2", "name": "Mother", "gender": gender,
+             "voice_preset": "v2/en_speaker_2"}]
+    out = CastLock().lock(script_json=_ledger(cast), cast_voice_policy="auto_registry")
+    rows = json.loads(out[0])["cast"]
+    assert rows[0]["voice_ref_id"] != rows[1]["voice_ref_id"]
+    entries, _ = load_voice_bank()
+    for row in rows:
+        assert row["gender"] == gender
+        assert row["voice_engine"] == "indextts2"
+        ref = next(e for e in entries if e.voice_ref_id == row["voice_ref_id"])
+        path = vnc._resolve_clone_ref_path(row["voice_engine"], row, 42)
+        assert path and path.endswith(ref.ref_path.replace("/", "\\").split("\\")[-1])
+        assert row["voice_cast_fallback"] == "gender_unservable"
+
+
 def test_the_unservable_row_is_not_refused_or_gender_restricted():
     """This is a LEDGER fix, not a content gate. The row still renders, and the
     roll that produced 'other' is untouched.
     """
diff --git a/tests/test_constrained_generate.py b/tests/test_constrained_generate.py
index 05ecd570..4a71c719 100644
--- a/tests/test_constrained_generate.py
+++ b/tests/test_constrained_generate.py
@@ -276,8 +276,19 @@ def _feed_json(prefix, text):
         ids.append(ord(char))
     return prefix(0, torch.tensor(ids))
 
 
+def test_scope_authorization_real_grammar_accepts_empty_or_omitted_spans_never_null():
+    from nodes._otr_ledger_clean import _ScopeAuthorization
+    for text in ('{"verdict":"already_spoken"}',
+                 '{"verdict":"already_spoken","spans":[]}'):
+        _, prefix = _real_constraint(_ScopeAuthorization)
+        assert 200 in _feed_json(prefix, text)
+    _, prefix = _real_constraint(_ScopeAuthorization)
+    allowed = _feed_json(prefix, '{"verdict":"already_spoken","spans":')
+    assert ord('[') in allowed and ord('n') not in allowed
+
+
 def test_real_lmfe_uses_shared_model_chat_eos_and_refreshes_without_mutating_history():
     from types import SimpleNamespace
     import torch
     from transformers import EosTokenCriteria
diff --git a/tests/test_credits_roll_spec.py b/tests/test_credits_roll_spec.py
index 570b47bf..d4ca2993 100644
--- a/tests/test_credits_roll_spec.py
+++ b/tests/test_credits_roll_spec.py
@@ -126,8 +126,39 @@ def _layout(**over):
                                        "total_target_frames": 400, "fps": 25,
                                        "clip_count": 3})
 
 
+@pytest.mark.parametrize("gender", [None, ""])
+def test_genderless_cast_wire_disk_render_and_credits_share_the_actual_reference(tmp_path, monkeypatch, gender):
+    from nodes import production_ledger as pl
+    from nodes.cast_lock import CastLock
+    from nodes import _otr_voice_node_common as vnc
+    saved = pl._CURRENT
+    try:
+        ledger = pl.new_ledger("credit_ref_contract", str(tmp_path))
+        data = _led()
+        data["meta"].pop("cast_contract", None)
+        data["cast"] = [{"char_id": "c01", "name": "Mother", "gender": gender,
+                         "voice_preset": "v2/en_speaker_2"}]
+        data["lines"] = []
+        ledger.data.update(data)
+        monkeypatch.setenv("OTR_TEST_MODE", "0")
+        result = CastLock().lock(script_json=json.dumps(data), cast_voice_policy="auto_registry")
+        wire = json.loads(result[0])
+        with open(ledger.path, encoding="utf-8") as handle:
+            durable = json.load(handle)
+        assert wire["cast"] == durable["cast"] == ledger.data["cast"]
+        row = durable["cast"][0]
+        assert row["gender"] == gender and row["voice_engine"] == "indextts2"
+        assert vnc._resolve_clone_ref_path(row["voice_engine"], row, 42)
+        layout = cr.build_credits_layout(durable, w=1920, h=1080, manifest={"clips": []})
+        assert layout["col2"]["cast_rows"][0]["line"] == (
+            "indextts2 · " + row["voice_ref_id"])
+    finally:
+        with pl._LEDGER_LOCK:
+            pl._CURRENT = saved
+
+
 def _flat(blocks):
     return json.dumps(blocks, default=str, ensure_ascii=False)
 
 
diff --git a/tests/test_ledger_clean_stage.py b/tests/test_ledger_clean_stage.py
index a24749dc..af4390d3 100644
--- a/tests/test_ledger_clean_stage.py
+++ b/tests/test_ledger_clean_stage.py
@@ -197,8 +197,45 @@ def test_genuine_whole_row_direction_remains_convertible_after_authorization():
     assert receipt["rows"][0]["scope"]["mode"] == "whole"
     assert slot.authorization_calls == 1 and slot.repair_calls == 1
 
 
+def test_scope_authorization_binds_once_and_reuses_bound_slot_for_repair():
+    original = "Until next time."
+    ledger = _ledger(original, bank="my_story")
+    slot = _Slot(judgements={original: [_dirty_judgement(original)]})
+    bindings, calls = [], []
+
+    def bind(schema):
+        bindings.append(schema)
+
+        def bound(messages, **kwargs):
+            calls.append(messages)
+            return json.dumps({"verdict": "already_spoken",
+                               "spans": None if len(calls) == 1 else []})
+        return bound
+
+    slot._otr_bind_schema = bind
+    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="my_story")
+    assert bindings == [lcl._ScopeAuthorization]
+    assert len(calls) == 2 and slot.authorization_calls == slot.repair_calls == 0
+    assert ledger["lines"][1]["text"] == original
+    assert "unclean_spoken_text" not in ledger["lines"][1].get("compose_flags", [])
+    assert receipt["rows"][0]["outcome"] == "already_spoken"
+
+
+def test_scope_binder_failure_propagates_without_unconstrained_fallback():
+    original = "Until next time."
+    slot = _Slot(judgements={original: [_dirty_judgement(original)]})
+
+    def bind(schema):
+        raise RuntimeError("schema owner unavailable")
+
+    slot._otr_bind_schema = bind
+    with pytest.raises(RuntimeError, match="schema owner unavailable"):
+        lcl.run_ledger_clean(_ledger(original), slot_fn=slot, bank_id="original")
+    assert slot.authorization_calls == slot.repair_calls == 0
+
+
 def test_whole_row_authorization_can_narrow_to_a_real_local_direction():
     original = "Stay here. (He sighs)"
     ledger = _ledger(original)
     slot = _Slot(judgements={original: [_dirty_judgement(original)]},
diff --git a/tests/test_my_story_runner.py b/tests/test_my_story_runner.py
index 29b08900..1d677f8a 100644
--- a/tests/test_my_story_runner.py
+++ b/tests/test_my_story_runner.py
@@ -1024,8 +1024,36 @@ def test_unusable_source_rewrites_stop_at_two_and_keep_a_usable_saved_ledger():
     assert len(slots.calls) == 12  # four author calls plus eight source attempts, no fourth/fifth retry
     validate_receipt(saved)
 
 
+def test_sparse_p0_source_reply_keeps_saved_metadata_and_records_unchanged():
+    from nodes._otr_content_authorship import validate_receipt
+
+    class Sparse(Slots):
+        def _answer(self, messages):
+            if messages[0]["content"].startswith("Check and rewrite"):
+                draft = json.loads(messages[1]["content"])["draft"]
+                if "setting_brief" in draft:
+                    return json.dumps({
+                        "requirements": [{"id": r["id"], "text": r["text"]}
+                                         for r in draft["requirements"]],
+                        "named_cast": [{"name": r["name"], "notes": r["notes"]}
+                                       for r in draft["named_cast"]]})
+            return super()._answer(messages)
+
+    slots = Sparse()
+    led, _ = _run(slots)
+    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
+    story = saved["meta"]["my_story"]
+    assert story["interpretation"] == MS.StoryInterpretation.model_validate(_interpretation()).model_dump()
+    receipt = story["source_rewrites"][0]
+    assert receipt["status"] == "unchanged" and not receipt["applied"]
+    assert receipt["input_sha256"] == receipt["output_sha256"]
+    assert receipt["returned_artifact"] == story["interpretation"]
+    assert len(receipt["attempts"]) == 1 and len(slots.calls) == 8
+    validate_receipt(saved)
+
+
 def test_source_failure_history_survives_a_provider_exception_after_a_checkpoint(tmp_path):
     class Failed(Slots):
         def _answer(self, messages):
             if messages[0]["content"].startswith("Check and rewrite"):
diff --git a/tests/test_my_story_visual_source.py b/tests/test_my_story_visual_source.py
index 7d5d02da..52b50cd0 100644
--- a/tests/test_my_story_visual_source.py
+++ b/tests/test_my_story_visual_source.py
@@ -141,8 +141,43 @@ def test_actual_shared_source_call_does_not_change_neutral_portrait_scope():
     portrait = mb._build_char_prompt_request(ledger["cast"][0], ledger["meta"], "kitchen")
     assert "CURRENT SCENE CONTEXT" not in portrait and "candidate_companions" not in portrait
 
 
+def test_scene_owner_receives_current_ages_and_shared_action_without_literalizing_memory():
+    ledger = _ledger()
+    ledger["meta"]["my_story"]["treatment"]["cast"] = [
+        {"name": "Ada", "age_band": "30s", "gender": "female"},
+        {"name": "Mother", "age_band": "50s", "gender": "female"},
+        {"name": "Tom", "age_band": "40s", "gender": "male"}]
+    ledger["lines"][0]["text"] = "Mother, remember when I was small? I love sharing dinner with you now."
+    before_hash = source.candidate_sha256(_context(ledger))
+    slot = Slot()
+    _compose(slot, ledger=ledger)
+    request = json.loads(slot.calls[0][1]["content"])
+    context = request["authoring_context"]
+    assert context["target_character"]["age_band"] == "30s"
+    assert context["candidate_companions"][0]["age_band"] == "50s"
+    visual_request = context["visual_request"]
+    assert "active speaker is the focus within that scene" in visual_request
+    assert "Spoken memories" in visual_request and "objects those actions require" in visual_request
+    assert "Do not force every act speaker into every frame" in slot.calls[0][0]["content"]
+    assert context["candidate_companions"][1]["speaks_in_scene"] is False
+    ledger["meta"]["my_story"]["treatment"]["cast"][0]["age_band"] = "40s"
+    assert source.candidate_sha256(_context(ledger)) != before_hash
+    portrait = mb._build_char_prompt_request(ledger["cast"][0], ledger["meta"], "kitchen")
+    assert "shared actions" not in portrait and "Spoken memories" not in portrait
+
+
+def test_scene_age_join_does_not_borrow_from_a_different_or_ambiguous_cast_name():
+    ledger = _ledger()
+    ledger["meta"]["my_story"]["treatment"]["cast"] = [
+        {"name": "Ada", "age_band": "30s"}, {"name": "ADA", "age_band": "60s"},
+        {"name": "Mother's friend", "age_band": "50s"}]
+    context = _context(ledger)["scene"]
+    assert context["target_character"]["age_band"] == ""
+    assert context["candidate_companions"][0]["age_band"] == ""
+
+
 def test_corrected_narrative_appearance_is_not_prepended_back_into_prompt():
     ledger = _ledger()
     ledger["cast"][0]["character_description"] = "round face, mourning her dead mother, waiting alone"
     (prompt, _), receipt = _compose(Slot(), ledger=ledger)
diff --git a/tests/test_story_source_review.py b/tests/test_story_source_review.py
index 9ac3f9c2..79677c0b 100644
--- a/tests/test_story_source_review.py
+++ b/tests/test_story_source_review.py
@@ -112,8 +112,106 @@ def test_existing_structural_validator_runs_on_the_exact_returned_model():
     assert result.people == ["MOTHER"]
     assert receipt["returned_artifact"]["people"] == ["MOTHER"]
 
 
+def _rewrite_interpretation(candidate, reply, **kwargs):
+    from nodes._otr_my_story import StoryInterpretation
+    slot = Slot(reply)
+    result, receipt = source.rewrite_story_source(
+        RawStoryFields(idea="Jeffrey and his mother share dinner."), candidate, slot,
+        schema=StoryInterpretation, receipts=[], pass_id="interpret",
+        preserve_omitted={("requirements",): "id", ("named_cast",): "name"}, **kwargs)
+    return result, receipt, slot
+
+
+def test_sparse_source_correction_conserves_metadata_by_identity_not_position():
+    from nodes._otr_my_story import StoryInterpretation
+    candidate = StoryInterpretation.model_validate({
+        "requirements": [{"id": "dinner", "text": "Share dinner", "kind": "event",
+                          "source_field": "plot", "strength": "preferred"},
+                         {"id": "place", "text": "In LA", "kind": "setting",
+                          "source_field": "setting"}],
+        "named_cast": [{"name": "Jeffrey", "stated_gender": "male"},
+                       {"name": "Mother", "notes": "Present throughout"}],
+        "assumptions": ["A warm evening"],
+    }).model_dump(mode="json")
+    before = copy.deepcopy(candidate)
+    reply = {"requirements": [{"id": "place", "text": "In LA"},
+                              {"id": "dinner", "text": "Share dinner"}],
+             "named_cast": [{"name": "Mother"}, {"name": "Jeffrey"}]}
+    result, receipt, slot = _rewrite_interpretation(candidate, reply)
+    assert result.requirements[1].source_field == "plot"
+    assert result.requirements[1].strength == "preferred"
+    assert result.requirements[0].kind == "setting"
+    assert result.named_cast[1].stated_gender == "male"
+    assert result.named_cast[0].notes == "Present throughout"
+    assert result.assumptions == candidate["assumptions"]
+    assert receipt["returned_artifact"] == result.model_dump(mode="json")
+    assert candidate == before and len(slot.calls) == 1
+
+
+def test_explicit_corrections_clears_and_list_membership_are_authoritative():
+    candidate = {"named_cast": [{"name": "Jeffrey", "stated_gender": "male", "notes": "old"},
+                                {"name": "Removed", "notes": "Do not resurrect"}],
+                 "requirements": [{"id": "old", "kind": "event"}],
+                 "assumptions": ["old"]}
+    reply = {"named_cast": [{"name": "Jeffrey", "stated_gender": "", "notes": "",
+                             "speaking": False, "required": False},
+                            {"name": "Added"}], "requirements": [], "assumptions": []}
+    result, _, _ = _rewrite_interpretation(candidate, reply)
+    assert [r.name for r in result.named_cast] == ["Jeffrey", "Added"]
+    assert result.named_cast[0].stated_gender == result.named_cast[0].notes == ""
+    assert not result.named_cast[0].speaking and not result.named_cast[0].required
+    assert not result.requirements and not result.assumptions
+
+
+@pytest.mark.parametrize("rows", [[{"name": "A"}, {"name": "A"}], [{}], [{"name": "Renamed"}]])
+def test_ambiguous_or_changed_identity_never_inherits_another_cast_member(rows):
+    result, _, _ = _rewrite_interpretation(
+        {"named_cast": [{"name": "A", "stated_gender": "female"}]}, {"named_cast": rows})
+    assert all(not row.stated_gender for row in result.named_cast)
+
+
+def test_rejected_proposal_is_not_the_next_baseline_and_validator_result_is_journaled():
+    from nodes._otr_my_story import StoryInterpretation
+    seen = []
+
+    def validate(model):
+        seen.append(model)
+        if model.setting_brief == "wrong":
+            return "wrong setting"
+        model.setting_brief = "normalized"
+
+    slot = Slot(lambda messages: {"setting_brief": "wrong", "assumptions": ["bad"]}
+                if len(slot.calls) == 1 else {"setting_brief": "right"})
+    result, receipt = source.rewrite_story_source(
+        RawStoryFields(idea="Dinner"), {"assumptions": ["original"]}, slot,
+        schema=StoryInterpretation, receipts=[], pass_id="interpret",
+        preserve_omitted={}, post_validator=validate)
+    assert len(slot.calls) == 2 and result is seen[-1]
+    assert result.assumptions == ["original"] and result.setting_brief == "normalized"
+    assert receipt["returned_artifact"] == result.model_dump(mode="json")
+
+
+@pytest.mark.parametrize("register,expected", [(None, "warm"), ("", ""), ("formal", "formal")])
+def test_treatment_alias_and_explicit_act_identity_preserve_the_correct_metadata(register, expected):
+    from nodes._otr_my_story import StoryTreatment
+    candidate = StoryTreatment.model_validate({
+        "cast": [{"name": "Jeffrey", "register": "warm"}],
+        "acts": [{"n": 1, "purpose": "dinner"}, {"n": 2, "purpose": "departure"}],
+    }).model_dump(mode="json")
+    member = {"name": "Jeffrey"}
+    if register is not None:
+        member["register"] = register
+    slot = Slot({"cast": [member], "acts": [{"n": 2}, {}]})
+    result, _ = source.rewrite_story_source(
+        RawStoryFields(idea="Dinner"), candidate, slot, schema=StoryTreatment,
+        receipts=[], pass_id="treatment", preserve_omitted={("cast",): "name", ("acts",): "n"})
+    assert result.cast[0].speech_register == expected
+    assert result.acts[0].purpose == "departure"
+    assert result.acts[1].purpose == ""  # default n=1 is not an explicit identity
+
+
 @pytest.mark.parametrize("error", [RuntimeError("provider failed"), MemoryError("real OOM"),
                                   KeyboardInterrupt("cancelled")])
 def test_real_failures_propagate_once_and_keep_attempt_evidence(error):
     def fail(messages):

```
