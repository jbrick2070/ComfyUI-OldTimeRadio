# Final bounded My Story repair-context QA

Read-only Sonnet review of the finished code diff. Root owns judgment and code.
Return <=700 words: actionable reachable must-fix, or no demonstrated must-fix;
state qualification limits. This is a narrow existing-owner correction, no new
architecture/schema/widget/graph. Do not demand another model pass or content gate.

## Live proof and grounded disposition

Full canonical03 on97f2d40f exhausted3P1 attempts at408.62seconds, no media.
All3raw_output="", while error.raw_completion retained2673/11581/1770chars.
No quotes after offset48: the model stayed inside logline, eventually repeating.
Native+grammar EOS aligned[248044,248046]; P0+its correction ended248046.
Actual tokenizer+LMFE CPU reconstruction from saved decoded text shows closing
quote token1 admitted after all3prefixes. This is NOT original token/logit replay.
Fresh parser pergenerate and unchanged binding are independently audited.

Concrete gap: sharedstructured_call intentionally passeslast_raw="" to its
typed repair when generation raises, with last_error carrying capturedtext.
My Story's own _full_artifact_repair ignored that error field and asked the
thirdcall to repair an empty assistant draft. The existingfullfailedartifact
PBUG20260910-01/Bible11.48 contract applies to this local owner.

Fix: nonempty returned failed_output wins exactly; otherwise string
error.raw_completion is supplied exactly; otherwise empty. No stripping,
truncation or stringify. Add interrupted-draft warning giving original source
authority over failed nonsense. Existing3attempts/source/binding/validators,
provider-capacity fit, failure receipts/counts unchanged. It reaches P0/P1/P2/P3
only through the existing local factory; generic sharedcaller policy untouched.

The new public-ladder test failed on base because thirdcall got emptydraft.
Afterfix the same ladder accepts a corrected treatment and durably saves it.
Two distinctfailedcompletions ensurelatestoneused, beyond400chars; sourcekept,
threecalls exact, failedfragmentsstillraw_completion notacceptedproposals.
One initialtestassertion expectedproposed_acts=None despiteacceptedfinalrepair;
correctedto1 (right result), previousall-raisedcountercontrolstillNone.
Finalfocused suite passed. Fullregression/Bible are running separately.

Root rejects prior panel overclaims: a single successful prompt/greedy run
doesn't prove causality; no-stringlengthgate is operatorrequired, not codebug;
seed capabilityalreadyexistsOTR_WRITER_SEED butproductionisintentionallyfresh;
no model/provider swap, no removal ofrawsource, no statsfromunfinishedfragments.
Metadataecho cast_plan.requested=3 is a model proposal beside actualrequested2,
not an actual castchange; don't widen this repair. Original degenerationcause
and reliable canonicalpublication remain unproven by this handoff repair.

## Finished diff

diff --git a/nodes/_otr_my_story.py b/nodes/_otr_my_story.py
index 8936529a..3e5c0e4f 100644
--- a/nodes/_otr_my_story.py
+++ b/nodes/_otr_my_story.py
@@ -417,17 +417,31 @@ def _call(pass_id: str, bundle: Any, *, attempt_receipts=None,
 
 
 def _full_artifact_repair(instruction: str):
-    """Give existing post-validation repair the entire parsed draft to revise.
+    """Give the existing repair the entire returned or interrupted draft.
 
     The generic repair's 400-character echo cannot show the end of a treatment
     or an act. Syntax and schema repair need that ending too. The same author
     attempt budget and structural validator still decide acceptance.
     """
     def repair(*, original_prompt, failed_output, error):
+        draft = failed_output
+        # A halted generation raises before the shared ladder assigns its
+        # return value. Its complete text belongs to the error instead. This
+        # lane's repair needs that evidence without treating it as an accepted
+        # proposal or changing the shared ladder's policy for other callers.
+        interrupted = False
+        if not draft:
+            completion = getattr(error, "raw_completion", None)
+            if isinstance(completion, str):
+                draft = completion
+                interrupted = bool(completion)
         return [
             *[dict(message) for message in original_prompt],
-            {"role": "assistant", "content": failed_output},
+            {"role": "assistant", "content": draft},
             {"role": "user", "content": (
+                ("The draft was interrupted during generation. Its repeated or "
+                 "unfinished text is failure evidence, not authority over the "
+                 "original source.\n" if interrupted else "") +
                 "Repair the complete draft above. %s\n"
                 "The validation problem is: %s\n"
                 "Preserve unaffected story events, relationships and ending; "
diff --git a/tests/test_my_story_runner.py b/tests/test_my_story_runner.py
index 1d677f8a..fc89ff19 100644
--- a/tests/test_my_story_runner.py
+++ b/tests/test_my_story_runner.py
@@ -975,6 +975,67 @@ def test_raised_completion_is_durable_evidence_not_a_completed_proposal(tmp_path
     assert story["counts"]["actual_acts"] is None
 
 
+def test_halted_treatment_repair_receives_the_latest_full_completion_and_rewrites():
+    from nodes._otr_generation_budget import GenerationDegeneracyError
+    from nodes._otr_content_authorship import validate_receipt
+    first = '{"title":"Harbor","logline":"' + "first interrupted fragment " * 30
+    last = '{"title":"Harbor","logline":"' + "second interrupted fragment " * 30
+    last += "THE FERRY TURNS AWAY"
+
+    class HaltThenRepair(Slots):
+        treatment_calls = 0
+
+        def _answer(self, messages):
+            if "radio dramatist" in messages[0]["content"]:
+                self.treatment_calls += 1
+                if self.treatment_calls < 3:
+                    raise GenerationDegeneracyError(
+                        "the output repeated a run of tokens verbatim",
+                        halt_reason="verbatim_cycle",
+                        raw_completion=first if self.treatment_calls == 1 else last,
+                    )
+                assert self.treatment_calls == 3
+                prior = [m for m in messages if m["role"] == "assistant"]
+                assert prior[-1]["content"] == last
+                assert prior[-1]["content"].index("THE FERRY TURNS AWAY") > 400
+                assert "a keeper hears a voice" in messages[1]["content"]
+                result = _treatment()
+                result["ending"] = "THE FERRY TURNS AWAY"
+                return json.dumps(result)
+            return super()._answer(messages)
+
+    slots = HaltThenRepair()
+    led, _ = _run(slots)
+    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
+    story = saved["meta"]["my_story"]
+    attempts = [a for a in story["attempts"] if a["pass_id"] == "treatment"]
+    assert slots.treatment_calls == 3
+    assert [a["status"] for a in attempts] == ["failed", "failed", "accepted"]
+    assert [a["raw_completion"] for a in attempts[:2]] == [first, last]
+    assert all(a["raw_output"] == "" for a in attempts[:2])
+    assert story["counts"]["proposed_acts"] == 1  # the returned repair, not either raised fragment
+    assert story["counts"]["actual_acts"] == 1
+    assert story["treatment"]["ending"] == "THE FERRY TURNS AWAY"
+    validate_receipt(saved)
+
+
+@pytest.mark.parametrize("returned,completion,expected", [
+    ('{"returned":true}', '{"older":true}', '{"returned":true}'),
+    ("", "  complete interrupted bytes\n", "  complete interrupted bytes\n"),
+    ("", None, ""),
+    ("", {"not": "text"}, ""),
+])
+def test_full_repair_keeps_returned_text_precedence_and_never_invents_a_draft(returned, completion, expected):
+    error = ValueError("failed")
+    error.raw_completion = completion
+    messages = MS._full_artifact_repair("Repair the treatment.")(
+        original_prompt=[{"role": "user", "content": "original source"}],
+        failed_output=returned, error=error,
+    )
+    assert messages[-2] == {"role": "assistant", "content": expected}
+    assert messages[0]["content"] == "original source"
+
+
 def test_combined_corrections_reach_the_saved_artifacts_and_spoken_ledger():
     from nodes._otr_content_authorship import validate_receipt
 

## Exact shared ladder repair handoff (unchanged)

998:     # SYNTAX failure. A re-prompt at lower temperature can shake loose a
999:     # parseable object when the model emitted malformed JSON. It does NOT help a
1000:     # ValidationError / PostValidationError (the JSON parsed; the SHAPE or
1001:     # CONTENT is wrong) -- a re-prompt just re-emits the same shape, burning a
1002:     # credit-billed call (the 2026-06-25 Opus normalize_length exhaustion: the
1003:     # structural rung never helped, it only spent tokens). So on a non-syntax
1004:     # failure skip straight to the typed repair; spend the structural retry only
1005:     # on json.JSONDecodeError. attempts_run advances ONLY when this branch runs.
1006:     # A-4: an `output_limit` capacity failure joins the syntax failure here.
1007:     # It is the SAME remedy for the same shape of problem -- the model did not
1008:     # finish, and the same prompt at a lower temperature is a real second
1009:     # chance rather than a re-emission of an identical wrong shape. It is NOT
1010:     # given to the typed repair for the reason the rung's own comment gives:
1011:     # `last_raw` is bound to "" before every call and only rebound when the
1012:     # call RETURNS, so a capacity raise leaves no artifact to repair, and the
1013:     # completion A-1 attached to the exception is deliberately not fed into a
1014:     # repair prompt (bounding that is a separate, unratified change).
1015:     if attempts_run < max_attempts and (
1016:         isinstance(last_error, json.JSONDecodeError)
1017:         or is_rerollable_capacity_error(last_error)
1018:     ):
1019:         attempts_run += 1
1020:         log.info(
1021:             "[OTR_StructuredCall] '%s' attempt %d/%d: structural retry at "
1022:             "temperature=%.3f (lowered from %.3f)",
1023:             helper_name, attempts_run, max_attempts,
1024:             structural_retry_temperature, base_temperature,
1025:         )
1026:         try:
1027:             last_raw = ""
1028:             last_raw = _invoke_slot(
1029:                 slot_fn, base_messages,
1030:                 temperature=structural_retry_temperature,
1031:                 max_new_tokens=max_new_tokens,
1032:                 force_json_object=text_parser is None,
1033:             )
1034:             result = _parse_and_validate(
1035:                 last_raw,
1036:                 schema,
1037:                 post_validator,
1038:                 text_parser,
1039:             )
1040:             notify_attempt(None)
1041:             return result
1042:         except _ATTEMPT_ERRORS as exc:
1043:             if not _attempt_is_retryable(exc):
1044:                 notify_attempt(exc)
1045:                 raise
1046:             last_error = exc
1047:             notify_attempt(exc)
1048:             log.warning(
1049:                 "[OTR_StructuredCall] '%s' attempt %d failed: %s | raw "
1050:                 "head: %s", helper_name, attempts_run, exc,
1051:                 _raw_head(last_raw, error=exc),
1052:             )
1053:         except Exception as exc:
1054:             notify_attempt(exc)
1055:             raise
1056: 
1057:     # --- Typed repair at a static low temperature (the final rung). ---
1058:     if attempts_run < max_attempts:
1059:         attempts_run += 1
1060:         log.info(
1061:             "[OTR_StructuredCall] '%s' attempt %d/%d: typed repair at "
1062:             "temperature=%.3f",
1063:             helper_name, attempts_run, max_attempts, _REPAIR_TEMPERATURE,
1064:         )
1065:         try:
1066:             repair_error: BaseException = (
1067:                 last_error
1068:                 if last_error is not None
1069:                 else ValueError("no prior error captured")
1070:             )
1071:             repair_prompt = factory(
1072:                 original_prompt=contract_prompt,
1073:                 failed_output=last_raw,
1074:                 error=repair_error,
1075:             )
1076:             # A typed repair factory MAY resolve the failure itself --
1077:             # e.g. cast_membership_repair remapping a phantom speaker to
1078:             # a locked-cast member via Levenshtein -- and hand back a
1079:             # finished `schema` instance instead of a repair prompt.
1080:             # Accept it directly: no LLM repair call is made. The
1081:             # instance still passes through post_validator so a
1082:             # deterministic "fix" that is itself content-invalid fails
1083:             # the ladder loudly rather than slipping through.
1084:             if isinstance(repair_prompt, schema):
1085:                 if post_validator is not None:
1086:                     content_error = post_validator(repair_prompt)
1087:                     if content_error is not None:
1088:                         raise PostValidationError(content_error)
1089:                 log.info(
1090:                     "[OTR_StructuredCall] '%s' attempt %d: repair factory "
1091:                     "resolved the failure deterministically; no LLM "
1092:                     "repair call made",
1093:                     helper_name, attempts_run,
1094:                 )
1095:                 # The attempt COMPLETED here, so report it like every other
1096:                 # successful return does. This was the one exit of six that
1097:                 # skipped the hook: `attempts_run` had already been
1098:                 # incremented for this rung, so a caller counting attempts
1099:                 # under-reported by one whenever a typed repair factory
1100:                 # resolved the failure itself. Unreachable from callers that
1101:                 # pass no `deterministic_repair` (the story-brief reflection),
1102:                 # but a lane MAY pass one.
1103:                 # Mutation-checked 2026-08-09: deleting this call turns
1104:                 # test_deterministic_repair_return_still_reports_its_attempt
1105:                 # red, so the guard is real rather than decorative.
1106:                 notify_attempt(None)
1107:                 return repair_prompt
1108:             repair_prompt = _inherit_generation_contract(
1109:                 contract_prompt, repair_prompt,
1110:             )
1111:             repair_messages = _prompt_to_messages(
1112:                 repair_prompt if text_parser is not None
1113:                 else _prompt_with_schema_contract(repair_prompt, schema)
1114:             )
1115:             last_raw = ""
1116:             last_raw = _invoke_slot(
1117:                 slot_fn, repair_messages,
1118:                 temperature=_REPAIR_TEMPERATURE,
1119:                 max_new_tokens=max_new_tokens,
1120:                 force_json_object=text_parser is None,
1121:             )
1122:             result = _parse_and_validate(
1123:                 last_raw,
1124:                 schema,
1125:                 post_validator,
1126:                 text_parser,
1127:             )
1128:             notify_attempt(None)
