# Authoritative final state

This snapshot supersedes the Bible hunks embedded in earlier review packets.
It supports Assign/AnnAssign, including typing.TypeAlias annotation (Python3.10),
not PEP695 type statements. Production code is unchanged since initial QA.

```diff
diff --git a/nodes/_otr_story_input.py b/nodes/_otr_story_input.py
index 76fedb6e..6b0d55fe 100644
--- a/nodes/_otr_story_input.py
+++ b/nodes/_otr_story_input.py
@@ -26,7 +26,7 @@ import hashlib
 import json
 import re
 from dataclasses import dataclass, field
-from typing import Any, Mapping
+from typing import Any, Literal, Mapping, get_args
 
 #: Bundle schema version. Bump only when the DIGESTED shape changes -- the
 #: digest is an identity, and a reader that cannot recompute it cannot verify
@@ -42,7 +42,8 @@ KNOWN_INPUT_MODES = frozenset({INPUT_MODE_LEGACY, INPUT_MODE_USER_FIELDS})
 #: The four creative fields, in the order they are shown and projected. The
 #: author is deliberately NOT one of them: naming who a story is by is not an
 #: idea for a story, and a run carrying only an author has nothing to write.
-CREATIVE_FIELDS = ("idea", "characters", "plot", "setting")
+CreativeFieldName = Literal["idea", "characters", "plot", "setting"]
+CREATIVE_FIELDS = get_args(CreativeFieldName)
 
 #: The three fields that belong to My Story alone. `custom_premise` is shared
 #: with every other bank and keeps its existing meaning there, so it is not in
diff --git a/nodes/_otr_story_source.py b/nodes/_otr_story_source.py
index 3e8c3bf5..48ccf04b 100644
--- a/nodes/_otr_story_source.py
+++ b/nodes/_otr_story_source.py
@@ -15,7 +15,7 @@ from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, Validat
 
 from ._otr_generation_budget import CAPACITY_ERRORS, PromptContextOverflowError, ProviderCapacityMessages
 from ._otr_source_document import build_source_document
-from ._otr_story_input import CREATIVE_FIELDS
+from ._otr_story_input import CREATIVE_FIELDS, CreativeFieldName
 from ._otr_structured_call import (
     PostValidationError, StructuredCallFailedError, inspect_structured_fit, structured_call,
 )
@@ -267,7 +267,7 @@ def rewrite_story_source(raw_fields, candidate, slot_fn, *, schema, receipts,
 class SpokenSourceEdit(BaseModel):
     model_config = ConfigDict(extra="forbid")
     line_id: StrictStr
-    source_field: StrictStr
+    source_field: CreativeFieldName
     source_quote: StrictStr = Field(min_length=1)
     original_quote: StrictStr = Field(min_length=1)
     replacement: StrictStr
@@ -363,9 +363,16 @@ def rewrite_spoken_from_source(ledger_data, *, slot_fn, slot_scheduler=None,
         slot_scheduler=slot_scheduler, configured_model_id=configured_model_id,
         instruction=("For this spoken ledger return edits containing actual replacement "
                      "text, grounded in an exact source quote and exact original interval. "
+                     "source_field names an original source key: %s. "
+                     "Copy source_quote exactly from source[source_field], never from "
+                     "the draft. Copy original_quote exactly from the draft line's text "
+                     "identified by line_id. Optional start_char/end_char are zero-based "
+                     "Python character offsets in that draft line, with end_char exclusive; "
+                     "they are not positions in the original source. "
                      "Use the edits schema instead of returning the full draft. Keep every "
                      "unrelated byte unchanged. Never change speakers, order or ids. "
-                     "Return an empty edits list when no source correction is needed."))
+                     "Return an empty edits list when no source correction is needed."
+                     % ", ".join(CREATIVE_FIELDS)))
     receipt["candidate_line_ids"] = [row["line_id"] for row in candidate["lines"]]
     if result is not None:
         updated = {row["line_id"]: row["text"] for row in accepted["lines"]}
diff --git a/tests/test_constrained_generate.py b/tests/test_constrained_generate.py
index 4a71c719..35eb75fd 100644
--- a/tests/test_constrained_generate.py
+++ b/tests/test_constrained_generate.py
@@ -288,6 +288,22 @@ def test_scope_authorization_real_grammar_accepts_empty_or_omitted_spans_never_n
     assert ord('[') in allowed and ord('n') not in allowed
 
 
+def test_spoken_source_grammar_excludes_candidate_field_and_accepts_all_source_keys():
+    from nodes._otr_story_source import SpokenSourceEdits
+    from nodes._otr_story_input import CREATIVE_FIELDS
+    schema = SpokenSourceEdits.model_json_schema()
+    assert tuple(schema['$defs']['SpokenSourceEdit']['properties']['source_field']['enum']) == CREATIVE_FIELDS
+    _, prefix = _real_constraint(SpokenSourceEdits)
+    allowed = _feed_json(prefix, '{"edits":[{"line_id":"l1","source_field":"')
+    assert ord('t') not in allowed  # live failure: source_field="text"
+    for field in CREATIVE_FIELDS:
+        _, prefix = _real_constraint(SpokenSourceEdits)
+        text = ('{"edits":[{"line_id":"l1","source_field":"' + field +
+                '","source_quote":"Mother is alive","original_quote":"Mother died.",'
+                '"replacement":"Mother lives."}]}')
+        assert 200 in _feed_json(prefix, text)
+
+
 def test_real_lmfe_uses_shared_model_chat_eos_and_refreshes_without_mutating_history():
     from types import SimpleNamespace
     import torch
diff --git a/tests/test_story_source_review.py b/tests/test_story_source_review.py
index 79677c0b..d71ee199 100644
--- a/tests/test_story_source_review.py
+++ b/tests/test_story_source_review.py
@@ -284,6 +284,31 @@ def test_spoken_correction_is_applied_without_changing_surrounding_bytes_ids_or_
     assert data["lines"][0]["char_count"] == len(data["lines"][0]["text"])
 
 
+def test_spoken_source_alias_repairs_to_an_applied_missing_action_within_two_calls():
+    data = _ledger()
+    data['meta']['source_meta']['story_input']['fields'] = {
+        'plot': 'Jeffrey briefly mentions his girlfriend as a separate person.'}
+    data['lines'][0]['text'] = '  Mom, I loved the carousel.  '
+    before = copy.deepcopy(data['lines'])
+    correction = _edit(
+        source_field='plot', source_quote='Jeffrey briefly mentions his girlfriend',
+        original_quote='Mom, I loved the carousel.',
+        replacement='Mom, I loved the carousel. My girlfriend would enjoy it too.')
+    responses = iter([{'edits': [dict(correction, source_field='text')]},
+                      {'edits': [correction]}])
+    slot = Slot(lambda _messages: next(responses))
+    receipt = source.rewrite_spoken_from_source(data, slot_fn=slot)
+    assert len(slot.calls) == 2 and receipt['applied']
+    assert data['lines'][0]['text'] == '  Mom, I loved the carousel. My girlfriend would enjoy it too.  '
+    assert data['lines'][1] == before[1]
+    assert data['lines'][0]['speaker'] == before[0]['speaker']
+    assert data['lines'][0]['line_id'] == before[0]['line_id']
+    assert receipt['input_sha256'] != receipt['output_sha256']
+    assert receipt['attempts'][0]['status'] == 'failed'
+    assert receipt['attempts'][1]['status'] == 'usable'
+    assert receipt['qualified'] is False  # application is not a semantic certificate
+
+
 @pytest.mark.parametrize("override", [
     {"source_field": "author"}, {"source_quote": "Invented source"},
     {"line_id": "unknown"}, {"original_quote": "invented original"},

```

{
  "nodes/_otr_story_input.py": "6ca61a41c65e4278c624f6d64ee5d60aa09cfcbbc4a86a3ba91c6c9abc9d9ed9",
  "nodes/_otr_story_source.py": "0bae5a77a7ae2def680d2f39a4972266ff9c81f0b5b7090763ce0506a3f22eb5",
  "tests/test_constrained_generate.py": "ee7873c576e785a37aee1bff9ea168b741d258e551c45693b68bdda4d6b020c4",
  "tests/test_story_source_review.py": "b1294e2a17120b189246696a3c7425baeec745162f6ac977fbe2a5e6f8ae5a33"
}
```diff
diff --git a/BUG_BIBLE.yaml b/BUG_BIBLE.yaml
index ccf3535..e593812 100644
--- a/BUG_BIBLE.yaml
+++ b/BUG_BIBLE.yaml
@@ -3760,6 +3760,20 @@ bugs:
     preservation of untouched bytes, and durable repair history after cleanup
     failure or rollback. Coverage: test_story_source_review.py and
     test_my_story_runner.py in the OTR production regression suite.
+    Keep an edit's original-source field namespace closed in the bound schema
+    when the application validator already accepts only those fields. Live
+    My Story pairlock_05 generated source_field=text, confusing candidate text
+    with original source; its remaining repair returned no edits. Generate
+    the same allowed field enum from the existing source authority. Teach
+    source_quote versus original_quote and candidate-line offsets in the
+    actual model instruction, not only unused schema descriptions. Verify
+    native grammar excludes the invalid alias and all original source keys
+    remain usable; these assertions discriminate the schema correction.
+    Separately retain integration coverage that a corrected replacement applies
+    inside the same two-call budget. That conservation test also passes before
+    this schema correction and is not evidence of semantic recovery.
+    This structural correction does not certify semantic fidelity or justify
+    an additional publication gate.
     Visual coverage: tests/test_my_story_visual_source.py verifies full source
     and scene context, actual corrected-prompt application, bounded malformed
     retries, no stale appearance prepend, neutral portrait isolation, failed
diff --git a/tests/bug_bible_regression.py b/tests/bug_bible_regression.py
index efac230..b9d92dc 100644
--- a/tests/bug_bible_regression.py
+++ b/tests/bug_bible_regression.py
@@ -1609,6 +1609,7 @@ class TestPhase07To12ProductionRegressionCatalog:
             "tests/test_story_source_review.py": (
                 "test_stubborn_failure_stops_at_two_actual_calls_without_a_fourth_or_fifth_round",
                 "test_spoken_correction_is_applied_without_changing_surrounding_bytes_ids_or_order",
+                "test_spoken_source_alias_repairs_to_an_applied_missing_action_within_two_calls",
                 "test_tail_persists_source_repair_before_propagating_later_cleanup_failure",
                 "test_tail_rollback_keeps_attempt_history_and_actual_retained_hash_without_rechecking",
             ),
@@ -1636,6 +1637,7 @@ class TestPhase07To12ProductionRegressionCatalog:
             ),
             "tests/test_constrained_generate.py": (
                 "test_generate_invoked_with_prefix_fn",
+                "test_spoken_source_grammar_excludes_candidate_field_and_accepts_all_source_keys",
             ),
         }
         for relative_path, names in expected.items():
@@ -1646,6 +1648,53 @@ class TestPhase07To12ProductionRegressionCatalog:
             for name in names:
                 assert f"def {name}(" in source, f"production regression missing: {relative_path}::{name}"
 
+    def test_otr_spoken_source_schema_uses_the_original_field_namespace(self, pack_dir):
+        """BUG-11.39: execute actual schema nodes without importing pack/GPU startup.
+
+        This portable scan may target an uninstalled pack. AST selection keeps
+        the production definitions, with explicit failures if they are moved;
+        the OTR suite separately exercises real imports and native decoding.
+        """
+        source_path = os.path.join(pack_dir, "nodes", "_otr_story_source.py")
+        input_path = os.path.join(pack_dir, "nodes", "_otr_story_input.py")
+        if not os.path.isfile(source_path):
+            pytest.skip("My Story source coordinates are OTR-local")
+        import typing
+        pydantic = pytest.importorskip("pydantic")
+        namespace = {"Literal": typing.Literal, "get_args": typing.get_args,
+                     "TypeAlias": typing.TypeAlias,
+                     "BaseModel": pydantic.BaseModel, "ConfigDict": pydantic.ConfigDict,
+                     "StrictStr": pydantic.StrictStr, "StrictInt": pydantic.StrictInt,
+                     "Field": pydantic.Field}
+        with open(input_path, encoding="utf-8") as handle:
+            input_tree = ast.parse(handle.read(), filename=input_path)
+        names = {"CREATIVE_FIELDS", "CreativeFieldName"}
+        fields, found = [], set()
+        for node in input_tree.body:
+            targets = (node.targets if isinstance(node, ast.Assign) else
+                       [node.target] if isinstance(node, ast.AnnAssign) else [])
+            selected = {target.id for target in targets
+                        if isinstance(target, ast.Name) and target.id in names}
+            if selected:
+                fields.append(node)
+                found.update(selected)
+        assert found == names, "Source namespace declarations moved; update this isolated schema guard"
+        exec(compile(ast.Module(body=fields, type_ignores=[]), input_path, "exec"), namespace)
+        with open(source_path, encoding="utf-8") as handle:
+            source_tree = ast.parse(handle.read(), filename=source_path)
+        definitions = [node for node in source_tree.body if isinstance(node, ast.ClassDef)
+                       and node.name == "SpokenSourceEdit"]
+        assert len(definitions) == 1
+        exec(compile(ast.Module(body=definitions, type_ignores=[]), source_path, "exec"), namespace)
+        model = namespace["SpokenSourceEdit"]
+        assert model.model_json_schema()["properties"]["source_field"].get("enum") == list(namespace["CREATIVE_FIELDS"])
+        for field in namespace["CREATIVE_FIELDS"]:
+            assert model(line_id="l1", source_field=field, source_quote="source",
+                         original_quote="candidate", replacement="corrected").source_field == field
+        with pytest.raises(pydantic.ValidationError):
+            model(line_id="l1", source_field="text", source_quote="candidate",
+                  original_quote="candidate", replacement="corrected")
+
     def test_otr_full_repair_uses_captured_text_when_generation_raises(self, pack_dir):
         """BUG-11.48: execute the lane owner, not a copied repair implementation."""
         path = os.path.join(pack_dir, "nodes", "_otr_my_story.py")

```

{
  "BUG_BIBLE.yaml": "57841dd6797a45418346b5eadcfbabe3b5103defec094bc08779d3fdc789732b",
  "tests/bug_bible_regression.py": "ce94de35679403173b2cc208f9dcce763e4a9c229a5829257c9667a9df1f22fa"
}
