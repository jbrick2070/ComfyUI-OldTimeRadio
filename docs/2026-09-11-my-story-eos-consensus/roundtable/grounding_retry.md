# Current final candidate grounding for bounded Opus retry
No file access assumed. Source excerpts and entire six-file diff below. Root judges all claims.

## nodes/_otr_model_loader.py:2270 _native_token_ids
```python
2270: def _native_token_ids(value) -> list[int]:
2271:     values = value if isinstance(value, (list, tuple, set)) else (value,)
2272:     normalized = [int(item) for item in values
2273:                   if isinstance(item, Integral) and not isinstance(item, bool) and item >= 0]
2274:     return sorted(normalized) if isinstance(value, set) else normalized
```

## nodes/_otr_model_loader.py:2277 native_eos_token_ids
```python
2277: def native_eos_token_ids(cache_entry: dict[str, Any]) -> list[int]:
2278:     """Share effective model and chat terminators without changing either owner.
2279: 
2280:     A native text decoder may inherit end-of-text from its config while its
2281:     chat tokenizer names end-of-turn. Grammar, generation and completion
2282:     classification must accept the same set. Preserve configured multi-EOS.
2283:     """
2284:     model = cache_entry["model"]
2285:     configured = _native_token_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
2286:     if not configured:
2287:         config = getattr(model, "config", None)
2288:         text_config = getattr(config, "text_config", None) or config
2289:         configured = _native_token_ids(getattr(text_config, "eos_token_id", None))
2290:     chat = _native_token_ids(getattr(cache_entry["tokenizer"], "eos_token_id", None))
2291:     return list(dict.fromkeys([*configured, *chat]))
```

## nodes/_otr_model_loader.py:2294 prepare_native_prompt
```python
2294: def prepare_native_prompt(cache_entry: dict[str, Any], messages) -> dict[str, Any]:
2295:     """Prepare the exact generation prompt on CPU; retain no model handles.
2296: 
2297:     Inputs belong to this one call. A fit inspector returns measurements only;
2298:     actual generation prepares again after any intervening slot transition.
2299:     """
2300:     from ._otr_loader_backends import chat_template_kwargs
2301:     from . import _otr_model_catalog as catalog
2302: 
2303:     flags = {
2304:         name: bool(getattr(messages, marker, False))
2305:         for name, marker in (
2306:             ("require_full_output", "_otr_require_full_output_budget"),
2307:             ("reserve_remaining", "_otr_reserve_remaining_output_capacity"),
2308:             ("fail_on_output_limit", "_otr_fail_on_output_limit"),
2309:             ("unbounded_json_field", "_otr_unbounded_json_field"),
2310:         )
2311:     }
2312:     tokenizer = cache_entry["tokenizer"]
2313:     normalized = _normalize_messages_for_cache_entry(cache_entry, messages)
2314:     prompt = tokenizer.apply_chat_template(
2315:         normalized, tokenize=False, add_generation_prompt=True,
2316:         **chat_template_kwargs(cache_entry.get("model_id", "")),
2317:     )
2318:     inputs = tokenizer(prompt, return_tensors="pt")
2319:     cap = catalog.normalized_context_pin(cache_entry.get("context_cap"))
2320:     source = cache_entry.get("context_capacity_source")
2321:     # Old third-party cache entries retain a disclosed estimate. Real native
2322:     # loads always stamp capacity and provenance at the loader boundary.
2323:     if cap is None:
2324:         cap = catalog.DEFAULT_CONTEXT_ESTIMATE
2325:         source = "legacy entry missing native capacity; default context estimate"
2326:     return {
2327:         "inputs": inputs,
2328:         "eos_token_ids": native_eos_token_ids(cache_entry),
2329:         "pad_token_id": next(iter(_native_token_ids(tokenizer.eos_token_id)), None),
2330:         "prompt_tokens": int(inputs["input_ids"].shape[-1]),
2331:         "context_cap": cap,
2332:         "capacity_source": str(source or "entry context setting; native capacity unreported"),
2333:         "capacity_known": catalog.normalized_context_pin(cache_entry.get("native_context_capacity")) is not None,
2334:         "model_id": str(cache_entry.get("model_id") or "<unknown>"),
2335:         **flags,
2336:     }
```

## nodes/_otr_constrained_generate.py:100 get_cached_transformers_schema_constraint
```python
100: def get_cached_transformers_schema_constraint(
101:     cache_entry: dict[str, Any],
102:     schema_model: Type[BaseModel],
103:     *, eos_token_ids: list[int] | None = None,
104: ) -> tuple[Any, Any]:
105:     """Return fresh request state while reusing resident tokenizer preprocessing.
106: 
107:     LMFE's expensive step is tokenizer-wide and schema-independent: it
108:     decodes/scans every token to build ``TokenEnforcerTokenizerData``. Gemma 4
109:     has a roughly 256K-token vocabulary, so rebuilding that data for every
110:     P0-P9 pass or retry is material. Cache it on the resident ``cache_entry``
111:     only. Parser/enforcer prefix history belongs to one generation; retaining it
112:     would accumulate past requests on the resident model. Unloading the model
113:     drops the tokenizer preprocessing naturally.
114:     """
115:     required = {"model", "tokenizer"}
116:     missing = required - set(cache_entry)
117:     if missing:
118:         raise ModelLoaderError(
119:             f"cache_entry missing required keys: {sorted(missing)}"
120:         )
121:     tokenizer = cache_entry["tokenizer"]
122: 
123:     _otr_lmfe_compat.ensure_lmfe_transformers_compat()
124:     from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
125:     from lmformatenforcer.integrations.transformers import (
126:         build_token_enforcer_tokenizer_data,
127:         build_transformers_prefix_allowed_tokens_fn,
128:     )
129: 
130:     eos_ids = tuple(native_eos_token_ids(cache_entry) if eos_token_ids is None else eos_token_ids)
131:     cache = cache_entry.get("_otr_lmfe_constraint_cache")
132:     if (not isinstance(cache, dict) or cache.get("tokenizer") is not tokenizer
133:             or cache.get("eos_token_ids") != eos_ids):
134:         tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
135:         # This data is newly owned by this cache entry, not the tokenizer.
136:         # LMFE supports multiple EOS IDs and only permits them at completion.
137:         tokenizer_data.eos_token_id = list(eos_ids)
138:         cache = {
139:             "tokenizer": tokenizer,
140:             "tokenizer_data": tokenizer_data,
141:             "eos_token_ids": eos_ids,
142:         }
143:         cache_entry["_otr_lmfe_constraint_cache"] = cache
144: 
145:     # Release history left by an already-warm entry from the earlier cache shape.
146:     cache.pop("by_schema", None)
147:     parser = JsonSchemaParser(
148:         schema_model.model_json_schema(),
149:         config=CharacterLevelParserConfig(max_json_array_length=0),
150:     )
151:     prefix_fn = build_transformers_prefix_allowed_tokens_fn(
152:         cache["tokenizer_data"], parser,
153:     )
154:     # The builder replaces config to install the tokenizer alphabet. Root lists
155:     # capture their bound during construction; nested lists read this later.
156:     # Disable only LMFE's implicit20-item default at BOTH boundaries; explicit
157:     # schema maxItems remains authoritative. Preserve the installed alphabet.
158:     parser.config.max_json_array_length = 0
159:     return parser, prefix_fn
```

## Writer generation, retry, classification (current)
```python
1051: 
1052:         # LIVE HEARTBEAT (2026-08-13). This transport had NO streamer, and it is
1053:         # the one that ran away: a P3 prose pass burned its whole 14,191-token
1054:         # allowance without a stop token, three times, ~20 minutes each, and
1055:         # nothing was visible while it happened. Read-only -- generate() hands
1056:         # the streamer sampled token ids and this never feeds any back, so the
1057:         # output is byte-identical with it attached.
1058:         _hb = _OTRHB.make_streamer(tokenizer, "OTR_LedgerScriptWriter")
1059:         if _hb is not None:
1060:             gen_kwargs = dict(gen_kwargs, streamer=_hb)
1061: 
1062:         _applied_seed = _seed_writer_sampling(inputs)
1063:         if _applied_seed is not None:
1064:             log.info("[OTR_LedgerScriptWriter] sampling seeded (%s) -- this "
1065:                      "pass is reproducible", WRITER_SEED_ENV)
1066: 
1067:         with torch.no_grad():
1068:             try:
1069:                 if schema_model is not None:
1070:                     _, gen_kwargs["prefix_allowed_tokens_fn"] = (
1071:                         get_cached_transformers_schema_constraint(
1072:                             cache_entry, schema_model, eos_token_ids=prepared["eos_token_ids"])
1073:                     )
1074:                 out = model.generate(**inputs, **gen_kwargs)
1075:             except TypeError as exc:
1076:                 # Tier 1 fix #8: min_p kwarg unsupported on
1077:                 # transformers < 4.43. Warn once and retry without it
1078:                 # for the rest of this run.
1079:                 if "min_p" in gen_kwargs and "min_p" in str(exc):
1080:                     log.warning(
1081:                         "[OTR_LedgerScriptWriter] min_p kwarg not "
1082:                         "supported by this transformers version; "
1083:                         "disabling for the remainder of this run "
1084:                         "(error was: %s)",
1085:                         str(exc),
1086:                     )
1087:                     _min_p_unsupported[0] = True
1088:                     gen_kwargs.pop("min_p", None)
1089:                     # Re-seed: the failed attempt above already consumed RNG
1090:                     # state, so without this the retry path would diverge from
1091:                     # a run that never hit the TypeError.
1092:                     _seed_writer_sampling(inputs)
1093:                     if schema_model is not None:
1094:                         _, gen_kwargs["prefix_allowed_tokens_fn"] = (
1095:                             get_cached_transformers_schema_constraint(
1096:                                 cache_entry, schema_model, eos_token_ids=prepared["eos_token_ids"])
1097:                         )
1098:                     out = model.generate(**inputs, **gen_kwargs)
1099:                 else:
1100:                     raise
1101:         _memory_log.memory_snapshot("writer_generation_returned", model_id=cache_entry.get("model_id"))
1102:         prompt_len = inputs["input_ids"].shape[1]
1103:         generated_ids = out[0][prompt_len:]
1104:         try:
1105:             generated_tokens = int(getattr(generated_ids, "shape", [len(generated_ids)])[-1])
1106:         except Exception:  # pragma: no cover - exotic backend sequence shape
1107:             generated_tokens = None
1108:         ended_with_eos = False
1109:         last_token = None
1110:         try:
1111:             last_token = int(generated_ids[-1])
1112:             eos_values = prepared["eos_token_ids"]
1113:             ended_with_eos = last_token in eos_values
1114:         except Exception:  # pragma: no cover - exotic token container
1115:             pass
1116:         log.info(
1117:             "[OTR_LedgerScriptWriter] DECODE RETURNED: generated_tokens=%s "
1118:             "last_token=%s eos_token_ids=%s ended_with_eos=%s",
1119:             generated_tokens, last_token, prepared["eos_token_ids"], ended_with_eos,
1120:         )
1121:         # A-1 (2026-07-30, writer repair): DECODE BEFORE THE RAISE.
1122:         # The output-limit raise used to fire HERE, above the decode, so a
1123:         # fail-closed leg threw away the only copy of what the model actually
1124:         # produced -- the ladder received an exception with no artifact, and the
1125:         # OUTPUT_TRUNCATED / OUTPUT_CAP arithmetic below never printed either,
1126:         # because the raise jumped over it. Whoever debugs a truncation needs
1127:         # the completion AND the arithmetic, and both were unreachable at the
1128:         # one moment they exist. Decoding here costs a decode on a leg that is
1129:         # already dying; the success path decodes exactly once, as before.
1130:         decoded = tokenizer.decode(
1131:             generated_ids, skip_special_tokens=True,
1132:         )
1133: 
1134:         # CLASSIFY THE HALT FIRST (2026-08-13). Order matters and this is the
1135:         # authoritative sequence from the settled design:
1136:         #   1. guard.hit          -> degeneracy, REGARDLESS of generated length
1137:         #   2. at the ceiling, no EOS -> capacity
1138:         #   3. ended_with_eos     -> clean termination
1139:         #   4. otherwise          -> some other criterion, e.g. a stop substring
1140:         # Degeneracy must be tested BEFORE capacity because a halted decode
1141:         # stops with room to spare -- reading it as anything else would report
1142:         # "ended at the provider capacity limit" about a decode that was
1143:         # deliberately stopped with ~11,000 tokens unspent.
1144:         if _degeneracy_guard is not None and getattr(
1145:             _degeneracy_guard, "hit", False
1146:         ):
1147:             telemetry = _degeneracy_guard.telemetry()
1148:             reason = ("an open JSON string exceeded its token allowance"
1149:                       if _degeneracy_guard.reason == "open_string"
1150:                       else "the output repeated a run of tokens verbatim")
1151:             log.error(
1152:                 "[OTR_LedgerScriptWriter] DECODE HALTED (%s): %s, after %s "
1153:                 "generated tokens of a %d-token allowance. Rerollable. Telemetry: %s",
1154:                 _degeneracy_guard.reason, reason,
1155:                 generated_tokens, effective_max_new_tokens,
1156:                 telemetry,
1157:             )
1158:             # Same evidence discipline as the capacity raise below: the head
1159:             # says what the model was writing, the tail says what it was doing
1160:             # when the guard stopped it. For a degeneracy halt the tail is the
1161:             # whole point -- it is where the loop is visible.
1162:             _halt_raw = decoded or ""
1163:             log.error(
1164:                 "[OTR_LedgerScriptWriter] RUNAWAY EVIDENCE (%d chars, %s "
1165:                 "tokens, halted)\n  HEAD: %s\n  TAIL: %s",
1166:                 len(_halt_raw), generated_tokens,
1167:                 _halt_raw[:400].replace("\n", " "),
1168:                 _halt_raw[-400:].replace("\n", " "),
1169:             )
1170:             raise GenerationDegeneracyError(
1171:                 "generation was halted by the in-decode liveness guard: " + reason,
1172:                 halt_reason=_degeneracy_guard.reason,
1173:                 open_string_tokens=telemetry.get("open_string_tokens"),
1174:                 repetition=telemetry,
1175:                 raw_completion=decoded,
1176:                 prompt_tokens=prompt_len,
1177:                 generated_tokens=generated_tokens,
1178:                 requested_output_tokens=requested_max_new_tokens,
1179:                 effective_output_tokens=effective_max_new_tokens,
1180:                 context_cap=context_cap,
1181:                 ended_with_eos=ended_with_eos,
1182:             )
1183: 
1184:         if generated_tokens == effective_max_new_tokens:
1185:             # The model stopped because it ran OUT OF ROOM, not because it was
1186:             # finished. When the room it was given is also LESS than the room
1187:             # its caller asked for, that is the silent catastrophe: the artifact
1188:             # is cut off mid-JSON and the ladder reports a bare JSONDecodeError
1189:             # three times, naming the model instead of the budget. Say the real
1190:             # cause once, LOUDLY, with the whole arithmetic -- a reader of the
1191:             # leg log must never have to reconstruct it.
1192:             if effective_max_new_tokens < requested_max_new_tokens:
1193:                 if reserve_remaining:
1194:                     # THE ADVICE WAS WRONG EXACTLY WHEN IT FIRED (2026-08-13).
1195:                     #
1196:                     # A ProviderCapacityMessages pass sets
1197:                     # _otr_reserve_remaining_output_capacity, so requested ==
1198:                     # the whole context window BY DESIGN, and
1199:                     # effective < requested is true the moment the prompt is
1200:                     # non-empty. The old text told the reader to "give this
```

## Complete candidate diff versus 2c9d47f2
```diff
diff --git a/nodes/OTR_LedgerScriptWriter.py b/nodes/OTR_LedgerScriptWriter.py
index 7ea854f7..c0d6ee2e 100644
--- a/nodes/OTR_LedgerScriptWriter.py
+++ b/nodes/OTR_LedgerScriptWriter.py
@@ -953,7 +953,8 @@ def _build_truncating_generate_fn(
             "temperature": float(temperature),
             "top_p": active_top_p,
             "max_new_tokens": effective_max_new_tokens,
-            "pad_token_id": tokenizer.eos_token_id,
+            "pad_token_id": prepared["pad_token_id"],
+            "eos_token_id": prepared["eos_token_ids"] or None,
         }
         # Only forward non-default values so older transformers
         # versions that don't accept `min_p` as a kwarg keep working
@@ -1067,7 +1068,8 @@ def _build_truncating_generate_fn(
             try:
                 if schema_model is not None:
                     _, gen_kwargs["prefix_allowed_tokens_fn"] = (
-                        get_cached_transformers_schema_constraint(cache_entry, schema_model)
+                        get_cached_transformers_schema_constraint(
+                            cache_entry, schema_model, eos_token_ids=prepared["eos_token_ids"])
                     )
                 out = model.generate(**inputs, **gen_kwargs)
             except TypeError as exc:
@@ -1090,7 +1092,8 @@ def _build_truncating_generate_fn(
                     _seed_writer_sampling(inputs)
                     if schema_model is not None:
                         _, gen_kwargs["prefix_allowed_tokens_fn"] = (
-                            get_cached_transformers_schema_constraint(cache_entry, schema_model)
+                            get_cached_transformers_schema_constraint(
+                                cache_entry, schema_model, eos_token_ids=prepared["eos_token_ids"])
                         )
                     out = model.generate(**inputs, **gen_kwargs)
                 else:
@@ -1103,15 +1106,18 @@ def _build_truncating_generate_fn(
         except Exception:  # pragma: no cover - exotic backend sequence shape
             generated_tokens = None
         ended_with_eos = False
+        last_token = None
         try:
             last_token = int(generated_ids[-1])
-            eos = tokenizer.eos_token_id
-            eos_values = {int(value) for value in (
-                eos if isinstance(eos, (list, tuple, set)) else (eos,)
-            ) if value is not None}
+            eos_values = prepared["eos_token_ids"]
             ended_with_eos = last_token in eos_values
         except Exception:  # pragma: no cover - exotic token container
             pass
+        log.info(
+            "[OTR_LedgerScriptWriter] DECODE RETURNED: generated_tokens=%s "
+            "last_token=%s eos_token_ids=%s ended_with_eos=%s",
+            generated_tokens, last_token, prepared["eos_token_ids"], ended_with_eos,
+        )
         # A-1 (2026-07-30, writer repair): DECODE BEFORE THE RAISE.
         # The output-limit raise used to fire HERE, above the decode, so a
         # fail-closed leg threw away the only copy of what the model actually
diff --git a/nodes/_otr_constrained_generate.py b/nodes/_otr_constrained_generate.py
index 7ac67a0f..77e5f18d 100644
--- a/nodes/_otr_constrained_generate.py
+++ b/nodes/_otr_constrained_generate.py
@@ -46,6 +46,7 @@ from ._otr_generation_budget import (
 )
 from ._otr_model_loader import (
     ModelLoaderError,
+    native_eos_token_ids,
     prepare_native_prompt,
 )
 
@@ -99,6 +100,7 @@ closes.
 def get_cached_transformers_schema_constraint(
     cache_entry: dict[str, Any],
     schema_model: Type[BaseModel],
+    *, eos_token_ids: list[int] | None = None,
 ) -> tuple[Any, Any]:
     """Return fresh request state while reusing resident tokenizer preprocessing.
 
@@ -125,11 +127,18 @@ def get_cached_transformers_schema_constraint(
         build_transformers_prefix_allowed_tokens_fn,
     )
 
+    eos_ids = tuple(native_eos_token_ids(cache_entry) if eos_token_ids is None else eos_token_ids)
     cache = cache_entry.get("_otr_lmfe_constraint_cache")
-    if not isinstance(cache, dict) or cache.get("tokenizer") is not tokenizer:
+    if (not isinstance(cache, dict) or cache.get("tokenizer") is not tokenizer
+            or cache.get("eos_token_ids") != eos_ids):
+        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
+        # This data is newly owned by this cache entry, not the tokenizer.
+        # LMFE supports multiple EOS IDs and only permits them at completion.
+        tokenizer_data.eos_token_id = list(eos_ids)
         cache = {
             "tokenizer": tokenizer,
-            "tokenizer_data": build_token_enforcer_tokenizer_data(tokenizer),
+            "tokenizer_data": tokenizer_data,
+            "eos_token_ids": eos_ids,
         }
         cache_entry["_otr_lmfe_constraint_cache"] = cache
 
@@ -320,13 +329,14 @@ def make_constrained_generate_fn(
             sampling = {"do_sample": False}
         with torch.no_grad():
             parser, prefix_fn = get_cached_transformers_schema_constraint(
-                cache_entry, schema_model,
+                cache_entry, schema_model, eos_token_ids=prepared["eos_token_ids"],
             )
             out = model.generate(
                 **inputs,
                 **sampling,
                 max_new_tokens=effective_max_new_tokens,
-                pad_token_id=tokenizer.eos_token_id,
+                pad_token_id=prepared["pad_token_id"],
+                eos_token_id=prepared["eos_token_ids"] or None,
                 stopping_criteria=StoppingCriteriaList([_guard]),
                 # The schema-binding argument. transformers passes
                 # this hook into the logits-processing path; lm-
@@ -373,10 +383,7 @@ def make_constrained_generate_fn(
                 prompt_tokens=prompt_len,
             )
         generated_ids = out[0][prompt_len:]
-        eos = tokenizer.eos_token_id
-        eos_values = {int(value) for value in (
-            eos if isinstance(eos, (list, tuple, set)) else (eos,)
-        ) if value is not None}
+        eos_values = prepared["eos_token_ids"]
         ended_with_eos = bool(len(generated_ids)) and int(generated_ids[-1]) in eos_values
         if (prepared["fail_on_output_limit"]
                 and len(generated_ids) >= effective_max_new_tokens and not ended_with_eos):
diff --git a/nodes/_otr_model_loader.py b/nodes/_otr_model_loader.py
index 8e300199..585d870a 100644
--- a/nodes/_otr_model_loader.py
+++ b/nodes/_otr_model_loader.py
@@ -43,6 +43,7 @@ import logging
 import os
 import threading
 import time
+from numbers import Integral
 from pathlib import Path
 from typing import Any
 
@@ -2266,6 +2267,30 @@ def _normalize_messages_for_cache_entry(
     )
 
 
+def _native_token_ids(value) -> list[int]:
+    values = value if isinstance(value, (list, tuple, set)) else (value,)
+    normalized = [int(item) for item in values
+                  if isinstance(item, Integral) and not isinstance(item, bool) and item >= 0]
+    return sorted(normalized) if isinstance(value, set) else normalized
+
+
+def native_eos_token_ids(cache_entry: dict[str, Any]) -> list[int]:
+    """Share effective model and chat terminators without changing either owner.
+
+    A native text decoder may inherit end-of-text from its config while its
+    chat tokenizer names end-of-turn. Grammar, generation and completion
+    classification must accept the same set. Preserve configured multi-EOS.
+    """
+    model = cache_entry["model"]
+    configured = _native_token_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
+    if not configured:
+        config = getattr(model, "config", None)
+        text_config = getattr(config, "text_config", None) or config
+        configured = _native_token_ids(getattr(text_config, "eos_token_id", None))
+    chat = _native_token_ids(getattr(cache_entry["tokenizer"], "eos_token_id", None))
+    return list(dict.fromkeys([*configured, *chat]))
+
+
 def prepare_native_prompt(cache_entry: dict[str, Any], messages) -> dict[str, Any]:
     """Prepare the exact generation prompt on CPU; retain no model handles.
 
@@ -2300,6 +2325,8 @@ def prepare_native_prompt(cache_entry: dict[str, Any], messages) -> dict[str, An
         source = "legacy entry missing native capacity; default context estimate"
     return {
         "inputs": inputs,
+        "eos_token_ids": native_eos_token_ids(cache_entry),
+        "pad_token_id": next(iter(_native_token_ids(tokenizer.eos_token_id)), None),
         "prompt_tokens": int(inputs["input_ids"].shape[-1]),
         "context_cap": cap,
         "capacity_source": str(source or "entry context setting; native capacity unreported"),
@@ -2316,6 +2343,7 @@ def inspect_native_prompt_fit(cache_entry: dict[str, Any], messages, *, max_new_
     prepared = prepare_native_prompt(cache_entry, messages)
     measured = {key: prepared[key] for key in (
         "prompt_tokens", "context_cap", "capacity_source", "capacity_known", "model_id",
+        "eos_token_ids",
     )}
     requested = (prepared["context_cap"] if prepared["reserve_remaining"]
                  else max(1, int(max_new_tokens)))
@@ -2454,7 +2482,8 @@ def make_generate_fn(cache_entry: dict[str, Any]):
                 temperature=temperature,
                 top_p=0.92,
                 max_new_tokens=effective_max_new_tokens,
-                pad_token_id=tokenizer.eos_token_id,
+                pad_token_id=prepared["pad_token_id"],
+                eos_token_id=prepared["eos_token_ids"] or None,
                 stopping_criteria=StoppingCriteriaList(
                     [_guard, _deadline_guard]),
                 # Read-only live heartbeat (2026-08-13). A reserve-remaining
@@ -2513,7 +2542,9 @@ def make_generate_fn(cache_entry: dict[str, Any]):
                 generated_tokens=len(generated_ids),
                 effective_output_tokens=effective_max_new_tokens,
             )
-        if fail_on_output_limit and len(generated_ids) >= effective_max_new_tokens:
+        if (fail_on_output_limit
+                and len(generated_ids) >= effective_max_new_tokens
+                and int(generated_ids[-1]) not in prepared["eos_token_ids"]):
             raise ModelLoaderError(
                 "prose generation exhausted the full remaining provider/context "
                 "capacity; the partial artifact is not eligible for reroll"
@@ -2651,7 +2682,8 @@ def make_polish_generate_fn(cache_entry: dict[str, Any]):
                 temperature=temperature,
                 top_p=_POLISH_TOP_P,
                 max_new_tokens=effective_max_new_tokens,
-                pad_token_id=tokenizer.eos_token_id,
+                pad_token_id=prepared["pad_token_id"],
+                eos_token_id=prepared["eos_token_ids"] or None,
                 stopping_criteria=StoppingCriteriaList(
                     [_guard, _deadline_guard]),
                 streamer=_OTRHB.make_streamer(
@@ -2695,7 +2727,9 @@ def make_polish_generate_fn(cache_entry: dict[str, Any]):
                 generated_tokens=len(generated_ids),
                 effective_output_tokens=effective_max_new_tokens,
             )
-        if fail_on_output_limit and len(generated_ids) >= effective_max_new_tokens:
+        if (fail_on_output_limit
+                and len(generated_ids) >= effective_max_new_tokens
+                and int(generated_ids[-1]) not in prepared["eos_token_ids"]):
             raise ModelLoaderError(
                 "prose generation exhausted the full remaining provider/context "
                 "capacity; the partial artifact is not eligible for reroll"
diff --git a/tests/test_constrained_generate.py b/tests/test_constrained_generate.py
index 8a0c8ff1..05ecd570 100644
--- a/tests/test_constrained_generate.py
+++ b/tests/test_constrained_generate.py
@@ -161,7 +161,7 @@ class TestFactoryContract:
         assert prefix_2 is not prefix_1
         internal = cache_entry["_otr_lmfe_constraint_cache"]
         assert internal["tokenizer"] is cache_entry["tokenizer"]
-        assert set(internal) == {"tokenizer", "tokenizer_data"}
+        assert set(internal) == {"tokenizer", "tokenizer_data", "eos_token_ids"}
         assert scan.call_count == 1
         # Warm entries may still carry the old history-owning shape.
         internal["by_schema"] = {_TinySchema: (parser_1, prefix_1)}
@@ -277,6 +277,34 @@ def _feed_json(prefix, text):
     return prefix(0, torch.tensor(ids))
 
 
+def test_real_lmfe_uses_shared_model_chat_eos_and_refreshes_without_mutating_history():
+    from types import SimpleNamespace
+    import torch
+    from transformers import EosTokenCriteria
+    from nodes._otr_constrained_generate import get_cached_transformers_schema_constraint
+    from nodes._otr_model_loader import native_eos_token_ids
+    entry = TestFactoryContract()._make_minimal_cache_entry()
+    entry["model"].generation_config = SimpleNamespace(eos_token_id=[202])
+    _, old_prefix = get_cached_transformers_schema_constraint(entry, _TinySchema)
+    ids = native_eos_token_ids(entry)
+    assert ids == [202, 200]
+    assert not set(ids).intersection(old_prefix(0, torch.tensor([201])))
+    allowed = _feed_json(old_prefix, '{"color":"red","count":1}')
+    for eos in ids:
+        assert eos in allowed
+        assert EosTokenCriteria(ids)(torch.tensor([[eos]]), None).item()
+    old_data = entry["_otr_lmfe_constraint_cache"]["tokenizer_data"]
+    entry["model"].generation_config.eos_token_id = [201]
+    _, new_prefix = get_cached_transformers_schema_constraint(entry, _TinySchema)
+    assert new_prefix is not old_prefix
+    assert entry["_otr_lmfe_constraint_cache"]["tokenizer_data"] is not old_data
+    assert old_data.eos_token_id == [202, 200]
+    assert old_prefix.token_enforcer.eos_token_id == [202, 200]
+    assert set(_feed_json(new_prefix, '{"color":"red","count":1}')) >= {201, 200}
+    assert entry["tokenizer"].eos_token_id == 200
+    assert entry["model"].generation_config.eos_token_id == [201]
+
+
 @pytest.mark.parametrize("nested", [False, True])
 def test_real_lmfe_accepts_twenty_five_items_without_implicit_limit(nested):
     import json
@@ -391,7 +419,7 @@ def test_each_actual_generation_has_fresh_collectible_history(kind, outcome, mon
     assert observations == [
         ("writer" if kind == "writer" else "constrained") + "_generation_returned"
     ] * (1 if outcome == "error" else 2)
-    assert set(entry["_otr_lmfe_constraint_cache"]) == {"tokenizer", "tokenizer_data"}
+    assert set(entry["_otr_lmfe_constraint_cache"]) == {"tokenizer", "tokenizer_data", "eos_token_ids"}
     assert not hasattr(fn, "prefix_allowed_tokens_fn")
     gc.collect()
     assert all(ref() is None for group in refs for ref in group)
diff --git a/tests/test_generation_budget.py b/tests/test_generation_budget.py
index 51e3bbae..2d386e12 100644
--- a/tests/test_generation_budget.py
+++ b/tests/test_generation_budget.py
@@ -58,7 +58,7 @@ def _exact_prompt_entry(monkeypatch, capacity=32768):
 
     monkeypatch.setattr(writer._OTRHB, "make_streamer", lambda *args: None)
     monkeypatch.setattr(constrained, "get_cached_transformers_schema_constraint",
-                        lambda *args: (None, lambda *args: []))
+                        lambda *args, **kwargs: (None, lambda *args: []))
     entry = {"model": Model(), "tokenizer": Tokenizer(), "model_id": "Qwen/Qwen3.5-4B",
              "context_cap": capacity, "native_context_capacity": capacity,
              "context_capacity_source": "loaded decoder config"}
@@ -120,6 +120,69 @@ def test_constrained_eos_at_exact_capacity_is_a_completed_reply(eos, monkeypatch
     assert result == '{"value":"ok"}' and generated[-1][1] == 1
 
 
+@pytest.mark.parametrize("route", ["writer", "constrained", "base", "polish"])
+@pytest.mark.parametrize("terminal", [19998, 19999, 20000])
+def test_all_native_routes_stop_on_configured_or_chat_eos_at_capacity(route, terminal, monkeypatch, caplog):
+    from types import SimpleNamespace
+    import torch
+    from transformers import EosTokenCriteria
+    from nodes import _otr_constrained_generate as constrained
+    from nodes._otr_generation_budget import ProviderCapacityMessages
+
+    entry, *_ = _exact_prompt_entry(monkeypatch)
+    caplog.set_level("INFO", logger=writer.log.name)
+    configured = [19998, 19999]
+    entry["model"].generation_config = SimpleNamespace(eos_token_id=configured)
+    entry["model"].config = SimpleNamespace(eos_token_id=12345)
+    messages = ProviderCapacityMessages([{"role": "user", "content": "Reply."}])
+    prepared = model_loader.prepare_native_prompt(entry, messages)
+    entry["context_cap"] = prepared["prompt_tokens"] + 1
+    calls = []
+
+    def generate(**kwargs):
+        calls.append(kwargs)
+        assert kwargs["eos_token_id"] == [19998, 19999, 20000]
+        assert kwargs["pad_token_id"] == 20000
+        output = torch.cat([kwargs["input_ids"], torch.tensor([[terminal]])], dim=1)
+        # Exercise Transformers' actual stop criterion with the transport kwargs.
+        assert EosTokenCriteria(kwargs["eos_token_id"])(output, None).item()
+        return output
+
+    entry["model"].generate = generate
+    factories = {"writer": writer._build_truncating_generate_fn,
+                 "constrained": lambda e: constrained.make_constrained_generate_fn(e, _FitSchema),
+                 "base": model_loader.make_generate_fn, "polish": model_loader.make_polish_generate_fn}
+    assert factories[route](entry)(messages, temperature=.2, max_new_tokens=None) == '{"value":"ok"}'
+    assert len(calls) == 1 and calls[0]["max_new_tokens"] == 1
+    assert configured == [19998, 19999] and entry["tokenizer"].eos_token_id == 20000
+    if route == "writer":
+        assert f"last_token={terminal} eos_token_ids=[19998, 19999, 20000] ended_with_eos=True" in caplog.text
+
+
+@pytest.mark.parametrize("configured,expected", [
+    ([7, 8, 7], [7, 8, 9]), (0, [0, 9]), (None, [5, 6, 9]),
+])
+def test_native_eos_respects_generation_precedence_and_nested_fallback(configured, expected):
+    from types import SimpleNamespace
+    import copy
+    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=configured),
+                            config=SimpleNamespace(eos_token_id=4,
+                                                   text_config=SimpleNamespace(eos_token_id=[5, 6])))
+    tokenizer = SimpleNamespace(eos_token_id=9)
+    before = copy.deepcopy((model, tokenizer))
+    assert model_loader.native_eos_token_ids({"model": model, "tokenizer": tokenizer}) == expected
+    assert (model, tokenizer) == before
+
+
+def test_native_eos_collection_normalization_keeps_padding_scalar(monkeypatch):
+    entry, *_ = _exact_prompt_entry(monkeypatch)
+    entry["tokenizer"].eos_token_id = {20000, 19999}
+    prepared = model_loader.prepare_native_prompt(entry, [{"role": "user", "content": "Reply."}])
+    assert prepared["eos_token_ids"] == [19999, 20000]
+    assert prepared["pad_token_id"] == 19999
+    assert model_loader._native_token_ids({True, -1, "bad", None, 8}) == [8]
+
+
 def test_unmarked_none_budget_is_a_programmer_error_before_prompt_preparation(monkeypatch):
     entry, runs, moves, refs, generated = _exact_prompt_entry(monkeypatch)
     with pytest.raises(TypeError, match="provider-capacity message contract"):
@@ -453,7 +516,7 @@ def test_local_structured_transport_adds_prefix_without_losing_sampling(
     parser = object()
     monkeypatch.setattr(
         "nodes._otr_constrained_generate.get_cached_transformers_schema_constraint",
-        lambda cache_entry, schema_model: (parser, prefix),
+        lambda cache_entry, schema_model, **kwargs: (parser, prefix),
     )
     model = Model()
     generate = writer._build_truncating_generate_fn(
diff --git a/tests/test_writer_sampling_seed.py b/tests/test_writer_sampling_seed.py
index 591bb124..f3a9c5bf 100644
--- a/tests/test_writer_sampling_seed.py
+++ b/tests/test_writer_sampling_seed.py
@@ -150,31 +150,60 @@ def test_two_different_seeds_really_do_diverge(monkeypatch):
 # WIRING: both generate call sites are covered.
 # --------------------------------------------------------------------------- #
 
-def test_both_generate_paths_are_seeded():
+@pytest.mark.parametrize("seeded", [True, False])
+def test_both_generate_paths_are_seeded(monkeypatch, seeded):
     """The min_p retry re-generates after a failed attempt has already consumed
     RNG state. Without a re-seed there, a run that hit the TypeError would
     diverge from one that did not -- reproducible only by luck."""
-    import ast
-    import inspect
-    import textwrap
-
-    source = inspect.getsource(writer)
-    tree = ast.parse(source)
-    generate_lines, seed_lines = [], []
-    for node in ast.walk(tree):
-        if not isinstance(node, ast.Call):
-            continue
-        func = node.func
-        if isinstance(func, ast.Attribute) and func.attr == "generate":
-            generate_lines.append(node.lineno)
-        if isinstance(func, ast.Name) and func.id == "_seed_writer_sampling":
-            seed_lines.append(node.lineno)
-    assert len(generate_lines) >= 2, generate_lines
-    for line in generate_lines:
-        assert any(0 < line - seed < 12 for seed in seed_lines), (
-            "model.generate at line %d has no _seed_writer_sampling call "
-            "shortly before it; that path is not reproducible" % line)
-    del textwrap
+    torch = pytest.importorskip("torch")
+    draws, min_p_values = [], []
+    if seeded:
+        monkeypatch.setenv(writer.WRITER_SEED_ENV, "42")
+    else:
+        monkeypatch.delenv(writer.WRITER_SEED_ENV, raising=False)
+
+    class Inputs(dict):
+        def to(self, device):
+            return self
+
+    class Tokenizer:
+        eos_token_id = 99
+
+        def apply_chat_template(self, messages, **kwargs):
+            return "Write this story."
+
+        def __call__(self, prompt, **kwargs):
+            return Inputs(input_ids=torch.tensor([[1, 2, 3]]))
+
+        def decode(self, tokens, **kwargs):
+            return "The story."
+
+    class Model:
+        device = "cpu"
+
+        def generate(self, **kwargs):
+            draws.append(torch.rand(4).tolist())
+            min_p_values.append(kwargs.get("min_p"))
+            if "min_p" in kwargs:
+                # An unsupported-argument failure can consume RNG state.
+                raise TypeError("unexpected keyword argument 'min_p'")
+            return torch.cat([kwargs["input_ids"], torch.tensor([[99]])], dim=1)
+
+    monkeypatch.setattr(writer._OTRHB, "make_streamer", lambda *args: None)
+    generate = writer._build_truncating_generate_fn(
+        {"model": Model(), "tokenizer": Tokenizer(), "context_cap": 1024},
+        min_p=.05,
+    )
+    messages = [{"role": "user", "content": "Write this story."}]
+    # First call exercises normal generation then the TypeError retry; the
+    # second uses the remembered unsupported-min_p path without another retry.
+    assert generate(messages, temperature=.5, max_new_tokens=20) == "The story."
+    assert generate(messages, temperature=.5, max_new_tokens=20) == "The story."
+    assert min_p_values == [.05, None, None]
+    if seeded:
+        assert draws[0] == draws[1] == draws[2]
+    else:
+        assert draws[0] != draws[1] != draws[2]
 
 
 def test_production_sampling_is_still_unseeded_by_default():

```
