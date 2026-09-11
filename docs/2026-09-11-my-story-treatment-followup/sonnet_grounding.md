# Final Sonnet grounding: no additional code revision

Your prior response labels a possible stale completion both "reachable must-fix"
and "plausible but unconfirmed". Root has read the missing owners below. Please
review only whether a concrete current path invalidates this disposition. Do
not convert a future arbitrary exception reuse into a present bug. Return <=500
words: remaining reachable must-fix or no demonstrated must-fix, with limits.

The final production code has not changed since your review. New tests distinguish
two actual sequential generation-error objects and prove third repair uses the
latest second completion, not thefirst. Focused193pass; Bible37/10inherited versus
35/12baseline, newcontextguardsfailbaseline/passcandidate. Fullsuiteinprogress.

## Root disposition

1. Every native degeneracy/output-limit exception is newly constructed AFTER
that call's out tensor is decoded. raw_completion=that_call_raw. Sharedladder
sets last_raw="" beforeeachcall and assigns last_error=exc inside its caught
failure; finalrepair gets this same last_error. There is no error object/cache
on theclosure that can preserve an older attempt's completion.
2. Ordinary JSON syntax/schema errors do not acquire raw_completion anywhere
in this path. Provider/OOM/cancel are nonretryable and escape beforethisfactory.
The source helper alreadyuses the same exception evidence for journaling; the
liveledger records allthree exact distinct completions. Failedtext stays only
failureevidence; counters neverparse it as accepted proposals.
3. The sharedcomment is intentionally about the defaultshared ladder, which
stillpasseslast_raw only and has NOT changed. The localfactory'snewcomment
explicitly names the lane-specific exception and why genericcallersunchanged.
The operator alreadyrequires thislane toreceive fullfailedartifact (PBUG01,
Bible11.48), and specifically requested checker+rewriter. No separate approval
needed tofix thisexistingowner. No source truncation or extra budgets added.
4. Duplicateasserts in thetest are harmless: inside ensuresno4thcall, outside
ensuresit actually reached3. No productionbug.
5. Interrupted meansgeneration raised before returning; it is not a claim of
operatorcancellation. Both actualcapacity anddecodelivenesserrors have this
property. Rawinvalidnonstringattributes are ignored, notstringified.

## Missing exact owners


### nodes/_otr_generation_budget.py

105:     which is documented pure and may not import the writer -- can name the
106:     type it is deciding about. The writer re-exports it, so
107:     ``writer.PromptContextOverflowError`` is the same object it always was.
108: 
109:     A-1 (2026-07-30): the output-limit raise carries the completion the model
110:     actually produced plus the token arithmetic, as FIELDS. Never in the
111:     message -- a ~14,000-token artifact inside an exception string floods
112:     every log and receipt that formats it, and the ladder's disposition lines
113:     are read by humans. A caller that wants the evidence asks for it by name.
114: 
115:     Every field is optional: the prompt-side re-wrap knows none of them, and
116:     reports ``None`` rather than a guess.
117:     """
118: 
119:     def __init__(
120:         self,
121:         message: str,
122:         *,
123:         phase: str = CAPACITY_PHASE_PROMPT_NO_ROOM,
124:         raw_completion: str | None = None,
125:         prompt_tokens: int | None = None,
126:         generated_tokens: int | None = None,
127:         requested_output_tokens: int | None = None,
128:         effective_output_tokens: int | None = None,
129:         context_cap: int | None = None,
130:         ended_with_eos: bool | None = None,
131:     ) -> None:
132:         super().__init__(message, phase=phase)
133:         self.raw_completion = raw_completion
134:         self.prompt_tokens = prompt_tokens
135:         self.generated_tokens = generated_tokens
136:         self.requested_output_tokens = requested_output_tokens
137:         self.effective_output_tokens = effective_output_tokens
138:         self.context_cap = context_cap
139:         self.ended_with_eos = ended_with_eos
140: 
141: 
142: class GenerationDegeneracyError(PromptContextOverflowError):
143:     """The transport HALTED a decode that had stopped steering.
144: 
145:     A subtype rather than a reused phase string, so a caller can tell "the model
146:     filled its allowance" from "we stopped the model" without string-matching.
147:     It inherits the evidence fields (`raw_completion`, the token arithmetic) and
148:     adds the halt's own: which signal fired, and how long the string had been
149:     open when it did.
150: 
151:     NOT a quality judgement and NOT terminal -- see `_otr_decode_guard`.
152:     """
153: 
154:     def __init__(
155:         self,
156:         message: str,
157:         *,
158:         halt_reason: str | None = None,
159:         open_string_tokens: int | None = None,
160:         repetition: dict | None = None,
161:         **kwargs: Any,
162:     ) -> None:
163:         kwargs.setdefault("phase", CAPACITY_PHASE_DECODE_DEGENERACY)
164:         super().__init__(message, **kwargs)
165:         self.halt_reason = halt_reason
166:         self.open_string_tokens = open_string_tokens
167:         self.repetition = repetition or {}
168: 
169: 
170: CAPACITY_ERRORS = (GenerationContextOverflowError, PromptContextOverflowError)
171: 
172: 
173: def is_rerollable_generation_error(error: Any) -> bool:
174:     """Return True only for a generation failure a re-roll could actually fix.
175: 
176:     ONE predicate, one owner: the transports raise the phase and the ladder
177:     asks this. A `prompt_no_room` failure answers False forever -- the
178:     arithmetic that refused it is deterministic. Both `output_limit` (the model
179:     filled its allowance) and `decode_degeneracy` (we halted it) are honest
180:     second chances, because sampling is stochastic and the ladder's next rung
181:     runs at a lower temperature.
182:     """
183:     return (
184:         isinstance(error, CAPACITY_ERRORS)
185:         and getattr(error, "phase", None) in REROLLABLE_PHASES
186:     )
187: 
188: 
189: #: Historical name. Kept because the ladder, the lower-temperature rung and the
190: #: candidate loop all call it, and renaming across three call sites in the same
191: #: change as a new failure mode is how a rename becomes a regression. The name
192: #: now understates what it covers, which is why the new name exists beside it.
193: is_rerollable_capacity_error = is_rerollable_generation_error
194: 
195: 
196: def estimate_prompt_tokens(messages: Any) -> int:
197:     """Return a conservative provider-independent prompt-token estimate.
198: 

### nodes/_otr_structured_call.py

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

### nodes/OTR_LedgerScriptWriter.py

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
1184:         if generated_tokens == effective_max_new_tokens and not ended_with_eos:
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
1201:                     # pass a slot whose window fits prompt+artifact" -- but
1202:                     # this pass already HAS every token there is, so there is
1203:                     # no bigger slot to give it and no config defect to find.
1204:                     # It sent a live session hunting one for twenty minutes.
1205:                     # When the pass reserved everything and still hit the
1206:                     # ceiling without an EOS, the model did not stop.
1207:                     log.error(
1208:                         "[OTR_LedgerScriptWriter] OUTPUT_TRUNCATED: this pass "
1209:                         "reserved ALL remaining output capacity (%d of the "
1210:                         "%d-token window, after a %d-token prompt) and still "
1211:                         "ran to the ceiling. THE MODEL DID NOT STOP -- there "
1212:                         "is no larger slot to move it to, so do not go looking "
