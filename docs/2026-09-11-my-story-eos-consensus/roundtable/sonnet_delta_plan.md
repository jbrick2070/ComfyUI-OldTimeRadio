# Sonnet scoped final QA: completion diagnostics delta
One final concrete correction after your finished-code QA. Review ONLY the actual writer classification block and associated regression below; <=400 words. No further architecture audit is requested.

Your earlier review requested the previously truncated capacity branch. It is now supplied in full. Root verified EOS-erasure concern cannot occur with a valid configured EOS: shared resolver takes nonempty generation_config IDs first. Explicit None disables EOS only when no supported config/chat terminator is present. No new rejection is being added for a hypothetical future malformed model config; Qwen has248044+248046. This is not an unresolved source-fidelity remedy.

Root found one actual diagnostic bug in the previously existing writer branch: at exact output capacity it logged OUTPUT_TRUNCATED/OUTPUT_CAP even when ended_with_eos=True and the later raise correctly did not fire. Changed exactly ONE condition to `if generated_tokens == effective_max_new_tokens and not ended_with_eos:` for those diagnostics. The actual later capacity raise already has not ended_with_eos. Guard.hit still runs first and always raises degeneracy. No sampling/length/correction/retry outcome change.

Existing parametrized test_all_native_routes_stop_on_configured_or_chat_eos_at_capacity drives all4factories and3terminalIDs through actual EosTokenCriteria at capacity, asserts return, and now asserts writer log has correct last_token/EOS/True but contains neither OUTPUT_CAP nor OUTPUT_TRUNCATED. Actual nonEOS capacity/decode regression runs alongside it. Full regression and controlled Bible are being repeated on final revision. Is there a concrete defect in THIS final diagnostic change? If none, say no remaining must-fix in this scoped delta. Live cure remains unproven; one fresh canonical retry after QA/push, unchanged controls.

## Full writer owner
```python
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
1213:                         "for one. Any JSON parse failure below is a runaway "
1214:                         "decode, not a budget defect.",
1215:                         effective_max_new_tokens, context_cap, prompt_len,
1216:                     )
1217:                 else:
1218:                     log.error(
1219:                         "[OTR_LedgerScriptWriter] OUTPUT_TRUNCATED: generation "
1220:                         "stopped at the ceiling after a CLAMP. The caller asked "
1221:                         "for %d output tokens; the %d-token context window left "
1222:                         "only %d after a %d-token prompt. Any JSON parse failure "
1223:                         "below is this budget, not the model. Give this pass a "
1224:                         "slot whose window fits prompt+artifact.",
1225:                         requested_max_new_tokens, context_cap,
1226:                         effective_max_new_tokens, prompt_len,
1227:                     )
1228:             else:
1229:                 log.warning(
1230:                     "[OTR_LedgerScriptWriter] OUTPUT_CAP: generation stopped at "
1231:                     "the caller's own ceiling (prompt_tokens=%d "
1232:                     "generated_tokens=%d max_new_tokens=%d); output may be "
1233:                     "truncated.",
1234:                     prompt_len, generated_tokens, effective_max_new_tokens,
1235:                 )
1236:         if (generated_tokens == effective_max_new_tokens
1237:                 and fail_on_output_limit and not ended_with_eos):
1238:             # LOG THE EVIDENCE BEFORE DISCARDING IT (2026-08-13).
1239:             #
1240:             # `raw_completion=decoded` below has been attached to this exception
1241:             # since A-1 and NOTHING has ever read it -- the leg log prints
1242:             # "raw head: <empty>". So at the one moment thousands of tokens of
1243:             # runaway text exist in memory, they are thrown away, and the next
1244:             # reader has to reproduce a 20-minute decode to learn what the model
1245:             # was actually saying. Two runaways in one night were diagnosed by
1246:             # inference for exactly this reason.
1247:             #
1248:             # Head AND tail, because they answer different questions: the head
1249:             # says what the model was writing, the tail says what it was doing
1250:             # when it ran out of room. A verbatim loop or digit run in the tail
1251:             # means degeneracy; varied run-on prose means it was hedging and
1252:             # could not find a way to end the sentence. That distinction decides
1253:             # whether the cure is a decode guard or the pack's own wording, and
1254:             # it is one log line away.
1255:             _raw = decoded or ""
1256:             _head = _raw[:400].replace("\n", " ")
1257:             _tail = _raw[-400:].replace("\n", " ")
1258:             log.error(
1259:                 "[OTR_LedgerScriptWriter] RUNAWAY EVIDENCE (%d chars, %d "
1260:                 "tokens, ended_with_eos=%s)\n  HEAD: %s\n  TAIL: %s",
1261:                 len(_raw), generated_tokens, ended_with_eos, _head, _tail,
1262:             )
1263:             raise PromptContextOverflowError(
1264:                 "prose generation exhausted the full remaining provider/context "
1265:                 f"capacity ({effective_max_new_tokens} output tokens after a "
1266:                 f"{prompt_len}-token prompt); the partial artifact is discarded, "
1267:                 "never repaired as prose",
1268:                 # A-4: THIS is the phase a re-roll can actually fix -- the call
1269:                 # RAN, and sampling is stochastic (nine engines in the live
1270:                 # 45-word campaign produced both a pass and a fail on
1271:                 # byte-identical code). The message lost its old tail, "not
1272:                 # eligible for a prose or structural reroll", because A-4 makes
1273:                 # the second half of that false: the ladder may now re-roll
1274:                 # this pass. What stays true, and stays said, is that the
1275:                 # partial artifact is never handed to a prose repair. Every
1276:                 # OTHER transport's capacity refusal carries no phase, so it
1277:                 # stays terminal and its own message stays accurate.
1278:                 phase=CAPACITY_PHASE_OUTPUT_LIMIT,
1279:                 raw_completion=decoded,
1280:                 prompt_tokens=prompt_len,
1281:                 generated_tokens=generated_tokens,
1282:                 requested_output_tokens=requested_max_new_tokens,
1283:                 effective_output_tokens=effective_max_new_tokens,
1284:                 context_cap=context_cap,
1285:                 ended_with_eos=ended_with_eos,
1286:             )
1287:         # Tier 1 fix #5 (2026-05-11): StoppingCriteria halts
1288:         # generation but leaves the trigger bytes in the output
1289:         # buffer. Slice at the first stop substring so leaked
1290:         # bracketed/parenthesized tails don't survive into the
1291:         # composer's strip_line_formatting -> ledger pipeline. With
1292:         # polish OFF (default), this is the last guard before the
1293:         # text lands. Earliest-match wins.
1294:         if stop:
1295:             cut = len(decoded)
1296:             for s in stop:
1297:                 if not s:
1298:                     continue
1299:                 idx = decoded.find(s)
1300:                 if idx >= 0 and idx < cut:
1301:                     cut = idx
1302:             decoded = decoded[:cut]
1303:         return decoded
1304: 
1305:     if schema_model is not None:
1306:         generate_fn.schema_model = schema_model  # type: ignore[attr-defined]
```

## Regression
```python
@pytest.mark.parametrize("route", ["writer", "constrained", "base", "polish"])
@pytest.mark.parametrize("terminal", [19998, 19999, 20000])
def test_all_native_routes_stop_on_configured_or_chat_eos_at_capacity(route, terminal, monkeypatch, caplog):
    from types import SimpleNamespace
    import torch
    from transformers import EosTokenCriteria
    from nodes import _otr_constrained_generate as constrained
    from nodes._otr_generation_budget import ProviderCapacityMessages

    entry, *_ = _exact_prompt_entry(monkeypatch)
    caplog.set_level("INFO", logger=writer.log.name)
    configured = [19998, 19999]
    entry["model"].generation_config = SimpleNamespace(eos_token_id=configured)
    entry["model"].config = SimpleNamespace(eos_token_id=12345)
    messages = ProviderCapacityMessages([{"role": "user", "content": "Reply."}])
    prepared = model_loader.prepare_native_prompt(entry, messages)
    entry["context_cap"] = prepared["prompt_tokens"] + 1
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        assert kwargs["eos_token_id"] == [19998, 19999, 20000]
        assert kwargs["pad_token_id"] == 20000
        output = torch.cat([kwargs["input_ids"], torch.tensor([[terminal]])], dim=1)
        # Exercise Transformers' actual stop criterion with the transport kwargs.
        assert EosTokenCriteria(kwargs["eos_token_id"])(output, None).item()
        return output

    entry["model"].generate = generate
    factories = {"writer": writer._build_truncating_generate_fn,
                 "constrained": lambda e: constrained.make_constrained_generate_fn(e, _FitSchema),
                 "base": model_loader.make_generate_fn, "polish": model_loader.make_polish_generate_fn}
    assert factories[route](entry)(messages, temperature=.2, max_new_tokens=None) == '{"value":"ok"}'
    assert len(calls) == 1 and calls[0]["max_new_tokens"] == 1
    assert configured == [19998, 19999] and entry["tokenizer"].eos_token_id == 20000
    if route == "writer":
        assert f"last_token={terminal} eos_token_ids=[19998, 19999, 20000] ended_with_eos=True" in caplog.text
        assert "OUTPUT_CAP:" not in caplog.text
        assert "OUTPUT_TRUNCATED:" not in caplog.text
```
