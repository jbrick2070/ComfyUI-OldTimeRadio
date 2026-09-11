# Exact source grounding for API Opus

The API reviewer receives these source excerpts; it cannot open local files. Paths and line numbers refer to the frozen five-file candidate where applicable. Other source is current unchanged code. Root grounds all claims.

## nodes/_otr_model_loader.py:691 -- _native_text_load_config
Source SHA256 468597c196625de8ef5700ff53ac1d68394aa893d6694a6ef914266c9713ec61
```python
691: def _native_text_load_config(model_config):
692:     """The text sub-config for a multimodal row loaded text-only.
693: 
694:     FAMILY-AGNOSTIC BY DESIGN. It asks only what it actually needs: does the
695:     parent config expose a ``text_config`` that names its own ``model_type``?
696:     Verified shapes at the time of writing, read from the real config.json of
697:     each cached row:
698: 
699:         gemma4          -> gemma4_text           (E2B, E4B)
700:         qwen3_5         -> qwen3_5_text          (Qwen3.5-4B)
701:         gemma4_unified  -> gemma4_unified_text   (12B -- NOT opted in)
702: 
703:     Note the deliberate absence of a model_type allowlist. The row opts in
704:     through the catalog's ``text_only_load`` field, which is a reviewed
705:     per-row decision; re-checking the family here would just be a second
706:     ladder to forget to update. What this function DOES enforce is that the
707:     config can actually be split, and it raises rather than silently handing
708:     back the composite -- a quiet fallback here would reintroduce the exact
709:     tower-loading bug this path exists to remove.
710: 
711:     This is a sibling of ``_e4b_text_offload_config`` rather than a
712:     replacement: that one serves the E4B CPU-offload RETRY and is pinned by
713:     its own exact-E4B test contracts. This one serves the INITIAL load.
714:     """
715:     from copy import deepcopy
716: 
717:     text_config = getattr(model_config, "text_config", None)
718:     parent_type = getattr(model_config, "model_type", None)
719:     text_type = getattr(text_config, "model_type", None)
720:     if text_config is None or not text_type:
721:         raise ModelLoaderError(
722:             f"native text load requires a text_config with a model_type; "
723:             f"{parent_type!r} exposes {text_type!r}. This row is marked "
724:             f"text_only_load=native_text_decoder in the catalog but its "
725:             f"checkpoint config cannot be split -- fix the row, do not fall "
726:             f"back to the composite load."
727:         )
728:     return deepcopy(text_config)
```

## nodes/_otr_model_loader.py:2270 -- _native_token_ids
Source SHA256 468597c196625de8ef5700ff53ac1d68394aa893d6694a6ef914266c9713ec61
```python
2270: def _native_token_ids(value) -> list[int]:
2271:     values = value if isinstance(value, (list, tuple, set)) else (value,)
2272:     normalized = [int(item) for item in values
2273:                   if isinstance(item, Integral) and not isinstance(item, bool) and item >= 0]
2274:     return sorted(normalized) if isinstance(value, set) else normalized
```

## nodes/_otr_model_loader.py:2277 -- native_eos_token_ids
Source SHA256 468597c196625de8ef5700ff53ac1d68394aa893d6694a6ef914266c9713ec61
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

## nodes/_otr_model_loader.py:2294 -- prepare_native_prompt
Source SHA256 468597c196625de8ef5700ff53ac1d68394aa893d6694a6ef914266c9713ec61
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

## nodes/_otr_model_loader.py:2339 -- inspect_native_prompt_fit
Source SHA256 468597c196625de8ef5700ff53ac1d68394aa893d6694a6ef914266c9713ec61
```python
2339: def inspect_native_prompt_fit(cache_entry: dict[str, Any], messages, *, max_new_tokens) -> dict[str, Any]:
2340:     """Non-generating exact-token measurement, with only primitive results."""
2341:     if max_new_tokens is None and not getattr(messages, "_otr_reserve_remaining_output_capacity", False):
2342:         raise TypeError("max_new_tokens=None requires the provider-capacity message contract")
2343:     prepared = prepare_native_prompt(cache_entry, messages)
2344:     measured = {key: prepared[key] for key in (
2345:         "prompt_tokens", "context_cap", "capacity_source", "capacity_known", "model_id",
2346:         "eos_token_ids",
2347:     )}
2348:     requested = (prepared["context_cap"] if prepared["reserve_remaining"]
2349:                  else max(1, int(max_new_tokens)))
2350:     measured.update(supported=True, requested_output_tokens=requested,
2351:                     available_output_tokens=max(0, prepared["context_cap"] - prepared["prompt_tokens"]))
2352:     try:
2353:         effective = fit_output_tokens(
2354:             requested, context_cap=prepared["context_cap"],
2355:             prompt_tokens=prepared["prompt_tokens"], label="prompt inspection",
2356:             require_full=(prepared["require_full_output"]
2357:                           or (prepared["reserve_remaining"] and max_new_tokens is not None)),
2358:         )
2359:     except GenerationContextOverflowError as exc:
2360:         measured.update(fits=False, effective_output_tokens=0, phase=exc.phase, reason=str(exc))
2361:     else:
2362:         measured.update(fits=True, effective_output_tokens=effective)
2363:     return measured
```

## nodes/_otr_model_loader.py:2366 -- make_generate_fn
Source SHA256 468597c196625de8ef5700ff53ac1d68394aa893d6694a6ef914266c9713ec61
```python
2366: def make_generate_fn(cache_entry: dict[str, Any]):
2367:     """Wrap a cache_entry into the GenerateFn callable.
2368: 
2369:     Returns a callable matching:
2370:         (messages, *, temperature, max_new_tokens) -> str
2371: 
2372:     where `messages` is a list[dict] in chat format
2373:     ([{"role": "system", "content": ...}, {"role": "user", "content": ...}])
2374:     and the return is the raw decoded string from the model with the
2375:     prompt prefix removed.
2376: 
2377:     Generation params hardcoded for the v2.0 path:
2378:         do_sample=True
2379:         top_p=0.92
2380: 
2381:     Caller controls temperature and max_new_tokens per call.
2382: 
2383:     Raises ModelLoaderError if the cache_entry is missing required
2384:     keys or if torch is not importable at first call time.
2385:     """
2386:     # [OpenRouter S3] Remote branch (FC2 seam 2). A provider-tagged
2387:     # remote entry has no model/tokenizer; return the remote generate_fn
2388:     # before the local-key check below. Uses zero local VRAM.
2389:     if cache_entry.get("provider") == "openrouter":
2390:         from ._otr_openrouter_backend import make_openrouter_generate_fn
2391:         return make_openrouter_generate_fn(cache_entry)
2392:     # BUG-LOCAL-299: Comfy Credits sibling -- same zero-VRAM remote seam.
2393:     if cache_entry.get("provider") == "comfy_credits":
2394:         from ._otr_comfy_backend import make_comfy_credits_generate_fn
2395:         return make_comfy_credits_generate_fn(cache_entry)
2396:     if cache_entry.get("provider") == "google_api":
2397:         from ._otr_google_api.llm import make_google_api_generate_fn
2398:         return make_google_api_generate_fn(cache_entry)
2399:     # Native GGUF lane: in-process llama-cpp-python, no daemon or port.
2400:     if cache_entry.get("provider") == "gguf_native":
2401:         from ._otr_gguf_backend import make_gguf_generate_fn
2402:         return make_gguf_generate_fn(cache_entry)
2403:     required = {"model", "tokenizer"}
2404:     missing = required - set(cache_entry)
2405:     if missing:
2406:         raise ModelLoaderError(
2407:             f"cache_entry missing required keys: {sorted(missing)}"
2408:         )
2409: 
2410:     model = cache_entry["model"]
2411:     tokenizer = cache_entry["tokenizer"]
2412: 
2413:     def generate_fn(messages, *, temperature, max_new_tokens):
2414:         # Lazy torch import. Raised as ModelLoaderError to match the
2415:         # facade's exception contract.
2416:         try:
2417:             import torch
2418:         except ImportError as exc:
2419:             raise ModelLoaderError("torch not available") from exc
2420: 
2421:         require_full_output = bool(getattr(
2422:             messages, "_otr_require_full_output_budget", False,
2423:         ))
2424:         reserve_remaining = bool(getattr(
2425:             messages, "_otr_reserve_remaining_output_capacity", False,
2426:         ))
2427:         bounded_capacity = reserve_remaining and max_new_tokens is not None
2428:         fail_on_output_limit = bool(getattr(
2429:             messages, "_otr_fail_on_output_limit", False,
2430:         ))
2431:         prepared = prepare_native_prompt(cache_entry, messages)
2432:         inputs = prepared["inputs"]
2433:         context_cap = prepared["context_cap"]
2434:         requested_tokens = context_cap if reserve_remaining else max_new_tokens
2435:         try:
2436:             effective_max_new_tokens = fit_output_tokens(
2437:                 requested_tokens,
2438:                 context_cap=context_cap,
2439:                 prompt_tokens=inputs["input_ids"].shape[1],
2440:                 label=f"local model {cache_entry.get('model_id', '<unknown>')}",
2441:                 require_full=require_full_output or bounded_capacity,
2442:             )
2443:         except GenerationContextOverflowError as exc:
2444:             # THE TWO LOCAL TRANSPORTS MUST AGREE (2026-08-13). This used to
2445:             # raise a bare ModelLoaderError with NO phase for the exact
2446:             # condition OTR_LedgerScriptWriter raises as a phase-carrying
2447:             # PromptContextOverflowError. The ladder reads the PHASE to decide
2448:             # whether a failure is rerollable, so an identical runaway was
2449:             # rerollable on one transport and terminal on the other, purely
2450:             # from which one the pass happened to take. The phase is read off
2451:             # the error rather than assumed here, so a pre-call refusal that IS
2452:             # retryable cannot be mislabelled by this line.
2453:             raise PromptContextOverflowError(
2454:                 str(exc), phase=exc.phase,
2455:             ) from exc
2456:         inputs = inputs.to(model.device)
2457:         # THE LIVENESS GUARD (2026-08-13). This transport was unprotected when
2458:         # the guard shipped, because the guard was installed per-WRAPPER in
2459:         # OTR_LedgerScriptWriter instead of at every local generate().
2460:         #
2461:         # Today's live callers all pass small numeric caps (8-300 tokens), so
2462:         # this is prophylaxis, not an emergency. It is worth doing anyway
2463:         # because THIS transport honours reserve-remaining -- the comment
2464:         # below concedes it "can legitimately be handed >14k output tokens" --
2465:         # so the first caller that reserves the window here would reopen the
2466:         # 22-minute runaway with nothing watching it. A guard installed only
2467:         # where the runaway happened to occur is a guard waiting to be missed.
2468:         from transformers import StoppingCriteriaList  # noqa: I001
2469:         try:
2470:             from ._otr_decode_guard import make_degeneracy_criterion
2471:         except ImportError:  # pragma: no cover - flat/standalone import path
2472:             from _otr_decode_guard import (  # type: ignore
2473:                 make_degeneracy_criterion,
2474:             )
2475:         _guard = make_degeneracy_criterion(inputs["input_ids"].shape[1])
2476:         _deadline_guard = _DeadlineStoppingCriteria()
2477: 
2478:         with torch.no_grad():
2479:             out = model.generate(
2480:                 **inputs,
2481:                 do_sample=True,
2482:                 temperature=temperature,
2483:                 top_p=0.92,
2484:                 max_new_tokens=effective_max_new_tokens,
2485:                 pad_token_id=prepared["pad_token_id"],
2486:                 eos_token_id=prepared["eos_token_ids"] or None,
2487:                 stopping_criteria=StoppingCriteriaList(
2488:                     [_guard, _deadline_guard]),
2489:                 # Read-only live heartbeat (2026-08-13). A reserve-remaining
2490:                 # pass can legitimately be handed >14k output tokens, and with
2491:                 # no streamer that is a silent twenty-minute wait whose only
2492:                 # signal arrives at the ceiling. None when disabled.
2493:                 streamer=_OTRHB.make_streamer(
2494:                     tokenizer,
2495:                     f"llm:{cache_entry.get('model_id', '<unknown>')}"),
2496:             )
2497:         _memory_log.memory_snapshot("base_generation_returned", model_id=cache_entry.get("model_id"))
2498:         # Strip prompt prefix from decoded output.
2499:         prompt_len = inputs["input_ids"].shape[1]
2500:         generated_ids = out[0][prompt_len:]
2501:         if _deadline_guard.hit:
2502:             # Checked FIRST: a deadline hit means this call's owning
2503:             # _run_with_timeout has (or is about to have) given up on it --
2504:             # raising here rather than returning text is what stops a
2505:             # truncated result from racing through as a silent success.
2506:             # See GenerationDeadlineExceededError's docstring.
2507:             log.warning(
2508:                 "[OTR.llm] generation deadline exceeded after %s of a "
2509:                 "%s-token allowance -- raising instead of returning "
2510:                 "truncated text (PBUG-20260825-04)",
2511:                 len(generated_ids), effective_max_new_tokens,
2512:             )
2513:             raise GenerationDeadlineExceededError(
2514:                 "generation was cut short by its caller's wall-clock "
2515:                 "deadline; the caller has abandoned this call",
2516:                 generated_tokens=len(generated_ids),
2517:             )
2518:         if getattr(_guard, "hit", False):
2519:             # Classified BEFORE the output-limit check below, for the same
2520:             # reason the writer classifies degeneracy first: a halted decode
2521:             # stops with its allowance unspent, so reporting it as capacity
2522:             # exhaustion would send the next reader hunting a budget defect
2523:             # that does not exist.
2524:             telemetry = _guard.telemetry()
2525:             log.error(
2526:                 "[OTR.llm] DECODE HALTED (%s): repeated a %s-token run "
2527:                 "verbatim %s times after %s generated tokens of a %s-token "
2528:                 "allowance. Telemetry: %s",
2529:                 _guard.reason, telemetry.get("cycle_tokens"),
2530:                 telemetry.get("required_repeats"), len(generated_ids),
2531:                 effective_max_new_tokens, telemetry,
2532:             )
2533:             raise GenerationDegeneracyError(
2534:                 "generation was halted by the liveness guard: the output "
2535:                 "repeated a run of tokens verbatim rather than progressing",
2536:                 halt_reason=_guard.reason,
2537:                 repetition=telemetry,
2538:                 raw_completion=tokenizer.decode(
2539:                     generated_ids, skip_special_tokens=True,
2540:                 ),
2541:                 prompt_tokens=prompt_len,
2542:                 generated_tokens=len(generated_ids),
2543:                 effective_output_tokens=effective_max_new_tokens,
2544:             )
2545:         ended_with_eos = bool(len(generated_ids)) and int(generated_ids[-1]) in prepared["eos_token_ids"]
2546:         if fail_on_output_limit and len(generated_ids) >= effective_max_new_tokens and not ended_with_eos:
2547:             raise ModelLoaderError(
2548:                 "prose generation exhausted the full remaining provider/context "
2549:                 "capacity; the partial artifact is not eligible for reroll"
2550:             )
2551:         return tokenizer.decode(
2552:             generated_ids,
2553:             skip_special_tokens=True,
2554:         )
2555: 
2556:     return generate_fn
```

## nodes/_otr_model_loader.py:2572 -- make_polish_generate_fn
Source SHA256 468597c196625de8ef5700ff53ac1d68394aa893d6694a6ef914266c9713ec61
```python
2572: def make_polish_generate_fn(cache_entry: dict[str, Any]):
2573:     """Build a polish-specific generate fn from `cache_entry`.
2574: 
2575:     LFC sprint commit 3, ADR section 6.4 (2026-05-11). Polish is a
2576:     short, targeted rewrite -- conceptually closer to a constrained
2577:     edit than the composer's long-form generation. The writer's main
2578:     `make_generate_fn` (via the OTR_LedgerScriptWriter
2579:     `_build_truncating_generate_fn` wrapper) bakes
2580:     repetition_penalty / min_p / top_p into its closure tuned for
2581:     composition. Those settings leak into polish via closure capture
2582:     and produce awkward substitutions on short rewrites.
2583: 
2584:     The polish fn here is a SEPARATE closure off the same cache_entry
2585:     with composer-independent sampling:
2586: 
2587:         temperature      -- caller-provided per call (defaults to 0.4
2588:                             via _otr_line_composer.polish_line)
2589:         top_p            -- 0.9 (slightly tighter than composer 0.92)
2590:         do_sample        -- True
2591:         min_p            -- not passed (transformers default 0)
2592:         repetition_penalty -- not passed (transformers default 1.0)
2593: 
2594:     Returns a callable with the same signature as `make_generate_fn`:
2595:         (messages, *, temperature, max_new_tokens) -> str
2596:     """
2597:     # [OpenRouter S3] Remote branch (FC2 seam 2). A provider-tagged
2598:     # remote entry has no model/tokenizer; the remote generate_fn applies
2599:     # the same sampling the caller passes (polish callers pass their own
2600:     # temperature), so one closure covers both factories.
2601:     if cache_entry.get("provider") == "openrouter":
2602:         from ._otr_openrouter_backend import make_openrouter_generate_fn
2603:         return make_openrouter_generate_fn(cache_entry)
2604:     # BUG-LOCAL-299: Comfy Credits sibling -- same zero-VRAM remote seam.
2605:     if cache_entry.get("provider") == "comfy_credits":
2606:         from ._otr_comfy_backend import make_comfy_credits_generate_fn
2607:         return make_comfy_credits_generate_fn(cache_entry)
2608:     if cache_entry.get("provider") == "google_api":
2609:         from ._otr_google_api.llm import make_google_api_generate_fn
2610:         return make_google_api_generate_fn(cache_entry)
2611:     # Native GGUF lane: in-process llama-cpp-python, no daemon or port.
2612:     if cache_entry.get("provider") == "gguf_native":
2613:         from ._otr_gguf_backend import make_gguf_generate_fn
2614:         return make_gguf_generate_fn(cache_entry)
2615:     required = {"model", "tokenizer"}
2616:     missing = required - set(cache_entry)
2617:     if missing:
2618:         raise ModelLoaderError(
2619:             f"cache_entry missing required keys: {sorted(missing)}"
2620:         )
2621: 
2622:     model = cache_entry["model"]
2623:     tokenizer = cache_entry["tokenizer"]
2624: 
2625:     def polish_generate_fn(messages, *, temperature, max_new_tokens):
2626:         try:
2627:             import torch
2628:         except ImportError as exc:
2629:             raise ModelLoaderError("torch not available") from exc
2630: 
2631:         require_full_output = bool(getattr(
2632:             messages, "_otr_require_full_output_budget", False,
2633:         ))
2634:         reserve_remaining = bool(getattr(
2635:             messages, "_otr_reserve_remaining_output_capacity", False,
2636:         ))
2637:         bounded_capacity = reserve_remaining and max_new_tokens is not None
2638:         fail_on_output_limit = bool(getattr(
2639:             messages, "_otr_fail_on_output_limit", False,
2640:         ))
2641:         prepared = prepare_native_prompt(cache_entry, messages)
2642:         inputs = prepared["inputs"]
2643:         context_cap = prepared["context_cap"]
2644:         requested_tokens = context_cap if reserve_remaining else max_new_tokens
2645:         try:
2646:             effective_max_new_tokens = fit_output_tokens(
2647:                 requested_tokens,
2648:                 context_cap=context_cap,
2649:                 prompt_tokens=inputs["input_ids"].shape[1],
2650:                 label=f"local polish {cache_entry.get('model_id', '<unknown>')}",
2651:                 require_full=require_full_output or bounded_capacity,
2652:             )
2653:         except GenerationContextOverflowError as exc:
2654:             # THE TWO LOCAL TRANSPORTS MUST AGREE (2026-08-13). This used to
2655:             # raise a bare ModelLoaderError with NO phase for the exact
2656:             # condition OTR_LedgerScriptWriter raises as a phase-carrying
2657:             # PromptContextOverflowError. The ladder reads the PHASE to decide
2658:             # whether a failure is rerollable, so an identical runaway was
2659:             # rerollable on one transport and terminal on the other, purely
2660:             # from which one the pass happened to take. The phase is read off
2661:             # the error rather than assumed here, so a pre-call refusal that IS
2662:             # retryable cannot be mislabelled by this line.
2663:             raise PromptContextOverflowError(
2664:                 str(exc), phase=exc.phase,
2665:             ) from exc
2666:         inputs = inputs.to(model.device)
2667:         from transformers import StoppingCriteriaList  # noqa: I001
2668:         try:
2669:             from ._otr_decode_guard import make_degeneracy_criterion
2670:         except ImportError:  # pragma: no cover - flat/standalone import path
2671:             from _otr_decode_guard import (  # type: ignore
2672:                 make_degeneracy_criterion,
2673:             )
2674:         _guard = make_degeneracy_criterion(inputs["input_ids"].shape[1])
2675:         _deadline_guard = _DeadlineStoppingCriteria()
2676: 
2677:         with torch.no_grad():
2678:             out = model.generate(
2679:                 **inputs,
2680:                 do_sample=_POLISH_DO_SAMPLE,
2681:                 temperature=temperature,
2682:                 top_p=_POLISH_TOP_P,
2683:                 max_new_tokens=effective_max_new_tokens,
2684:                 pad_token_id=prepared["pad_token_id"],
2685:                 eos_token_id=prepared["eos_token_ids"] or None,
2686:                 stopping_criteria=StoppingCriteriaList(
2687:                     [_guard, _deadline_guard]),
2688:                 streamer=_OTRHB.make_streamer(
2689:                     tokenizer,
2690:                     f"polish:{cache_entry.get('model_id', '<unknown>')}"),
2691:             )
2692:         _memory_log.memory_snapshot("polish_generation_returned", model_id=cache_entry.get("model_id"))
2693:         prompt_len = inputs["input_ids"].shape[1]
2694:         generated_ids = out[0][prompt_len:]
2695:         if _deadline_guard.hit:
2696:             log.warning(
2697:                 "[OTR.llm] polish generation deadline exceeded after %s of "
2698:                 "a %s-token allowance -- raising instead of returning "
2699:                 "truncated text (PBUG-20260825-04)",
2700:                 len(generated_ids), effective_max_new_tokens,
2701:             )
2702:             raise GenerationDeadlineExceededError(
2703:                 "polish generation was cut short by its caller's "
2704:                 "wall-clock deadline; the caller has abandoned this call",
2705:                 generated_tokens=len(generated_ids),
2706:             )
2707:         if getattr(_guard, "hit", False):
2708:             telemetry = _guard.telemetry()
2709:             log.error(
2710:                 "[OTR.llm] POLISH DECODE HALTED (%s): repeated a %s-token run "
2711:                 "verbatim %s times after %s generated tokens of a %s-token "
2712:                 "allowance. Telemetry: %s",
2713:                 _guard.reason, telemetry.get("cycle_tokens"),
2714:                 telemetry.get("required_repeats"), len(generated_ids),
2715:                 effective_max_new_tokens, telemetry,
2716:             )
2717:             raise GenerationDegeneracyError(
2718:                 "polish generation was halted by the liveness guard: the output "
2719:                 "repeated a run of tokens verbatim rather than progressing",
2720:                 halt_reason=_guard.reason,
2721:                 repetition=telemetry,
2722:                 raw_completion=tokenizer.decode(
2723:                     generated_ids, skip_special_tokens=True,
2724:                 ),
2725:                 prompt_tokens=prompt_len,
2726:                 generated_tokens=len(generated_ids),
2727:                 effective_output_tokens=effective_max_new_tokens,
2728:             )
2729:         ended_with_eos = bool(len(generated_ids)) and int(generated_ids[-1]) in prepared["eos_token_ids"]
2730:         if fail_on_output_limit and len(generated_ids) >= effective_max_new_tokens and not ended_with_eos:
2731:             raise ModelLoaderError(
2732:                 "prose generation exhausted the full remaining provider/context "
2733:                 "capacity; the partial artifact is not eligible for reroll"
2734:             )
2735:         return tokenizer.decode(
2736:             generated_ids,
2737:             skip_special_tokens=True,
2738:         )
2739: 
2740:     return polish_generate_fn
```

## nodes/OTR_LedgerScriptWriter.py:548 -- _SlotScheduler
Source SHA256 c25ff162e086182b1b4d290c58bc3469f312917d1540468e5cde71b90c412920
```python
548: class _SlotScheduler:
549:     """Writer-side slot scheduler for the S30 two-model selector.
550: 
551:     Holds the resolved per-slot model ids + the writer's sampling
552:     config. for_slot(slot) returns a generate_fn closure that lazily
553:     request_slot's the right model on every invocation. for_polish()
554:     returns a polish-tuned closure that always routes through the
555:     creative slot.
556: 
557:     Counts transitions and per-slot calls for forensic meta stamping
558:     (meta["slot_transitions"], meta["slot_calls_by_slot"]).
559:     """
560: 
561:     _ALLOWED_SLOTS = ("creative", "technical")
562: 
563:     def __init__(
564:         self,
565:         *,
566:         creative_id: str,
567:         technical_id: str,
568:         top_p: float,
569:         min_p: float,
570:         repetition_penalty: float,
571:         policy: Any = None,
572:         load_config_by_slot: dict | None = None,
573:     ):
574:         self.ids = {
575:             "creative": creative_id,
576:             "technical": technical_id,
577:         }
578:         # S1 platform-portability: the frozen LLMRuntimePolicy threaded
579:         # into every request_slot call (None = nv50 baseline, resolved
580:         # by request_slot itself).
581:         self.policy = policy
582:         # GGUF row registry (2026-07-16): the immutable per-slot GGUF
583:         # load_config (gguf slots only) threaded into request_slot -> backend
584:         # load. Empty for non-GGUF runs (request_slot then uses the policy).
585:         self.load_config_by_slot = load_config_by_slot or {}
586:         self.sampling = {
587:             "top_p": float(top_p),
588:             "min_p": float(min_p or 0.0),
589:             "repetition_penalty": float(repetition_penalty or 1.0),
590:         }
591:         self.transitions = 0
592:         self.calls_by_slot = {"creative": 0, "technical": 0}
593:         self._last_resolved_id: str | None = None
594:         # S32 B6: per-helper / per-phase accounting for forensic meta
595:         # stamping. `slot_calls_by_helper` maps helper-name -> per-slot
596:         # call counts; `slot_transitions_by_phase` is the ordered list
597:         # of (phase_label, from_slot, to_slot, from_id, to_id) tuples
598:         # captured every time a slot transition fires.
599:         self.slot_calls_by_helper: dict[str, dict[str, int]] = {}
600:         self.slot_transitions_by_phase: list[dict] = []
601:         self._current_helper: str | None = None
602:         # Episode-local successful returns only; never retain model handles.
603:         self.successful_model_calls: list[dict] = []
604: 
605:     def _record_successful_model_call(self, slot, helper, entry, base):
606:         def identity(value):
607:             return value if isinstance(value, str) and value.strip() else None
608: 
609:         provider = identity(entry.get("provider")) or "local"
610:         requested = identity(entry.get({
611:             "comfy_credits": "slug", "google_api": "google_model",
612:         }.get(provider, "model_id")))
613:         executed, reported, basis = requested, None, "request_identity"
614:         if provider == "openrouter":
615:             receipt = getattr(base, "_otr_response_model_receipt", None) or {}
616:             requested = identity(receipt.get("requested_model_id")) or identity(entry.get("slug"))
617:             reported = identity(receipt.get("reported_model_id"))
618:             executed = reported
619:             basis = "response_model" if reported else "unreported"
620:         self.successful_model_calls.append({
621:             "helper": identity(helper), "slot": slot, "provider": provider,
622:             "configured_model_id": identity(self.ids[slot]),
623:             "requested_model_id": requested, "executed_model_id": executed,
624:             "reported_model_id": reported, "identity_basis": basis,
625:         })
626: 
627:     def _account_and_get_entry(self, slot: str, *, count_generation: bool = True) -> dict:
628:         """Acquire the configured slot and record actual model transitions.
629: 
630:         Fit inspection shares acquisition without incrementing generation or
631:         helper call counts. Lazy import keeps module loading lightweight.
632:         """
633:         from . import _otr_model_loader as _OTRML
634: 
635:         resolved_id = self.ids[slot]
636:         cache_entry = _OTRML.request_slot(
637:             slot, resolved_id, policy=self.policy,
638:             load_config=self.load_config_by_slot.get(slot),
639:         )
640:         if (
641:             self._last_resolved_id is not None
642:             and self._last_resolved_id != resolved_id
643:         ):
644:             self.transitions += 1
645:             # S32 B6: capture the transition with phase context.
646:             # `_current_helper` is set by the writer via the
647:             # `helper_context()` manager around each helper call.
648:             self.slot_transitions_by_phase.append({
649:                 "phase": self._current_helper or "<unknown>",
650:                 "from_slot": None,  # populated below from prior id
651:                 "to_slot": slot,
652:                 "from_id": self._last_resolved_id,
653:                 "to_id": resolved_id,
654:             })
655:             # Backfill from_slot: which slot did `_last_resolved_id`
656:             # belong to? Look it up in self.ids.
657:             for s, sid in self.ids.items():
658:                 if sid == self._last_resolved_id:
659:                     self.slot_transitions_by_phase[-1]["from_slot"] = s
660:                     break
661:         self._last_resolved_id = resolved_id
662:         if not count_generation:
663:             return cache_entry
664:         self.calls_by_slot[slot] = self.calls_by_slot.get(slot, 0) + 1
665:         # S32 B6: per-helper accounting. When `_current_helper` is
666:         # unset (helper context not entered), bucket calls under
667:         # `"<unattributed>"` so we still capture totals; in practice
668:         # the writer wraps every helper call site so this fallback
669:         # bucket should stay at 0 in production.
670:         helper = self._current_helper or "<unattributed>"
671:         bucket = self.slot_calls_by_helper.setdefault(
672:             helper, {"creative": 0, "technical": 0}
673:         )
674:         bucket[slot] = bucket.get(slot, 0) + 1
675:         return cache_entry
676: 
677:     def helper_context(self, helper_name: str):
678:         """Context manager: attribute slot calls made within `with` to
679:         `helper_name`. Used by the writer to wrap each helper call so
680:         the per-helper bucket in `slot_calls_by_helper` and the
681:         `phase` field on `slot_transitions_by_phase` get populated.
682:         """
683:         scheduler = self
684: 
685:         class _HelperCtx:
686:             def __enter__(self):
687:                 self._prior = scheduler._current_helper
688:                 scheduler._current_helper = helper_name
689:                 return scheduler
690: 
691:             def __exit__(self, exc_type, exc, tb):
692:                 scheduler._current_helper = self._prior
693:                 return False
694: 
695:         return _HelperCtx()
696: 
697:     def _slot_transport_markers(
698:         self, slot: str,
699:     ) -> dict[str, bool | str | None]:
700:         """Return the declared transport capability for one configured slot.
701: 
702:         ``for_slot`` deliberately defers model acquisition until an actual
703:         generation call. Its wrapper retains the catalog's structured transport
704:         behavior: local models expose lazy schema binding and remote providers
705:         retain their JSON-object capability markers.
706:         """
707:         try:
708:             row = _otr_model_catalog._by_repo_id().get(self.ids[slot])
709:             provider = str(getattr(row, "provider", "") or "")
710:         except Exception:  # noqa: BLE001 -- capability is a safe false default
711:             provider = ""
712:         return {
713:             "_otr_local_schema_binding": provider == "local",
714:             "_otr_openrouter": provider == "openrouter",
715:             "_otr_comfy_credits": provider == "comfy_credits",
716:             "_otr_google_api": provider == "google_api",
717:             "_otr_gguf_native": provider == "gguf_native",
718:             # Providers whose backend accepts a json_object response_format:
719:             # invoke_structured_slot forces json_object for a schema-less
720:             # structured pass on these. OpenRouter (frontier prose->JSON) AND
721:             # native GGUF (llama-cpp json_object). The local transformers lane
722:             # is excluded (it has no json_object mode).
723:             "_otr_supports_json_object": provider in ("openrouter", "gguf_native"),
724:             # The plain scheduler closure does not bind a schema itself.
725:             # Local-transformers closures expose `_otr_bind_schema`; the SciFi
726:             # structured invoker uses it to bind each pass's exact Pydantic
727:             # result type. OpenRouter/GGUF retain their existing json_object
728:             # response-format behavior and are intentionally unchanged.
729:             "_otr_response_format": None,
730:         }
731: 
732:     def for_slot(self, slot: str):
733:         """Return a generate_fn closure that targets `slot`. Each call
734:         ensures the right model is resident before generation fires."""
735:         if slot not in self._ALLOWED_SLOTS:
736:             raise ValueError(
737:                 f"_SlotScheduler.for_slot: slot must be one of "
738:                 f"{self._ALLOWED_SLOTS!r}; got {slot!r}"
739:             )
740:         scheduler = self
741: 
742:         transport_markers = scheduler._slot_transport_markers(slot)
743: 
744:         def _make_generate_fn(schema_model=None):
745:             def generate_fn(
746:                 messages, *, temperature, max_new_tokens, stop=None,
747:                 response_format=None,
748:             ):
749:                 helper = scheduler._current_helper
750:                 cache_entry = scheduler._account_and_get_entry(slot)
751:                 base = _build_truncating_generate_fn(
752:                     cache_entry,
753:                     schema_model=schema_model,
754:                     **scheduler.sampling,
755:                 )
756:                 kwargs = {
757:                     "temperature": temperature,
758:                     "max_new_tokens": max_new_tokens,
759:                     "stop": stop,
760:                 }
761:                 if response_format is not None:
762:                     kwargs["response_format"] = response_format
763:                 output = base(messages, **kwargs)
764:                 scheduler._record_successful_model_call(slot, helper, cache_entry, base)
765:                 return output
766: 
767:             def inspect_fit(messages, *, max_new_tokens, **kwargs):
768:                 if any(transport_markers[marker] for marker in (
769:                     "_otr_openrouter", "_otr_comfy_credits", "_otr_google_api", "_otr_gguf_native",
770:                 )):
771:                     return {"supported": False, "reason": "native tokenizer inspection unavailable"}
772:                 from . import _otr_model_loader as loader
773:                 entry = scheduler._account_and_get_entry(slot, count_generation=False)
774:                 return loader.inspect_native_prompt_fit(
775:                     entry, messages, max_new_tokens=max_new_tokens,
776:                 )
777: 
778:             generate_fn._otr_inspect_fit = inspect_fit
779:             for marker, value in transport_markers.items():
780:                 setattr(generate_fn, marker, value)
781:             if transport_markers["_otr_local_schema_binding"]:
782:                 # Lazy schema binding preserves slot accounting and model
783:                 # transitions while making invalid JSON un-sampleable on the
784:                 # local Transformers lane.
785:                 generate_fn._otr_bind_schema = _make_generate_fn  # type: ignore[attr-defined]
786:                 generate_fn._otr_bound_schema_model = schema_model  # type: ignore[attr-defined]
787:             return generate_fn
788: 
789:         return _make_generate_fn()
```

## nodes/OTR_LedgerScriptWriter.py:792 -- _build_truncating_generate_fn
Source SHA256 c25ff162e086182b1b4d290c58bc3469f312917d1540468e5cde71b90c412920
```python
792: def _build_truncating_generate_fn(
793:     cache_entry: dict,
794:     *,
795:     top_p: float = 0.92,
796:     min_p: float = 0.0,
797:     repetition_penalty: float = 1.0,
798:     schema_model: Any = None,
799: ):
800:     """Return a generate_fn that NEVER truncates a prompt.
801: 
802:     The name is historical. This wrapper used to left-slice an oversized prompt;
803:     it no longer can, and no longer does. The output request is fitted to the
804:     MEASURED prompt (see below), which makes the input allowance at least the
805:     prompt's own length by construction, and a prompt with no honest room left
806:     for an artifact raises ``PromptContextOverflowError`` instead of quietly
807:     losing its system/schema prefix.
808: 
809:     Closure captures the episode-level sampling knobs from the
810:     writer widgets: top_p, min_p, repetition_penalty. The per-call
811:     args (`temperature`, `max_new_tokens`, optional `stop`) are
812:     whatever the line composer / outline / picker passes.
813: 
814:     Phase 4 v4 (2026-05-11): min_p and repetition_penalty added as
815:     closure-captured params, plus per-call `stop` support via a
816:     StoppingCriteria subclass that matches on substring at the tail
817:     of the decoded output. Defaults are conservative for the 7B-14B
818:     class:
819:       top_p              = 0.92   (current default, preserved)
820:       min_p              = 0.0    (disabled; 0.05 is the safe non-
821:                                    trivial improvement)
822:       repetition_penalty = 1.0    (disabled; 1.03 is gentle and
823:                                    doesn't damage short outputs)
824:     Each widget overrides per-episode from the workflow.
825: 
826:     The output request is first fitted against the measured prompt. This is
827:     important for structured passes whose artifact-derived reservation can be
828:     larger than the local context cap: the old ``context_cap - requested``
829:     arithmetic reduced the input allowance to 64 tokens and silently deleted
830:     the contract. A prompt is never truncated merely because its caller asked
831:     for a generous output ceiling.
832:     """
833:     # [OpenRouter S3] Remote branch (FC2 seam 2). A provider-tagged remote
834:     # entry has no model/tokenizer/context_cap to close over; return the
835:     # remote generate_fn before capturing local handles below. The remote
836:     # model does its own prompt budgeting server-side and honours the
837:     # caller's per-call temperature + stop. Zero local VRAM.
838:     if cache_entry.get("provider") == "openrouter":
839:         from . import _otr_openrouter_backend as _orb
840:         return _orb.make_openrouter_generate_fn(cache_entry)
841:     # [Comfy Credits] sibling remote seam (2026-06-01). Same provider-tag
842:     # dispatch as OpenRouter: a credit-billed entry has no model/tokenizer/
843:     # context_cap to close over; return the remote generate_fn before
844:     # capturing local handles below. Server-side budgeting; zero local VRAM.
845:     if cache_entry.get("provider") == "comfy_credits":
846:         from . import _otr_comfy_backend as _occ
847:         return _occ.make_comfy_credits_generate_fn(cache_entry)
848:     if cache_entry.get("provider") == "google_api":
849:         from ._otr_google_api import llm as _gai_llm
850:         return _gai_llm.make_google_api_generate_fn(cache_entry)
851:     # [Local OpenAI] External local server lane for Gemma 4 12B. Same
852:     # provider-tag dispatch; zero ComfyUI-process VRAM.
853:     if cache_entry.get("provider") == "gguf_native":
854:         from . import _otr_gguf_backend as _gguf
855:         # GGUF row registry (2026-07-16): the native GGUF lane now HONORS the
856:         # writer's episode sampling widgets (previously discarded, so gemma ran
857:         # at llama-cpp defaults). top_k stays pinned to GGUF_TOP_K inside the
858:         # backend; the base seed rides the cache_entry from the preflight
859:         # load_config. Announced behavior change for all gguf rows.
860:         return _gguf.make_gguf_generate_fn(
861:             cache_entry,
862:             sampling={
863:                 "top_p": top_p,
864:                 "min_p": min_p,
865:                 "repeat_penalty": repetition_penalty,
866:             },
867:         )
868:     model = cache_entry["model"]
869:     tokenizer = cache_entry["tokenizer"]
870:     active_top_p = float(top_p)
871:     active_min_p = float(min_p or 0.0)
872:     active_rep_penalty = float(repetition_penalty or 1.0)
873:     # Tier 1 fix #8 (2026-05-11): one-shot warning + auto-fallback
874:     # for transformers versions < 4.43 that don't accept `min_p` as
875:     # a kwarg on model.generate. Closure-scoped mutable cell so the
876:     # disable persists across calls within one run without spamming
877:     # the warning more than once.
878:     _min_p_unsupported = [False]
879:     if schema_model is not None:
880:         from ._otr_constrained_generate import (
881:             get_cached_transformers_schema_constraint,
882:         )
883: 
884:     def generate_fn(messages, *, temperature, max_new_tokens, stop=None):
885:         import torch  # local import; never load torch at module import
886:         from ._otr_model_loader import prepare_native_prompt
887:         require_full_output = bool(getattr(
888:             messages, "_otr_require_full_output_budget", False,
889:         ))
890:         reserve_remaining = bool(getattr(
891:             messages, "_otr_reserve_remaining_output_capacity", False,
892:         ))
893:         bounded_capacity = reserve_remaining and max_new_tokens is not None
894:         fail_on_output_limit = bool(getattr(
895:             messages, "_otr_fail_on_output_limit", False,
896:         ))
897:         unbounded_json_field = bool(getattr(
898:             messages, "_otr_unbounded_json_field", False,
899:         ))
900:         prepared = prepare_native_prompt(cache_entry, messages)
901:         inputs = prepared["inputs"]
902:         input_len = prepared["prompt_tokens"]
903:         context_cap = prepared["context_cap"]
904:         requested_max_new_tokens = (
905:             context_cap if reserve_remaining else max(1, int(max_new_tokens))
906:         )
907:         try:
908:             effective_max_new_tokens = fit_output_tokens(
909:                 requested_max_new_tokens,
910:                 context_cap=context_cap,
911:                 prompt_tokens=input_len,
912:                 label="prompt",
913:                 require_full=require_full_output or bounded_capacity,
914:             )
915:         except GenerationContextOverflowError as exc:
916:             # A prompt that leaves no honest room for a usable artifact is a
917:             # hard failure for EVERY caller, not only the ones that opt in via
918:             # `_otr_prompt_must_fit`: left-truncating it would delete the
919:             # system/schema prefix and the model would answer from whatever
920:             # fragment survived. Both arms of the old branch already raised the
921:             # same error, so the flag decided nothing here; the honest guard is
922:             # unconditional. `prompt_must_fit` still selects fail-loud behavior
923:             # at the lane preflights that own an artifact's provenance.
924:             # A-4: the phase travels with the re-wrap rather than being
925:             # re-derived. `fit_output_tokens` refuses BEFORE the call, so this
926:             # is always `prompt_no_room` and the ladder must not re-roll it --
927:             # but the phase is read off the error, not assumed here, so a
928:             # future pre-call refusal that IS retryable cannot be mislabelled
929:             # by this line.
930:             raise PromptContextOverflowError(
931:                 str(exc), phase=exc.phase,
932:             ) from exc
933:         inputs = inputs.to(model.device)
934:         if (not reserve_remaining
935:                 and effective_max_new_tokens != requested_max_new_tokens):
936:             log.warning(
937:                 "[OTR_LedgerScriptWriter] OUTPUT_BUDGET: requested %d -> %d "
938:                 "tokens (prompt_tokens=%d, context_cap=%d)",
939:                 requested_max_new_tokens, effective_max_new_tokens,
940:                 input_len, context_cap,
941:             )
942:         # NO PROMPT TRUNCATION HAPPENS HERE, BY CONSTRUCTION. `fit_output_tokens`
943:         # returns at most `context_cap - input_len`, so `context_cap -
944:         # effective_max_new_tokens` is always >= input_len: the old PROMPT_GUARD
945:         # left-slice could never fire once the output budget was fitted to the
946:         # MEASURED prompt instead of the REQUESTED ceiling. It is deleted rather
947:         # than left unreachable -- a dead lever is worse than no lever, because
948:         # the next reader repairs the branch that never runs (operator, 2026-07-11)
949:         # and the live defect keeps its hiding place. A prompt that genuinely
950:         # cannot fit now raises PromptContextOverflowError above.
951:         gen_kwargs = {
952:             "do_sample": True,
953:             "temperature": float(temperature),
954:             "top_p": active_top_p,
955:             "max_new_tokens": effective_max_new_tokens,
956:             "pad_token_id": prepared["pad_token_id"],
957:             "eos_token_id": prepared["eos_token_ids"] or None,
958:         }
959:         # Only forward non-default values so older transformers
960:         # versions that don't accept `min_p` as a kwarg keep working
961:         # silently when the widget is at its disabled default.
962:         if active_min_p > 0.0 and not _min_p_unsupported[0]:
963:             gen_kwargs["min_p"] = active_min_p
964:         if active_rep_penalty != 1.0:
965:             gen_kwargs["repetition_penalty"] = active_rep_penalty
966:         if schema_model is not None:
967:             # Local structured passes are constrained at token selection:
968:             # tokens that cannot continue a schema-valid JSON document are
969:             # never sampleable. Keep one beam; constrained sampling does not
970:             # benefit from multiplying parser state across beams.
971:             gen_kwargs["num_beams"] = 1
972: 
973:         # THE LIVENESS GUARD (2026-08-13). Installed UNCONDITIONALLY, and NOT
974:         # behind `if stop:` -- the structured passes never pass `stop`, which is
975:         # exactly why nothing watched the decode that ran away for 22 minutes.
976:         #
977:         # It is also NOT gated on `schema_model is not None`, though the settled
978:         # design proposed that. A schema gate installs the guard on the lane
979:         # that now ALSO has structural string ceilings, and skips the lane with
980:         # full remaining capacity, no schema and no stop. A liveness contract
981:         # belongs on every local generate() call; the guard costs nothing on a
982:         # decode that never opens a long string.
983:         #
984:         # CONSTRUCTION FAILURE IS LOUD. The stop-string block below swallows its
985:         # exception because a missing quality stop is a nice-to-have; a missing
986:         # liveness guard is a silent removal of protection the log then claims
987:         # to have. So this one raises.
988:         from transformers import StoppingCriteriaList  # noqa: I001
989:         try:
990:             from ._otr_decode_guard import (
991:                 make_degeneracy_criterion, MAX_OPEN_STRING_TOKENS,
992:             )
993:         except ImportError:  # pragma: no cover - flat/standalone import path
994:             from _otr_decode_guard import (  # type: ignore
995:                 make_degeneracy_criterion,
996:                 MAX_OPEN_STRING_TOKENS,
997:             )
998:         # NO try/except AROUND THE CONSTRUCTION. An r1 panel caught the first
999:         # version claiming "construction failure must be loud" in a comment
1000:         # while the code quietly set the guard to None and ran without it --
1001:         # a comment describing a fix is not a fix. If the guard cannot be
1002:         # built, that is a broken install and the render must say so.
1003:         # The tokenizer is passed ONLY when a schema is bound, which turns on
1004:         # the second signal: the open-string counter that catches an
1005:         # ELABORATION SPIRAL (a runaway that never repeats -- specimen P2,
1006:         # 15,355 tokens, which the cycle detector is structurally blind to).
1007:         # It reads quotes as STRUCTURE, so it must never run on a free-prose
1008:         # or markup pass where a quotation mark is dialogue.
1009:         # Provider-capacity prose also opts out of the open-string size limit;
1010:         # cycle detection stays active on every route.
1011:         _degeneracy_guard = make_degeneracy_criterion(
1012:             inputs["input_ids"].shape[1],
1013:             tokenizer=tokenizer if schema_model is not None else None,
1014:             max_open_string_tokens=(
1015:                 None if unbounded_json_field else MAX_OPEN_STRING_TOKENS),
1016:         )
1017:         gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
1018:             [_degeneracy_guard]
1019:         )
1020: 
1021:         # Stop-string support (Phase 4 v4). Tier 2 fix #16
1022:         # (2026-05-11): the StoppingCriteria subclass is now defined
1023:         # once at module scope by `_get_substring_stop_class()` and
1024:         # reuses a rolling buffer instead of decoding the last 64
1025:         # tokens every step. Falls back silently on import error
1026:         # (stop strings are quality nice-to-have, not correctness).
1027:         if stop:
1028:             try:
1029:                 from transformers import (  # noqa: I001
1030:                     StoppingCriteriaList,
1031:                 )
1032:                 prompt_len_now = inputs["input_ids"].shape[1]
1033:                 stop_strings = tuple(s for s in stop if s)
1034:                 _SubstringStop = _get_substring_stop_class()
1035:                 # APPEND, never assign. This block used to own
1036:                 # `stopping_criteria` outright; with the liveness guard
1037:                 # installed above, assigning here would silently REMOVE the
1038:                 # guard on exactly the calls that also asked for stop strings.
1039:                 # StoppingCriteriaList ORs its members, so both signals stay
1040:                 # live and the guard's `hit` flag remains the discriminator.
1041:                 _criteria = list(gen_kwargs.get("stopping_criteria") or [])
1042:                 _criteria.append(
1043:                     _SubstringStop(tokenizer, stop_strings, prompt_len_now)
1044:                 )
1045:                 gen_kwargs["stopping_criteria"] = StoppingCriteriaList(_criteria)
1046:             except Exception as exc:  # noqa: BLE001
1047:                 log.debug(
1048:                     "[OTR_LedgerScriptWriter] stop-strings disabled: %s",
1049:                     exc,
1050:                 )
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
1109:         try:
1110:             last_token = int(generated_ids[-1])
1111:             eos_values = prepared["eos_token_ids"]
1112:             ended_with_eos = last_token in eos_values
1113:         except Exception:  # pragma: no cover - exotic token container
1114:             pass
1115:         # A-1 (2026-07-30, writer repair): DECODE BEFORE THE RAISE.
1116:         # The output-limit raise used to fire HERE, above the decode, so a
1117:         # fail-closed leg threw away the only copy of what the model actually
1118:         # produced -- the ladder received an exception with no artifact, and the
1119:         # OUTPUT_TRUNCATED / OUTPUT_CAP arithmetic below never printed either,
1120:         # because the raise jumped over it. Whoever debugs a truncation needs
1121:         # the completion AND the arithmetic, and both were unreachable at the
1122:         # one moment they exist. Decoding here costs a decode on a leg that is
1123:         # already dying; the success path decodes exactly once, as before.
1124:         decoded = tokenizer.decode(
1125:             generated_ids, skip_special_tokens=True,
1126:         )
1127: 
1128:         # CLASSIFY THE HALT FIRST (2026-08-13). Order matters and this is the
1129:         # authoritative sequence from the settled design:
1130:         #   1. guard.hit          -> degeneracy, REGARDLESS of generated length
1131:         #   2. at the ceiling, no EOS -> capacity
1132:         #   3. ended_with_eos     -> clean termination
1133:         #   4. otherwise          -> some other criterion, e.g. a stop substring
1134:         # Degeneracy must be tested BEFORE capacity because a halted decode
1135:         # stops with room to spare -- reading it as anything else would report
1136:         # "ended at the provider capacity limit" about a decode that was
1137:         # deliberately stopped with ~11,000 tokens unspent.
1138:         if _degeneracy_guard is not None and getattr(
1139:             _degeneracy_guard, "hit", False
1140:         ):
1141:             telemetry = _degeneracy_guard.telemetry()
1142:             reason = ("an open JSON string exceeded its token allowance"
1143:                       if _degeneracy_guard.reason == "open_string"
1144:                       else "the output repeated a run of tokens verbatim")
1145:             log.error(
1146:                 "[OTR_LedgerScriptWriter] DECODE HALTED (%s): %s, after %s "
1147:                 "generated tokens of a %d-token allowance. Rerollable. Telemetry: %s",
1148:                 _degeneracy_guard.reason, reason,
1149:                 generated_tokens, effective_max_new_tokens,
1150:                 telemetry,
1151:             )
1152:             # Same evidence discipline as the capacity raise below: the head
1153:             # says what the model was writing, the tail says what it was doing
1154:             # when the guard stopped it. For a degeneracy halt the tail is the
1155:             # whole point -- it is where the loop is visible.
1156:             _halt_raw = decoded or ""
1157:             log.error(
1158:                 "[OTR_LedgerScriptWriter] RUNAWAY EVIDENCE (%d chars, %s "
1159:                 "tokens, halted)\n  HEAD: %s\n  TAIL: %s",
1160:                 len(_halt_raw), generated_tokens,
1161:                 _halt_raw[:400].replace("\n", " "),
1162:                 _halt_raw[-400:].replace("\n", " "),
1163:             )
1164:             raise GenerationDegeneracyError(
1165:                 "generation was halted by the in-decode liveness guard: " + reason,
1166:                 halt_reason=_degeneracy_guard.reason,
1167:                 open_string_tokens=telemetry.get("open_string_tokens"),
1168:                 repetition=telemetry,
1169:                 raw_completion=decoded,
1170:                 prompt_tokens=prompt_len,
1171:                 generated_tokens=generated_tokens,
1172:                 requested_output_tokens=requested_max_new_tokens,
1173:                 effective_output_tokens=effective_max_new_tokens,
1174:                 context_cap=context_cap,
1175:                 ended_with_eos=ended_with_eos,
1176:             )
1177: 
1178:         if generated_tokens == effective_max_new_tokens:
1179:             # The model stopped because it ran OUT OF ROOM, not because it was
1180:             # finished. When the room it was given is also LESS than the room
1181:             # its caller asked for, that is the silent catastrophe: the artifact
1182:             # is cut off mid-JSON and the ladder reports a bare JSONDecodeError
1183:             # three times, naming the model instead of the budget. Say the real
1184:             # cause once, LOUDLY, with the whole arithmetic -- a reader of the
1185:             # leg log must never have to reconstruct it.
1186:             if effective_max_new_tokens < requested_max_new_tokens:
1187:                 if reserve_remaining:
1188:                     # THE ADVICE WAS WRONG EXACTLY WHEN IT FIRED (2026-08-13).
1189:                     #
1190:                     # A ProviderCapacityMessages pass sets
1191:                     # _otr_reserve_remaining_output_capacity, so requested ==
1192:                     # the whole context window BY DESIGN, and
1193:                     # effective < requested is true the moment the prompt is
1194:                     # non-empty. The old text told the reader to "give this
1195:                     # pass a slot whose window fits prompt+artifact" -- but
1196:                     # this pass already HAS every token there is, so there is
1197:                     # no bigger slot to give it and no config defect to find.
1198:                     # It sent a live session hunting one for twenty minutes.
1199:                     # When the pass reserved everything and still hit the
1200:                     # ceiling without an EOS, the model did not stop.
1201:                     log.error(
1202:                         "[OTR_LedgerScriptWriter] OUTPUT_TRUNCATED: this pass "
1203:                         "reserved ALL remaining output capacity (%d of the "
1204:                         "%d-token window, after a %d-token prompt) and still "
1205:                         "ran to the ceiling. THE MODEL DID NOT STOP -- there "
1206:                         "is no larger slot to move it to, so do not go looking "
1207:                         "for one. Any JSON parse failure below is a runaway "
1208:                         "decode, not a budget defect.",
1209:                         effective_max_new_tokens, context_cap, prompt_len,
1210:                     )
1211:                 else:
1212:                     log.error(
1213:                         "[OTR_LedgerScriptWriter] OUTPUT_TRUNCATED: generation "
1214:                         "stopped at the ceiling after a CLAMP. The caller asked "
1215:                         "for %d output tokens; the %d-token context window left "
1216:                         "only %d after a %d-token prompt. Any JSON parse failure "
1217:                         "below is this budget, not the model. Give this pass a "
1218:                         "slot whose window fits prompt+artifact.",
1219:                         requested_max_new_tokens, context_cap,
1220:                         effective_max_new_tokens, prompt_len,
1221:                     )
1222:             else:
1223:                 log.warning(
1224:                     "[OTR_LedgerScriptWriter] OUTPUT_CAP: generation stopped at "
1225:                     "the caller's own ceiling (prompt_tokens=%d "
1226:                     "generated_tokens=%d max_new_tokens=%d); output may be "
1227:                     "truncated.",
1228:                     prompt_len, generated_tokens, effective_max_new_tokens,
1229:                 )
1230:         if (generated_tokens == effective_max_new_tokens
1231:                 and fail_on_output_limit and not ended_with_eos):
1232:             # LOG THE EVIDENCE BEFORE DISCARDING IT (2026-08-13).
1233:             #
1234:             # `raw_completion=decoded` below has been attached to this exception
1235:             # since A-1 and NOTHING has ever read it -- the leg log prints
1236:             # "raw head: <empty>". So at the one moment thousands of tokens of
1237:             # runaway text exist in memory, they are thrown away, and the next
1238:             # reader has to reproduce a 20-minute decode to learn what the model
1239:             # was actually saying. Two runaways in one night were diagnosed by
1240:             # inference for exactly this reason.
1241:             #
1242:             # Head AND tail, because they answer different questions: the head
1243:             # says what the model was writing, the tail says what it was doing
1244:             # when it ran out of room. A verbatim loop or digit run in the tail
1245:             # means degeneracy; varied run-on prose means it was hedging and
1246:             # could not find a way to end the sentence. That distinction decides
1247:             # whether the cure is a decode guard or the pack's own wording, and
1248:             # it is one log line away.
1249:             _raw = decoded or ""
1250:             _head = _raw[:400].replace("\n", " ")
1251:             _tail = _raw[-400:].replace("\n", " ")
1252:             log.error(
1253:                 "[OTR_LedgerScriptWriter] RUNAWAY EVIDENCE (%d chars, %d "
1254:                 "tokens, ended_with_eos=%s)\n  HEAD: %s\n  TAIL: %s",
1255:                 len(_raw), generated_tokens, ended_with_eos, _head, _tail,
1256:             )
1257:             raise PromptContextOverflowError(
1258:                 "prose generation exhausted the full remaining provider/context "
1259:                 f"capacity ({effective_max_new_tokens} output tokens after a "
1260:                 f"{prompt_len}-token prompt); the partial artifact is discarded, "
1261:                 "never repaired as prose",
1262:                 # A-4: THIS is the phase a re-roll can actually fix -- the call
1263:                 # RAN, and sampling is stochastic (nine engines in the live
1264:                 # 45-word campaign produced both a pass and a fail on
1265:                 # byte-identical code). The message lost its old tail, "not
1266:                 # eligible for a prose or structural reroll", because A-4 makes
1267:                 # the second half of that false: the ladder may now re-roll
1268:                 # this pass. What stays true, and stays said, is that the
1269:                 # partial artifact is never handed to a prose repair. Every
1270:                 # OTHER transport's capacity refusal carries no phase, so it
1271:                 # stays terminal and its own message stays accurate.
1272:                 phase=CAPACITY_PHASE_OUTPUT_LIMIT,
1273:                 raw_completion=decoded,
1274:                 prompt_tokens=prompt_len,
1275:                 generated_tokens=generated_tokens,
1276:                 requested_output_tokens=requested_max_new_tokens,
1277:                 effective_output_tokens=effective_max_new_tokens,
1278:                 context_cap=context_cap,
1279:                 ended_with_eos=ended_with_eos,
1280:             )
1281:         # Tier 1 fix #5 (2026-05-11): StoppingCriteria halts
1282:         # generation but leaves the trigger bytes in the output
1283:         # buffer. Slice at the first stop substring so leaked
1284:         # bracketed/parenthesized tails don't survive into the
1285:         # composer's strip_line_formatting -> ledger pipeline. With
1286:         # polish OFF (default), this is the last guard before the
1287:         # text lands. Earliest-match wins.
1288:         if stop:
1289:             cut = len(decoded)
1290:             for s in stop:
1291:                 if not s:
1292:                     continue
1293:                 idx = decoded.find(s)
1294:                 if idx >= 0 and idx < cut:
1295:                     cut = idx
1296:             decoded = decoded[:cut]
1297:         return decoded
1298: 
1299:     if schema_model is not None:
1300:         generate_fn.schema_model = schema_model  # type: ignore[attr-defined]
1301:     return generate_fn
```

## nodes/_otr_constrained_generate.py:100 -- get_cached_transformers_schema_constraint
Source SHA256 d8298bf94f283ca21286415ac16b70af49a619f060103506a5b4d36ad0fa94f8
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

## nodes/_otr_constrained_generate.py:167 -- make_constrained_generate_fn
Source SHA256 d8298bf94f283ca21286415ac16b70af49a619f060103506a5b4d36ad0fa94f8
```python
167: def make_constrained_generate_fn(
168:     cache_entry: dict[str, Any],
169:     schema_model: Type[BaseModel],
170:     heartbeat_label: Optional[str] = None,
171: ) -> ConstrainedGenerateFn:
172:     """Wrap a cache_entry into a grammar-constrained generate closure.
173: 
174:     Args:
175:         cache_entry: dict produced by _otr_model_loader.load_llm. Must
176:             contain `model` and `tokenizer` keys.
177:         schema_model: a pydantic BaseModel subclass. The lm-format-
178:             enforcer JsonSchemaParser binds to its
179:             model_json_schema(); generate() will only emit token
180:             sequences that the parser accepts.
181:         heartbeat_label: opt-in live-visibility label. When set (e.g.
182:             "EditorPass"), the closure attaches a read-only
183:             _HeartbeatStreamer to model.generate() that logs token
184:             count, tok/s, elapsed, and a decoded tail every ~32 tokens
185:             so long blocking passes are visible in real time. The
186:             streamer only observes tokens -- it never feeds any back,
187:             so generated output is identical with or without it.
188:             Default None -> no streamer -> byte-identical to the
189:             prior behaviour for every existing caller.
190: 
191:     Returns:
192:         A callable (messages, *, temperature, max_new_tokens) -> str.
193: 
194:     Raises:
195:         ModelLoaderError on cache_entry missing required keys.
196: 
197:     The closure is independent of any specific call site -- it just
198:     knows the schema. Stage 1, Stage 3 LLM validators (step 5), and
199:     the whole-episode critic (step 7) each bind their own schema and
200:     get their own closure.
201:     """
202:     # [OpenRouter S4] Remote branch: a provider-tagged remote entry has
203:     # no tokenizer to bind a grammar parser to. Map the call's Pydantic
204:     # schema -> OpenRouter response_format (json_schema, strict) and
205:     # return the remote generate_fn. Integrity is enforced FAIL-CLOSED
206:     # (C4) by the SAME downstream the local path uses -- either the
207:     # structured_call validate + bounded-repair ladder (raises
208:     # StructuredCallFailedError on exhaustion) or the call site's direct
209:     # _parse_and_validate -- so malformed remote output can never reach
210:     # the ledger. A model that lacks json_schema support returns a 4xx,
211:     # which the backend surfaces as OpenRouterCallFailedError (also
212:     # fail-closed). Zero NEW validation logic; reuses the existing path.
213:     if cache_entry.get("provider") == "openrouter":
214:         from . import _otr_openrouter_backend as _orb
215:         response_format = _orb.schema_to_response_format(
216:             schema_model, name=getattr(schema_model, "__name__", "otr_schema"),
217:         )
218:         return _orb.make_openrouter_generate_fn(
219:             cache_entry, response_format=response_format,
220:         )
221:     # BUG-LOCAL-299: Comfy Credits sibling. The Comfy lane is "OpenRouter over
222:     # Comfy's proxy", so the json_schema response_format is byte-identical --
223:     # reuse the OpenRouter schema mapper, then hand it to the Comfy generate_fn.
224:     # Same fail-closed downstream (structured_call validate + repair ladder).
225:     if cache_entry.get("provider") == "comfy_credits":
226:         from . import _otr_openrouter_backend as _orb
227:         from . import _otr_comfy_backend as _occ
228:         response_format = _orb.schema_to_response_format(
229:             schema_model, name=getattr(schema_model, "__name__", "otr_schema"),
230:         )
231:         return _occ.make_comfy_credits_generate_fn(
232:             cache_entry, response_format=response_format,
233:         )
234:     # Native GGUF lane. It accepts llama-cpp-python response_format, so map the
235:     # existing OpenRouter-style json_schema wrapper at the backend boundary.
236:     if cache_entry.get("provider") == "gguf_native":
237:         from . import _otr_openrouter_backend as _orb
238:         from . import _otr_gguf_backend as _gguf
239:         response_format = _orb.schema_to_response_format(
240:             schema_model, name=getattr(schema_model, "__name__", "otr_schema"),
241:         )
242:         return _gguf.make_gguf_generate_fn(
243:             cache_entry, response_format=response_format,
244:         )
245:     required = {"model", "tokenizer"}
246:     missing = required - set(cache_entry)
247:     if missing:
248:         raise ModelLoaderError(
249:             f"cache_entry missing required keys: {sorted(missing)}"
250:         )
251: 
252:     model = cache_entry["model"]
253:     tokenizer = cache_entry["tokenizer"]
254: 
255:     def constrained_generate_fn(
256:         messages: List[dict],
257:         *,
258:         temperature: float,
259:         max_new_tokens: int,
260:     ) -> str:
261:         # Lazy torch import keeps the module importable in test
262:         # environments where torch may be slow / partial.
263:         try:
264:             import torch
265:         except ImportError as exc:
266:             raise ModelLoaderError("torch not available") from exc
267: 
268:         unbounded_json_field = bool(getattr(
269:             messages, "_otr_unbounded_json_field", False,
270:         ))
271:         prepared = prepare_native_prompt(cache_entry, messages)
272:         context_cap = prepared["context_cap"]
273:         requested_tokens = context_cap if prepared["reserve_remaining"] else max_new_tokens
274:         try:
275:             effective_max_new_tokens = fit_output_tokens(
276:                 requested_tokens, context_cap=context_cap,
277:                 prompt_tokens=prepared["prompt_tokens"], label="constrained prompt",
278:                 require_full=(prepared["require_full_output"] or
279:                               (prepared["reserve_remaining"] and max_new_tokens is not None)),
280:             )
281:         except GenerationContextOverflowError as exc:
282:             raise PromptContextOverflowError(str(exc), phase=exc.phase) from exc
283:         inputs = prepared["inputs"].to(model.device)
284: 
285:         # Opt-in live heartbeat (read-only; does not alter sampled tokens).
286:         streamer = (
287:             _HeartbeatStreamer(tokenizer, heartbeat_label)
288:             if heartbeat_label
289:             else None
290:         )
291: 
292:         # THE LIVENESS GUARD (2026-08-13). This route is LIVE -- the writer
293:         # builds it for the slot-drama contract and calls it once per voiced
294:         # beat -- and it was missed when the guard shipped, because the guard
295:         # was installed per-WRAPPER in OTR_LedgerScriptWriter rather than at
296:         # every local generate(). A six-agent audit of every `model.generate`
297:         # in nodes/ found this and `_otr_model_loader.make_generate_fn`
298:         # unprotected. A unit test can pass while a production route runs bare.
299:         #
300:         # Cost here is bounded today (192 output tokens, two attempts per slot),
301:         # so this is prophylaxis rather than an emergency -- but "bounded by
302:         # whatever the caller happened to pass" is not a liveness contract.
303:         from transformers import StoppingCriteriaList  # noqa: I001
304:         try:
305:             from ._otr_decode_guard import (
306:                 make_degeneracy_criterion, MAX_OPEN_STRING_TOKENS,
307:             )
308:         except ImportError:  # pragma: no cover - flat/standalone import path
309:             from _otr_decode_guard import (  # type: ignore
310:                 make_degeneracy_criterion,
311:                 MAX_OPEN_STRING_TOKENS,
312:             )
313:         # This route is schema-bound. Ordinary calls retain open-string tracking;
314:         # provider-capacity prose opts out of that size limit, keeping cycles.
315:         _guard = make_degeneracy_criterion(
316:             inputs["input_ids"].shape[1], tokenizer=tokenizer,
317:             max_open_string_tokens=(
318:                 None if unbounded_json_field else MAX_OPEN_STRING_TOKENS),
319:         )
320: 
321:         # temperature 0.0 means GREEDY. transformers raises on
322:         # do_sample=True with temperature 0.0 (story_orchestrator worked around
323:         # it with 0.05 in 2026-04); the honest shape is to switch sampling off,
324:         # which is what a caller asking for 0.0 means. Every temperature > 0
325:         # call keeps the exact kwargs it always had.
326:         if temperature and temperature > 0.0:
327:             sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.92}
328:         else:
329:             sampling = {"do_sample": False}
330:         with torch.no_grad():
331:             parser, prefix_fn = get_cached_transformers_schema_constraint(
332:                 cache_entry, schema_model, eos_token_ids=prepared["eos_token_ids"],
333:             )
334:             out = model.generate(
335:                 **inputs,
336:                 **sampling,
337:                 max_new_tokens=effective_max_new_tokens,
338:                 pad_token_id=prepared["pad_token_id"],
339:                 eos_token_id=prepared["eos_token_ids"] or None,
340:                 stopping_criteria=StoppingCriteriaList([_guard]),
341:                 # The schema-binding argument. transformers passes
342:                 # this hook into the logits-processing path; lm-
343:                 # format-enforcer reuses it to keep the sampler in
344:                 # the JSON-schema-valid subset.
345:                 prefix_allowed_tokens_fn=prefix_fn,
346:                 # num_beams=1 keeps memory + latency manageable.
347:                 # Constrained sampling does not need beams to land
348:                 # a valid object; beams would multiply the parser
349:                 # state cost without quality gain on structured
350:                 # output.
351:                 num_beams=1,
352:                 # Read-only observer for live tok/s visibility on long
353:                 # passes; None when heartbeat_label is unset.
354:                 streamer=streamer,
355:             )
356: 
357:         _memory_log.memory_snapshot("constrained_generation_returned", model_id=cache_entry.get("model_id"))
358:         prompt_len = inputs["input_ids"].shape[1]
359:         decoded = tokenizer.decode(
360:             out[0][prompt_len:],
361:             skip_special_tokens=True,
362:         )
363:         if getattr(_guard, "hit", False):
364:             # A halted decode is NOT a short answer. Returning the truncated
365:             # text would hand the caller a fragment that parses as a real
366:             # reply -- the silent-truncation trap. Raise the same rerollable
367:             # phase the writer transport raises, so a caller that has a retry
368:             # path uses it and one that does not fails loudly instead of
369:             # quietly accepting half a JSON object.
370:             telemetry = _guard.telemetry()
371:             reason = ("an open JSON string exceeded its token allowance"
372:                       if _guard.reason == "open_string"
373:                       else "the output repeated a run of tokens verbatim")
374:             log.error("[%s] DECODE HALTED (%s): %s. Rerollable. Telemetry: %s",
375:                       heartbeat_label or "constrained-generate", _guard.reason,
376:                       reason, telemetry)
377:             raise GenerationDegeneracyError(
378:                 "constrained generation was halted by the liveness guard: " + reason,
379:                 halt_reason=_guard.reason,
380:                 open_string_tokens=telemetry.get("open_string_tokens"),
381:                 repetition=telemetry,
382:                 raw_completion=decoded,
383:                 prompt_tokens=prompt_len,
384:             )
385:         generated_ids = out[0][prompt_len:]
386:         eos_values = prepared["eos_token_ids"]
387:         ended_with_eos = bool(len(generated_ids)) and int(generated_ids[-1]) in eos_values
388:         if (prepared["fail_on_output_limit"]
389:                 and len(generated_ids) >= effective_max_new_tokens and not ended_with_eos):
390:             raise PromptContextOverflowError(
391:                 "constrained generation consumed its output allowance before stopping",
392:                 phase="output_limit", raw_completion=decoded,
393:                 prompt_tokens=prompt_len, generated_tokens=len(generated_ids),
394:                 requested_output_tokens=requested_tokens,
395:                 effective_output_tokens=effective_max_new_tokens,
396:                 context_cap=context_cap, ended_with_eos=False,
397:             )
398:         return decoded
399: 
400:     # Schema metadata is stable. Mutable parser/prefix state must stay call-local.
401:     constrained_generate_fn.schema_model = schema_model       # type: ignore[attr-defined]
402: 
403:     return constrained_generate_fn
```

## nodes/_otr_my_story.py:230 -- StoryTreatment
Source SHA256 d18d200d83d940f4b308879eb79469c79f0f90a6391ef691b7544174b3683366
```python
230: class StoryTreatment(BaseModel):
231:     title: str = ""
232:     logline: str = ""
233:     dramatic_question: str = ""
234:     setting: str = ""
235:     time_of_day: str = "night"
236:     cast: "list[CastMember]" = Field(min_length=1)
237:     acts: "list[ActPlan]" = Field(min_length=1)
238:     ending: str = ""
239: 
240:     def names(self) -> "list[str]":
241:         return [c.name.strip() for c in self.cast]
```

## nodes/_otr_my_story.py:369 -- _call
Source SHA256 d18d200d83d940f4b308879eb79469c79f0f90a6391ef691b7544174b3683366
```python
369: def _call(pass_id: str, bundle: Any, *, attempt_receipts=None,
370:           source_rewrite_receipts=None, slot_scheduler=None, configured_model_id=None,
371:           **kwargs) -> Any:
372:     """Use the shared capacity contract and retain actual attempt evidence."""
373:     author_context = [dict(message) for message in kwargs["prompt"]]
374:     prompt = [dict(message) for message in author_context]
375:     prompt[-1]["content"] = _SOURCE.raw_source_block(bundle.fields) + "\n\n" + prompt[-1]["content"]
376:     kwargs["prompt"] = ProviderCapacityMessages(prompt)
377:     kwargs["max_new_tokens"] = None
378: 
379:     def completed(number, raw, error):
380:         if attempt_receipts is not None:
381:             attempt_receipts.append({
382:                 "pass_id": pass_id, "attempt": number, "raw_output": raw,
383:                 "raw_completion": getattr(error, "raw_completion", None),
384:                 "status": "accepted" if error is None else "failed",
385:                 "error": None if error is None else str(error),
386:             })
387: 
388:     # LLM slot: per-sub-pass -- caller supplies the creative or technical slot.
389:     authored = structured_call(on_attempt_complete=completed, **kwargs)
390:     if source_rewrite_receipts is None:
391:         return authored
392:     original = authored.model_dump(mode="json")
393:     corrected, receipt = _SOURCE.rewrite_story_source(
394:         bundle.fields, original, kwargs["slot_fn"], schema=kwargs["schema"],
395:         receipts=source_rewrite_receipts, pass_id=pass_id,
396:         post_validator=kwargs.get("post_validator"), slot_scheduler=slot_scheduler,
397:         configured_model_id=configured_model_id, author_context=author_context)
398:     # This runs once AFTER author acceptance, never inside its validator. A
399:     # source rewrite cannot restart the author ladder or check its own output.
400:     if corrected is None:
401:         return authored
402:     accepted = corrected.model_dump(mode="json")
403:     receipt.update(output_sha256=_SOURCE.candidate_sha256(accepted),
404:                    applied=accepted != original,
405:                    status="rewritten" if accepted != original else "unchanged")
406:     return corrected
```

## nodes/_otr_my_story.py:485 -- _pass_treatment
Source SHA256 d18d200d83d940f4b308879eb79469c79f0f90a6391ef691b7544174b3683366
```python
485: def _pass_treatment(creative_fn, pack, bundle, interp: StoryInterpretation,
486:                     *, act_count: int, requested_characters: int,
487:                     include_act_breaks: bool, attempt_receipts=None, **source_kwargs) -> StoryTreatment:
488:     base, retry = _TEMP["treatment"]
489:     bind_schema = getattr(creative_fn, "_otr_bind_schema", None)
490:     treatment_fn = bind_schema(StoryTreatment) if callable(bind_schema) else creative_fn
491:     return _call(
492:         "treatment", bundle, attempt_receipts=attempt_receipts, **source_kwargs,
493:         prompt=[
494:             {"role": "system", "content": _seam(pack, "my_story_treatment_system")},
495:             {"role": "user", "content": (
496:                 "THE INTERPRETATION:\n%s\n\n"
497:                 "SELECTED ACTS: %d (binding). REQUESTED SPEAKING CHARACTERS: %d "
498:                 "(flexible, announcer excluded).\n"
499:                 "Let the supplied story guide the cast; preserve its people. Music cues between acts: %d.\n"
500:                 "Plan the episode now."
501:                 % (json.dumps(interp.model_dump(), ensure_ascii=False, indent=2),
502:                    act_count, requested_characters,
503:                    _interstitial_count(act_count, include_act_breaks))
504:             )},
505:         ],
506:         schema=StoryTreatment,
507:         slot_fn=treatment_fn,
508:         base_temperature=base,
509:         structural_retry_temperature=retry,
510:         repair_prompt_factory=_full_artifact_repair(
511:             "Reorganize the treatment into exactly %d acts. The requested "
512:             "character count is flexible; preserve the listener's people, "
513:             "story material, relationships and ending; change the act grouping to fit."
514:             % act_count),
515:         post_validator=_make_treatment_validator(act_count),
516:         max_attempts=3,
517:         helper_name="my_story_treatment",
518:     )
```

## nodes/_otr_loader_backends.py:88 -- chat_template_kwargs
Source SHA256 8e92a40ec84381553069c6354b7039362847d79eb8432c8d2f64cb60cab36cdb
```python
88: def chat_template_kwargs(model_id: str) -> dict:
89:     """Use Qwen3.5's official direct-response switch on every native surface.
90: 
91:     Other models keep their exact previous template arguments. This controls
92:     template formatting, not sampling or a synthetic prompt rewrite.
93:     """
94:     if str(model_id or "").split(" ", 1)[0] == "Qwen/Qwen3.5-4B":
95:         return {"enable_thinking": False}
96:     return {}
```

## C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_model_loader.py:1225
```python
1225:             )
1226: 
1227:             # NATIVE TEXT DECODER ON THE **INITIAL** LOAD (PBUG-20260906-07).
1228:             # Not on a retry: gemma-4-E2B-it proved a retry is unreachable
1229:             # here. bitsandbytes' validate_environment only refuses a
1230:             # device_map that is a DICT (quantizer_bnb_4bit.py: `isinstance(
1231:             # device_map, dict)`), and this path passes the STRING "auto", so
1232:             # no ValueError is raised, the composite load "succeeds" with its
1233:             # towers dispatched to CPU, and only the post-load tripwire notices
1234:             # -- 193.79 seconds later. The towers must never be built at all.
1235:             _init_config = model_config
1236:             _init_kwargs = dict(common_kwargs)
1237:             if _native_text_row:
1238:                 _init_config = _native_text_load_config(model_config)
1239:                 _init_kwargs["output_loading_info"] = True
1240:                 _text_type = getattr(_init_config, "model_type", "")
1241:                 if not _registry_supplies_text_prefix_mapping(_text_type):
1242:                     _init_kwargs["key_mapping"] = {
1243:                         r"^model\.language_model\.": "model.",
1244:                     }
1245:                 _runtime_log(
1246:                     f"[StoryOrchestrator] {_stripped_model_id} loading NATIVE "
1247:                     f"TEXT DECODER ({_text_type}); towers are not "
1248:                     f"materialized. key_mapping supplied by "
1249:                     f"{'OTR' if 'key_mapping' in _init_kwargs else 'transformers registry'}"
1250:                 )
1251: 
1252:             try:
1253:                 if _native_text_row:
1254:                     model, _native_info = AutoModelForCausalLM.from_pretrained(
1255:                         load_target,
1256:                         local_files_only=True,
1257:                         config=_init_config,
1258:                         **_init_kwargs,
1259:                     )
1260:                     _dropped = _validate_native_text_loading_info(
1261:                         _native_info, model_id=_stripped_model_id)
1262:                     _runtime_log(
1263:                         f"[StoryOrchestrator] native text coverage OK for "
1264:                         f"{_stripped_model_id}; dropped tower prefixes="
1265:                         f"{_dropped!r}"
1266:                     )
1267:                 else:
1268:                     model = AutoModelForCausalLM.from_pretrained(
1269:                         load_target,
1270:                         local_files_only=True,
1271:                         config=_init_config,
1272:                         **_init_kwargs,
1273:                     )
1274:             except ValueError as _dispatch_err:
1275:                 # Operator directive 2026-08-29: guards do not kill a render;
```

## C:\Users\jeffr\Documents\ComfyUI\.venv\Lib\site-packages\lmformatenforcer\integrations\transformers.py:68
```python
68: 
69: 
70: def _decode_function(tokenizer: PreTrainedTokenizerBase, tokens: List[int]) -> str:
71:     decoded = tokenizer.decode(tokens)
72:     cleaned = decoded.rstrip('�')
73:     return cleaned
74: 
75: 
76: def build_token_enforcer_tokenizer_data(tokenizer: PreTrainedTokenizerBase, 
77:                                         use_bitmask: bool = False,
78:                                         vocab_size: Optional[int] = None,
79:                                         ) -> TokenEnforcerTokenizerData:
80:     vocab_size = vocab_size or len(tokenizer)
81:     regular_tokens = _build_regular_tokens_list(tokenizer, vocab_size)
82:     decode_fn = functools.partial(_decode_function, tokenizer)
83:     return TokenEnforcerTokenizerData(regular_tokens, decode_fn, tokenizer.eos_token_id, use_bitmask, vocab_size)
84: 
85: 
86: class TransformersPrefixAllowedTokensFn:
87:     def __init__(self, token_enforcer: TokenEnforcer):
88:         self.token_enforcer = token_enforcer
89:         
90:     def __call__(self, batch_id: int, sent: torch.Tensor) -> List[int]:
91:         token_sequence = sent.tolist()
92:         return self.token_enforcer.get_allowed_tokens(token_sequence).allowed_tokens
93: 
94: 
95: def build_transformers_prefix_allowed_tokens_fn(tokenizer_data: Union[PreTrainedTokenizerBase, TokenEnforcerTokenizerData], 
96:                                                 character_level_parser: CharacterLevelParser) -> TransformersPrefixAllowedTokensFn:
97:     """Build the prefix allowed tokens function that transformers will use to filter the tokens generated by the model. The result
98:     can be passed to the prefix_allowed_tokens_fn parameter of the generate() method of transformers models or pipeline configurations."""
99:     if isinstance(tokenizer_data, PreTrainedTokenizerBase):
```

## C:\Users\jeffr\Documents\ComfyUI\.venv\Lib\site-packages\lmformatenforcer\tokenenforcer.py:10
```python
10: 
11: 
12: class TokenEnforcerTokenizerData:
13:     """TokenEnforcerTokenizerData contains all of the preprocessing for preparing the TokenEnforcer to work with a 
14:     specific tokenizer. It does some calculations, so it is recommended to reuse it for multiple TokenEnforcers"""
15:     def __init__(self, 
16:                  regular_tokens: List[Tuple[int, str, bool]], 
17:                  decoder: Callable[[List[int]], str],
18:                  eos_token_id: Union[int, List[int]],
19:                  use_bitmask: bool,
20:                  vocab_size: int):
21:         """
22:         Create the tokenizer data that the TokenEnforcer needs. This can be reused for multiple TokenEnforcers if they work with the same tokenizer.
23:         :param regular_tokens: A list of tuples (token_id, token_string, is_new_word_token) for all the regular (not special) tokens in the tokenizer vocabulary.
24:         Note that token_string is expected to include leading / trailing whitespaces if relevant.
25:         :param decoder: A function that decodes a list of token ids into a string.
26:         :param eos_token_id: The token id(s) of the end-of-string token(s).
27:         """
28:         filtered_regular_tokens = [token_tuple for token_tuple in regular_tokens if token_tuple[0] <= vocab_size]
29:         self.regular_tokens = filtered_regular_tokens
30:         self.tokenizer_tree = TokenizerPrefixTree(self.regular_tokens, use_bitmask, vocab_size)
31:         self.decoder = decoder
32:         self.eos_token_id = eos_token_id
33:         self.tokenizer_alphabet = "".join(token_str for token_str in self.tokenizer_tree.root.children.keys() if len(token_str) == 1)
34:         self.vocab_size = vocab_size
35:         self.use_bitmask = use_bitmask
36: 
37: 
38: class TokenEnforcer:
39:     """TokenEnforcer provides a token filtering mechanism, given a CharacterLevelParser and some information about the tokenizer.
40:     It is the main entry point for extending lm-format-enforcer to new inference libraries. See __init__() and get_allowed_tokens()"""
41:     @dataclass
42:     class OutputTensorState:
43:         parser: CharacterLevelParser
44:         allowed_tokens: TokenList | None = field(default=None)
45:         current_word_tokens: List[int] = field(default_factory=list)
46:         
47: 
48:     def __init__(self, tokenizer_data: TokenEnforcerTokenizerData, parser: CharacterLevelParser):
49:         """
50:         Create a new TokenEnforcer.
51:         :param tokenizer_data: Per tokenizer data that the token enforcer needs in order to operate.
52:         :param parser: A CharacterLevelParser that defines the allowed strings.
53:         """
54:         self.prefix_states: Dict[Tuple, TokenEnforcer.OutputTensorState] = {}
55:         self.root_parser = parser
56:         self.tokenizer_tree = tokenizer_data.tokenizer_tree
57:         self.decoder = tokenizer_data.decoder
58:         self.eos_token_id = tokenizer_data.eos_token_id
59:         self.regular_tokens = tokenizer_data.regular_tokens
60:         self.allowed_token_cache: Dict[Hashable, Any] = {}
61:         self.use_bitmask = tokenizer_data.use_bitmask
62:         self.vocab_size = tokenizer_data.vocab_size
```

## C:\Users\jeffr\Documents\ComfyUI\.venv\Lib\site-packages\lmformatenforcer\tokenenforcer.py:98
```python
98:     def _compute_allowed_tokens(self, state_tokens: Tuple, state: 'TokenEnforcer.OutputTensorState'):
99:         try:
100:             allowed_tokens: TokenList = TokenList(self.use_bitmask, self.vocab_size)
101:             
102:             cache_key = state.parser.cache_key()
103:             if cache_key is not None and cache_key in self.allowed_token_cache:
104:                 state.allowed_tokens = self.allowed_token_cache[cache_key]
105:                 return
106:             shortcut_key = state.parser.shortcut_key()
107:             self._collect_allowed_tokens(state.parser, self.tokenizer_tree.root, allowed_tokens, shortcut_key)
108:             if state.parser.can_end():
109:                 if isinstance(self.eos_token_id, list):
110:                     allowed_tokens.extend(self.eos_token_id)                    
111:                 else:
112:                     allowed_tokens.append(self.eos_token_id)
113:             if not allowed_tokens:
114:                 raise ValueError(f"Parser reached state with no allowed tokens")
115:             # root_state = next(state for state in self.prefix_states.values() if state.parser == self.root_parser)
116:             # print(f"Allowing {len(allowed_tokens)} tokens after {state.str_so_far[len(root_state.str_so_far):]}")
117:             state.allowed_tokens = allowed_tokens
118:             if cache_key is not None:
119:                 self.allowed_token_cache[cache_key] = allowed_tokens
120:         except LMFormatEnforcerException:
121:             # Getting an LMFormatEnforcerException means that we know what the user did wrong, 
122:             # and we can give a nice error message for them to fix.
123:             raise
124:         except Exception:
125:             # Other exceptions are potential bugs and should be reported
126:             logging.basicConfig(level=logging.ERROR)  # Initialize if no loggers
127:             prefix = self.decoder(list(state_tokens))
128:             logging.exception(f"Unknown LMFormatEnforcer Problem. Prefix: '{prefix}'\n"
129:                               "Terminating the parser. Please open an issue at \n"
130:                               "https://github.com/noamgat/lm-format-enforcer/issues with the prefix and "
131:                               "CharacterLevelParser parameters")
132:             state.allowed_tokens = TokenList(self.use_bitmask, self.vocab_size)
133:             if isinstance(self.eos_token_id, list):
134:                 state.allowed_tokens.extend(self.eos_token_id)
135:             else:
136:                 state.allowed_tokens.append(self.eos_token_id)
137: 
138:     def _collect_allowed_tokens(self, parser: CharacterLevelParser, tree_node: TokenizerPrefixTreeNode, allowed_tokens: TokenList, shortcut_key: Optional[Hashable]):
139:         allowed_tokens.extend(tree_node.tokens)
140:         allowed_characters = parser.get_allowed_characters()
```
