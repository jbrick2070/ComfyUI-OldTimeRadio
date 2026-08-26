"""The GGUF lane honours the generation deadline (PBUG-20260825-04 deferral).

Closes the gap where `make_generate_fn` returned the GGUF closure before any
deadline guard was built, so an abandoned worker kept decoding. llama-cpp's
`create_chat_completion` accepts no `stopping_criteria`, so the GGUF lane uses
DEADLINE-CONDITIONAL STREAMING instead.

Behavioural, with a fake llm object -- no llama-cpp, no CUDA, no model file.
The point of every test here is the OBSERVABLE contract, not the source text:
* with NO deadline registered the call must be byte-identical to the legacy
  one, `stream` absent from kwargs entirely;
* with a deadline it must stop, RAISE, and never hand back partial text.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_gguf_backend as gb  # noqa: E402
from nodes import _otr_model_loader as ml  # noqa: E402


class FakeLlama:
    """Records the kwargs it was called with; can stream or not."""

    def __init__(self, *, chunks=None, text="ok", finish_reason="stop",
                 final_delay=0.0):
        self.calls: list[dict] = []
        self._chunks = chunks
        self._final_delay = final_delay
        self._text = text
        self._finish_reason = finish_reason
        self.closed = False

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs.get("stream"):
            return {"choices": [{
                "finish_reason": self._finish_reason,
                "message": {"content": self._text},
            }]}
        return self._make_stream()

    def _make_stream(self):
        outer = self

        class _Stream:
            def __init__(self):
                self._it = iter(outer._chunks or [])

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    item = next(self._it)
                except StopIteration:
                    # Stall on the FINAL advance. The loop checks expiry only
                    # AFTER a successful advance, so a deadline that passes
                    # during the StopIteration advance is caught by the
                    # POST-call check and nothing else -- the only way to
                    # exercise that branch deterministically.
                    if outer._final_delay:
                        time.sleep(outer._final_delay)
                    raise
                # A chunk may be a CALLABLE so a test can make a side effect
                # (expiring the deadline) happen DURING iteration. Building
                # it eagerly in the list would fire before the call starts
                # and trip the pre-call gate instead of the mid-stream one.
                return item() if callable(item) else item

            def close(self):
                outer.closed = True

        return _Stream()


def _content_chunk(text):
    return {"choices": [{"index": 0, "delta": {"content": text},
                         "finish_reason": None}]}


def _role_chunk():
    return {"choices": [{"index": 0, "delta": {"role": "assistant"},
                         "finish_reason": None}]}


def _terminal_chunk(reason="stop"):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def _model(llm):
    return {"model": llm, "model_id": "unsloth/gemma-4-12b-it-GGUF",
            "n_ctx": 4096, "stop_tokens": ()}


@pytest.fixture(autouse=True)
def _clear_deadline():
    ml.set_generation_deadline(None)
    yield
    ml.set_generation_deadline(None)


class _Messages(list):
    """A list that can carry the private flags generate() reads off the
    messages object itself (`_otr_fail_on_output_limit` and friends)."""


def _generate(llm, *, fail_on_output_limit=False, **kw):
    msgs = _Messages([{"role": "user", "content": "hi"}])
    if fail_on_output_limit:
        msgs._otr_fail_on_output_limit = True
    return gb.GGUFNativeBackend().generate(
        _model(llm), msgs, max_new_tokens=64, **kw,
    )


# --------------------------------------------------------------------------- #
# The ordinary path must not change at all.
# --------------------------------------------------------------------------- #

def test_no_deadline_makes_the_legacy_nonstreaming_call():
    """`stream` must be ABSENT, not merely False.

    Passing stream=False would still be a different call than the one that
    shipped, and this lane's whole safety argument is that an un-deadlined
    GGUF call is untouched.
    """
    llm = FakeLlama(text="hello world")
    out = _generate(llm)
    assert out == "hello world"
    assert len(llm.calls) == 1
    assert "stream" not in llm.calls[0]


def test_no_deadline_still_returns_text_when_a_deadline_is_set_elsewhere():
    """A deadline far in the future must not trip anything."""
    ml.set_generation_deadline(time.monotonic() + 300.0)
    llm = FakeLlama(chunks=[_role_chunk(), _content_chunk("streamed ok"),
                            _terminal_chunk()])
    assert _generate(llm) == "streamed ok"
    assert llm.calls[0]["stream"] is True


# --------------------------------------------------------------------------- #
# Expiry: before, during, and after.
# --------------------------------------------------------------------------- #

def test_already_expired_deadline_never_starts_the_call():
    """The pre-call gate. This is the case that actually pays: the budget is
    usually eaten by the model LOAD upstream, and no mechanism can interrupt
    a load once under way."""
    ml.set_generation_deadline(time.monotonic() - 1.0)
    llm = FakeLlama(text="should never be produced")
    with pytest.raises(ml.GenerationDeadlineExceededError):
        _generate(llm)
    assert llm.calls == [], "an expired call must not reach llama-cpp at all"


def test_deadline_expiring_mid_stream_raises_and_returns_no_partial_text():
    """The whole point: never hand back what was decoded so far."""
    # A SLOW chunk that outlives the budget -- the real shape of the bug.
    # (Mutating the registered deadline mid-stream would prove nothing:
    # _stream_until_deadline uses the value captured when the call started,
    # which is correct because _run_with_timeout sets exactly one deadline
    # per call and never changes it.)
    def _slow_second_half():
        time.sleep(0.15)
        return _content_chunk("second half")

    llm = FakeLlama(chunks=[
        _role_chunk(),
        _content_chunk("first half "),
        _slow_second_half,          # CALLABLE: runs during iteration
        _terminal_chunk(),
    ])
    ml.set_generation_deadline(time.monotonic() + 0.05)
    with pytest.raises(ml.GenerationDeadlineExceededError) as exc:
        _generate(llm)
    assert "first half" not in str(exc.value), (
        "the partial decode must not leak back through the exception either"
    )


def test_stream_is_closed_when_the_deadline_fires():
    """An abandoned generator would keep the llama context busy for the next
    caller on this cached model."""
    def _slow_b():
        time.sleep(0.15)
        return _content_chunk("b")

    llm = FakeLlama(chunks=[_content_chunk("a"), _slow_b, _terminal_chunk()])
    ml.set_generation_deadline(time.monotonic() + 0.05)
    with pytest.raises(ml.GenerationDeadlineExceededError):
        _generate(llm)
    assert llm.closed is True


# --------------------------------------------------------------------------- #
# Assembly fidelity.
# --------------------------------------------------------------------------- #

def test_role_only_chunks_contribute_nothing_and_deltas_concatenate_in_order():
    ml.set_generation_deadline(time.monotonic() + 300.0)
    llm = FakeLlama(chunks=[_role_chunk(), _content_chunk("one "),
                            _content_chunk("two "), _content_chunk("three"),
                            _terminal_chunk()])
    assert _generate(llm) == "one two three"


def test_terminal_length_finish_reason_survives_reassembly():
    """`finish_reason == "length"` drives the output-limit refusal, so it has
    to survive streaming or a truncated artifact becomes eligible for reroll."""
    ml.set_generation_deadline(time.monotonic() + 300.0)
    llm = FakeLlama(chunks=[_content_chunk("cut off"),
                            _terminal_chunk("length")])
    with pytest.raises(gb.GGUFNativeCallFailedError, match="output capacity"):
        _generate(llm, fail_on_output_limit=True)


def test_empty_stream_fails_loud_rather_than_returning_empty_text():
    ml.set_generation_deadline(time.monotonic() + 300.0)
    llm = FakeLlama(chunks=[_terminal_chunk()])
    with pytest.raises(gb.GGUFNativeCallFailedError):
        _generate(llm)


# --------------------------------------------------------------------------- #
# Error routing.
# --------------------------------------------------------------------------- #

def test_deadline_error_is_not_relabelled_as_a_call_failure():
    """_run_with_timeout catches GenerationDeadlineExceededError BY TYPE.
    If the backend's broad `except Exception` swallowed and relabelled it,
    an abandoned generation would route into the ordinary reroll path
    instead of timeout recovery -- silently undoing the fix."""
    ml.set_generation_deadline(time.monotonic() - 1.0)
    llm = FakeLlama()
    with pytest.raises(ml.GenerationDeadlineExceededError):
        _generate(llm)


def test_a_real_backend_error_is_still_wrapped_normally():
    """The deadline plumbing must not become a catch-all that hides faults."""
    class Boom(FakeLlama):
        def create_chat_completion(self, **kwargs):
            raise RuntimeError("cuda blew up")

    ml.set_generation_deadline(time.monotonic() + 300.0)
    with pytest.raises(gb.GGUFNativeCallFailedError, match="cuda blew up"):
        _generate(Boom())


def test_deadline_is_not_visible_to_other_threads():
    """threading.local: a deadline set here must not make a sibling thread's
    GGUF call start streaming."""
    import threading

    ml.set_generation_deadline(time.monotonic() - 1.0)
    seen: dict = {}

    def _worker():
        llm = FakeLlama(text="sibling ok")
        try:
            seen["out"] = _generate(llm)
            seen["kwargs"] = llm.calls[0]
        except BaseException as exc:  # noqa: BLE001
            seen["err"] = exc

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert seen.get("err") is None, f"sibling thread saw a deadline: {seen.get('err')!r}"
    assert seen["out"] == "sibling ok"
    assert "stream" not in seen["kwargs"]


# --------------------------------------------------------------------------- #
# Gaps named by the r3 panel (Cursor + Sonnet) and by r2's own test matrix.
# --------------------------------------------------------------------------- #

def test_a_mid_stream_deadline_is_not_relabelled_as_a_call_failure():
    """Cursor r3: the earlier relabel test only exercised the PRE-call raise,
    which happens OUTSIDE the try. The `except _DeadlineExc: raise` clause
    that actually guards the broad wrapper is on the IN-TRY path, so it was
    named but never tested. Expire mid-iteration to reach it."""
    def _slow():
        time.sleep(0.15)
        return _content_chunk("late")

    llm = FakeLlama(chunks=[_content_chunk("early "), _slow, _terminal_chunk()])
    ml.set_generation_deadline(time.monotonic() + 0.05)
    with pytest.raises(ml.GenerationDeadlineExceededError) as exc:
        _generate(llm)
    assert not isinstance(exc.value, gb.GGUFNativeCallFailedError), (
        "the in-try deadline raise must escape the broad except Exception "
        "intact; relabelled, it would route into the reroll ladder instead "
        "of timeout recovery"
    )


def test_a_stream_that_finishes_late_is_discarded_by_the_post_call_check():
    """The post-call branch. The mid-stream check runs only after a SUCCESSFUL
    advance, so a budget that expires during the final (StopIteration) advance
    reaches only this branch -- which could otherwise regress with the rest of
    this file fully green."""
    llm = FakeLlama(chunks=[_content_chunk("complete answer"),
                            _terminal_chunk()], final_delay=0.15)
    ml.set_generation_deadline(time.monotonic() + 0.05)
    with pytest.raises(ml.GenerationDeadlineExceededError, match="after its deadline"):
        _generate(llm)


def test_response_format_is_forwarded_unchanged_when_streaming():
    """Structured JSON is the highest-risk thing streaming could disturb: the
    grammar is built by llama-cpp's chat handler from response_format, so the
    kwarg must arrive identically in both modes or constrained output silently
    stops being constrained."""
    rf = {"type": "json_object"}
    ml.set_generation_deadline(time.monotonic() + 300.0)
    streamed = FakeLlama(chunks=[_content_chunk('{"a": 1}'), _terminal_chunk()])
    assert _generate(streamed, response_format=rf) == '{"a": 1}'
    ml.set_generation_deadline(None)
    plain = FakeLlama(text='{"a": 1}')
    _generate(plain, response_format=rf)
    assert streamed.calls[0]["response_format"] == plain.calls[0]["response_format"]


def test_merged_stop_list_is_identical_streaming_and_not():
    """Pins that THIS code forwards stops identically. (The documented
    upstream 0.3.33 difference -- non-stream picks first-in-list-order, the
    streaming flush picks earliest-occurrence -- lives inside llama-cpp and
    is not ours to assert with a fake; what IS ours is that we hand both
    modes the same list.)"""
    stops = ["\n[", "\n(", "END"]
    ml.set_generation_deadline(time.monotonic() + 300.0)
    streamed = FakeLlama(chunks=[_content_chunk("x"), _terminal_chunk()])
    _generate(streamed, stop=stops)
    ml.set_generation_deadline(None)
    plain = FakeLlama(text="x")
    _generate(plain, stop=stops)
    assert streamed.calls[0].get("stop") == plain.calls[0].get("stop")


def test_qwen3_think_envelope_is_stripped_on_a_streamed_reply():
    """The think-strip runs after extraction, so it must see reassembled
    streamed text exactly as it sees a non-streamed reply -- otherwise a
    qwen3 row under a deadline returns its reasoning wrapper as the answer."""
    ml.set_generation_deadline(time.monotonic() + 300.0)
    llm = FakeLlama(chunks=[_content_chunk("<think>hmm</think>"),
                            _content_chunk("the answer"), _terminal_chunk()])
    model = _model(llm)
    model["think_policy"] = "qwen3_no_think"
    msgs = _Messages([{"role": "user", "content": "hi"}])
    out = gb.GGUFNativeBackend().generate(model, msgs, max_new_tokens=64)
    assert "<think>" not in out
    assert out.strip() == "the answer"


def test_the_model_is_reusable_after_a_cancelled_stream():
    """Cancel-then-reuse. If an abandoned stream wedged the llama context,
    the NEXT call on the same cached model would fail -- which is the whole
    reason the iterator is explicitly closed."""
    def _slow():
        time.sleep(0.15)
        return _content_chunk("never used")

    llm = FakeLlama(chunks=[_content_chunk("a"), _slow, _terminal_chunk()])
    ml.set_generation_deadline(time.monotonic() + 0.05)
    with pytest.raises(ml.GenerationDeadlineExceededError):
        _generate(llm)
    assert llm.closed is True
    # Second call on the SAME instance, no deadline: must behave normally.
    ml.set_generation_deadline(None)
    llm._text = "second call fine"
    assert _generate(llm) == "second call fine"
    assert "stream" not in llm.calls[-1]


def test_an_error_raised_mid_iteration_is_wrapped_not_leaked():
    """Exception translation must cover iterator ADVANCEMENT, not just the
    initial create_chat_completion call."""
    def _boom():
        raise RuntimeError("decode exploded")

    llm = FakeLlama(chunks=[_content_chunk("a"), _boom, _terminal_chunk()])
    ml.set_generation_deadline(time.monotonic() + 300.0)
    with pytest.raises(gb.GGUFNativeCallFailedError, match="decode exploded"):
        _generate(llm)
