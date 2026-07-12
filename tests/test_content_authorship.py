import copy

import pytest

from nodes._otr_content_authorship import (
    ContentAuthorshipError, receipt_sha256, stamp_receipt, validate_receipt,
)


def _ledger():
    return {
        "meta": {"source_bank": "bank_a"},
        "lines": [
            {"line_id": "l1", "text": "Exact raw text.", "skip_tts": False},
            {"line_id": "l2", "text": "Second line.", "skip_tts": False},
            {"line_id": "note", "text": "Not voiced.", "skip_tts": True},
        ],
    }


def test_stamp_validate_and_save_reopen_hash_are_stable():
    ledger = _ledger()
    stamp_receipt(ledger, owner_bank="bank_a", accepted_artifacts={"final": {"x": 1}})
    before = receipt_sha256(ledger)
    reopened = copy.deepcopy(ledger)
    assert validate_receipt(reopened)["coverage"]["complete"] is True
    assert receipt_sha256(reopened) == before


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "tamper"])
def test_line_proof_coverage_and_tamper_fail_closed(mutation):
    ledger = _ledger()
    stamp_receipt(ledger, owner_bank="bank_a", accepted_artifacts={"final": {"x": 1}})
    proofs = ledger["meta"]["content_authorship"]["line_proofs"]
    if mutation == "missing":
        proofs.pop()
    elif mutation == "extra":
        proofs.append({"line_id": "extra", "text_sha256": "0" * 64})
    elif mutation == "duplicate":
        proofs.append(copy.deepcopy(proofs[0]))
    else:
        ledger["lines"][0]["text"] += " changed"
    with pytest.raises(ContentAuthorshipError):
        validate_receipt(ledger)


def test_owner_and_artifact_contract_fail_closed():
    ledger = _ledger()
    with pytest.raises(ContentAuthorshipError):
        stamp_receipt(ledger, owner_bank="wrong", accepted_artifacts={"final": {}})
    with pytest.raises(ContentAuthorshipError):
        stamp_receipt(ledger, owner_bank="bank_a", accepted_artifacts={})
