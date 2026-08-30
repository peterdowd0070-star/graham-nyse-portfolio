from graham_nyse.reporting.payload import validate_report_document


def test_report_rejects_unsupported_number():
    context = {
        "claims": [{"claim_id": "metric_cagr", "value": 0.1, "display_value": "10.00%"}]
    }
    valid = {
        "sections": [
            {
                "heading": "Result",
                "text": "The CAGR was 10.00%.",
                "claim_ids": ["metric_cagr"],
            }
        ]
    }
    invalid = {
        "sections": [
            {
                "heading": "Result",
                "text": "The CAGR was 12.00%.",
                "claim_ids": ["metric_cagr"],
            }
        ]
    }
    assert validate_report_document(valid, context)["passed"]
    assert not validate_report_document(invalid, context)["passed"]
