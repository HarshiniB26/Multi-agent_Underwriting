from underwriting.domain.case import CaseState, CaseStatus


def test_new_case_has_received_status():
    case = CaseState()

    assert case.status == CaseStatus.RECEIVED


def test_new_case_has_unique_identifiers():
    first_case = CaseState()
    second_case = CaseState()

    assert first_case.metadata.case_id != second_case.metadata.case_id
    assert first_case.metadata.trace_id != second_case.metadata.trace_id


def test_new_case_starts_with_empty_processing_results():
    case = CaseState()

    assert case.application is None
    assert case.enrichment is None
    assert case.risk_assessment is None
    assert case.recommendation is None
    assert case.errors == []


def test_cases_do_not_share_mutable_error_state():
    first_case = CaseState()
    second_case = CaseState()

    first_case.errors.append("simulated error")

    assert first_case.errors == ["simulated error"]
    assert second_case.errors == []