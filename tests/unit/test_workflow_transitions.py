import pytest

from underwriting.domain.case import CaseState, CaseStatus
from underwriting.workflow.transitions import (
    InvalidCaseTransitionError,
    can_transition,
    transition_case,
)


def test_received_case_can_complete_intake():
    assert can_transition(
        CaseStatus.RECEIVED,
        CaseStatus.INTAKE_COMPLETED,
    )


def test_received_case_cannot_skip_to_risk_assessment():
    assert not can_transition(
        CaseStatus.RECEIVED,
        CaseStatus.RISK_ASSESSED,
    )


def test_valid_transition_returns_updated_copy():
    case = CaseState()

    transitioned = transition_case(
        case,
        CaseStatus.INTAKE_COMPLETED,
    )

    assert transitioned.status == CaseStatus.INTAKE_COMPLETED

    # Original object remains unchanged.
    assert case.status == CaseStatus.RECEIVED


def test_invalid_transition_raises_error():
    case = CaseState()

    with pytest.raises(InvalidCaseTransitionError):
        transition_case(
            case,
            CaseStatus.RISK_ASSESSED,
        )


@pytest.mark.parametrize(
    "status",
    [
        CaseStatus.RECEIVED,
        CaseStatus.INTAKE_COMPLETED,
        CaseStatus.ENRICHMENT_COMPLETED,
        CaseStatus.RISK_ASSESSED,
    ],
)
def test_active_processing_states_can_escalate_to_human_review(status):
    case = CaseState(status=status)

    transitioned = transition_case(
        case,
        CaseStatus.HUMAN_REVIEW,
    )

    assert transitioned.status == CaseStatus.HUMAN_REVIEW


@pytest.mark.parametrize(
    "status",
    [
        CaseStatus.RECEIVED,
        CaseStatus.INTAKE_COMPLETED,
        CaseStatus.ENRICHMENT_COMPLETED,
        CaseStatus.RISK_ASSESSED,
    ],
)
def test_active_processing_states_can_fail(status):
    case = CaseState(status=status)

    transitioned = transition_case(
        case,
        CaseStatus.PROCESSING_FAILED,
    )

    assert transitioned.status == CaseStatus.PROCESSING_FAILED


@pytest.mark.parametrize(
    "terminal_status",
    [
        CaseStatus.RECOMMENDATION_COMPLETED,
        CaseStatus.HUMAN_REVIEW,
        CaseStatus.PROCESSING_FAILED,
    ],
)
def test_terminal_states_cannot_transition(terminal_status):
    for target_status in CaseStatus:
        assert not can_transition(
            terminal_status,
            target_status,
        )