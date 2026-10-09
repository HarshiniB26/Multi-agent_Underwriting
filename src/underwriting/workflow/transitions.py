from underwriting.domain.case import CaseState, CaseStatus


class InvalidCaseTransitionError(Exception):
    """Raised when an invalid underwriting workflow transition is attempted."""


_ALLOWED_TRANSITIONS: dict[CaseStatus, frozenset[CaseStatus]] = {
    CaseStatus.RECEIVED: frozenset(
        {
            CaseStatus.INTAKE_COMPLETED,
            CaseStatus.HUMAN_REVIEW,
            CaseStatus.PROCESSING_FAILED,
        }
    ),
    CaseStatus.INTAKE_COMPLETED: frozenset(
        {
            CaseStatus.ENRICHMENT_COMPLETED,
            CaseStatus.HUMAN_REVIEW,
            CaseStatus.PROCESSING_FAILED,
        }
    ),
    CaseStatus.ENRICHMENT_COMPLETED: frozenset(
        {
            CaseStatus.RISK_ASSESSED,
            CaseStatus.HUMAN_REVIEW,
            CaseStatus.PROCESSING_FAILED,
        }
    ),
    CaseStatus.RISK_ASSESSED: frozenset(
        {
            CaseStatus.RECOMMENDATION_COMPLETED,
            CaseStatus.HUMAN_REVIEW,
            CaseStatus.PROCESSING_FAILED,
        }
    ),
    CaseStatus.RECOMMENDATION_COMPLETED: frozenset(),
    CaseStatus.HUMAN_REVIEW: frozenset(),
    CaseStatus.PROCESSING_FAILED: frozenset(),
}


def can_transition(
    current_status: CaseStatus,
    target_status: CaseStatus,
) -> bool:
    """Return whether the requested workflow transition is allowed."""

    return target_status in _ALLOWED_TRANSITIONS[current_status]


def transition_case(
    case: CaseState,
    target_status: CaseStatus,
) -> CaseState:
    """
    Return a copy of the case transitioned to the requested status.

    The original case is not mutated.
    """

    if not can_transition(case.status, target_status):
        raise InvalidCaseTransitionError(
            f"Invalid case transition: "
            f"{case.status.value} -> {target_status.value}"
        )

    transitioned_case = case.model_copy(deep=True)
    transitioned_case.status = target_status

    return transitioned_case