import sqlite3
from uuid import uuid4

import pytest

from underwriting.domain.case import CaseMetadata, CaseState, CaseStatus
from underwriting.repositories.case_repository import (
    CaseAlreadyExistsError,
    CaseNotFoundError,
    ConcurrencyConflictError,
)
from underwriting.repositories.sqlite_case_repository import (
    SQLiteCaseRepository,
)


@pytest.fixture
def repository(tmp_path):
    database_path = tmp_path / "test_underwriting.db"
    return SQLiteCaseRepository(database_path)


def test_create_and_get_case(repository):
    case = CaseState()

    repository.create(case)

    loaded = repository.get(case.metadata.case_id)

    assert loaded == case


def test_create_duplicate_case_raises_error(repository):
    case = CaseState()

    repository.create(case)

    with pytest.raises(CaseAlreadyExistsError):
        repository.create(case)


def test_get_missing_case_raises_error(repository):
    missing_case_id = uuid4()

    with pytest.raises(CaseNotFoundError):
        repository.get(missing_case_id)


def test_save_updates_case_and_increments_version(repository):
    case = CaseState()
    repository.create(case)

    case.status = CaseStatus.INTAKE_COMPLETED

    saved = repository.save(case)

    assert saved.status == CaseStatus.INTAKE_COMPLETED
    assert saved.metadata.version == 2

    loaded = repository.get(case.metadata.case_id)

    assert loaded.status == CaseStatus.INTAKE_COMPLETED
    assert loaded.metadata.version == 2


def test_stale_case_update_raises_concurrency_conflict(repository):
    case = CaseState()
    repository.create(case)

    worker_a_case = repository.get(case.metadata.case_id)
    worker_b_case = repository.get(case.metadata.case_id)

    worker_a_case.status = CaseStatus.INTAKE_COMPLETED
    repository.save(worker_a_case)

    worker_b_case.status = CaseStatus.HUMAN_REVIEW

    with pytest.raises(ConcurrencyConflictError):
        repository.save(worker_b_case)


def test_idempotency_key_finds_its_case(repository):
    case = CaseState(
        metadata=CaseMetadata(idempotency_key="submit-123")
    )
    repository.create(case)

    assert repository.find_by_idempotency_key("submit-123") == case
    assert repository.find_by_idempotency_key("other") is None


def test_idempotency_key_cannot_be_reused(repository):
    repository.create(
        CaseState(metadata=CaseMetadata(idempotency_key="submit-123"))
    )

    with pytest.raises(CaseAlreadyExistsError):
        repository.create(
            CaseState(
                metadata=CaseMetadata(idempotency_key="submit-123")
            )
        )


def test_cases_without_idempotency_key_do_not_collide(repository):
    repository.create(CaseState())
    repository.create(CaseState())


def test_find_open_by_applicant_excludes_closed_and_other_applicants(
    repository,
):
    def create_case(applicant_key: str, status: CaseStatus) -> CaseState:
        case = CaseState(
            metadata=CaseMetadata(applicant_key=applicant_key),
            status=status,
        )
        repository.create(case)
        return case

    in_progress = create_case("applicant-a", CaseStatus.RISK_ASSESSED)
    in_review = create_case("applicant-a", CaseStatus.HUMAN_REVIEW)
    create_case("applicant-a", CaseStatus.RECOMMENDATION_COMPLETED)
    create_case("applicant-a", CaseStatus.PROCESSING_FAILED)
    create_case("applicant-b", CaseStatus.HUMAN_REVIEW)

    open_cases = repository.find_open_by_applicant("applicant-a")

    assert [case.metadata.case_id for case in open_cases] == [
        in_progress.metadata.case_id,
        in_review.metadata.case_id,
    ]


def test_find_approved_by_applicant_returns_only_approved_completed_cases(
    repository,
):
    def create_case(
        applicant_key: str,
        status: CaseStatus,
        decision: str | None,
    ) -> CaseState:
        case = CaseState(
            metadata=CaseMetadata(applicant_key=applicant_key),
            status=status,
            recommendation=(
                {"recommendation": {"decision": decision}}
                if decision is not None
                else None
            ),
        )
        repository.create(case)
        return case

    first = create_case(
        "applicant-a", CaseStatus.RECOMMENDATION_COMPLETED, "approve"
    )
    create_case("applicant-a", CaseStatus.RECOMMENDATION_COMPLETED, "deny")
    create_case("applicant-a", CaseStatus.HUMAN_REVIEW, "refer")
    create_case("applicant-a", CaseStatus.HUMAN_REVIEW, "approve")
    create_case("applicant-a", CaseStatus.PROCESSING_FAILED, None)
    create_case("applicant-b", CaseStatus.RECOMMENDATION_COMPLETED, "approve")
    second = create_case(
        "applicant-a", CaseStatus.RECOMMENDATION_COMPLETED, "approve"
    )

    approved = repository.find_approved_by_applicant("applicant-a")

    assert [case.metadata.case_id for case in approved] == [
        first.metadata.case_id,
        second.metadata.case_id,
    ]


def test_save_updates_applicant_key(repository):
    case = CaseState()
    repository.create(case)

    case.metadata.applicant_key = "applicant-a"
    case.status = CaseStatus.INTAKE_COMPLETED
    repository.save(case)

    assert [
        open_case.metadata.case_id
        for open_case in repository.find_open_by_applicant("applicant-a")
    ] == [case.metadata.case_id]


def test_legacy_schema_gains_new_columns(tmp_path):
    database_path = tmp_path / "legacy.db"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE underwriting_cases (
                case_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                case_data TEXT NOT NULL
            )
            """
        )
    connection.close()

    repository = SQLiteCaseRepository(database_path)

    case = CaseState(
        metadata=CaseMetadata(
            idempotency_key="submit-123",
            applicant_key="applicant-a",
        )
    )
    repository.create(case)

    assert repository.find_by_idempotency_key("submit-123") == case
    assert repository.find_open_by_applicant("applicant-a") == [case]


def test_save_missing_case_raises_not_found(repository):
    with pytest.raises(CaseNotFoundError):
        repository.save(CaseState())


def test_cases_remain_isolated(repository):
    first_case = CaseState()
    second_case = CaseState()

    repository.create(first_case)
    repository.create(second_case)

    first_case.status = CaseStatus.HUMAN_REVIEW
    repository.save(first_case)

    loaded_first = repository.get(first_case.metadata.case_id)
    loaded_second = repository.get(second_case.metadata.case_id)

    assert loaded_first.status == CaseStatus.HUMAN_REVIEW
    assert loaded_second.status == CaseStatus.RECEIVED