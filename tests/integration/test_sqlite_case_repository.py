from uuid import uuid4

import pytest

from underwriting.domain.case import CaseState, CaseStatus
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