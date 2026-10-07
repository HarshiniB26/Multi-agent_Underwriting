import pytest

from underwriting.repositories.case_repository import CaseRepository


def test_case_repository_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        CaseRepository()