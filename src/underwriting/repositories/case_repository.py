from abc import ABC, abstractmethod
from uuid import UUID

from underwriting.domain.case import CaseState


class CaseNotFoundError(Exception):
    """Raised when an underwriting case cannot be found."""


class CaseAlreadyExistsError(Exception):
    """Raised when attempting to create a case that already exists."""


class ConcurrencyConflictError(Exception):
    """Raised when attempting to save a stale version of a case."""


class CaseRepository(ABC):
    """
    Persistence contract for underwriting case state.

    Domain and orchestration code depend on this abstraction rather than
    on a specific database implementation.
    """

    @abstractmethod
    def create(self, case: CaseState) -> None:
        """Persist a new underwriting case."""
        raise NotImplementedError

    @abstractmethod
    def get(self, case_id: UUID) -> CaseState:
        """Retrieve an underwriting case by its unique identifier."""
        raise NotImplementedError

    @abstractmethod
    def save(self, case: CaseState) -> CaseState:
        """
        Persist changes to an existing case.

        Implementations must protect against stale writes using optimistic
        concurrency control and return the newly persisted case state.
        """
        raise NotImplementedError