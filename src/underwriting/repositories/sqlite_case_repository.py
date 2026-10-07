import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from underwriting.domain.case import CaseState
from underwriting.repositories.case_repository import (
    CaseAlreadyExistsError,
    CaseNotFoundError,
    CaseRepository,
    ConcurrencyConflictError,
)


class SQLiteCaseRepository(CaseRepository):
    """SQLite implementation of the underwriting case repository."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS underwriting_cases (
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
    def create(self, case: CaseState) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO underwriting_cases (
                        case_id,
                        trace_id,
                        status,
                        version,
                        created_at,
                        updated_at,
                        case_data
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(case.metadata.case_id),
                        str(case.metadata.trace_id),
                        case.status.value,
                        case.metadata.version,
                        case.metadata.created_at.isoformat(),
                        case.metadata.updated_at.isoformat(),
                        case.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise CaseAlreadyExistsError(
                f"Case {case.metadata.case_id} already exists."
            ) from exc
    def get(self, case_id: UUID) -> CaseState:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT case_data
                FROM underwriting_cases
                WHERE case_id = ?
                """,
                (str(case_id),),
            ).fetchone()

        if row is None:
            raise CaseNotFoundError(
                f"Case {case_id} was not found."
            )

        return CaseState.model_validate_json(row["case_data"])

    def save(self, case: CaseState) -> CaseState:
        current_version = case.metadata.version
        new_version = current_version + 1
        updated_at = datetime.now(UTC)

        persisted_case = case.model_copy(deep=True)

        persisted_case.metadata.version = new_version
        persisted_case.metadata.updated_at = updated_at

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE underwriting_cases
                SET
                    trace_id = ?,
                    status = ?,
                    version = ?,
                    updated_at = ?,
                    case_data = ?
                WHERE case_id = ?
                  AND version = ?
                """,
                (
                    str(persisted_case.metadata.trace_id),
                    persisted_case.status.value,
                    new_version,
                    updated_at.isoformat(),
                    persisted_case.model_dump_json(),
                    str(persisted_case.metadata.case_id),
                    current_version,
                ),
            )

            if cursor.rowcount == 0:
                exists = connection.execute(
                    """
                    SELECT 1
                    FROM underwriting_cases
                    WHERE case_id = ?
                    """,
                    (str(case.metadata.case_id),),
                ).fetchone()

                if exists is None:
                    raise CaseNotFoundError(
                        f"Case {case.metadata.case_id} was not found."
                    )

                raise ConcurrencyConflictError(
                    f"Case {case.metadata.case_id} was modified "
                    "by another process."
                )

        return persisted_case