import json
from pathlib import Path
from typing import Any
from uuid import UUID


class SyntheticDataStoreError(Exception):
    """Raised when a synthetic enrichment dataset cannot be loaded."""


class SyntheticDataStore:
    """
    Read-only access to a synthetic external enrichment dataset.

    Records are keyed by application ID to provide deterministic,
    auditable test scenarios.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def get_record(
        self,
        application_id: UUID,
    ) -> dict[str, Any] | None:
        records = self._load_records()

        record = records.get(
            str(application_id)
        )

        if record is None:
            return None

        if not isinstance(record, dict):
            raise SyntheticDataStoreError(
                f"Record for application {application_id} "
                "must be a JSON object."
            )

        return record

    def _load_records(self) -> dict[str, Any]:
        try:
            with self.file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (OSError, json.JSONDecodeError) as exc:
            raise SyntheticDataStoreError(
                f"Unable to load synthetic dataset: "
                f"{self.file_path}"
            ) from exc

        if not isinstance(data, dict):
            raise SyntheticDataStoreError(
                f"Synthetic dataset must contain "
                f"a JSON object: {self.file_path}"
            )

        return data