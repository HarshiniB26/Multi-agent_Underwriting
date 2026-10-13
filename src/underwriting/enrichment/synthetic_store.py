import json
from pathlib import Path
from typing import Any


class SyntheticDataStoreError(Exception):
    """Raised when a synthetic enrichment dataset cannot be loaded."""


class SyntheticDataStore:
    """
    Read-only access to a synthetic external enrichment dataset.

    Records are keyed by the applicant's SSN, mirroring how external
    data vendors are searched by a strong personal identifier rather
    than by the insurer's internal application ID.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def get_record(
        self,
        key: str,
    ) -> dict[str, Any] | None:
        records = self._load_records()

        record = records.get(key)

        if record is None:
            return None

        if not isinstance(record, dict):
            raise SyntheticDataStoreError(
                f"Record in {self.file_path.name} "
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