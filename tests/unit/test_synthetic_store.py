import json

import pytest

from underwriting.enrichment.synthetic_store import (
    SyntheticDataStore,
    SyntheticDataStoreError,
)

SSN = "901-00-0001"


def write_dataset(tmp_path, content: str):
    data_file = tmp_path / "records.json"
    data_file.write_text(content, encoding="utf-8")
    return data_file


def test_get_record_returns_matching_record(tmp_path):
    record = {"status": "available"}
    data_file = write_dataset(
        tmp_path,
        json.dumps({SSN: record}),
    )

    store = SyntheticDataStore(data_file)

    assert store.get_record(SSN) == record


def test_get_record_returns_none_for_unknown_application(tmp_path):
    data_file = write_dataset(tmp_path, "{}")

    store = SyntheticDataStore(data_file)

    assert store.get_record(SSN) is None


def test_non_object_record_is_rejected(tmp_path):
    data_file = write_dataset(
        tmp_path,
        json.dumps({SSN: ["available"]}),
    )

    store = SyntheticDataStore(data_file)

    with pytest.raises(
        SyntheticDataStoreError,
        match="must be a JSON object",
    ):
        store.get_record(SSN)


def test_missing_dataset_file_is_rejected(tmp_path):
    store = SyntheticDataStore(tmp_path / "missing.json")

    with pytest.raises(
        SyntheticDataStoreError,
        match="Unable to load synthetic dataset",
    ):
        store.get_record(SSN)


def test_malformed_dataset_json_is_rejected(tmp_path):
    data_file = write_dataset(tmp_path, "{not-json")

    store = SyntheticDataStore(data_file)

    with pytest.raises(
        SyntheticDataStoreError,
        match="Unable to load synthetic dataset",
    ):
        store.get_record(SSN)


def test_non_object_dataset_is_rejected(tmp_path):
    data_file = write_dataset(tmp_path, "[]")

    store = SyntheticDataStore(data_file)

    with pytest.raises(
        SyntheticDataStoreError,
        match="must contain a JSON object",
    ):
        store.get_record(SSN)
