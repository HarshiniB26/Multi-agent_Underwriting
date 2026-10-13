from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from underwriting.domain.application import Application, TobaccoUse


def valid_application_data():
    return {
        "ssn": "901-00-0001",
        "first_name": "Aarav",
        "last_name": "Sharma",
        "date_of_birth": date(1990, 5, 15),
        "state": "PA",
        "annual_income": 90000,
        "coverage_amount": 500000,
        "occupation": "Software Engineer",
        "tobacco_use": "never",
        "height_cm": 175,
        "weight_kg": 75,
        "medical_conditions_declared": [],
        "medications_declared": [],
    }


def test_valid_application_is_created():
    application = Application(**valid_application_data())

    assert application.first_name == "Aarav"
    assert application.tobacco_use == TobaccoUse.NEVER
    assert application.annual_income == 90000


def test_application_normalizes_state():
    data = valid_application_data()
    data["state"] = "pa"

    application = Application(**data)

    assert application.state == "PA"


def test_application_strips_text_fields():
    data = valid_application_data()
    data["first_name"] = "  Aarav  "
    data["occupation"] = "  Software Engineer  "

    application = Application(**data)

    assert application.first_name == "Aarav"
    assert application.occupation == "Software Engineer"


def test_blank_name_is_rejected():
    data = valid_application_data()
    data["first_name"] = "   "

    with pytest.raises(ValidationError):
        Application(**data)


@pytest.mark.parametrize(
    "name",
    ["O'Brien", "Mary-Jane", "José", "St. John", "D’Souza", "Van der Berg"],
)
def test_real_world_names_are_accepted(name):
    data = valid_application_data()
    data["last_name"] = name

    assert Application(**data).last_name == name


@pytest.mark.parametrize("field", ["first_name", "last_name"])
@pytest.mark.parametrize(
    "name",
    ["1234", "Aarav2", "@@@", "-Smith", "'Neil", "John_Doe", "Ann!"],
)
def test_names_with_digits_or_symbols_are_rejected(field, name):
    data = valid_application_data()
    data[field] = name

    with pytest.raises(ValidationError) as exc_info:
        Application(**data)

    assert exc_info.value.errors()[0]["loc"] == (field,)
    assert name not in str(exc_info.value.errors()[0]["msg"])


@pytest.mark.parametrize(
    "field",
    ["first_name", "state"],
)
def test_non_string_text_field_is_rejected(field):
    data = valid_application_data()
    data[field] = 12

    with pytest.raises(ValidationError) as exc_info:
        Application(**data)

    assert exc_info.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize(
    "ssn",
    ["901-00-0001", "901000001", "901 00 0001"],
)
def test_ssn_is_normalized(ssn):
    data = valid_application_data()
    data["ssn"] = ssn

    assert Application(**data).ssn == "901-00-0001"


@pytest.mark.parametrize(
    "ssn",
    ["90100001", "901-00-000A", "901/00/0001", 901000001],
)
def test_invalid_ssn_is_rejected(ssn):
    data = valid_application_data()
    data["ssn"] = ssn

    with pytest.raises(ValidationError) as exc_info:
        Application(**data)

    assert exc_info.value.errors()[0]["loc"] == ("ssn",)


def test_ssn_is_hidden_from_repr():
    application = Application(**valid_application_data())

    assert "901-00-0001" not in repr(application)


def test_applicant_key_is_stable_per_ssn():
    first = Application(**valid_application_data())

    other_data = valid_application_data()
    other_data["first_name"] = "Someone"
    same_person = Application(**other_data)

    other_data["ssn"] = "902-00-0002"
    different_person = Application(**other_data)

    assert first.applicant_key == same_person.applicant_key
    assert first.applicant_key != different_person.applicant_key
    assert len(first.applicant_key) == 64
    assert "901-00-0001" not in first.applicant_key


def test_unknown_state_code_is_rejected():
    data = valid_application_data()
    data["state"] = "zz"

    with pytest.raises(
        ValidationError,
        match="'ZZ' is not a US state or DC postal code",
    ):
        Application(**data)


def test_district_of_columbia_is_accepted():
    data = valid_application_data()
    data["state"] = "dc"

    assert Application(**data).state == "DC"


def test_future_date_of_birth_is_rejected():
    data = valid_application_data()
    data["date_of_birth"] = date.today() + timedelta(days=1)

    with pytest.raises(
        ValidationError,
        match="Date of birth cannot be in the future",
    ):
        Application(**data)


def test_invalid_tobacco_value_is_rejected():
    data = valid_application_data()
    data["tobacco_use"] = "sometimes"

    with pytest.raises(ValidationError):
        Application(**data)


def test_negative_income_is_rejected():
    data = valid_application_data()
    data["annual_income"] = -50000

    with pytest.raises(ValidationError):
        Application(**data)


def test_invalid_height_is_rejected():
    data = valid_application_data()
    data["height_cm"] = 20

    with pytest.raises(ValidationError):
        Application(**data)


def test_declared_lists_are_normalized():
    data = valid_application_data()
    data["medical_conditions_declared"] = [
        " hypertension ",
        "",
        "   ",
    ]

    application = Application(**data)

    assert application.medical_conditions_declared == [
        "hypertension"
    ]