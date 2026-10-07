from datetime import date

import pytest
from pydantic import ValidationError

from underwriting.domain.application import Application, TobaccoUse


def valid_application_data():
    return {
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