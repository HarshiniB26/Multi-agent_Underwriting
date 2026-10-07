from datetime import date
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class TobaccoUse(StrEnum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class Application(BaseModel):
    """
    Validated applicant-provided information used by the underwriting workflow.

    This model represents synthetic educational data and does not reproduce
    any insurer's proprietary underwriting application.
    """

    application_id: UUID = Field(default_factory=uuid4)

    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)

    date_of_birth: date
    state: str = Field(min_length=2, max_length=2)

    annual_income: int = Field(gt=0)
    coverage_amount: int = Field(gt=0)

    occupation: str = Field(min_length=1, max_length=150)

    tobacco_use: TobaccoUse

    height_cm: int = Field(ge=100, le=250)
    weight_kg: float = Field(ge=30, le=300)

    medical_conditions_declared: list[str] = Field(default_factory=list)
    medications_declared: list[str] = Field(default_factory=list)

    @field_validator(
        "first_name",
        "last_name",
        "occupation",
        mode="before",
    )
    @classmethod
    def strip_text_fields(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        return value.strip()

    @field_validator("state", mode="before")
    @classmethod
    def normalize_state(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        return value.strip().upper()

    @field_validator(
        "medical_conditions_declared",
        "medications_declared",
    )
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        return [
            value.strip()
            for value in values
            if value.strip()
        ]