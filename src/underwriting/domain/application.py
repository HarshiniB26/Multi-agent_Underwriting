import hashlib
from datetime import date
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

US_STATE_CODES = frozenset(
    {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
        "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
        "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
        "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
        "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
        "WY",
    }
)


NAME_PUNCTUATION = frozenset(" -'’.")

LLM_EXCLUDED_FIELDS = frozenset(
    {
        "ssn",
        "first_name",
        "last_name",
        "date_of_birth",
    }
)


class TobaccoUse(StrEnum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class Application(BaseModel):
    """
    Validated applicant-provided information used by the underwriting workflow.

    This model represents synthetic educational data and does not reproduce
    any insurer's proprietary underwriting application.

    application_id is assigned by the server when the application is
    received. ssn is the strong identifier used to match external
    evidence. Direct identifiers (LLM_EXCLUDED_FIELDS) are never sent to
    the LLM; no agent's reasoning depends on them.
    """

    application_id: UUID = Field(default_factory=uuid4)

    ssn: str = Field(
        pattern=r"^\d{3}-\d{2}-\d{4}$",
        repr=False,
    )

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

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name_characters(cls, value: str) -> str:
        if not value[0].isalpha() or not all(
            char.isalpha() or char in NAME_PUNCTUATION
            for char in value
        ):
            raise ValueError(
                "must start with a letter and contain only letters, "
                "spaces, hyphens, apostrophes, or periods."
            )

        return value

    @field_validator("ssn", mode="before")
    @classmethod
    def normalize_ssn(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        compact = value.replace("-", "").replace(" ", "")

        if len(compact) == 9 and compact.isdigit():
            return f"{compact[:3]}-{compact[3:5]}-{compact[5:]}"

        return value

    @field_validator("state", mode="before")
    @classmethod
    def normalize_state(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        return value.strip().upper()

    @field_validator("state")
    @classmethod
    def validate_state_code(cls, value: str) -> str:
        if value not in US_STATE_CODES:
            raise ValueError(
                f"{value!r} is not a US state or DC postal code."
            )

        return value

    @field_validator("date_of_birth")
    @classmethod
    def reject_future_date_of_birth(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("Date of birth cannot be in the future.")

        return value

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

    @property
    def applicant_key(self) -> str:
        """
        Key for finding other cases belonging to the same person.

        An unsalted hash of a 9-digit SSN can be brute-forced; a
        production system should use a keyed HMAC with a managed secret.
        """

        return hashlib.sha256(
            self.ssn.encode("utf-8")
        ).hexdigest()