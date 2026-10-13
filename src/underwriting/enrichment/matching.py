"""
Shared rules for comparing applicant declarations with external evidence.

Enrichment and risk scoring both depend on these rules, so they live in
one place to keep the two stages from disagreeing.
"""

from underwriting.enrichment.models import (
    FinancialEvidence,
    IdentityEvidence,
)

INCOME_MATCH_TOLERANCE = 0.05

_CONDITION_ALIASES: dict[str, str] = {
    "high blood pressure": "hypertension",
    "htn": "hypertension",
    "type ii diabetes": "type 2 diabetes",
    "diabetes type 2": "type 2 diabetes",
    "diabetes mellitus type 2": "type 2 diabetes",
    "type 2 diabetes mellitus": "type 2 diabetes",
    "t2d": "type 2 diabetes",
    "t2dm": "type 2 diabetes",
    "underactive thyroid": "hypothyroidism",
    "hypercholesterolemia": "high cholesterol",
    "hyperlipidemia": "high cholesterol",
}


def normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def normalize_condition(value: str) -> str:
    normalized = normalize_text(value)

    return _CONDITION_ALIASES.get(normalized, normalized)


def identity_mismatches(
    identity: IdentityEvidence,
) -> list[str]:
    """Return the identity checks that external evidence explicitly failed."""

    checks = {
        "identity_verified": identity.identity_verified,
        "name_match": identity.name_match,
        "date_of_birth_match": identity.date_of_birth_match,
        "state_match": identity.state_match,
    }

    return [
        name
        for name, value in checks.items()
        if value is False
    ]


def income_matches(
    verified_annual_income: int,
    declared_annual_income: int,
) -> bool:
    difference = abs(
        verified_annual_income - declared_annual_income
    )

    return (
        difference
        <= declared_annual_income * INCOME_MATCH_TOLERANCE
    )


def income_discrepancy(
    financial: FinancialEvidence,
    declared_annual_income: int,
) -> bool:
    """Whether available financial evidence fails to support declared income."""

    if financial.income_verified is False:
        return True

    return (
        financial.verified_annual_income is not None
        and not income_matches(
            financial.verified_annual_income,
            declared_annual_income,
        )
    )
