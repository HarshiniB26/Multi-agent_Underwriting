import pytest

from underwriting.enrichment.matching import (
    identity_mismatches,
    income_discrepancy,
    income_matches,
    normalize_condition,
    normalize_text,
)
from underwriting.enrichment.models import (
    EvidenceStatus,
    FinancialEvidence,
    IdentityEvidence,
)


def test_normalize_text_collapses_whitespace_and_case():
    assert normalize_text("  Type   2\tDiabetes ") == "type 2 diabetes"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("High Blood Pressure", "hypertension"),
        ("HTN", "hypertension"),
        ("Type II Diabetes", "type 2 diabetes"),
        ("T2DM", "type 2 diabetes"),
        ("Hyperlipidemia", "high cholesterol"),
        ("Underactive Thyroid", "hypothyroidism"),
        ("Asthma", "asthma"),
        ("Synthetic Unknown Condition", "synthetic unknown condition"),
    ],
)
def test_normalize_condition_maps_synonyms(value, expected):
    assert normalize_condition(value) == expected


def test_identity_mismatches_lists_only_explicit_failures():
    identity = IdentityEvidence(
        status=EvidenceStatus.AVAILABLE,
        identity_verified=True,
        name_match=False,
        date_of_birth_match=None,
        state_match=False,
    )

    assert identity_mismatches(identity) == [
        "name_match",
        "state_match",
    ]


@pytest.mark.parametrize(
    ("verified", "declared", "expected"),
    [
        (100000, 100000, True),
        (95000, 100000, True),
        (105000, 100000, True),
        (94999, 100000, False),
        (105001, 100000, False),
    ],
)
def test_income_matches_within_five_percent(
    verified,
    declared,
    expected,
):
    assert income_matches(verified, declared) is expected


@pytest.mark.parametrize(
    ("financial", "expected"),
    [
        (
            FinancialEvidence(
                status=EvidenceStatus.AVAILABLE,
                income_verified=True,
                verified_annual_income=99000,
            ),
            False,
        ),
        (
            FinancialEvidence(
                status=EvidenceStatus.AVAILABLE,
                income_verified=True,
                verified_annual_income=60000,
            ),
            True,
        ),
        (
            FinancialEvidence(
                status=EvidenceStatus.AVAILABLE,
                income_verified=False,
            ),
            True,
        ),
        (
            FinancialEvidence(
                status=EvidenceStatus.AVAILABLE,
                income_verified=None,
            ),
            False,
        ),
    ],
)
def test_income_discrepancy(financial, expected):
    assert income_discrepancy(financial, 100000) is expected
