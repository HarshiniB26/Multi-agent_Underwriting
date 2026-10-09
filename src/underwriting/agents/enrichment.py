import json
from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError, model_validator

from underwriting.domain.application import Application
from underwriting.enrichment.models import (
    EnrichmentEvidence,
    EvidenceStatus,
)
from underwriting.enrichment.providers import (
    FinancialProvider,
    IdentityProvider,
    MedicalHistoryProvider,
    PrescriptionProvider,
)
from underwriting.llm.client import LLMClient, LLMResponse


class EnrichmentStatus(StrEnum):
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"


class EvidenceConsistency(StrEnum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    INCOMPLETE = "incomplete"


class EnrichmentReview(BaseModel):
    """
    Structured cross-source synthesis produced by the LLM.

    The LLM interprets relationships between validated evidence.
    It does not create or remove authoritative discrepancies.
    """

    evidence_consistency: EvidenceConsistency
    summary: str = Field(min_length=1)
    material_concerns: list[str] = Field(default_factory=list)
    corroborating_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review_consistency(self):
        if (
            self.evidence_consistency
            == EvidenceConsistency.CONSISTENT
            and self.material_concerns
        ):
            raise ValueError(
                "Consistent evidence must not contain material concerns."
            )

        return self


class EnrichmentResult(BaseModel):
    status: EnrichmentStatus
    evidence: EnrichmentEvidence
    discrepancies: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    review: EnrichmentReview
    llm_response: LLMResponse


ENRICHMENT_SYSTEM_PROMPT = """
You are the cross-source evidence synthesis component of a synthetic
educational life-insurance underwriting system.

You receive:

1. validated applicant-provided information
2. validated identity evidence
3. validated medical evidence
4. validated prescription evidence
5. validated financial evidence
6. deterministic discrepancies detected by application code
7. deterministic missing-evidence findings detected by application code

Your responsibility is to reason about relationships among those supplied
facts.

You should identify useful cross-source relationships such as:

- external medical and prescription evidence that corroborate one another
- external evidence that conflicts with applicant declarations
- multiple evidence sources that support the same concern
- incomplete evidence that limits interpretation
- evidence that is mutually consistent

The deterministic discrepancies and missing-evidence findings supplied
to you are authoritative.

You must not:

- remove or contradict an authoritative discrepancy
- remove or contradict a missing-evidence finding
- invent medical conditions
- invent medications
- invent identity information
- invent financial information
- infer that missing evidence means no risk exists
- calculate a risk score
- assign a risk tier
- approve or deny the applicant
- make the final underwriting recommendation
- invent underwriting policy

Use evidence_consistency="consistent" when:
- no deterministic discrepancies exist
- no evidence sources are missing

Use evidence_consistency="inconsistent" when:
- one or more deterministic discrepancies exist
- no evidence sources are missing

Use evidence_consistency="incomplete" when:
- one or more evidence sources are missing

If both discrepancies and missing evidence exist, use "incomplete"
because the complete evidence picture is unavailable.

Every material concern and corroborating relationship must be grounded
in the supplied application, evidence, or deterministic findings.

Return JSON only using this structure:

{
  "evidence_consistency": "consistent",
  "summary": "brief cross-source evidence synthesis",
  "material_concerns": [],
  "corroborating_evidence": []
}
""".strip()


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _find_discrepancies(
    application: Application,
    evidence: EnrichmentEvidence,
) -> list[str]:
    """
    Detect factual discrepancies deterministically.

    These findings are authoritative.
    """

    discrepancies: list[str] = []

    identity = evidence.identity

    if identity.status == EvidenceStatus.CONFLICT:
        discrepancies.append(
            "Identity evidence conflicts with the application."
        )

    elif identity.status == EvidenceStatus.AVAILABLE:
        if identity.identity_verified is False:
            discrepancies.append(
                "External identity evidence could not verify the applicant."
            )

        if identity.name_match is False:
            discrepancies.append(
                "Applicant name does not match external identity evidence."
            )

        if identity.date_of_birth_match is False:
            discrepancies.append(
                "Applicant date of birth does not match "
                "external identity evidence."
            )

        if identity.state_match is False:
            discrepancies.append(
                "Applicant state does not match external identity evidence."
            )

    declared_conditions = {
        _normalize(condition)
        for condition in application.medical_conditions_declared
    }

    if evidence.medical.status == EvidenceStatus.CONFLICT:
        discrepancies.append(
            "Medical evidence contains a provider-level conflict."
        )

    elif evidence.medical.status == EvidenceStatus.AVAILABLE:
        for condition in evidence.medical.conditions:
            if _normalize(condition.condition) not in declared_conditions:
                discrepancies.append(
                    "External medical evidence contains an undeclared "
                    f"condition: {condition.condition}."
                )

    declared_medications = {
        _normalize(medication)
        for medication in application.medications_declared
    }

    if evidence.prescription.status == EvidenceStatus.CONFLICT:
        discrepancies.append(
            "Prescription evidence contains a provider-level conflict."
        )

    elif evidence.prescription.status == EvidenceStatus.AVAILABLE:
        for prescription in evidence.prescription.prescriptions:
            if (
                prescription.active
                and _normalize(prescription.medication)
                not in declared_medications
            ):
                discrepancies.append(
                    "External prescription evidence contains an undeclared "
                    f"active medication: {prescription.medication}."
                )

    financial = evidence.financial

    if financial.status == EvidenceStatus.CONFLICT:
        discrepancies.append(
            "Financial evidence conflicts with the application."
        )

    elif financial.status == EvidenceStatus.AVAILABLE:
        if financial.income_verified is False:
            discrepancies.append(
                "Applicant income could not be verified."
            )

        if (
            financial.verified_annual_income is not None
            and financial.verified_annual_income
            != application.annual_income
        ):
            discrepancies.append(
                "Verified annual income does not match "
                "applicant-declared annual income."
            )

    return discrepancies


def _find_missing_evidence(
    evidence: EnrichmentEvidence,
) -> list[str]:
    """
    Detect unavailable external evidence sources deterministically.
    """

    missing: list[str] = []

    if evidence.identity.status == EvidenceStatus.NOT_FOUND:
        missing.append("identity")

    if evidence.medical.status == EvidenceStatus.NOT_FOUND:
        missing.append("medical")

    if evidence.prescription.status == EvidenceStatus.NOT_FOUND:
        missing.append("prescription")

    if evidence.financial.status == EvidenceStatus.NOT_FOUND:
        missing.append("financial")

    return missing


def _build_review_prompt(
    application: Application,
    evidence: EnrichmentEvidence,
    discrepancies: list[str],
    missing_evidence: list[str],
) -> str:
    payload = {
        "application": application.model_dump(mode="json"),
        "external_evidence": evidence.model_dump(mode="json"),
        "authoritative_discrepancies": discrepancies,
        "authoritative_missing_evidence": missing_evidence,
    }

    return (
        "Perform a cross-source synthesis of the following validated "
        "application and enrichment evidence.\n\n"
        + json.dumps(payload, indent=2)
    )


def _parse_enrichment_review(
    content: str,
) -> EnrichmentReview:
    try:
        parsed = json.loads(content)
        return EnrichmentReview.model_validate(parsed)

    except (
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise ValueError(
            "LLM returned an invalid enrichment review."
        ) from exc


def enrichment_agent(
    application: Application,
    identity_provider: IdentityProvider,
    medical_provider: MedicalHistoryProvider,
    prescription_provider: PrescriptionProvider,
    financial_provider: FinancialProvider,
    llm_client: LLMClient,
) -> EnrichmentResult:
    """
    Retrieve external evidence and perform cross-source synthesis.

    Deterministic code:
    - retrieves provider evidence
    - detects factual discrepancies
    - detects missing evidence
    - controls authoritative enrichment status

    LLM:
    - synthesizes relationships among validated evidence
    - identifies corroborating evidence
    - explains material concerns

    Provider failures intentionally propagate. Retry, backoff, and
    escalation belong to the orchestration/resilience layer.
    """

    evidence = EnrichmentEvidence(
        identity=identity_provider.get_identity_evidence(
            application
        ),
        medical=medical_provider.get_medical_evidence(
            application
        ),
        prescription=prescription_provider.get_prescription_evidence(
            application
        ),
        financial=financial_provider.get_financial_evidence(
            application
        ),
    )

    discrepancies = _find_discrepancies(
        application,
        evidence,
    )

    missing_evidence = _find_missing_evidence(
        evidence
    )

    status = (
        EnrichmentStatus.NEEDS_REVIEW
        if discrepancies or missing_evidence
        else EnrichmentStatus.COMPLETED
    )

    llm_response = llm_client.generate(
        system_prompt=ENRICHMENT_SYSTEM_PROMPT,
        user_prompt=_build_review_prompt(
            application=application,
            evidence=evidence,
            discrepancies=discrepancies,
            missing_evidence=missing_evidence,
        ),
    )

    review = _parse_enrichment_review(
        llm_response.content
    )

    return EnrichmentResult(
        status=status,
        evidence=evidence,
        discrepancies=discrepancies,
        missing_evidence=missing_evidence,
        review=review,
        llm_response=llm_response,
    )