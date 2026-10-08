import json
from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError

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
    """Outcome of the enrichment processing stage."""

    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"


class EnrichmentReview(BaseModel):
    """
    Structured LLM interpretation of already collected enrichment evidence.

    This review is explanatory only. It cannot create, remove, or modify
    external evidence or make an underwriting recommendation.
    """

    summary: str = Field(min_length=1)
    observations: list[str] = Field(default_factory=list)


class EnrichmentResult(BaseModel):
    """Result returned by the standalone enrichment agent."""

    status: EnrichmentStatus
    evidence: EnrichmentEvidence

    discrepancies: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    review: EnrichmentReview
    llm_response: LLMResponse


ENRICHMENT_SYSTEM_PROMPT = """
You are the enrichment review component of a synthetic educational
life-insurance underwriting system.

You will receive:
1. applicant-provided information,
2. validated synthetic external evidence,
3. deterministic discrepancies identified by application code, and
4. external evidence sources that were not found.

Your responsibility is limited to explaining the supplied information
clearly and concisely.

You must not:
- invent applicant or external facts
- modify external evidence
- decide whether a discrepancy exists
- calculate an underwriting risk score
- approve or deny insurance coverage
- make a final underwriting recommendation
- treat missing evidence as favorable evidence

Return JSON only using exactly this structure:

{
  "summary": "brief evidence summary",
  "observations": ["observation 1", "observation 2"]
}

Use only facts supplied in the prompt.
""".strip()


def _normalize_text(value: str) -> str:
    """Normalize text for deterministic comparisons."""

    return value.strip().casefold()


def _find_discrepancies(
    application: Application,
    evidence: EnrichmentEvidence,
) -> list[str]:
    """
    Compare applicant declarations with available external evidence.

    These comparisons are deterministic and do not rely on the LLM.
    """

    discrepancies: list[str] = []

    identity = evidence.identity

    if identity.status == EvidenceStatus.CONFLICT:
        discrepancies.append(
            "External identity evidence has a conflict."
        )

    elif identity.status == EvidenceStatus.AVAILABLE:
        if identity.identity_verified is False:
            discrepancies.append(
                "External identity evidence did not verify the applicant."
            )

        if identity.name_match is False:
            discrepancies.append(
                "Applicant name does not match external identity evidence."
            )

        if identity.date_of_birth_match is False:
            discrepancies.append(
                "Applicant date of birth does not match external identity evidence."
            )

        if identity.state_match is False:
            discrepancies.append(
                "Applicant state does not match external identity evidence."
            )

    medical = evidence.medical

    if medical.status == EvidenceStatus.CONFLICT:
        discrepancies.append(
            "External medical evidence has a conflict."
        )

    if medical.status == EvidenceStatus.AVAILABLE:
        declared_conditions = {
            _normalize_text(condition)
            for condition in application.medical_conditions_declared
        }

        for condition in medical.conditions:
            if (
                _normalize_text(condition.condition)
                not in declared_conditions
            ):
                discrepancies.append(
                    "External medical evidence contains an undeclared "
                    f"condition: {condition.condition}."
                )

    prescription = evidence.prescription

    if prescription.status == EvidenceStatus.CONFLICT:
        discrepancies.append(
            "External prescription evidence has a conflict."
        )

    if prescription.status == EvidenceStatus.AVAILABLE:
        declared_medications = {
            _normalize_text(medication)
            for medication in application.medications_declared
        }

        for record in prescription.prescriptions:
            if (
                record.active
                and _normalize_text(record.medication)
                not in declared_medications
            ):
                discrepancies.append(
                    "External prescription evidence contains an undeclared "
                    f"active medication: {record.medication}."
                )

    financial = evidence.financial

    if financial.status == EvidenceStatus.CONFLICT:
        discrepancies.append(
            "External financial evidence has a conflict."
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
                "Applicant annual income differs from verified "
                "external annual income."
            )

    return discrepancies


def _find_missing_evidence(
    evidence: EnrichmentEvidence,
) -> list[str]:
    """Identify external sources for which no record was available."""

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
    """Construct explicit case context for the enrichment LLM invocation."""

    payload = {
        "application": application.model_dump(
            mode="json"
        ),
        "external_evidence": evidence.model_dump(
            mode="json"
        ),
        "deterministic_discrepancies": discrepancies,
        "missing_evidence": missing_evidence,
    }

    return (
        "Explain the following validated synthetic enrichment result.\n\n"
        + json.dumps(
            payload,
            indent=2,
        )
    )


def _parse_enrichment_review(
    content: str,
) -> EnrichmentReview:
    """Validate untrusted LLM output."""

    try:
        parsed = json.loads(content)

        return EnrichmentReview.model_validate(
            parsed
        )

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
    Retrieve external evidence and perform bounded LLM interpretation.

    External providers are authoritative for retrieved evidence.
    Deterministic code identifies discrepancies and missing evidence.
    The LLM is used only to explain the resulting evidence.
    """

    identity = identity_provider.get_identity_evidence(
        application
    )

    medical = medical_provider.get_medical_evidence(
        application
    )

    prescription = (
        prescription_provider.get_prescription_evidence(
            application
        )
    )

    financial = financial_provider.get_financial_evidence(
        application
    )

    evidence = EnrichmentEvidence(
        identity=identity,
        medical=medical,
        prescription=prescription,
        financial=financial,
    )

    discrepancies = _find_discrepancies(
        application,
        evidence,
    )

    missing_evidence = _find_missing_evidence(
        evidence
    )

    llm_response = llm_client.generate(
        system_prompt=ENRICHMENT_SYSTEM_PROMPT,
        user_prompt=_build_review_prompt(
            application,
            evidence,
            discrepancies,
            missing_evidence,
        ),
    )

    review = _parse_enrichment_review(
        llm_response.content
    )

    status = (
        EnrichmentStatus.NEEDS_REVIEW
        if discrepancies or missing_evidence
        else EnrichmentStatus.COMPLETED
    )

    return EnrichmentResult(
        status=status,
        evidence=evidence,
        discrepancies=discrepancies,
        missing_evidence=missing_evidence,
        review=review,
        llm_response=llm_response,
    )