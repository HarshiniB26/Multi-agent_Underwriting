# Synthetic Recommendation Policy

## Purpose

This document defines the deterministic recommendation policy used by the
educational multi-agent life-insurance underwriting project.

The recommendation rules are synthetic coursework rules. They do not
represent the proprietary underwriting rules of any insurer and must not
be used for real insurance decisions.

## Design Principles

The recommendation stage is:

1. deterministic
2. explainable
3. auditable
4. conservative when evidence is incomplete
5. independent of LLM-generated decisions

The LLM explains the authoritative recommendation but does not choose
APPROVE, DENY, or REFER.

## Inputs

The recommendation policy consumes the authoritative RiskAssessment
produced by synthetic-life-v1.

The RiskAssessment contains:

- risk score
- risk tier
- risk factors
- review flags
- requires_review
- policy version

## Decision Precedence

Recommendation rules are evaluated in the following order:

1. Mandatory review
2. Very-high risk
3. High risk
4. Low or moderate risk

The ordering is important because evidence-quality problems take
precedence over automatic approval or denial.

## REFER

A case receives REFER when:

requires_review = true

This includes cases with conditions such as:

- missing evidence
- conflicting evidence
- undeclared medical evidence
- undeclared active medication
- unsupported medical condition
- excessive coverage-to-income ratio

REFER means that the automated workflow does not have sufficient
confidence or policy authority to produce an automatic approval or denial.

REFER is a business recommendation for human underwriting review.

## DENY

A case receives DENY when:

- requires_review = false
- risk tier = VERY_HIGH

For synthetic-life-v1, VERY_HIGH corresponds to a deterministic risk
score of 50 or greater.

A case requiring review is never automatically denied solely because of
its numerical score. Review takes precedence.

## APPROVE

A case receives APPROVE when:

- requires_review = false
- risk tier is LOW, MODERATE, or HIGH

This simplified coursework policy intentionally treats these tiers as
eligible for automated approval.

The risk tier remains available in the result so downstream systems could
apply additional pricing, coverage, or underwriting rules in a more
complete implementation.

## Decision Matrix

| Requires Review | Risk Tier | Recommendation |
|---|---|---|
| Yes | Any | REFER |
| No | VERY_HIGH | DENY |
| No | HIGH | APPROVE |
| No | MODERATE | APPROVE |
| No | LOW | APPROVE |

## Important Precedence Example

Suppose a case has:

score = 55
tier = VERY_HIGH
requires_review = true

The recommendation is:

REFER

not:

DENY

The unresolved evidence-quality issue takes precedence over automatic
denial.

## Separation of Responsibilities

The deterministic recommendation policy owns:

- APPROVE
- DENY
- REFER
- decision reason code
- policy version

The LLM owns only:

- explanation
- summarization
- presentation of the already calculated recommendation

The LLM must not:

- change APPROVE to DENY
- change DENY to APPROVE
- change REFER to another decision
- remove review requirements
- modify the risk score
- modify the risk tier
- invent applicant facts
- invent underwriting rules

## Human Review

REFER routes the case to a human underwriting review path.

Examples include:

- missing external evidence
- provider evidence conflict
- undeclared medical information
- unsupported medical conditions
- excessive requested coverage relative to income

Human review is therefore a first-class business outcome rather than an
exception or system failure.

## Policy Version

Recommendation policy version:

synthetic-recommendation-v1

This version is stored with the recommendation so historical decisions
can be traced back to the rules used at the time.

## Educational Scope

All recommendation thresholds and decision rules are synthetic and exist
only to demonstrate deterministic multi-agent decision architecture.