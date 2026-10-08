# Synthetic Underwriting Policy

## Purpose

This document defines the deterministic risk-scoring policy used by the
educational multi-agent life-insurance underwriting project.

The rules, thresholds, weights, and risk tiers in this document are
synthetic coursework rules. They do not represent the proprietary
underwriting rules of any insurer and must not be used for real insurance
decisions.

## Design Principles

The synthetic policy is designed to be:

1. deterministic
2. explainable
3. auditable
4. testable
5. independent of the LLM
6. conservative when evidence is incomplete or conflicting

The LLM does not create policy rules or calculate the authoritative risk
score.

## Input Sources

The policy engine consumes:

- validated applicant information
- identity evidence
- medical evidence
- prescription evidence
- financial evidence

Derived features may be calculated deterministically from those facts.

## Derived Features

### Age

Age is calculated from the applicant's date of birth relative to the
policy evaluation date.

### Body Mass Index

BMI is calculated as:

BMI = weight_kg / (height_m ^ 2)

BMI is used only as a synthetic coursework risk feature.

### Coverage-to-Income Ratio

Coverage-to-income ratio is calculated as:

requested coverage / annual income

This is used as a synthetic financial proportionality indicator.

## Base Score

Every case begins with a synthetic risk score of:

0

Risk points are added according to the rules below.

A higher score represents higher synthetic underwriting risk.

## Age Rules

| Age | Points |
|---|---:|
| 18-39 | 0 |
| 40-49 | 5 |
| 50-59 | 10 |
| 60-69 | 20 |
| 70+ | 30 |

## BMI Rules

| BMI | Points |
|---|---:|
| 18.5-29.9 | 0 |
| 30.0-34.9 | 5 |
| 35.0-39.9 | 10 |
| <18.5 | 10 |
| >=40.0 | 20 |

## Tobacco Rules

| Tobacco Status | Points |
|---|---:|
| Never | 0 |
| Former | 5 |
| Current | 20 |

## Synthetic Medical Rules

The following weights are educational examples only.

| External Condition | Points |
|---|---:|
| Asthma | 5 |
| Hypothyroidism | 5 |
| High Cholesterol | 5 |
| Hypertension | 10 |
| Type 2 Diabetes | 20 |

A condition is scored only when supported by available external medical
evidence.

Conditions not defined by this synthetic policy do not receive an
invented score. They instead generate a policy-review flag.

## Prescription Evidence

Prescription evidence is used primarily to corroborate medical evidence.

A medication does not independently add risk points when it corresponds
to a condition already scored by the medical rules.

An active external medication that was not declared by the applicant
generates an evidence-review flag.

This avoids double-counting the same underlying risk through both a
diagnosis and its treatment.

## Coverage-to-Income Rules

| Coverage / Income | Points |
|---|---:|
| <= 5 | 0 |
| > 5 and <= 8 | 5 |
| > 8 and <= 10 | 10 |
| > 10 | Review flag |

Coverage above ten times annual income is not assigned arbitrary
additional risk points. It requires policy review.

## Evidence Quality Rules

The following situations generate a mandatory review flag:

- identity evidence conflict
- financial evidence conflict
- missing identity evidence
- missing medical evidence
- missing prescription evidence
- missing financial evidence
- undeclared condition found in external medical evidence
- undeclared active medication found in prescription evidence
- unsupported medical condition not defined in the synthetic policy
- coverage-to-income ratio greater than 10

Evidence-quality review flags are separate from numerical risk points.

The system must not interpret missing evidence as favorable evidence.

## Risk Tiers

When no mandatory review flag exists:

| Score | Risk Tier |
|---|---|
| 0-14 | LOW |
| 15-29 | MODERATE |
| 30-49 | HIGH |
| 50+ | VERY_HIGH |

## Mandatory Review

If one or more mandatory review flags exist, the policy engine marks the
assessment as requiring review regardless of the numerical score.

The numerical score may still be calculated from evidence that is
available, but it must not be interpreted as a complete assessment.

Example:

A case may have a score of 20 but also have conflicting identity evidence.

The result is therefore:

score = 20
tier = MODERATE
requires_review = true

The downstream recommendation stage must consider the mandatory-review
flag.

## Separation of Responsibilities

The deterministic policy engine owns:

- feature calculation
- policy rule evaluation
- numerical score
- risk tier
- policy flags
- mandatory-review determination

The LLM owns only:

- explanation
- summarization
- contextual presentation of already calculated results

The LLM must not:

- change the score
- change the risk tier
- remove policy flags
- invent medical facts
- approve or deny an application
- create new underwriting rules

## Safety and Educational Scope

All applicant records, enrichment evidence, scoring weights, thresholds,
and policy rules in this project are synthetic.

The project demonstrates software architecture and multi-agent workflow
design rather than real-world insurance underwriting policy.