# State and Memory Design

## Decision

The underwriting pipeline uses a case-scoped shared state model with
controlled agent access and deterministic orchestration.

Each underwriting application is assigned a unique case identifier.
Information produced during processing is persisted as part of that case
rather than being stored in global application memory or relying on an
LLM to remember previous interactions.

## Rationale

Insurance underwriting is a multi-stage process that may span multiple
agent executions and potentially multiple sessions. Intake information,
enrichment results, risk assessments, and recommendations therefore need
to remain available throughout the lifecycle of a case.

A shared case state provides a consistent source of truth while explicit
agent contracts limit which portions of the state each component may
produce or consume.

This approach was selected over pure message passing because case data may
need to be retrieved after the original processing step has completed.
However, agents do not receive unrestricted access to mutable global state.
The orchestrator controls execution and provides each agent only the data
required for its responsibility.

## State Ownership

| Component | Reads | Produces |
|---|---|---|
| Intake Agent | Raw application | Validated application |
| Enrichment Agent | Validated application | Enrichment evidence |
| Risk Scoring Agent | Application, enrichment evidence | Risk assessment |
| Recommendation Agent | Application, enrichment evidence, risk assessment | Recommendation |

## Short-Term Memory

Short-term memory exists only during an individual agent invocation.

Examples include:

- the prompt sent to the model
- relevant case information supplied to that invocation
- temporary tool results
- intermediate reasoning context

Short-term memory is never shared globally between cases and is discarded
after the invocation completes.

## Persistent Case State

Information required across workflow stages is stored in the case record.

The case state contains:

- case metadata
- validated application
- enrichment evidence
- risk assessment
- recommendation
- workflow status
- failure information

For the lab implementation, SQLite will provide durable local persistence
behind a repository abstraction. A production implementation could replace
this adapter with an enterprise transactional database without changing
agent business contracts.

## Case Isolation

Every application receives a unique case ID.

Agents operate only on state associated with the current case. No global
conversation history or shared applicant context is used.

This prevents information from one applicant from leaking into another
applicant's processing context.

Case isolation will also be verified through automated tests.

## Agent Access Boundaries

Agents have explicit responsibilities:

- Intake owns creation of the validated application.
- Enrichment owns external enrichment evidence.
- Risk Scoring owns the risk assessment.
- Recommendation owns the final recommendation.

Agents must not overwrite data owned by another stage.

## LLM Memory

LLM calls are stateless.

The system never assumes that a model remembers previous calls. Required
context is explicitly constructed from the current case and supplied to
each model invocation.

Persistent business state belongs in the case repository, not in model
conversation history.

## Production Evolution

The lab uses SQLite for lightweight durable persistence.

The repository interface separates domain logic from storage technology.
A production deployment could replace SQLite with a managed transactional
database and introduce asynchronous workers or event-driven processing
without redesigning the core case model.

## Key Design Goals

The state architecture prioritizes:

1. Case isolation
2. Auditability
3. Explicit ownership
4. Persistence
5. Testability
6. Replaceable infrastructure
7. Prevention of cross-case context leakage