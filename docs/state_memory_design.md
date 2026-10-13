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

## Identifiers and Repeat Submissions

| Identifier | Assigned by | Purpose |
|---|---|---|
| application_id | Server, at intake | Identifies one request for coverage |
| case_id / trace_id | Server, at case creation | Identifies one underwriting run and its traces |
| ssn | Applicant | Strong identifier used to search external evidence |
| applicant_key | Server (hash of ssn) | Finds other cases for the same person |
| idempotency_key | Client (optional) | Makes a resubmitted request safe |

Intake rejects a client-supplied application_id.

External evidence providers are searched by SSN, not by any insurer
identifier. Name and date of birth alone are not unique, so medical,
prescription, and financial records found by SSN are only attributed to
the applicant when the record's date of birth also matches; otherwise the
evidence is marked as a conflict for human review. The identity provider
reports name, date of birth, and state matches itself.

## Tracing and PII

Every stage attempt writes one trace record. Together they answer, for
one case:

| Question | Trace fields |
|---|---|
| Which agent ran? | stage, component |
| What did it read, and from whom? | received, input_sources, tools_used |
| What did it produce and write to shared state? | produced, state_writes |
| When, and for how long? | started_at, finished_at, duration_ms, llm_latency_ms |
| What did it cost? | model, input_tokens, output_tokens, estimated_cost_usd |
| Did it fail or retry? | status, attempt, error_type, error_message |

input_sources names the producer of each received value: user_input for
the submission, otherwise the agent that wrote it. LLM usage is metered
per attempt, so an attempt that fails after a model call (for example on
unparseable output) is still costed.

The case repository is the only store that holds applicant PII, and it
needs encryption at rest and access control in production. Everything
else is minimized:

- LLM prompts never include SSN, name, or date of birth. No agent's
  reasoning depends on them.
- Trace payloads always mask SSN and name, and reduce date of birth to
  its year so age-driven decisions remain debuggable.
- Every string in a trace, including error messages, is scrubbed of
  SSN-shaped values as a backstop for fields added later.
- applicant_key is a plain SHA-256 of the SSN in this lab. Because SSNs
  have only about one billion values, production should use a keyed
  HMAC with the key held in a secrets manager.

Repeat submissions are handled in two ways:

- Same idempotency_key: the existing case is returned and no new
  processing starts. A unique index enforces one case per key.
- Same applicant with an open case (any status other than
  RECOMMENDATION_COMPLETED or PROCESSING_FAILED): a new case is
  processed, recorded in possible_duplicate_of, and routed to
  HUMAN_REVIEW so an underwriter sees both applications together.

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