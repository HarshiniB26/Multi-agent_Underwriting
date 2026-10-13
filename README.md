# Multi-agent life insurance underwriting

Lab pipeline with four standalone agents (intake, enrichment,
risk scoring, recommendation), a shared case store, structured
traces, and retry-then-escalate failure handling.

Underwriting decisions come from deterministic policy code
(`src/underwriting/risk/policy.py`,
`src/underwriting/recommendation/policy.py`). The LLM only reviews
and summarizes; it never sets a score, tier, or decision.

## Setup

```bash
pip install -e ".[dev]"
```

The devcontainer runs this automatically. Python 3.12+.

## Run tests

```bash
ruff check src tests
pytest -q --cov
```

This is what CI runs (`.github/workflows/ci.yml`). Coverage must stay
at 100% (`fail_under` in `pyproject.toml`). Ollama integration tests
are skipped when no Ollama server is reachable.

## How a case flows

1. **Intake** validates the submission. The server assigns
   `application_id`; a client-supplied one is rejected.
2. **Enrichment** looks up identity, medical, prescription, and
   financial records by SSN. A record whose date of birth does not
   match the applicant is a conflict, not the applicant's evidence.
3. **Risk scoring** applies `docs/synthetic_underwriting_policy.md`.
   Mismatches (undeclared conditions or medications, income off by
   more than 5%, identity mismatch, missing records) raise review
   flags instead of adding points.
4. **Recommendation** applies `docs/synthetic_recommendation_policy.md`:
   any review flag means REFER, then very high risk means DENY,
   otherwise APPROVE.

Final status is `recommendation_completed`, `human_review`, or
`processing_failed`.

### Repeat submissions

- Pass `idempotency_key` to `start_case` to make resubmits safe: the
  same key returns the existing case.
- A new application from someone who already has an open case is
  processed, recorded in `possible_duplicate_of`, and sent to human
  review.
- Coverage from the applicant's earlier approved cases is added to the
  new request before the coverage-to-income rules apply, so a second
  application can push the total over 10x income and trigger review.

## State and memory

Documented in `docs/state_memory_design.md`. The pipeline uses a
**case-scoped shared state store** (SQLite behind a repository).
Agents do not share chat memory; the orchestrator passes each agent
only the fields it needs.

## Tracing

Every stage attempt writes one trace record with: which agent ran
(`component`), what it read and where that came from (`received`,
`input_sources`, `tools_used`), what it produced and wrote to shared
state (`produced`, `state_writes`), timing (`started_at`,
`finished_at`, `duration_ms`, `llm_latency_ms`), token usage and
equivalent cost, and any error or retry (`status`, `attempt`,
`error_type`, `error_message`).

## PII handling

- LLM prompts never include SSN, name, or date of birth.
- Traces always mask SSN and names, keep only the birth year, and
  scrub SSN-shaped text from every string, including error messages.
- The case database is the only store holding PII. In production it
  needs encryption at rest and access control, and `applicant_key`
  should be a keyed HMAC rather than a plain SHA-256 of the SSN.

All data in `data/` is synthetic. SSNs use the never-issued
`9NN-00-NNNN` pattern.

## Capture assignment evidence (3 cases + timeout)

```bash
python scripts/capture_lab_evidence.py
python scripts/build_deliverable_pdf.py
```

The first writes traces and test output to `artifacts/lab_evidence/`
using a deterministic fake LLM. The second packs those, the design
docs, and key source files into `deliverable/assignment_deliverable.pdf`.

## Live Ollama run

```bash
python scripts/run_underwriting.py --cases case_01 case_02 case_03
```

Prints each case's result and a one-line summary per stage. Add
`--trace` to also print every stage's full (PII-masked) inputs and
outputs. Traces are saved to the database either way.

## Cost / latency write-up

See `docs/cost_latency_analysis.md`.
