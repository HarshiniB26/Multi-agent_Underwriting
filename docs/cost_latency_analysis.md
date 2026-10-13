# Cost and Latency Analysis

This analysis uses traces from three mock applications
(`case_01`, `case_02`, `case_03`) run through the full
intake → enrichment → risk scoring → recommendation
pipeline.

Local Ollama inference has a cash cost of **$0.00**.
Traces still record an **equivalent commercial cost** using
published gpt-4o-mini-class rates:

- $0.15 / 1M input tokens
- $0.60 / 1M output tokens

Those rates are implemented in
`underwriting.observability.cost.estimate_llm_cost_usd`.

The captured lab evidence uses a deterministic fake LLM so
the packet can be regenerated without a live model. Token
counts are therefore representative of a small structured
JSON call (420 input / 90 output per agent), not a long
chat session. Wall-clock duration in that mode is
sub-second. A live Ollama run on `qwen3:8b` is typically
**8–40 seconds per case** depending on host load; that is
the latency figure used for the real-time vs batch judgment.

## Per-case totals from captured traces

| Case | Stages | Input tokens | Output tokens | Equivalent USD | Stage duration (fake LLM) |
|---|---|---|---|---|---|
| case_01 | 4 | 1680 | 360 | $0.000468 | < 1 s |
| case_02 | 4 | 1680 | 360 | $0.000468 | < 1 s |
| case_03 | 4 | 1680 | 360 | $0.000468 | < 1 s (routes to human_review) |

Formula used for each stage attempt (failed attempts are
costed too, for every model call that returned before the
failure):

```
cost = (input_tokens / 1e6) * 0.15 + (output_tokens / 1e6) * 0.60
```

Four successful calls × (420, 90) tokens:

```
4 * ((420 / 1e6) * 0.15 + (90 / 1e6) * 0.60) = 0.000468
```

## Real-time vs overnight batch

**Not acceptable as a hard real-time / interactive
underwriting API** if the product needs a sub-second or
even 2-second response. Four sequential LLM calls add
serial latency. On a local 8B model that commonly lands
in the **tens of seconds** per application. That is fine
for a human underwriter waiting on a workstation, but not
for a quote-in-the-browser experience.

**Acceptable for overnight / batch underwriting.** Cost is
negligible even at commercial rates (well under a tenth of
a cent per case). Overnight jobs routinely spend minutes
per file when humans or third-party data vendors are in
the loop. Retry-then-escalate also fits batch: a timed-out
risk score becomes `processing_failed` for a human queue
instead of a silent wrong decision.

**Borderline for same-day assisted underwriting.** If a
human reviewer already waits 5–15 minutes for MIB /
prescription / financial hits, adding ~30 seconds of model
time is reasonable. The system should still show progress
and never return an incomplete recommendation as if it
were final.

## Failure-path cost note

When risk scoring times out three times, the case is
flagged `processing_failed`. Intake and enrichment tokens
are still spent; risk scoring adds duration but $0 extra
because those attempts never returned a completion. That
is cheaper than inventing a score.
