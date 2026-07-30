# Company Research Agent

An evidence-first, independently deployable research service for pharmaceutical companies. It
resolves an entity conservatively, plans research, searches Exa through MCP, extracts only
quote-supported claims, reflects on information quality, performs targeted follow-up rounds, and
returns a structured report with evidence and a complete state-transition trace.

The service is designed to fail closed: input fields are hints, model output is untrusted, URLs and
source metadata come only from the search provider, unsupported claims are discarded, conflicting
facts remain visible, and absent information is returned as `Unknown`.

## Quick start

Prerequisites: Python 3.11+ for local development, or Docker with Compose.

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# POSIX: source .venv/bin/activate
python -m pip install -e ".[dev]"
copy .env.example .env  # POSIX: cp .env.example .env
uvicorn app.main:app --reload
```

For a containerized start:

```bash
docker compose up --build
```

The API listens on `http://localhost:8000`. Liveness is at `/health/live`, configuration readiness
at `/health/ready`, and OpenAPI documentation at `/docs`.

## Configuration

All settings use the `CRA_` prefix and may be supplied through the environment or `.env`. Start from
`.env.example`; never commit real credentials.

Required provider settings:

| Variable | Purpose |
| --- | --- |
| `CRA_LLM_PROVIDER` | `openai` or `anthropic` compatible request format. |
| `CRA_LLM_API_KEY` | LLM credential. |
| `CRA_LLM_BASE_URL` | Explicit provider base URL; no provider URL is hardcoded. |
| `CRA_LLM_MODEL` | Provider model identifier. |
| `CRA_EXA_MCP_URL` | Exa MCP Streamable HTTP endpoint. |
| `CRA_EXA_API_KEY` | Optional Exa key, sent in the `x-api-key` header. |
| `CRA_SERVICE_API_KEY` | Required in production; callers send it as `X-API-Key`. |

The default Exa tool is `web_search_advanced_exa`. The client adds it to the endpoint's `tools`
parameter, negotiates MCP, verifies the exposed tool schema, requests bounded page text, and accepts
only structured/JSON results containing provider-owned HTTP(S) URLs. See the
[Exa MCP reference](https://exa.ai/docs/reference/exa-mcp) and
[MCP Python transport documentation](https://py.sdk.modelcontextprotocol.io/client/transports/).

Important controls include:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CRA_MAX_SEARCH_ROUNDS` | `3` | Maximum research/follow-up rounds. |
| `CRA_MAX_TOTAL_QUERIES` | `50` | Per-run logical query budget. |
| `CRA_SEARCH_MAX_CONCURRENCY` | `8` | Concurrent Exa calls per process. |
| `CRA_SEARCH_TIMEOUT_SECONDS` | `315` | Timeout for one search attempt. |
| `CRA_SEARCH_MAX_RETRIES` | `1` | Application retries after the initial attempt. |
| `CRA_MAX_CONCURRENT_RUNS` | `4` | Concurrent API research runs per process. |
| `CRA_RUN_TIMEOUT_SECONDS` | `900` | End-to-end deadline, including queue time. |
| `CRA_TOKEN_BUDGET` | `100000` | Per-run LLM token ceiling. |
| `CRA_COST_BUDGET_USD` | `25` | Per-run estimated LLM cost ceiling. |
| `CRA_EVIDENCE_LIMIT` | `500` | Maximum returned evidence records. |
| `CRA_EXA_MAX_TEXT_CHARACTERS` | `20000` | Maximum text requested and retained per result. |
| `CRA_EXTRACTION_MAX_PROMPT_BYTES` | `80000` | Aggregate source-payload limit per extraction call. |
| `CRA_SOURCE_TYPE_DOMAIN_RULES` | `{}` | JSON mapping of publisher domains to source classes. |

Set the LLM input/output prices for the configured model. Production configuration rejects a
positive cost budget with zero prices. For newer OpenAI-compatible APIs the default output field is
`max_completion_tokens`; set `CRA_LLM_OPENAI_MAX_TOKENS_FIELD=max_tokens` for older compatible
servers. Both provider URLs must use HTTPS in production. Development and test permit HTTP only on
loopback. URL credentials, fragments, secret query parameters, and LLM query strings are rejected.

Source authority is application-owned. Provider-supplied labels cannot mark a result as official or
reliable. Configure publisher rules explicitly, for example:

```dotenv
CRA_SOURCE_TYPE_DOMAIN_RULES={"acmepharma.com":"official","fda.gov":"regulatory","reuters.com":"news"}
```

Public suffixes such as `com` or `co.uk` are rejected because they would trust unrelated publishers.
Unclassified domains remain `other`; `.gov` and `.edu` receive conservative built-in regulatory and
academic classifications.

## API

Only `canonical_name` is required. `website`, `linkedin`, `country`, `industry`, and `summary` are
accepted strictly as unverified search/disambiguation hints.

```bash
curl -X POST http://localhost:8000/v1/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CRA_SERVICE_API_KEY" \
  -d '{
    "canonical_name": "Example Pharma",
    "website": "https://example-pharma.com",
    "country": "US",
    "industry": "Pharmaceuticals"
  }'
```

The response has four top-level parts:

```json
{
  "company": {
    "canonical_name": "Example Pharma",
    "identity_status": "partially_confirmed",
    "confidence": 0.7,
    "claim_ids": ["clm_..."]
  },
  "research": {
    "overview": {"findings": [{"statement": "...", "claim_ids": ["clm_..."]}]},
    "products": {"findings": []},
    "unknowns": {"findings": [{"statement": "Unknown: ...", "claim_ids": ["clm_..."]}]}
  },
  "evidence": {
    "sources": [{"source_id": "src_...", "url": "https://...", "content_hash": "cnt_..."}],
    "evidence": [{
      "claim_id": "clm_...",
      "claim": "Clinical stage",
      "value": "Phase 2",
      "status": "single_source",
      "source_id": "src_...",
      "supporting_quote": "The trial is Phase 2."
    }],
    "conflicts": []
  },
  "run_trace": {
    "run_id": "run_...",
    "rounds": [],
    "transitions": [],
    "stop_reason": "max_rounds_reached"
  }
}
```

Research prose contains claim IDs, never URLs. Evidence contains the retrieved source lineage and a
short exact quote. Raw page bodies are deliberately excluded from API serialization.

Expected operational errors use a stable envelope:

```json
{"error":{"code":"research_provider_error","message":"..."}}
```

Missing configuration returns 503, end-to-end timeout returns 504, invalid service authentication
returns 401, and permanent provider/authentication errors return 502 rather than silently producing a
successful empty report.

## Truth and evidence model

- `verified_fact`: the same controlled claim/value is supported by at least two reliable,
  registrably independent publisher domains.
- `single_source`: exactly one application-classified reliable publisher domain supports the claim.
- `inference`: reserved for future deterministic, typed derivation rules. Free-form model-generated
  inferences are rejected and are never promoted into evidence.
- `unknown`: available evidence is insufficient or only untrusted publishers agree.

Extraction responses cannot supply titles or URLs. For a direct fact, every referenced source must be
in the current allowlist, every short quote must exist exactly in normalized retrieved content, the
value must match at token boundaries, the quote must link the target company to the predicate, and it must directly
express a non-negated controlled predicate/value relationship rather than mere co-occurrence.
Validation happens before a claim record is created, so rejected model text is not echoed as an
`Unknown` fact. Scalar predicate disagreements become explicit conflicts; set-valued facts such as
different products coexist. Identity fields use a conservative allowlist of company-scoped claim
labels. Ambiguous, partially confirmed, or unconfirmed entity resolution suppresses non-identity
facts from the final bundle so same-name candidates cannot be blended; summaries, assessments, and
explicit `Unknown` items are then recomputed against that constrained bundle.
The company website is returned only when an exact company-domain claim corroborates the hinted
domain and a source classified as official was actually retrieved from that domain; an official
article on an acquirer or partner domain is not treated as the target's website.

Evidence is deduplicated first by canonical URL and then by normalized content hash across rounds.
A URL's first retrieved body is an immutable evidence snapshot; a later changed body cannot rewrite
its citation lineage or bridge unrelated sources. Exact-content duplicates under distinct URLs are
deduplicated deterministically, historical IDs are reconciled in the final trace, and research
statements are rebuilt from validated evidence rather than trusting model-authored prose.

Follow-up query labels and model-declared gap mappings are advisory only. A query must independently
target the resolved company and contain multiple terms from exactly one missing research dimension;
model output can prioritize dimensions, but executable text is synthesized from deterministic
pharmaceutical-domain templates. Any still-unaddressed content, quality, freshness, or diversity gap
also receives a safe template query before search execution.

## Workflow and architecture

The orchestration is an explicit legal-transition state machine:

```text
ResolveCompany -> PlanResearch -> GenerateSearchQueries -> ParallelSearch
  -> ProcessEvidence -> Summarize -> AssessInformation -> RefineCompanyIdentity
  -> GenerateFollowupQueries -> ParallelSearch ...
  -> RefineCompanyIdentity -> Finalize -> ReturnResult
```

Each transition records timestamps, duration, round, outcome, safe error details, and the run ID.
Search events additionally record query ID/text, results, retries, cache outcomes, duration, and run
correlation. Identity is refined before every sufficiency decision; unresolved rounds are assessed
against an identity-constrained view without destroying the accumulated internal search evidence.
Historical round assessments remain immutable, while `final_research_summary` and
`final_assessment` record the separately reconciled final view. Stop precedence is deterministic:
sufficient information, continuous search failure, budget, maximum rounds, then no new evidence.

Main modules:

```text
app/api/          FastAPI routes, dependency wiring, error envelopes
app/core/         state machine, budgets, stop policy, protocols, enums
app/resolver/     evidence-constrained company identity resolution
app/planner/      topic planning and non-duplicate query generation
app/search/       Exa MCP adapter and bounded parallel executor
app/evidence/     source registry, claim validation, status/conflict policy
app/reflection/   evidence summary, dimension scoring, follow-up planning
app/llm/          OpenAI/Anthropic-compatible structured-output adapters
app/prompts/      six versionable prompt templates
app/cache/        typed memory/SQLite TTL caches and split page cache
app/services/     per-run composition, orchestration, final report sanitizer
app/schemas/      strict Pydantic v2 API/domain contracts
```

Shared infrastructure is constructed once. Every request receives a new budget ledger, LLM client,
source registry, evidence processor, and orchestrator, preventing evidence or trace leakage between
runs. Business modules depend on protocols and injected implementations, so fake providers exercise
the same workflow in tests.

## Caching

Search metadata, page contents, and validated LLM outputs use separate TTL namespaces. Search cache
identity includes the endpoint, tool, parser version, page-text limit, and authority rules. Page bodies
are stored separately so a stale/missing body makes the whole typed search entry a miss. Extraction
cache payloads omit volatile retrieval timestamps and query IDs. Set `CRA_CACHE_ENABLED=false` to use
the no-op cache.

SQLite is the default single-host store at `CRA_CACHE_PATH`; Docker Compose persists `/data` in a
named volume.

## Testing and quality gates

All network-dependent code is tested with fakes or `httpx` transports; tests do not require provider
credentials.

```bash
pytest -q
ruff format --check app tests
ruff check app tests
mypy app
python -m compileall -q app tests
```

The suite covers strict schemas, company hint handling, identity provenance, explicit transitions,
parallelism, timeout/retry/partial failure, run log correlation, Exa MCP parsing and authentication,
URL/content deduplication, split caches, quote and semantic adversarial cases, conflicts, inference
rejection,
reflection, follow-up deduplication, stop precedence, budgets, API authentication, provider response
validation, and the full multi-round workflow.

## Deployment notes

The image runs as a non-root user, exposes a liveness health check, and persists only the configured
cache volume. Terminate TLS at a trusted ingress, keep the service key in a secret manager, restrict
outbound traffic to configured providers, and set current model pricing before production use. Scale
carefully: `CRA_MAX_CONCURRENT_RUNS` and SQLite are process-local; multiple replicas need a shared
rate-limit/queue policy and a shared cache implementation if cross-replica coordination is required.

## Known limitations

- `/health/ready` validates local dependency configuration; the first Exa call performs the live MCP
  handshake, authentication, and tool-schema check.
- The USD budget covers configured LLM token pricing. Exa cost metadata is not yet included, so use
  Exa-side spend controls in addition to this service's query cap.
- Publisher authority requires curated domain rules. The service intentionally does not infer that an
  arbitrary company or media domain is reliable.
- Controlled predicate normalization is conservative and extensible, not a full pharmaceutical
  ontology. Unrecognized paraphrases may remain separate instead of being merged. Direct-fact
  validation intentionally rejects complex, multi-statement, speculative, or subject-ambiguous
  excerpts; this trades recall for citation safety.
- MCP sessions are short-lived per provider call. This favors isolation and simple cancellation over
  connection reuse.
- The shared API key is service-level authentication, not tenant identity, per-tenant quota, or a
  distributed rate limiter.
- Search quality and freshness remain bounded by provider coverage and source text returned by Exa;
  the service does not bypass paywalls or execute page JavaScript.
