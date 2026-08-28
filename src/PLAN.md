## Plan: AION Cost-First Rebuild + Mamba-First LLM Lab

Build AION in `src` with two parallel goals:
- A practical conversational product that runs well on weak machines.
- A research lab where Mamba/SSM is the primary from-scratch model family.

The strategy is local-first, reversible decisions, and strict cost gates.

Project independence rule:
- No external model API dependency is allowed in any mode.
- If this changes in the future, it requires an explicit planning decision and plan update.

### Exploration doctrine

1. No sacred components: every layer can be removed, replaced, or merged.
2. Prefer high-information experiments over safe incrementalism.
3. Kill assumptions quickly, then scale only what proves itself.
4. Innovation is required, but every innovation must have rollback.

### North-star constraints

1. Zero dependency on external model APIs.
2. Strong behavior on low-power hardware before scaling up.
3. Same product contract regardless of model provider.
4. Every architecture decision must be measurable and reversible.

### Hardware and budget envelopes (hard defaults)

Weak laptop profile (WL-1):
- 4 cores / 8 threads
- 8 GB RAM
- integrated GPU only

Mid desktop profile (MD-1):
- 8+ cores
- 16-32 GB RAM
- optional entry GPU

Nano mode hard targets (WL-1):
1. Idle RAM <= 1.2 GB (stretch: <= 800 MB)
2. Idle CPU <= 8 percent (stretch: <= 4 percent)
3. Cold start <= 8 s (stretch: <= 5 s)
4. p95 first token latency <= 1.8 s on deterministic/local routes
5. External model API calls = 0 in all environments
6. Local fallback success ratio >= 99 percent of turns

### Operating modes (first-class)

1. Nano mode (default)
   - Single backend service + lightweight web UI.
   - SQLite persistence.
   - Optional tiny local model or no-model deterministic path.
   - Runs via Docker Compose (single-container or minimal multi-container).
2. Standard mode
   - Frontend + BFF + backend + Postgres.
   - SSE end-to-end.
   - Better separation and scaling path.
3. Lab mode
   - Dataset, tokenizer, training, eval jobs.
   - CPU-first workflows; optional GPU profile for burst runs.

### Architecture (target, with cost-first rollout)

1. `frontend` (Vue 3 + Vite + TS)
   - Chat UX and session lifecycle.
   - Talks to BFF in Standard mode.
   - Can talk directly to backend in Nano mode.
2. `bff` (Node 22 + Fastify + TS)
   - Contract normalization and SSE relay.
   - Optional in early Nano iterations.
3. `backend` (Python 3.12 + FastAPI)
   - Chat orchestration, routing, inference endpoints.
   - Local-provider adapter with deterministic fallbacks.
4. `llm_lab` (Python + PyTorch)
   - Data curation, tokenizer, Mamba/SSM pretraining, instruction tuning, evaluation.
   - Compact transformer baseline for fair comparison.
5. `infra` (Compose + env contracts)
   - Docker Compose is the single entry point for all modes.
   - Compose profiles: nano (default), standard, lab-cpu, lab-gpu.
   - All services run in containers; no host-install requirement beyond Docker.

### Phase plan

Phase 0 - Cost envelope, experiments, and decision gates (blocking)
1. Define hardware profiles:
   - Weak laptop profile.
   - Mid desktop profile.
   - Burst cloud profile (optional).
2. Define operating budgets:
   - RAM budget for app stack.
   - Idle CPU budget.
   - Max p95 latency target.
   - Monthly compute/storage budget with no external model API usage.
3. Define quality baselines:
   - Fixed prompt suite.
   - Task-level score rubric.
   - Failure taxonomy.
4. Define go/no-go gates:
   - Ship only if quality improves beyond threshold and cost does not exceed threshold.
   - Reject experiments that are not reversible.

Phase 0.5 - Assumption-kill sprint (10 days, mandatory)
1. T1: Measurement harness reliability
   - Hypothesis: quality/cost/latency metrics are stable enough for decisions.
   - Pass: metric variance <= 5 percent over 3 repeated runs.
2. T2: Cascade economics
   - Hypothesis: deterministic -> local tiny -> local advanced beats deterministic-only on quality while staying within Nano resource budgets.
   - Pass: >= 95 percent turns resolved by deterministic + tiny local path and quality gain >= 8 percent.
3. T3: Retrieval + semantic cache + context budget
   - Hypothesis: token and latency reductions with minimal quality loss.
   - Pass: >= 25 percent token reduction and quality drop <= 5 percent.
4. T4: Mamba CPU stability smoke
   - Hypothesis: small Mamba runs are stable and resumable under CPU-first budget.
   - Pass: stable loss decline and no recurring NaN/divergence in 3 runs.
5. T5: Equal-budget Mamba vs compact transformer
   - Hypothesis: Mamba wins at least one efficiency frontier (quality, latency, or RAM).
   - Pass: quality parity within 3 percent plus >= 20 percent RAM gain, or equivalent efficiency win.

If any test fails, freeze the affected branch and shift effort to the next highest information-per-cost branch.

Phase 1 - Monorepo foundation and mode topology (depends on Phase 0)
1. Organize `src` into `frontend/`, `bff/`, `backend/`, `llm_lab/`, `infra/`.
2. Define shared env contract and per-service env files.
3. Add mode switches:
   - `AION_MODE=nano|standard|lab`.
   - `AION_PROVIDER=deterministic|local-mamba|local-transformer`.
4. Create root compose setup with profiles for all modes (nano, standard, lab).
   - `docker compose up` starts Nano mode by default.
   - `docker compose --profile standard up` adds BFF + Postgres.
   - `docker compose --profile lab-cpu up` adds training jobs.

Phase 2 - Nano app MVP (depends on Phase 1)
1. Backend endpoints:
   - `POST /chat`
   - `POST /chat/stream` (SSE)
   - `GET /health`
   - `GET /ready`
2. Minimal frontend shell with streaming and error handling.
3. SQLite conversation persistence.
4. Deterministic + tiny model cascade:
   - Rule/template path first.
   - Tiny local model second.
   - Local advanced model fallback only when necessary.
5. Enforce guardrails:
   - Hard token budget per request.
   - Adaptive context compaction.
   - Local route circuit breaker with deterministic degradation mode.

Phase 3 - Standard app track (depends on Phase 2)
1. Add BFF with schema validation and SSE passthrough.
2. Switch persistence to PostgreSQL while preserving storage abstraction.
3. Validate unchanged frontend contract across providers.

Phase 4 - LLM lab foundations (parallel with Phase 3 after Phase 1)
1. Dataset pipeline:
   - raw ingestion
   - cleaning/dedup/filtering
   - train/val split with manifests
2. Tokenizer pipeline:
   - train and version tokenizer artifacts
3. Training infra:
   - reproducible configs
   - checkpoint/resume
   - scheduler, grad clipping, mixed precision where useful
4. Evaluation infra:
   - perplexity and token-level metrics
   - prompt/task harness with versioned reports

Phase 5 - Mamba/SSM from scratch (depends on Phase 4)
1. Implement Mamba/SSM blocks and LM stack.
2. Run small-scale pretraining with frequent checkpoints.
3. Perform stability sweeps (sequence length, state size, batch settings).
4. Export local inference bundle.

Phase 6 - Transformer baseline and fair comparison (depends on Phase 4, parallel with late Phase 5)
1. Implement compact transformer baseline (RoPE + GQA when feasible).
2. Train under same data and budget envelope as Mamba.
3. Compare quality/latency/memory under equal constraints.

Phase 6.5 - Model portfolio decision gate (mandatory)
1. If Mamba does not win a clear efficiency frontier, move it to research-only.
2. If transformer dominates under equal budget, make transformer the default local model.
3. If neither local path meets Nano constraints, default product provider remains deterministic+retrieval-only until local models improve.

Phase 7 - Instruction tuning and serving integration (depends on Phases 3, 5, and 6)
1. Build instruction format (`instruction`, `context`, `response`).
2. Tune Mamba and optional transformer baseline.
3. Add backend router that keeps app contract stable:
   - `provider=deterministic`
   - `provider=local-mamba`
   - `provider=local-transformer`

Phase 8 - Observability, comparison, and docs (depends on Phases 3 and 5-7)
1. Add experiment metadata (run id, params, metrics, checkpoints).
2. Add side-by-side evaluation workflow (mamba vs transformer vs deterministic-retrieval).
3. Write practical docs:
   - internals notes for Mamba/SSM in this repo
   - how to run CPU and GPU jobs
   - how to switch local provider modes safely

### Unusual and non-traditional routes to test

1. Retrieval-first responses with generation only for missing spans.
2. Aggressive semantic caching of model outputs.
3. Prompt budget enforcement and adaptive context truncation.
4. Early-exit cascade routing based on confidence thresholds.
5. Optional byte-level tokenizer experiment for simpler pipelines.
6. Distill task-specific micro-models rather than one larger general model.
7. Negative caching for expensive low-confidence query classes.
8. Thermal-aware runtime mode that shortens outputs under throttling.
9. Lazy-load second local model tier only when route confidence requires it.

### Radical architecture tracks (deliberately non-safe)

Track R1 - Deterministic-first product
1. Product works fully without model generation.
2. Model generation is an augmentation path only.

Track R2 - Offline-first edge appliance
1. Local retrieval + local tiny model only.
2. External provider is disabled by policy.

Track R3 - Split-brain runtime
1. Cheap brain handles 95 percent of traffic under strict budget.
2. Expensive brain is isolated, rate-limited, and fully audited.

Only one radical track is promoted at a time; others remain behind flags.

### Decision scoreboard (used for every major choice)

Each decision is scored from 1 to 5 on:
1. Monthly cost impact.
2. Idle resource impact.
3. Complexity tax.
4. Reversibility.
5. Quality gain.
6. Learning value.

Ship only when total score beats current baseline and rollback is clear.

### Stop rules and rollback discipline

1. Stop experimenting on any branch that misses budget or latency targets for 2 consecutive check-ins.
2. Every risky feature must have a single config flag to disable it.
3. Rollback target state is always deterministic + retrieval-only mode.
4. Rollback execution target: under 30 minutes, documented before promotion.
5. Any non-local model API integration attempt is blocked unless the policy is explicitly revised.

### LLM learning scope (explicit)

Scale positioning: AION is a **Small Language Model (SLM)** lab. "LLM" denotes the general
language-modeling field, but the models trained here are small (a few million up to a few hundred
million parameters; largest config is a 235M Transformer). SLMs match the cost-first,
weak-hardware, local-first constraints: cheap to train, fast/CPU-friendly to run, private (offline),
and quick to fine-tune. Frontier-scale **LLMs** (tens of billions+ parameters) trade efficiency and
controllability for broad general reasoning at high compute cost and are out of scope.

Included:
- Mamba/SSM internals from scratch.
- Compact transformer baseline for controlled comparisons.
- Training loop mechanics (optimizer, scheduler, clipping, checkpoints).
- Tokenization and data curation.
- Instruction tuning and local inference integration.
- Efficiency analysis (quality vs memory vs latency).

Excluded initially:
- Frontier-scale from-scratch training.
- Full RLHF/DPO pipelines.
- Multi-node distributed orchestration.

### Repository surfaces to create/use

- `src/frontend`
- `src/bff`
- `src/backend`
- `src/llm_lab`
- `src/infra`
- `src/docker-compose.yml`
- `src/README.md`

### Verification strategy

1. Nano mode E2E:
   - Verify frontend/backend chat flow with SQLite.
   - Verify SSE streaming and cancellation.
   - Verify deterministic fallback behavior.
2. Standard mode E2E:
   - Verify frontend -> BFF -> backend -> Postgres flow.
   - Verify identical response contract to Nano mode.
3. Lab training:
   - Run tokenizer training and validate artifacts.
   - Run Mamba smoke pretraining and validate loss trend.
   - Run transformer smoke pretraining and validate loss trend.
   - Run short instruction tuning and validate chat behavior.
4. Integration:
   - Switch provider among deterministic, local-mamba, and local-transformer without frontend changes.
   - Validate conversation persistence across provider modes.
5. Independence control:
   - Verify no external model API calls are possible in runtime paths.
   - Verify policy checks reject non-local provider configuration.
6. Weak-hardware soak:
   - 24h WL-1 run with memory growth <= 15 percent and crash-free sessions >= 99.5 percent.

### Progress tracking system

Goal:
1. Keep tracking lightweight and personal.
2. Capture only what was done and what should happen next.

Simple tracking loop (no fixed schedule):
1. When you finish a work session, write a short session note.
2. When you return, read the last note and continue from the Next list.

Session note format:
1. Date
2. Done:
   - what changed
   - key decision made
3. Next:
   - 1 to 3 concrete next actions
4. Parking lot:
   - ideas to explore later (no pressure)
5. Blockers (optional):
   - what is stuck and why

Optional personal views:
1. Now: what is active right now.
2. Next up: what you will likely do in the next session.
3. Later: ideas worth keeping but not acting on yet.

Rule of thumb:
1. If tracking feels heavy, reduce it.
2. If restart feels confusing, add one more line to Next.
3. Progress quality matters more than process consistency.

### Confirmed decisions

1. Base directory: `src`.
2. Streaming protocol: SSE end-to-end.
3. Primary from-scratch model family: Mamba/SSM.
4. Cost-first development starts in Nano mode and scales to Standard mode.
5. Exploration-first process: assumption-kill sprint before major buildout.
6. External model API dependency is prohibited unless explicitly re-decided.
