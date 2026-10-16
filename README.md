# AI-Powered Agentic Framework for Financial Fraud Detection and Investigation



Detection models answer *"is this suspicious?"*. This system adds the layer investigators actually need: *"why, what else is connected, and what should happen next?"* A flagged transaction is investigated by four agents that correlate classifier, behavioural and graph evidence, and an LLM turns that evidence into a report in which every claim is mechanically checked against the evidence it cites. A human reviewer always makes the final decision.

Section numbers (§) below refer to the project design document.

## Quick start

```bash
pip install -r requirements.txt
./run_all.sh                      # synthetic data -> models -> experiments -> tests (about 3 minutes)
uvicorn backend.app:app           # then open http://127.0.0.1:8000
```

In the dashboard, click **Replay 20 transactions** to stream held-out transactions through the system, then open an alert. Click any evidence chip (E12) in the report to jump to the evidence row it cites.

Everything runs offline. Without an API key the Investigation Agent uses a deterministic template narrator that produces the same JSON report schema, so the pipeline, validator and evaluation all work. To use Claude:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export NARRATOR=claude            # or "auto": Claude if a key is set, template otherwise
```

With Docker: `docker build -t fraudinv . && docker run -p 8000:8000 fraudinv`.

## Architecture

```
Transaction ─► features (past-only) ─► XGBoost ──────┐
                                    ─► Isolation Forest ├─► deterministic fusion ─► alert?
                                    ─► graph score ─────┘                              │
                                                                                       ▼
 triage ─► evidence correlation ─► supervisor gate ─┬─► investigation (LLM) ─► supervisor final ─► human review
              ▲                                     │
              └──── widen k-hop 2→3, once ◄─────────┤ medium confidence
                                                    └─► low / conflicting ─► human review, raw evidence, no narrative
```

The agent graph in `docs/agent_graph.mmd` is exported from the compiled LangGraph itself, so the diagram in the black book cannot drift from the code.

| Layer (§6) | Module | Choice and reason |
|---|---|---|
| 1 Ingestion | `backend/app.py`, `scripts/replay.py` | FastAPI + controlled-rate replay ("near-real-time, simulated", §19) |
| 2 Processing | `preprocessing/features.py` | 25 features from §11, computed only from history before each transaction |
| 3 Detection | `models/train.py` | XGBoost, `scale_pos_weight` + threshold tuning, no SMOTE (§36) |
| 4 Behaviour | `models/detectors.py` | Isolation Forest on per-account deviation features, trained without labels |
| 5 Graph | `graph/engine.py` | NetworkX: k-hop, fan-in/out, cycles, scatter-gather, shared devices |
| 6 Explainability | `explainability/shap_explain.py` | TreeSHAP: exact Shapley values for tree models |
| Fusion (§31) | `fusion/fusion.py` | Learned logistic regression over the 3 signals, or 2-of-3 rule |
| 7 Agents | `agents/pipeline.py` | Triage, Evidence Correlation, Investigation, Supervisor (LangGraph) |
| Grounding (§28) | `agents/validator.py` | Rejects uncited claims, unknown evidence IDs, and numbers not in the cited evidence |
| Storage (§10) | `backend/db.py` | All §10 tables; SQLite by default, PostgreSQL via `DATABASE_URL` |

### The four agents

| Agent | Sees | Decides | Failure behaviour |
|---|---|---|---|
| Triage | scores, features | priority P1–P3, whether to investigate | — |
| Evidence Correlation | graph engine, SHAP, features | builds the evidence bundle (≈35 items, each with an ID) | graph failure: continues without graph evidence and marks it degraded |
| Investigation | **only** the evidence bundle | the narrative, nothing else | API error or failed validation: retries twice with the validator's errors, then gives up |
| Supervisor | fusion output, evidence counts, report | routing; overrides the LLM's recommended action | any failure routes to a human with raw evidence; nothing is silently dropped |

Confidence always comes from the deterministic fusion, never from the LLM's own self-report. The allowed actions are `escalate_for_review`, `request_more_evidence` and `monitor`; there is no code path that blocks a transaction or closes a case.

## Using the real datasets

The synthetic generator is for development only. **Results for the black book come from the real datasets**, downloaded from Kaggle into `datasets/data/raw/`:

- PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1 → `PS_20174392719_1491204439457_log.csv`
- IBM AML: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml → `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` only

```bash
./run_real.sh paysim     # results -> reports_paysim/
./run_real.sh ibm        # results -> reports_ibm/
```

Each run replaces the data, models and dashboard contents; the header shows which dataset is live. What the loaders do:

- Keep **every** fraud or laundering row and sample legitimate activity (`--frac`): 10% of PaySim rows, 20% of IBM accounts. IBM is sampled by account, so each kept account keeps its full history and laundering patterns stay complete.
- Convert IBM's mixed currencies to US dollars using exchange rates implied by its own cross-currency transactions.
- Attach IBM typology labels (fan-out, cycle, scatter-gather and so on) from `HI-Small_Patterns.txt`.

Known properties of the real data, worth stating in the black book:

- **PaySim customers almost never repeat**, so per-account behaviour features and the graph add little; detection there rests on the classifier. That is the reason the design uses IBM AML for the graph layer.
- Neither dataset has device, IP or account-age data, so those features are constant.
- PaySim's balance columns leak the label (§35) and are excluded unless `USE_BALANCE_FEATURES = True` in `configs/settings.py`. Reporting both runs is a good ablation.

## Experiments and where the outputs go

| Experiment (§37) | Command | Output in `reports/` |
|---|---|---|
| 1–4 Baselines, feature engineering, behaviour, graph, fusion | `python -m models.train` | `baseline_comparison.md`, `recall_by_pattern.csv` |
| 6 Report quality + ablation | `python -m evaluation.evaluate_reports` | `report_eval_summary.md`, `rubric_sheet.csv`, `rubric_key.csv` |
| 7 Latency at scale | `python -m evaluation.latency --tiers 100,1000,10000` | `latency.md`, `latency.png` |

`rubric_sheet.csv` is the blind sheet for §32: reports from all ablation variants are shuffled and the variant is hidden. Give it to someone who did not write the report code, have them score correctness, completeness, grounding, readability and usefulness from 1 to 5, then join their scores to `rubric_key.csv`.

## Results on synthetic data

These numbers show the pipeline works end to end. **They are not findings**: the synthetic data is easier than real fraud (ROC-AUC is close to 1 for most models), and nearly every alert routes to "escalate". Re-run on PaySim and IBM AML before writing the results chapter.

Detection, chronological test split (17,733 transactions, 100 frauds):

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Rule-based threshold | 0.688 | 0.640 | 0.663 | — |
| Logistic Regression | 0.966 | 0.570 | 0.717 | 0.805 |
| XGBoost, raw features | 0.173 | 0.130 | 0.149 | 0.142 |
| XGBoost, engineered features | 0.974 | 0.760 | 0.854 | 0.940 |
| + behaviour + graph, learned fusion | 0.986 | 0.730 | 0.839 | 0.955 |

Report quality and ablation (60 alerted test cases, template narrator):

| Variant | Valid reports | Claims grounded | Claims per report | Evidence coverage |
|---|---|---|---|---|
| Full system | 100% | 100% | 11.4 | 1.00 |
| Without graph agent | 100% | 100% | 9.2 | 1.00 |
| Without SHAP | 100% | 100% | 8.4 | 1.00 |
| Without evidence correlation | 100% | 100% | 4.0 | 0.26 |

Removing correlation drops evidence coverage from 1.00 to 0.26, which directly answers "why does this component exist". Without the graph, recall into the escalate route falls from 1.00 to 0.91.

Latency (template narrator, single process): detection P50 121 ms, P95 164 ms, P99 236 ms; investigation P50 72 ms. The detection P95 meets the 200 ms target from §22 but P99 does not. With Claude, investigation time will be dominated by the API call (expect a few seconds), which is the honest bottleneck to report.

## Checking your own transactions (CSV upload)

On the dashboard, **Upload transactions** opens a page where you can upload a CSV. Every row is treated as a new payment: scored, alerted and investigated exactly like live traffic, in time order so patterns such as fan-in can build up. **Download the sample file** there for a ready-made demo built from the live system's own accounts (normal payments, a payment to a known fraud account, a fan-in build-up, a night-time burst and a shared-device ring), with an `expected` column so the results page can show whether each result matched. Up to 300 rows; PDF is not supported.

## "How often is this right?" (calibration)

`python -m evaluation.calibrate` measures, on the held-out test period, how often alerts at each risk level were real fraud (adjusted for the sampling of normal activity), and saves it to `artifacts/calibration.json`. The case page uses it to say, for example, "In testing, about 7 in 10 alerts at this risk level were real fraud." Run it after every retrain.

## API (§17)

| Method | Path | Purpose |
|---|---|---|
| POST | `/transactions` | ingest and score; investigation runs in the background if it alerts |
| POST | `/predict/{tx_id}` | detection output and features for a scored transaction |
| POST | `/investigate/{tx_id}` | run the agent pipeline now, even without an alert |
| GET | `/alerts`, `/alerts/{id}` | alert queue, highest fused risk first |
| GET | `/graph/{account_id}?k=2` | k-hop subgraph |
| GET | `/explanation/{tx_id}` | SHAP breakdown |
| GET | `/report/{case_id}` | report, evidence, validation result and agent trace |
| POST | `/review/{case_id}` | `approve`, `escalate` or `reject`; approving marks the receiver as a known perpetrator |
| POST | `/demo/replay?n=20` | demo helper: replays held-out transactions |
| POST | `/upload` | check every row of a CSV (sent as text) through the full pipeline |
| GET | `/upload/sample` | download a demo CSV built from the live system |
| GET | `/entity/{id}` | an account's or device's connections (used by the network view) |

Interactive docs are at `/docs` once the server is running.

## Tests

`pytest -q` runs 13 tests, including the ones an examiner is most likely to probe:

- the online feature path produces exactly the same values as the batch training path (no train/serve skew);
- the validator rejects invented numbers, non-existent evidence IDs, uncited claims and "block account";
- conflicting signals skip the LLM entirely;
- medium confidence widens the graph search exactly once, then produces a partial report;
- an LLM that errors or hallucinates is retried, then the case goes to a human with raw evidence;
- dataset ground-truth labels never appear in the evidence the agents see.

## Repository layout

```
agents/          evidence bundle, narrators (Claude + template), validator, LangGraph pipeline, text renderer
backend/         FastAPI app, SQLAlchemy schema
configs/         every threshold in one file (settings.py)
datasets/        synthetic generator, PaySim / IBM AML loaders
docs/            Mermaid agent graph and ER diagram (generated)
evaluation/      report-quality + ablation, latency benchmark
explainability/  SHAP wrapper and feature descriptions
frontend/        single-file analyst dashboard (Cytoscape.js for the graph)
fusion/          learned and rule-based evidence fusion
graph/           graph construction, typology detection, graph score
models/          training script and inference bundle
reports/         experiment outputs
scripts/         replay into the API, diagram export
tests/           pytest suite
```

## Known limitations and next steps

- The graph suspicion score uses documented heuristic weights for its individual signals (`GraphEngine.score`); the learned fusion then re-weights the combined score against labels. Replacing it with GraphSAGE is the §12 SHOULD BUILD.
- Graph features at evaluation time use a graph built from the training period plus everything replayed so far; they never see the future.
- The in-memory history grows with every transaction; a production version would use a feature store (§22).
- The Claude narrator is implemented but has not yet been evaluated at scale; run Experiment 6 with `NARRATOR=claude` on around 20 cases and compare its validation pass rate and retries against the template.
