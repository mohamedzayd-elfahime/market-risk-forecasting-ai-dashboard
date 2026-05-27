# Market Risk Forecasting AI Dashboard

End-to-end market risk forecasting and AI-assisted risk analysis platform, applied to the Moroccan MASI stock index.

This project combines quantitative risk research, production-oriented machine learning pipelines, VaR/Expected Shortfall backtesting, volatility regime analysis, a FastAPI dashboard, and a controlled local AI assistant for interpreting risk outputs.

It is designed as a research-to-application system: the repository includes the runnable application, trained artifacts for local inference, runtime dashboard data, API services, ML workflows, tests, and public architecture documentation.

> This project is for research, education, and risk interpretation. It does not provide financial advice.

## Highlights

- FastAPI backend serving a static web dashboard and REST API.
- Forecasting workflow for returns, VaR, Expected Shortfall, volatility, and HMM regimes.
- Statistical backtesting with Kupiec, Christoffersen, violation analysis, and ES diagnostics.
- Economic backtest outputs and model validation reports.
- Trained model artifacts included for local demo and inference without retraining.
- Local AI assistant powered by Ollama, embedding-based intent routing, Chroma vector RAG, response policies, and guardrails.
- Tests for temporal leakage, sequence construction, chatbot behavior, streaming, and answer safety.

## System Layers

```text
Research layer
  -> statistical validation, benchmarks, model evaluation

ML application layer
  -> data pipelines, training, inference, artifacts, backtesting

API and dashboard layer
  -> FastAPI routes, services, schemas, static dashboard

AI assistant layer
  -> local LLM, intent routing, vector RAG, dashboard grounding, guardrails
```

## Repository Structure

```text
app/
  backend/        FastAPI routes, services, chatbot, RAG, schemas, configuration
  dashboard/      Static HTML/CSS/JS dashboard served by FastAPI
  ml/             Training, inference, utilities, and trained artifacts
  pipelines/      Reusable data workflows
  jobs/           CLI entrypoints for full or partial workflows
  data/           Runtime data, reports, forecasts, plots, and minimal inputs
  tests/          Leakage, time-series, API, and chatbot quality tests

docs/
  ARCHITECTURE.md
  CHATBOT_ARCHITECTURE.md
  CHATBOT_DIAGRAMS.md
  CHATBOT_DIAGRAMS.tex

research/
  README.md       Link to the dedicated research notebook repository

tools/
  README.md
  test_local_llm.py
```

## Quick Start

### 1. Create the environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

On macOS/Linux:

```bash
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Install the local LLM model

The chatbot uses Ollama by default:

```powershell
ollama pull qwen2.5:3b
```

### 3. Build the vector RAG index

```powershell
cd app
..\.venv\Scripts\python.exe -m backend.chatbot.rag.build_index
```

On macOS/Linux:

```bash
cd app
../.venv/bin/python -m backend.chatbot.rag.build_index
```

### 4. Launch the application

```powershell
cd app
..\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- Dashboard: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Dashboard Features

- `Forecast`: expected return, VaR, Expected Shortfall, EGARCH volatility, and HMM volatility regime.
- `Backtest`: VaR violations, Kupiec/Christoffersen diagnostics, ES diagnostics, and economic simulation.
- `Report`: latest generated risk report.
- `Admin`: data upload and workflow controls.
- `Chat`: local AI assistant specialized in market risk interpretation.

## Chatbot Architecture

The chatbot is not a generic LLM wrapper. It is a controlled assistant designed around market risk interpretation.

```text
User question
  -> FastAPI chat endpoint
  -> embedding intent router
  -> routed context builder
  -> dashboard state + vector RAG
  -> response policy
  -> prompt builder
  -> local LLM
  -> answer repair + guardrails
  -> final dashboard response
```

The assistant uses:

- `sentence-transformers/all-MiniLM-L6-v2` for lightweight embeddings;
- Chroma for local vector retrieval;
- Ollama for local generation;
- dashboard-state grounding for exact metrics;
- response policies to constrain each answer type;
- guardrails against hallucinated numbers, VaR/ES confusion, and financial advice.

See:

- [Global architecture](docs/ARCHITECTURE.md)
- [Detailed chatbot architecture](docs/CHATBOT_ARCHITECTURE.md)
- [Chatbot diagrams](docs/CHATBOT_DIAGRAMS.md)
- [LaTeX chatbot diagrams](docs/CHATBOT_DIAGRAMS.tex)

## Configuration

Useful environment variables:

```text
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434
AUTO_START_OLLAMA=true
RAG_RETRIEVER_BACKEND=chroma
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_EMBEDDING_DEVICE=cpu
RAG_LOCAL_FILES_ONLY=true
WARM_RAG_ON_STARTUP=true
```

The vector database is generated locally under `app/backend/chatbot/rag/vector_db/` and is ignored by Git. Rebuild it after cloning or after editing the RAG documents.

## ML Workflows

From `app/`:

```powershell
# Full workflow: data, validation, training/inference, report, plots
..\.venv\Scripts\python.exe .\jobs\run_full_masi_pipeline.py

# Reuse existing artifacts and regenerate outputs
..\.venv\Scripts\python.exe .\jobs\run_full_masi_pipeline.py --no-train

# Forecast pipeline only
..\.venv\Scripts\python.exe .\jobs\run_forecast_pipeline.py --no-train

# Refresh validation reports
..\.venv\Scripts\python.exe .\jobs\run_validation_report_pipeline.py
```

## Tests

From `app/`:

```powershell
..\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
```

## Related Repositories

- Research notebooks: https://github.com/mohamedzayd-elfahime/masi-risk-research-notebooks
- Chatbot architecture reference: https://github.com/mohamedzayd-elfahime/market-risk-rag-chatbot

## Publication Notes

This repository intentionally includes trained artifacts and minimal runtime files so the dashboard can be launched locally without retraining. Generated caches, virtual environments, logs, local vector databases, and experimental notebooks are excluded.

## License

MIT.
