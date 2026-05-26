# Guide rapide MASI Risk Engine

La documentation complete est dans le dossier `../docs/` et peut etre publiee sur Read the Docs.

## Lancer l'application

Depuis `app/` :

```powershell
..\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Ouvrir :

```text
http://127.0.0.1:8000/
```

## Pages du dashboard

- Forecast : prevision MASI, VaR, ES, EGARCH, regime HMM et horizons 1D/10D/25D.
- Backtest : violations VaR, Kupiec, Christoffersen, diagnostic ES et simulation economique.
- Report : dernier rapport Markdown genere par le pipeline.
- Admin : upload de donnees, lancement des pipelines, suivi des runs et logs.

## Pipelines principaux

Depuis `app/` :

```powershell
..\.venv\Scripts\python.exe .\jobs\run_full_masi_pipeline.py
..\.venv\Scripts\python.exe .\jobs\run_full_masi_pipeline.py --no-train
..\.venv\Scripts\python.exe .\jobs\run_forecast_pipeline.py --no-train
..\.venv\Scripts\python.exe .\jobs\run_validation_report_pipeline.py
```

## Optuna

```powershell
..\.venv\Scripts\python.exe .\jobs\run_hyperparameter_search.py --max-trials 30 --epochs 25 --patience 8
..\.venv\Scripts\python.exe .\jobs\run_return_hyperparameter_search.py --max-trials 25 --epochs 25 --patience 8
```

## Chatbot

Le chatbot utilise Ollama par defaut :

```powershell
ollama serve
ollama pull qwen2.5:3b
```

Il utilise le contexte du dashboard, le RAG documentaire, les response policies, `answer_repair` et les guardrails.

## Tests

```powershell
..\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
```

## Documentation locale

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe -m pip install -r docs\requirements.txt
.\.venv\Scripts\python.exe -m sphinx -b html docs docs\_build\html
```
