# MASI Risk Engine

MASI Risk Engine est une application locale de prevision et d'analyse du risque pour l'indice MASI. Elle combine une API FastAPI, un dashboard web statique, des pipelines ML, des artefacts de modele, des rapports de backtesting et un chatbot controle par RAG.

Le projet sert a explorer les previsions de rendement, la VaR, l'Expected Shortfall, les regimes de volatilite HMM et les resultats de validation. Il ne fournit pas de conseil d'investissement.

## Fonctionnalites

- Dashboard web servi par FastAPI.
- API REST pour les previsions, series de prix, backtests, rapports, plots, contexte dashboard et chatbot.
- Pipelines de donnees, entrainement, inference, validation et generation de rapport.
- Modeles hybrides autour de LSTM, Ridge, EGARCH/GARCH, HMM et modeles tabulaires.
- Chatbot local via Ollama, avec routage d'intention, RAG Markdown, contexte numerique, politiques de reponse et garde-fous.
- Tests anti-leakage, sequences temporelles et qualite chatbot.

## Prerequis

- Python 3.11 ou plus recent.
- Ollama pour utiliser le chatbot local.
- Le modele Ollama configure par defaut:

```powershell
ollama pull qwen2.5:3b
```

## Installation

Depuis la racine du projet, installation standard pour lancer l'application:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

Sur macOS/Linux, remplace le chemin Python par:

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

Installation complete, avec les dependances de test:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-all.txt
```

Construire ou reconstruire l'index RAG vectoriel:

```powershell
cd app
..\.venv\Scripts\python.exe -m backend.chatbot.rag.build_index
```

## Demarrer l'application

Depuis `app/`:

```powershell
cd app
..\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Puis ouvre:

- Dashboard: `http://127.0.0.1:8000/`
- Documentation API interactive: `http://127.0.0.1:8000/docs`

## Utiliser le dashboard

Le dashboard regroupe les vues principales:

- `Forecast`: prevision de rendement, VaR, ES, volatilite EGARCH et regime HMM.
- `Backtest`: violations de VaR, tests Kupiec/Christoffersen, diagnostics ES et simulation economique.
- `Report`: dernier rapport de prevision genere par les jobs.
- `Admin`: chargement de donnees MASI et lancement des workflows backend.
- `Chat`: assistant specialise MASI Risk Dashboard.

Le chatbot utilise les valeurs affichees dans l'interface quand elles sont envoyees avec la question. Il refuse les demandes de recommandation d'achat/vente et se limite a l'interpretation du dashboard.

## Configuration utile

Variables d'environnement principales:

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

Le mode RAG `chroma` est le mode par defaut. Il utilise la base vectorielle locale generee dans `app/backend/chatbot/rag/vector_db/`. Cette base est un artefact local ignore par Git; il faut la reconstruire apres un clone ou apres modification des documents RAG.

Avec `WARM_RAG_ON_STARTUP=true`, FastAPI precharge le modele d'embedding et la base Chroma au demarrage. Le startup peut prendre un peu plus de temps, mais le premier message chatbot evite le cold start du retrieval. Avec `RAG_LOCAL_FILES_ONLY=true`, le runtime utilise le modele deja telecharge localement et n'appelle pas Hugging Face pendant le warmup.

```powershell
cd app
..\.venv\Scripts\python.exe -m backend.chatbot.rag.build_index
```

Si tu veux lancer Ollama manuellement:

```powershell
ollama serve
```

Pour desactiver le demarrage automatique par l'API:

```powershell
$env:AUTO_START_OLLAMA="false"
```

## Workflows ML

Depuis `app/`:

```powershell
# Pipeline complet: donnees, validation, entrainement/inference, rapport et plots
..\.venv\Scripts\python.exe .\jobs\run_full_masi_pipeline.py

# Reutiliser les artefacts existants et regenerer les sorties
..\.venv\Scripts\python.exe .\jobs\run_full_masi_pipeline.py --no-train

# Pipeline de prevision uniquement
..\.venv\Scripts\python.exe .\jobs\run_forecast_pipeline.py --no-train

# Rafraichir les rapports de validation
..\.venv\Scripts\python.exe .\jobs\run_validation_report_pipeline.py
```

## Tests

Depuis `app/`:

```powershell
..\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
```

## Documentation projet

- [Architecture globale](docs/ARCHITECTURE.md)
- [Architecture detaillee du chatbot](docs/CHATBOT_ARCHITECTURE.md)

## Notes de publication

Le depot ignore les environnements virtuels, caches Python, sorties temporaires, bases vectorielles locales, donnees de recherche generees et notebooks experimentaux. Les chemins locaux de developpement ne sont pas requis pour lancer l'application.

## Licence

MIT.
