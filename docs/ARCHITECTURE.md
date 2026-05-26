# Architecture globale

Ce document decrit l'organisation generale de MASI Risk Engine et la maniere dont les modules collaborent.

## Vue d'ensemble

```text
Utilisateur
  |
  v
Dashboard web statique
  |
  v
FastAPI backend
  |
  +--> Services metier
  |      +--> Forecast, backtest, report, chatbot
  |
  +--> Fichiers applicatifs
  |      +--> donnees, artefacts ML, rapports, plots
  |
  +--> Pipelines et jobs
         +--> preparation donnees, entrainement, inference, validation
```

L'application est organisee autour d'une separation simple:

- `app/` contient l'application executable.
- `research/` contient les travaux exploratoires et le code de recherche.
- `tools/` contient les utilitaires de developpement.
- `docs/` contient uniquement la documentation publique du projet.

## Arborescence principale

```text
app/
  backend/
    api/routes/          Routes FastAPI exposees au dashboard et aux clients HTTP
    chatbot/             Orchestration du chatbot, RAG, prompts et garde-fous
    core/                Configuration et chemins applicatifs centralises
    dashboard_state/     Lecture des fichiers de sortie pour produire du contexte
    llm/                 Client Ollama local
    schemas/             Schemas Pydantic des requetes et reponses
    services/            Facades metier utilisees par les routes
    main.py              Point d'entree FastAPI
  dashboard/             Interface HTML/CSS/JS servie par FastAPI
  data/                  Donnees applicatives, previsions, rapports et plots
  jobs/                  Commandes executables pour workflows complets ou partiels
  ml/
    artifacts/           Modeles et scalers utilises par l'inference
    inference/           Moteur de prevision et logique d'horizon
    training/            Entrainement des modeles
    utils/               Chargement, preprocessing, backtesting, reporting
  pipelines/             Workflows reutilisables appeles par jobs et services
  tests/                 Tests de non-leakage, sequences et chatbot
research/
  src/                   Code de recherche reusable
  notebooks/             Analyses experimentales non necessaires au runtime
  data/                  Donnees de recherche ignorees a la publication
tools/                   Scripts de smoke test et aide locale
```

## Backend FastAPI

`app/backend/main.py` cree l'application FastAPI, active CORS, monte le dashboard statique et enregistre les routeurs:

- `health`: verification de disponibilite.
- `forecast`: dernieres previsions, historique, series prix/risque.
- `backtest`: resume statistique et economique.
- `reports`: rapport Markdown de prevision.
- `plots`: images generees.
- `context`: contexte lisible pour le dashboard/chatbot.
- `chat`: endpoints chatbot standard et streaming.
- `admin`: upload de donnees et lancement de workflows.

Les routes restent fines: elles valident les entrees, appellent les services et retournent des schemas Pydantic.

## Services et schemas

`backend/services/` sert de couche metier stable. Les routes n'ont pas besoin de connaitre les details des fichiers, des modeles ou du chatbot.

`backend/schemas/` centralise les contrats API avec Pydantic. Cela rend les reponses previsibles pour le dashboard et pour la documentation OpenAPI.

## Donnees et artefacts

`backend/core/paths.py` definit les chemins applicatifs a partir de l'emplacement du projet, sans chemin absolu local. Les principales familles de fichiers sont:

- `app/data/final/`: dataset final utilise par l'application.
- `app/data/forecasts/`: journal de previsions.
- `app/data/reports/`: diagnostics, backtests et rapport courant.
- `app/data/reports/plots/`: figures rendues dans le dashboard.
- `app/ml/artifacts/`: artefacts charges par l'inference.

## Pipelines et jobs

Les scripts de `app/jobs/` sont les entrees CLI. Ils orchestrent les modules reutilisables de `app/pipelines/` et `app/ml/`.

Flux typique:

```text
Donnees MASI brutes
  -> nettoyage
  -> transformation
  -> dataset final
  -> entrainement ou chargement artefacts
  -> inference VaR/ES/rendement/regime
  -> backtesting
  -> rapport et plots
  -> dashboard/API
```

## Dashboard

`app/dashboard/index.html` charge `assets/css/app.css` et `assets/js/app.js`. L'interface consomme les endpoints FastAPI pour afficher les previsions, graphes, rapports et conversations.

Le dashboard est volontairement statique: aucun bundler n'est requis pour lancer l'application.

## Tests

Les tests couvrent les points critiques:

- construction des sequences temporelles sans fuite de donnees futures;
- splits chronologiques;
- comportement du chatbot, policies, garde-fous et streaming.

Cette architecture garde une frontiere claire entre exploration, application executable et documentation publique.
