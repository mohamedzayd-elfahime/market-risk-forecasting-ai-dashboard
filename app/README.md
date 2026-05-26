# Application Boundary

`app/` is the runnable MASI Risk Engine application.

It contains the FastAPI backend, the static dashboard, production jobs, ML code, model artifacts, app-local data and tests. Code inside this directory should be able to run without importing from `research/`.

Main entry points:

- `backend/main.py`: FastAPI application
- `backend/chatbot/`: chatbot domain, response policies, RAG and prompt control
- `backend/dashboard_state/`: readers and formatters for dashboard output files
- `dashboard/`: browser UI served by FastAPI
- `jobs/`: executable workflows used by the CLI and Admin page
- `pipelines/`: reusable application workflows called by jobs and services
- `ml/`: training, inference, utilities and artifacts
- `data/`: app runtime data and generated outputs only, no executable pipeline code
- `tests/`: regression, leakage and chatbot tests
