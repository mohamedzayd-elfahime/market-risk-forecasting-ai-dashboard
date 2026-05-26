# Research Workspace

`research/` contains exploratory and report-oriented material that is not required by the running dashboard.

Folder purpose:

- `notebooks/`: research notebooks grouped by modeling stage
- `src/`: reusable notebook helpers and experimental analysis code
- `data/`: research datasets used by notebooks
- `reports/`: research figures and tables

Production code should live in `app/`. When a research workflow becomes part of the dashboard or Admin page, promote it deliberately into `app/jobs`, `app/ml` or `app/backend/services`.
