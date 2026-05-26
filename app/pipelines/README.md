# Pipelines

`pipelines/` contains reusable application workflows.

These modules are not HTTP routes and they are not raw data. They hold domain pipelines that can be called from CLI jobs, backend services or the Admin page.

Current pipeline packages:

- `data_pipeline/`: MASI raw-file cleaning, history merge, feature transformation and master-dataset generation.
