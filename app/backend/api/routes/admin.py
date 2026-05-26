"""Admin API for launching MASI backend pipelines from the dashboard."""

from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.core.paths import APP_ROOT, LOGS_DIR, ensure_backend_dirs
from pipelines.data_pipeline.paths import RAW_DIR, ensure_pipeline_dirs


router = APIRouter(prefix="/admin", tags=["admin"])

RunStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass(frozen=True)
class PipelineDefinition:
    id: str
    name: str
    description: str
    args: tuple[str, ...]
    estimated_duration: str
    category: str


class PipelineRun(BaseModel):
    id: str
    pipeline_id: str
    pipeline_name: str
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    return_code: int | None = None
    log_path: str
    command: list[str]


class PipelineRunDetail(PipelineRun):
    log_tail: str


class UploadedDataFile(BaseModel):
    filename: str
    saved_path: str
    size_bytes: int


PIPELINES: dict[str, PipelineDefinition] = {
    "data_cleaning": PipelineDefinition(
        id="data_cleaning",
        name="Importer et nettoyer les donnees",
        description="Nettoie les fichiers uploades, fusionne l'historique, transforme les donnees et recalcule les features EGARCH.",
        args=("jobs/run_masi_update.py",),
        estimated_duration="2-10 min",
        category="Donnees",
    ),
    "forecast_inference": PipelineDefinition(
        id="forecast_inference",
        name="Generer la prevision",
        description="Utilise les modeles existants pour produire la prevision MASI, le rapport et le graphique.",
        args=("jobs/run_forecast_pipeline.py", "--no-train"),
        estimated_duration="1-3 min",
        category="Production",
    ),
    "validation_report": PipelineDefinition(
        id="validation_report",
        name="Valider et regenerer le rapport",
        description="Met a jour le backtest statistique/economique puis regenere le rapport final.",
        args=("jobs/run_validation_report_pipeline.py",),
        estimated_duration="1-5 min",
        category="Production",
    ),
    "forecast_train": PipelineDefinition(
        id="forecast_train",
        name="Appliquer Optuna puis prevoir",
        description="Reentraine les modeles avec les meilleurs hyperparametres Optuna sauvegardes, puis genere les sorties forecast.",
        args=("jobs/run_forecast_pipeline.py", "--use-best-params"),
        estimated_duration="10-30 min",
        category="Modele",
    ),
    "forecast_train_return": PipelineDefinition(
        id="forecast_train_return",
        name="Appliquer Optuna return puis prevoir",
        description="Reentraine avec Optuna VaR/ES verrouille, puis applique Optuna return seulement si ses parametres sont acceptes.",
        args=("jobs/run_forecast_pipeline.py", "--use-best-params", "--use-best-return-params"),
        estimated_duration="8-25 min",
        category="Modele",
    ),
    "full_pipeline": PipelineDefinition(
        id="full_pipeline",
        name="Executer le workflow complet",
        description="Nettoyage donnees, validation, reentrainement, forecast, rapport et graphique.",
        args=("jobs/run_full_masi_pipeline.py", "--use-best-params"),
        estimated_duration="15-40 min",
        category="Modele",
    ),
    "hyperopt_robust": PipelineDefinition(
        id="hyperopt_robust",
        name="Tuning Optuna robuste",
        description="Optimise d'abord les p-values Kupiec et Christoffersen; sauvegarde les meilleurs parametres sans remplacer les artefacts.",
        args=("jobs/run_hyperparameter_search.py", "--max-trials", "30", "--epochs", "25", "--patience", "8"),
        estimated_duration="1-3 h",
        category="Recherche",
    ),
    "hyperopt_return": PipelineDefinition(
        id="hyperopt_return",
        name="Tuning Optuna return",
        description="Optimise seulement le LSTM de rendement sur validation chronologique; le test reste reserve a l'audit anti-leakage.",
        args=("jobs/run_return_hyperparameter_search.py", "--max-trials", "25", "--epochs", "25", "--patience", "8"),
        estimated_duration="45-120 min",
        category="Recherche",
    ),
}

_runs: dict[str, PipelineRun] = {}
_lock = threading.Lock()


@router.get("/pipelines")
def list_pipelines() -> dict[str, object]:
    return {
        "pipelines": [
            {
                "id": pipeline.id,
                "name": pipeline.name,
                "description": pipeline.description,
                "estimated_duration": pipeline.estimated_duration,
                "category": pipeline.category,
            }
            for pipeline in PIPELINES.values()
        ],
        "runs": [_run_summary(run) for run in _sorted_runs()],
    }


@router.post("/data/upload")
async def upload_masi_data(file: UploadFile = File(...)) -> UploadedDataFile:
    suffix = Path(file.filename or "").suffix.lower()
    allowed_suffixes = {".csv", ".txt", ".xlsx", ".xls"}
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail="Format non supporte. Utilise .csv, .txt, .xlsx ou .xls.",
        )

    ensure_pipeline_dirs()
    original_name = _safe_filename(file.filename or f"masi_upload{suffix}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = RAW_DIR / f"{Path(original_name).stem}_{timestamp}{suffix}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Le fichier envoye est vide.")

    target_path.write_bytes(content)
    return UploadedDataFile(
        filename=original_name,
        saved_path=str(target_path),
        size_bytes=len(content),
    )


@router.post("/pipelines/{pipeline_id}/run")
def run_pipeline(pipeline_id: str) -> PipelineRun:
    pipeline = PIPELINES.get(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Pipeline inconnu: {pipeline_id}")

    with _lock:
        active = next((run for run in _runs.values() if run.status in {"queued", "running"}), None)
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Un pipeline est deja en cours: {active.pipeline_name}",
            )

        ensure_backend_dirs()
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now().isoformat(timespec="seconds")
        log_path = LOGS_DIR / f"admin_pipeline_{pipeline.id}_{run_id}.log"
        command = [sys.executable, *pipeline.args]
        run = PipelineRun(
            id=run_id,
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            status="queued",
            started_at=started_at,
            log_path=str(log_path),
            command=command,
        )
        _runs[run_id] = run

    thread = threading.Thread(target=_execute_pipeline, args=(run_id, pipeline, log_path), daemon=True)
    thread.start()
    return run


@router.get("/pipelines/runs")
def list_runs() -> dict[str, object]:
    return {"runs": [_run_summary(run) for run in _sorted_runs()]}


@router.get("/pipelines/runs/{run_id}")
def get_run(run_id: str) -> PipelineRunDetail:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run inconnu: {run_id}")
    return PipelineRunDetail(**_dump_model(run), log_tail=_read_log_tail(Path(run.log_path)))


def _execute_pipeline(run_id: str, pipeline: PipelineDefinition, log_path: Path) -> None:
    _update_run(run_id, status="running")
    command = [sys.executable, *pipeline.args]

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            handle.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
            handle.write(f"Command: {' '.join(command)}\n\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=APP_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = process.wait()
    except Exception as exc:
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"\nAdmin runner failed: {exc}\n")
        _update_run(run_id, status="failed", return_code=-1)
        return

    _update_run(
        run_id,
        status="succeeded" if return_code == 0 else "failed",
        return_code=return_code,
    )


def _update_run(run_id: str, **updates: object) -> None:
    with _lock:
        run = _runs[run_id]
        payload = _dump_model(run)
        payload.update(updates)
        if payload.get("status") in {"succeeded", "failed"} and payload.get("finished_at") is None:
            payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _runs[run_id] = PipelineRun(**payload)


def _sorted_runs() -> list[PipelineRun]:
    return sorted(_runs.values(), key=lambda run: run.started_at, reverse=True)


def _run_summary(run: PipelineRun) -> dict[str, object]:
    payload = _dump_model(run)
    payload["log_tail"] = _read_log_tail(Path(run.log_path), max_chars=2400)
    return payload


def _dump_model(model: BaseModel) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _read_log_tail(path: Path, max_chars: int = 6000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _safe_filename(filename: str) -> str:
    keep = []
    for char in filename:
        if char.isalnum() or char in {" ", ".", "_", "-"}:
            keep.append(char)
    cleaned = "".join(keep).strip().replace(" ", "_")
    return cleaned or "masi_upload.csv"
