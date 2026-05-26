"""Dynamic context builder for the MASI Risk Dashboard chatbot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.paths import (
    ECON_BACKTEST_PATH,
    FORECAST_LOG_PATH,
    METADATA_PATH,
    REPORT_DIR,
    STAT_BACKTEST_PATH,
    TEST_PREDICTIONS_PATH,
    TRAINING_DIAGNOSTICS_PATH,
    TRAINING_HISTORY_PATH,
)


LATEST_REPORT_PATH = REPORT_DIR / "latest_forecast_report.md"


def safe_read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def safe_read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None


def safe_read_csv(path: str | Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(Path(path))
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    if df.empty:
        return None
    return df.copy()


def safe_read_csv_tail(path: str | Path, n: int = 5) -> pd.DataFrame | None:
    df = safe_read_csv(path)
    if df is None:
        return None
    return df.tail(n).copy()


def _fmt_pct(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "indisponible"
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "indisponible"


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "indisponible"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "indisponible"


def _fmt_int(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "indisponible"
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "indisponible"


def _fmt_date(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "indisponible"
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(value)


def _fmt_datetime(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "indisponible"
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)


def _fmt_list(values: Any) -> str:
    if not values:
        return "indisponible"
    if isinstance(values, (list, tuple)):
        return ", ".join(str(v) for v in values)
    return str(values)


def _safe_get(row: pd.Series, name: str, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        value = row.get(name, default)
    except Exception:
        return default
    return default if pd.isna(value) else value


def _latest_forecast_batch(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "run_date" in df.columns:
        latest = df.copy()
        latest["run_date"] = pd.to_datetime(latest["run_date"], errors="coerce")
        last_run = latest["run_date"].max()
        if pd.notna(last_run):
            latest = latest.loc[latest["run_date"] == last_run].copy()
            if "horizon" in latest.columns:
                latest["horizon_num"] = pd.to_numeric(latest["horizon"], errors="coerce")
                latest = latest.sort_values(["horizon_num", "target_date"], kind="stable")
                latest = latest.drop(columns=["horizon_num"])
            return latest

    return df.tail(3).copy()


def build_forecast_context() -> str:
    forecast_df = safe_read_csv(FORECAST_LOG_PATH)
    report_text = safe_read_text(LATEST_REPORT_PATH)
    test_tail = safe_read_csv_tail(TEST_PREDICTIONS_PATH, n=5)

    lines: list[str] = []

    if forecast_df is not None:
        latest_batch = _latest_forecast_batch(forecast_df)
        if latest_batch is not None and not latest_batch.empty:
            first_row = latest_batch.iloc[0]
            lines.append(
                f"La derniere execution disponible date du {_fmt_datetime(_safe_get(first_row, 'run_date'))}."
            )
            lines.append(
                f"Le modele utilise est {str(_safe_get(first_row, 'model_name', 'indisponible'))}, "
                f"version {str(_safe_get(first_row, 'model_version', 'indisponible'))}, "
                f"avec la version de donnees {str(_safe_get(first_row, 'data_version', 'indisponible'))}."
            )

            for _, row in latest_batch.iterrows():
                lines.append(
                    f"Pour l'horizon {_fmt_int(_safe_get(row, 'horizon'))} jour(s), la date cible est {_fmt_date(_safe_get(row, 'target_date'))}. "
                    f"Le regime estime est {str(_safe_get(row, 'regime', 'indisponible'))}. "
                    f"La moyenne conditionnelle est {_fmt_pct(_safe_get(row, 'mean_forecast'), 3)}, "
                    f"la volatilite prevue est {_fmt_pct(_safe_get(row, 'volatility_forecast'), 3)}, "
                    f"le rendement prevu est {_fmt_pct(_safe_get(row, 'return_forecast'), 3)}, "
                    f"la VaR 5% est {_fmt_pct(_safe_get(row, 'var_forecast'), 3)} et "
                    f"l'ES 5% est {_fmt_pct(_safe_get(row, 'es_forecast'), 3)}."
                )

            one_day = latest_batch.copy()
            if "horizon" in one_day.columns:
                one_day = one_day.loc[pd.to_numeric(one_day["horizon"], errors="coerce") == 1]
            if not one_day.empty:
                row_1 = one_day.iloc[0]
                lines.append(
                    "Lecture rendement-risque : pour l'horizon principal a 1 jour, "
                    f"le rendement prevu est {_fmt_pct(_safe_get(row_1, 'return_forecast'), 3)}, "
                    f"la VaR 5% est {_fmt_pct(_safe_get(row_1, 'var_forecast'), 3)}, "
                    f"l'ES 5% est {_fmt_pct(_safe_get(row_1, 'es_forecast'), 3)}, "
                    f"et le regime courant est {str(_safe_get(row_1, 'regime', 'indisponible'))}. "
                    "Cette combinaison doit etre interpretee comme un diagnostic conditionnel du risque, "
                    "pas comme une recommandation d'investissement."
                )
    elif report_text:
        lines.append(
            "Le rapport de prevision le plus recent est disponible, mais forecast_log.csv n'a pas pu etre exploite proprement. "
            "Les valeurs numeriques detaillees ne sont donc pas reprises ici."
        )
    else:
        lines.append("Le rapport de prevision le plus recent est indisponible.")

    if test_tail is not None:
        lines.append(
            "Les lignes suivantes correspondent aux dernieres observations historiques de la periode de test, "
            "et non a des previsions futures."
        )
        for _, row in test_tail.iterrows():
            violation_raw = row.get("violation", row.get("breach"))
            violation_txt = "indisponible"
            if violation_raw is not None and not pd.isna(violation_raw):
                try:
                    violation_txt = "oui" if int(float(violation_raw)) == 1 else "non"
                except (TypeError, ValueError):
                    violation_txt = str(violation_raw)
            lines.append(
                f"- {_fmt_date(row.get('date'))} : "
                f"rendement realise {_fmt_pct(row.get('realized_return'), 3)}, "
                f"rendement predit {_fmt_pct(row.get('return_pred'), 3)}, "
                f"VaR {_fmt_pct(row.get('var_pred'), 3)}, "
                f"ES {_fmt_pct(row.get('es_pred'), 3)}, "
                f"violation VaR : {violation_txt}."
            )
    else:
        lines.append("Les dernieres observations du fichier test_predictions.csv sont indisponibles.")

    return "\n".join(lines)


def build_statistical_backtest_context() -> str:
    statistical = safe_read_json(STAT_BACKTEST_PATH)
    diagnostics = safe_read_json(TRAINING_DIAGNOSTICS_PATH)

    if statistical is None:
        return (
            "Le fichier de backtesting statistique est indisponible. "
            "Le taux de violation, les tests de Kupiec, de Christoffersen et le diagnostic ES ne peuvent pas etre resumes."
        )

    lines = [
        f"Taux de violation observe : {_fmt_pct(statistical.get('violation_rate'))}.",
        f"Niveau de violation attendu : {_fmt_pct(statistical.get('expected_violation_rate'))}.",
        "Precision importante : le niveau de violation attendu est le taux theorique attendu de violations VaR, pas une p-value.",
        f"Nombre de breaches VaR : {_fmt_int(statistical.get('n_var_breaches'))} sur {_fmt_int(statistical.get('n_observations'))} observations.",
        f"Test de Kupiec : statistique {_fmt_num(statistical.get('kupiec_pof_stat'))}, p-value {_fmt_num(statistical.get('kupiec_pof_p_value'))}.",
        (
            f"Test d'independance de Christoffersen : statistique "
            f"{_fmt_num(statistical.get('christoffersen_independence_stat'))}, "
            f"p-value {_fmt_num(statistical.get('christoffersen_independence_p_value'))}."
        ),
        (
            f"Test de couverture conditionnelle : statistique "
            f"{_fmt_num(statistical.get('christoffersen_cc_stat'))}, "
            f"p-value {_fmt_num(statistical.get('christoffersen_cc_p_value'))}."
        ),
        (
            f"Diagnostic de queue pour l'ES : {_fmt_int(statistical.get('n_es_tail_observations'))} observations de queue, "
            f"statistique {_fmt_num(statistical.get('es_tail_calibration_stat'))}, "
            f"p-value {_fmt_num(statistical.get('es_tail_calibration_p_value'))}, "
            f"residuel moyen {_fmt_pct(statistical.get('es_tail_residual_mean'), 3)}."
        ),
    ]

    if diagnostics is not None and isinstance(diagnostics.get("es_model"), dict):
        es_diag = diagnostics["es_model"]
        lines.append(
            "Diagnostic d'entrainement ES : "
            f"methode {es_diag.get('method', 'indisponible')}, "
            f"violations d'entrainement utilisees {_fmt_int(es_diag.get('n_train_violations_used'))}."
        )

    return "\n".join(lines)


def build_economic_backtest_context() -> str:
    economic = safe_read_json(ECON_BACKTEST_PATH)

    if economic is None:
        return (
            "Le fichier de backtesting economique est indisponible. "
            "La comparaison de richesse simulee, de Sharpe historique et de drawdown ne peut pas etre resume."
        )

    lines = [
        "Le backtesting economique correspond a une simulation historique d'une regle risk-managed comparee a un scenario Buy & Hold.",
        f"Richesse finale simulee : {_fmt_num(economic.get('final_wealth'), 3)} contre {_fmt_num(economic.get('buy_hold_final_wealth'), 3)} pour Buy & Hold.",
        f"Sharpe annualise historique : {_fmt_num(economic.get('annualized_sharpe'), 3)} contre {_fmt_num(economic.get('buy_hold_sharpe'), 3)} pour Buy & Hold.",
        f"Maximum drawdown historique : {_fmt_pct(economic.get('max_drawdown'))} contre {_fmt_pct(economic.get('buy_hold_max_drawdown'))} pour Buy & Hold.",
        f"Poids simule moyen : {_fmt_num(economic.get('average_weight_used'), 3)}. Turnover moyen : {_fmt_num(economic.get('average_turnover'), 3)}.",
        "Ces metriques relevent d'un backtesting economique historique simule et ne constituent pas un conseil d'investissement.",
    ]

    final_wealth = economic.get("final_wealth")
    buy_hold_final_wealth = economic.get("buy_hold_final_wealth")
    annualized_sharpe = economic.get("annualized_sharpe")
    buy_hold_sharpe = economic.get("buy_hold_sharpe")
    max_drawdown = economic.get("max_drawdown")
    buy_hold_max_drawdown = economic.get("buy_hold_max_drawdown")

    comparable_values = [
        final_wealth,
        buy_hold_final_wealth,
        annualized_sharpe,
        buy_hold_sharpe,
        max_drawdown,
        buy_hold_max_drawdown,
    ]
    if all(value is not None and not pd.isna(value) for value in comparable_values):
        wealth_read = "inferieure" if float(final_wealth) < float(buy_hold_final_wealth) else "superieure"
        sharpe_read = "plus eleve" if float(annualized_sharpe) > float(buy_hold_sharpe) else "plus faible"
        drawdown_read = (
            "moins severe"
            if abs(float(max_drawdown)) < abs(float(buy_hold_max_drawdown))
            else "plus severe"
        )
        lines.append(
            "Lecture economique : "
            f"la richesse finale simulee est {wealth_read} au Buy & Hold, "
            f"le Sharpe historique est {sharpe_read}, "
            f"et le maximum drawdown historique est {drawdown_read}. "
            "Cette lecture indique un profil rendement-risque historique simule, "
            "sans constituer une preuve de superiorite future ni une recommandation d'investissement."
        )

    return "\n".join(lines)


def build_model_metadata_context() -> str:
    metadata = safe_read_json(METADATA_PATH)
    history = safe_read_json(TRAINING_HISTORY_PATH)

    if metadata is None:
        return "Les metadonnees du modele sont indisponibles."

    lines = [
        f"Type de modele : {metadata.get('model_name', 'indisponible')} (version {metadata.get('model_version', 'indisponible')}).",
        f"Niveau alpha : {_fmt_num(metadata.get('alpha'), 2)}.",
        "L'horizon principal du systeme est 1 jour, avec des extensions operationnelles 10 jours et 25 jours.",
        f"Longueur de sequence LSTM : {_fmt_int(metadata.get('seq_len'))}.",
        f"Variables de la branche VaR : {_fmt_list(metadata.get('var_feature_cols') or metadata.get('feature_cols'))}.",
        f"Variables de la branche rendement : {_fmt_list(metadata.get('return_feature_cols'))}.",
        f"Periode d'apprentissage : {_fmt_date(metadata.get('train_start'))} a {_fmt_date(metadata.get('train_end'))}.",
        f"Periode de validation : {_fmt_date(metadata.get('validation_start'))} a {_fmt_date(metadata.get('validation_end'))}.",
        f"Periode de test : {_fmt_date(metadata.get('test_start'))} a {_fmt_date(metadata.get('test_end'))}.",
        f"Version de donnees : {metadata.get('data_version', 'indisponible')}.",
    ]

    if history is not None:
        var_hist = history.get("var_history", {})
        ret_hist = history.get("return_history", {})
        if isinstance(var_hist, dict):
            lines.append(
                "Historique d'entrainement VaR disponible : "
                f"{_fmt_int(len(var_hist.get('train_loss', [])))} epoques enregistrees."
            )
        if isinstance(ret_hist, dict):
            lines.append(
                "Historique d'entrainement rendement disponible : "
                f"{_fmt_int(len(ret_hist.get('train_loss', [])))} epoques enregistrees."
            )

    return "\n".join(lines)


def build_dashboard_context() -> str:
    sections = [
        "# Contexte dynamique du MASI Risk Dashboard",
        "",
        "## Derniere prevision disponible",
        build_forecast_context(),
        "",
        "## Backtesting statistique",
        build_statistical_backtest_context(),
        "",
        "## Backtesting economique simule",
        build_economic_backtest_context(),
        "",
        "## Metadonnees du modele",
        build_model_metadata_context(),
        "",
        "## Rappel methodologique important",
        (
            "Les previsions portent sur les rendements logarithmiques du MASI, meme si le dashboard peut afficher "
            "les niveaux de prix historiques. La VaR et l'ES sont exprimees dans l'espace des rendements et du risque. "
            "Les horizons 10 jours et 25 jours sont des extensions mathematiques du modele 1 jour, sauf si un modele "
            "multi-horizon independamment entraine est explicitement disponible. Les metriques economiques telles que "
            "la richesse, le Sharpe et le drawdown correspondent a un backtesting economique simule. Les poids de "
            "risk-targeting, lorsqu'ils sont presents, doivent etre interpretes comme des poids simules et non comme "
            "des allocations recommandees."
        ),
    ]
    return "\n".join(sections).strip()


if __name__ == "__main__":
    print(build_dashboard_context())
