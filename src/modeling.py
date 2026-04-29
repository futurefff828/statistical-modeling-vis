from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import TABLES_DIAGNOSTIC_DIR, TABLES_MODEL_DIR

LOGGER = logging.getLogger(__name__)


BASE_Y = "quality_efficiency_index"
BASE_X = "annual_ai_log"
CONTROL_VARS = ["ln_assets", "leverage", "rd_intensity", "firm_age", "revenue_growth"]


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    q_low, q_high = series.quantile([lower, upper])
    return series.clip(q_low, q_high)


def prepare_model_data(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [BASE_Y, BASE_X, "mda_ai_log", "ai_invest_log", "ai_application_index", "tfp"] + CONTROL_VARS
    data = panel.copy()
    for col in cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            if col not in ["year"]:
                data[f"{col}_w"] = winsorize_series(data[col])
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data.loc[:, numeric_cols] = data.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)
    return data


def fit_ols(formula: str, data: pd.DataFrame, cluster: bool = False):
    model = smf.ols(formula, data=data)
    if cluster:
        fit_data = model.data.frame
        return model.fit(cov_type="cluster", cov_kwds={"groups": fit_data["code"]})
    return model.fit(cov_type="HC3")


def model_specs() -> list[dict[str, str | bool]]:
    controls = " + ".join([f"{v}_w" for v in CONTROL_VARS])
    return [
        {
            "model": "M1_fe_baseline",
            "formula": f"{BASE_Y}_w ~ {BASE_X}_w + C(year) + C(industry_2digit)",
            "cluster": True,
            "note": "Fixed-effect baseline with year and manufacturing subindustry effects; firm-clustered standard errors.",
        },
        {
            "model": "M2_fe_controls",
            "formula": f"{BASE_Y}_w ~ {BASE_X}_w + {controls} + C(year)",
            "cluster": True,
            "note": "Adds controls and year fixed effects; firm-clustered standard errors.",
        },
        {
            "model": "M3_fe_controls_industry",
            "formula": f"{BASE_Y}_w ~ {BASE_X}_w + {controls} + C(year) + C(industry_2digit)",
            "cluster": True,
            "note": "Adds manufacturing subindustry and year fixed effects; firm-clustered standard errors.",
        },
    ]


def summarize_fit(name: str, fit, core_var: str) -> dict[str, float | str]:
    return {
        "model": name,
        "core_var": core_var,
        "coef": fit.params.get(core_var, np.nan),
        "std_err": fit.bse.get(core_var, np.nan),
        "t_value": fit.tvalues.get(core_var, np.nan),
        "p_value": fit.pvalues.get(core_var, np.nan),
        "nobs": int(fit.nobs),
        "r_squared": fit.rsquared,
        "adj_r_squared": fit.rsquared_adj,
    }


def run_regressions(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    data = prepare_model_data(panel)
    fits = {}
    rows = []
    for spec in model_specs():
        formula = str(spec["formula"])
        needed = [BASE_Y, BASE_X, "code", "year", "industry_2digit"] + CONTROL_VARS
        model_data = data.dropna(subset=[c if c in ["code", "year", "industry_2digit"] else f"{c}_w" for c in needed]).copy()
        fit = fit_ols(formula, model_data, cluster=bool(spec["cluster"]))
        fits[str(spec["model"])] = fit
        row = summarize_fit(str(spec["model"]), fit, f"{BASE_X}_w")
        row["formula"] = formula
        row["specification_note"] = str(spec["note"])
        rows.append(row)
        LOGGER.info("Fitted %s with %s observations", spec["model"], int(fit.nobs))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "baseline_regression_results.csv", index=False, encoding="utf-8-sig")
    return table, fits


def run_robustness(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_model_data(panel)
    controls = " + ".join([f"{v}_w" for v in CONTROL_VARS])
    specs = [
        ("R1_mda_frequency_x", f"{BASE_Y}_w ~ mda_ai_log_w + {controls} + C(year) + C(industry_2digit)", "mda_ai_log_w"),
        ("R2_composite_ai_x", f"{BASE_Y}_w ~ ai_application_index_w + {controls} + C(year) + C(industry_2digit)", "ai_application_index_w"),
        ("R3_tfp_y", f"tfp_w ~ {BASE_X}_w + {controls} + C(year) + C(industry_2digit)", f"{BASE_X}_w"),
    ]
    rows = []
    for name, formula, core in specs:
        required = [core, "code", "year", "industry_2digit"] + [f"{v}_w" for v in CONTROL_VARS]
        y_var = formula.split("~", maxsplit=1)[0].strip()
        required.append(y_var)
        model_data = data.dropna(subset=required).copy()
        if len(model_data) < 100:
            continue
        fit = fit_ols(formula, model_data, cluster=True)
        row = summarize_fit(name, fit, core)
        row["formula"] = formula
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "robustness_regression_results.csv", index=False, encoding="utf-8-sig")
    return table


def descriptive_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        BASE_Y,
        BASE_X,
        "mda_ai_log",
        "ai_application_index",
        "ai_invest_log",
        "tfp",
        "ln_assets",
        "leverage",
        "rd_intensity",
        "firm_age",
        "revenue_growth",
    ]
    available = [c for c in cols if c in panel.columns]
    desc = panel[available].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"50%": "median"})
    desc.to_csv(TABLES_DIAGNOSTIC_DIR / "descriptive_statistics.csv", encoding="utf-8-sig")
    return desc


def missingness_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in panel.columns:
        rows.append({"variable": col, "missing": panel[col].isna().sum(), "missing_rate": panel[col].isna().mean()})
    table = pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    table.to_csv(TABLES_DIAGNOSTIC_DIR / "missingness_summary.csv", index=False, encoding="utf-8-sig")
    return table


def vif_table(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_model_data(panel)
    cols = [f"{v}_w" for v in [BASE_X] + CONTROL_VARS]
    model_data = data[cols].dropna()
    if len(model_data) < 100:
        table = pd.DataFrame(columns=["variable", "vif"])
    else:
        x = model_data.assign(const=1.0)
        table = pd.DataFrame(
            [{"variable": col, "vif": variance_inflation_factor(x.values, i)} for i, col in enumerate(x.columns) if col != "const"]
        )
    table.to_csv(TABLES_DIAGNOSTIC_DIR / "vif_diagnostics.csv", index=False, encoding="utf-8-sig")
    return table


def heteroskedasticity_test(fits: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, fit in fits.items():
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(fit.resid, fit.model.exog)
        rows.append(
            {
                "model": name,
                "bp_lm_stat": lm,
                "bp_lm_pvalue": lm_pvalue,
                "bp_f_stat": fvalue,
                "bp_f_pvalue": f_pvalue,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIAGNOSTIC_DIR / "heteroskedasticity_tests.csv", index=False, encoding="utf-8-sig")
    return table
