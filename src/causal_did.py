from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import FIGURES_DIR, RANDOM_STATE, TABLES_DIAGNOSTIC_DIR, TABLES_MODEL_DIR
from .modeling import winsorize_series
from .research_models import two_way_demean
from .visualization import save_figure, setup_style

LOGGER = logging.getLogger(__name__)

OUTCOME = "quality_efficiency_index"
POLICY_YEAR = 2017
POST_START_YEAR = 2018
BASE_EVENT_YEAR = 2016
PRE_EXPOSURE_END_YEAR = 2016


def _cluster_fit(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    return sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})


def _result_row(model: str, fit, variable: str, data: pd.DataFrame, purpose: str) -> dict[str, object]:
    return {
        "model": model,
        "purpose": purpose,
        "variable": variable,
        "coef": fit.params.get(variable, np.nan),
        "std_err": fit.bse.get(variable, np.nan),
        "t_value": fit.tvalues.get(variable, np.nan),
        "p_value": fit.pvalues.get(variable, np.nan),
        "nobs": int(fit.nobs),
        "firms": data["code"].nunique(),
        "years": data["year"].nunique(),
        "r_squared_within": fit.rsquared,
    }


def prepare_did_data(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")

    variables = [
        OUTCOME,
        "annual_ai_log",
        "ln_assets",
        "leverage",
        "rd_intensity",
        "firm_age",
        "revenue_growth",
    ]
    for col in variables:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        data[f"{col}_w"] = winsorize_series(data[col])

    data["rd_missing"] = data["rd_intensity"].isna().astype(int)
    data["rd_intensity_imputed"] = data["rd_intensity"].fillna(0)
    data["rd_intensity_imputed_w"] = winsorize_series(data["rd_intensity_imputed"])

    pre = data[data["year"].le(PRE_EXPOSURE_END_YEAR)].copy()
    exposure = (
        pre.groupby("industry_2digit", dropna=False)
        .agg(pre_ai_exposure=("annual_ai_log", "mean"), pre_obs=("annual_ai_log", "count"))
        .reset_index()
    )
    valid_exposure = exposure.loc[exposure["pre_obs"] >= 30, "pre_ai_exposure"]
    threshold = valid_exposure.median()
    exposure["high_pre_ai_industry"] = (
        (exposure["pre_ai_exposure"] >= threshold) & (exposure["pre_obs"] >= 30)
    ).astype(int)
    data = data.merge(exposure, on="industry_2digit", how="left")
    data["high_pre_ai_industry"] = data["high_pre_ai_industry"].fillna(0).astype(int)
    data["post_ai_policy"] = data["year"].ge(POST_START_YEAR).astype(int)
    data["did_policy_exposure"] = data["high_pre_ai_industry"] * data["post_ai_policy"]
    return data


def run_policy_exposure_did(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_did_data(panel)
    specs = [
        (
            "DID1_no_rd_controls",
            ["did_policy_exposure", "ln_assets_w", "leverage_w", "firm_age_w", "revenue_growth_w"],
            "政策暴露DID：不纳入研发强度，缓解研发缺失导致的样本选择",
        ),
        (
            "DID2_with_rd_complete",
            [
                "did_policy_exposure",
                "ln_assets_w",
                "leverage_w",
                "rd_intensity_w",
                "firm_age_w",
                "revenue_growth_w",
            ],
            "政策暴露DID：完整研发强度样本",
        ),
        (
            "DID3_rd_missing_imputed",
            [
                "did_policy_exposure",
                "ln_assets_w",
                "leverage_w",
                "rd_intensity_imputed_w",
                "rd_missing",
                "firm_age_w",
                "revenue_growth_w",
            ],
            "政策暴露DID：研发缺失填0并加入缺失虚拟变量",
        ),
    ]
    rows = []
    for name, variables, purpose in specs:
        needed = [f"{OUTCOME}_w", "code", "year"] + variables
        model_data = data.dropna(subset=needed).copy()
        dm = two_way_demean(model_data, [f"{OUTCOME}_w"] + variables)
        fit = _cluster_fit(dm[f"{OUTCOME}_w"], dm[variables], model_data["code"])
        rows.append(_result_row(name, fit, "did_policy_exposure", model_data, purpose))
        LOGGER.info("Fitted %s with %s observations", name, len(model_data))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "did_policy_exposure_results.csv", index=False, encoding="utf-8-sig")
    return table


def run_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_did_data(panel)
    controls = ["ln_assets_w", "leverage_w", "firm_age_w", "revenue_growth_w"]
    event_terms = []
    for year in sorted(data["year"].dropna().astype(int).unique()):
        if year == BASE_EVENT_YEAR:
            continue
        col = f"event_{year}"
        data[col] = data["high_pre_ai_industry"] * data["year"].eq(year).astype(int)
        event_terms.append(col)
    variables = event_terms + controls
    needed = [f"{OUTCOME}_w", "code", "year"] + variables
    model_data = data.dropna(subset=needed).copy()
    dm = two_way_demean(model_data, [f"{OUTCOME}_w"] + variables)
    fit = _cluster_fit(dm[f"{OUTCOME}_w"], dm[variables], model_data["code"])

    rows = []
    for col in event_terms:
        year = int(col.replace("event_", ""))
        rows.append(
            {
                "model": "policy_exposure_event_study",
                "base_year": BASE_EVENT_YEAR,
                "year": year,
                "event_time": year - POLICY_YEAR,
                "period": "pre" if year < POLICY_YEAR else ("policy_year" if year == POLICY_YEAR else "post"),
                "coef": fit.params.get(col, np.nan),
                "std_err": fit.bse.get(col, np.nan),
                "t_value": fit.tvalues.get(col, np.nan),
                "p_value": fit.pvalues.get(col, np.nan),
                "nobs": int(fit.nobs),
                "firms": model_data["code"].nunique(),
            }
        )
    table = pd.DataFrame(rows).sort_values("year")

    pre_terms = [col for col in event_terms if int(col.replace("event_", "")) < BASE_EVENT_YEAR]
    if pre_terms:
        hypothesis = " = 0, ".join(pre_terms) + " = 0"
        ftest = fit.f_test(hypothesis)
        table.attrs["pretrend_f_stat"] = float(ftest.fvalue)
        table.attrs["pretrend_p_value"] = float(ftest.pvalue)
    table.to_csv(TABLES_MODEL_DIR / "did_policy_event_study_results.csv", index=False, encoding="utf-8-sig")
    return table


def run_pretrend_placebo(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_did_data(panel)
    data = data[data["year"].between(2013, 2016)].copy()
    data["placebo_post_2016"] = data["year"].ge(2016).astype(int)
    data["placebo_did_2016"] = data["high_pre_ai_industry"] * data["placebo_post_2016"]
    variables = ["placebo_did_2016", "ln_assets_w", "leverage_w", "firm_age_w", "revenue_growth_w"]
    needed = [f"{OUTCOME}_w", "code", "year"] + variables
    model_data = data.dropna(subset=needed).copy()
    dm = two_way_demean(model_data, [f"{OUTCOME}_w"] + variables)
    fit = _cluster_fit(dm[f"{OUTCOME}_w"], dm[variables], model_data["code"])
    table = pd.DataFrame(
        [_result_row("DID_placebo_pre_2016", fit, "placebo_did_2016", model_data, "政策前安慰剂：仅使用2013-2016年")]
    )
    table.to_csv(TABLES_MODEL_DIR / "did_pretrend_placebo_results.csv", index=False, encoding="utf-8-sig")
    return table


def save_event_study_plot(event_table: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    table = event_table.sort_values("year").copy()
    x = table["event_time"]
    y = table["coef"]
    err = 1.96 * table["std_err"]
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.axvline(0, color="#b33f3f", linewidth=0.8, linestyle="--")
    ax.errorbar(x, y, yerr=err, fmt="o-", color="#2f6f8f", ecolor="#7aa6bb", capsize=3, linewidth=1.2)
    ax.set_xlabel("相对政策年份")
    ax.set_ylabel("处理组相对变化系数")
    ax.set_title("政策暴露DID事件研究")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig8_did_policy_event_study.png")


def build_identification_assessment(
    did_table: pd.DataFrame, event_table: pd.DataFrame, placebo_table: pd.DataFrame
) -> pd.DataFrame:
    pretrend_p = event_table.attrs.get("pretrend_p_value", np.nan)
    placebo_p = float(placebo_table.loc[0, "p_value"]) if len(placebo_table) else np.nan
    main_p = float(did_table.loc[did_table["model"].eq("DID1_no_rd_controls"), "p_value"].iloc[0])
    main_coef = float(did_table.loc[did_table["model"].eq("DID1_no_rd_controls"), "coef"].iloc[0])
    conclusion = "不建议用于因果主结论"
    if main_coef > 0 and main_p < 0.1 and pretrend_p > 0.1 and placebo_p > 0.1:
        conclusion = "可作为谨慎的补充因果证据"
    rows = [
        {
            "item": "policy_shock",
            "assessment": "2017年国家人工智能规划可作为统一时间冲击，但处理强度来自政策前行业AI暴露而非随机分配",
            "passed": True,
        },
        {
            "item": "parallel_trend_joint_test",
            "assessment": f"事件研究政策前联合检验p值={pretrend_p:.4f}",
            "passed": bool(pretrend_p > 0.1) if pd.notna(pretrend_p) else False,
        },
        {
            "item": "pre_period_placebo",
            "assessment": f"2016年前置安慰剂p值={placebo_p:.4f}",
            "passed": bool(placebo_p > 0.1) if pd.notna(placebo_p) else False,
        },
        {
            "item": "main_effect",
            "assessment": f"主DID系数={main_coef:.4f}, p值={main_p:.4f}",
            "passed": bool(main_coef > 0 and main_p < 0.1),
        },
        {"item": "overall", "assessment": conclusion, "passed": conclusion != "不建议用于因果主结论"},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIAGNOSTIC_DIR / "did_identification_assessment.csv", index=False, encoding="utf-8-sig")
    return table


def run_all_did_checks(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    np.random.seed(RANDOM_STATE)
    did = run_policy_exposure_did(panel)
    event = run_event_study(panel)
    placebo = run_pretrend_placebo(panel)
    save_event_study_plot(event)
    assessment = build_identification_assessment(did, event, placebo)
    return {"did": did, "event": event, "placebo": placebo, "assessment": assessment}
