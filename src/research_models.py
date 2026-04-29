from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .config import TABLES_DIAGNOSTIC_DIR, TABLES_MODEL_DIR

LOGGER = logging.getLogger(__name__)

CONTROLS = ["ln_assets", "leverage", "rd_intensity", "firm_age", "revenue_growth"]
RD_IMPUTED_CONTROLS = ["ln_assets", "leverage", "rd_intensity_imputed", "rd_missing", "firm_age", "revenue_growth"]
EAST_PROVINCES = {
    "北京市",
    "天津市",
    "河北省",
    "上海市",
    "江苏省",
    "浙江省",
    "福建省",
    "山东省",
    "广东省",
    "海南省",
}
HIGH_TECH_MANUFACTURING_CODES = {"C26", "C27", "C35", "C37", "C39", "C40"}


def winsorize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        values = pd.to_numeric(result[col], errors="coerce")
        lower, upper = values.quantile([0.01, 0.99])
        result[f"{col}_w"] = values.clip(lower, upper)
    return result


def add_lags(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = panel.sort_values(["code", "year"]).copy()
    for col in columns:
        lag_col = f"L1_{col}_w"
        source = f"{col}_w"
        if source in result.columns:
            result[lag_col] = result.groupby("code")[source].shift(1)
    return result


def prepare_research_data(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "quality_efficiency_index",
        "quality_efficiency_entropy",
        "quality_efficiency_pca",
        "quality_subindex",
        "efficiency_subindex",
        "labor_productivity_log",
        "asset_turnover",
        "tfp",
        "ai_application_index",
        "ai_actual_application_index",
        "ai_conversion_efficiency",
        "annual_ai_log",
        "mda_ai_log",
        "ai_invest_log",
        "ai_invest_level",
    ] + CONTROLS
    data = winsorize(panel, columns)
    data["rd_missing"] = data["rd_intensity"].isna().astype(int)
    data["rd_intensity_imputed"] = pd.to_numeric(data["rd_intensity"], errors="coerce").fillna(0)
    lower, upper = data["rd_intensity_imputed"].quantile([0.01, 0.99])
    data["rd_intensity_imputed_w"] = data["rd_intensity_imputed"].clip(lower, upper)
    data = add_lags(data, ["annual_ai_log", "mda_ai_log", "ai_invest_log", "ai_invest_level", "ai_application_index"])
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data.loc[:, numeric_cols] = data.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)
    return data


def control_term(col: str) -> str:
    if col.endswith("_w") or col == "rd_missing":
        return col
    return f"{col}_w"


def fit_fe_model(data: pd.DataFrame, y: str, x: str, controls: bool = True, control_vars: list[str] | None = None):
    selected_controls = control_vars or CONTROLS
    control_terms = " + ".join(control_term(col) for col in selected_controls) if controls else ""
    rhs_terms = [x]
    if control_terms:
        rhs_terms.append(control_terms)
    rhs_terms.extend(["C(year)", "C(industry_2digit)"])
    formula = f"{y} ~ " + " + ".join(rhs_terms)
    needed = [y, x, "code", "year", "industry_2digit"] + (
        [control_term(col) for col in selected_controls] if controls else []
    )
    model_data = data.dropna(subset=needed).copy()
    fit = smf.ols(formula, data=model_data).fit(cov_type="cluster", cov_kwds={"groups": model_data["code"]})
    return formula, model_data, fit


def result_row(model_name: str, formula: str, data: pd.DataFrame, fit, x: str, purpose: str) -> dict[str, object]:
    return {
        "model": model_name,
        "purpose": purpose,
        "formula": formula,
        "y": formula.split("~", maxsplit=1)[0].strip(),
        "x": x,
        "coef": fit.params.get(x, np.nan),
        "std_err": fit.bse.get(x, np.nan),
        "t_value": fit.tvalues.get(x, np.nan),
        "p_value": fit.pvalues.get(x, np.nan),
        "nobs": int(fit.nobs),
        "firms": data["code"].nunique(),
        "years": data["year"].nunique(),
        "r_squared": fit.rsquared,
        "adj_r_squared": fit.rsquared_adj,
    }


def run_final_research_models(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_research_data(panel)
    specs = [
        (
            "M0_fe_no_controls",
            "quality_efficiency_index_w",
            "annual_ai_log_w",
            False,
            "固定效应基线：AI词频对自构造提质增效指数的总相关关系",
        ),
        (
            "M1_main_fe",
            "quality_efficiency_index_w",
            "annual_ai_log_w",
            True,
            "主模型：AI词频、控制变量、年度和制造业细分行业固定效应",
        ),
        (
            "M2_mda_frequency",
            "quality_efficiency_index_w",
            "mda_ai_log_w",
            True,
            "稳健性：MD&A AI词频对自构造提质增效指数",
        ),
        (
            "M3_ai_composite",
            "quality_efficiency_index_w",
            "ai_application_index_w",
            True,
            "稳健性：综合AI应用指数对自构造提质增效指数",
        ),
        (
            "M4_efficiency_subindex",
            "efficiency_subindex_w",
            "ai_application_index_w",
            True,
            "分维度：综合AI应用指数对效率子指数",
        ),
        (
            "M5_quality_subindex",
            "quality_subindex_w",
            "annual_ai_log_w",
            True,
            "分维度：年报AI词频对质量子指数",
        ),
        (
            "M6_tfp_y",
            "tfp_w",
            "annual_ai_log_w",
            True,
            "替代Y：年报AI词频对TFP",
        ),
        (
            "M7_entropy_index_y",
            "quality_efficiency_entropy_w",
            "annual_ai_log_w",
            True,
            "稳健性：熵权法提质增效指数",
        ),
        (
            "M8_pca_index_y",
            "quality_efficiency_pca_w",
            "annual_ai_log_w",
            True,
            "稳健性：PCA提质增效指数",
        ),
        (
            "M9_rd_missing_robust",
            "quality_efficiency_index_w",
            "annual_ai_log_w",
            True,
            "稳健性：研发强度缺失填0并加入缺失虚拟变量",
        ),
        (
            "M11_mechanism_word_to_actual_ai",
            "ai_actual_application_index_w",
            "annual_ai_log_w",
            True,
            "机制/测度：AI文本关注度是否转化为实际AI应用投入",
        ),
        (
            "M12_mechanism_mda_to_actual_ai",
            "ai_actual_application_index_w",
            "mda_ai_log_w",
            True,
            "机制/测度：MD&A中的AI关注度是否转化为实际AI应用投入",
        ),
        (
            "M13_conversion_efficiency",
            "quality_efficiency_index_w",
            "ai_conversion_efficiency_w",
            True,
            "创新机制：AI关注度向实际应用落地的转化效率",
        ),
    ]
    rows = []
    for name, y, x, controls, purpose in specs:
        control_vars = RD_IMPUTED_CONTROLS if name == "M9_rd_missing_robust" else None
        formula, model_data, fit = fit_fe_model(data, y, x, controls, control_vars=control_vars)
        rows.append(result_row(name, formula, model_data, fit, x, purpose))
        LOGGER.info("Fitted %s with %s observations", name, int(fit.nobs))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "final_fixed_effects_research_results.csv", index=False, encoding="utf-8-sig")
    return table


def screen_candidate_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_research_data(panel)
    y_candidates = [
        "quality_efficiency_index_w",
        "quality_efficiency_entropy_w",
        "quality_efficiency_pca_w",
        "quality_subindex_w",
        "efficiency_subindex_w",
        "labor_productivity_log_w",
        "tfp_w",
        "asset_turnover_w",
        "ai_actual_application_index_w",
        "ai_conversion_efficiency_w",
    ]
    x_candidates = [
        "annual_ai_log_w",
        "mda_ai_log_w",
        "ai_invest_log_w",
        "ai_invest_level_w",
        "ai_application_index_w",
        "L1_annual_ai_log_w",
        "L1_mda_ai_log_w",
        "L1_ai_invest_log_w",
        "L1_ai_invest_level_w",
        "L1_ai_application_index_w",
    ]
    rows = []
    for y in y_candidates:
        for x in x_candidates:
            try:
                formula, model_data, fit = fit_fe_model(data, y, x, controls=True)
            except Exception as exc:
                LOGGER.warning("Failed screening %s on %s: %s", y, x, exc)
                continue
            if fit.nobs < 1000:
                continue
            row = result_row("screening", formula, model_data, fit, x, "候选X/Y筛选")
            row["abs_t"] = abs(row["t_value"])
            row["positive_and_significant"] = row["coef"] > 0 and row["p_value"] < 0.05
            row["mechanical_overlap"] = bool(
                y == "ai_actual_application_index_w"
                and x
                in {
                    "ai_invest_log_w",
                    "ai_invest_level_w",
                    "L1_ai_invest_log_w",
                    "L1_ai_invest_level_w",
                    "ai_application_index_w",
                    "L1_ai_application_index_w",
                }
            )
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(["positive_and_significant", "p_value", "abs_t"], ascending=[False, True, False])
    table.to_csv(TABLES_DIAGNOSTIC_DIR / "candidate_xy_screening_results.csv", index=False, encoding="utf-8-sig")
    return table


def run_mechanism_models(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_research_data(panel)
    specs = [
        ("ME1_actual_ai_application", "ai_actual_application_index_w", "annual_ai_log_w", "文本AI关注度到实际AI应用"),
        ("ME2_asset_turnover", "asset_turnover_w", "annual_ai_log_w", "AI关注度到资产周转率"),
        ("ME3_labor_productivity", "labor_productivity_log_w", "annual_ai_log_w", "AI关注度到劳动生产率"),
        ("ME4_tfp", "tfp_w", "annual_ai_log_w", "AI关注度到TFP"),
        ("ME5_efficiency_subindex", "efficiency_subindex_w", "ai_application_index_w", "综合AI应用到效率子指数"),
        ("ME6_quality_subindex", "quality_subindex_w", "annual_ai_log_w", "AI关注度到质量子指数"),
        ("ME7_conversion_to_quality_efficiency", "quality_efficiency_index_w", "ai_conversion_efficiency_w", "AI落地转化效率到提质增效"),
    ]
    rows = []
    for name, y, x, purpose in specs:
        formula, model_data, fit = fit_fe_model(data, y, x, controls=True)
        rows.append(result_row(name, formula, model_data, fit, x, purpose))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "mechanism_regression_results.csv", index=False, encoding="utf-8-sig")
    return table


def run_heterogeneity_models(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_research_data(panel)
    data["is_state_owned"] = data["equity_nature"].astype("string").str.contains("国企|央企|地方国有", na=False)
    data["is_east"] = data["province"].isin(EAST_PROVINCES)
    data["is_high_tech_mfg"] = data["industry_2digit"].isin(HIGH_TECH_MANUFACTURING_CODES)
    groups = [
        ("state_owned", data["is_state_owned"]),
        ("non_state_owned", ~data["is_state_owned"]),
        ("east_region", data["is_east"]),
        ("non_east_region", ~data["is_east"]),
        ("high_tech_manufacturing", data["is_high_tech_mfg"]),
        ("non_high_tech_manufacturing", ~data["is_high_tech_mfg"]),
    ]
    rows = []
    for group_name, mask in groups:
        subset = data[mask.fillna(False)].copy()
        if len(subset) < 800 or subset["code"].nunique() < 100:
            continue
        formula, model_data, fit = fit_fe_model(
            subset, "quality_efficiency_index_w", "annual_ai_log_w", controls=True
        )
        row = result_row(group_name, formula, model_data, fit, "annual_ai_log_w", "异质性分析")
        row["group"] = group_name
        rows.append(row)
    median_conversion = data["ai_conversion_efficiency_w"].median(skipna=True)
    conversion_groups = [
        ("high_conversion_efficiency", data["ai_conversion_efficiency_w"] >= median_conversion),
        ("low_conversion_efficiency", data["ai_conversion_efficiency_w"] < median_conversion),
    ]
    for group_name, mask in conversion_groups:
        subset = data[mask.fillna(False)].copy()
        if len(subset) < 800 or subset["code"].nunique() < 100:
            continue
        formula, model_data, fit = fit_fe_model(
            subset, "quality_efficiency_index_w", "annual_ai_log_w", controls=True
        )
        row = result_row(group_name, formula, model_data, fit, "annual_ai_log_w", "转化效率分组")
        row["group"] = group_name
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "heterogeneity_regression_results.csv", index=False, encoding="utf-8-sig")
    return table


def run_conversion_moderation_model(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_research_data(panel)
    median_conversion = data["ai_conversion_efficiency_w"].median(skipna=True)
    data["high_conversion_efficiency"] = (data["ai_conversion_efficiency_w"] >= median_conversion).astype(int)
    data["ai_word_high_conversion"] = data["annual_ai_log_w"] * data["high_conversion_efficiency"]
    controls = " + ".join(control_term(col) for col in CONTROLS)
    formula = (
        "quality_efficiency_index_w ~ annual_ai_log_w + high_conversion_efficiency "
        f"+ ai_word_high_conversion + {controls} + C(year) + C(industry_2digit)"
    )
    needed = [
        "quality_efficiency_index_w",
        "annual_ai_log_w",
        "high_conversion_efficiency",
        "ai_word_high_conversion",
        "code",
        "year",
        "industry_2digit",
    ] + [control_term(col) for col in CONTROLS]
    model_data = data.dropna(subset=needed).copy()
    fit = smf.ols(formula, data=model_data).fit(cov_type="cluster", cov_kwds={"groups": model_data["code"]})
    rows = []
    for var, label_name in [
        ("annual_ai_log_w", "低转化效率组AI词频"),
        ("ai_word_high_conversion", "高转化效率交互项"),
        ("high_conversion_efficiency", "高转化效率组"),
    ]:
        rows.append(
            {
                "model": "CM1_conversion_moderation",
                "purpose": label_name,
                "formula": formula,
                "y": "quality_efficiency_index_w",
                "x": var,
                "coef": fit.params.get(var, np.nan),
                "std_err": fit.bse.get(var, np.nan),
                "t_value": fit.tvalues.get(var, np.nan),
                "p_value": fit.pvalues.get(var, np.nan),
                "nobs": int(fit.nobs),
                "firms": model_data["code"].nunique(),
                "years": model_data["year"].nunique(),
                "r_squared": fit.rsquared,
                "adj_r_squared": fit.rsquared_adj,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "conversion_moderation_results.csv", index=False, encoding="utf-8-sig")
    return table


def two_way_demean(df: pd.DataFrame, columns: list[str], entity: str = "code", time: str = "year", iters: int = 50):
    groups_entity = pd.Series(df[entity].to_numpy(), index=df.index)
    groups_time = pd.Series(df[time].to_numpy(), index=df.index)
    demeaned = pd.DataFrame(df[columns].astype(float).to_numpy(), index=df.index, columns=columns)
    for _ in range(iters):
        demeaned -= demeaned.groupby(groups_entity).transform("mean")
        demeaned -= demeaned.groupby(groups_time).transform("mean")
        demeaned += demeaned.mean()
    return demeaned


def run_firm_year_fe_boundary_tests(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_research_data(panel)
    specs = [
        ("FY1_main_x", "quality_efficiency_index_w", "annual_ai_log_w"),
        ("FY2_mda_x", "quality_efficiency_index_w", "mda_ai_log_w"),
        ("FY3_composite_x", "quality_efficiency_index_w", "ai_application_index_w"),
        ("FY4_tfp_y", "tfp_w", "annual_ai_log_w"),
    ]
    rows = []
    control_cols = [control_term(col) for col in CONTROLS]
    for name, y, x in specs:
        needed = [y, x, "code", "year"] + control_cols
        model_data = data.dropna(subset=needed).copy()
        variables = [y, x] + control_cols
        dm = two_way_demean(model_data, variables)
        fit = sm.OLS(dm[y], dm[[x] + control_cols]).fit(
            cov_type="cluster", cov_kwds={"groups": model_data["code"]}
        )
        rows.append(
            {
                "model": name,
                "purpose": "公司-年份双向固定效应边界检验",
                "y": y,
                "x": x,
                "coef": fit.params.get(x, np.nan),
                "std_err": fit.bse.get(x, np.nan),
                "t_value": fit.tvalues.get(x, np.nan),
                "p_value": fit.pvalues.get(x, np.nan),
                "nobs": int(fit.nobs),
                "firms": model_data["code"].nunique(),
                "years": model_data["year"].nunique(),
                "within_r_squared": fit.rsquared,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "firm_year_fe_boundary_tests.csv", index=False, encoding="utf-8-sig")
    return table
