from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TABLES_FINAL_DIR, TABLES_MODEL_DIR


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def fmt_num(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(number):
        return ""
    return f"{number:.{digits}f}"


VARIABLE_LABELS = {
    "quality_efficiency_index_w": "提质增效指数",
    "quality_efficiency_entropy_w": "提质增效指数(熵权)",
    "quality_efficiency_pca_w": "提质增效指数(PCA)",
    "quality_subindex_w": "质量子指数",
    "efficiency_subindex_w": "效率子指数",
    "tfp_w": "TFP",
    "ai_actual_application_index_w": "实际AI应用指数",
    "ai_conversion_efficiency_w": "AI落地转化效率",
    "ai_word_high_conversion": "AI词频×高转化效率",
    "high_conversion_efficiency": "高转化效率",
    "annual_ai_log_centered": "AI词频一次项",
    "annual_ai_log_centered_sq": "AI词频二次项",
    "asset_turnover_w": "资产周转率",
    "labor_productivity_log_w": "劳动生产率",
    "annual_ai_log_w": "年报AI词频",
    "mda_ai_log_w": "MD&A AI词频",
    "ai_application_index_w": "综合AI应用指数",
}


MODEL_LABELS = {
    "M0_fe_no_controls": "固定效应基线",
    "M1_main_fe": "主模型",
    "M2_mda_frequency": "替换X：MD&A词频",
    "M3_ai_composite": "替换X：综合AI指数",
    "M6_tfp_y": "替换Y：TFP",
    "M7_entropy_index_y": "替换Y：熵权指数",
    "M8_pca_index_y": "替换Y：PCA指数",
    "M9_rd_missing_robust": "研发缺失处理",
    "ME1_actual_ai_application": "实际AI应用",
    "ME2_asset_turnover": "资产周转率",
    "ME3_labor_productivity": "劳动生产率",
    "ME4_tfp": "TFP",
    "ME5_efficiency_subindex": "效率子指数",
    "ME6_quality_subindex": "质量子指数",
    "ME7_conversion_to_quality_efficiency": "AI落地转化效率",
    "annual_ai_log_w": "低转化效率组AI词频",
    "ai_word_high_conversion": "AI词频×高转化效率",
    "high_conversion_efficiency": "高转化效率",
    "state_owned": "国企",
    "non_state_owned": "非国企",
    "east_region": "东部地区",
    "non_east_region": "非东部地区",
    "high_tech_manufacturing": "高技术制造业",
    "non_high_tech_manufacturing": "非高技术制造业",
    "high_conversion_efficiency": "高转化效率",
    "low_conversion_efficiency": "低转化效率",
    "FY1_main_x": "企业FE：主模型",
    "FY2_mda_x": "企业FE：MD&A词频",
    "FY3_composite_x": "企业FE：综合AI指数",
    "FY4_tfp_y": "企业FE：TFP",
}


def label(value: object, mapping: dict[str, str]) -> str:
    text = str(value)
    return mapping.get(text, text)


def format_regression_table(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    table = df[df["model"].isin(model_order)].copy()
    table["model"] = pd.Categorical(table["model"], categories=model_order, ordered=True)
    table = table.sort_values("model")
    rows = []
    for _, row in table.iterrows():
        stars = significance_stars(float(row["p_value"]))
        fit_index = row.get("adj_r_squared", np.nan)
        if pd.isna(fit_index):
            fit_index = row.get("within_r_squared", np.nan)
        if pd.isna(fit_index):
            fit_index = row.get("pseudo_r_squared", np.nan)
        if pd.isna(fit_index):
            fit_index = row.get("r_squared", np.nan)
        rows.append(
            {
                "模型": label(row["model"], MODEL_LABELS),
                "被解释变量": label(row["y"], VARIABLE_LABELS),
                "核心解释变量": label(row["x"], VARIABLE_LABELS),
                "系数": f"{fmt_num(row['coef'])}{stars}",
                "标准误": f"({fmt_num(row['std_err'])})",
                "t值": fmt_num(row["t_value"]),
                "p值": fmt_num(row["p_value"]),
                "样本量": int(float(row["nobs"])),
                "拟合指标": fmt_num(fit_index),
                "说明": row.get("purpose", ""),
            }
        )
    return pd.DataFrame(rows)


def build_paper_tables() -> None:
    final = pd.read_csv(TABLES_MODEL_DIR / "final_fixed_effects_research_results.csv")
    mechanism = pd.read_csv(TABLES_MODEL_DIR / "mechanism_regression_results.csv")
    heterogeneity = pd.read_csv(TABLES_MODEL_DIR / "heterogeneity_regression_results.csv")
    boundary = pd.read_csv(TABLES_MODEL_DIR / "firm_year_fe_boundary_tests.csv")
    conversion = pd.read_csv(TABLES_MODEL_DIR / "conversion_moderation_results.csv")
    quantile = pd.read_csv(TABLES_MODEL_DIR / "quantile_fe_results.csv")
    nonlinear = pd.read_csv(TABLES_MODEL_DIR / "nonlinear_ai_effect_results.csv")
    partial_linear = pd.read_csv(TABLES_MODEL_DIR / "cross_fitted_partial_linear_results.csv")

    main_table = format_regression_table(
        final,
        ["M0_fe_no_controls", "M1_main_fe", "M2_mda_frequency", "M3_ai_composite"],
    )
    robustness_table = format_regression_table(
        final,
        ["M6_tfp_y", "M7_entropy_index_y", "M8_pca_index_y", "M9_rd_missing_robust"],
    )
    mechanism_table = format_regression_table(
        mechanism,
        [
            "ME1_actual_ai_application",
            "ME2_asset_turnover",
            "ME3_labor_productivity",
            "ME4_tfp",
            "ME5_efficiency_subindex",
            "ME6_quality_subindex",
            "ME7_conversion_to_quality_efficiency",
        ],
    )
    heterogeneity_table = format_regression_table(heterogeneity, list(heterogeneity["model"]))
    boundary_table = format_regression_table(boundary, list(boundary["model"]))

    conversion_table = conversion.copy()
    conversion_table["model"] = conversion_table["x"]
    conversion_table = format_regression_table(conversion_table, list(conversion_table["model"]))

    quantile_table = quantile.copy()
    quantile_table["model"] = quantile_table["model"] + "_" + quantile_table["quantile"].astype(str)
    quantile_table["model"] = quantile_table["model"].replace(
        {
            "quantile_fe_0.25": "分位数25%",
            "quantile_fe_0.5": "分位数50%",
            "quantile_fe_0.75": "分位数75%",
        }
    )
    advanced = pd.concat([quantile_table, nonlinear, partial_linear], ignore_index=True)
    advanced.loc[advanced["term"] == "AI词频一次项", "model"] = "非线性一次项"
    advanced.loc[advanced["term"] == "AI词频二次项", "model"] = "非线性二次项"
    advanced["model"] = [
        f"{model}_{idx}" if list(advanced["model"]).count(model) > 1 else model
        for idx, model in enumerate(advanced["model"], start=1)
    ]
    advanced_table = format_regression_table(advanced, list(advanced["model"]))

    outputs = {
        "paper_table_1_main_regression.csv": main_table,
        "paper_table_2_robustness.csv": robustness_table,
        "paper_table_3_mechanism.csv": mechanism_table,
        "paper_table_4_heterogeneity.csv": heterogeneity_table,
        "paper_table_A1_boundary.csv": boundary_table,
        "paper_table_5_conversion_moderation.csv": conversion_table,
        "paper_table_6_advanced_models.csv": advanced_table,
    }
    for filename, table in outputs.items():
        table.to_csv(TABLES_FINAL_DIR / filename, index=False, encoding="utf-8-sig")
