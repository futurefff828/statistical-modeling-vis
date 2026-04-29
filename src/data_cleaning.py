from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR, RANDOM_STATE, RAW_FILES, TABLES_DIAGNOSTIC_DIR


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanedPanels:
    ai_word: pd.DataFrame
    ai_investment: pd.DataFrame
    controls: pd.DataFrame
    rd_investment: pd.DataFrame
    tfp: pd.DataFrame
    y_candidates: pd.DataFrame
    modeling_panel: pd.DataFrame


def read_structured_excel(path: Path) -> pd.DataFrame:
    """Read Excel files where row 1 and row 2 store labels and units."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(path, skiprows=[1, 2])
    return df.dropna(how="all").copy()


def normalize_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return f"{int(float(text)):06d}"
    except ValueError:
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(6) if digits else None


def extract_year(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(int(value))
    dt = pd.to_datetime(value, errors="coerce")
    if pd.notna(dt):
        return float(dt.year)
    text = str(value)
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    for item in digits:
        if len(item) >= 4:
            return float(int(item[:4]))
    return np.nan


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def safe_log1p(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.log1p(values.where(values > -1))


def add_key_columns(df: pd.DataFrame, code_col: str, date_col: str = "EndDate") -> pd.DataFrame:
    result = df.copy()
    result["code"] = result[code_col].map(normalize_code)
    result["year"] = result[date_col].map(extract_year)
    result = result.dropna(subset=["code", "year"])
    result["year"] = result["year"].astype(int)
    return result


def is_manufacturing(df: pd.DataFrame) -> pd.Series:
    code = df["industry_code"].astype("string")
    return code.str.startswith("C", na=False)


def is_st_or_pt(name: pd.Series) -> pd.Series:
    text = name.astype("string").fillna("")
    return text.str.contains("ST|PT", case=False, regex=True)


def filter_to_manufacturing_universe(df: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    """Keep only firm-year observations that are in the verified manufacturing controls panel."""
    keys = controls[["code", "year"]].drop_duplicates()
    return df.merge(keys, on=["code", "year"], how="inner")


def clean_ai_word() -> pd.DataFrame:
    raw = read_structured_excel(RAW_FILES["ai_word"])
    df = add_key_columns(raw, "Symbol")
    rename = {
        "ShortName": "firm_name",
        "AnnualReportAIWordFre": "annual_ai_word_freq",
        "AnnualReportAISentenceFre": "annual_ai_sentence_freq",
        "AnnualReportSentence": "annual_sentence_count",
        "AnnualReportWord": "annual_word_count",
        "AnnualReportWordNum": "annual_text_chars",
        "MDAAIWordFre": "mda_ai_word_freq",
        "MDAAIAISentenceFre": "mda_ai_sentence_freq",
        "MDASentence": "mda_sentence_count",
        "MDAWord": "mda_word_count",
        "MDAWordNum": "mda_text_chars",
    }
    df = df.rename(columns=rename)
    keep = ["code", "year", "firm_name"] + [v for v in rename.values() if v != "firm_name"]
    numeric = [c for c in keep if c not in ["code", "year", "firm_name"]]
    df = coerce_numeric(df[keep], numeric)
    df["annual_ai_log"] = safe_log1p(df["annual_ai_word_freq"])
    df["mda_ai_log"] = safe_log1p(df["mda_ai_word_freq"])
    df["annual_ai_word_share"] = df["annual_ai_word_freq"] / df["annual_word_count"].replace(0, np.nan)
    df["mda_ai_word_share"] = df["mda_ai_word_freq"] / df["mda_word_count"].replace(0, np.nan)
    return df


def clean_ai_investment() -> pd.DataFrame:
    raw = read_structured_excel(RAW_FILES["ai_investment"])
    df = add_key_columns(raw, "Symbol")
    rename = {
        "ShortName": "firm_name",
        "Category": "report_category",
        "AISoftInvest": "ai_soft_invest",
        "AISoftInvestValueAdd": "ai_soft_invest_add",
        "AIHardInvest": "ai_hard_invest",
        "AIHardInvestValueAdd": "ai_hard_invest_add",
        "AIInvestTotal": "ai_invest_total",
        "AIInvestTotalValueAdd": "ai_invest_total_add",
        "AIInvestLevel": "ai_invest_level",
    }
    df = df.rename(columns=rename)
    keep = ["code", "year", "firm_name"] + [v for v in rename.values() if v != "firm_name"]
    numeric = [c for c in keep if c not in ["code", "year", "firm_name"]]
    df = coerce_numeric(df[keep], numeric)
    df["ai_invest_log"] = safe_log1p(df["ai_invest_total"])
    df["ai_invest_add_log"] = safe_log1p(df["ai_invest_total_add"])
    return df


def clean_controls() -> pd.DataFrame:
    raw = read_structured_excel(RAW_FILES["controls"])
    df = add_key_columns(raw, "code")
    rename = {
        "stknme": "firm_name",
        "listingDate": "listing_date",
        "STK_LISTEDCOINFOANL-IndustryNameD": "industry_name",
        "STK_LISTEDCOINFOANL-IndustryCodeD": "industry_code",
        "TMTLI_CGInfo-ProvinceName": "province",
        "BDT_ExcessiveDebt-PropertyRightsNature": "property_rights",
        "BDT_ExcessiveDebt-LargestHolderRate": "largest_holder_rate",
        "EN_EquityNatureAll-EquityNature": "equity_nature",
        "BDT_FinConstFC-ListingAge": "firm_age",
        "PT_LCMAINFIN-TotalAssets": "total_assets",
        "PT_LCMAINFIN-TotalLiability": "total_liability",
        "DEBT_LOANSTATIS-DebtToAssetratio": "debt_asset_ratio",
        "FI_T8-F081601B": "revenue_growth",
        "FS_Comins-B001209000": "selling_expense",
        "FS_Comins-B001210000": "admin_expense",
        "FS_Comins-B001211000": "finance_expense",
    }
    df = df.rename(columns=rename)
    keep = ["code", "year"] + list(rename.values())
    numeric = [
        "largest_holder_rate",
        "firm_age",
        "total_assets",
        "total_liability",
        "debt_asset_ratio",
        "revenue_growth",
        "selling_expense",
        "admin_expense",
        "finance_expense",
    ]
    df = coerce_numeric(df[keep], numeric)
    df = df[is_manufacturing(df)].copy()
    df = df[~is_st_or_pt(df["firm_name"])].copy()
    df["industry_2digit"] = df["industry_code"].astype("string").str[:3]
    df["ln_assets"] = np.log(df["total_assets"].where(df["total_assets"] > 0))
    if df["debt_asset_ratio"].dropna().median() > 1:
        df["leverage"] = df["debt_asset_ratio"] / 100
    else:
        df["leverage"] = df["debt_asset_ratio"]
    fallback = df["total_liability"] / df["total_assets"].replace(0, np.nan)
    df["leverage"] = df["leverage"].fillna(fallback)
    return df


def clean_rd_investment() -> pd.DataFrame:
    raw = read_structured_excel(RAW_FILES["rd_investment"])
    df = add_key_columns(raw, "code")
    rename = {
        "stknme": "firm_name",
        "listingDate": "listing_date",
        "PT_LCRDSPENDING-RDSpendSumRatio": "rd_investment_intensity_pct",
        "PT_LCRDSPENDING-RDSpendSum": "rd_investment_amount",
        "STK_LISTEDCOINFOANL-IndustryCodeD": "industry_code",
        "STK_LISTEDCOINFOANL-IndustryNameD": "industry_name",
    }
    df = df.rename(columns=rename)
    keep = ["code", "year"] + list(rename.values())
    numeric = [
        "rd_investment_intensity_pct",
        "rd_investment_amount",
    ]
    df = coerce_numeric(df[keep], numeric)
    df = df[is_manufacturing(df)].copy()
    df = df[~is_st_or_pt(df["firm_name"])].copy()
    return df


def clean_y_candidates() -> pd.DataFrame:
    raw = read_structured_excel(RAW_FILES["y_candidates"])
    df = add_key_columns(raw, "code")
    rename = {
        "stknme": "firm_name",
        "STK_LISTEDCOINFOANL-IndustryNameD": "industry_name",
        "STK_LISTEDCOINFOANL-IndustryCodeD": "industry_code",
        "FS_Comins-B001101000": "revenue",
        "CG_Ybasic-Y0601b": "employees",
        "FI_T5-F050201B": "roa",
        "FI_T5-F050501B": "roe",
        "FI_T5-F051401B": "operating_margin",
        "FI_T4-F041701B": "asset_turnover",
    }
    df = df.rename(columns=rename)
    keep = ["code", "year"] + list(rename.values())
    numeric = [c for c in keep if c not in ["code", "year", "firm_name", "industry_name", "industry_code"]]
    df = coerce_numeric(df[keep], numeric)
    df = df[is_manufacturing(df)].copy()
    df = df[~is_st_or_pt(df["firm_name"])].copy()
    return df


def clean_tfp_inputs() -> pd.DataFrame:
    raw = read_structured_excel(RAW_FILES["tfp_inputs"])
    df = add_key_columns(raw, "code")
    rename = {
        "stknme": "firm_name",
        "STK_LISTEDCOINFOANL-IndustryNameD": "industry_name",
        "STK_LISTEDCOINFOANL-IndustryCodeD": "industry_code",
        "FS_Comins-B001101000": "revenue",
        "FS_Comins-B001201000": "operating_cost",
        "CG_Ybasic-Y0601b": "employees",
        "FS_Combas-A001212000": "fixed_assets_net",
        "FS_Comscfd-C001014000": "purchase_goods_cash",
        "FS_Comscfd-C002006000": "capex",
        "FI_T6-F061201B": "depreciation_amortization",
    }
    df = df.rename(columns=rename)
    keep = ["code", "year"] + list(rename.values())
    numeric = [c for c in keep if c not in ["code", "year", "firm_name", "industry_name", "industry_code"]]
    df = coerce_numeric(df[keep], numeric)
    df = df[is_manufacturing(df)].copy()
    df = df[~is_st_or_pt(df["firm_name"])].copy()
    df["industry_2digit"] = df["industry_code"].astype("string").str[:3]
    return construct_tfp(df)


def construct_tfp(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["value_added"] = (
        result["revenue"] - result["operating_cost"] + result["depreciation_amortization"].fillna(0)
    )
    log_sources = {
        "value_added": "ln_output",
        "employees": "ln_labor",
        "fixed_assets_net": "ln_capital",
        "operating_cost": "ln_intermediate",
    }
    for source, target in log_sources.items():
        result[target] = np.log(result[source].where(result[source] > 0))
    model_data = result.dropna(
        subset=["ln_output", "ln_labor", "ln_capital", "ln_intermediate", "year", "industry_2digit", "code"]
    ).copy()
    if len(model_data) < 100:
        result["tfp"] = np.nan
        return result

    model_data = model_data.sort_values(["code", "year"]).copy()
    model_data["ln_capital_sq"] = model_data["ln_capital"] ** 2
    model_data["ln_intermediate_sq"] = model_data["ln_intermediate"] ** 2
    model_data["capital_intermediate"] = model_data["ln_capital"] * model_data["ln_intermediate"]
    model_data["ln_capital_cu"] = model_data["ln_capital"] ** 3
    model_data["ln_intermediate_cu"] = model_data["ln_intermediate"] ** 3
    model_data["capital_sq_intermediate"] = model_data["ln_capital_sq"] * model_data["ln_intermediate"]
    model_data["capital_intermediate_sq"] = model_data["ln_capital"] * model_data["ln_intermediate_sq"]

    first_stage_formula = (
        "ln_output ~ ln_labor + ln_capital + ln_intermediate + ln_capital_sq + ln_intermediate_sq "
        "+ capital_intermediate + ln_capital_cu + ln_intermediate_cu + capital_sq_intermediate "
        "+ capital_intermediate_sq + C(year) + C(industry_2digit)"
    )
    first_stage = smf.ols(first_stage_formula, data=model_data).fit()
    beta_labor = first_stage.params["ln_labor"]
    model_data["phi_hat"] = first_stage.fittedvalues - beta_labor * model_data["ln_labor"]
    model_data["lag_phi_hat"] = model_data.groupby("code")["phi_hat"].shift(1)
    model_data["lag_ln_capital"] = model_data.groupby("code")["ln_capital"].shift(1)
    second_stage = model_data.dropna(subset=["lag_phi_hat", "lag_ln_capital"]).copy()

    def gmm_objective(beta_capital: float) -> float:
        omega = second_stage["phi_hat"] - beta_capital * second_stage["ln_capital"]
        lag_omega = second_stage["lag_phi_hat"] - beta_capital * second_stage["lag_ln_capital"]
        transition = pd.DataFrame(
            {
                "const": 1.0,
                "lag_omega": lag_omega,
                "lag_omega_sq": lag_omega**2,
                "lag_omega_cu": lag_omega**3,
            },
            index=second_stage.index,
        )
        innovation = sm.OLS(omega, transition).fit().resid
        instruments = np.column_stack(
            [
                np.ones(len(second_stage)),
                second_stage["lag_ln_capital"].to_numpy(),
                second_stage["ln_capital"].to_numpy(),
            ]
        )
        moments = (instruments * innovation.to_numpy()[:, None]).mean(axis=0)
        scale = instruments.std(axis=0)
        scale[scale == 0] = 1.0
        return float(((moments / scale) ** 2).sum())

    if len(second_stage) >= 100:
        optimization = minimize_scalar(gmm_objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-4})
        beta_capital = float(optimization.x)
    else:
        beta_capital = float(first_stage.params["ln_capital"])

    model_data["tfp"] = model_data["ln_output"] - beta_labor * model_data["ln_labor"] - beta_capital * model_data["ln_capital"]
    result = result.merge(model_data[["code", "year", "tfp"]], on=["code", "year"], how="left")
    return result


def build_ai_application_index(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    components = [
        "annual_ai_log",
        "mda_ai_log",
        "annual_ai_word_share",
        "mda_ai_word_share",
        "ai_invest_log",
        "ai_invest_add_log",
        "ai_invest_level",
    ]
    available = [c for c in components if c in result.columns]
    counts = result[available].notna().sum(axis=1)
    matrix = result[available].copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    scaled = pd.DataFrame(index=result.index)
    scaler = StandardScaler()
    for col in available:
        values = matrix[[col]]
        valid = values[col].notna()
        if valid.sum() <= 1 or values[col].std(skipna=True) == 0:
            scaled[col] = np.nan
            continue
        transformed = np.full(len(values), np.nan)
        transformed[valid.to_numpy()] = scaler.fit_transform(values.loc[valid, [col]]).ravel()
        scaled[col] = transformed
    result["ai_index_component_count"] = counts
    result["ai_application_index"] = scaled.mean(axis=1, skipna=True)
    result.loc[counts < 2, "ai_application_index"] = np.nan
    result["ai_application_index_pos"] = result["ai_application_index"] - result["ai_application_index"].min(skipna=True)
    return result


def zscore_components(df: pd.DataFrame, columns: list[str], min_components: int) -> pd.Series:
    scaled = pd.DataFrame(index=df.index)
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = values.notna()
        scaled[col] = np.nan
        if valid.sum() > 1 and values.std(skipna=True) > 0:
            scaled.loc[valid, col] = (values.loc[valid] - values.loc[valid].mean()) / values.loc[valid].std()
    result = scaled.mean(axis=1, skipna=True)
    result[scaled.notna().sum(axis=1) < min_components] = np.nan
    return result


def minmax_components(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = pd.DataFrame(index=df.index)
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        min_value = values.min(skipna=True)
        max_value = values.max(skipna=True)
        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            scaled[col] = np.nan
        else:
            scaled[col] = (values - min_value) / (max_value - min_value)
    return scaled


def entropy_weight_index(df: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    scaled = minmax_components(df, columns)
    filled = scaled.fillna(scaled.median())
    eps = 1e-12
    positive = filled + eps
    proportions = positive.div(positive.sum(axis=0), axis=1)
    n = len(filled)
    entropy = -(proportions * np.log(proportions)).sum(axis=0) / np.log(n)
    diversity = 1 - entropy
    if diversity.sum() <= 0:
        weights = pd.Series(1 / len(columns), index=columns)
    else:
        weights = diversity / diversity.sum()
    index = filled.mul(weights, axis=1).sum(axis=1)
    weight_table = pd.DataFrame({"component": columns, "entropy_weight": weights.reindex(columns).to_numpy()})
    return index, weight_table


def pca_first_component_index(df: pd.DataFrame, columns: list[str], orient_to: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    matrix = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    filled = matrix.fillna(matrix.median())
    scaled = StandardScaler().fit_transform(filled)
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    values = pca.fit_transform(scaled).ravel()
    index = pd.Series(values, index=df.index)
    if index.corr(orient_to) < 0:
        index = -index
        loadings = -pca.components_[0]
    else:
        loadings = pca.components_[0]
    loading_table = pd.DataFrame(
        {
            "component": columns,
            "pca_loading": loadings,
            "explained_variance_ratio": pca.explained_variance_ratio_[0],
        }
    )
    return index, loading_table


def build_research_indexes(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["labor_productivity_log"] = np.log(
        result["revenue"].where(result["revenue"] > 0) / result["employees"].where(result["employees"] > 0)
    )
    quality_components = ["roa", "operating_margin", "tfp"]
    efficiency_components = ["asset_turnover", "labor_productivity_log"]
    all_components = quality_components + efficiency_components
    for col in all_components:
        values = pd.to_numeric(result[col], errors="coerce")
        lower, upper = values.quantile([0.01, 0.99])
        result[f"{col}_qe"] = values.clip(lower, upper)
    result["quality_subindex"] = zscore_components(result, [f"{col}_qe" for col in quality_components], min_components=2)
    result["efficiency_subindex"] = zscore_components(
        result, [f"{col}_qe" for col in efficiency_components], min_components=1
    )
    result["quality_efficiency_index"] = zscore_components(
        result, ["quality_subindex", "efficiency_subindex"], min_components=2
    )
    component_cols = [f"{col}_qe" for col in all_components]
    entropy_index, entropy_weights = entropy_weight_index(result, component_cols)
    pca_index, pca_loadings = pca_first_component_index(result, component_cols, result["quality_efficiency_index"])
    result["quality_efficiency_entropy"] = entropy_index
    result["quality_efficiency_pca"] = pca_index
    result.attrs["quality_efficiency_entropy_weights"] = entropy_weights
    result.attrs["quality_efficiency_pca_loadings"] = pca_loadings

    actual_ai_components = ["ai_invest_log", "ai_invest_add_log", "ai_invest_level"]
    for col in actual_ai_components:
        values = pd.to_numeric(result[col], errors="coerce")
        lower, upper = values.quantile([0.01, 0.99])
        result[f"{col}_actual"] = values.clip(lower, upper)
    result["ai_actual_application_index"] = zscore_components(
        result, [f"{col}_actual" for col in actual_ai_components], min_components=2
    )
    conversion_data = result.dropna(
        subset=["ai_actual_application_index", "annual_ai_log", "mda_ai_log", "year", "industry_2digit"]
    ).copy()
    if len(conversion_data) >= 100:
        conversion_model = smf.ols(
            "ai_actual_application_index ~ annual_ai_log + mda_ai_log + C(year) + C(industry_2digit)",
            data=conversion_data,
        ).fit()
        conversion_data["ai_conversion_efficiency"] = conversion_model.resid
        result = result.merge(
            conversion_data[["code", "year", "ai_conversion_efficiency"]], on=["code", "year"], how="left"
        )
    else:
        result["ai_conversion_efficiency"] = np.nan
    return result


def build_modeling_panel() -> CleanedPanels:
    LOGGER.info("Cleaning raw data files with random_state=%s", RANDOM_STATE)
    ai_word = clean_ai_word()
    ai_investment = clean_ai_investment()
    controls = clean_controls()
    rd_investment = clean_rd_investment()
    ai_word = filter_to_manufacturing_universe(ai_word, controls)
    ai_investment = filter_to_manufacturing_universe(ai_investment, controls)
    rd_investment = filter_to_manufacturing_universe(rd_investment, controls)
    tfp = clean_tfp_inputs()
    y_candidates = clean_y_candidates()

    panel = controls.merge(ai_word.drop(columns=["firm_name"], errors="ignore"), on=["code", "year"], how="left")
    panel = panel.merge(ai_investment.drop(columns=["firm_name"], errors="ignore"), on=["code", "year"], how="left")
    rd_keep = [
        "code",
        "year",
        "rd_investment_intensity_pct",
        "rd_investment_amount",
    ]
    panel = panel.merge(rd_investment[rd_keep], on=["code", "year"], how="left")
    y_keep = [
        "code",
        "year",
        "revenue",
        "employees",
        "roa",
        "roe",
        "operating_margin",
        "asset_turnover",
    ]
    panel = panel.merge(y_candidates[y_keep], on=["code", "year"], how="left")
    panel = panel.merge(tfp[["code", "year", "tfp"]], on=["code", "year"], how="left")
    panel["rd_expenses"] = panel["rd_investment_amount"]
    panel["rd_intensity"] = panel["rd_investment_amount"] / panel["revenue"].replace(0, np.nan)
    panel["sales_expense_ratio"] = panel["selling_expense"] / panel["revenue"].replace(0, np.nan)
    panel["admin_expense_ratio"] = panel["admin_expense"] / panel["revenue"].replace(0, np.nan)
    panel = build_ai_application_index(panel)
    panel = build_research_indexes(panel)
    panel_attrs = dict(panel.attrs)
    panel = panel.sort_values(["code", "year"]).reset_index(drop=True)
    panel.attrs.update(panel_attrs)

    return CleanedPanels(
        ai_word=ai_word,
        ai_investment=ai_investment,
        controls=controls,
        rd_investment=rd_investment,
        tfp=tfp,
        y_candidates=y_candidates,
        modeling_panel=panel,
    )


def write_cleaned_outputs(panels: CleanedPanels) -> dict[str, Path]:
    outputs = {
        "ai_word": PROCESSED_DIR / "manufacturing_ai_word_panel.csv",
        "ai_investment": PROCESSED_DIR / "manufacturing_ai_investment_panel.csv",
        "controls": PROCESSED_DIR / "manufacturing_controls_panel.csv",
        "rd_investment": PROCESSED_DIR / "manufacturing_rd_investment_panel.csv",
        "tfp": PROCESSED_DIR / "manufacturing_tfp_panel.csv",
        "y_candidates": PROCESSED_DIR / "manufacturing_y_candidates_panel.csv",
        "modeling_panel": PROCESSED_DIR / "manufacturing_modeling_panel.csv",
    }
    for legacy in [PROCESSED_DIR / "ai_word_panel.csv", PROCESSED_DIR / "ai_investment_panel.csv"]:
        if legacy.exists():
            legacy.unlink()
    for name, path in outputs.items():
        getattr(panels, name).to_csv(path, index=False, encoding="utf-8-sig")
        LOGGER.info("Wrote %s rows to %s", len(getattr(panels, name)), path)
    return outputs


def build_data_summary(panels: CleanedPanels) -> pd.DataFrame:
    records = []
    for name in ["ai_word", "ai_investment", "controls", "rd_investment", "tfp", "y_candidates", "modeling_panel"]:
        df = getattr(panels, name)
        records.append(
            {
                "dataset": name,
                "rows": len(df),
                "firms": df["code"].nunique() if "code" in df.columns else np.nan,
                "year_min": df["year"].min() if "year" in df.columns else np.nan,
                "year_max": df["year"].max() if "year" in df.columns else np.nan,
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(TABLES_DIAGNOSTIC_DIR / "data_cleaning_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def build_sample_integrity_checks(panels: CleanedPanels) -> pd.DataFrame:
    datasets = {
        "ai_word": panels.ai_word,
        "ai_investment": panels.ai_investment,
        "controls": panels.controls,
        "rd_investment": panels.rd_investment,
        "tfp": panels.tfp,
        "y_candidates": panels.y_candidates,
        "modeling_panel": panels.modeling_panel,
    }
    records = []
    manufacturing_keys = panels.controls[["code", "year"]].drop_duplicates()
    for name, df in datasets.items():
        check = df.merge(manufacturing_keys.assign(in_manufacturing_controls=True), on=["code", "year"], how="left")
        bad_keys = check["in_manufacturing_controls"].isna().sum()
        non_c_rows = 0
        if "industry_code" in df.columns:
            non_c_rows = (~df["industry_code"].astype("string").str.startswith("C", na=False)).sum()
        pingan_rows = 0
        if "code" in df.columns:
            pingan_rows += df["code"].astype(str).str.zfill(6).eq("000001").sum()
        if "firm_name" in df.columns:
            pingan_rows = max(pingan_rows, df["firm_name"].astype(str).str.contains("平安银行", regex=False, na=False).sum())
        records.append(
            {
                "dataset": name,
                "rows": len(df),
                "non_manufacturing_key_rows": int(bad_keys),
                "non_c_industry_rows": int(non_c_rows),
                "pingan_bank_rows": int(pingan_rows),
                "passed": bad_keys == 0 and non_c_rows == 0 and pingan_rows == 0,
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(TABLES_DIAGNOSTIC_DIR / "sample_integrity_checks.csv", index=False, encoding="utf-8-sig")
    return result


def build_variable_dictionary() -> pd.DataFrame:
    rows = [
        ("ai_application_index", "核心解释变量", "AI词频、词频占比、AI投资及投资水平标准化后的等权综合指数"),
        ("ai_actual_application_index", "机制变量/应用水平变量", "仅由AI投资总额、AI投资增加值和AI投资水平构造，不包含词频信息"),
        ("ai_conversion_efficiency", "创新机制变量", "在给定AI文本关注度、年度和行业后，实际AI应用指数的残差，衡量AI关注度向实际应用落地的转化效率"),
        ("quality_efficiency_index", "首选被解释变量", "自构造提质增效指数，由质量子指数和效率子指数等权合成"),
        ("quality_efficiency_entropy", "稳健性被解释变量", "五个提质增效组成变量经极差标准化后用熵权法加权合成"),
        ("quality_efficiency_pca", "稳健性被解释变量", "五个提质增效组成变量标准化后提取第一主成分，并按主指数方向调整符号"),
        ("quality_subindex", "被解释变量分维度", "质量维度，由ROA、营业利润率和TFP标准化合成"),
        ("efficiency_subindex", "被解释变量分维度", "效率维度，由总资产周转率和劳动生产率标准化合成"),
        ("tfp", "备选/稳健性变量", "采用中间投入作为代理变量的控制函数方法估计企业层面TFP"),
        ("ln_assets", "控制变量", "总资产取自然对数，反映企业规模"),
        ("leverage", "控制变量", "资产负债率，统一为比例口径"),
        ("rd_intensity", "控制变量", "研发投入金额除以营业收入，研发投入金额来自研发投入情况表PT_LCRDSPENDING"),
        ("rd_investment_amount", "控制变量来源", "研发投入金额，来自研发投入情况表PT_LCRDSPENDING"),
        ("firm_age", "控制变量", "上市公司年龄"),
        ("revenue_growth", "控制变量", "营业收入增长率"),
        ("industry_2digit", "固定效应", "证监会行业代码前三位，制造业二级行业"),
        ("year", "固定效应", "年度"),
    ]
    dictionary = pd.DataFrame(rows, columns=["variable", "role", "definition"])
    dictionary.to_csv(TABLES_DIAGNOSTIC_DIR / "variable_dictionary.csv", index=False, encoding="utf-8-sig")
    return dictionary


def build_index_component_dictionary() -> pd.DataFrame:
    rows = [
        ("质量维度", "roa", "总资产净利润率", "越高表示单位资产创造利润能力越强"),
        ("质量维度", "operating_margin", "营业利润率", "越高表示主营业务盈利质量越好"),
        ("质量维度", "tfp", "TFP", "越高表示资本和劳动投入之外的相对产出效率越强"),
        ("效率维度", "asset_turnover", "总资产周转率", "越高表示资产使用效率越高"),
        ("效率维度", "labor_productivity_log", "劳动生产率对数", "营业收入/员工人数取对数，越高表示人均产出越高"),
    ]
    table = pd.DataFrame(rows, columns=["dimension", "component", "meaning", "interpretation"])
    table.to_csv(TABLES_DIAGNOSTIC_DIR / "quality_efficiency_index_components.csv", index=False, encoding="utf-8-sig")
    return table


def write_index_weight_tables(panels: CleanedPanels) -> None:
    attrs = panels.modeling_panel.attrs
    entropy = attrs.get("quality_efficiency_entropy_weights")
    pca = attrs.get("quality_efficiency_pca_loadings")
    if isinstance(entropy, pd.DataFrame):
        entropy.to_csv(TABLES_DIAGNOSTIC_DIR / "quality_efficiency_entropy_weights.csv", index=False, encoding="utf-8-sig")
    if isinstance(pca, pd.DataFrame):
        pca.to_csv(TABLES_DIAGNOSTIC_DIR / "quality_efficiency_pca_loadings.csv", index=False, encoding="utf-8-sig")
