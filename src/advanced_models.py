from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from .config import RANDOM_STATE, TABLES_MODEL_DIR
from .research_models import CONTROLS, control_term, prepare_research_data

LOGGER = logging.getLogger(__name__)


def base_model_data(panel: pd.DataFrame) -> pd.DataFrame:
    data = prepare_research_data(panel)
    needed = [
        "quality_efficiency_index_w",
        "annual_ai_log_w",
        "code",
        "year",
        "industry_2digit",
    ] + [control_term(col) for col in CONTROLS]
    return data.dropna(subset=needed).copy()


def run_quantile_fixed_effect_models(panel: pd.DataFrame) -> pd.DataFrame:
    data = base_model_data(panel)
    controls = " + ".join(control_term(col) for col in CONTROLS)
    formula = f"quality_efficiency_index_w ~ annual_ai_log_w + {controls} + C(year) + C(industry_2digit)"
    rows = []
    for quantile in [0.25, 0.5, 0.75]:
        fit = smf.quantreg(formula, data=data).fit(q=quantile, max_iter=2000)
        rows.append(
            {
                "model": "quantile_fe",
                "quantile": quantile,
                "y": "quality_efficiency_index_w",
                "x": "annual_ai_log_w",
                "coef": fit.params.get("annual_ai_log_w", np.nan),
                "std_err": fit.bse.get("annual_ai_log_w", np.nan),
                "t_value": fit.tvalues.get("annual_ai_log_w", np.nan),
                "p_value": fit.pvalues.get("annual_ai_log_w", np.nan),
                "nobs": int(fit.nobs),
                "pseudo_r_squared": fit.prsquared,
                "purpose": "分位数固定效应：检验AI作用在提质增效分布不同位置是否异质",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "quantile_fe_results.csv", index=False, encoding="utf-8-sig")
    return table


def run_nonlinear_ai_effect_model(panel: pd.DataFrame) -> pd.DataFrame:
    data = base_model_data(panel)
    data["annual_ai_log_centered"] = data["annual_ai_log_w"] - data["annual_ai_log_w"].mean()
    data["annual_ai_log_centered_sq"] = data["annual_ai_log_centered"] ** 2
    controls = " + ".join(control_term(col) for col in CONTROLS)
    formula = (
        "quality_efficiency_index_w ~ annual_ai_log_centered + annual_ai_log_centered_sq "
        f"+ {controls} + C(year) + C(industry_2digit)"
    )
    fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["code"]})
    rows = []
    for var, label in [
        ("annual_ai_log_centered", "AI词频一次项"),
        ("annual_ai_log_centered_sq", "AI词频二次项"),
    ]:
        rows.append(
            {
                "model": "nonlinear_fe",
                "term": label,
                "y": "quality_efficiency_index_w",
                "x": var,
                "coef": fit.params.get(var, np.nan),
                "std_err": fit.bse.get(var, np.nan),
                "t_value": fit.tvalues.get(var, np.nan),
                "p_value": fit.pvalues.get(var, np.nan),
                "nobs": int(fit.nobs),
                "r_squared": fit.rsquared,
                "adj_r_squared": fit.rsquared_adj,
                "purpose": "非线性固定效应：检验AI关注度是否存在边际递减或门槛特征",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_MODEL_DIR / "nonlinear_ai_effect_results.csv", index=False, encoding="utf-8-sig")
    return table


def cross_fitted_residuals(data: pd.DataFrame, target: str, features: list[str]) -> np.ndarray:
    numeric_features = [f for f in features if f not in ["year", "industry_2digit"]]
    categorical_features = ["year", "industry_2digit"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    learner = make_pipeline(
        preprocessor,
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    residuals = np.full(len(data), np.nan)
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    x = data[features]
    y = data[target].to_numpy()
    for train_idx, test_idx in kfold.split(data):
        model = clone(learner)
        model.fit(x.iloc[train_idx], y[train_idx])
        pred = model.predict(x.iloc[test_idx])
        residuals[test_idx] = y[test_idx] - pred
    return residuals


def run_cross_fitted_partial_linear_model(panel: pd.DataFrame) -> pd.DataFrame:
    data = base_model_data(panel).reset_index(drop=True)
    features = [control_term(col) for col in CONTROLS] + ["year", "industry_2digit"]
    data["y_resid"] = cross_fitted_residuals(data, "quality_efficiency_index_w", features)
    data["x_resid"] = cross_fitted_residuals(data, "annual_ai_log_w", features)
    fit = sm.OLS(data["y_resid"], sm.add_constant(data["x_resid"])).fit(
        cov_type="cluster", cov_kwds={"groups": data["code"]}
    )
    table = pd.DataFrame(
        [
            {
                "model": "cross_fitted_partial_linear",
                "y": "quality_efficiency_index_w",
                "x": "annual_ai_log_w",
                "coef": fit.params.get("x_resid", np.nan),
                "std_err": fit.bse.get("x_resid", np.nan),
                "t_value": fit.tvalues.get("x_resid", np.nan),
                "p_value": fit.pvalues.get("x_resid", np.nan),
                "nobs": int(fit.nobs),
                "r_squared": fit.rsquared,
                "purpose": "交叉拟合部分线性模型：用随机森林净化控制变量、年份和行业影响后估计AI净效应",
            }
        ]
    )
    table.to_csv(TABLES_MODEL_DIR / "cross_fitted_partial_linear_results.csv", index=False, encoding="utf-8-sig")
    return table


def run_all_advanced_models(panel: pd.DataFrame) -> None:
    run_quantile_fixed_effect_models(panel)
    run_nonlinear_ai_effect_model(panel)
    run_cross_fitted_partial_linear_model(panel)
