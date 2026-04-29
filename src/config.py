from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_ROOT_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR = FIGURES_ROOT_DIR / "final"
TABLES_FINAL_DIR = TABLES_DIR / "final"
TABLES_MODEL_DIR = TABLES_DIR / "model_results"
TABLES_DIAGNOSTIC_DIR = TABLES_DIR / "diagnostics"
LOGS_DIR = OUTPUTS_DIR / "logs"

RAW_FILES = {
    "ai_word": RAW_DIR / "AI_词频.xlsx",
    "ai_investment": RAW_DIR / "AI_InvestmentLevel.xlsx",
    "controls": RAW_DIR / "raw_controls_heterogeneity.xlsx",
    "rd_investment": RAW_DIR / "研发投入.xlsx",
    "tfp_inputs": RAW_DIR / "raw_tfp.xlsx",
    "y_candidates": RAW_DIR / "raw_y_candidates.xlsx",
}


LEGACY_OUTPUT_FILES = {
    TABLES_DIR: {
        "baseline_regression_results.csv",
        "candidate_xy_screening_results.csv",
        "conversion_moderation_results.csv",
        "cross_fitted_partial_linear_results.csv",
        "data_cleaning_summary.csv",
        "descriptive_statistics.csv",
        "final_fixed_effects_research_results.csv",
        "firm_year_fe_boundary_tests.csv",
        "heterogeneity_regression_results.csv",
        "heteroskedasticity_tests.csv",
        "mechanism_regression_results.csv",
        "missingness_summary.csv",
        "nonlinear_ai_effect_results.csv",
        "paper_table_1_main_regression.csv",
        "paper_table_2_robustness.csv",
        "paper_table_3_mechanism.csv",
        "paper_table_4_heterogeneity.csv",
        "paper_table_5_conversion_moderation.csv",
        "paper_table_6_advanced_models.csv",
        "paper_table_A1_boundary.csv",
        "quality_efficiency_entropy_weights.csv",
        "quality_efficiency_index_components.csv",
        "quality_efficiency_pca_loadings.csv",
        "quantile_fe_results.csv",
        "robustness_regression_results.csv",
        "sample_integrity_checks.csv",
        "variable_dictionary.csv",
        "vif_diagnostics.csv",
    },
    TABLES_DIAGNOSTIC_DIR: {
        "no_rd_rescue_model_check.csv",
        "no_rd_xy_rescue_screening.csv",
        "tfp_variant_screening.csv",
    },
    FIGURES_ROOT_DIR: {
        "fig1_ai_trend.pdf",
        "fig1_ai_trend.png",
        "fig2_quality_efficiency_trend.pdf",
        "fig2_quality_efficiency_trend.png",
        "fig3_baseline_robustness_coefficients.pdf",
        "fig3_baseline_robustness_coefficients.png",
        "fig4_final_model_coefficients.pdf",
        "fig4_final_model_coefficients.png",
        "fig5_mechanism_coefficients.pdf",
        "fig5_mechanism_coefficients.png",
        "fig6_heterogeneity_coefficients.pdf",
        "fig6_heterogeneity_coefficients.png",
        "fig7_ai_application_quality_bins.pdf",
        "fig7_ai_application_quality_bins.png",
        "fig8_ai_quality_partial_bins.pdf",
        "fig8_ai_quality_partial_bins.png",
    },
}


def cleanup_legacy_output_files() -> None:
    """Remove only known generated files from the old flat output layout."""
    for directory, filenames in LEGACY_OUTPUT_FILES.items():
        if not directory.exists():
            continue
        for filename in filenames:
            path = directory / filename
            if path.is_file():
                path.unlink()


def ensure_project_dirs() -> None:
    for path in [
        PROCESSED_DIR,
        FIGURES_ROOT_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        TABLES_FINAL_DIR,
        TABLES_MODEL_DIR,
        TABLES_DIAGNOSTIC_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
