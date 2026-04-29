from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LOGS_DIR, RANDOM_STATE, cleanup_legacy_output_files, ensure_project_dirs
from src.advanced_models import run_all_advanced_models
from src.data_cleaning import (
    build_data_summary,
    build_index_component_dictionary,
    build_modeling_panel,
    build_sample_integrity_checks,
    build_variable_dictionary,
    write_index_weight_tables,
    write_cleaned_outputs,
)
from src.modeling import (
    descriptive_statistics,
    heteroskedasticity_test,
    missingness_summary,
    run_regressions,
    run_robustness,
    vif_table,
)
from src.table_outputs import build_paper_tables
from src.research_models import (
    run_final_research_models,
    run_conversion_moderation_model,
    run_firm_year_fe_boundary_tests,
    run_heterogeneity_models,
    run_mechanism_models,
    screen_candidate_pairs,
)
from src.visualization import save_all_figures, save_extended_figures, save_final_research_coefficient_plot


def setup_logging() -> None:
    ensure_project_dirs()
    cleanup_legacy_output_files()
    log_path = LOGS_DIR / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    setup_logging()
    logging.info("Starting reproducible pipeline with random_state=%s", RANDOM_STATE)
    panels = build_modeling_panel()
    write_cleaned_outputs(panels)
    build_data_summary(panels)
    build_sample_integrity_checks(panels)
    build_variable_dictionary()
    build_index_component_dictionary()
    write_index_weight_tables(panels)
    descriptive_statistics(panels.modeling_panel)
    missingness_summary(panels.modeling_panel)
    reg_table, fits = run_regressions(panels.modeling_panel)
    robustness = run_robustness(panels.modeling_panel)
    vif_table(panels.modeling_panel)
    heteroskedasticity_test(fits)
    screen_candidate_pairs(panels.modeling_panel)
    final_results = run_final_research_models(panels.modeling_panel)
    run_mechanism_models(panels.modeling_panel)
    run_heterogeneity_models(panels.modeling_panel)
    run_conversion_moderation_model(panels.modeling_panel)
    run_firm_year_fe_boundary_tests(panels.modeling_panel)
    run_all_advanced_models(panels.modeling_panel)
    save_all_figures(panels.modeling_panel, reg_table, robustness)
    save_final_research_coefficient_plot(final_results)
    save_extended_figures()
    build_paper_tables()
    logging.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
