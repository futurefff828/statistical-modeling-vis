from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, TABLES_MODEL_DIR

LOGGER = logging.getLogger(__name__)

# Journal-style palette: restrained, color-blind friendly, and printable.
INK = "#1F1F1F"
MUTED = "#6E6E6E"
GRID = "#E6E6E6"
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
LEGACY_FIGURES = (
    "ai_application_trend.png",
    "ai_core_coefficient_plot.png",
    "final_fixed_effects_coefficient_plot.png",
    "heterogeneity_coefficients.png",
    "mechanism_path_coefficients.png",
    "quality_efficiency_index_trend.png",
    "figures_preview_contactsheet.png",
    "fig8_ai_quality_partial_bins.png",
    "fig8_ai_quality_partial_bins.pdf",
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 360,
            "savefig.transparent": False,
            "font.size": 8.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, length=3, width=0.75)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def save_figure(fig, filename: str) -> None:
    fig.tight_layout(pad=0.55)
    out = FIGURES_DIR / filename
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cleanup_legacy_figures() -> None:
    figures_root = FIGURES_DIR.resolve()
    for filename in LEGACY_FIGURES:
        target = (FIGURES_DIR / filename).resolve()
        if target.parent == figures_root and target.exists():
            target.unlink()


def coefficient_panel(
    table: pd.DataFrame,
    label_col: str,
    filename: str,
    title: str,
    xlabel: str = "估计系数及95%置信区间",
    color: str = BLUE,
) -> None:
    setup_style()
    data = table.dropna(subset=["coef", "std_err"]).copy()
    if data.empty:
        return
    data["lower"] = data["coef"] - 1.96 * data["std_err"]
    data["upper"] = data["coef"] + 1.96 * data["std_err"]
    data = data.iloc[::-1].reset_index(drop=True)

    fig_height = max(3.1, 0.38 * len(data) + 0.95)
    fig, ax = plt.subplots(figsize=(5.8, fig_height))
    y = np.arange(len(data))
    xerr = np.vstack([data["coef"] - data["lower"], data["upper"] - data["coef"]])
    ax.errorbar(
        data["coef"],
        y,
        xerr=xerr,
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=0.95,
        capsize=2.7,
        capthick=0.95,
        markersize=3.6,
    )
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(data[label_col])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=6)
    polish_axis(ax, grid_axis="x")
    save_figure(fig, filename)


def _winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    q_low, q_high = values.quantile([lower, upper])
    return values.clip(q_low, q_high)


def _equal_count_groups(data: pd.DataFrame, x: str, y: str, groups: int) -> pd.DataFrame:
    work = data[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        work["bin"] = pd.Series(dtype="int64")
        return work
    work = work.sort_values(x, kind="mergesort").reset_index(drop=True)
    positions = np.arange(len(work))
    work["bin"] = np.minimum(np.floor(positions * groups / len(work)).astype(int), groups - 1)
    return work


def _binned_means(data: pd.DataFrame, x: str, y: str, bins: int = 10) -> pd.DataFrame:
    work = _equal_count_groups(data, x, y, bins)
    grouped = work.groupby("bin", observed=True).agg(
        x_mean=(x, "mean"),
        y_mean=(y, "mean"),
        y_se=(y, lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=(y, "size"),
    )
    grouped["y_low"] = grouped["y_mean"] - 1.96 * grouped["y_se"]
    grouped["y_high"] = grouped["y_mean"] + 1.96 * grouped["y_se"]
    return grouped.reset_index(drop=True)


def _fit_line(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_values = x.to_numpy()
    y_values = y.to_numpy()
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    grid = np.linspace(x_values.min(), x_values.max(), 100)
    return grid, intercept + slope * grid


def save_ai_qe_raw_bins(panel: pd.DataFrame) -> None:
    setup_style()
    data = panel[["ai_actual_application_index", "quality_efficiency_index"]].copy()
    data.columns = ["x", "y"]
    data["x"] = _winsorize(pd.to_numeric(data["x"], errors="coerce"))
    data["y"] = _winsorize(pd.to_numeric(data["y"], errors="coerce"))
    work = _equal_count_groups(data, "x", "y", 5)
    summary = work.groupby("bin", observed=True).agg(
        y_mean=("y", "mean"),
        y_se=("y", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=("y", "size"),
    )
    labels = ["低", "较低", "中", "较高", "高"][: len(summary)]
    summary = summary.reset_index(drop=True)
    summary["label"] = labels
    summary["y_low"] = summary["y_mean"] - 1.96 * summary["y_se"]
    summary["y_high"] = summary["y_mean"] + 1.96 * summary["y_se"]
    diff = summary["y_mean"].iloc[-1] - summary["y_mean"].iloc[0]

    fig, ax = plt.subplots(figsize=(5.9, 3.55))
    x_pos = np.arange(len(summary))
    colors = ["#D9D9D9", "#C7D6E5", "#9EC3DD", "#5EA6C8", TEAL][: len(summary)]
    ax.bar(x_pos, summary["y_mean"], color=colors, edgecolor="white", linewidth=0.6, width=0.72)
    ax.errorbar(
        x_pos,
        summary["y_mean"],
        yerr=np.vstack([summary["y_mean"] - summary["y_low"], summary["y_high"] - summary["y_mean"]]),
        fmt="none",
        color=INK,
        elinewidth=0.8,
        capsize=2.4,
        capthick=0.8,
    )
    ax.plot(x_pos, summary["y_mean"], color=INK, linewidth=1.0, marker="o", markersize=3.0)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.text(
        0.02,
        0.95,
        f"高组 - 低组 = {diff:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=INK,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary["label"])
    ax.set_title("AI应用水平分组与提质增效", loc="left", pad=6)
    ax.set_xlabel("实际AI应用水平分组")
    ax.set_ylabel("提质增效指数均值")
    polish_axis(ax, grid_axis="y")
    save_figure(fig, "fig7_ai_application_quality_bins.png")


def save_ai_trend(panel: pd.DataFrame) -> None:
    setup_style()
    trend = panel.groupby("year", as_index=False).agg(
        annual_ai_log=("annual_ai_log", "mean"),
        mda_ai_log=("mda_ai_log", "mean"),
        ai_actual_application_index=("ai_actual_application_index", "mean"),
    )

    fig, ax = plt.subplots(figsize=(5.9, 3.35))
    ax.plot(trend["year"], trend["annual_ai_log"], color=BLUE, linewidth=1.35, marker="o", markersize=2.8, label="年报词频")
    ax.plot(trend["year"], trend["mda_ai_log"], color=MUTED, linewidth=1.1, marker="s", markersize=2.6, label="MD&A词频")
    ax2 = ax.twinx()
    ax2.plot(
        trend["year"],
        trend["ai_actual_application_index"],
        color=TEAL,
        linewidth=1.15,
        linestyle=(0, (4, 2)),
        marker="^",
        markersize=2.6,
        label="实际应用",
    )
    ax.set_title("AI关注与应用趋势", loc="left", pad=6)
    ax.set_xlabel("年份")
    ax.set_ylabel("词频对数均值")
    ax2.set_ylabel("应用指数均值")
    polish_axis(ax, grid_axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(INK)
    ax2.tick_params(colors=INK, length=3, width=0.75)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    save_figure(fig, "fig1_ai_trend.png")


def save_y_trend(panel: pd.DataFrame) -> None:
    setup_style()
    trend = panel.groupby("year", as_index=False).agg(
        quality_efficiency_index=("quality_efficiency_index", "mean"),
        quality_subindex=("quality_subindex", "mean"),
        efficiency_subindex=("efficiency_subindex", "mean"),
    )

    fig, ax = plt.subplots(figsize=(5.9, 3.35))
    ax.plot(trend["year"], trend["quality_efficiency_index"], color=INK, linewidth=1.45, marker="o", markersize=2.8, label="综合指数")
    ax.plot(trend["year"], trend["quality_subindex"], color=BLUE, linewidth=1.1, marker="s", markersize=2.6, label="质量")
    ax.plot(trend["year"], trend["efficiency_subindex"], color=TEAL, linewidth=1.1, marker="^", markersize=2.6, label="效率")
    ax.set_title("提质增效趋势", loc="left", pad=6)
    ax.set_xlabel("年份")
    ax.set_ylabel("指数均值")
    polish_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    save_figure(fig, "fig2_quality_efficiency_trend.png")


def save_coefficient_plot(reg_table: pd.DataFrame, robustness: pd.DataFrame) -> None:
    combined = pd.concat([reg_table, robustness], ignore_index=True)
    if combined.empty:
        return
    label_map = {
        "M1_fe_baseline": "基准",
        "M2_fe_controls": "加控制",
        "M3_fe_controls_industry": "行业FE",
        "R1_mda_frequency_x": "MD&A",
        "R2_composite_ai_x": "综合AI",
        "R3_tfp_y": "TFP",
    }
    combined["label"] = combined["model"].map(label_map).fillna(combined["model"])
    coefficient_panel(combined, "label", "fig3_baseline_robustness_coefficients.png", "基准与稳健性", color=BLUE)


def save_all_figures(panel: pd.DataFrame, reg_table: pd.DataFrame, robustness: pd.DataFrame) -> None:
    cleanup_legacy_figures()
    save_ai_trend(panel)
    save_y_trend(panel)
    save_ai_qe_raw_bins(panel)
    save_coefficient_plot(reg_table, robustness)


def save_final_research_coefficient_plot(final_results: pd.DataFrame) -> None:
    label_map = {
        "M1_main_fe": "主模型",
        "M2_mda_frequency": "MD&A",
        "M3_ai_composite": "综合AI",
        "M7_entropy_index_y": "熵权Y",
        "M8_pca_index_y": "PCA Y",
        "M9_rd_missing_robust": "研发缺失",
        "M11_mechanism_word_to_actual_ai": "实际应用",
        "M12_mechanism_mda_to_actual_ai": "MD&A应用",
        "M13_conversion_efficiency": "转化效率",
    }
    keep = list(label_map)
    table = final_results[final_results["model"].isin(keep)].copy()
    table["label"] = table["model"].map(label_map)
    coefficient_panel(table, "label", "fig4_final_model_coefficients.png", "最终模型汇总", color=BLUE)


def save_mechanism_plot(mechanism_results: pd.DataFrame) -> None:
    label_map = {
        "ME1_actual_ai_application": "实际应用",
        "ME2_asset_turnover": "资产效率",
        "ME3_labor_productivity": "劳动效率",
        "ME4_tfp": "TFP",
        "ME5_efficiency_subindex": "效率指数",
        "ME6_quality_subindex": "质量指数",
        "ME7_conversion_to_quality_efficiency": "转化效率",
    }
    table = mechanism_results.copy()
    table["label"] = table["model"].map(label_map).fillna(table["model"])
    coefficient_panel(table, "label", "fig5_mechanism_coefficients.png", "机制路径", color=TEAL)


def save_heterogeneity_plot(heterogeneity_results: pd.DataFrame) -> None:
    label_map = {
        "state_owned": "国企",
        "non_state_owned": "非国企",
        "east_region": "东部",
        "non_east_region": "非东部",
        "high_tech_manufacturing": "高技术",
        "non_high_tech_manufacturing": "一般制造",
        "high_conversion_efficiency": "高转化",
        "low_conversion_efficiency": "低转化",
    }
    table = heterogeneity_results.copy()
    table["label"] = table["group"].map(label_map).fillna(table["group"])
    coefficient_panel(table, "label", "fig6_heterogeneity_coefficients.png", "异质性分析", color=ORANGE)


def save_extended_figures() -> None:
    mechanism_path = TABLES_MODEL_DIR / "mechanism_regression_results.csv"
    heterogeneity_path = TABLES_MODEL_DIR / "heterogeneity_regression_results.csv"
    if mechanism_path.exists():
        save_mechanism_plot(pd.read_csv(mechanism_path))
    if heterogeneity_path.exists():
        save_heterogeneity_plot(pd.read_csv(heterogeneity_path))
