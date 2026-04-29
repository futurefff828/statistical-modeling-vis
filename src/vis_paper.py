"""论文精美图表模块 — 统一全局风格，全部输出 PNG + PDF 双格式。

可单独运行每个 save_*() 函数，也可调用 run_all() 一键生成全部。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline

from .config import (
    FIGURES_DIR,
    PROCESSED_DIR,
    TABLES_DIAGNOSTIC_DIR,
    TABLES_FINAL_DIR,
    TABLES_MODEL_DIR,
)

LOGGER = logging.getLogger(__name__)

# ── 全局统一视觉规范 ─────────────────────────────────────────────
INK = "#1F1F1F"
MUTED = "#6E6E6E"
GRID = "#E6E6E6"
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
RED_ACCENT = "#CC3311"
PURPLE = "#8E6AB3"
GOLD = "#C49B44"
BLUE_LIGHT = "#7CB9D5"
TEAL_LIGHT = "#7EC8B8"
ORANGE_LIGHT = "#E8A87C"

SEABORN_PALETTE = [BLUE, TEAL, ORANGE, PURPLE, RED_ACCENT, GOLD, MUTED]
FIGURE_W = 5.9
NOTE_FMT = "注：标准误为公司层面聚类稳健标准误。{extra}"
PANEL_LABELS = [chr(97 + i) for i in range(26)]  # a, b, c, ...


def setup_style() -> None:
    """全局 matplotlib 样式 (与现有 visualization.py 完全一致)。"""
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
            "legend.fontsize": 7.6,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axis(ax, grid_axis: str = "y") -> None:
    """去除上右边框 + 统一网格样式 (与现有 visualization.py 完全一致)。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, length=3, width=0.75)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def save_figure(fig, filename: str, extra_note: str = "") -> None:
    """统一保存为 PNG + PDF，底部可加注释行。"""
    out = FIGURES_DIR / filename
    if extra_note:
        fig.text(
            0.02,
            -0.01,
            NOTE_FMT.format(extra=extra_note),
            ha="left",
            va="top",
            fontsize=6.5,
            color=MUTED,
            style="italic",
        )
    fig.tight_layout(pad=0.55)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=360)
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", dpi=360)
    plt.close(fig)
    LOGGER.info("Saved %s", out)


# ── 数据读取工具 ──────────────────────────────────────────────────

def _load_panel() -> pd.DataFrame:
    p = PROCESSED_DIR / "manufacturing_modeling_panel.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到面板数据: {p}")
    df = pd.read_csv(p)
    return df


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    df = pd.read_csv(path)
    # Clean BOM from column names
    df.columns = df.columns.str.lstrip("\ufeff")
    return df


def _parse_coef(val) -> float:
    """从 '0.0490***' 或 '-0.0175**' 之类的字符串中提取数字。"""
    s = str(val).strip().replace("(", "").replace(")", "")
    # 移除尾部的星号
    s = s.rstrip("*")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_se(val) -> float:
    """从 '(0.0079)' 之类的字符串中提取数字。"""
    s = str(val).strip().replace("(", "").replace(")", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    q_low, q_high = values.quantile([lower, upper])
    return values.clip(q_low, q_high)


def _equal_count_groups(series: pd.Series, groups: int) -> pd.Series:
    ranks = series.rank(method="average", na_option="bottom")
    valid = ranks.notna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result[valid] = np.minimum(
        np.floor(ranks[valid] * groups / (ranks[valid].max() + 1)).astype(int),
        groups - 1,
    )
    return result


def _add_footnote(fig, text: str) -> None:
    fig.text(0.02, 0.005, text, ha="left", va="bottom", fontsize=6.5, color=MUTED, style="italic")


def _panel_label(ax, idx: int) -> None:
    ax.text(
        -0.08,
        1.06,
        f"({PANEL_LABELS[idx]})",
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=INK,
    )


# ═══════════════════════════════════════════════════════════════════
#  图表函数 (共 18 个)
# ═══════════════════════════════════════════════════════════════════

# ── ① 数据筛选流程 / fig_s1_sample_flow ──────────────────────────

def save_sample_flow() -> None:
    """数据筛选流程图 - 第2章 §2.1 数据来源"""
    setup_style()
    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.8))

    steps = [
        ("原始A股制造业\n上市公司样本", "N=29,621  企业=3,843  2013-2024", BLUE),
        ("剔除关键变量缺失\n(AI词频/提质增效/控制变量)", "N=28,450", TEAL),
        ("剔除研发强度缺失\n(主模型样本)", "N=26,869", ORANGE),
        ("最终建模样本\n(企业=3,615  年份=12)", "N=26,869", INK),
    ]

    y_start = 3.3
    dy = 0.72
    box_h = 0.55
    box_w = 3.6
    left = 1.15

    for i, (label, detail, color) in enumerate(steps):
        y = y_start - i * dy
        rect = plt.Rectangle((left, y - box_h / 2), box_w, box_h,
                              facecolor=color, alpha=0.12, edgecolor=color,
                              linewidth=1.0, zorder=2)
        ax.add_patch(rect)
        ax.text(left + 0.15, y + 0.08, label, va="center", fontsize=8.4,
                color=INK, fontweight="bold")
        ax.text(left + 0.15, y - 0.16, detail, va="center", fontsize=7.4,
                color=MUTED)
        if i < len(steps) - 1:
            ax.annotate("", xy=(left + box_w / 2, y - box_h / 2 - 0.02),
                        xytext=(left + box_w / 2, y_start - (i + 1) * dy + box_h / 2 + 0.02),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3.8)
    ax.set_title("数据筛选流程", loc="left", pad=6)
    ax.axis("off")

    save_figure(fig, "fig_s1_sample_flow.png",
                extra_note="制造业A股上市公司企业-年份面板。研发强度缺失约占3%，主模型采用完整样本。")


# ── ② 提质增效指数结构 / fig9_index_structure ─────────────────────

def save_index_structure() -> None:
    """提质增效指数构成结构图 - 第2章 §2.1 变量构造"""
    setup_style()
    fig, ax = plt.subplots(figsize=(FIGURE_W, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # 指标 → 子指数 → 综合指数
    metrics = [
        ("总资产净利润率\n(ROA)", "质量维度\nQuality", 1.2),
        ("营业利润率\n(Operating Margin)", "质量维度\nQuality", 2.2),
        ("全要素生产率\n(TFP, LP法)", "质量维度\nQuality", 3.2),
        ("总资产周转率\n(Asset Turnover)", "效率维度\nEfficiency", 6.8),
        ("劳动生产率对数\n(Labor Productivity)", "效率维度\nEfficiency", 7.8),
    ]

    colors_metrics = [TEAL_LIGHT, TEAL_LIGHT, TEAL_LIGHT, ORANGE_LIGHT, ORANGE_LIGHT]

    for i, (label, _, x) in enumerate(metrics):
        rect = plt.Rectangle((x - 0.7, 3.6), 1.4, 0.9, facecolor=colors_metrics[i],
                              alpha=0.35, edgecolor=colors_metrics[i], linewidth=0.9)
        ax.add_patch(rect)
        ax.text(x, 4.05, label, ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.3)

    # 质量子指数盒子
    rect_q = plt.Rectangle((1.2 - 0.9, 2.0), 1.8, 1.0, facecolor=TEAL, alpha=0.2,
                            edgecolor=TEAL, linewidth=1.3)
    ax.add_patch(rect_q)
    ax.text(1.2, 2.5, "质量子指数\nQuality Subindex", ha="center", va="center",
            fontsize=8.2, color=INK, fontweight="bold")

    # 效率子指数盒子
    rect_e = plt.Rectangle((7.0 - 0.9, 2.0), 1.8, 1.0, facecolor=ORANGE, alpha=0.2,
                            edgecolor=ORANGE, linewidth=1.3)
    ax.add_patch(rect_e)
    ax.text(7.0, 2.5, "效率子指数\nEfficiency Subindex", ha="center", va="center",
            fontsize=8.2, color=INK, fontweight="bold")

    # 箭头上层
    for x_src, x_dst in [(1.2, 1.2), (2.2, 1.2), (3.2, 1.2)]:
        ax.annotate("", xy=(x_dst, 3.2), xytext=(x_src, 3.55),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    for x_src, x_dst in [(6.8, 7.0), (7.8, 7.0)]:
        ax.annotate("", xy=(x_dst, 3.2), xytext=(x_src, 3.55),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))

    # 综合指数盒子
    rect_c = plt.Rectangle((3.6 - 1.4, 0.6), 2.8, 1.1, facecolor=INK, alpha=0.08,
                            edgecolor=INK, linewidth=1.6)
    ax.add_patch(rect_c)
    ax.text(3.6, 1.15, "提质增效综合指数", ha="center", va="center",
            fontsize=9.0, color=INK, fontweight="bold")

    # 箭头下层
    ax.annotate("", xy=(3.6 + 0.3, 1.72), xytext=(1.2, 1.95),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.annotate("", xy=(3.6 - 0.3, 1.72), xytext=(7.0, 1.95),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    # 标注
    ax.text(4.1, 2.2, "等权合成\nw=0.5 : 0.5", fontsize=7.0, color=MUTED)
    ax.text(0.5, 3.3, "标准化\n(Min-Max)", fontsize=7.0, color=MUTED, rotation=90, va="center")
    ax.text(9.0, 3.3, "标准化\n(Min-Max)", fontsize=7.0, color=MUTED, rotation=90, va="center")

    ax.set_title("提质增效指数构造框架", loc="left", pad=6)

    save_figure(fig, "fig9_index_structure.png",
                extra_note="各分项指标标准化到[0,1]区间后等权合成。TFP采用LP法估计。")


# ── ③ 描述性统计组合图 / fig10_descriptive_stats ──────────────────

def save_descriptive_dashboard() -> None:
    """主要变量描述性统计 2×2 面板 - 第2章 §2.1"""
    setup_style()
    panel = _load_panel()
    vars_plot = {
        "quality_efficiency_index": "提质增效指数",
        "annual_ai_log": "年报AI词频(log)",
        "tfp": "TFP",
        "ln_assets": "企业规模(log)",
        "leverage": "资产负债率",
        "revenue_growth": "营收增长率",
    }

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_W + 1.2, 5.5))
    axes = axes.flatten()

    # (a) 小提琴图
    ax = axes[0]
    violin_data = []
    violin_labels = []
    for col, label in vars_plot.items():
        s = pd.to_numeric(panel[col], errors="coerce").dropna()
        s = _winsorize(s, 0.005, 0.995)
        violin_data.append(s.values)
        violin_labels.append(label)
    vp = ax.violinplot(violin_data, positions=np.arange(len(violin_data)),
                       showmeans=False, showmedians=True, widths=0.7)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(SEABORN_PALETTE[i % len(SEABORN_PALETTE)])
        body.set_alpha(0.55)
    for part in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if part in vp:
            vp[part].set_color(INK)
            vp[part].set_linewidth(0.8)
    ax.set_xticks(np.arange(len(violin_data)))
    ax.set_xticklabels(violin_labels, rotation=30, ha="right", fontsize=7.0)
    ax.set_ylabel("标准化值")
    _panel_label(ax, 0)
    ax.set_title("主要变量分布", loc="left", fontsize=8.6)
    polish_axis(ax)

    # (b) 相关性热力图
    ax = axes[1]
    corr_vars = list(vars_plot.keys())
    corr_labels = list(vars_plot.values())
    corr_data = panel[corr_vars].apply(pd.to_numeric, errors="coerce")
    corr = corr_data.corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr_vars)))
    ax.set_yticks(np.arange(len(corr_vars)))
    ax.set_xticklabels(corr_labels, rotation=30, ha="right", fontsize=6.8)
    ax.set_yticklabels(corr_labels, fontsize=6.8)
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(corr.values[i, j]) > 0.45 else INK)
    _panel_label(ax, 1)
    ax.set_title("变量相关性", loc="left", fontsize=8.6)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    # (c) 样本年份分布
    ax = axes[2]
    yr_cnt = panel.groupby("year").size().sort_index()
    ax.bar(yr_cnt.index.astype(int), yr_cnt.values, color=BLUE, alpha=0.7, width=0.7,
           edgecolor="white", linewidth=0.4)
    ax.set_xlabel("年份")
    ax.set_ylabel("观测数")
    _panel_label(ax, 2)
    ax.set_title("样本年份分布", loc="left", fontsize=8.6)
    polish_axis(ax)

    # (d) 行业分布 Top10
    ax = axes[3]
    top_ind = panel.groupby("industry_name").size().sort_values(ascending=False).head(10)
    short_names = [n[:12] for n in top_ind.index]
    ax.barh(np.arange(len(short_names))[::-1], top_ind.values,
            color=TEAL, alpha=0.7, height=0.65, edgecolor="white", linewidth=0.4)
    ax.set_yticks(np.arange(len(short_names))[::-1])
    ax.set_yticklabels(short_names, fontsize=6.8)
    ax.set_xlabel("观测数")
    _panel_label(ax, 3)
    ax.set_title("制造业二级行业 Top10", loc="left", fontsize=8.6)
    polish_axis(ax)

    fig.tight_layout(pad=1.2)
    save_figure(fig, "fig10_descriptive_stats.png",
                extra_note="变量经1%双侧缩尾处理。数据来源：A股制造业上市公司2013-2024年面板。")


# ── ④ 缺失率柱状图 / fig_s2_missingness ─────────────────────────

def save_missingness_bar() -> None:
    """关键变量缺失率 - 附录"""
    setup_style()
    df = _load_csv(TABLES_DIAGNOSTIC_DIR / "missingness_summary.csv")
    key_vars = [
        "annual_ai_log", "mda_ai_log", "ai_application_index",
        "quality_efficiency_index", "tfp", "rd_intensity",
        "ln_assets", "leverage", "firm_age", "revenue_growth",
        "labor_productivity_log", "asset_turnover",
    ]
    df = df[df["variable"].isin(key_vars)].copy()
    df["missing_rate_pct"] = df["missing_rate"] * 100
    df = df.sort_values("missing_rate_pct", ascending=True)

    labels = {
        "annual_ai_log": "年报AI词频(log)",
        "mda_ai_log": "MD&A AI词频(log)",
        "ai_application_index": "AI应用指数",
        "quality_efficiency_index": "提质增效指数",
        "tfp": "TFP",
        "rd_intensity": "研发强度",
        "ln_assets": "企业规模(log)",
        "leverage": "资产负债率",
        "firm_age": "企业年龄",
        "revenue_growth": "营收增长率",
        "labor_productivity_log": "劳动生产率(log)",
        "asset_turnover": "资产周转率",
    }
    df["label"] = df["variable"].map(labels)

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.8))
    colors_list = [BLUE if v < 5 else (TEAL if v < 10 else ORANGE)
                   for v in df["missing_rate_pct"]]

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["missing_rate_pct"], color=colors_list, alpha=0.75,
            height=0.6, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=7.8)
    ax.set_xlabel("缺失率 (%)")
    ax.axvline(5, color=ORANGE, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(5.1, len(df) - 0.5, "5%", fontsize=7.2, color=ORANGE)

    for i, (rate, missing) in enumerate(zip(df["missing_rate_pct"], df["missing"])):
        ax.text(rate + 0.3, i, f"{rate:.1f}%  (n={int(missing)})",
                va="center", fontsize=6.8, color=MUTED)

    ax.set_title("关键变量缺失率", loc="left", pad=6)
    polish_axis(ax)

    save_figure(fig, "fig_s2_missingness.png",
                extra_note="总观测29,621条。研发强度缺失约3%是主模型样本量差异的主要来源。")


# ── ⑤ VIF柱状图 / fig_s3_vif ─────────────────────────────────────

def save_vif_bar() -> None:
    """多重共线性诊断 VIF - 第2章 §2.2"""
    setup_style()
    df = _load_csv(TABLES_DIAGNOSTIC_DIR / "vif_diagnostics.csv")
    labels_map = {
        "annual_ai_log_w": "年报AI词频",
        "ln_assets_w": "企业规模",
        "leverage_w": "资产负债率",
        "rd_intensity_w": "研发强度",
        "firm_age_w": "企业年龄",
        "revenue_growth_w": "营收增长率",
    }
    df["label"] = df["variable"].map(labels_map)
    df = df.sort_values("vif", ascending=True)

    fig, ax = plt.subplots(figsize=(FIGURE_W, 2.8))
    y_pos = np.arange(len(df))
    colors_vif = [BLUE if v < 5 else (TEAL if v < 10 else ORANGE) for v in df["vif"]]
    ax.barh(y_pos, df["vif"], color=colors_vif, alpha=0.75, height=0.55,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=7.8)

    for i, v in enumerate(df["vif"]):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=7.2, color=INK)

    ax.axvline(5, color=ORANGE, linewidth=0.8, linestyle="--", alpha=0.5, label="VIF=5")
    ax.axvline(10, color=RED_ACCENT, linewidth=0.8, linestyle="--", alpha=0.5, label="VIF=10")
    ax.set_xlabel("VIF")
    ax.set_title("方差膨胀因子 (VIF) 诊断", loc="left", pad=6)
    ax.legend(fontsize=7.2, frameon=False)
    polish_axis(ax)

    save_figure(fig, "fig_s3_vif.png",
                extra_note="所有变量VIF<2，不存在严重多重共线性问题。")


# ── ⑥ 异方差诊断 / fig_s4_heteroskedasticity ────────────────────

def save_heteroskedasticity() -> None:
    """异方差检验结果 - 第2章 §2.2"""
    setup_style()
    df = _load_csv(TABLES_DIAGNOSTIC_DIR / "heteroskedasticity_tests.csv")
    labels = {
        "M1_fe_baseline": "固定效应基线",
        "M2_fe_controls": "加控制变量",
        "M3_fe_controls_industry": "行业FE+控制",
    }
    df["label"] = df["model"].map(labels)

    fig, ax = plt.subplots(figsize=(FIGURE_W, 2.6))
    x = np.arange(len(df))
    ax.bar(x, df["bp_lm_pvalue"], color=BLUE, alpha=0.7, width=0.5,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=7.8)
    ax.set_ylabel("Breusch-Pagan p值")
    ax.axhline(0.01, color=RED_ACCENT, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0.05, color=ORANGE, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(len(df) - 0.5, 0.06, "p=0.05", fontsize=7.0, color=ORANGE, ha="right")
    ax.text(len(df) - 0.5, 0.015, "p=0.01", fontsize=7.0, color=RED_ACCENT, ha="right")
    ax.set_title("Breusch-Pagan 异方差检验", loc="left", pad=6)
    polish_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(1e-300, 1)

    for i, pval in enumerate(df["bp_lm_pvalue"]):
        ax.text(i, max(pval * 1.5, 1e-280), f"p={pval:.2e}", ha="center", fontsize=7.0)

    save_figure(fig, "fig_s4_heteroskedasticity.png",
                extra_note="所有模型p<0.01，拒绝同方差原假设，采用公司聚类稳健标准误。")


# ── ⑦ AI应用分组柱状图 (优化 fig7) ─────────────────────────────

def save_ai_quality_bins_improved() -> None:
    """fig7优化 - 添加每组样本量标注 - §3.1 描述性分析"""
    setup_style()
    panel = _load_panel()
    data = panel[["ai_actual_application_index", "quality_efficiency_index"]].copy()
    data.columns = ["x", "y"]
    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data["x"] = _winsorize(data["x"])
    data["y"] = _winsorize(data["y"])

    work = data.dropna().sort_values("x").reset_index(drop=True)
    groups = 5
    positions = np.arange(len(work))
    work["bin"] = np.minimum(np.floor(positions * groups / len(work)).astype(int), groups - 1)

    summary = work.groupby("bin").agg(
        y_mean=("y", "mean"),
        y_se=("y", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=("y", "size"),
    ).reset_index(drop=True)
    labels = ["低", "较低", "中", "较高", "高"]
    summary["label"] = [f"{labels[i]}\n(n={int(summary['n'].iloc[i])})" for i in range(len(summary))]
    summary["y_low"] = summary["y_mean"] - 1.96 * summary["y_se"]
    summary["y_high"] = summary["y_mean"] + 1.96 * summary["y_se"]
    diff = summary["y_mean"].iloc[-1] - summary["y_mean"].iloc[0]

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.6))
    x_pos = np.arange(len(summary))
    colors_bins = ["#D9D9D9", "#C7D6E5", "#9EC3DD", "#5EA6C8", BLUE]
    ax.bar(x_pos, summary["y_mean"], color=colors_bins, edgecolor="white",
           linewidth=0.6, width=0.72)
    ax.errorbar(x_pos, summary["y_mean"],
                yerr=np.vstack([summary["y_mean"] - summary["y_low"],
                                summary["y_high"] - summary["y_mean"]]),
                fmt="none", color=INK, elinewidth=0.8, capsize=2.4, capthick=0.8)
    ax.plot(x_pos, summary["y_mean"], color=INK, linewidth=1.0, marker="o", markersize=3.0)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.text(0.02, 0.95, f"高组−低组 = {diff:+.3f}  (未控制其他因素)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color=INK)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary["label"], fontsize=7.4)
    ax.set_title("AI应用水平分组与提质增效均值", loc="left", pad=6)
    ax.set_xlabel("实际AI应用水平分组")
    ax.set_ylabel("提质增效指数均值")
    polish_axis(ax)

    save_figure(fig, "fig7_ai_application_quality_bins.png",
                extra_note="按AI应用指数等样本量五等分。未控制年份、行业和企业特征。")


# ── ⑧ 分箱散点图 (20组) / fig11_binned_scatter ───────────────

def save_binned_scatter() -> None:
    """20组等样本量分箱散点 - §3.1 描述性分析"""
    setup_style()
    panel = _load_panel()
    data = panel[["annual_ai_log", "quality_efficiency_index"]].copy()
    data.columns = ["x", "y"]
    for c in ["x", "y"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        data[c] = _winsorize(data[c])
    data = data.dropna()

    bins = 20
    data["bin"] = _equal_count_groups(data["x"], bins)
    binned = data.groupby("bin").agg(
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        y_se=("y", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=("y", "size"),
    ).reset_index(drop=True)
    binned["y_low"] = binned["y_mean"] - 1.96 * binned["y_se"]
    binned["y_high"] = binned["y_mean"] + 1.96 * binned["y_se"]

    # Fit line
    x_arr = data["x"].values
    y_arr = data["y"].values
    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    grid = np.linspace(x_arr.min(), x_arr.max(), 100)
    fit_y = intercept + slope * grid
    r = np.corrcoef(x_arr, y_arr)[0, 1]

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.8))
    # 背景散点 (半透明)
    ax.scatter(data["x"].sample(min(5000, len(data)), random_state=42),
               data["y"].sample(min(5000, len(data)), random_state=42),
               s=0.8, color=MUTED, alpha=0.15, zorder=1)
    # 分箱均值 + 置信区间
    ax.errorbar(binned["x_mean"], binned["y_mean"],
                yerr=np.vstack([binned["y_mean"] - binned["y_low"],
                                binned["y_high"] - binned["y_mean"]]),
                fmt="o", color=BLUE, ecolor=BLUE, elinewidth=0.9,
                capsize=2.0, capthick=0.9, markersize=4.0, zorder=3)
    ax.plot(grid, fit_y, color=ORANGE, linewidth=1.3, zorder=2)
    ax.text(0.97, 0.08, f"r = {r:.3f}\ny = {intercept:.3f} + {slope:.3f}x",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec=MUTED))
    ax.set_xlabel("年报AI词频 (对数)")
    ax.set_ylabel("提质增效指数")
    ax.set_title("AI词频与提质增效：20组等样本量分箱", loc="left", pad=6)
    polish_axis(ax)

    save_figure(fig, "fig11_binned_scatter.png",
                extra_note="未控制其他因素。散点为随机抽样5,000个观测。误差线为95%置信区间。")


# ── ⑨ 控制后偏相关图 / fig12_partial_correlation ──────────────

def save_partial_correlation() -> None:
    """控制后偏相关图 - 第3章 §3.2 基准回归 [最核心新增]"""
    setup_style()
    panel = _load_panel()

    # 选取变量
    y_var = "quality_efficiency_index"
    x_var = "annual_ai_log"
    controls = ["ln_assets", "leverage", "rd_intensity", "firm_age", "revenue_growth"]

    work = panel[[y_var, x_var] + controls + ["year", "industry_2digit"]].copy()
    for col in [y_var, x_var] + controls:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna().reset_index(drop=True)

    # Step 1: Y ~ controls + year FE + industry FE → residuals
    formula_y = f"{y_var} ~ {' + '.join(controls)} + C(year) + C(industry_2digit)"
    mod_y = sm.OLS.from_formula(formula_y, data=work).fit()
    e_y = mod_y.resid

    # Step 2: X ~ controls + year FE + industry FE → residuals
    formula_x = f"{x_var} ~ {' + '.join(controls)} + C(year) + C(industry_2digit)"
    mod_x = sm.OLS.from_formula(formula_x, data=work).fit()
    e_x = mod_x.resid

    # Residuals
    res_df = pd.DataFrame({"e_x": e_x, "e_y": e_y}).dropna()
    res_df["e_x_w"] = _winsorize(res_df["e_x"])
    res_df["e_y_w"] = _winsorize(res_df["e_y"])
    res_df = res_df.dropna()

    bins = 20
    res_df["bin"] = _equal_count_groups(res_df["e_x_w"], bins)
    binned = res_df.groupby("bin").agg(
        x_mean=("e_x_w", "mean"),
        y_mean=("e_y_w", "mean"),
        y_se=("e_y_w", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    ).reset_index(drop=True)
    binned["y_low"] = binned["y_mean"] - 1.96 * binned["y_se"]
    binned["y_high"] = binned["y_mean"] + 1.96 * binned["y_se"]

    x_arr = res_df["e_x_w"].values
    y_arr = res_df["e_y_w"].values
    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    grid = np.linspace(x_arr.min(), x_arr.max(), 100)
    partial_r = np.corrcoef(x_arr, y_arr)[0, 1]

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.8))
    sample_idx = res_df.sample(min(5000, len(res_df)), random_state=42).index
    ax.scatter(res_df.loc[sample_idx, "e_x_w"], res_df.loc[sample_idx, "e_y_w"],
               s=0.7, color=MUTED, alpha=0.14, zorder=1)
    ax.errorbar(binned["x_mean"], binned["y_mean"],
                yerr=np.vstack([binned["y_mean"] - binned["y_low"],
                                binned["y_high"] - binned["y_mean"]]),
                fmt="o", color=BLUE, ecolor=BLUE, elinewidth=0.9,
                capsize=2.0, capthick=0.9, markersize=4.0, zorder=3)
    ax.plot(grid, intercept + slope * grid, color=ORANGE, linewidth=1.3, zorder=2)
    ax.axhline(0, color=GRID, linewidth=0.7, linestyle="--")
    ax.axvline(0, color=GRID, linewidth=0.7, linestyle="--")

    partial_coef = mod_y.params.get(x_var, np.nan)
    ax.text(0.97, 0.08,
            f"偏相关系数 r = {partial_r:.3f}\n回归系数 β = {partial_coef:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec=MUTED))

    ax.set_xlabel("年报AI词频残差 (控制后)")
    ax.set_ylabel("提质增效指数残差 (控制后)")
    ax.set_title("控制后偏相关：AI关注度与提质增效", loc="left", pad=6)
    polish_axis(ax)

    ctrl_str = "、".join(["企业规模", "资产负债率", "研发强度", "企业年龄", "营收增长率"])
    save_figure(fig, "fig12_partial_correlation.png",
                extra_note=f"控制变量：{ctrl_str}，以及年份固定效应和制造业二级行业固定效应。")


# ── ⑩ 稳健性检验汇总森林图 / fig13_robustness_forest ─────────

def save_robustness_forest() -> None:
    """稳健性检验汇总森林图 - §3.3"""
    setup_style()
    t1 = _load_csv(TABLES_FINAL_DIR / "paper_table_1_main_regression.csv")
    t2 = _load_csv(TABLES_FINAL_DIR / "paper_table_2_robustness.csv")

    records = [
        ("主模型", t1[t1["模型"] == "主模型"],
         "主模型", BLUE),
        ("MD&A词频(X)", t1[t1["模型"].str.contains("MD&A")],
         "替换X", TEAL),
        ("综合AI(X)", t1[t1["模型"].str.contains("综合AI")],
         "替换X", TEAL),
        ("TFP(Y)", t2[t2["模型"].str.contains("TFP")],
         "替换Y", ORANGE),
        ("熵权指数(Y)", t2[t2["模型"].str.contains("熵权")],
         "替换Y", ORANGE),
        ("PCA指数(Y)", t2[t2["模型"].str.contains("PCA")],
         "替换Y", ORANGE),
        ("研发缺失处理", t2[t2["模型"].str.contains("研发缺失")],
         "样本处理", MUTED),
    ]

    labels = []
    coefs = []
    ses = []
    colors_list = []
    for label, row, group, color in records:
        if row.empty:
            continue
        cf = _parse_coef(row["系数"].values[0])
        se = _parse_se(row["标准误"].values[0])
        if np.isnan(cf) or np.isnan(se):
            continue
        labels.append(label)
        coefs.append(cf)
        ses.append(se)
        colors_list.append(color)

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.6))
    y = np.arange(len(labels))
    for i, (cf, se, clr, lb) in enumerate(zip(coefs, ses, colors_list, labels)):
        lo = cf - 1.96 * se
        hi = cf + 1.96 * se
        ax.errorbar(cf, i, xerr=[[cf - lo], [hi - cf]], fmt="o",
                    color=clr, ecolor=clr, elinewidth=1.0,
                    capsize=2.5, capthick=1.0, markersize=3.8)
    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_xlabel("估计系数及95%置信区间")

    # 分组标签
    group_positions = [0, 3, 6]
    group_names = ["基准", "替换X", "替换Y", "样本"]
    for gp, gn in zip(group_positions, group_names):
        ax.text(-0.15, gp, gn, ha="right", va="center", fontsize=7.2,
                color=MUTED, fontweight="bold", transform=ax.get_yaxis_transform())

    ax.set_title("稳健性检验：替换X / 替换Y / 样本处理", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")

    save_figure(fig, "fig13_robustness_forest.png",
                extra_note="替换X包括MD&A AI词频和综合AI应用指数；替换Y包括TFP、熵权指数和PCA指数。")


# ── ⑪ 机制路径图 (重做 fig5) ─────────────────────────────────────

def save_mechanism_path() -> None:
    """机制路径图 - §3.4 机制检验"""
    setup_style()
    df = _load_csv(TABLES_MODEL_DIR / "mechanism_regression_results.csv")

    path_map = {
        "ME1_actual_ai_application": ("实际AI应用", BLUE),
        "ME2_asset_turnover": ("资产周转率", BLUE),
        "ME3_labor_productivity": ("劳动生产率", MUTED),
        "ME4_tfp": ("TFP", BLUE),
        "ME5_efficiency_subindex": ("效率子指数", BLUE),
        "ME6_quality_subindex": ("质量子指数", BLUE),
        "ME7_conversion_to_quality_efficiency": ("转化效率", MUTED),
    }

    rows = []
    for model_name, (label, color) in path_map.items():
        row = df[df["model"] == model_name]
        if row.empty:
            continue
        r = row.iloc[0]
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else
                                                   ("*" if r["p_value"] < 0.1 else ""))
        rows.append({
            "label": label,
            "coef": r["coef"],
            "std_err": r["std_err"],
            "sig": sig,
            "color": color if r["p_value"] < 0.05 else MUTED,
            "significant": r["p_value"] < 0.05,
        })

    fig, ax = plt.subplots(figsize=(FIGURE_W, len(rows) * 0.42 + 1.5))
    for i, r in enumerate(rows):
        lower = r["coef"] - 1.96 * r["std_err"]
        upper = r["coef"] + 1.96 * r["std_err"]
        marker = "o" if r["significant"] else "s"
        ax.errorbar(r["coef"], i,
                    xerr=[[r["coef"] - lower], [upper - r["coef"]]],
                    fmt=marker, color=r["color"], ecolor=r["color"],
                    elinewidth=1.0, capsize=2.8, capthick=1.0,
                    markersize=4.0 if r["significant"] else 3.2)
        txt = f"{r['coef']:+.4f}{r['sig']}"
        x_txt = upper + 0.015 if r["coef"] > 0 else lower - 0.015
        ha_txt = "left" if r["coef"] > 0 else "right"
        ax.text(x_txt, i, txt, va="center", ha=ha_txt, fontsize=7.2,
                color=r["color"], fontweight="bold" if r["significant"] else "normal")

    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.8)
    ax.set_xlabel("AI关注度系数及95%置信区间")

    # Legend for significance
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=6,
               label="p<0.05 (显著)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=MUTED, markersize=6,
               label="p≥0.05 (不显著)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=7.2)

    ax.set_title("机制相关性检验：AI关注度如何与经营环节相关", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")

    save_figure(fig, "fig5_mechanism_coefficients.png",
                extra_note="机制检验非严格中介因果检验。实心圆表示p<0.05，空心方块表示不显著。")


# ── ⑫ 配对异质性森林图 (重做 fig6) ────────────────────────────

def save_paired_heterogeneity() -> None:
    """配对异质性森林图 - §3.5"""
    setup_style()
    df = _load_csv(TABLES_MODEL_DIR / "heterogeneity_regression_results.csv")

    pairs = [
        ("state_owned", "non_state_owned", "产权性质", BLUE),
        ("east_region", "non_east_region", "地区", TEAL),
        ("high_tech_manufacturing", "non_high_tech_manufacturing", "技术属性", ORANGE),
        ("high_conversion_efficiency", "low_conversion_efficiency", "转化效率", MUTED),
    ]
    pair_labels = ["国企 vs 非国企", "东部 vs 非东部", "高技术 vs 一般制造", "高转化 vs 低转化"]

    fig, ax = plt.subplots(figsize=(FIGURE_W, 4.5))
    all_data = []
    y = 0
    y_ticks = []
    y_labels = []

    for (g1, g2, cat, color), pl in zip(pairs, pair_labels):
        r1 = df[df["group"] == g1]
        r2 = df[df["group"] == g2]
        if r1.empty or r2.empty:
            continue

        # Group 1
        coef1 = r1.iloc[0]["coef"]; se1 = r1.iloc[0]["std_err"]
        lo1 = coef1 - 1.96 * se1; hi1 = coef1 + 1.96 * se1
        ax.errorbar(coef1, y + 0.15, xerr=[[coef1 - lo1], [hi1 - coef1]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.1,
                    capsize=2.8, capthick=1.1, markersize=4.2)
        y_labels.append(f"{g1}_label"); y_ticks.append(y + 0.15)

        # Group 2
        coef2 = r2.iloc[0]["coef"]; se2 = r2.iloc[0]["std_err"]
        lo2 = coef2 - 1.96 * se2; hi2 = coef2 + 1.96 * se2
        fill_style = "full" if r2.iloc[0]["p_value"] < 0.05 else "none"
        ax.errorbar(coef2, y - 0.15, xerr=[[coef2 - lo2], [hi2 - coef2]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.1,
                    capsize=2.8, capthick=1.1, markersize=4.2,
                    markerfacecolor="white" if fill_style == "none" else color)
        y_labels.append(f"{g2}_label"); y_ticks.append(y - 0.15)

        # Pair bracket
        ax.text(-0.35, y, pl, ha="right", va="center", fontsize=7.8,
                fontweight="bold", color=color, transform=ax.get_yaxis_transform())

        y -= 1.0

    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y_ticks)
    label_map = {
        "state_owned_label": "国企", "non_state_owned_label": "非国企",
        "east_region_label": "东部", "non_east_region_label": "非东部",
        "high_tech_manufacturing_label": "高技术制造",
        "non_high_tech_manufacturing_label": "一般制造",
        "high_conversion_efficiency_label": "高转化效率",
        "low_conversion_efficiency_label": "低转化效率",
    }
    ax.set_yticklabels([label_map.get(lb, lb) for lb in y_labels], fontsize=7.5)
    ax.set_xlabel("AI词频系数及95%置信区间")
    ax.set_title("异质性分析：AI关注度与提质增效", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")

    save_figure(fig, "fig6_heterogeneity_coefficients.png",
                extra_note="按产权性质、地区和制造业技术属性分组。实心点表示p<0.05。")


# ── ⑬ 分位数效应折线图 / fig15_quantile_effect ────────────────

def save_quantile_effect() -> None:
    """分位数效应折线图 - §3.6 扩展模型"""
    setup_style()
    df = _load_csv(TABLES_MODEL_DIR / "quantile_fe_results.csv")

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.2))
    qs = df["quantile"].values
    coefs = df["coef"].values
    ses = df["std_err"].values
    lower = coefs - 1.96 * ses
    upper = coefs + 1.96 * ses

    # 平滑曲线插值
    x_smooth = np.linspace(qs.min(), qs.max(), 50)
    spl_coef = make_interp_spline(qs, coefs, k=2)(x_smooth) if len(qs) >= 3 else coefs

    ax.plot(x_smooth, spl_coef if len(qs) >= 3 else coefs, color=BLUE, linewidth=1.5, zorder=2)
    ax.fill_between(qs, lower, upper, alpha=0.15, color=BLUE, zorder=1)
    ax.errorbar(qs, coefs, yerr=np.vstack([coefs - lower, upper - coefs]),
                fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.0,
                capsize=3.0, capthick=1.0, markersize=5.0, zorder=3)

    for q, c, lo, up in zip(qs, coefs, lower, upper):
        ax.text(q + 0.01, up + 0.003, f"{c:.4f}", ha="center", fontsize=7.2, color=INK)

    ax.axhline(0, color=MUTED, linewidth=0.7, linestyle="--")
    ax.set_xlabel("分位点")
    ax.set_ylabel("AI词频系数")
    ax.set_title("分位数固定效应：AI系数随提质增效水平变化", loc="left", pad=6)
    ax.set_xticks(qs)
    ax.set_xticklabels([f"Q{int(q*100)}" for q in qs])
    polish_axis(ax)

    if len(coefs) >= 2:
        decline = (coefs[0] - coefs[-1]) / abs(coefs[0]) * 100 if abs(coefs[0]) > 1e-8 else 0
        ax.text(0.97, 0.05, f"Q25→Q75 下降约 {decline:.0f}%",
                transform=ax.transAxes, ha="right", fontsize=7.4, color=INK)

    save_figure(fig, "fig15_quantile_effect.png",
                extra_note="分位数固定效应模型，控制年份和行业固定效应及企业特征。")


# ── ⑭ 模型对比图 / fig16_model_comparison ─────────────────────────

def save_model_comparison() -> None:
    """固定效应 vs 交叉拟合 vs 企业FE - §3.6"""
    setup_style()
    t1 = _load_csv(TABLES_FINAL_DIR / "paper_table_1_main_regression.csv")
    cf = _load_csv(TABLES_MODEL_DIR / "cross_fitted_partial_linear_results.csv")
    ta = _load_csv(TABLES_FINAL_DIR / "paper_table_A1_boundary.csv")

    main_row = t1[t1["模型"] == "主模型"]
    fe_row = ta[ta["模型"].str.contains("主模型")]

    models = [
        ("固定效应主模型", _parse_coef(main_row["系数"].values[0]),
         _parse_se(main_row["标准误"].values[0]), BLUE),
        ("交叉拟合部分线性", cf.iloc[0]["coef"], cf.iloc[0]["std_err"], TEAL),
    ]
    if not fe_row.empty:
        models.append(
            ("企业FE边界模型",
             _parse_coef(fe_row["系数"].values[0]),
             _parse_se(fe_row["标准误"].values[0]), MUTED)
        )

    fig, ax = plt.subplots(figsize=(FIGURE_W, 2.2))
    for i, (label, coef, se, color) in enumerate(models):
        lo = coef - 1.96 * se
        hi = coef + 1.96 * se
        ax.errorbar(coef, i, xerr=[[coef - lo], [hi - coef]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.3,
                    capsize=3.5, capthick=1.3, markersize=5.5)
        sig = "***" if abs(coef / se) > 2.58 else ("**" if abs(coef / se) > 1.96 else "")
        ax.text(hi + 0.005, i, f"{coef:+.4f}{sig}",
                va="center", fontsize=7.8, color=color, fontweight="bold")

    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([m[0] for m in models], fontsize=8.0)
    ax.set_xlabel("AI词频系数及95%置信区间")
    ax.set_title("模型设定对比：固定效应 vs 交叉拟合 vs 企业FE", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")

    save_figure(fig, "fig16_model_comparison.png",
                extra_note="交叉拟合部分线性模型使用随机森林净化控制变量后估计AI净效应。")


# ── ⑮ 边界检验对比图 / fig17_boundary_comparison ──────────────

def save_boundary_comparison() -> None:
    """主模型 vs 企业FE边界检验 - §4.2 结论边界 [核心新增]"""
    setup_style()
    t1 = _load_csv(TABLES_FINAL_DIR / "paper_table_1_main_regression.csv")
    ta = _load_csv(TABLES_FINAL_DIR / "paper_table_A1_boundary.csv")

    pairs = [
        ("主模型 (行业FE)", t1[t1["模型"] == "主模型"],
         ta[ta["模型"].str.contains("主模型")], BLUE),
        ("MD&A词频", t1[t1["模型"].str.contains("MD&A")],
         ta[ta["模型"].str.contains("MD&A")], TEAL),
        ("综合AI指数", t1[t1["模型"].str.contains("综合AI")],
         ta[ta["模型"].str.contains("综合AI")], ORANGE),
        ("TFP", None, ta[ta["模型"].str.contains("TFP")], MUTED),
    ]

    rows_data = []
    for name, row_main, row_fe, color in pairs:
        if row_main is not None and not row_main.empty:
            coef_m = _parse_coef(row_main["系数"].values[0])
            se_m = _parse_se(row_main["标准误"].values[0])
        else:
            coef_m, se_m = np.nan, np.nan
        if not row_fe.empty:
            coef_f = _parse_coef(row_fe["系数"].values[0])
            se_f = _parse_se(row_fe["标准误"].values[0])
        else:
            coef_f, se_f = np.nan, np.nan
        rows_data.append((name, coef_m, se_m, coef_f, se_f, color))

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.5))
    y_positions = []
    y_labels = []

    for i, (name, cm, sm, cf, sf, color) in enumerate(rows_data):
        y_base = i * 1.0

        # 主模型 (行业FE)
        if not np.isnan(cm):
            lo = cm - 1.96 * sm
            hi = cm + 1.96 * sm
            ax.errorbar(cm, y_base + 0.15, xerr=[[cm - lo], [hi - cm]],
                        fmt="o", color=color, ecolor=color, elinewidth=1.0,
                        capsize=2.8, capthick=1.0, markersize=4.5)
            ax.text(hi + 0.01, y_base + 0.15, f"{cm:+.4f}", va="center",
                    fontsize=7.2, color=color, fontweight="bold")
            y_positions.append(y_base + 0.15)
            y_labels.append(f"{name}\n(行业FE)")

        # 企业FE
        if not np.isnan(cf):
            lo = cf - 1.96 * sf
            hi = cf + 1.96 * sf
            is_sig = abs(cf / sf) > 1.96 if sf > 0 else False
            marker_style = "o" if is_sig else "s"
            marker_fc = color if is_sig else "white"
            ax.errorbar(cf, y_base - 0.15, xerr=[[cf - lo], [hi - cf]],
                        fmt=marker_style, color=color, ecolor=color, elinewidth=1.0,
                        capsize=2.8, capthick=1.0, markersize=4.5,
                        markerfacecolor=marker_fc)
            ax.text(hi + 0.01, y_base - 0.15, f"{cf:+.4f}", va="center",
                    fontsize=7.2, color=color,
                    fontweight="bold" if is_sig else "normal")
            y_positions.append(y_base - 0.15)
            y_labels.append(f"         (企业FE)")

    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7.2)
    ax.set_xlabel("系数及95%置信区间")
    ax.set_title("边界检验：行业固定效应 vs 企业固定效应", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")

    # 添加说明
    ax.text(0.98, 0.02, "实心=显著  空心=不显著\n企业FE下系数明显减弱",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.0, color=RED_ACCENT)

    save_figure(fig, "fig17_boundary_comparison.png",
                extra_note="企业固定效应下系数明显减弱，部分系数不再显著。结论应谨慎解释为统计相关关系。")


# ── ⑯ LOESS局部拟合 / fig14_loess_fit ─────────────────────────

def save_loess_fit() -> None:
    """LOESS局部拟合曲线 - §3.4 机制检验"""
    setup_style()
    panel = _load_panel()

    x_col = "annual_ai_log"
    y_col = "quality_efficiency_index"
    data = panel[[x_col, y_col]].copy()
    for c in [x_col, y_col]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        data[c] = _winsorize(data[c])
    data = data.dropna()

    x = data[x_col].values
    y = data[y_col].values

    # LOESS via lowess
    from statsmodels.nonparametric.smoothers_lowess import lowess
    loess_result = lowess(y, x, frac=0.3, return_sorted=True)
    x_loess = loess_result[:, 0]
    y_loess = loess_result[:, 1]

    # Linear fit for comparison
    slope, intercept = np.polyfit(x, y, deg=1)
    y_linear = intercept + slope * x_loess

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.8))
    sample_idx = np.random.RandomState(42).choice(len(data), min(4000, len(data)), replace=False)
    ax.scatter(x[sample_idx], y[sample_idx], s=0.6, color=MUTED, alpha=0.12, zorder=1)

    ax.plot(x_loess, y_loess, color=BLUE, linewidth=1.8, label="LOESS (f=0.3)", zorder=3)
    ax.plot(x_loess, y_linear, color=ORANGE, linewidth=1.2, linestyle="--",
            label="线性拟合", zorder=2)

    ax.set_xlabel("年报AI词频 (对数)")
    ax.set_ylabel("提质增效指数")
    ax.set_title("局部加权回归 (LOESS)：AI关注度与提质增效", loc="left", pad=6)
    ax.legend(frameon=False, fontsize=7.8)
    polish_axis(ax)

    save_figure(fig, "fig14_loess_fit.png",
                extra_note="LOESS带宽frac=0.3。曲线整体近似线性上升，非线性二次项不显著。")


# ── ⑰ 变量筛选热力图 / fig_s5_xy_screening ─────────────────────

def save_xy_screening_heatmap() -> None:
    """候选X-Y组合筛选热力图 - 附录"""
    setup_style()
    df = _load_csv(TABLES_DIAGNOSTIC_DIR / "candidate_xy_screening_results.csv")

    # 提取核心X-Y组合 (主表中的AI相关结果)
    x_vars = ["annual_ai_log_w", "mda_ai_log_w", "ai_application_index_w",
              "ai_actual_application_index_w", "ai_invest_log_w", "ai_invest_level_w"]
    y_vars = ["quality_efficiency_index_w", "tfp_w", "quality_efficiency_entropy_w",
              "asset_turnover_w", "quality_efficiency_pca_w",
              "ai_actual_application_index_w"]

    x_labels = {"annual_ai_log_w": "年报AI词频", "mda_ai_log_w": "MD&A词频",
                "ai_application_index_w": "综合AI指数", "ai_actual_application_index_w": "实际AI应用",
                "ai_invest_log_w": "AI投资(log)", "ai_invest_level_w": "AI投资水平"}
    y_labels = {"quality_efficiency_index_w": "提质增效", "tfp_w": "TFP",
                "quality_efficiency_entropy_w": "熵权指数", "asset_turnover_w": "资产周转",
                "quality_efficiency_pca_w": "PCA指数", "ai_actual_application_index_w": "实际AI应用"}

    matrix = np.zeros((len(y_vars), len(x_vars)))
    annot = np.empty((len(y_vars), len(x_vars)), dtype=object)

    for i, yv in enumerate(y_vars):
        for j, xv in enumerate(x_vars):
            row = df[(df["y"].str.contains(yv.replace("_w", ""))) &
                     (df["x"].str.contains(xv.replace("_w", "")))].head(1)
            if not row.empty:
                t_val = row.iloc[0]["t_value"]
                matrix[i, j] = min(abs(t_val), 15)
                annot[i, j] = f"t={t_val:.1f}" if row.iloc[0]["positive_and_significant"] else "ns"
            else:
                matrix[i, j] = 0
                annot[i, j] = "-"

    fig, ax = plt.subplots(figsize=(FIGURE_W + 0.6, 3.8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=15)
    ax.set_xticks(np.arange(len(x_vars)))
    ax.set_yticks(np.arange(len(y_vars)))
    ax.set_xticklabels([x_labels.get(v, v) for v in x_vars], rotation=30, ha="right", fontsize=7.2)
    ax.set_yticklabels([y_labels.get(v, v) for v in y_vars], fontsize=7.2)

    for i in range(len(y_vars)):
        for j in range(len(x_vars)):
            color = "white" if matrix[i, j] > 5 else INK
            ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=6.8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="|t-value|")
    ax.set_title("候选X-Y组合筛选：|t值|热力图", loc="left", pad=6)
    ax.set_xlabel("核心解释变量 (X)")
    ax.set_ylabel("被解释变量 (Y)")

    save_figure(fig, "fig_s5_xy_screening.png",
                extra_note="仅展示主要候选X-Y组合。颜色越深表示t值越大。ns表示不显著或方向为负。")


# ── ⑱ 调节效应图 / fig_s6_conversion_moderation ───────────────

def save_conversion_moderation() -> None:
    """转化效率调节效应图 - 附录"""
    setup_style()
    df = _load_csv(TABLES_MODEL_DIR / "conversion_moderation_results.csv")

    rows_data = []
    for _, r in df.iterrows():
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else
                                                   ("*" if r["p_value"] < 0.1 else ""))
        is_sig = r["p_value"] < 0.1
        rows_data.append({
            "label": r["x"],
            "coef": r["coef"],
            "std_err": r["std_err"],
            "sig": sig,
            "significant": is_sig,
        })

    fig, ax = plt.subplots(figsize=(FIGURE_W, 2.4))
    label_map = {
        "annual_ai_log_w": "低转化×AI词频",
        "ai_word_high_conversion": "AI词频×高转化 (交互项)",
        "high_conversion_efficiency": "高转化效率虚拟变量",
    }

    for i, row in enumerate(rows_data):
        label = label_map.get(row["label"], row["label"])
        lo = row["coef"] - 1.96 * row["std_err"]
        hi = row["coef"] + 1.96 * row["std_err"]
        color = BLUE if row["significant"] else MUTED
        marker = "o" if row["significant"] else "s"
        ax.errorbar(row["coef"], i, xerr=[[row["coef"] - lo], [hi - row["coef"]]],
                    fmt=marker, color=color, ecolor=color, elinewidth=1.0,
                    capsize=2.8, capthick=1.0, markersize=4.2,
                    markerfacecolor="white" if not row["significant"] else color)
        txt = f"{row['coef']:+.4f}{row['sig']}"
        ax.text(hi + 0.01, i, txt, va="center", fontsize=7.4,
                color=color, fontweight="bold" if row["significant"] else "normal")

    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(np.arange(len(rows_data)))
    ax.set_yticklabels([label_map.get(r["label"], r["label"]) for r in rows_data], fontsize=7.6)
    ax.set_xlabel("系数及95%置信区间")
    ax.set_title("AI落地转化效率调节效应 (探索性分析)", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")
    ax.text(0.98, 0.02, "交互项不显著 (p>0.1)\n转化效率调节效应证据不足",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.0, color=RED_ACCENT)

    save_figure(fig, "fig_s6_conversion_moderation.png",
                extra_note="交互项不显著，转化效率调节效应证据不足。属探索性分析，不宜作为核心结论。")


# ── 附加：DID事件研究图 (已有fig8，这里做优化版) ───────────────

def save_did_event_study() -> None:
    """DID事件研究图 (优化) - §4.1"""
    setup_style()
    path = TABLES_MODEL_DIR / "did_policy_event_study_results.csv"
    if not path.exists():
        LOGGER.warning("DID结果文件不存在，跳过")
        return
    df = _load_csv(path)

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.5))
    df = df.sort_values("event_time")

    ax.errorbar(df["event_time"], df["coef"],
                yerr=1.96 * df["std_err"],
                fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.0,
                capsize=2.5, capthick=1.0, markersize=3.8)
    ax.axhline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.axvline(-0.5, color=RED_ACCENT, linewidth=0.8, linestyle="--", alpha=0.6,
               label="政策时点 (2017)")

    ax.set_xlabel("相对政策时点 (年)")
    ax.set_ylabel("估计系数")
    ax.set_title("DID事件研究：AI政策冲击与提质增效", loc="left", pad=6)
    ax.legend(fontsize=7.5, frameon=False)
    polish_axis(ax)

    save_figure(fig, "fig8_did_policy_event_study.png",
                extra_note="基准年为2016。虚线表示2017年政策时点。当前结果不能支撑严格因果关系。")


# ═══════════════════════════════════════════════════════════════════
#  一键运行入口
# ═══════════════════════════════════════════════════════════════════

def run_all() -> None:
    """一键生成全部论文图表。"""
    LOGGER.info("=" * 60)
    LOGGER.info("开始生成全部论文图表 ...")
    LOGGER.info("=" * 60)

    print("\n[1/17] 数据筛选流程图...")
    save_sample_flow()

    print("[2/17] 提质增效指数结构图...")
    save_index_structure()

    print("[3/17] 描述性统计组合图...")
    save_descriptive_dashboard()

    print("[4/17] 缺失率诊断图...")
    save_missingness_bar()

    print("[5/17] VIF诊断图...")
    save_vif_bar()

    print("[6/17] 异方差诊断图...")
    save_heteroskedasticity()

    print("[7/17] AI应用分组图 (优化)...")
    save_ai_quality_bins_improved()

    print("[8/17] 分箱散点图 (20组)...")
    save_binned_scatter()

    print("[9/17] 控制后偏相关图...")
    save_partial_correlation()

    print("[10/17] 稳健性检验森林图...")
    save_robustness_forest()

    print("[11/17] 机制路径图 (重做)...")
    save_mechanism_path()

    print("[12/17] 配对异质性森林图 (重做)...")
    save_paired_heterogeneity()

    print("[13/17] 分位数效应折线图...")
    save_quantile_effect()

    print("[14/17] 模型对比图...")
    save_model_comparison()

    print("[15/17] 边界检验对比图...")
    save_boundary_comparison()

    print("[16/17] LOESS局部拟合...")
    save_loess_fit()

    print("[17/17] 其他 (DID + 热力图 + 调节效应)...")
    save_did_event_study()
    save_xy_screening_heatmap()
    save_conversion_moderation()

    LOGGER.info("=" * 60)
    LOGGER.info("全部图表生成完毕！输出目录: %s", FIGURES_DIR)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    run_all()
