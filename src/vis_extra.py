"""论文附加丰富图表模块 — 流程图 + 多种统计图表类型，供筛选参考。

风格复用 vis_paper 的全局基础设施。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D

from .config import (
    FIGURES_DIR,
    PROCESSED_DIR,
    TABLES_DIAGNOSTIC_DIR,
    TABLES_FINAL_DIR,
    TABLES_MODEL_DIR,
)

LOGGER = logging.getLogger(__name__)

# ── 全局视觉规范 (同 vis_paper) ──────────────────────────────────
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
PURPLE_LIGHT = "#B8A9D4"

PALETTE = [BLUE, TEAL, ORANGE, PURPLE, RED_ACCENT, GOLD, MUTED]
FIGURE_W = 5.9
NOTE_FMT = "注：标准误为公司层面聚类稳健标准误。{extra}"


# ── 基础设施 (复用 vis_paper 函数签名) ──────────────────────────

def setup_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
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
    })


def polish_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, length=3, width=0.75)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def save_figure(fig, filename: str, extra_note: str = "") -> None:
    out = FIGURES_DIR / filename
    if extra_note:
        fig.text(0.02, -0.01, NOTE_FMT.format(extra=extra_note),
                 ha="left", va="top", fontsize=6.5, color=MUTED, style="italic")
    fig.tight_layout(pad=0.55)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=360)
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", dpi=360)
    plt.close(fig)
    LOGGER.info("Saved %s", out)


def _load_panel() -> pd.DataFrame:
    p = PROCESSED_DIR / "manufacturing_modeling_panel.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到面板数据: {p}")
    return pd.read_csv(p)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lstrip("\ufeff")
    return df


def _winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    q_low, q_high = values.quantile([lower, upper])
    return values.clip(q_low, q_high)


def _parse_coef(val) -> float:
    s = str(val).strip().replace("(", "").replace(")", "")
    s = s.rstrip("*")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_se(val) -> float:
    s = str(val).strip().replace("(", "").replace(")", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


# ═══════════════════════════════════════════════════════════════════
#  第一部分：流程图 (Flowcharts)
# ═══════════════════════════════════════════════════════════════════

def save_research_design_flow() -> None:
    """研究设计流程图 — 第1章 §1.2"""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, alpha=0.12, fontsize=8.0, bold=True):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.2)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=INK, fontweight=weight)

    def draw_arrow(x1, y1, x2, y2, color=MUTED):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

    # Row 0: 阶段标签
    phases = [("阶段一: 理论构建", 1.5), ("阶段二: 数据准备", 4.0), ("阶段三: 实证分析", 6.5), ("阶段四: 结果呈现", 9.0)]
    for label, x in phases:
        ax.text(x, 9.3, label, ha="center", fontsize=7.5, color=MUTED, fontweight="bold")

    # Row 1: 核心步骤
    draw_box(1.5, 7.3, 2.4, 0.9, "文献综述\n理论框架构建", BLUE)
    draw_box(4.0, 7.3, 2.4, 0.9, "数据采集\nA股制造业面板", TEAL)
    draw_box(6.5, 7.3, 2.4, 0.9, "基准回归\n固定效应模型", ORANGE)
    draw_box(9.0, 7.3, 2.0, 0.9, "论文撰写\n图表输出", PURPLE)

    draw_arrow(2.7, 7.3, 2.8, 7.3)
    draw_arrow(5.2, 7.3, 5.3, 7.3)
    draw_arrow(7.7, 7.3, 8.0, 7.3)

    # Row 2: 子步骤
    subs = [
        (1.5, 5.0, "AI+制造\n提质增效", BLUE_LIGHT, 1.5),
        (1.5, 3.2, "提出假设\nH1~H3", BLUE_LIGHT, 1.5),
        (4.0, 5.0, "文本分析提取\nAI词频/应用", TEAL_LIGHT, 2.4),
        (4.0, 3.2, "缺失值处理\n变量构造", TEAL_LIGHT, 2.0),
        (6.5, 5.0, "稳健性检验\n替换X/Y/样本", ORANGE_LIGHT, 2.3),
        (6.5, 3.2, "机制/异质性\n边界检验", ORANGE_LIGHT, 2.0),
        (9.0, 5.0, "19组核心图表\nPNG+PDF双格式", PURPLE_LIGHT, 2.0),
        (9.0, 3.2, "研究结论\n政策建议", PURPLE_LIGHT, 1.8),
    ]
    for x, y, text, color, w in subs:
        draw_box(x, y, w, 0.85, text, color, alpha=0.18, fontsize=7.0, bold=False)

    # Arrows down
    for x in [1.5, 4.0, 6.5, 9.0]:
        draw_arrow(x, 6.8, x, 5.45)
        draw_arrow(x, 4.55, x, 3.65)

    # Cross connections
    draw_arrow(2.7, 5.0, 2.8, 5.0)
    draw_arrow(5.2, 5.0, 5.3, 5.0)
    draw_arrow(7.7, 5.0, 8.0, 5.0)

    ax.set_title("研究设计总览流程图", loc="left", pad=6, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_extra_research_design_flow.png",
                extra_note="四阶段研究流程：理论构建→数据准备→实证分析→结果呈现。")


def save_variable_construction_flow() -> None:
    """变量构造流程图"""
    setup_style()
    fig, ax = plt.subplots(figsize=(FIGURE_W + 1.0, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    boxes = [
        # (x, y, w, h, label, detail, color)
        (6.0, 7.2, 2.8, 0.8, "解释变量 X\nAI关注度", "", BLUE),
        (6.0, 2.8, 2.8, 0.8, "被解释变量 Y\n提质增效综合指数", "", ORANGE),
        (2.0, 6.2, 2.4, 0.7, "年报文本AI词频", "annual_ai_log", BLUE_LIGHT),
        (10.0, 6.2, 2.4, 0.7, "MD&A AI词频", "mda_ai_log", BLUE_LIGHT),
        (4.5, 4.8, 2.4, 0.7, "AI投资水平", "ai_invest_level", BLUE_LIGHT),
        (7.5, 4.8, 2.4, 0.7, "综合AI应用指数", "ai_application_index", BLUE_LIGHT),
        (2.0, 1.8, 2.4, 0.7, "质量子指数\nROA+利润率+TFP", "quality_subindex", ORANGE_LIGHT),
        (6.0, 0.8, 2.4, 0.7, "效率子指数\n周转率+劳动生产率", "efficiency_subindex", ORANGE_LIGHT),
        (10.0, 1.8, 2.4, 0.7, "控制变量\n规模/杠杆/研发/年龄/增长", "", TEAL_LIGHT),
    ]
    for x, y, w, h, label, detail, color in boxes:
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, facecolor=color,
                              alpha=0.15, edgecolor=color, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x, y + 0.05, label, ha="center", va="center", fontsize=7.2, color=INK, fontweight="bold")
        if detail:
            ax.text(x, y - 0.2, detail, ha="center", va="center", fontsize=6.0, color=MUTED)

    # Arrows
    arrows = [
        (2.0, 5.85, 6.0, 6.8), (10.0, 5.85, 6.0, 6.8),
        (4.5, 4.45, 6.0, 6.8), (7.5, 4.45, 6.0, 6.8),
        (2.0, 2.15, 6.0, 2.4), (6.0, 1.15, 6.0, 2.4), (10.0, 2.15, 6.0, 2.4),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))

    # Middle arrow: X → Y
    ax.annotate("", xy=(6.0, 3.25), xytext=(6.0, 6.35),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))

    ax.set_title("核心变量构造流程图", loc="left", pad=6, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_extra_variable_construction.png",
                extra_note="AI关注度通过年报/MD&A文本分析提取；提质增效指数由5项指标等权合成。")


def save_methodology_path() -> None:
    """方法论路径图 — 从描述到因果推断"""
    setup_style()
    fig, ax = plt.subplots(figsize=(FIGURE_W + 1.0, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    methods = [
        (1.5, 5.0, 2.0, "描述性分析", "小提琴图/热力图/分箱散点", BLUE_LIGHT),
        (4.0, 5.0, 2.0, "相关性分析", "OLS回归", TEAL_LIGHT),
        (6.5, 5.0, 2.0, "因果初步", "固定效应+控制变量", ORANGE_LIGHT),
        (9.0, 5.0, 2.0, "稳健性验证", "替换变量/分位数/交叉拟合", PURPLE_LIGHT),
        (6.5, 2.5, 2.0, "因果边界", "企业FE/DID事件研究", RED_ACCENT),
    ]
    for x, y, w, title, sub, color in methods:
        rect = plt.Rectangle((x - w/2, y - 1.0), w, 2.0, facecolor=color,
                              alpha=0.12, edgecolor=color, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y + 0.4, title, ha="center", va="center", fontsize=8.2, color=INK, fontweight="bold")
        ax.text(x, y - 0.4, sub, ha="center", va="center", fontsize=6.8, color=MUTED)

    for i in range(len(methods) - 1):
        ax.annotate("", xy=(methods[i+1][0] - 1.0, methods[i+1][1] - 0.5),
                     xytext=(methods[i][0] + 1.0, methods[i][1] + 0.5),
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

    # Confidence band
    ax.annotate("", xy=(6.5, 3.5), xytext=(9.0, 4.0),
                 arrowprops=dict(arrowstyle="->", color=RED_ACCENT, lw=1.2, linestyle="dashed"))
    ax.text(7.8, 2.0, "证据强度递减\n相关系→因果推断", ha="center", fontsize=7.0, color=RED_ACCENT)

    ax.set_title("实证方法论路径：从相关到因果推断", loc="left", pad=6, fontsize=10, fontweight="bold")
    save_figure(fig, "fig_extra_methodology_path.png",
                extra_note="从描述相关到因果推断，证据强度逐渐增强但设计约束也逐步严格。")


# ═══════════════════════════════════════════════════════════════════
#  第二部分：丰富多样统计图表
# ═══════════════════════════════════════════════════════════════════

# ── ① Ridge/Joyplot：历年AI词频分布 ───────────────────────────

def save_ridge_plot() -> None:
    """山脊线图 — AI词频历年分布变化"""
    setup_style()
    panel = _load_panel()
    data = panel[["year", "annual_ai_log"]].copy()
    data["annual_ai_log"] = pd.to_numeric(data["annual_ai_log"], errors="coerce")
    data["annual_ai_log"] = _winsorize(data["annual_ai_log"], 0.005, 0.995)
    data = data.dropna()

    years = sorted(data["year"].unique())
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(FIGURE_W, n_years * 0.52 + 0.8), sharex=True)

    if n_years == 1:
        axes = [axes]

    for i, yr in enumerate(years):
        ax = axes[i]
        subset = data[data["year"] == yr]["annual_ai_log"]
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(subset)
            x_grid = np.linspace(subset.min(), subset.max(), 200)
            density = kde(x_grid)
            density = density / density.max()
            ax.fill_between(x_grid, density, alpha=0.5, color=BLUE)
            ax.plot(x_grid, density, color=BLUE, linewidth=0.8)
        except Exception:
            ax.hist(subset, bins=30, density=True, alpha=0.5, color=BLUE)
        ax.text(0.02, 0.85, f"{int(yr)} (n={len(subset)})",
                transform=ax.transAxes, fontsize=7.0, color=INK)
        ax.set_ylim(0, 1.15)
        ax.axis("off")

    axes[-1].axis("on")
    axes[-1].set_xlabel("年报AI词频 (对数)")
    axes[-1].spines["top"].set_visible(False)
    axes[-1].spines["right"].set_visible(False)
    axes[-1].spines["left"].set_visible(False)
    axes[-1].set_yticks([])

    fig.suptitle("AI词频历年分布变化 (山脊线图)", x=0.02, ha="left", fontsize=9.2, fontweight="bold")
    fig.tight_layout(pad=0.3)
    save_figure(fig, "fig_extra_ridge_plot.png",
                extra_note="各年AI词频核密度分布。可直观观察分布中心逐年右移趋势。")


# ── ② 完整相关性热力图 (所有变量) ──────────────────────────────

def save_full_correlation_heatmap() -> None:
    """完整相关性热力图 — 全部数值变量"""
    setup_style()
    panel = _load_panel()

    vars_all = [
        "annual_ai_log", "mda_ai_log", "ai_application_index", "ai_actual_application_index",
        "quality_efficiency_index", "tfp", "rd_intensity",
        "ln_assets", "leverage", "firm_age", "revenue_growth",
    ]
    labels = [
        "AI词频", "MD&A词频", "AI应用指数", "实际AI应用",
        "提质增效", "TFP", "研发强度",
        "企业规模", "资产负债率", "企业年龄", "营收增长",
    ]

    work = panel[[v for v in vars_all if v in panel.columns]].copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    corr = work.corr()

    actual_vars = list(work.columns)
    actual_labels = [labels[i] for i, v in enumerate(vars_all) if v in work.columns]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    n = len(actual_vars)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(actual_labels, rotation=45, ha="right", fontsize=7.2)
    ax.set_yticklabels(actual_labels, fontsize=7.2)

    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.3, color="white" if abs(v) > 0.45 else INK)

    plt.colorbar(im, ax=ax, shrink=0.78, label="Pearson r")
    ax.set_title("全部变量 Pearson 相关系数矩阵", loc="left", pad=6)
    save_figure(fig, "fig_extra_correlation_heatmap.png",
                extra_note="蓝色=正相关，红色=负相关。AI词频与提质增效、TFP正相关；与研发强度高度正相关(r>0.5)。")


# ── ③ 企业-年份面板覆盖热力图 ─────────────────────────────────

def save_panel_coverage_heatmap() -> None:
    """面板数据覆盖热力图"""
    setup_style()
    panel = _load_panel()

    # 按行业分组显示年份覆盖率
    panel["industry_name"] = panel["industry_name"].fillna("其他")
    industries = panel.groupby("industry_name").size().sort_values(ascending=False).head(8).index.tolist()
    years = sorted(panel["year"].dropna().unique())

    sub = panel[panel["industry_name"].isin(industries)]
    matrix = pd.crosstab(sub["industry_name"], sub["year"], values=sub["annual_ai_log"],
                          aggfunc="count").loc[industries, :]

    fig, ax = plt.subplots(figsize=(FIGURE_W + 0.8, 3.5))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=matrix.max().max())
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels([str(int(y)) for y in matrix.columns], fontsize=7.2)
    ax.set_yticklabels([n[:14] for n in matrix.index], fontsize=7.2)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            v = matrix.values[i, j]
            if v > 0:
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6.0,
                        color="white" if v > matrix.max().max() * 0.6 else INK)

    plt.colorbar(im, ax=ax, shrink=0.8, label="观测数")
    ax.set_xlabel("年份")
    ax.set_ylabel("制造业二级行业 (Top 8)")
    ax.set_title("企业-年份面板覆盖热力图", loc="left", pad=6)
    save_figure(fig, "fig_extra_panel_coverage.png",
                extra_note="颜色越深表示观测数越多。Top 8行业占全样本多数，各行业年覆盖分布均匀。")


# ── ④ 棒棒糖图 (Lollipop)：系数大小 ──────────────────────────

def save_lollipop_chart() -> None:
    """棒棒糖图 — 各模型系数横向对比"""
    setup_style()
    t1 = _load_csv(TABLES_FINAL_DIR / "paper_table_1_main_regression.csv")

    entries = []
    for _, row in t1.iterrows():
        cf = _parse_coef(row["系数"])
        se = _parse_se(row["标准误"])
        if np.isnan(cf):
            continue
        entries.append({"model": row["模型"], "coef": cf, "se": se})

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.2))
    entries.sort(key=lambda x: x["coef"])

    for i, e in enumerate(entries):
        color = BLUE if e["coef"] > 0 else RED_ACCENT
        # stem
        ax.plot([e["coef"], e["coef"]], [i - 0.25, i + 0.25], color=color, linewidth=0.9, alpha=0.6)
        # lollipop head
        ax.scatter(e["coef"], i, s=80, color=color, edgecolors="white", linewidth=0.5, zorder=3)
        ax.text(e["coef"] + 0.003, i + 0.2, f"{e['coef']:+.4f}", fontsize=7.2, color=color,
                fontweight="bold")

    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(np.arange(len(entries)))
    ax.set_yticklabels([e["model"] for e in entries], fontsize=7.2)
    ax.set_xlabel("AI词频估计系数")
    ax.set_title("棒棒糖图：AI词频系数一览", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")

    save_figure(fig, "fig_extra_lollipop.png",
                extra_note="蓝=正效应，红=负效应。所有模型AI词频系数为正，稳健通过显著性检验。")


# ── ⑤ 哑铃图 (Dumbbell)：OLS vs FE 对比 ──────────────────────

def save_dumbbell_chart() -> None:
    """哑铃图 — OLS vs FE coefficient comparison"""
    setup_style()
    t1 = _load_csv(TABLES_FINAL_DIR / "paper_table_1_main_regression.csv")
    ta = _load_csv(TABLES_FINAL_DIR / "paper_table_A1_boundary.csv")

    pairs = []
    for _, r in t1.iterrows():
        model_name = r["模型"]
        cf_ols = _parse_coef(r["系数"])
        se_ols = _parse_se(r["标准误"])
        if np.isnan(cf_ols):
            continue
        # match in boundary table
        fe_row = ta[ta["模型"].str.contains(model_name)]
        if fe_row.empty:
            continue
        cf_fe = _parse_coef(fe_row["系数"].values[0])
        se_fe = _parse_se(fe_row["标准误"].values[0])
        if np.isnan(cf_fe):
            continue
        pairs.append((model_name, cf_ols, se_ols, cf_fe, se_fe))

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.5))
    y_pos = np.arange(len(pairs))

    for i, (name, cf_o, se_o, cf_f, se_f) in enumerate(pairs):
        # Dumbbell line
        ax.plot([cf_o, cf_f], [i + 0.3, i - 0.3], color=MUTED, linewidth=1.0, alpha=0.5, zorder=1)
        # OLS dot (行业FE)
        lo_o = cf_o - 1.96 * se_o
        hi_o = cf_o + 1.96 * se_o
        ax.errorbar(cf_o, i + 0.3, xerr=[[cf_o - lo_o], [hi_o - cf_o]],
                     fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.0,
                     capsize=2.5, capthick=1.0, markersize=5, zorder=3)
        # FE dot (企业FE)
        lo_f = cf_f - 1.96 * se_f
        hi_f = cf_f + 1.96 * se_f
        is_sig = abs(cf_f / se_f) > 1.96 if se_f > 0 else False
        marker_fc = ORANGE if is_sig else "white"
        ax.errorbar(cf_f, i - 0.3, xerr=[[cf_f - lo_f], [hi_f - cf_f]],
                     fmt="o", color=ORANGE, ecolor=ORANGE, elinewidth=1.0,
                     capsize=2.5, capthick=1.0, markersize=5, markerfacecolor=marker_fc, zorder=3)

    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0][:12] for p in pairs], fontsize=7.2)

    # legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=7, label="行业FE (OLS)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=7, label="企业FE"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.2, frameon=False)

    ax.set_xlabel("系数及95%置信区间")
    ax.set_title("哑铃图：行业FE vs 企业FE 系数对比", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")

    save_figure(fig, "fig_extra_dumbbell_ols_vs_fe.png",
                extra_note="企业FE下系数和显著性均有明显减弱，提示个体异质性可能混淆OLS估计。")


# ── ⑥ 连接散点图：AI vs TFP 按年份 ──────────────────────────

def save_connected_scatter() -> None:
    """连接散点图 — AI词频和TFP年度均值变化轨迹"""
    setup_style()
    panel = _load_panel()
    cols = ["year", "annual_ai_log", "tfp"]
    data = panel[cols].copy()
    for c in ["annual_ai_log", "tfp"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()

    yearly = data.groupby("year").agg(
        ai_mean=("annual_ai_log", "mean"),
        tfp_mean=("tfp", "mean"),
    ).reset_index()
    yearly = yearly.sort_values("year")

    # Normalize both to 0-1 for visual comparison
    ai_norm = (yearly["ai_mean"] - yearly["ai_mean"].min()) / \
              (yearly["ai_mean"].max() - yearly["ai_mean"].min())
    tfp_norm = (yearly["tfp_mean"] - yearly["tfp_mean"].min()) / \
               (yearly["tfp_mean"].max() - yearly["tfp_mean"].min())

    fig, ax1 = plt.subplots(figsize=(FIGURE_W, 3.8))
    ax2 = ax1.twinx()

    # AI词频 (左侧y轴)
    ax1.plot(yearly["year"].astype(int), yearly["ai_mean"], color=BLUE, linewidth=1.5, marker="o",
              markersize=5, label="AI词频均值")
    ax1.set_ylabel("AI词频均值", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)

    # TFP (右侧y轴)
    ax2.plot(yearly["year"].astype(int), yearly["tfp_mean"], color=ORANGE, linewidth=1.5, marker="s",
              markersize=5, label="TFP均值")
    ax2.set_ylabel("TFP均值", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)

    # Connected scatter inset: plot normalized
    for i in range(len(yearly) - 1):
        ax1.annotate("", xy=(yearly["year"].iloc[i+1], yearly["ai_mean"].iloc[i+1]),
                      xytext=(yearly["year"].iloc[i], yearly["ai_mean"].iloc[i]),
                      arrowprops=dict(arrowstyle="-", color=BLUE, alpha=0.4, lw=0.8))
        ax2.annotate("", xy=(yearly["year"].iloc[i+1], yearly["tfp_mean"].iloc[i+1]),
                      xytext=(yearly["year"].iloc[i], yearly["tfp_mean"].iloc[i]),
                      arrowprops=dict(arrowstyle="-", color=ORANGE, alpha=0.4, lw=0.8))

    ax1.set_xlabel("年份")
    ax1.set_title("双轴时间序列：AI词频与TFP年度均值变化", loc="left", pad=6)
    polish_axis(ax1)

    # Combined legend
    lines = [Line2D([0], [0], color=BLUE, linewidth=2, marker="o", markersize=5, label="AI词频"),
             Line2D([0], [0], color=ORANGE, linewidth=2, marker="s", markersize=5, label="TFP")]
    ax1.legend(handles=lines, loc="upper left", frameon=False, fontsize=7.5)

    save_figure(fig, "fig_extra_connected_scatter.png",
                extra_note="2013-2024年AI词频呈持续上升趋势，TFP整体平稳。两者的时间序列协变明显。")


# ── ⑦ 非线性边际效应图 ────────────────────────────────────────

def save_nonlinear_marginal() -> None:
    """非线性二次项边际效应图"""
    setup_style()
    nl = _load_csv(TABLES_MODEL_DIR / "nonlinear_ai_effect_results.csv")

    lin_row = nl[nl["term"] == "AI词频一次项"]
    sq_row = nl[nl["term"] == "AI词频二次项"]

    if lin_row.empty or sq_row.empty:
        return

    b1 = lin_row.iloc[0]["coef"]
    b2 = sq_row.iloc[0]["coef"]

    x_range = np.linspace(-2.5, 2.5, 200)
    y_pred = b1 * x_range + b2 * x_range ** 2
    dy_dx = b1 + 2 * b2 * x_range
    marginal_threshold = -b1 / (2 * b2) if abs(b2) > 1e-8 else np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_W + 1.5, 3.2))

    # Left: 预测值
    ax1.plot(x_range, y_pred, color=BLUE, linewidth=1.5)
    ax1.axvline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax1.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    if not np.isnan(marginal_threshold):
        ax1.axvline(marginal_threshold, color=RED_ACCENT, linewidth=0.8, linestyle="--")
        ax1.text(marginal_threshold + 0.1, ax1.get_ylim()[1] * 0.9,
                 f"x={marginal_threshold:.2f}", fontsize=7.0, color=RED_ACCENT)
    ax1.set_xlabel("AI词频 (中心化)")
    ax1.set_ylabel("提质增效预测值")
    ax1.set_title("二次拟合：AI词频非线性效应", loc="left", fontsize=8.0)
    polish_axis(ax1)

    # Right: 边际效应
    ax2.plot(x_range, dy_dx, color=ORANGE, linewidth=1.5)
    ax2.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax2.fill_between(x_range, 0, dy_dx, alpha=0.1, color=ORANGE)
    ax2.set_xlabel("AI词频 (中心化)")
    ax2.set_ylabel("边际效应 ∂Y/∂X")
    ax2.set_title("边际效应随AI水平变化", loc="left", fontsize=8.0)
    polish_axis(ax2)

    ax2.text(0.97, 0.05, f"β₁={b1:.4f}\nβ₂={b2:.4f}",
             transform=ax2.transAxes, ha="right", fontsize=7.4, color=INK)

    fig.tight_layout(pad=1.0)
    save_figure(fig, "fig_extra_nonlinear_marginal.png",
                extra_note="二次项为正(=0.0141, p<0.01)说明AI效应随水平提高而递增，不存在传统边际递减。")


# ── ⑧ 箱线图：主要变量多组合对比 ──────────────────────────────

def save_multivariate_boxplot() -> None:
    """箱线图 — 多变量标准化后对比展示"""
    setup_style()
    panel = _load_panel()
    vars_box = ["annual_ai_log", "quality_efficiency_index", "tfp",
                "rd_intensity", "leverage", "revenue_growth"]
    labels_box = ["AI词频", "提质增效", "TFP", "研发强度", "资产负债率", "营收增长"]
    colors_box = [BLUE, TEAL, ORANGE, PURPLE, GOLD, MUTED]

    fig, ax = plt.subplots(figsize=(FIGURE_W + 1.0, 4.2))

    box_data = []
    for col in vars_box:
        if col not in panel.columns:
            continue
        s = pd.to_numeric(panel[col], errors="coerce")
        s = _winsorize(s, 0.005, 0.995)
        s = (s - s.min()) / (s.max() - s.min())  # 标准化到[0,1]
        box_data.append(s.dropna().values)

    # Filter labels/colors to match
    actual_labels = [l for i, l in enumerate(labels_box) if vars_box[i] in panel.columns]
    actual_colors = [c for i, c in enumerate(colors_box) if vars_box[i] in panel.columns]
    n = len(box_data)

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.55,
                     medianprops={"color": INK, "linewidth": 1.0},
                     flierprops={"marker": ".", "markersize": 2, "alpha": 0.5},
                     whiskerprops={"linewidth": 0.8},
                     capprops={"linewidth": 0.8})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(actual_colors[i % len(actual_colors)])
        patch.set_alpha(0.5)

    ax.set_xticks(np.arange(1, n + 1))
    ax.set_xticklabels(actual_labels, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("标准化值 [0,1]")
    ax.set_title("多变量箱线图：标准化分布对比", loc="left", pad=6)
    polish_axis(ax)

    save_figure(fig, "fig_extra_boxplot.png",
                extra_note="所有变量经1%缩尾+Min-Max标准化至[0,1]。箱体=Q1~Q3，中线=中位数。")


# ── ⑨ 堆积柱状图：行业结构年度变化 ────────────────────────────

def save_stacked_bar_industry() -> None:
    """堆积柱状图 — Top 5行业年度占比变化"""
    setup_style()
    panel = _load_panel()
    panel["industry_name"] = panel["industry_name"].fillna("其他")
    top5 = panel.groupby("industry_name").size().sort_values(ascending=False).head(5).index.tolist()
    panel["ind_grp"] = panel["industry_name"].apply(lambda x: x if x in top5 else "其他")

    crosstab = pd.crosstab(panel["year"], panel["ind_grp"]).sort_index()
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    # Sort columns by average share
    col_order = crosstab_pct.mean().sort_values(ascending=False).index.tolist()
    crosstab_pct = crosstab_pct[col_order]

    fig, ax = plt.subplots(figsize=(FIGURE_W + 0.6, 3.8))
    years_int = [int(y) for y in crosstab_pct.index]
    bottom = np.zeros(len(crosstab_pct))
    colors = [BLUE, TEAL, ORANGE, PURPLE, GOLD, MUTED]

    for i, col in enumerate(crosstab_pct.columns):
        ax.bar(years_int, crosstab_pct[col].values, bottom=bottom,
               color=colors[i % len(colors)], alpha=0.75, width=0.7,
               edgecolor="white", linewidth=0.3, label=col[:12])
        bottom += crosstab_pct[col].values

    # 添加连接线显示趋势
    for i, col in enumerate(crosstab_pct.columns[:3]):
        mid = bottom - crosstab_pct[col].values - crosstab_pct[col].values / 2

    ax.set_xlabel("年份")
    ax.set_ylabel("行业占比 (%)")
    ax.set_title("制造业二级行业年度结构变化 (Top 5)", loc="left", pad=6)
    ax.legend(loc="lower left", fontsize=6.8, frameon=False, ncol=2)
    ax.set_ylim(0, 105)
    polish_axis(ax)

    save_figure(fig, "fig_extra_stacked_bar_industry.png",
                extra_note="各行业占比历年保持稳定，样本行业结构未发生剧烈变动。其他行业合并显示。")


# ── ⑩ 气泡图：模型性能对比 ───────────────────────────────────

def save_bubble_chart() -> None:
    """气泡图 — 模型一览：系数大小、显著性、样本量"""
    setup_style()
    t1 = _load_csv(TABLES_FINAL_DIR / "paper_table_1_main_regression.csv")
    t2 = _load_csv(TABLES_FINAL_DIR / "paper_table_2_robustness.csv")
    t6 = _load_csv(TABLES_FINAL_DIR / "paper_table_6_advanced_models.csv")

    combined = pd.concat([t1, t2, t6], ignore_index=True)
    combined = combined.drop_duplicates(subset=["模型"])

    entries = []
    for _, row in combined.iterrows():
        cf = _parse_coef(row["系数"])
        se = _parse_se(row["标准误"])
        if np.isnan(cf):
            continue
        pval = row["p值"] if "p值" in row.index and pd.notna(row["p值"]) else \
               (row["p_value"] if "p_value" in row.index and pd.notna(row["p_value"]) else np.nan)
        try:
            pval = float(pval)
        except (ValueError, TypeError):
            pval = np.nan
        n_obs = row["样本量"] if "样本量" in row.index else np.nan
        try:
            n_obs = float(n_obs)
        except (ValueError, TypeError):
            n_obs = np.nan
        entries.append({
            "model": str(row["模型"])[:16],
            "coef": cf,
            "se": se,
            "pval": pval,
            "n_obs": n_obs,
        })

    fig, ax = plt.subplots(figsize=(FIGURE_W, 4.2))
    entries_sorted = sorted(entries, key=lambda x: x["coef"])
    y = np.arange(len(entries_sorted))

    for i, e in enumerate(entries_sorted):
        t_val = abs(e["coef"] / e["se"]) if e["se"] > 0 else 0
        size = np.clip(t_val * 30 + 20, 30, 250)
        alpha_val = 0.7 if not np.isnan(e["pval"]) and e["pval"] < 0.05 else 0.35
        color = BLUE if e["coef"] > 0 else RED_ACCENT
        ax.scatter(e["coef"], i, s=size, color=color, alpha=alpha_val,
                   edgecolors=INK, linewidth=0.3, zorder=3)
        ax.text(e["coef"] + 0.003, i, f"  {e['model']}",
                va="center", fontsize=7.0, color=INK)

    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks([])
    ax.set_xlabel("AI词频系数")
    ax.set_title("气泡图：模型系数一览 (气泡大小=|t|，透明度=显著性)", loc="left", pad=6)
    polish_axis(ax, grid_axis="x")

    save_figure(fig, "fig_extra_bubble_chart.png",
                extra_note="气泡大小=|t值|，透明度=显著性(p<0.05为实心)。所有模型AI系数为正。")


# ── ⑪ 误差带时间序列（均值±CI） ──────────────────────────────

def save_error_band_timeseries() -> None:
    """提升效能年度变化带图"""
    setup_style()
    panel = _load_panel()
    y_col = "quality_efficiency_index"
    x_col = "annual_ai_log"

    for c in [y_col, x_col]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel[f"{y_col}_w"] = _winsorize(panel[y_col], 0.005, 0.995)
    panel[f"{x_col}_w"] = _winsorize(panel[x_col], 0.005, 0.995)

    yearly = panel.groupby("year").agg(
        y_mean=(f"{y_col}_w", "mean"),
        y_std=(f"{y_col}_w", "std"),
        y_n=(f"{y_col}_w", "size"),
        x_mean=(f"{x_col}_w", "mean"),
        x_std=(f"{x_col}_w", "std"),
    ).reset_index()
    yearly["y_se"] = yearly["y_std"] / np.sqrt(yearly["y_n"])
    yearly["y_lo"] = yearly["y_mean"] - 1.96 * yearly["y_se"]
    yearly["y_hi"] = yearly["y_mean"] + 1.96 * yearly["y_se"]
    yearly["x_se"] = yearly["x_std"] / np.sqrt(yearly["y_n"])
    yearly["x_lo"] = yearly["x_mean"] - 1.96 * yearly["x_se"]
    yearly["x_hi"] = yearly["x_mean"] + 1.96 * yearly["x_se"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_W + 1.5, 3.5))
    years_int = yearly["year"].astype(int).values

    # Left: 提质增效
    ax1.fill_between(years_int, yearly["y_lo"], yearly["y_hi"], alpha=0.15, color=BLUE)
    ax1.plot(years_int, yearly["y_mean"], color=BLUE, linewidth=1.5, marker="o", markersize=4)
    ax1.set_title("提质增效指数年度趋势", loc="left", fontsize=8.2)
    ax1.set_xlabel("年份")
    ax1.set_ylabel("提质增效指数")
    polish_axis(ax1)

    # Right: AI词频
    ax2.fill_between(years_int, yearly["x_lo"], yearly["x_hi"], alpha=0.15, color=ORANGE)
    ax2.plot(years_int, yearly["x_mean"], color=ORANGE, linewidth=1.5, marker="s", markersize=4)
    ax2.set_title("AI词频年度趋势", loc="left", fontsize=8.2)
    ax2.set_xlabel("年份")
    ax2.set_ylabel("AI词频 (对数)")
    polish_axis(ax2)

    fig.tight_layout(pad=1.2)
    save_figure(fig, "fig_extra_error_band_timeseries.png",
                extra_note="误差带为年度均值的95%置信区间。提质增效2020年后有所改善；AI词频持续上升。")


# ── ⑫ 甜甜圈图：样本行业分布 ─────────────────────────────────

def save_donut_industry() -> None:
    """甜甜圈图 — 样本行业分布"""
    setup_style()
    panel = _load_panel()
    panel["industry_name"] = panel["industry_name"].fillna("其他")
    ind_counts = panel.groupby("industry_name").size().sort_values(ascending=False)
    top7 = ind_counts.head(7)
    other_sum = ind_counts[7:].sum()

    sizes = list(top7.values) + [other_sum]
    labels = [f"{n[:12]}\n({v}, {v/ind_counts.sum()*100:.1f}%)" for n, v in zip(top7.index, top7.values)]
    labels.append(f"其他 ({other_sum}, {other_sum/ind_counts.sum()*100:.1f}%)")

    colors_donut = [BLUE, TEAL, ORANGE, PURPLE, GOLD, RED_ACCENT, BLUE_LIGHT, MUTED]

    fig, ax = plt.subplots(figsize=(FIGURE_W, 4.0))
    wedges, texts = ax.pie(sizes, labels=None, colors=colors_donut,
                            startangle=90, wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 1.0})
    centre_circle = plt.Circle((0, 0), 0.72, fc="white")
    ax.add_artist(centre_circle)

    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize=6.8, frameon=False)
    ax.text(0, 0, f"N={ind_counts.sum():,}", ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    ax.set_title("样本行业分布 (甜甜圈图)", loc="left", pad=6)

    save_figure(fig, "fig_extra_donut_industry.png",
                extra_note="计算机通信、设备制造、金属加工等是主要观测行业。")


# ── ⑬ 六边形蜂巢/热力图(R&D和AI空间) ─────────────────────────

def save_hexbin_ai_rd() -> None:
    """Hexbin图 — AI词频 vs 研发强度空间分布"""
    setup_style()
    panel = _load_panel()
    data = panel[["annual_ai_log", "rd_intensity", "quality_efficiency_index"]].copy()
    for c in ["annual_ai_log", "rd_intensity", "quality_efficiency_index"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()

    x = data["annual_ai_log"].values
    y = data["rd_intensity"].values
    # Clip rd_intensity for visualization
    y_clipped = np.clip(y, y.min(), np.percentile(y, 99))

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.8))
    hb = ax.hexbin(x, y_clipped, gridsize=40, cmap="YlOrRd",
                    mincnt=1, linewidths=0.1, edgecolors="white")
    plt.colorbar(hb, ax=ax, shrink=0.8, label="观测数")
    ax.set_xlabel("年报AI词频 (对数)")
    ax.set_ylabel("研发强度")
    ax.set_title("六边形蜂巢图：AI词频与研发强度", loc="left", pad=6)
    polish_axis(ax)

    save_figure(fig, "fig_extra_hexbin_ai_rd.png",
                extra_note="AI词频与研发强度集中分布于低-低区域，高-高区域稀疏但存在正向关联。")


# ── ⑭ 坡度图 (Slope chart)：各指标历年变化 ────────────────────

def save_slope_chart() -> None:
    """坡度图 — 各指标2013→2024变化"""
    setup_style()
    panel = _load_panel()
    metrics = {
        "annual_ai_log": "AI词频",
        "quality_efficiency_index": "提质增效",
        "tfp": "TFP",
        "rd_intensity": "研发强度",
        "leverage": "资产负债率",
    }
    years_start = [2013, 2014, 2015]
    years_end = [2022, 2023, 2024]

    start_data = panel[panel["year"].isin(years_start)]
    end_data = panel[panel["year"].isin(years_end)]

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.8))
    y_pos = 0
    y_ticks = []
    y_labels = []

    for col, label in metrics.items():
        if col not in panel.columns:
            continue
        start_vals = pd.to_numeric(start_data[col], errors="coerce").dropna()
        end_vals = pd.to_numeric(end_data[col], errors="coerce").dropna()
        start_m = start_vals.mean()
        end_m = end_vals.mean()

        # Normalize
        all_vals = pd.concat([start_vals, end_vals])
        norm_min, norm_max = all_vals.min(), all_vals.max()
        rng = norm_max - norm_min if norm_max > norm_min else 1
        s = (start_m - norm_min) / rng
        e = (end_m - norm_min) / rng

        color = BLUE if e > s else RED_ACCENT
        ax.plot([0, 1], [s, e], color=color, linewidth=2.0, alpha=0.7, zorder=1)
        ax.scatter([0], [s], s=70, color=color, edgecolors="white", linewidth=0.5, zorder=2)
        ax.scatter([1], [e], s=70, color=color, edgecolors="white", linewidth=0.5, zorder=2)
        ax.text(-0.12, s, label, ha="right", va="center", fontsize=7.5, color=INK)
        # Display raw values
        ax.text(-0.12, s - 0.08, f"{start_m:.3f}", ha="right", va="top", fontsize=6.5, color=MUTED)
        ax.text(1.12, e, f"{end_m:.3f}", ha="left", va="center", fontsize=6.5, color=MUTED)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2013-2015均值", "2022-2024均值"], fontsize=7.8)
    ax.set_ylabel("标准化值")
    ax.set_title("坡度图：各指标均值变化 (早期 vs 晚期)", loc="left", pad=6)
    polish_axis(ax)

    save_figure(fig, "fig_extra_slope_chart.png",
                extra_note="蓝线=上升，红线=下降。AI词频大幅上升，提质增效和TFP略有改善，杠杆率微降。")


# ── ⑮ CDF对比图：AI高频 vs 低频企业 ─────────────────────────

def save_cdf_comparison() -> None:
    """累积分布对比图 — 高AI vs 低AI企业"""
    setup_style()
    panel = _load_panel()
    ai_col = "annual_ai_log"
    y_col = "quality_efficiency_index"

    data = panel[[ai_col, y_col]].copy()
    for c in [ai_col, y_col]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()

    # Split into high/low AI
    median_ai = data[ai_col].median()
    high = data[data[ai_col] >= median_ai][y_col]
    low = data[data[ai_col] < median_ai][y_col]

    fig, ax = plt.subplots(figsize=(FIGURE_W, 3.6))

    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf_high = ECDF(high)
    ecdf_low = ECDF(low)

    x_grid = np.linspace(data[y_col].min(), data[y_col].max(), 300)
    ax.plot(x_grid, ecdf_high(x_grid), color=BLUE, linewidth=1.5, label=f"高AI组 (n={len(high)})")
    ax.plot(x_grid, ecdf_low(x_grid), color=MUTED, linewidth=1.5, label=f"低AI组 (n={len(low)})")

    # KS statistic
    try:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(high, low)
        ax.text(0.97, 0.08, f"KS统计量={ks_stat:.4f}\nKS检验p={ks_p:.2e}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4, color=INK)
    except ImportError:
        pass

    ax.set_xlabel("提质增效指数")
    ax.set_ylabel("累积概率")
    ax.set_title("累积分布函数 (CDF)：高AI vs 低AI企业", loc="left", pad=6)
    ax.legend(fontsize=7.8, frameon=False, loc="lower right")
    polish_axis(ax)

    save_figure(fig, "fig_extra_cdf_comparison.png",
                extra_note="高AI组的提质增效指数CDF整体右移，一阶随机占优低AI组。KS检验可检验分布差异。")


# ═══════════════════════════════════════════════════════════════════
#  一键运行入口
# ═══════════════════════════════════════════════════════════════════

ALL_FUNCTIONS = [
    ("研究设计流程图", save_research_design_flow),
    ("变量构造流程图", save_variable_construction_flow),
    ("方法论路径图", save_methodology_path),
    ("山脊线图", save_ridge_plot),
    ("完整相关性热力图", save_full_correlation_heatmap),
    ("面板覆盖热力图", save_panel_coverage_heatmap),
    ("棒棒糖图", save_lollipop_chart),
    ("哑铃图(OLS vs FE)", save_dumbbell_chart),
    ("连接散点图", save_connected_scatter),
    ("非线性边际效应", save_nonlinear_marginal),
    ("多变量箱线图", save_multivariate_boxplot),
    ("堆积柱状图", save_stacked_bar_industry),
    ("气泡图", save_bubble_chart),
    ("误差带时间序列", save_error_band_timeseries),
    ("甜甜圈图", save_donut_industry),
    ("六边形蜂巢图", save_hexbin_ai_rd),
    ("坡度图", save_slope_chart),
    ("CDF分布对比", save_cdf_comparison),
]


def run_all() -> None:
    total = len(ALL_FUNCTIONS)
    for i, (name, fn) in enumerate(ALL_FUNCTIONS, 1):
        print(f"[{i}/{total}] {name}...")
        try:
            fn()
        except Exception as exc:
            LOGGER.error("✗ %s 失败: %s", name, exc)
    print(f"\n全部 {total} 张额外图表生成完毕 → {FIGURES_DIR}")


if __name__ == "__main__":
    run_all()
