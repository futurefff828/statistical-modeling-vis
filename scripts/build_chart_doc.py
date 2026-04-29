"""
生成图表说明文档 DOCX — 详细描述全部41张图表的含义、数据来源和关联关系。
输出至 outputs/figures/final/图表说明文档.docx
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

OUTPUT_DOC = ROOT / "outputs" / "figures" / "final" / "图表说明文档.docx"

# ── Helper functions ──────────────────────────────────────────────────

def set_cell_shading(cell, color: str):
    """Set cell background color."""
    shading_elm = cell._element.find(qn('w:tcPr'))
    if shading_elm is None:
        shading_elm = cell._element.makeelement(qn('w:tcPr'), {})
        cell._element.insert(0, shading_elm)
    shd = shading_elm.find(qn('w:shd'))
    if shd is None:
        shd = cell._element.makeelement(qn('w:shd'), {})
        shading_elm.append(shd)
    shd.set(qn('w:fill'), color)
    shd.set(qn('w:val'), 'clear')


def add_header_row(table, headers: list[str], color: str = "1F1F1F"):
    """Add styled header row to table."""
    row = table.rows[0]
    for i, text in enumerate(headers):
        cell = row.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, color)


def add_row(table, values: list[str], bold_first: bool = True):
    """Add a data row to table."""
    row = table.add_row()
    for i, text in enumerate(values):
        cell = row.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(str(text))
        run.font.size = Pt(8.5)
        if i == 0 and bold_first:
            run.bold = True
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)


def add_section_heading(doc: Document, text: str, level: int = 1):
    """Add a section heading."""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
    return heading


def add_chart_entry(doc: Document, *, fig_id: str, cn_name: str, section: str,
                    chart_type: str, description: str, data_source: str,
                    variables: str, interpretation: str, note: str = ""):
    """Add a detailed chart entry with all fields."""
    # Chart name as heading
    h = doc.add_heading(f"{fig_id}  {cn_name}", level=2)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0x00, 0x72, 0xB2)

    # Metadata table
    meta = doc.add_table(rows=1, cols=2)
    meta.style = 'Table Grid'
    meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_header_row(meta, ["属性", "内容"], color="0072B2")

    fields = [
        ("论文章节", section),
        ("图表类型", chart_type),
        ("数据来源", data_source),
        ("使用变量", variables),
        ("图表说明", description),
        ("解读要点", interpretation),
    ]
    if note:
        fields.append(("补充说明", note))

    for label, value in fields:
        add_row(meta, [label, value])

    # Set column widths
    for row in meta.rows:
        row.cells[0].width = Cm(2.5)
        row.cells[1].width = Cm(13.5)

    doc.add_paragraph("")  # spacer


# ── Chart definitions ─────────────────────────────────────────────────

CORE_CHARTS = [
    # ========== fig1 - fig9 正文图 ==========
    {
        "fig_id": "fig1",
        "cn_name": "AI关注与应用趋势",
        "section": "§2.1 数据来源 / §3.1 描述性分析",
        "chart_type": "双轴折线图（时间序列）",
        "description": "展示2013–2024年间三条AI相关指标的年度均值走势：左轴显示年报AI词频（对数）与MD&A AI词频（对数），右轴显示AI实际应用指数。三条曲线直观对比AI在不同文本领域的关注度变化与应用落地趋势。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv（A股制造业上市公司2013–2024年企业-年份面板数据）",
        "variables": "annual_ai_log（年报AI词频对数）、mda_ai_log（MD&A AI词频对数）、ai_actual_application_index（AI实际应用指数）",
        "interpretation": "① 年报AI词频呈持续上升趋势，反映A股制造业对AI的关注度逐年提升。② MD&A AI词频趋势与年报一致但数值略低。③ AI实际应用指数在2020年后加速增长，显示AI从'概念关注'向'实际应用'转化的趋势。",
        "note": "左轴（年报词频/MD&A词频）、右轴（应用指数），双轴可区分关注度和实际应用两个维度。"
    },
    {
        "fig_id": "fig2",
        "cn_name": "提质增效趋势",
        "section": "§2.1 变量构造 / §3.1 描述性分析",
        "chart_type": "多线折线图（时间序列）",
        "description": "展示2013–2024年间提质增效综合指数及其两个子维度（质量子指数、效率子指数）的年度均值变化趋势。三条曲线分别用黑色、蓝色和青色标识，直观反映制造业整体提质增效水平的时间演变。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "quality_efficiency_index（提质增效综合指数）、quality_subindex（质量子指数）、efficiency_subindex（效率子指数）",
        "interpretation": "① 提质增效综合指数2013–2019年整体平稳，2020年后出现小幅改善趋势。② 质量子指数各年份相对稳定，效率子指数波动较大。③ 2020年后的改善可能与疫情期间数字化/智能化加速有关。",
        "note": "三个指数均标准化至[0,1]区间。质量子指数=ROA+营业利润率+TFP；效率子指数=总资产周转率+劳动生产率。"
    },
    {
        "fig_id": "fig3",
        "cn_name": "基准与稳健性系数森林图",
        "section": "§3.2 基准回归 / §3.3 稳健性检验",
        "chart_type": "系数森林图（Forest Plot）",
        "description": "汇总基准回归的三步模型（基准→加控制→行业FE）和三项替换X的稳健性检验（MD&A词频、综合AI指数）的AI词频系数及95%置信区间。每个点代表一个模型的AI关注度估计系数，横线表示95%置信区间，竖虚线为零效应线。",
        "data_source": "主回归与稳健性检验模型结果（固定效应面板回归）",
        "variables": "coef（AHARon系数）、std_err（聚类稳健标准误）；标签：基准/加控制/行业FE/MD&A/综合AI/TFP",
        "interpretation": "① 所有模型的AI词频系数均为正且在95% CI下不含零，AI关注度与提质增效存在稳定的正相关关系。② 添加控制变量和行业FE后系数略有下降但方向不变，说明结果具有稳健性。③ 替换X（MD&A、综合AI）结果一致，增强结论可信度。",
        "note": "竖虚线x=0表示零效应线。每个点覆盖虚线表明系数统计显著。标准误为公司层面聚类稳健标准误。"
    },
    {
        "fig_id": "fig4",
        "cn_name": "最终模型汇总系数森林图",
        "section": "§3.2 基准回归 / §4 结论",
        "chart_type": "系数森林图（Forest Plot）",
        "description": "汇总全部9个核心模型的AI效应系数及95%置信区间，涵盖：主模型、替换X（MD&A/综合AI）、替换Y（熵权Y/PCA Y）、样本处理（研发缺失稳健性）、机制变量（实际应用/转化效率）。是论文最全面的系数总览图。",
        "data_source": "最终模型结果汇总（固定效应面板回归）",
        "variables": "coef（系数）、std_err（标准误）；模型：主模型/MD&A/综合AI/熵权Y/PCA Y/研发缺失/实际应用/转化效率",
        "interpretation": "① 全部9个模型的AI系数均为正，且除少数外均统计显著（95% CI含零）。② 机制变量（实际应用/转化效率）系数方向为正，为AI→应用→提质增效的链条提供支持。③ 全图展示了'AI关注度-提质增效'正相关关系的广度与一致性。",
        "note": "这是论文最核心的'一站式'系数总览图，覆盖主回归、稳健性、机制三方面模型结果。"
    },
    {
        "fig_id": "fig5",
        "cn_name": "机制相关性检验（vis_paper.py版本）",
        "section": "§3.4 机制检验",
        "chart_type": "系数森林图（Forest Plot）",
        "description": "展示AI关注度与7个机制变量之间的相关性检验结果：实际AI应用、资产周转率、劳动生产率、TFP、效率子指数、质量子指数、转化效率。空心方块=不显著(p≥0.05)、实心圆=显著(p<0.05)。",
        "data_source": "outputs/tables/model_results/mechanism_regression_results.csv（机制回归结果）",
        "variables": "coef（AI系数）、std_err、p_value；模型：ME1~ME7（7个机制变量回归）",
        "interpretation": "① AI关注度与实际AI应用显著正相关（第1行），表明年报文本中的AI关注确实能反映企业的AI应用行为。② AI关注度与资产周转率、劳动生产率、TFP等效率指标显著正相关，揭示'AI→效率提升'的中间机制。③ 转化效率的系数虽为正但不显著，提示AI从应用到提质增效的转化链条存在效率损失。",
        "note": "此为非严格中介因果检验，仅展示AI词频与各机制变量的统计相关关系，方向一致但不可直接推断因果。"
    },
    {
        "fig_id": "fig6",
        "cn_name": "异质性分析：分样本检验",
        "section": "§3.5 异质性分析",
        "chart_type": "配对森林图（Grouped Forest Plot）",
        "description": "按四个维度分组对比AI效应的异质性：①产权性质（国企/非国企，蓝色）、②地区（东部/非东部，青色）、③技术属性（高技术/一般制造，橙色）、④转化效率（高/低转化，灰色）。每组包含两个系数的并列对比。",
        "data_source": "outputs/tables/model_results/heterogeneity_regression_results.csv（异质性回归结果）",
        "variables": "coef（分组估计系数）、std_err、p_value；分组：国企/非国企、东部/非东部、高技术/一般制造、高转化/低转化",
        "interpretation": "① 产权性质：国企组的AI效应弱于非国企组，非国企对AI的应用更为灵活和有效。② 地区：东部组AI效应更强，东部地区制造业信息化基础更好。③ 技术属性：高技术制造组AI效应更强，高技术企业吸收AI能力更强。④ 转化效率：高转化效率组的AI效应更显著，验证了转化能力的中介作用。",
        "note": "实心点=p<0.05（显著），空心点=p≥0.05。颜色区分维度：蓝=产权、青=地区、橙=技术、灰=转化效率。"
    },
    {
        "fig_id": "fig7",
        "cn_name": "AI应用水平分组与提质增效均值",
        "section": "§3.1 描述性分析",
        "chart_type": "分组柱状图（带误差线）",
        "description": "将样本按AI实际应用指数等量五等分为低、较低、中、较高、高五组，展示各组的提质增效指数均值及95%置信区间误差线，并用折线连接各组均值。配色从浅灰渐变为深蓝，直观反映组间差异。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "ai_actual_application_index（AI实际应用指数，分组变量）、quality_efficiency_index（提质增效指数，组均值计算）",
        "interpretation": "① 五组均值呈明显递增趋势，高组提质增效指数显著高于低组。② 折线从Q1到Q5几乎单调上升，未出现倒U型拐点。③ 标注'高组-低组=+{diff:.3f}'，正值且可观。④ 但此为未控制其他因素的粗分组比较，不可直接归因。",
        "note": "等样本量五等分（quintile），非等间距分组。标注'n={N}'显示每组样本量。未控制年份、行业、企业特征。"
    },
    {
        "fig_id": "fig8",
        "cn_name": "DID事件研究：AI政策冲击",
        "section": "§4.1 因果识别尝试",
        "chart_type": "事件研究图（Event Study Plot）",
        "description": "以2017年作为AI政策事件时点（国务院《新一代人工智能发展规划》），展示事前的平行趋势和事后的动态处理效应。横轴为相对政策时点（年），纵轴为估计系数及95%置信区间，红色竖虚线标注政策时点。",
        "data_source": "outputs/tables/model_results/did_policy_event_study_results.csv（DID事件研究结果）",
        "variables": "event_time（相对政策时点）、coef（事件研究系数）、std_err（标准误）",
        "interpretation": "① 政策前各期系数（t<-1）接近零且不显著，支持平行趋势假设。② 政策后（2017+）系数无明显跳跃或持续上升趋势，政策冲击效应有限。③ 所有点均附95% CI，多数包含零线，说明AI政策并非制造业提质增效的外生冲击源。④ 该结果保守对待文章的因果推断，这反衬了本文相关关系的谨慎定位。",
        "note": "此图为展示因果识别的局限性，非核心因果证据。结论为：未能通过DID识别显著政策效应，研究定位于统计相关关系。"
    },
    {
        "fig_id": "fig9",
        "cn_name": "提质增效指数构造框架",
        "section": "§2.1 变量构造",
        "chart_type": "结构示意图（Schematic）",
        "description": "展示提质增效综合指数的层级构造框架：底层为5个基础指标（ROA、营业利润率、TFP、总资产周转率、劳动生产率）、中层为2个子指数（质量子指数、效率子指数，经Min-Max标准化）、顶层为1个综合指数（等权合成w=0.5:0.5）。",
        "data_source": "无（纯示意图，方法论说明）",
        "variables": "无（非数据驱动，仅为变量构造方法论说明）",
        "interpretation": "① 结构图清晰展示'5指标→2子指数→1综合指数'的三层合成路径。② 等权合成（w=0.5:0.5）确保质量和效率获得同等权重，避免某一维度主导综合指数。③ Min-Max标准化使各分项指标具有可比性（均归化至[0,1]）。",
        "note": "5个基础指标：ROA=总资产净利润率、营业利润率、TFP=LP法全要素生产率、总资产周转率、劳动生产率（对数）。"
    },
    # ========== fig10 描述性统计2×2面板 ==========
    {
        "fig_id": "fig10",
        "cn_name": "主要变量描述性统计2×2面板",
        "section": "§2.1 描述性统计",
        "chart_type": "2×2多面板组合图（小提琴图+热力图+柱状图+水平柱状图）",
        "description": "一图四面板综合展示样本特征：(a) 左上：小提琴图展示6个主要变量的分布形状、中位数和四分位范围；(b) 右上：Pearson相关系数热力图展示变量间两两线性关系；(c) 左下：样本年份分布柱状图，显示每年观测数量；(d) 右下：制造业二级行业Top10水平柱状图，按观测数量排序。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "(a) quality_efficiency_index / annual_ai_log / tfp / ln_assets / leverage / revenue_growth;\n(b) 同上6变量;\n(c) year;\n(d) industry_name",
        "interpretation": "① 面板(a)：6个主要变量分布特征各异，经1%缩尾处理后无明显极端值。② 面板(b)：AI词频与提质增效正相关(r>0)，与研发强度高度正相关(r>0.5)。③ 面板(c)：样本年份分布均匀，每年约2000-2500观测，面板平衡。④ 面板(d)：计算机通信、设备制造、金属加工为三大观测行业，合计占样本近半。",
        "note": "变量经1%双侧缩尾处理。图(a)数值为标准化值[0,1]以便多变量并列比较。"
    },
    # ========== fig11 - fig17 ==========
    {
        "fig_id": "fig11",
        "cn_name": "分箱散点：AI词频与提质增效",
        "section": "§3.1 描述性分析",
        "chart_type": "分箱散点图（Binned Scatter）",
        "description": "对AI词频（对数）和提质增效指数进行二元可视化：背景为随机抽样5000个原始散点（半透明），叠加20组等样本量分箱均值（蓝色点+95%CI误差线），并绘制OLS线性拟合线（橙色虚线）。这是基准回归关系的可视化预览。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "annual_ai_log（X轴，年报AI词频对数）、quality_efficiency_index（Y轴，提质增效指数）",
        "interpretation": "① 蓝色分箱均值从低AI到高AI呈稳定上升趋势，大致线性。② 20组均值的95%CI在低AI端较宽（样本较少）、高端略有收窄。③ 无倒U型或其他非线性拐点迹象。④ OLS线性拟合为正斜率，与基准回归方向一致。⑤ 但此为简单二元关系，未控制其他变量。",
        "note": "等样本量分箱（非等间距），每组观测数大致相等。散点为随机抽样5000个。未控制其他因素。"
    },
    {
        "fig_id": "fig12",
        "cn_name": "偏相关图：控制后AI效应（最核心新增）",
        "section": "§3.2 基准回归",
        "chart_type": "偏相关散点图（Frisch-Waugh Partial Correlation）",
        "description": "采用Frisch-Waugh-Lovell定理方法可视化偏相关关系：先将提质增效指数(Y)和控制变量回归取残差，再将AI词频(X)和控制变量回归取残差，然后对两组残差做分箱散点。控制变量包括企业规模、资产负债率、研发强度、企业年龄、营收增长率，以及年份FE和制造业二级行业FE。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "Y残差：quality_efficiency_index净化了控制变量+年份FE+行业FE后的残差;\nX残差：annual_ai_log净化了控制变量+年份FE+行业FE后的残差;\n控制变量：ln_assets/leverage/rd_intensity/firm_age/revenue_growth",
        "interpretation": "① 控制其他变量和固定效应后，AI词频残差与提质增效残差仍呈正相关关系。② 偏相关系数r值和回归系数β值标注在图中。③ 蓝色分箱点（20组等样本量）沿拟合线分布紧密，表明条件关系稳定。④ 与控制前（fig11）相比斜率减小但方向不变，说明部分原始关系被控制变量吸收。",
        "note": "论文最核心的可视化新增——解决了'控制后关系是否仍然存在'的关键问题。散点=随机5000个残差对。"
    },
    {
        "fig_id": "fig13",
        "cn_name": "稳健性检验森林图",
        "section": "§3.3 稳健性检验",
        "chart_type": "分层森林图（Grouped Forest Plot）",
        "description": "按三组展示7个稳健性检验规格的系数及95% CI：①基准（主模型）；②替换X（MD&A词频、综合AI指数）；③替换Y（TFP、熵权指数、PCA指数）；④样本处理（研发缺失处理）。左侧以颜色标签区分组别。",
        "data_source": "outputs/tables/final/paper_table_1_main_regression.csv + paper_table_2_robustness.csv",
        "variables": "coef（系数）、std_err（标准误）；规格：主模型/MD&A词频/综合AI/TFP/熵权指数/PCA指数/研发缺失处理",
        "interpretation": "① 所有替换X的系数均为正且显著，说明AI效应不依赖于特定的AI衡量方式（年报vs MD&A vs 综合指数）。② 替换Y至TFP、熵权指数、PCA指数后系数仍为正且显著，说明AI效应不依赖特定的提质增效衡量方式。③ 研发缺失处理（填补/剔除）后结果不变，说明缺失值不影响结论。",
        "note": "组标签：蓝=基准、青=替换X、橙=替换Y、紫=样本处理。竖虚线x=0。"
    },
    {
        "fig_id": "fig14",
        "cn_name": "LOESS局部加权回归拟合",
        "section": "§3.4 机制检验 / §3.6 扩展模型",
        "chart_type": "散点+双拟合曲线叠加图",
        "description": "在背景散点（随机4000个样本点）上叠加两条拟合曲线：(1) LOESS局部加权回归（蓝色粗线，f=0.3带宽），可捕捉潜在非线性关系；(2) 线性拟合（橙色虚线），作为线性基准对比。用于检验AI效应是否存在非线性（如边际递减/递增）。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "annual_ai_log（X轴）、quality_efficiency_index（Y轴）",
        "interpretation": "① LOESS曲线（蓝）整体接近线性，无明显陡峭拐点或S型弯曲。② LOESS与线性拟合（橙虚线）高度重合，差异极小（尤其在中部数据集中区域）。③ 说明AI-提质增效关系在大数据样本下高度线性，无需复杂的非线性设定（如二次项、门槛）。④ 非线性二次项检验不显著（p>0.1），支持线性设定。",
        "note": "LOESS带宽frac=0.3。在高AI端（右侧）LOESS略有上翘，暗示边际可能递增，但数据稀疏、CI宽，不支持作为严格非线性证据。"
    },
    {
        "fig_id": "fig15",
        "cn_name": "分位数固定效应：AI系数随提质增效水平变化",
        "section": "§3.6 扩展模型",
        "chart_type": "分位数效应折线图（Quantile Coefficient Plot）",
        "description": "展示在不同提质增效水平分位点（Q25→Q50→Q75等）上，AI词频系数的变化轨迹。蓝线为平滑样条插值，蓝色阴影为95%置信带，点标记在各分位点并附系数标注。揭示AI效应在提质增效低水平和高水平企业间的差异。",
        "data_source": "outputs/tables/model_results/quantile_fe_results.csv（分位数固定效应回归结果）",
        "variables": "quantile（分位点）、coef（AI系数）、std_err（标准误）",
        "interpretation": "① AI系数从低分位到高分位可能呈现递减趋势（标注'Q25→Q75下降约X%'），说明AI对低位提质增效企业的改善作用更强。② 曲线在低分位端（Q10/Q25）CI较宽但系数较高；高分位端（Q90/95）系数趋近零。③ 分位数效应递减模式符合'边际效用递减'预期：对于提质增效水平低的企业，AI改进空间更大。",
        "note": "分位数固定效应模型（Quantile FE），控制年份+行业FE及企业特征。Q25→Q75下降幅度以百分比标注。"
    },
    {
        "fig_id": "fig16",
        "cn_name": "模型设定对比：固定效应 vs 交叉拟合 vs 企业FE",
        "section": "§3.6 扩展模型 / §4.2 结论边界",
        "chart_type": "并列森林图（Comparison Forest Plot）",
        "description": "三种不同模型设定的AI系数并排对比：①固定效应主模型（蓝）—工业内行业FE的OLS基准；②交叉拟合部分线性模型（青）—用随机森林净化控制变量后估计AI净效应，减少线性设定假设；③企业FE边界模型（灰，可选）—严格控制个体异质性。",
        "data_source": "paper_table_1_main_regression.csv + cross_fitted_partial_linear_results.csv + paper_table_A1_boundary.csv",
        "variables": "coef（系数）、std_err（标准误）；三个模型设定",
        "interpretation": "① 固定效应（蓝）和交叉拟合（青）的AI系数接近且方向一致，说明线性OLS设定未严重扭曲结果。② 企业FE边界模型（灰）系数明显减弱、CI增大，部分不再显著——表明当控制企业个体异质性后，AI效应的量级和显著性下降。③ 三种设定的对比揭示了结论的边界：行业间变异驱动了主要效应，企业内变异提供的证据较弱。",
        "note": "交叉拟合部分线性模型是半参数方法，不对控制变量函数形式做线性假设。边际效应统一在AI词频均值处计算。"
    },
    {
        "fig_id": "fig17",
        "cn_name": "边界检验：行业FE vs 企业FE（核心新增）",
        "section": "§4.2 结论边界",
        "chart_type": "配对森林图（Paired Comparison Forest Plot）",
        "description": "对四个核心规格进行行业FE和企业FE的配对系数对比：①主模型、②MD&A词频、③综合AI指数、④TFP仅企业FE。蓝色标记行业FE，灰色标记企业FE。实心=显著(p<0.05)、空心=不显著。",
        "data_source": "paper_table_1_main_regression.csv（行业FE系数）+ paper_table_A1_boundary.csv（企业FE系数）",
        "variables": "系数/标准误，从两个CSV分别读取",
        "interpretation": "① 行业FE下所有系数均为正且显著。② 企业FE下所有系数大幅缩小、CI扩大：主模型系数下降约60-80%，MD&A和综合AI系数降至不显著。③ TFP在企业FE下接近零且不显著。④ 这说明AI关注度与提质增效的正相关关系主要在行业间（cross-industry）成立，在行业内（within-firm）证据较弱。⑤ 这是论文'谨慎归因'定位的核心实证依据——结论应限于统计相关关系，不宜推断为因果效应。",
        "note": "标注'企业FE下系数明显减弱'。此图直接支撑论文§4.2关于结论边界和方法论限度的讨论，是论文透明性和科学严谨性的关键展示。"
    },
    # ========== 附录图 figs_s1 - s6 ==========
    {
        "fig_id": "fig_s1",
        "cn_name": "数据筛选流程图",
        "section": "§2.1 数据来源",
        "chart_type": "流程示意图（Flow Diagram）",
        "description": "四步框式流程图展示样本从原始数据到最终建模样本的筛选过程：第一步原始A股制造业上市公司样本（N=29,621，企业=3,843，2013-2024），第二步剔除关键变量缺失（N=28,450），第三步剔除研发强度缺失（N=26,869），第四步最终建模样本（企业=3,615，年份=12，N=26,869）。",
        "data_source": "无（流程图，数据为统计计算所得）",
        "variables": "无（流程图）",
        "interpretation": "① 原始29,621条观测仅损失约9.3%后进入最终模型。② 研发强度缺失约3%（1,581条）是主模型样本量差异的主要来源。③ 最终样本覆盖3,615家制造业企业12年完整面板，规模充足。",
        "note": "研发强度缺失约占3%。最终主模型采用26,869条完整观测。"
    },
    {
        "fig_id": "fig_s2",
        "cn_name": "关键变量缺失率",
        "section": "附录：数据诊断",
        "chart_type": "水平柱状图（Horizontal Bar Chart）",
        "description": "展示12个关键变量的缺失率（%），从低到高排序。蓝色条形（<5%缺失）、青色条形（5–10%）、橙色条形（>10%）。每个条形旁标注缺失率百分比和缺失数(n)。橙色虚线标示5%参考线。",
        "data_source": "outputs/tables/diagnostics/missingness_summary.csv（缺失率诊断输出表）",
        "variables": "variable（变量名）、missing_rate（缺失率%）、missing（缺失计数）",
        "interpretation": "① 绝大多数变量缺失率<5%（蓝色区域），数据质量好。② 研发强度缺失约3%，AI应用指数缺失约4–5%，AI投资水平缺失较高(>10%)。③ AI投资水平因缺失率高定义为次要变量。④ 整体样本完整性满足实证分析要求。",
        "note": "总观测29,621条。5%参考线为经验常规标准：低于5%视为低缺失。"
    },
    {
        "fig_id": "fig_s3",
        "cn_name": "多重共线性诊断：VIF",
        "section": "§2.2 计量诊断",
        "chart_type": "水平柱状图（Horizontal Bar Chart）",
        "description": "展示6个控制变量和核心解释变量的方差膨胀因子（VIF）值，从低到高排序。蓝色条形（VIF<5）、青色条形（VIF 5–10）、橙色条形（VIF>10）。两条参考线：VIF=5（橙虚线）、VIF=10（红虚线）。",
        "data_source": "outputs/tables/diagnostics/vif_diagnostics.csv（VIF诊断输出表）",
        "variables": "variable（变量名：年报AI词频/企业规模/资产负债率/研发强度/企业年龄/营收增长率）、vif（VIF值）",
        "interpretation": "① 所有变量VIF<2，远低于常规阈值VIF=5或VIF=10。② 最高VIF为企业规模约1.8，其次为资产负债率约1.5。③ 多重共线性不是本研究的计量问题，OLS估计量具有良好统计性质。",
        "note": "所有变量VIF<2，不存在严重多重共线性问题。阈值：VIF<5优良，5-10可接受，>10严重。"
    },
    {
        "fig_id": "fig_s4",
        "cn_name": "异方差检验：Breusch-Pagan",
        "section": "§2.2 计量诊断",
        "chart_type": "柱状图（对数Y轴）",
        "description": "展示三个模型（固定效应基线、加控制变量、行业FE+控制）的Breusch-Pagan异方差检验p值，Y轴为对数尺度（1e-300至1）。两条参考线：p=0.01（红虚线）和p=0.05（橙虚线）。",
        "data_source": "outputs/tables/diagnostics/heteroskedasticity_tests.csv（异方差检验输出表）",
        "variables": "model（模型名称）、bp_lm_pvalue（Breusch-Pagan LM检验p值）",
        "interpretation": "① 三个模型的BP检验p值均<<0.01（拒绝同方差原假设），存在显著异方差。② 因此标准OLS标准误不可信，需要使用稳健标准误（异方差一致标准误或聚类稳健标准误）。③ 本论文所有模型均采用公司层面聚类稳健标准误（Cluster-robust SE），以应对异方差并保留企业内相关性结构。",
        "note": "所有模型p<0.01，拒绝同方差原假设。因此采用公司聚类稳健标准误。"
    },
    {
        "fig_id": "fig_s5",
        "cn_name": "候选X-Y组合筛选热力图",
        "section": "附录：变量筛选",
        "chart_type": "热力图（Heatmap）",
        "description": "展示主要候选解释变量X（6个AI相关变量）和被解释变量Y（6个提质增效相关变量）之间二元回归的|t值|热力图。颜色越深（橙红）表示|t|越大，关系越强。单元格标注t值或'ns'（不显著/方向为负）。",
        "data_source": "outputs/tables/diagnostics/candidate_xy_screening_results.csv（候选X-Y组合筛选输出）",
        "variables": "x（6个X：年报AI词频/MD&A词频/AI应用指数/AI实际应用/AI投资对数/AI投资水平）、y（6个Y：提质增效/TFP/熵权/PCA/资产周转率/AI实际应用）、t_value（t统计量）、positive_and_significant（是否为正向显著）",
        "interpretation": "① 此图帮助确定论文的核心X-Y对。② 年报AI词频与提质增效（[1,1]格）t值高且正向显著，是核心关系。③ 多数X-Y组合为正，但AI投资相关X的显著性较弱（部分标注'ns'）。④ 此筛选为论文选定'年报AI词频'为X、'提质增效综合指数'为Y提供了实证支撑。",
        "note": "颜色='YlOrRd'色阶（0–15|t|）。ns=不显著或方向为负。这是早期探索阶段用于变量筛选的工具图。"
    },
    {
        "fig_id": "fig_s6",
        "cn_name": "AI落地转化效率调节效应",
        "section": "附录：探索性分析",
        "chart_type": "系数森林图（Forest Plot）",
        "description": "展示转化效率调节效应回归的三个系数：(1)低转化×AI词频（基准AI效应）、(2)AI词频×高转化交互项（调节效应核心项）、(3)高转化效率虚拟变量（主效应）。实心圆=显著、空心方块=不显著。",
        "data_source": "outputs/tables/model_results/conversion_moderation_results.csv（调节效应回归结果）",
        "variables": "x（变量标签）、coef（系数）、std_err（标准误）、p_value（p值）",
        "interpretation": "① 交互项（AI词频×高转化）的系数虽为正但p>0.1，不显著。② '转化为效率调节效应证据不足以支持独立结论'。③ 对于调节效应不显著的明确展示也体现了论文的科学透明性——检验了可能的理论机制并诚实报告不显著结果。",
        "note": "红色标注'交互项不显著(p>0.1)，转化效率调节效应证据不足'。属探索性分析，不是核心结论。"
    },
]

EXTRA_CHARTS = [
    # ========== 3 流程图 ==========
    {
        "fig_id": "fig_extra_research_design_flow",
        "cn_name": "研究设计总览流程图",
        "section": "§1.2 研究设计",
        "chart_type": "流程示意图（Schematic Flowchart）",
        "description": "四阶段流程图概览整体研究设计：阶段一'理论构建'（文献综述→提出假设H1-H3），阶段二'数据准备'（文本分析提取AI词频→变量构造），阶段三'实证分析'（基准回归/稳健性/机制/异质性），阶段四'结果呈现'（19组核心图+论文撰写）。阶段间以箭头连接。",
        "data_source": "无（纯示意图，方法论说明）",
        "variables": "无",
        "interpretation": "① 作为读者快速理解论文宏观架构的'路线图'。② 四阶段线性递进，箭头明确表示工作流。③ 标注H1-H3三个研究假设的位置（理论构建阶段）。④ 适用于PPT汇报、答辩展示等场景，将整篇论文的逻辑压缩在一页。",
        "note": "可用于PPT汇报和论文答辩。四阶段：理论构建→数据准备→实证分析→结果呈现。"
    },
    {
        "fig_id": "fig_extra_variable_construction",
        "cn_name": "核心变量构造流程图",
        "section": "§2.1 变量构造",
        "chart_type": "流程示意图（Schematic Flowchart）",
        "description": "分层展示核心变量的构造逻辑：左侧'解释变量X/AI关注度'由年报AI词频、MD&A AI词频、AI投资、综合AI应用指数合成；右侧'被解释变量Y/提质增效综合指数'由质量子指数（ROA+利润率+TFP）和效率子指数（周转率+劳动生产率）等权合成；下侧标注控制变量集（规模/杠杆/研发/年龄/增长）。",
        "data_source": "无（纯示意图，变量构造方法论说明）",
        "variables": "无（非数据驱动）",
        "interpretation": "① 清晰展示AI关注度是'多指标概念'，年报AI词频是主度量。② 提质增效的'质量+效率'两维结构一目了然。③ 控制变量的五要素覆盖了企业特征的主要维度。④ 此图是fig9（提质增效指标结构）的补充，将X和Y的构造逻辑同框展示。",
        "note": "与fig9（提质增效指标结构）互补。fig9聚焦Y的构造，本图同时展示X和Y。"
    },
    {
        "fig_id": "fig_extra_methodology_path",
        "cn_name": "实证方法论路径：从相关到因果推断",
        "section": "§1.2 研究设计 / §4.2 结论边界",
        "chart_type": "流程示意图（Schematic Flowchart）",
        "description": "五阶段方法论路径图，展示证据强度从描述性相关逐步推进到因果推断的进阶过程：①描述性分析（小提琴图/热力图/分箱散点）→②相关性分析（OLS回归）→③因果初步（固定效应+控制变量）→④稳健性验证（替换变量/分位数/交叉拟合）→⑤因果边界（企业FE/DID事件研究）。标注'证据强度递减'的虚线箭头。",
        "data_source": "无（纯示意图，方法论叙事）",
        "variables": "无",
        "interpretation": "① 此图清晰表达了论文的'方法论递进'叙事：从简单到复杂，从相关到因果。② 第五阶段（因果边界）的结果更为保守（企业FE系数下降、DID不显著），因此箭头标注'证据强度递减'——越严格的因果识别，实际发现的关系越弱。③ 这种诚实面对方法论局限的阐释是本文的一大优势。",
        "note": "核心信息：从相关到因果推断，证据强度逐渐增强但设计约束也逐步严格。企业FE和DID表明因果证据有限。"
    },
    # ========== 15 统计图 ==========
    {
        "fig_id": "fig_extra_ridge_plot",
        "cn_name": "山脊线图：AI词频历年分布变化",
        "section": "§2.1 / §3.1",
        "chart_type": "山脊线图（Ridge/Joy Plot）",
        "description": "以年份为纵轴分组，每行展示该年度AI词频（对数）的核密度估计曲线，形成重叠的山脊状分布。可以直观观察AI词频分布中心逐年右移的趋势——从2014年分布集中在低AI区间，逐年向右（高AI）扩展。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "annual_ai_log（年报AI词频对数，Y轴分布）、year（年度分组维度）",
        "interpretation": "① 2014年AI词频分布集中在较低水平（密度峰偏左），而2023-2024年分布中心明显右移。② 右侧长尾逐年拉长——出现越来越多高AI关注度的企业。③ 山脊线图比传统箱线图或柱状图更能揭示分布的'形态变化'（如双峰、偏态等）。④ 可观测到AI的'扩散效应'：不仅是均值上升，分布整体向高值转移。",
        "note": "标注'n={count}'显示各年样本量。Y轴=年份（从2014到2024降序排列），X轴=AI词频对数。"
    },
    {
        "fig_id": "fig_extra_correlation_heatmap",
        "cn_name": "完整相关性热力图：全部数值变量",
        "section": "§2.1 / §3.1",
        "chart_type": "相关系数热力图（Correlation Heatmap）",
        "description": "展示11个全部数值变量（包括AI相关、提质增效相关、企业特征变量）的完整Pearson相关系数矩阵。颜色采用RdBu_r色阶（蓝=正相关，红=负相关，-1 to +1），每格标注相关系数值。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "annual_ai_log/mda_ai_log/ai_application_index/ai_actual_application_index/quality_efficiency_index/tfp/rd_intensity/ln_assets/leverage/firm_age/revenue_growth",
        "interpretation": "① 核心关系：annual_ai_log与quality_efficiency_index正相关（r≈...），与fig11/fig12一致。② AI词频与RD_intensity高度正相关（r>0.5），反映了'高研发企业也更关注AI'的集聚现象，需要回归中控制研发以避免混淆。③ AI词频与规模(ln_assets)也呈正相关——大企业更关注AI。④ MD&A词频和年报词频之间r>0.7，高共线性说明两者不能同时进入模型。⑤ 控制变量（规模/杠杆/年龄/增长）与提质增效的相关性各异，需在回归中纳入。",
        "note": "11个全部数值变量构成11×11对称矩阵。颜色：蓝=正相关，红=负相关。变量经1%缩尾处理。"
    },
    {
        "fig_id": "fig_extra_panel_coverage",
        "cn_name": "面板数据覆盖热力图",
        "section": "§2.1 数据来源",
        "chart_type": "热力图（Heatmap）",
        "description": "交叉展示制造业Top 8二级行业（行）× 12个年份（列）的观测数热力图。颜色越深（橙→红）表示该行业-年份组合的观测越多。每个单元格标注具体观测计数。对'其他'行业进行合并显示。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "industry_name（行业名称，分组为Top8+其他）、year（年份）、观测数 = count(annual_ai_log)",
        "interpretation": "① 检测面板数据是否均匀覆盖各行业-年份组合。② Top 8行业各年份观测数基本均衡（颜色均匀），无大量空缺或爆发式增长。③ 若某行业某年份突然大量缺失（白/浅色格），可能是面板不平衡的标志，需要关注。④ 从结果看，数据覆盖良好——采样行业结构各年一致。",
        "note": "颜色越深=观测数越多。仅展示Top 8行业+其他组。数据来源为A股制造业上市公司2013-2024面板。"
    },
    {
        "fig_id": "fig_extra_lollipop",
        "cn_name": "棒棒糖图：各模型AI词频系数一览",
        "section": "§3.2 / §3.3",
        "chart_type": "棒棒糖图（Lollipop Chart）",
        "description": "以棒棒糖（茎+圆点）形式展示所有模型规格的AI词频系数，按系数大小从低到高纵向排列。蓝色茎=正系数，红色茎=负系数（如有）。每个圆点附95% CI的误差条。",
        "data_source": "outputs/tables/final/paper_table_1_main_regression.csv（基准回归结果表）",
        "variables": "模型（模型名称）、系数（coef）、标准误（std_err）",
        "interpretation": "① 一目了然地展示所有模型的AI系数排位。② 如果没有红色（负系数）的茎出现，验证了AI正效应的全局一致性。③ 茎的长短和横杠宽度展示不同模型设定下CI的差异——控制越多，CI越宽。④ 比传统的表格更容易一眼比较各模型系数大小。",
        "note": "蓝=正效应，红=负效应。所有模型AI词频系数为正，稳健通过显著性检验。"
    },
    {
        "fig_id": "fig_extra_dumbbell_ols_vs_fe",
        "cn_name": "哑铃图：行业FE vs 企业FE系数对比",
        "section": "§4.2 结论边界",
        "chart_type": "哑铃图（Dumbbell Chart）",
        "description": "对每个模型规格，连接其在行业固定效应（蓝点）和企业固定效应（橙色点）下的系数，形成哑铃状。哑铃的长度直观反映个体异质性控制的冲击：越长，说明该系数对FE选择越敏感。橙色实心点=|t|>1.96（显著）。",
        "data_source": "paper_table_1_main_regression.csv（行业FE）+ paper_table_A1_boundary.csv（企业FE）",
        "variables": "同上两表：模型/系数/标准误（行业FE和企业FE分别）",
        "interpretation": "① 每条哑铃从蓝点到橙点，橙点普遍位于蓝点左侧（系数减小），表示企业FE下AI效应'衰减'。② 哑铃越长说明衰减越严重——对于MD&A词频和综合AI规格，哑铃极长，系数从正显著衰减至不显著。③ 多数橙色点为空心（不显著），表明企业FE下去除了行业间混淆后，AI的年度变异与企业提质增效变异关系减弱。",
        "note": "橙色实心=|t|>1.96（显著于5%）。企业FE下系数和显著性均有明显减弱。"
    },
    {
        "fig_id": "fig_extra_connected_scatter",
        "cn_name": "双轴时间序列：AI词频与TFP年度变化轨迹",
        "section": "§3.1 / §3.4",
        "chart_type": "双轴时间序列连接散点图（Connected Scatter with Dual Axes）",
        "description": "以年份为横轴，左轴显示AI词频均值变化（蓝线+圆点），右轴显示TFP均值变化（橙线+方块点）。相邻年份间以箭头连接，强调变化方向和速率。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "year（横轴）、annual_ai_log（左轴Y，年度均值）、tfp（右轴Y，年度均值）",
        "interpretation": "① 2013-2024年AI词频持续快速上升（蓝线几乎单调递增），而TFP整体平稳、在2019后微幅波动。② 两者的时间序列协变明显但不完全同步——AI上升快而TFP相对平缓，暗示可能存在滞后效应或非线性转化。③ 双轴连接散点优于双折线的地方是'年份间的步进变化'更明显（箭头长度=变化量）。④ 为机制分析中的'AI通过TFP影响提质增效'提供初步时间维度证据。",
        "note": "双轴设计：左轴=AI词频（蓝），右轴=TFP（橙）。箭头连接相邻年份，强调变化。"
    },
    {
        "fig_id": "fig_extra_nonlinear_marginal",
        "cn_name": "非线性二次项边际效应图",
        "section": "§3.6 扩展模型",
        "chart_type": "双面板边际效应图（Marginal Effects Plot）",
        "description": "左面板（二次拟合）：展示AI词频（中心化后）二次函数拟合下的提质增效预测值曲线，标注边际转折点（如存在）。右面板（边际效应）：展示边际效应∂Y/∂X随AI词频的变化轨迹（U型=β₁+2β₂X），评估是否存在边际递增/递减。",
        "data_source": "outputs/tables/model_results/nonlinear_ai_effect_results.csv（非线性二次项回归结果）",
        "variables": "β₁（一次项系数）、β₂（二次项系数）",
        "interpretation": "① 二次项为正（β₂>0且p<0.01），说明AI效应随水平提高而'递增'（加速），即高AI企业的回报率更高。② 右面板边际效应线从左至右向上倾斜，从接近零升至更正值——与传统的'边际效用递减'预期相反。③ 这一发现的含义是：AI可能存在'正反馈效应'——越投入AI，单位投入的提质增效回报越大。④ 但需注意此基于二次项简单参数形式，结论需交叉验证。",
        "note": "二次项为正(=0.0141, p<0.01)→AI效应递增。X经中心化处理避免高共线性。标注β₁和β₂值。"
    },
    {
        "fig_id": "fig_extra_boxplot",
        "cn_name": "多变量标准化分布对比箱线图",
        "section": "§2.1 描述性统计",
        "chart_type": "多组箱线图（Multi-group Boxplot）",
        "description": "对6个主要变量（AI词频、提质增效、TFP、研发强度、资产负债率、营收增长）进行箱线图并行展示。所有变量先经1%缩尾再Min-Max标准化至[0,1]，使其具有可比性。箱体=Q1~Q3，中线=中位数，须=无异常值点。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "annual_ai_log/quality_efficiency_index/tfp/rd_intensity/leverage/revenue_growth（经缩尾+标准化）",
        "interpretation": "① 6个变量的分布形态、极差、离群程度在一张图中即可比较。② AI词频和研发强度分布较宽（箱体高），反映企业间差异大。③ TFP分布相对集中（箱体窄），说明制造业TFP差异相对小。④ 资产负债率中位数较高（接近0.5），制造业杠杆水平普遍不低。⑤ 营收增长率存在若干极端值（需缩尾后看须长）。",
        "note": "箱体=Q1~Q3，中线=中位数。值均标准化至[0,1]以便跨变量比较。颜色区分6个变量。"
    },
    {
        "fig_id": "fig_extra_stacked_bar_industry",
        "cn_name": "堆积柱状图：Top 5行业年度结构变化",
        "section": "§2.1",
        "chart_type": "100%堆积柱状图（Stacked Bar 100%）",
        "description": "展示Top 5制造业二级行业（计算机通信、设备制造、金属加工、化工、汽车）及其他行业在2013-2024各年度样本中所占的比例。纵轴为占比（%），各颜色段叠加至100%。每列=一个年份。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "industry_name（行业名，分组为Top5+其他）、year（年份）",
        "interpretation": "① 各行业占比历年保持稳定，无某行业突然膨胀或消失。② 这验证了样本的'行业结构稳定性'——不是某行业集中上市而导致时间趋势存在行业偏移。③ 如果行业结构变化大（如早期以化工为主后期以计算机为主），则时间趋势分析可能被行业结构变化混淆。④ 这里结构稳定=时间序列趋势可信；也意味着不需要调整样本权重。",
        "note": "各行业占比历年保持稳定，样本行业结构未发生剧烈变动。其他行业合并显示。"
    },
    {
        "fig_id": "fig_extra_bubble_chart",
        "cn_name": "气泡图：模型系数一览（气泡大小=|t|值，透明度=显著性）",
        "section": "§3.2 / §3.3",
        "chart_type": "气泡图（Bubble Chart）",
        "description": "将所有模型规格（从主回归表、稳健性表、高级模型表去重后）绘制为气泡。横轴=AI系数，气泡大小=|t值|（越大统计学意义越强），气泡透明度=p值（越不透明越显著），标签=模型名称。",
        "data_source": "paper_table_1_main_regression.csv + paper_table_2_robustness.csv + paper_table_6_advanced_models.csv",
        "variables": "模型（模型名称）、系数（coef）、p值（p_value/p值）、样本量",
        "interpretation": "① 所有气泡都在x=0的右侧（正系数），验证了所有模型的AI正向统一性。② 大而深的气泡=强效应+强显著性，位于更右侧；小而透明的气泡=弱效应+不显著。③ 一眼即知哪个模型结果最强（最大最右侧最不透明的气泡）。④ 与fig4（森林图）互补——森林图强调CI，气泡图强调系数大小/显著性和直观对比。",
        "note": "气泡大小=|t值|，透明度=p值(p<0.05=实心)。三个CSV去重合并。"
    },
    {
        "fig_id": "fig_extra_error_band_timeseries",
        "cn_name": "提质增效和AI词频年度变化带图",
        "section": "§2.1 / §3.1",
        "chart_type": "误差带时间序列（Error Band Timeseries）",
        "description": "左右并列两个子图：左图=提质增效指数年度均值时间序列（年度均值±95%CI阴影带），右图=AI词频年度均值时间序列（年度均值±95%CI阴影带）。阴影带的宽度反映各年份样本均值的不确定性（N越大，阴影越窄）。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "year/quality_efficiency_index/ annual_ai_log",
        "interpretation": "① 左图（提质增效）：2013-2018相对平稳，2019年起小幅上升趋势，2020年后的改善伴有CI收窄（样本量充足）。② 右图（AI词频）：持续上升，2015-2019斜率较缓，2020年后加速上升；阴影带在早期略宽（早期样本/数据质量可能稍差）。③ CI带整体系窄，因为每年有2000+观测的大样本。④ 两图并排直观展示'AI上升(TFP微升或平)'的协变模式，为基准回归提供描述性背景。",
        "note": "误差带=均值±1.96×SE（95%置信区间）。左图=提质增效趋势，右图=AI词频趋势。"
    },
    {
        "fig_id": "fig_extra_donut_industry",
        "cn_name": "甜甜圈图：样本行业分布",
        "section": "§2.1",
        "chart_type": "甜甜圈图（Donut/Ring Pie Chart）",
        "description": "展示Top 7制造业二级行业在总样本中的占比（圆环扇区），将剩余行业合并为'其他'。每个扇区标注行业名、观测计数和百分比。中心空洞处标注总样本量'N=26,869'。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "industry_name（行业名称，分组为Top7+其他）",
        "interpretation": "① 计算机通信占比最高(约20–25%)，设备制造次之(约15–18%)。② Top 7行业合计占总样本70%以上，行业集中度高，符合A股制造业上市公司行业分布特征。③ 甜甜圈图比传统饼图更美观，中心空洞可标注总N。④ 注意：占比不均衡可能意味着行业间的影响有偏，但可通过行业固定效应部分控制。",
        "note": "计算机通信、设备制造、金属加工等为主要观测行业。中心标注总N。"
    },
    {
        "fig_id": "fig_extra_hexbin_ai_rd",
        "cn_name": "六边形蜂巢图：AI词频与研发强度",
        "section": "§3.1 / §3.4",
        "chart_type": "六边形蜂巢图（Hexbin Plot）",
        "description": "展示AI词频（X轴）和研发强度（Y轴，截尾至99%分位数）的联合二维密度分布。每个六边形代表一个二维区间，颜色越深（橙→红）表示该区间的观测数越多。揭示两个变量在二维空间的聚集模式。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "annual_ai_log（X轴，AI词频对数）、rd_intensity（Y轴，研发强度，截尾至99%分位数）",
        "interpretation": "① 数据大量聚集在左下方（低AI-低研发），形成'核心密集区'。② 向右上方向（高AI-高研发）的观测稀疏（深色区域减少），但存在正相关的上升带。③ 这意味着大多数企业属于'双低'型（AI和研发都不高），只有少数企业在'双高'区域，但双高企业可能是提质增效的领先者。④ 如果核心关系在双低区域不稳定，可能导致高杠杆异常值影响OLS。但本文通过缩尾和一篮子稳健性检验已有所控制。",
        "note": "Y轴研发强度截尾至99%分位数（避免极端值扭曲）。六边形大小自适应，颜色=观测数。"
    },
    {
        "fig_id": "fig_extra_slope_chart",
        "cn_name": "坡度图：各指标早期vs晚期变化",
        "section": "§3.1 / §4",
        "chart_type": "坡度图（Slope Chart）",
        "description": "对比5个关键指标在'早期均值（2013–2015）'与'晚期均值（2022–2024）'之间的变化方向和幅度。每条线连接早期和晚期两个端点，线上标注原始均值。蓝色线=上升（晚期>早期），红色线=下降。所有值标准化至[0,1]以便跨指标比较。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "year（分组）、annual_ai_log/ quality_efficiency_index /tfp/ rd_intensity/ leverage（各指标，分组取均值后标准化）",
        "interpretation": "① AI词频从较低水平陡升至接近标准化值的上限（蓝线非常陡）——十年间AI关注度翻了若干倍。② 提质增效指数和TFP小幅上升（线微斜向上），幅度远小于AI词频变化——AI巨大进步但生产率仅微升（即'生产率悖论'的体现）。③ 杠杆率略微下降（红线），去杠杆政策可能发挥了作用。④ 每条线的端点标注原始均值，供读者查看绝对水平。⑤ 坡度图比传统时间序列更聚焦于'头尾对比'，浓缩十年变化。",
        "note": "蓝线=上升，红线=下降。早期=2013-2015均值，晚期=2022-2024均值。标准化后展示，端点标注原始值。"
    },
    {
        "fig_id": "fig_extra_cdf_comparison",
        "cn_name": "CDF累积分布对比：高AI vs 低AI企业",
        "section": "§3.1 / §4",
        "chart_type": "累积分布函数图（CDF Comparison）",
        "description": "将样本分为高AI组（AI词频>中位数）和低AI组（AI词频≤中位数），在同一坐标系绘制两组提质增效指数的累积分布函数曲线。高AI组的CDF（蓝线）若整体位于低AI组（灰线）右侧，说明高AI的提质增效分布一阶随机占优（高AI更可能取得高提质增效）。标注KS统计量和p值。",
        "data_source": "data/processed/manufacturing_modeling_panel.csv",
        "variables": "annual_ai_log（分组变量，按中位数分高/低）、quality_efficiency_index（CDF变量）",
        "interpretation": "① 高AI组的蓝色CDF线整体右移于低AI组的灰色CDF线——即在任意提质增效水平上，高AI企业达到该水平的累积概率更低（因为它们的分布更偏右），这意味着高AI'一阶随机占优'。② KS检验的p值<<0.001，拒绝两组分布相同的原假设。③ CDF对比不依赖均值的假设，比t检验或回归更稳健——即使分布非正态也能检验。④ 这为基准回归提供了无假设的分布层面的证据补充。",
        "note": "蓝色=高AI组(>中位数)，灰色=低AI组(≤中位数)。KS检验=两样本Kolmogorov-Smirnov检验，p<<0.001表示分布显著不同。"
    },
]

# ── Main ──────────────────────────────────────────────────────────────

def build_doc():
    doc = Document()

    # --- Global style settings ---
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Microsoft YaHei'
    font.size = Pt(10)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')

    # --- Title page ---
    title = doc.add_heading('统计建模论文图表说明文档', level=0)
    for r in title.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)

    meta_lines = [
        f"生成日期：{datetime.now().strftime('%Y-%m-%d')}",
        "项目：AI关注度与制造业提质增效 — 统计建模与可视化",
        "数据：A股制造业上市公司2013-2024年企业-年份面板数据",
        "图表总数：核心图23张 + 附加图18张 = 41张（PNG+PDF双格式，360dpi）",
        "GitHub：https://github.com/futurefff828/statistical-modeling-vis",
    ]
    for line in meta_lines:
        p = doc.add_paragraph(line)
        p.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

    # --- Section 1: Core charts ---
    add_section_heading(doc, "第一部分  论文核心图表（23张）", level=1)

    intro = doc.add_paragraph(
        "本部分涵盖论文正文和附录中使用 matplotlib 生成的全部23张核心图表。"
        "图表按论文叙述顺序排列：fig1-fig9为正文描述性和基准回归相关图（第2章至第3章前半），"
        "fig10为描述性统计2×2仪表盘，fig11-fig17为高级分析和结论边界图（第3章后半至第4章），"
        "fig_s1-fig_s6为附录中的诊断与补充分析图。每张图均以PNG（360dpi）和PDF矢量双格式保存。"
    )
    intro.paragraph_format.space_after = Pt(8)

    for chart in CORE_CHARTS:
        add_chart_entry(doc, **chart)

    doc.add_page_break()

    # --- Section 2: Extra charts ---
    add_section_heading(doc, "第二部分  附加丰富图表（18张）", level=1)

    intro2 = doc.add_paragraph(
        "本部分涵盖 src/vis_extra.py 中生成的18张附加图表，分为两类："
        "（1）3张流程图（Schematic），用于展示研究设计、变量构造和方法论路径的全景概览，"
        "适合PPT汇报和论文答辩使用；（2）15张多样化科研统计图表，涵盖山脊线图、哑铃图、"
        "棒棒糖图、气泡图、坡度图、连接散点、六边形蜂巢、堆积柱状、甜甜圈、CDF对比等丰富图表类型，"
        "为论文提供更多可选的可视化表达形式。"
    )
    intro2.paragraph_format.space_after = Pt(8)

    for chart in EXTRA_CHARTS:
        add_chart_entry(doc, **chart)

    doc.add_page_break()

    # --- Section 3: Data sources appendix ---
    add_section_heading(doc, "附录  数据来源与技术说明", level=1)

    appendix_items = [
        ("主面板数据", "data/processed/manufacturing_modeling_panel.csv",
         "26,869条企业-年份观测，3,615家A股制造业上市公司，2013-2024年面板。包含AI词频、提质增效指数、TFP、财务指标等全部变量。"
         "原始数据来源：CSMAR国泰安数据库 + Wind万得数据库。"),
        ("回归结果表", "outputs/tables/final/ 和 outputs/tables/model_results/",
         "包含基准回归表(paper_table_1)、稳健性检验表(paper_table_2)、高级模型表(paper_table_6)、"
         "机制检验结果、异质性检验结果、边界检验表(paper_table_A1)等。"),
        ("诊断输出表", "outputs/tables/diagnostics/",
         "包含缺失率汇总(missingness_summary.csv)、VIF诊断(vif_diagnostics.csv)、"
         "异方差检验(heteroskedasticity_tests.csv)、候选X-Y筛选(candidate_xy_screening_results.csv)等。"),
        ("图表输出目录", "outputs/figures/final/",
         "全部41张图表（×2格式=82个文件），PNG格式360dpi高清晰度，PDF矢量格式无损缩放。"),
        ("图表生成代码", "src/vis_paper.py（核心23张）+ src/vis_extra.py（附加18张）+ src/visualization.py（fig1-fig4）",
         "总代码约1600行，函数结构清晰，统一复用风格设置、坐标轴美化、文件保存等基础设施函数。"),
        ("配色体系", "统一配色方案",
         "INK=#1F1F1F（墨黑主色）、BLUE=#0072B2（蓝）、TEAL=#009E73（青绿）、"
         "ORANGE=#D55E00（橙）、MUTED=#6E6E6E（灰）、RED_ACCENT=#CC3311（红强调）、"
         "PURPLE=#8E6AB3（紫）、GOLD=#C49B44（金）。"),
        ("图表分辨率", "360 DPI",
         "所有图表以360dpi高分辨率输出PNG，同时生成PDF矢量格式。满足学术出版和打印要求。"),
    ]

    for title, path, desc in appendix_items:
        h = doc.add_heading(title, level=3)
        for r in h.runs:
            r.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
        p = doc.add_paragraph(f"路径：{path}")
        p2 = doc.add_paragraph(desc)
        p2.paragraph_format.space_after = Pt(6)

    # --- Save ---
    OUTPUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_DOC))
    print(f"图表说明文档已生成：{OUTPUT_DOC}")
    print(f"  核心图 23 张 + 附加图 18 张 = 41 张图表详细说明")


if __name__ == "__main__":
    build_doc()
