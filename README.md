# 统计建模 — 制造业"人工智能+"提质增效

A股制造业上市公司面板数据实证研究。代码可直接在 Spyder 中运行。

## 环境

```bash
pip install -r requirements.txt
```

核心依赖: pandas, numpy, statsmodels, scikit-learn, matplotlib, openpyxl

## 快速运行

1. 在 Spyder 中打开 `scripts/run_vis_paper.py`
2. 点击 Run (F5)
3. 所有图表输出到 `outputs/figures/final/`

## 目录结构

```
├── src/                # 核心代码
│   ├── config.py       # 路径、全局常量和参数
│   ├── visualization.py # 原有8张基础图表
│   └── vis_paper.py    # 新增14张论文精美图表
├── scripts/            # 一键运行入口
│   ├── run_pipeline.py
│   └── run_vis_paper.py
├── data/
│   ├── raw/            # 原始Excel（不上传）
│   └── processed/      # 清洗后数据
├── outputs/
│   ├── figures/final/  # 论文图表（PNG+PDF双格式）
│   ├── tables/final/   # 论文回归表
│   └── tables/         # 诊断/模型结果表
└── requirements.txt
```

## 图表清单

| 图表 | 文件名 | 论文位置 |
|------|--------|----------|
| 数据筛选流程 | fig_s1_sample_flow | 第2章 研究设计 |
| 提质增效指数结构 | fig9_index_structure | §2.1 变量构造 |
| 描述性统计 | fig10_descriptive_stats | §2.1 变量构造 |
| 变量缺失率 | fig_s2_missingness | 附录 |
| 多重共线性VIF | fig_s3_vif | §2.2 模型诊断 |
| 异方差诊断 | fig_s4_heteroskedasticity | §2.2 模型诊断 |
| AI应用分箱图 | fig7 (优化) | §3.1 描述性分析 |
| 分箱散点图 | fig11_binned_scatter | §3.1 描述性分析 |
| 控制后偏相关 | fig12_partial_correlation | §3.2 基准回归 |
| 稳健性森林图 | fig13_robustness_forest | §3.3 稳健性 |
| 机制路径图 | fig5 (重做) | §3.4 机制检验 |
| 局部拟合曲线 | fig14_loess_fit | §3.4 机制检验 |
| 配对异质性 | fig6 (重做) | §3.5 异质性分析 |
| 分位数效应 | fig15_quantile_effect | §3.6 扩展模型 |
| 模型对比 | fig16_model_comparison | §3.6 扩展模型 |
| 边界检验 | fig17_boundary_comparison | §4.2 结论边界 |
| 调节效应 | fig_s6_conversion_moderation | 附录 |
| 变量筛选热力 | fig_s5_xy_screening | 附录 |

## 全局风格

- 配色: 蓝(#0072B2) 青(#009E73) 橙(#D55E00) 灰(#6E6E6E)
- 字体: 微软雅黑 + 无衬线备选
- 分辨率: 360 DPI, PNG+PDF双格式
- 标准误: 公司聚类稳健标准误
