"""论文图表一键生成脚本 — 在 Spyder 中打开后点击 Run (F5) 即可。

生成的所有图表将输出到 outputs/figures/final/ 目录。
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vis_paper import run_all

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════╗
║   统计建模论文图表一键生成脚本       ║
║   制造业"人工智能+"提质增效研究       ║
╚══════════════════════════════════════╝
""")
    run_all()
