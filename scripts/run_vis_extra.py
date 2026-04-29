"""一键运行附加丰富图表 — 在 Spyder 中直接 Run 即可。

生成 3 张流程图 + 15 张多样统计图 = 18 张额外可选图表。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vis_extra import run_all

if __name__ == "__main__":
    run_all()
