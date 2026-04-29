from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_did import run_all_did_checks
from src.config import LOGS_DIR, ensure_project_dirs


def setup_logging() -> None:
    ensure_project_dirs()
    log_path = LOGS_DIR / "did_checks.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    setup_logging()
    panel_path = ROOT / "data" / "processed" / "manufacturing_modeling_panel.csv"
    logging.info("Reading modeling panel from %s", panel_path)
    panel = pd.read_csv(panel_path, dtype={"code": str})
    outputs = run_all_did_checks(panel)
    for name, table in outputs.items():
        logging.info("%s result rows: %s", name, len(table))
    logging.info("DID checks finished successfully")


if __name__ == "__main__":
    main()
