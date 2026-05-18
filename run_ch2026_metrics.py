# ================================
# run_ch2026_metrics.py
#
# Command-line entry point for the CH2026 metrics baseline pipeline.
#
# Functions
#   - main() -> None : Parse paths and run the pipeline.
# ================================

from __future__ import annotations

import argparse
from pathlib import Path

from src.ch2026_metrics import run_pipeline


def main() -> None:
    """Parse command-line arguments and run the CH2026 metrics pipeline."""

    parser = argparse.ArgumentParser(description="Train CH2026 metrics baseline and create a submission CSV.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing CH2026 data files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated CSV outputs.")
    parser.add_argument(
        "--metric",
        choices=["f1", "accuracy", "xgb-variants", "lstm"],
        default="f1",
        help="Selection metric or experiment family for final target strategies.",
    )
    parser.add_argument("--install-check", action="store_true", help="Print optional dependency availability before running.")
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir, metric=args.metric, install_check=args.install_check)


if __name__ == "__main__":
    main()
