import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))

from prepare_dataset import SOURCE, prepare_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the pothole dataset for training.")
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        default=[],
        help="Optional dataset root to merge before training. Repeat to combine multiple sources.",
    )
    args = parser.parse_args()
    selected_sources = [Path(value) for value in args.sources] if args.sources else [SOURCE]
    print(json.dumps(prepare_dataset(source_roots=selected_sources), indent=2))
