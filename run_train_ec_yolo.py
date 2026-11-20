import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from train_ec_yolo import train_ec_yolo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom EC-YOLO on the prepared pothole dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size (default: 4)")
    parser.add_argument("--image-size", type=int, default=640, help="Square detector input resolution (default: 640)")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate (default: 1e-3)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from artifacts/ec_yolo_training/last.pt if it exists",
    )
    args = parser.parse_args()

    summary = train_ec_yolo(
        ROOT / "prepared_dataset",
        ROOT / "artifacts" / "ec_yolo_training",
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))