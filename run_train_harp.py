import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from train_harp import train_harp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HARP-UNet on the prepared ROI pothole dataset")
    parser.add_argument("--data-root", type=Path, default=ROOT / "prepared_roi_dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "harp_roi_training")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs (default: 30)")
    parser.add_argument("--image-size", type=int, default=128, help="Square input resolution (default: 128)")
    parser.add_argument("--batch-size", type=int, default=4, help="Samples per mini-batch (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate (default: 1e-3)")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the selected output directory's last.pt checkpoint",
    )
    args = parser.parse_args()

    summary = train_harp(
        args.data_root,
        args.output_dir,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))