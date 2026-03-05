from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-subject 3D pose reconstruction (YOLO person detection + SAM-3D-Body)"
    )
    p.add_argument("--image", required=True, help="Path to input image (jpg/png)")
    p.add_argument("--out_dir", default="outputs", help="Output directory (default: outputs)")

    # pipeline options (mirrors run_pipeline signature)
    p.add_argument("--hf_repo", default="facebook/sam-3d-body-dinov3", help="Hugging Face repo for checkpoints")
    p.add_argument("--ckpt_dir", default="checkpoints/sam-3d-body", help="Checkpoint directory")
    p.add_argument("--sam_repo_url", default="https://github.com/facebookresearch/sam-3d-body.git", help="SAM-3D repo URL")
    p.add_argument("--sam_repo_dir", default="third_party/sam-3d-body", help="Local clone directory for SAM-3D repo")

    p.add_argument("--yolo_weights", default="yolov8x.pt", help="Ultralytics YOLO weights (default: yolov8x.pt)")
    p.add_argument("--conf", type=float, default=0.20, help="YOLO confidence threshold (default: 0.20)")
    p.add_argument("--iou", type=float, default=0.50, help="YOLO IoU threshold (default: 0.50)")
    p.add_argument("--pad", type=float, default=0.15, help="Padding around bbox for crop (default: 0.15)")

    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: auto/cpu/cuda (default: auto)",
    )
    p.add_argument("--hf_token", default=None, help="Optional HF token. If not set, uses env var HF_TOKEN")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device_arg = None if args.device == "auto" else args.device

    merged_obj, individual_objs = run_pipeline(
        image=str(image_path),
        out_dir=args.out_dir,
        hf_repo=args.hf_repo,
        ckpt_dir=args.ckpt_dir,
        sam_repo_url=args.sam_repo_url,
        sam_repo_dir=args.sam_repo_dir,
        yolo_weights=args.yolo_weights,
        conf=args.conf,
        iou=args.iou,
        pad=args.pad,
        device=device_arg,
        hf_token=args.hf_token,
    )

    print("\n✅ Done.")
    print(f"✅ Merged mesh: {merged_obj}")
    for pth in individual_objs:
        print(f"✅ Person mesh: {pth}")


if __name__ == "__main__":
    main()
