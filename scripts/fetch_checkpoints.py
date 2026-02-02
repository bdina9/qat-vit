#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import urllib.request


DEFAULTS = {
    # If you want, we can add more model types later
    "ViT-B_16": {
        # Official Google ViT NPZ weights (ImageNet21k pretrain) are commonly used
        # NOTE: if this URL changes/breaks, pass --teacher-url explicitly.
        "teacher_url": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz",
        "teacher_filename": "ViT-B_16.npz",
    }
}


def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[=] Exists, skipping: {out_path}")
        return

    print(f"[+] Downloading:\n    {url}\n -> {out_path}")
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(out_path)
    print(f"[+] Saved: {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")


def main():
    p = argparse.ArgumentParser(description="Fetch required checkpoints into a folder.")
    p.add_argument("--out-dir", required=True, help="Directory to place checkpoints into.")
    p.add_argument("--model-type", default="ViT-B_16", help="Model type (default: ViT-B_16)")
    p.add_argument("--teacher-url", default=None,
                   help="Override URL for teacher checkpoint (NPZ). If not set, uses default known URL.")
    p.add_argument("--teacher-name", default=None,
                   help="Override output filename for teacher checkpoint.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type not in DEFAULTS:
        raise SystemExit(f"Unknown model-type {args.model_type}. Known: {list(DEFAULTS.keys())}")

    teacher_url = args.teacher_url or DEFAULTS[args.model_type]["teacher_url"]
    teacher_name = args.teacher_name or DEFAULTS[args.model_type]["teacher_filename"]
    teacher_path = out_dir / teacher_name

    download(teacher_url, teacher_path)

    print("\n[+] Done. Checkpoints available:")
    print(f"    {teacher_path}")
    print("\nUse with:")
    print(f"  bash calib.sh --data-dir <DATASET_ROOT> --ckpt-dir {out_dir}")
    print(f"  bash qat.sh   --data-dir <DATASET_ROOT> --ckpt-dir {out_dir}")


if __name__ == "__main__":
    main()
