#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Download a HF image classification dataset and export to ImageFolder layout.")
    p.add_argument("--dataset", default="slegroux/tiny-imagenet-200-clean",
                   help="HF dataset repo id (default: tiny-imagenet-200-clean).")
    p.add_argument("--out", required=True, help="Output root directory. Will create train/ and val/ inside.")
    p.add_argument("--train-split", default="train", help="Dataset split name for training.")
    p.add_argument("--val-split", default="valid", help="Dataset split name for validation (common: valid or validation).")
    p.add_argument("--image-col", default="image", help="Image column name (default: image).")
    p.add_argument("--label-col", default="label", help="Label column name (default: label).")
    p.add_argument("--limit", type=int, default=0, help="Optional limit per split (0 = no limit).")
    p.add_argument("--num-proc", type=int, default=4, help="Parallel workers for export.")
    args = p.parse_args()

    from datasets import load_dataset

    out = Path(args.out).expanduser().resolve()
    train_dir = out / "train"
    val_dir = out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)

    if args.train_split not in ds:
        raise SystemExit(f"Train split '{args.train_split}' not found. Available: {list(ds.keys())}")
    if args.val_split not in ds:
        raise SystemExit(f"Val split '{args.val_split}' not found. Available: {list(ds.keys())}")

    train = ds[args.train_split]
    val = ds[args.val_split]

    # Optional downsample
    if args.limit and args.limit > 0:
        train = train.select(range(min(args.limit, len(train))))
        val = val.select(range(min(args.limit, len(val))))

    # Labels -> class names
    label_feat = train.features.get(args.label_col, None)
    if label_feat is None or not hasattr(label_feat, "names"):
        raise SystemExit("Label feature does not expose .names; dataset may not be standard classif. "
                         "Inspect dataset features and set --label-col accordingly.")
    class_names = list(label_feat.names)

    print(f"[+] Dataset: {args.dataset}")
    print(f"[+] Classes: {len(class_names)}")
    print(f"[+] Exporting to: {out}")
    print(f"[+] Train split: {args.train_split} ({len(train)} samples)")
    print(f"[+] Val split:   {args.val_split} ({len(val)} samples)")

    # Pre-create class dirs
    for cn in class_names:
        (train_dir / cn).mkdir(parents=True, exist_ok=True)
        (val_dir / cn).mkdir(parents=True, exist_ok=True)

    def export_split(split_ds, split_out_dir: Path, split_name: str):
        def _save(example, idx):
            img = example[args.image_col]  # PIL image
            y = example[args.label_col]
            cn = class_names[int(y)]
            # keep it simple: sequential filenames
            fp = split_out_dir / cn / f"{split_name}_{idx:08d}.jpg"
            img.save(fp, format="JPEG", quality=95)
            return {}

        # map with indices
        split_ds.map(_save, with_indices=True, num_proc=args.num_proc)
        print(f"[+] Exported {split_name} to {split_out_dir}")

    export_split(train, train_dir, "train")
    export_split(val, val_dir, "val")

    print("[+] Done.")
    print("Expected layout:")
    print(f"  {out}/train/<class>/*.jpg")
    print(f"  {out}/val/<class>/*.jpg")

if __name__ == "__main__":
    main()
