#!/usr/bin/env python3
import argparse
from pathlib import Path


def pick_split(ds, preferred: str, fallbacks):
    if preferred in ds:
        return preferred
    for s in fallbacks:
        if s in ds:
            return s
    raise SystemExit(f"Split '{preferred}' not found. Available: {list(ds.keys())}")


def main():
    p = argparse.ArgumentParser(description="Download a HF image dataset and export to ImageFolder layout.")
    p.add_argument("--dataset", default="slegroux/tiny-imagenet-200-clean",
                   help="HF dataset repo id.")
    p.add_argument("--out", required=True,
                   help="Base output directory. A dataset root folder will be created under this path.")
    p.add_argument("--root-name", default="imagenet1k",
                   help="Name of dataset root folder created under --out (default: imagenet1k).")

    p.add_argument("--train-split", default="train",
                   help="Preferred training split name.")
    p.add_argument("--val-split", default="validation",
                   help="Preferred validation split name (default: validation).")

    p.add_argument("--image-col", default="image", help="Image column name.")
    p.add_argument("--label-col", default="label", help="Label column name.")
    p.add_argument("--limit", type=int, default=0, help="Optional limit per split (0 = no limit).")
    p.add_argument("--num-proc", type=int, default=4, help="Parallel workers for export.")
    args = p.parse_args()

    from datasets import load_dataset

    base_out = Path(args.out).expanduser().resolve()
    root_out = base_out / args.root_name
    train_dir = root_out / "train"
    val_dir = root_out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)

    train_split = pick_split(ds, args.train_split, fallbacks=["training"])
    val_split = pick_split(ds, args.val_split, fallbacks=["valid", "val", "dev", "eval"])

    train = ds[train_split]
    val = ds[val_split]

    if args.limit and args.limit > 0:
        train = train.select(range(min(args.limit, len(train))))
        val = val.select(range(min(args.limit, len(val))))

    label_feat = train.features.get(args.label_col, None)
    if label_feat is None or not hasattr(label_feat, "names"):
        raise SystemExit(
            "Label feature does not expose .names. "
            "Inspect dataset features and set --label-col accordingly."
        )
    class_names = list(label_feat.names)

    print(f"[+] Dataset: {args.dataset}")
    print(f"[+] Splits: train={train_split} ({len(train)}), val={val_split} ({len(val)})")
    print(f"[+] Classes: {len(class_names)}")
    print(f"[+] Export root: {root_out}")
    print(f"[+] Layout: {root_out}/train/<class>/*.jpg and {root_out}/val/<class>/*.jpg")

    for cn in class_names:
        (train_dir / cn).mkdir(parents=True, exist_ok=True)
        (val_dir / cn).mkdir(parents=True, exist_ok=True)

    def export_split(split_ds, split_out_dir: Path, split_tag: str):
        def _save(example, idx):
            img = example[args.image_col]  # PIL image
            y = int(example[args.label_col])
            cn = class_names[y]
            fp = split_out_dir / cn / f"{split_tag}_{idx:08d}.jpg"
            img.save(fp, format="JPEG", quality=95)
            return {}

        split_ds.map(_save, with_indices=True, num_proc=args.num_proc)
        print(f"[+] Exported {split_tag} -> {split_out_dir}")

    export_split(train, train_dir, "train")
    export_split(val, val_dir, "val")

    print("[+] Done.")
    print(f"Use this as your --data-path: {root_out}")


if __name__ == "__main__":
    main()
