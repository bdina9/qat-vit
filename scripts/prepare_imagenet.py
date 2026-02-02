#!/usr/bin/env python3
import argparse
import os
import tarfile
import shutil
from pathlib import Path


def extract_tar(tar_path: Path, dst: Path):
    print(f"[+] Extracting: {tar_path} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=dst)


def expand_train_class_tars(train_dir: Path):
    # ImageNet train tar expands into many <wnid>.tar files in train_dir
    tars = sorted(train_dir.glob("*.tar"))
    if not tars:
        print("[!] No class tar files found in train/ (maybe already expanded).")
        return

    print(f"[+] Expanding {len(tars)} class tars in {train_dir}")
    for tf in tars:
        wnid = tf.stem
        wnid_dir = train_dir / wnid
        wnid_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tf, "r:*") as tar:
            tar.extractall(path=wnid_dir)
        tf.unlink()  # remove class tar
    print("[+] Train class tar expansion complete.")


def organize_val(val_dir: Path, val_map_file: Path):
    """
    val_map_file must contain lines:
      ILSVRC2012_val_00000001.JPEG n01440764
    """
    print(f"[+] Organizing val images using map: {val_map_file}")

    # Read mapping
    mapping = {}
    with open(val_map_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad mapping line: {line}")
            img, wnid = parts
            mapping[img] = wnid

    # Move files
    moved = 0
    for img_name, wnid in mapping.items():
        src = val_dir / img_name
        if not src.exists():
            continue
        dst_dir = val_dir / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_dir / img_name))
        moved += 1

    print(f"[+] Moved {moved} val images into class folders.")


def main():
    p = argparse.ArgumentParser(description="Prepare ImageNet train/val folder structure from ILSVRC2012 tarballs.")
    p.add_argument("--root", required=True, help="Root directory where ImageNet will be prepared.")
    p.add_argument("--train-tar", default="ILSVRC2012_img_train.tar", help="Train tar filename inside --root")
    p.add_argument("--val-tar", default="ILSVRC2012_img_val.tar", help="Val tar filename inside --root")
    p.add_argument("--val-map", default=None,
                   help="Optional mapping file: '<val_image_name> <wnid>' per line to organize val/ into wnid folders.")
    p.add_argument("--force", action="store_true", help="Overwrite existing train/val directories")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    train_tar = root / args.train_tar
    val_tar = root / args.val_tar

    if not train_tar.exists():
        raise FileNotFoundError(f"Missing train tar: {train_tar}")
    if not val_tar.exists():
        raise FileNotFoundError(f"Missing val tar: {val_tar}")

    train_dir = root / "train"
    val_dir = root / "val"

    if args.force:
        if train_dir.exists():
            print(f"[!] Removing existing: {train_dir}")
            shutil.rmtree(train_dir)
        if val_dir.exists():
            print(f"[!] Removing existing: {val_dir}")
            shutil.rmtree(val_dir)

    # Extract train tar -> produces class tar files
    extract_tar(train_tar, train_dir)
    expand_train_class_tars(train_dir)

    # Extract val tar -> flat images
    extract_tar(val_tar, val_dir)

    # Optional: organize val into wnid folders
    if args.val_map:
        val_map_file = Path(args.val_map).expanduser().resolve()
        if not val_map_file.exists():
            raise FileNotFoundError(f"Missing val map: {val_map_file}")
        organize_val(val_dir, val_map_file)
    else:
        print("[!] Val images extracted but NOT organized into wnid folders.")
        print("    Provide --val-map to sort val images into class directories.")

    print("\n[+] Done.")
    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")


if __name__ == "__main__":
    main()
