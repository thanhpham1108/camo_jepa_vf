#!/usr/bin/env python3
"""Package the validation (eval) dataset into /workspaces/result for 1-click download from Azure DevOps."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package CaMo-JEPA evaluation split into an archive for download."
    )
    parser.add_argument(
        "--converted-dir",
        type=Path,
        default=Path(os.environ.get("DATASETS_CONVERTED_DIR", "/dataset/camo_jepa/datasets_converted_interp")),
        help="Path to converted dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("RUN_DIR", "/workspaces/result/run_dir")),
        help="Destination directory (defaults to $RUN_DIR or /workspaces/result/run_dir).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("val", "train"),
        help="Split to package (default: val).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Package only .npz episodes and manifest (exclude image files for a tiny ~15MB archive).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    converted_dir = args.converted_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_dir = converted_dir / "episodes" / args.split
    if not episodes_dir.is_dir():
        print(f"❌ Error: Không tìm thấy thư mục {episodes_dir}")
        return 1

    npz_files = sorted(episodes_dir.glob("*.npz"))
    if not npz_files:
        print(f"❌ Error: Không có file .npz nào trong {episodes_dir}")
        return 1

    archive_name = f"camo_jepa_{args.split}_dataset"
    if args.no_images:
        archive_name += "_npz_only"
    archive_tar_path = output_dir / f"{archive_name}.tar.gz"

    staging_dir = output_dir / archive_name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"📦 ĐÓNG GÓI TẬP EVAL ({args.split.upper()}) ĐỂ TẢI VỀ MÁY")
    print("=" * 70)
    print(f"📁 Nguồn dataset : {converted_dir}")
    print(f"📁 Nơi lưu gói   : {archive_tar_path}")
    print(f"📄 Số lượng scene: {len(npz_files)} episodes")
    print(f"🖼️  Kèm ảnh thật : {'Không (chỉ lấy .npz)' if args.no_images else 'Có (dereference symlinks)'}")
    print("=" * 70)

    # 1. Copy dataset_info.json
    info_src = converted_dir / "dataset_info.json"
    if info_src.exists():
        shutil.copy2(info_src, staging_dir / "dataset_info.json")

    # 2. Copy các file .npz vào episodes/<split>/
    staging_episodes = staging_dir / "episodes" / args.split
    staging_episodes.mkdir(parents=True, exist_ok=True)
    val_scene_names = set()
    for npz in npz_files:
        val_scene_names.add(npz.stem)
        shutil.copy2(npz, staging_episodes / npz.name)
    print(f"  ✅ Đã gom {len(npz_files)} file .npz vào episodes/{args.split}/")

    # 3. Lọc riêng manifest cho tập val
    manifest_src = converted_dir / "manifest.jsonl"
    if manifest_src.exists():
        val_manifest_lines = []
        with manifest_src.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    ep = data.get("episode", "")
                    # Khớp episode thuộc split val
                    if f"episodes/{args.split}/" in ep or any(sc in ep for sc in val_scene_names):
                        val_manifest_lines.append(line)
                except Exception:
                    pass
        if val_manifest_lines:
            (staging_dir / f"manifest_{args.split}.jsonl").write_text("".join(val_manifest_lines), encoding="utf-8")
            print(f"  ✅ Đã trích xuất {len(val_manifest_lines):,} frames vào manifest_{args.split}.jsonl")

    # 4. Copy ảnh thật (dereference symlinks) nếu không bật --no-images
    if not args.no_images:
        images_src = converted_dir / "images" / args.split
        staging_images = staging_dir / "images" / args.split
        staging_images.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        print("\n🖼️  Đang gom ảnh thật (copy dereference symlinks từ camera)...")
        for idx, scene_name in enumerate(sorted(val_scene_names), 1):
            scene_img_src = images_src / scene_name
            scene_img_dst = staging_images / scene_name
            scene_img_dst.mkdir(parents=True, exist_ok=True)
            if scene_img_src.exists():
                for img_file in scene_img_src.glob("*.jpg"):
                    real_file = img_file.resolve()
                    if real_file.exists():
                        shutil.copy2(real_file, scene_img_dst / img_file.name)
                        copied_images += 1
            if idx % 10 == 0 or idx == len(val_scene_names):
                print(f"   [{idx}/{len(val_scene_names)}] {scene_name} (Tổng ảnh đã copy: {copied_images:,})")
        print(f"  ✅ Hoàn tất copy {copied_images:,} ảnh vào images/{args.split}/")

    # 5. Nén toàn bộ thư mục staging thành .tar.gz
    print(f"\n🗜️  Đang nén thành file {archive_tar_path.name}...")
    with tarfile.open(archive_tar_path, "w:gz") as tar:
        tar.add(staging_dir, arcname=archive_name)

    # Dọn dẹp staging folder sau khi nén
    shutil.rmtree(staging_dir, ignore_errors=True)

    # Đảm bảo file có mặt ở cả RUN_DIR lẫn /workspaces/result để chắc chắn được tải lên Azure
    result_parent = Path("/workspaces/result")
    if result_parent.exists() and output_dir != result_parent:
        alt_tar = result_parent / archive_tar_path.name
        try:
            if not alt_tar.exists():
                shutil.copy2(archive_tar_path, alt_tar)
                print(f"  🔗 Đã sao lưu bổ sung sang: {alt_tar}")
        except Exception:
            pass

    size_mb = archive_tar_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f"🎉 ĐÓNG GÓI THÀNH CÔNG GÓI DỮ LIỆU EVAL!")
    print(f"📦 Vị trí file: {archive_tar_path}")
    print(f"⚖️  Dung lượng : {size_mb:.1f} MB ({size_mb / 1024:.2f} GB)")
    print(f"👉 File đã nằm trong $RUN_DIR ({output_dir}).")
    print(f"   Toàn bộ thư mục này sẽ được nén và đẩy lên Azure DevOps Artifacts.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
