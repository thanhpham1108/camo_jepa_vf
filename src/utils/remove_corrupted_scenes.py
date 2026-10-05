#!/usr/bin/env python3
"""Script loại bỏ các scenes bị hỏng / thiếu ảnh khỏi datasets_converted."""

import os
import shutil
from pathlib import Path
import numpy as np

TARGET_CORRUPTED = [
    "20260318_1603_VF6_01_1773824636_1773828236_1773826644_1773827121_colored_scans_latlon.npz",
    "20260303_1428_VF6_01_1772522906_1772524656_1772523588_1772523888_colored_scans_latlon_0.75.npz",
    "20260318_1603_VF6_01_1773824636_1773828236_1773827681_1773827881_colored_scans_latlon.npz",
    "20260318_1603_VF6_01_1773824636_1773828236_1773827881_1773828001_colored_scans_latlon.npz",
]


def main():
    dataset_root = Path(os.environ.get("DATASET_ROOT", "/dataset/camo_jepa/datasets_converted"))
    episodes_dir = dataset_root / "episodes" / "train"
    images_dir = dataset_root / "images" / "train"

    print("=" * 65)
    print("🧹 LOẠI BỎ CÁC SCENES HỎNG KHỎI DATASETS_CONVERTED")
    print(f"📁 Thư mục dataset: {dataset_root}")
    print("=" * 65)

    if not episodes_dir.exists():
        print(f"❌ Không tìm thấy thư mục episodes: {episodes_dir}")
        return

    # Thư mục cách ly (nếu muốn giữ lại backup thay vì xoá hẳn)
    quarantine_dir = dataset_root / "quarantined_episodes"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    removed_count = 0
    for scene_npz in TARGET_CORRUPTED:
        npz_file = episodes_dir / scene_npz
        scene_name = scene_npz.replace(".npz", "")
        img_folder = images_dir / scene_name

        if npz_file.exists():
            # Di chuyển sang thư mục quarantine để cách ly an toàn
            dest_npz = quarantine_dir / scene_npz
            shutil.move(str(npz_file), str(dest_npz))
            print(f"  📦 Đã cách ly .npz: {scene_npz} -> quarantined_episodes/")
            removed_count += 1
        else:
            print(f"  ℹ️ Không tìm thấy (có thể đã xoá): {scene_npz}")

        # Xoá folder symlink ảnh rỗng nếu có
        if img_folder.exists() and img_folder.is_dir():
            shutil.rmtree(str(img_folder), ignore_errors=True)
            print(f"  🗑️ Đã xoá folder ảnh rỗng: images/train/{scene_name}")

    remaining_npz = len(list(episodes_dir.glob("*.npz")))
    print("\n" + "=" * 65)
    print(f"✅ ĐÃ CÁCH LY THÀNH CÔNG {removed_count} SCENES BỊ LỖI.")
    print(f"🎉 Số scenes sạch còn lại trong train: {remaining_npz} scenes.")
    print("=" * 65)


if __name__ == "__main__":
    main()
