#!/usr/bin/env python3
"""Script quét toàn bộ Dataset trên cụm SLURM để thống kê chính xác bao nhiêu % dữ liệu bị hỏng/thiếu."""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    dataset_root = Path(os.environ.get("DATASET_ROOT", "/dataset/camo_jepa/datasets_converted"))
    episodes_dir = dataset_root / "episodes" / "train"
    images_dir = dataset_root

    print("="*60)
    print("🕵️  BẮT ĐẦU QUÉT TOÀN BỘ DATASET ĐỂ TÌM FILE LỖI")
    print(f"📁 Thư mục gốc: {dataset_root}")
    print("="*60)

    if not episodes_dir.exists():
        print(f"❌ Không tìm thấy thư mục episodes: {episodes_dir}")
        return

    npz_files = list(episodes_dir.glob("*.npz"))
    print(f"📊 Tìm thấy tổng cộng {len(npz_files)} scenes (.npz files). Bắt đầu quét...\n")

    total_images = 0
    total_missing = 0
    corrupted_scenes = 0
    corrupted_scene_names = []

    for npz_path in tqdm(npz_files, desc="Đang quét file .npz"):
        try:
            with np.load(npz_path) as data:
                image_paths = data["image_paths"]
                
            scene_missing = 0
            for rel_path in image_paths:
                total_images += 1
                full_path = images_dir / str(rel_path)
                if not full_path.exists():
                    scene_missing += 1
                    total_missing += 1
                    
            if scene_missing > 0:
                corrupted_scenes += 1
                corrupted_scene_names.append((npz_path.name, scene_missing, len(image_paths)))
                
        except Exception as e:
            print(f"\n❌ Lỗi khi đọc {npz_path.name}: {e}")
            corrupted_scenes += 1

    print("\n" + "="*60)
    print("✨ KẾT QUẢ QUÉT DATASET ✨")
    print("="*60)
    print(f"🔹 Tổng số Scenes: {len(npz_files)}")
    print(f"🔹 Tổng số Ảnh cần có: {total_images:,}")
    print(f"🔴 Tổng số Ảnh bị THIẾU: {total_missing:,}")
    
    if total_images > 0:
        missing_percent = (total_missing / total_images) * 100
        print(f"📉 Tỷ lệ dữ liệu bị hỏng: {missing_percent:.4f}%")
    
    print(f"🔴 Số Scenes bị lỗi một phần hoặc toàn bộ: {corrupted_scenes} / {len(npz_files)}")

    if corrupted_scenes > 0:
        print("\n📝 DANH SÁCH TOP 10 SCENES LỖI NẶNG NHẤT:")
        # Sắp xếp theo số lượng ảnh thiếu giảm dần
        corrupted_scene_names.sort(key=lambda x: x[1], reverse=True)
        for scene_name, missing, total in corrupted_scene_names[:10]:
            print(f"   - {scene_name} (Thiếu {missing}/{total} ảnh)")
            
    print("="*60)

if __name__ == "__main__":
    main()
