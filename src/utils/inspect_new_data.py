#!/usr/bin/env python3
"""Script kiểm tra và đánh giá cấu trúc dữ liệu mới tại /dataset/data260926."""

import os
import sys
from pathlib import Path

def inspect():
    target_path = Path("/dataset/data260926")
    dataset_parent = Path("/dataset")

    print("=" * 60)
    print("🔍 KIỂM TRA HẠ TẦNG & DỮ LIỆU MỚI: /dataset/data260926")
    print("=" * 60)

    # 1. Kiểm tra thư mục cha /dataset
    if dataset_parent.exists():
        print(f"\n📂 Các thư mục hiện có trong {dataset_parent}:")
        try:
            items = os.listdir(dataset_parent)
            for item in sorted(items):
                p = dataset_parent / item
                suffix = "/" if p.is_dir() else ""
                print(f"  ├── {item}{suffix}")
        except Exception as e:
            print(f"  ❌ Không thể đọc {dataset_parent}: {e}")
    else:
        print(f"❌ Thư mục {dataset_parent} không tồn tại bên trong container!")
        return

    # 2. Kiểm tra thư mục mục tiêu /dataset/data260926
    if not target_path.exists():
        print(f"\n❌ THƯ MỤC {target_path} CHƯA ĐƯỢC MOUNT HOẶC KHÔNG TỒN TẠI!")
        print("💡 Gợi ý: Kiểm tra lại cấu hình --container-mounts của Slurm/Airflow.")
        return

    print(f"\n✅ Tìm thấy thư mục: {target_path}")
    sub_items = sorted(os.listdir(target_path))
    print(f"📊 Tổng số mục con bên trong: {len(sub_items)}")
    for item in sub_items[:15]:
        p = target_path / item
        suffix = "/" if p.is_dir() else ""
        print(f"  ├── {item}{suffix}")
    if len(sub_items) > 15:
        print(f"  └── ... và {len(sub_items) - 15} mục khác.")

    # 3. Phán đoán định dạng dữ liệu
    has_manifest = (target_path / "manifest.jsonl").exists()
    has_episodes = (target_path / "episodes").exists()
    has_images = (target_path / "images").exists()

    if has_manifest and has_episodes:
        print("\n🎉 KẾT QUẢ ĐÁNH GIÁ: DỮ LIỆU ĐÃ ĐƯỢC CONVERT SẴN (CaMo-JEPA v1 Format)!")
        print("Mô hình có thể đọc trực tiếp bằng cách trỏ DATASET_ROOT vào đây.")
        
        # Đếm số dòng manifest
        try:
            with open(target_path / "manifest.jsonl") as f:
                lines = sum(1 for _ in f)
            print(f"  * Số lượng clips/episodes trong manifest.jsonl: {lines:,}")
        except Exception as e:
            print(f"  * Lỗi đọc manifest.jsonl: {e}")

        # Thử chạy audit nếu có script
        try:
            from src.utils.audit_dataset import audit_dataset
            print("\n📈 Chạy phân tích động học dữ liệu (Audit):")
            audit_dataset(target_path)
        except Exception as e:
            print(f"  * Không thể chạy audit tự động: {e}")

    else:
        print("\n⚠️ KẾT QUẢ ĐÁNH GIÁ: ĐÂY CÓ THỂ LÀ DỮ LIỆU THÔ (RAW VINFAST RECORDINGS)!")
        print("Cần chạy script 'convert_vf_to_camo.py' trước khi huấn luyện.")
        
        # Kiểm tra chuyến xe đầu tiên
        for item in sub_items:
            trip_path = target_path / item
            if trip_path.is_dir():
                print(f"\n🔎 Soi chi tiết thư mục chuyến xe mẫu: {item}")
                trip_contents = os.listdir(trip_path)
                for tc in sorted(trip_contents):
                    print(f"    ├── {tc}")
                break

    print("\n" + "=" * 60)
    print("🏁 HOÀN TẤT KIỂM TRA")
    print("=" * 60)

if __name__ == "__main__":
    inspect()
