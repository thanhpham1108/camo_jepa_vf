#!/usr/bin/env python3
"""Script chẩn đoán chi tiết cấu trúc thư mục data raw để tìm vị trí thực của file ảnh."""

import os
from pathlib import Path

def main():
    print("=" * 70)
    print("🕵️  ĐIỀU TRA CHI TIẾT NƠI CẤT GIỮ ẢNH RAW")
    print("=" * 70)

    # 1. Soi thư mục /dataset/data260926/VF6_01_02
    vf_dir = Path("/dataset/data260926/VF6_01_02")
    if vf_dir.exists():
        subdirs = list(vf_dir.iterdir())
        print(f"\n📁 /dataset/data260926/VF6_01_02: Có {len(subdirs)} mục con.")
        print("   5 mục con đầu tiên:")
        for s in subdirs[:5]:
            print(f"     - {s.name} ({'DIR' if s.is_dir() else 'FILE'})")

        # Thử kiểm tra chính xác scene bị lỗi
        test_scene = vf_dir / "20260303_1239_VF6_01_1772516366_1772518447_1772517541_1772517558"
        print(f"\n🔍 Kiểm tra scene lỗi: {test_scene.name}")
        print(f"   -> Tồn tại thư mục scene? : {test_scene.exists()}")
        if test_scene.exists():
            cam_dir = test_scene / "CAMERA" / "CAM_P_F"
            print(f"   -> Tồn tại CAMERA/CAM_P_F?: {cam_dir.exists()}")
            if cam_dir.exists():
                jpgs = list(cam_dir.glob("*.jpg"))
                print(f"   -> Số lượng file .jpg bên trong: {len(jpgs)} ảnh")
                if jpgs:
                    print(f"      (Ví dụ ảnh: {jpgs[0].name})")
            else:
                others = list(test_scene.glob("*"))
                print(f"   -> Các mục bên trong scene: {[o.name for o in others]}")
    else:
        print(f"❌ Thư mục {vf_dir} không tồn tại!")

    # 2. Soi thư mục /dataset/camo_jepa/datasets_raw
    raw_dir = Path("/dataset/camo_jepa/datasets_raw")
    if raw_dir.exists():
        raw_items = list(raw_dir.iterdir())
        print(f"\n📁 /dataset/camo_jepa/datasets_raw: Có {len(raw_items)} mục con.")
        for item in raw_items:
            print(f"   - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
            
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
