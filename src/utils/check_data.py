#!/usr/bin/env python3
"""Script quét sâu toàn bộ data thô để kiểm tra cấu trúc từng scene (có CAMERA/ và OTHERS/ không)."""

import os
from pathlib import Path


SOURCE_DIRS = [
    "/dataset/data260926",
    "/dataset/camo_jepa/datasets_raw",
]


def scan_scenes(base_paths: list[str]):
    """
    Đi sâu qua tất cả các thư mục để tìm scenes (thư mục chứa nhánh CAMERA/).
    Với mỗi scene tìm được, kiểm tra xem có đủ CAMERA/ và OTHERS/ không.
    """
    ok_scenes = []
    missing_others = []
    missing_camera = []

    for base_str in base_paths:
        base = Path(base_str)
        if not base.exists():
            print(f"⚠️  Bỏ qua (không tồn tại): {base}")
            continue

        print(f"\n🔍 Đang quét: {base}")
        for root, dirs, _ in os.walk(base):
            root_path = Path(root)
            has_camera = "CAMERA" in dirs
            has_others = "OTHERS" in dirs

            if has_camera or has_others:
                scene_name = root_path.relative_to(base)
                if has_camera and has_others:
                    ok_scenes.append((base_str, str(scene_name)))
                elif has_camera and not has_others:
                    missing_others.append((base_str, str(scene_name)))
                else:
                    missing_camera.append((base_str, str(scene_name)))
                # Không đi sâu hơn vào bên trong scene
                dirs.clear()

    return ok_scenes, missing_others, missing_camera


def main():
    print("=" * 70)
    print("🕵️  QUÉT SÂU CẤU TRÚC DATA RAW — KIỂM TRA CAMERA/ & OTHERS/")
    print("=" * 70)

    ok, missing_others, missing_camera = scan_scenes(SOURCE_DIRS)

    total = len(ok) + len(missing_others) + len(missing_camera)
    print(f"\n📊 Tổng số scenes tìm thấy: {total}")
    print(f"   ✅ Đầy đủ (có cả CAMERA/ & OTHERS/) : {len(ok)}")
    print(f"   🔴 Thiếu OTHERS/                     : {len(missing_others)}")
    print(f"   ⚠️  Thiếu CAMERA/ (bất thường)        : {len(missing_camera)}")

    if missing_others:
        print(f"\n{'='*70}")
        print("🔴 DANH SÁCH SCENES CÒN THIẾU OTHERS/ (cần anh hạ tầng fix):")
        print(f"{'='*70}")
        for base, scene in missing_others:
            print(f"   📁 [{base}]  {scene}")

    if missing_camera:
        print(f"\n{'='*70}")
        print("⚠️  DANH SÁCH SCENES KHÔNG CÓ CAMERA/ (bất thường — kiểm tra lại):")
        print(f"{'='*70}")
        for base, scene in missing_camera:
            print(f"   📁 [{base}]  {scene}")

    print(f"\n{'='*70}")
    if len(missing_others) == 0:
        print("🎉 TẤT CẢ SCENES ĐÃ ĐẦY ĐỦ! Sẵn sàng để Convert.")
    else:
        print(f"❌ CÒN {len(missing_others)} SCENES THIẾU OTHERS/. Chưa nên Convert vội.")
    print("=" * 70)


if __name__ == "__main__":
    main()

