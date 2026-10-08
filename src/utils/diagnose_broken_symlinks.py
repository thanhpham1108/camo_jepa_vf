#!/usr/bin/env python3
"""Script chẩn đoán nguyên nhân đứt symlink ảnh trong datasets_converted."""

import os
from pathlib import Path

def main():
    dataset_root = Path("/dataset/camo_jepa/datasets_converted")
    images_dir = dataset_root / "images" / "train"

    print("=" * 70)
    print("🔍 CHẨN ĐOÁN LỖI 'IMAGE DOES NOT EXIST' / ĐỨT SYMLINK")
    print("=" * 70)

    # 1. Kiểm tra sự tồn tại của các thư mục gốc raw data
    raw_dirs = [
        Path("/dataset/data260926"),
        Path("/dataset/camo_jepa/datasets_raw"),
        Path("/dataset/camo_jepa"),
    ]
    print("\n📁 1. KIỂM TRA CÁC THƯ MỤC RAW TRÊN SERVER:")
    for rd in raw_dirs:
        exists = rd.exists()
        is_dir = rd.is_dir() if exists else False
        item_count = len(list(rd.iterdir())) if is_dir else 0
        status = f"✅ TỒN TẠI ({item_count} mục)" if exists else "❌ KHÔNG TỒN TẠI (ĐÃ BỊ XÓA/MOVE HOẶC CHƯA MOUNT)"
        print(f"   - {rd}: {status}")

    # 2. Kiểm tra một vài file ảnh symlink bị lỗi
    print("\n🔗 2. SOI CHI TIẾT CÁC CON TRỎ SYMLINK BỊ LỖI:")
    if not images_dir.exists():
        print(f"❌ Thư mục {images_dir} không tồn tại!")
        return

    sample_count = 0
    broken_count = 0
    valid_count = 0

    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.endswith(".jpg"):
                continue
            file_path = Path(root) / f
            is_symlink = file_path.is_symlink()
            target = os.readlink(file_path) if is_symlink else "Not a symlink"
            resolved = file_path.resolve()
            target_exists = file_path.exists()

            if not target_exists:
                broken_count += 1
                if sample_count < 5:
                    print(f"\n   [LỖI #{broken_count}] File: {file_path.name}")
                    print(f"      - Là symlink?        : {is_symlink}")
                    print(f"      - Trỏ tới (relpath)  : {target}")
                    print(f"      - Đường dẫn thực tế  : {resolved}")
                    print(f"      - Thực tế có tồn tại?: {target_exists}")
                    sample_count += 1
            else:
                valid_count += 1

            if broken_count + valid_count >= 1000:
                break
        if broken_count + valid_count >= 1000:
            break

    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ QUÉT 1,000 ẢNH MẪU ĐẦU TIÊN:")
    print(f"   ✅ Ảnh hợp lệ: {valid_count} / 1000")
    print(f"   🔴 Ảnh bị đứt link / mất: {broken_count} / 1000")
    print("=" * 70)

if __name__ == "__main__":
    main()
