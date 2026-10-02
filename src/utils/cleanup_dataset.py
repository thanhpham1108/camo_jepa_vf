#!/usr/bin/env python3
"""Script dọn dẹp và tổ chức lại thư mục /dataset/camo_jepa trên server."""

import os
import shutil
from pathlib import Path

def organize_dataset():
    root = Path("/dataset/camo_jepa")
    
    if not root.exists():
        print(f"❌ Không tìm thấy thư mục {root}")
        return

    # Định nghĩa cấu trúc mới
    dirs = {
        "raw": root / "datasets_raw",
        "models": root / "models"
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    print("=" * 60)
    print("🧹 BẮT ĐẦU DỌN DẸP VÀ TỔ CHỨC LẠI /dataset/camo_jepa")
    print("=" * 60)

    # 1. Xoá bộ data convert cũ
    legacy_items = ["episodes", "images", "manifest.jsonl", "dataset_info.json"]
    print("\n🗑️ Đang xóa vĩnh viễn dữ liệu đã convert cũ...")
    for item in legacy_items:
        src = root / item
        if src.exists():
            if src.is_dir():
                shutil.rmtree(str(src))
            else:
                src.unlink()
            print(f"  ✅ Đã xóa: {item}")

    # 2. Xử lý các Raw Data và Models đang nằm la liệt ở root
    print("\n🗂️ Đang phân loại các file/folder còn lại...")
    for item in root.iterdir():
        # Bỏ qua các mục đã quy hoạch hoặc được chỉ định giữ nguyên
        if item.name in ["datasets_converted", "datasets_raw", "models", "nuscenes", "label_cvat_sample"]:
            continue
            
        if item.name.endswith(".pt") or item.name.endswith(".pth"):
            shutil.move(str(item), str(dirs["models"] / item.name))
            print(f"  👉 Move '{item.name}' -> models/")
            
        elif item.is_dir():
            # Những thư mục raw data rải rác đưa hết vào datasets_raw
            shutil.move(str(item), str(dirs["raw"] / item.name))
            print(f"  👉 Move '{item.name}' -> datasets_raw/")

    print("\n" + "=" * 60)
    print("✨ DỌN DẸP HOÀN TẤT! Cấu trúc mới của /dataset/camo_jepa:")
    os.system(f"ls -la {root}")
    print("=" * 60)

if __name__ == "__main__":
    organize_dataset()
