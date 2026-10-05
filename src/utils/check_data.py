#!/usr/bin/env python3
"""Script kiểm tra nhanh nội dung của data_260924 và rawdata_260924 trên server."""

import os
from pathlib import Path

def peek_directory(path_str, max_items=20):
    p = Path(path_str)
    print("=" * 60)
    print(f"👀 Đang kiểm tra: {p}")
    print("=" * 60)
    
    if not p.exists():
        print("❌ Thư mục không tồn tại!")
        return
        
    items = list(p.iterdir())
    print(f"📊 Tổng số mục con (trực tiếp): {len(items)}")
    
    # In ra danh sách tối đa `max_items`
    for item in items[:max_items]:
        type_str = "📁 DIR " if item.is_dir() else "📄 FILE"
        size_str = ""
        if item.is_file():
            size_str = f" ({item.stat().st_size / (1024*1024):.2f} MB)"
        print(f"  ├── {type_str}: {item.name}{size_str}")
        
    if len(items) > max_items:
        print(f"  └── ... và {len(items) - max_items} mục khác.")
        
    # Thử check xem bên trong có giống cấu trúc đã convert không
    if (p / "episodes").exists() and (p / "images").exists():
        print("\n👉 NHẬN XÉT: Thư mục này có cấu trúc giống hệt một bộ data ĐÃ CONVERT (có episodes/, images/).")
    elif (p / "CAMERA").exists() or any(d.is_dir() and (d/"CAMERA").exists() for d in items[:5]):
        print("\n👉 NHẬN XÉT: Thư mục này có vẻ chứa RAW DATA (có nhánh CAMERA/).")
    print("\n")

if __name__ == "__main__":
    peek_directory("/dataset/data260926")
    peek_directory("/dataset/camo_jepa/datasets_raw")
