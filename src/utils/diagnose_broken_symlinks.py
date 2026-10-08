#!/usr/bin/env python3
"""Script điều tra toàn diện: Liệt kê chính xác raw data còn lại trên server và tìm kiếm VF6_01_02."""

import os
import subprocess
from pathlib import Path


def run_cmd(cmd: str):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Error running '{cmd}': {e}"


def main():
    print("=" * 75)
    print("🔍 ĐIỀU TRA TOÀN DIỆN THỰC TRẠNG DATA TRÊN SERVER (/dataset)")
    print("=" * 75)

    # 1. Liệt kê toàn bộ thư mục cấp 1 bên trong /dataset
    print("\n📁 1. CÁC THƯ MỤC TRỰC TIẾP TRONG /dataset:")
    print(run_cmd("ls -la /dataset"))

    # 2. Liệt kê chi tiết bên trong /dataset/data260926
    print("\n📁 2. CHI TIẾT BÊN TRONG /dataset/data260926:")
    if Path("/dataset/data260926").exists():
        print(run_cmd("ls -la /dataset/data260926"))
        
        # Nếu có thư mục con, soi tiếp 1 cấp nữa
        for item in sorted(Path("/dataset/data260926").iterdir()):
            if item.is_dir():
                sub_count = len(list(item.iterdir()))
                print(f"   └── 📂 {item.name}: có {sub_count} mục con")
    else:
        print("❌ /dataset/data260926 không tồn tại!")

    # 3. Liệt kê chi tiết bên trong /dataset/camo_jepa/datasets_raw
    print("\n📁 3. CHI TIẾT BÊN TRONG /dataset/camo_jepa/datasets_raw:")
    if Path("/dataset/camo_jepa/datasets_raw").exists():
        print(run_cmd("ls -la /dataset/camo_jepa/datasets_raw"))
    else:
        print("❌ /dataset/camo_jepa/datasets_raw không tồn tại!")

    # 4. Tìm kiếm từ khóa 'VF6_01' hoặc '1772517548' trên toàn bộ /dataset
    print("\n🔎 4. TÌM KIẾM DẤU VẾT 'VF6_01' TRÊN TOÀN BỘ CỤM /dataset:")
    find_vf6 = run_cmd("find /dataset -maxdepth 3 -name '*VF6_01*' 2>/dev/null")
    if find_vf6:
        print("   Tìm thấy các đường dẫn sau có tên VF6_01:")
        for line in find_vf6.splitlines():
            print(f"     👉 {line}")
    else:
        print("   ❌ KHÔNG TÌM THẤY bất kỳ thư mục/file nào có tên 'VF6_01' trong /dataset (độ sâu 3 cấp)!")

    # 5. Thử tìm kiếm 1 file ảnh cụ thể xem nó có bị move đi đâu không
    sample_img = "1772517548-432551158.jpg"
    print(f"\n🔎 5. TÌM KIẾM FILE ẢNH MẪU ({sample_img}) TRÊN TOÀN BỘ /dataset:")
    find_img = run_cmd(f"find /dataset -name '{sample_img}' 2>/dev/null")
    if find_img:
        print(f"   Tìm thấy ảnh mẫu tại:")
        for line in find_img.splitlines():
            is_link = Path(line).is_symlink()
            link_target = f" -> {os.readlink(line)}" if is_link else " (FILE THẬT)"
            print(f"     👉 {line}{link_target}")
    else:
        print(f"   ❌ KHÔNG TÌM THẤY file ảnh thật '{sample_img}' ở bất kỳ ngóc ngách nào trên /dataset!")

    print("\n" + "=" * 75)
    print("🏁 KẾT THÚC ĐIỀU TRA")
    print("=" * 75)


if __name__ == "__main__":
    main()
