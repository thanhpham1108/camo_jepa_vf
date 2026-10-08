#!/usr/bin/env python3
"""Script soi chi tiết cấu trúc data/ & label/ mới trong /dataset/data260926 
và tạo symlink cầu nối (bridge) để hồi sinh toàn bộ datasets_converted ngay lập tức."""

import os
from pathlib import Path


def main():
    root = Path("/dataset/data260926")

    print("=" * 75)
    print("🔍 KHÁM PHÁ CẤU TRÚC MỚI: /dataset/data260926 (DATA & LABEL)")
    print("=" * 75)

    if not root.exists():
        print(f"❌ Không tìm thấy thư mục {root}!")
        return

    # 1. Liệt kê các thư mục trực tiếp bên trong /dataset/data260926
    print("\n📁 1. CÁC THƯ MỤC CHÍNH BÊN TRONG /dataset/data260926:")
    for item in sorted(root.iterdir()):
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"   ├── 📂 [{item_type}] {item.name}")

    # 2. Khám phá thư mục LABEL
    label_dir = root / "label"
    if not label_dir.exists():
        # Tìm thư mục nào có chữ label
        label_candidates = [d for d in root.iterdir() if "label" in d.name.lower()]
        label_dir = label_candidates[0] if label_candidates else None

    if label_dir and label_dir.exists():
        print(f"\n🏷️ 2. SOI CHI TIẾT THƯ MỤC NHÃN: {label_dir.name}/")
        label_items = sorted(list(label_dir.iterdir()))
        print(f"   * Tổng số mục con trong label: {len(label_items)}")
        print("   * 10 mục con đầu tiên:")
        for item in label_items[:10]:
            print(f"     - {item.name} ({'DIR' if item.is_dir() else f'FILE, {item.stat().st_size} bytes'})")

        # Soi cấu trúc bên trong 1 nhãn mẫu
        sample_label = label_items[0] if label_items else None
        if sample_label:
            if sample_label.is_dir():
                sub_label_files = sorted(list(sample_label.iterdir()))
                print(f"\n   * Soi sâu vào thư mục con '{sample_label.name}': có {len(sub_label_files)} files")
                for sf in sub_label_files[:5]:
                    print(f"       └── {sf.name} ({sf.stat().st_size} bytes)")
            elif sample_label.is_file() and sample_label.stat().st_size < 10000:
                print(f"\n   * Nội dung sơ bộ file nhãn mẫu '{sample_label.name}':")
                try:
                    with open(sample_label, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [f.readline().strip() for _ in range(5)]
                        for l in lines:
                            print(f"       | {l}")
                except Exception as e:
                    print(f"       (Không thể đọc dạng text: {e})")
    else:
        print("\n⚠️ Không tìm thấy thư mục 'label' rõ ràng nào.")

    # 3. Khám phá thư mục DATA
    data_dir = root / "data"
    if data_dir.exists():
        print(f"\n📦 3. SOI CHI TIẾT THƯ MỤC DATA: {data_dir.name}/")
        data_items = sorted(list(data_dir.iterdir()))
        for item in data_items:
            if item.is_dir():
                sub_scenes = len(list(item.iterdir()))
                print(f"   ├── 🚗 Nhóm xe '{item.name}': có {sub_scenes} scenes/chuyến đi")

    # 4. HÀNH ĐỘNG CỨU HỘ: TẠO SYMLINK CẦU NỐI ĐỂ HỒI SINH DATASET
    print("\n" + "=" * 75)
    print("🛠️ 4. TỰ ĐỘNG TẠO SYMLINK CẦU NỐI (BRIDGE) ĐỂ HỒI SINH DATASET")
    print("=" * 75)

    bridge_created = 0
    if data_dir.exists():
        for vehicle_dir in data_dir.iterdir():
            if vehicle_dir.is_dir():
                target_link = root / vehicle_dir.name
                if not target_link.exists() and not target_link.is_symlink():
                    try:
                        target_link.symlink_to(vehicle_dir)
                        print(f"   ✅ Đã tạo bridge: {target_link} -> {vehicle_dir}")
                        bridge_created += 1
                    except Exception as e:
                        print(f"   ❌ Lỗi tạo link cho {vehicle_dir.name}: {e}")
                else:
                    print(f"   ℹ️ Đã tồn tại: {target_link}")

    # 5. Kiểm tra thử nghiệm 1 ảnh sau khi tạo cầu nối
    test_img = Path("/dataset/camo_jepa/datasets_converted/images/train/20260303_1239_VF6_01_1772516366_1772518447_1772517541_1772517558/1772517548-432551158.jpg")
    print(f"\n🧪 KIỂM TRA THỬ NGHIỆM ẢNH CONVERT SAU KHI TẠO CẦU NỐI:")
    print(f"   - File kiểm tra: {test_img.name}")
    print(f"   - Kết quả tồn tại: {'🎉 HỒI SINH THÀNH CÔNG (TỒN TẠI VÀ ĐỌC ĐƯỢC)!' if test_img.exists() else '❌ VẪN LỖI'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
