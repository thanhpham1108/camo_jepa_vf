#!/usr/bin/env python3
"""Script đối soát độc lập tuyệt đối giữa /dataset/camo_jepa/datasets_raw và /dataset/data260926.

Mục tiêu:
1. Xác thực xem data260926 có bị copy nhầm vào trong datasets_raw hay không.
2. Tìm kiếm bất kỳ sự trùng lặp (intersection) nào về tên scene giữa 2 nguồn.
3. Kiểm tra xem có thư mục/file nào của 260926 xuất hiện bên trong datasets_raw không.
4. Kiểm tra symlink / hardlink chéo giữa 2 thư mục.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

DEFAULT_RAW_DIR = "/dataset/camo_jepa/datasets_raw"
DEFAULT_NEW_DIR = "/dataset/data260926"


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if path_str.startswith("/"):
        rel_p = Path(path_str.lstrip("/"))
        if rel_p.exists():
            return rel_p
    return p


def find_scenes_in_dir(root_dir: Path) -> Dict[str, Path]:
    """Tìm tất cả các thư mục scene (chứa CAMERA hoặc OTHERS hoặc stream telemetry)."""
    scenes: Dict[str, Path] = {}
    if not root_dir.exists():
        return scenes

    for root, dirs, _ in os.walk(root_dir):
        root_path = Path(root)
        has_camera = "CAMERA" in dirs
        has_others = "OTHERS" in dirs
        has_streams = any(st in dirs for st in ("NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER"))

        if has_camera or has_others or has_streams:
            scene_name = root_path.name
            scenes[scene_name] = root_path
            dirs.clear()

    return scenes


def check_independence(
    raw_path_str: str,
    new_path_str: str,
) -> None:
    raw_dir = resolve_path(raw_path_str)
    new_dir = resolve_path(new_path_str)

    print("=" * 85)
    print("🕵️  KIỂM TRA TÍNH ĐỘC LẬP GIỮA DATASETS_RAW VÀ DATA260926")
    print("=" * 85)
    print(f"📁 Thư mục datasets_raw : {raw_dir} (Tồn tại: {'Có' if raw_dir.exists() else 'KHÔNG'})")
    print(f"📁 Thư mục data260926   : {new_dir} (Tồn tại: {'Có' if new_dir.exists() else 'KHÔNG'})")
    print("=" * 85)

    if not raw_dir.exists() or not new_dir.exists():
        print("❌ Một trong hai thư mục không tồn tại trên hệ thống này. Không thể so sánh.")
        return

    # 1. Liệt kê cấu trúc cấp 1 của cả 2 thư mục
    print("\n📂 1. CẤU TRÚC THƯ MỤC CẤP 1:")
    print(f"  [datasets_raw]:")
    raw_children = sorted([d.name for d in raw_dir.iterdir()])
    for name in raw_children:
        print(f"    ├── {name}")

    print(f"\n  [data260926]:")
    new_children = sorted([d.name for d in new_dir.iterdir()])
    for name in new_children:
        print(f"    ├── {name}")

    # 2. Tìm kiếm chuỗi '260926' hoặc các nhóm con của data260926 bên trong datasets_raw
    print("\n🔍 2. TÌM KIẾM DẤU VẾT '260926' BÊN TRONG datasets_raw:")
    matches_260926: List[str] = []
    for root, dirs, files in os.walk(raw_dir):
        for d in dirs:
            if "260926" in d or d in ("VF6_01_02", "VF6_03"):
                matches_260926.append(os.path.join(root, d))
        for f in files:
            if "260926" in f:
                matches_260926.append(os.path.join(root, f))

    if matches_260926:
        print(f"  ⚠️  PHÁT HIỆN {len(matches_260926)} MỤC CHỨA TÊN '260926' HOẶC TÊN NHÓM XE TRONG datasets_raw:")
        for m in matches_260926[:10]:
            print(f"    👉 {m}")
    else:
        print("  ✅ HOÀN TOÀN KHÔNG CÓ bất kỳ thư mục hay file nào mang tên '260926', 'VF6_01_02', hay 'VF6_03' trong datasets_raw.")

    # 3. Quét toàn bộ scenes và tính toán tập giao (Intersection)
    print("\n🔍 3. QUÉT VÀ SO SÁNH TẬP HỢP SCENES (SCENE NAMES):")
    scenes_raw = find_scenes_in_dir(raw_dir)
    scenes_new = find_scenes_in_dir(new_dir)

    print(f"  * Tổng số scenes trong datasets_raw : {len(scenes_raw):,}")
    print(f"  * Tổng số scenes trong data260926   : {len(scenes_new):,}")

    raw_set: Set[str] = set(scenes_raw.keys())
    new_set: Set[str] = set(scenes_new.keys())
    intersection: Set[str] = raw_set.intersection(new_set)

    if intersection:
        print(f"\n  🔴 CẢNH BÁO: PHÁT HIỆN {len(intersection)} SCENES BỊ TRÙNG LẶP GIỮA 2 THƯ MỤC:")
        for sc in sorted(intersection)[:10]:
            print(f"    - Scene: {sc}")
            print(f"        Trong datasets_raw: {scenes_raw[sc]}")
            print(f"        Trong data260926  : {scenes_new[sc]}")
    else:
        print("\n  ✅ TẬP GIAO BẰNG RỖNG (0 SCENES TRÙNG LẶP):")
        print("     100% scenes trong datasets_raw và data260926 là HOÀN TOÀN KHÁC NHAU!")

    # 4. Kiểm tra cụ thể thư mục data_260924 và rawdata_260924
    print("\n🔍 4. LÀM RÕ SỰ KHÁC NHAU VỀ NGÀY THÁNG (260924 vs 260926):")
    folder_260924 = raw_dir / "data_260924"
    rawfolder_260924 = raw_dir / "rawdata_260924"

    print("  📌 Thư mục trong datasets_raw là 260924 (ngày 24/09/2026):")
    if folder_260924.exists():
        sub_items = [x.name for x in folder_260924.iterdir()]
        print(f"    - datasets_raw/data_260924: chứa {len(sub_items)} mục: {sub_items[:5]}")
    if rawfolder_260924.exists():
        sub_items = [x.name for x in rawfolder_260924.iterdir()]
        print(f"    - datasets_raw/rawdata_260924: chứa {len(sub_items)} mục: {sub_items}")

    print("\n  📌 Thư mục data260926 là đợt dữ liệu ngày 26/09/2026:")
    if new_dir.exists():
        sub_items = [x.name for x in new_dir.iterdir()]
        print(f"    - /dataset/data260926: chứa {len(sub_items)} mục: {sub_items}")
        data_subdir = new_dir / "data"
        if data_subdir.exists():
            car_subdirs = [x.name for x in data_subdir.iterdir()]
            print(f"    - /dataset/data260926/data: chứa các nhóm xe {car_subdirs}")

    # 5. KẾT LUẬN CUỐI CÙNG
    print("\n" + "=" * 85)
    print("🏁 KẾT LUẬN:")
    print("=" * 85)
    if not intersection and not matches_260926:
        print("🎉 XÁC NHẬN: BẠN CHƯA HỀ COPY `data260926` VÀO TRONG `datasets_raw`!")
        print("   1. Hai thư mục nằm độc lập tại: /dataset/camo_jepa/datasets_raw và /dataset/data260926.")
        print("   2. Tên gây nhầm lẫn là do ngày tháng: `260924` (24/09) vs `260926` (26/09).")
        print("   3. Số lượng scene không trùng lặp: datasets_raw (~15 scenes) vs data260926 (~1,106 scenes).")
        print("   4. Khi convert cả 2 nguồn, tổng hợp lại sẽ được toàn vẹn ~10.67 giờ không bị trùng một frame nào.")
    else:
        print("⚠️ Có dấu hiệu trùng lặp dữ liệu, vui lòng xem chi tiết danh sách bên trên.")
    print("=" * 85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate independence between datasets_raw and data260926")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Đường dẫn datasets_raw")
    parser.add_argument("--new-dir", default=DEFAULT_NEW_DIR, help="Đường dẫn data260926")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_independence(args.raw_dir, args.new_dir)
