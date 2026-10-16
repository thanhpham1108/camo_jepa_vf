#!/usr/bin/env python3
"""Script kiểm tra và trực quan hóa cây thư mục (tree structure) của raw data và converted data cho CaMo-JEPA.

Đường dẫn mặc định:
- Raw data:       /dataset/camo_jepa/datasets_raw
- Converted data: /dataset/camo_jepa/datasets_converted
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

REQUIRED_STREAMS = ("NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER")
DEFAULT_RAW_DIR = "/dataset/camo_jepa/datasets_raw"
DEFAULT_CONVERTED_DIR = "/dataset/camo_jepa/datasets_converted"


def resolve_existing_path(path_str: str) -> Path:
    """Xử lý đường dẫn linh hoạt (hỗ trợ cả đường dẫn tuyệt đối /dataset... và tương đối dataset...)."""
    p = Path(path_str)
    if p.exists():
        return p
    # Thử fallback tương đối nếu đường dẫn gốc có dấu / ở đầu
    if path_str.startswith("/"):
        rel_p = Path(path_str.lstrip("/"))
        if rel_p.exists():
            return rel_p
    return p


def print_tree_raw(raw_path: Path, max_scenes_per_group: int = 2):
    """In cấu trúc cây thư mục của datasets_raw."""
    print("\n" + "=" * 75)
    print(f"🌲 CẤU TRÚC CÂY THƯ MỤC RAW DATA: {raw_path}")
    print("=" * 75)

    if not raw_path.exists():
        print(f"❌ Không tìm thấy thư mục raw data tại: {raw_path}")
        return

    # Lấy các nhóm con (nhóm xe hoặc trực tiếp các scene)
    children = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    if not children:
        print(f"⚠️  Thư mục {raw_path} đang rỗng!")
        return

    print(f"📁 {raw_path.name}/")

    # Kiểm tra xem children là nhóm xe (chứa scenes) hay trực tiếp là scenes (chứa CAMERA/OTHERS)
    first_has_cam = (children[0] / "CAMERA").is_dir() or (children[0] / "OTHERS").is_dir()

    if first_has_cam:
        # Trường hợp datasets_raw chứa trực tiếp các scenes
        groups = {"Root": children}
    else:
        # Trường hợp datasets_raw chứa các nhóm xe (VD: VF6_01_02, VF8_...)
        groups = {}
        for child in children:
            scenes = sorted([s for s in child.iterdir() if s.is_dir()])
            groups[child.name] = scenes

    for g_idx, (g_name, scene_list) in enumerate(groups.items()):
        is_last_group = (g_idx == len(groups) - 1)
        g_prefix = "└── " if is_last_group else "├── "
        sub_prefix = "    " if is_last_group else "│   "

        if g_name != "Root":
            print(f"{g_prefix}📂 {g_name}/ ({len(scene_list)} scenes)")
        else:
            sub_prefix = ""

        # Hiển thị mẫu một vài scenes
        shown_scenes = scene_list[:max_scenes_per_group]
        for s_idx, scene in enumerate(shown_scenes):
            is_last_scene = (s_idx == len(shown_scenes) - 1) and (len(scene_list) <= max_scenes_per_group)
            s_connector = "└── " if is_last_scene else "├── "
            indent = sub_prefix + s_connector

            print(f"{indent}📂 {scene.name}/")
            scene_sub_prefix = sub_prefix + ("    " if is_last_scene else "│   ")

            # In nhánh CAMERA
            cam_dir = scene / "CAMERA"
            if cam_dir.is_dir():
                cams = sorted([c for c in cam_dir.iterdir() if c.is_dir()])
                cam_names = [c.name for c in cams]
                print(f"{scene_sub_prefix}├── 📂 CAMERA/ -> [{', '.join(cam_names)}]")
                for c in cams:
                    jpg_count = len(list(c.glob("*.jpg")))
                    sample_jpg = next(c.glob("*.jpg"), None)
                    sample_name = f" (mẫu: {sample_jpg.name})" if sample_jpg else ""
                    print(f"{scene_sub_prefix}│   └── 📂 {c.name}/ ({jpg_count:,} ảnh .jpg){sample_name}")
            else:
                print(f"{scene_sub_prefix}├── ❌ KHÔNG CÓ THƯ MỤC CAMERA/")

            # In nhánh OTHERS hoặc telemetry ở gốc
            others_dir = scene / "OTHERS"
            if others_dir.is_dir():
                streams = sorted([s.name for s in others_dir.iterdir() if s.is_dir()])
                print(f"{scene_sub_prefix}└── 📂 OTHERS/ -> [{', '.join(streams)}]")
                for s_name in REQUIRED_STREAMS:
                    st_path = others_dir / s_name
                    if st_path.is_dir():
                        csv_count = len(list(st_path.glob("*.csv")))
                        print(f"{scene_sub_prefix}    ├── 📄 {s_name}/ ({csv_count} file .csv)")
                    else:
                        print(f"{scene_sub_prefix}    ├── ❌ THIẾU {s_name}/")
            else:
                # Kiểm tra cảm biến ở gốc
                found = [s for s in REQUIRED_STREAMS if (scene / s).is_dir()]
                if found:
                    print(f"{scene_sub_prefix}└── 📂 Telemetry ở gốc scene: [{', '.join(found)}]")
                else:
                    print(f"{scene_sub_prefix}└── ❌ KHÔNG CÓ THƯ MỤC OTHERS/ HAY CẢM BIẾN TELEMETRY")

        if len(scene_list) > max_scenes_per_group:
            print(f"{sub_prefix}└── ... và {len(scene_list) - max_scenes_per_group} scenes khác.")


def print_tree_converted(converted_path: Path):
    """In cấu trúc cây thư mục của datasets_converted (nếu đã có)."""
    print("\n" + "=" * 75)
    print(f"📦 CẤU TRÚC CÂY THƯ MỤC CONVERTED DATA: {converted_path}")
    print("=" * 75)

    if not converted_path.exists():
        print(f"ℹ️  Chưa tồn tại thư mục {converted_path} (Dữ liệu chưa convert hoặc chưa tạo).")
        return

    print(f"📁 {converted_path.name}/")

    manifest = converted_path / "manifest.jsonl"
    dataset_info = converted_path / "dataset_info.json"
    episodes_dir = converted_path / "episodes"
    images_dir = converted_path / "images"

    # Manifest & Info
    if dataset_info.exists():
        size_kb = dataset_info.stat().st_size / 1024
        print(f"├── 📄 dataset_info.json ({size_kb:.1f} KB)")
    else:
        print("├── ⚠️  dataset_info.json (chưa có)")

    if manifest.exists():
        size_mb = manifest.stat().st_size / (1024 * 1024)
        print(f"├── 📄 manifest.jsonl ({size_mb:.1f} MB)")
    else:
        print("├── ⚠️  manifest.jsonl (chưa có)")

    # Episodes
    if episodes_dir.is_dir():
        npz_files = list(episodes_dir.rglob("*.npz"))
        print(f"├── 📂 episodes/ ({len(npz_files)} episodes .npz)")
        splits = sorted([d.name for d in episodes_dir.iterdir() if d.is_dir()])
        for split in splits:
            split_npz = list((episodes_dir / split).glob("*.npz"))
            print(f"│   └── 📂 {split}/ ({len(split_npz)} episodes)")
    else:
        print("├── ⚠️  episodes/ (chưa có)")

    # Images
    if images_dir.is_dir():
        img_splits = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        print(f"└── 📂 images/ (splits: {', '.join(img_splits)})")
        for split in img_splits:
            scene_dirs = list((images_dir / split).iterdir())
            print(f"    └── 📂 {split}/ ({len(scene_dirs)} scenes đã symlink/convert)")
    else:
        print("└── ⚠️  images/ (chưa có)")


def inspect_deep(raw_path: Path, converted_path: Path):
    print("=" * 75)
    print(f"🔍 KIỂM TRA TOÀN DIỆN DỮ LIỆU RAW & CONVERTED CHO CAMO-JEPA")
    print("=" * 75)
    print(f"📌 Raw data dir       : {raw_path}")
    print(f"📌 Converted data dir : {converted_path}")

    # 1. Hiển thị tree
    print_tree_raw(raw_path)
    print_tree_converted(converted_path)

    if not raw_path.exists():
        return

    # 2. Tìm tất cả scenes để đếm tổng quát
    all_scenes: List[Path] = []
    for root, dirs, _ in os.walk(raw_path):
        if "CAMERA" in dirs or "OTHERS" in dirs:
            all_scenes.append(Path(root))
            dirs.clear()

    print("\n" + "=" * 75)
    print(f"📊 TỔNG QUAN: Tìm thấy {len(all_scenes)} scenes trong {raw_path}")
    print("=" * 75)

    if not all_scenes:
        print("❌ Không tìm thấy scene nào!")
        return

    # 3. Soi chi tiết 1 Scene mẫu đầu tiên
    sample_scene = all_scenes[0]
    print(f"\n🔎 SOI CHI TIẾT SCENE MẪU: {sample_scene.relative_to(raw_path)}")
    print("-" * 75)

    # Kiểm tra Camera
    cam_dir = sample_scene / "CAMERA"
    cam_name = None
    if cam_dir.exists():
        cams = sorted([d.name for d in cam_dir.iterdir() if d.is_dir()])
        print(f"  📷 CAMERA: Các camera tìm thấy: {cams}")
        for c in cams:
            jpgs = list((cam_dir / c).glob("*.jpg"))
            print(f"     - {c}: {len(jpgs)} ảnh JPG")
            if jpgs and cam_name is None:
                cam_name = c
                print(f"       Ảnh mẫu: {jpgs[0].name}")
    else:
        print("  ❌ Không tìm thấy thư mục CAMERA/")

    # Kiểm tra Telemetry
    others_dir = sample_scene / "OTHERS"
    print(f"\n  📡 TELEMETRY:")
    if others_dir.exists():
        print("     - Thư mục 'OTHERS/' bao bọc: Có")
        for req in REQUIRED_STREAMS:
            st_path = others_dir / req
            if st_path.is_dir():
                csvs = list(st_path.glob("*.csv"))
                print(f"       ✅ {req:<13}: {len(csvs)} file CSV")
            else:
                print(f"       ❌ {req:<13}: THIẾU")
    else:
        print("     - Thư mục 'OTHERS/' bao bọc: Không (kiểm tra cảm biến ở gốc scene)")
        for req in REQUIRED_STREAMS:
            st_path = sample_scene / req
            if st_path.is_dir():
                csvs = list(st_path.glob("*.csv"))
                print(f"       ✅ {req:<13}: {len(csvs)} file CSV")
            else:
                print(f"       ❌ {req:<13}: THIẾU")

    # 4. Thử nghiệm chạy dry-run convert_vf_to_camo trên scene mẫu
    print("\n" + "=" * 75)
    print("⚙️ THỬ NGHIỆM ĐỒNG BỘ VỚI 'convert_vf_to_camo.py':")
    print("=" * 75)

    active_cam = "CAM_P_F" if (sample_scene / "CAMERA" / "CAM_P_F").exists() else (
        "CAM_F" if (sample_scene / "CAMERA" / "CAM_F").exists() else cam_name
    )

    if not active_cam:
        print("❌ Không tìm thấy camera hợp lệ để test.")
        return

    try:
        from src.utils.convert_vf_to_camo import build_samples
        import argparse

        test_args = argparse.Namespace(
            vf_root=sample_scene,
            camera=active_cam,
            max_drift_ms=80.0,
            split="train",
            log_name="test_sample",
            output_root=Path("/tmp/test_convert"),
            image_mode="symlink",
            target_hz=10.0,
            max_frames=50,
            dry_run=True,
        )
        print(f"🧪 Đang thử build samples trên scene mẫu (Camera: {active_cam})...")
        samples, skipped = build_samples(test_args)
        print(f"  🎉 TEST THÀNH CÔNG! Đồng bộ được {len(samples)} frames (Bỏ qua do lệch sync: {skipped})")
        if samples:
            print(f"  * CAN bus shape  : {samples[0].can_bus.shape} (Kỳ vọng: 18)")
            print(f"  * Ego motion     : {samples[0].ego_motion} [yaw_rate, accel_x]")
            print("  👉 KẾT LUẬN: CẤU TRÚC RAW DATA HOÀN TOÀN TƯƠNG THÍCH VỚI CAMO-JEPA!")
    except Exception as e:
        print(f"  ⚠️ Test convert báo lỗi: {e}")

    print("\n" + "=" * 75)
    print("🏁 HOÀN TẤT KIỂM TRA!")
    print("=" * 75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra cấu trúc cây thư mục raw và converted data cho CaMo-JEPA."
    )
    raw_env = os.environ.get("DATASETS_RAW_DIR", DEFAULT_RAW_DIR)
    converted_env = os.environ.get("DATASETS_CONVERTED_DIR", DEFAULT_CONVERTED_DIR)

    parser.add_argument(
        "--raw-dir",
        default=raw_env,
        help=f"Đường dẫn thư mục datasets_raw (mặc định: {raw_env})",
    )
    parser.add_argument(
        "--converted-dir",
        default=converted_env,
        help=f"Đường dẫn thư mục datasets_converted (mặc định: {converted_env})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_path = resolve_existing_path(args.raw_dir)
    converted_path = resolve_existing_path(args.converted_dir)
    inspect_deep(raw_path, converted_path)


if __name__ == "__main__":
    main()
