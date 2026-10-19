#!/usr/bin/env python3
"""Script quét toàn bộ data thô và chạy batch conversion ra định dạng CaMo-JEPA dataset.

Mặc định quét CẢ 2 nguồn raw:
  - /dataset/camo_jepa/datasets_raw
  - /dataset/data260926

Tất cả scenes convert vào split='train' trước.
Sau đó chạy create_eval_split.py để tách 20% data260926 sang split='val'.
"""

import os
import subprocess
from pathlib import Path

DEFAULT_SOURCE_DIRS = [
    "/dataset/camo_jepa/datasets_raw",
    "/dataset/data260926",
]


def find_scenes(base_paths: list) -> list:
    """Tìm tất cả thư mục scene (có CAMERA + OTHERS) trong các thư mục nguồn."""
    scenes = []
    for base in base_paths:
        base = Path(base)
        if not base.exists() and str(base).startswith("/"):
            rel_base = Path(str(base).lstrip("/"))
            if rel_base.exists():
                base = rel_base
        if not base.exists():
            print(f"⚠️  Bỏ qua (đường dẫn không tồn tại): {base}")
            continue

        for root, dirs, _ in os.walk(base):
            if "CAMERA" in dirs and "OTHERS" in dirs:
                scenes.append(Path(root))
    return scenes


def get_camera_name(scene_path: Path) -> str | None:
    """Ưu tiên CAM_P_F, fallback CAM_F."""
    cam_dir = scene_path / "CAMERA"
    if (cam_dir / "CAM_P_F").exists():
        return "CAM_P_F"
    elif (cam_dir / "CAM_F").exists():
        return "CAM_F"
    return None


def main() -> None:
    env_sources = os.environ.get("DATASETS_RAW_DIRS") or os.environ.get("DATASETS_RAW_DIR")
    source_dirs = [s.strip() for s in env_sources.split(",")] if env_sources else DEFAULT_SOURCE_DIRS

    output_root = Path(os.environ.get("DATASETS_CONVERTED_DIR", "/dataset/camo_jepa/datasets_converted"))
    sync_mode = os.environ.get("SYNC_MODE", "interpolation")
    max_interp_gap_ms = os.environ.get("MAX_INTERP_GAP_MS", "300.0")

    print("=" * 70)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH BATCH CONVERSION DỮ LIỆU")
    print("=" * 70)
    print(f"📌 Nguồn raw data : {source_dirs}")
    print(f"📌 Thư mục output : {output_root}")
    print(f"📌 Chế độ sync    : {sync_mode} (max_interp_gap={max_interp_gap_ms} ms)")
    print("=" * 70)

    output_root.mkdir(parents=True, exist_ok=True)
    episodes_train_dir = output_root / "episodes" / "train"
    episodes_train_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tìm tất cả scenes hợp lệ từ cả 2 nguồn
    print("\n🔍 Đang quét tìm các scenes từ tất cả nguồn raw...")
    scenes = find_scenes(source_dirs)
    print(f"✅ Tìm thấy tổng cộng {len(scenes)} scenes hợp lệ.")

    if not scenes:
        print("❌ Không tìm thấy scenes nào có đủ nhánh CAMERA/ và OTHERS/. Thoát.")
        return

    # 2. Convert từng scene — skip những scene đã có .npz (idempotent)
    success_count = 0
    skip_count = 0
    fail_count = 0

    convert_script = Path(__file__).parent / "convert_vf_to_camo.py"

    print("\n⚙️  Bắt đầu Convert — tất cả vào split='train'...")
    print("   (Sau khi xong, chạy create_eval_split.py để tách 20% data260926 → val)\n")

    for idx, scene in enumerate(scenes, 1):
        scene_name = scene.name

        # Idempotent: skip nếu .npz đã tồn tại
        if (episodes_train_dir / f"{scene_name}.npz").exists():
            print(f"[{idx}/{len(scenes)}] ⏭️  Đã có, bỏ qua: {scene_name}")
            skip_count += 1
            continue

        cam_name = get_camera_name(scene)
        if not cam_name:
            print(f"[{idx}/{len(scenes)}] ⚠️  Bỏ qua {scene_name} (Không có CAM_P_F hoặc CAM_F)")
            fail_count += 1
            continue

        print(f"[{idx}/{len(scenes)}] 🔄 Converting: {scene_name}  (cam={cam_name})")

        cmd = [
            "python3", str(convert_script),
            "--vf-root", str(scene),
            "--camera", cam_name,
            "--log-name", scene_name,
            "--output-root", str(output_root),
            "--split", "train",
            "--image-mode", "symlink",
            "--sync-mode", sync_mode,
            "--max-interp-gap-ms", max_interp_gap_ms,
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                success_count += 1
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                print(f"    ❌ Lỗi: {error_msg.split(chr(10))[-1]}")
                fail_count += 1
        except Exception as e:
            print(f"    ❌ Lỗi hệ thống: {e}")
            fail_count += 1

    print("\n" + "=" * 70)
    print("🏁 TỔNG KẾT BATCH CONVERSION")
    print(f"  ✅ Thành công mới      : {success_count} scenes")
    print(f"  ⏭️  Đã có, bỏ qua       : {skip_count} scenes")
    print(f"  ❌ Thất bại/Bỏ qua     : {fail_count} scenes")
    print(f"  📦 Tổng episodes/train : {success_count + skip_count} scenes")
    print(f"\n👉 Bước tiếp theo:")
    print(f"   python3 -m src.utils.create_eval_split")
    print("=" * 70)


if __name__ == "__main__":
    main()
