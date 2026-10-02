#!/usr/bin/env python3
"""Script kiểm tra chi tiết cấu trúc các scene bên trong /dataset/data260926."""

import os
import glob
from pathlib import Path

def inspect_deep():
    target_path = Path("/dataset/data260926")

    print("=" * 65)
    print("🔍 SOI CHI TIẾT CẤU TRÚC SCENE DỮ LIỆU MỚI: /dataset/data260926")
    print("=" * 65)

    if not target_path.exists():
        print(f"❌ Không tìm thấy {target_path}")
        return

    # 1. Liệt kê toàn bộ các scene folders
    vehicle_groups = [d for d in target_path.iterdir() if d.is_dir()]
    print(f"\n🚗 Các nhóm xe: {[vg.name for vg in vehicle_groups]}")

    all_scenes = []
    for vg in vehicle_groups:
        scenes = sorted([s for s in vg.iterdir() if s.is_dir()])
        print(f"  * {vg.name}: có {len(scenes)} scenes/clips")
        all_scenes.extend(scenes)

    print(f"\n📊 TỔNG CỘNG TOÀN BỘ DATASET: {len(all_scenes)} scenes/clips!")

    if not all_scenes:
        print("❌ Không tìm thấy scene nào!")
        return

    # 2. Soi cực kỳ chi tiết 1 Scene mẫu đầu tiên
    sample_scene = all_scenes[0]
    print("\n" + "=" * 65)
    print(f"🔎 SOI CHI TIẾT SCENE MẪU: {sample_scene.parent.name}/{sample_scene.name}")
    print("=" * 65)

    try:
        sub_items = sorted(os.listdir(sample_scene))
        print(f"📁 Các mục trực tiếp bên trong scene:")
        for item in sub_items:
            ip = sample_scene / item
            if ip.is_dir():
                print(f"  ├── 📂 {item}/")
            else:
                size_kb = ip.stat().st_size / 1024
                print(f"  ├── 📄 {item} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"❌ Lỗi đọc scene: {e}")
        return

    # 3. Kiểm tra nhánh CAMERA
    print("\n📷 1. KIỂM TRA NHÁNH CAMERA:")
    camera_dir = sample_scene / "CAMERA"
    if camera_dir.exists():
        cams = sorted(os.listdir(camera_dir))
        print(f"  * Tìm thấy thư mục CAMERA, các camera hiện có: {cams}")
        for cam in cams:
            cam_path = camera_dir / cam
            if cam_path.is_dir():
                jpgs = list(cam_path.glob("*.jpg"))
                print(f"    - {cam}: {len(jpgs)} ảnh JPG")
                if jpgs:
                    print(f"      Mẫu tên file: {jpgs[0].name}")
    else:
        print("  ❌ KHÔNG thấy thư mục CAMERA trực tiếp!")

    # 4. Kiểm tra nhánh CẢM BIẾN / TELEMETRY
    print("\n📡 2. KIỂM TRA NHÁNH CẢM BIẾN (CAN BUS & TELEMETRY):")
    others_dir = sample_scene / "OTHERS"
    if others_dir.exists():
        print("  * Tìm thấy thư mục 'OTHERS/' bao bọc.")
        streams = sorted(os.listdir(others_dir))
        for st in streams:
            st_path = others_dir / st
            csvs = list(st_path.glob("*.csv")) if st_path.is_dir() else []
            print(f"    - {st}: {len(csvs)} file CSV")
    else:
        print("  ⚠️ Không có thư mục 'OTHERS/' bao bọc. Kiểm tra cảm biến ở thư mục gốc:")
        required = ["NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER"]
        for req in required:
            req_path = sample_scene / req
            if req_path.exists():
                csvs = list(req_path.glob("*.csv")) if req_path.is_dir() else []
                print(f"    - Tìm thấy {req}/ ở gốc scene: {len(csvs)} file CSV")
            else:
                print(f"    - ❌ THIẾU {req}/")

    # 5. Đánh giá tính tương thích với convert_vf_to_camo.py
    print("\n" + "=" * 65)
    print("⚙️ 3. KIỂM TRA TƯƠNG THÍCH VỚI 'convert_vf_to_camo.py':")
    print("=" * 65)
    has_others = others_dir.exists()
    has_cam_p_f = (sample_scene / "CAMERA" / "CAM_P_F").exists() or (sample_scene / "CAMERA" / "CAM_F").exists()
    
    print(f"  * Chuẩn đường dẫn OTHERS/: {'✅ ĐẠT' if has_others else '⚠️ CẦN TẠO SYMLINK HOẶC SỬA SCRIPT ĐỌC TRỰC TIẾP'}")
    print(f"  * Camera phía trước: {'✅ ĐẠT' if has_cam_p_f else '❌ CẦN XÁC ĐỊNH TÊN CAMERA CHÍNH'}")

    # Thử chạy dry-run trên scene này nếu script tồn tại
    try:
        from src.utils.convert_vf_to_camo import build_samples
        import argparse
        
        vf_root_to_test = sample_scene
        cam_name = "CAM_P_F" if (sample_scene / "CAMERA" / "CAM_P_F").exists() else "CAM_F"
        
        args = argparse.Namespace(
            vf_root=vf_root_to_test,
            camera=cam_name,
            max_drift_ms=80.0,
            split="train",
            log_name="test_sample",
            output_root=Path("/tmp/test_convert"),
            image_mode="symlink",
            target_hz=10.0,
            max_frames=50,
            dry_run=True
        )
        print(f"\n🧪 Thử nghiệm build samples trên scene mẫu (Camera: {cam_name})...")
        samples, skipped = build_samples(args)
        print(f"  🎉 THỬ NGHIỆM THÀNH CÔNG! Đã khớp được {len(samples)} frames (Bỏ qua do lệch sync: {skipped})")
        print(f"  * Mẫu CAN bus vector 18 chiều: shape = {samples[0].can_bus.shape}")
        print(f"  * Mẫu Ego motion [yaw_rate, accel_x]: {samples[0].ego_motion}")
        print("  👉 KẾT LUẬN: SCRIPT CONVERT HOÀN TOÀN CHẠY ĐƯỢC NGAY TRÊN DỮ LIỆU NÀY!")
    except Exception as e:
        print(f"  ⚠️ Thử nghiệm convert báo lỗi: {e}")
        print("  (Đừng lo, có thể do lệch đường dẫn OTHERS hoặc tên camera, ta sẽ tinh chỉnh script convert cho khớp)")

    print("\n" + "=" * 65)
    print("🏁 HOÀN TẤT PHÂN TÍCH CHI TIẾT")
    print("=" * 65)

if __name__ == "__main__":
    inspect_deep()
