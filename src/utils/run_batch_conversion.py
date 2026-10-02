#!/usr/bin/env python3
"""Script quét toàn bộ data thô và chạy batch conversion ra định dạng CaMo-JEPA dataset."""

import os
import subprocess
from pathlib import Path

def find_scenes(base_paths):
    scenes = []
    for base in base_paths:
        base = Path(base)
        if not base.exists():
            continue
        
        # Tìm tất cả các thư mục có chứa thư mục con là "CAMERA" và "OTHERS"
        for root, dirs, files in os.walk(base):
            if "CAMERA" in dirs and "OTHERS" in dirs:
                scenes.append(Path(root))
    return scenes

def get_camera_name(scene_path):
    # Kiểm tra xem có CAM_P_F hay CAM_F
    cam_dir = scene_path / "CAMERA"
    if (cam_dir / "CAM_P_F").exists():
        return "CAM_P_F"
    elif (cam_dir / "CAM_F").exists():
        return "CAM_F"
    return None

def main():
    source_dirs = [
        "/dataset/data260926",
        "/dataset/camo_jepa/datasets_raw"
    ]
    
    output_root = Path("/dataset/camo_jepa/datasets_converted")
    
    print("=" * 60)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH BATCH CONVERSION DỮ LIỆU")
    print("=" * 60)
    
    # 1. Khởi tạo / Dọn dẹp manifest cũ nếu có để chạy mới 100%
    manifest_path = output_root / "manifest.jsonl"
    if manifest_path.exists():
        print("🗑️ Đang xóa manifest.jsonl cũ để bắt đầu bản convert mới hoàn toàn...")
        manifest_path.unlink()
        
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 2. Tìm tất cả các scenes hợp lệ
    print("\n🔍 Đang quét tìm các scenes...")
    scenes = find_scenes(source_dirs)
    print(f"✅ Tìm thấy tổng cộng {len(scenes)} scenes hợp lệ.")
    
    if not scenes:
        print("❌ Không tìm thấy scenes nào có đủ nhánh CAMERA/ và OTHERS/. Thoát.")
        return
        
    # 3. Lặp qua từng scene và gọi script convert
    success_count = 0
    fail_count = 0
    
    convert_script = Path(__file__).parent / "convert_vf_to_camo.py"
    
    print("\n⚙️ Bắt đầu Convert (Có thể sẽ tốn một khoảng thời gian dài)...")
    for idx, scene in enumerate(scenes, 1):
        scene_name = scene.name
        cam_name = get_camera_name(scene)
        
        if not cam_name:
            print(f"[{idx}/{len(scenes)}] ⚠️ Bỏ qua {scene_name} (Không tìm thấy CAM_P_F hoặc CAM_F)")
            fail_count += 1
            continue
            
        print(f"[{idx}/{len(scenes)}] 🔄 Đang xử lý: {scene_name} (Camera: {cam_name})")
        
        cmd = [
            "python3", str(convert_script),
            "--vf-root", str(scene),
            "--camera", cam_name,
            "--log-name", scene_name,
            "--output-root", str(output_root),
            "--image-mode", "symlink"
        ]
        
        try:
            # Chạy subprocess
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                success_count += 1
            else:
                # Chỉ in ra dòng lỗi từ stderr nếu có vấn đề
                error_msg = result.stderr.strip() or result.stdout.strip()
                print(f"    ❌ Lỗi convert: {error_msg.split(chr(10))[-1]}")
                fail_count += 1
        except Exception as e:
            print(f"    ❌ Lỗi hệ thống: {e}")
            fail_count += 1
            
    print("\n" + "=" * 60)
    print(f"🏁 TỔNG KẾT BATCH CONVERSION")
    print(f"  * Thành công: {success_count} scenes")
    print(f"  * Thất bại/Bỏ qua: {fail_count} scenes")
    print(f"👉 Dữ liệu đầu ra được lưu trọn gói tại: {output_root}")
    print("=" * 60)

if __name__ == "__main__":
    main()
