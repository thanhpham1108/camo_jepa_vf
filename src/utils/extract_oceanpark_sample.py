#!/usr/bin/env python3
"""Script tìm và trích xuất file video summary từ data_260924 và rawdata_260924."""

import os
import shutil
from pathlib import Path

def main():
    target_dirs = [
        Path("/dataset/camo_jepa/data_260924"),
        Path("/dataset/camo_jepa/rawdata_260924")
    ]
    
    # Lấy thư mục RUN_DIR từ môi trường (do pipeline set), nếu không có thì lấy mặc định
    run_dir = Path(os.environ.get("RUN_DIR", "/workspaces/result/run_dir"))
    out_dir = run_dir / "oceanpark_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    found_videos = []
    
    print("=" * 60)
    print("🔍 TÌM KIẾM VIDEO SUMMARY TRONG data_260924 & rawdata_260924")
    print("=" * 60)
    
    for d in target_dirs:
        if not d.exists():
            print(f"❌ Không tìm thấy {d}")
            continue
            
        print(f"\nĐang quét thư mục: {d}...")
        # Tìm các file mp4
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith(".mp4") and "summary" in file.lower():
                    src_path = Path(root) / file
                    
                    # Đổi tên file đích để không bị trùng (vd: data_260924_Default_oceanpark1_summary.mp4)
                    rel_path = src_path.relative_to(d)
                    safe_name = f"{d.name}_{str(rel_path).replace('/', '_')}"
                    dst_path = out_dir / safe_name
                    
                    print(f"  📸 Đã tìm thấy: {src_path.name}")
                    print(f"     -> Copying to: {dst_path.name}")
                    
                    try:
                        shutil.copy2(str(src_path), str(dst_path))
                        found_videos.append(dst_path)
                    except Exception as e:
                        print(f"     ❌ Lỗi copy: {e}")
                    
                    # Giới hạn copy 2-4 file để tránh bị quá tải pipeline
                    if len(found_videos) >= 4:
                        print("\n👉 Đã copy đủ số lượng mẫu (4 files), dừng tìm kiếm để tiết kiệm thời gian.")
                        break
            if len(found_videos) >= 4:
                break

    if not found_videos:
        print("\n⚠️ Không tìm thấy file summary *.mp4 nào trong các thư mục trên!")
    else:
        print(f"\n✅ Đã copy {len(found_videos)} video ra thư mục: {out_dir}")
        print("💡 Pipeline sẽ tự động nén thư mục này và trả về cho bạn trong Artifact (EvaluationResults).")

if __name__ == "__main__":
    main()
