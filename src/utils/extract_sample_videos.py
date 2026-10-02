#!/usr/bin/env python3
"""Trích xuất các video MP4 mẫu từ các chuyến xe khác nhau vào /workspaces/result để tải về."""

import argparse
import glob
import os
import shutil
from pathlib import Path

def extract_videos(num_videos: int = 10, output_dir: Path | None = None):
    dataset_dir = Path("/dataset/data260926")
    
    # Lấy thư mục từ biến môi trường RUN_DIR (mặc định là /workspaces/result/run_dir)
    if output_dir is None:
        run_dir = Path(os.environ.get("RUN_DIR", "/workspaces/result/run_dir"))
        output_dir = run_dir / "sample_videos"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"🎬 TRÍCH XUẤT {num_videos} VIDEO MP4 MẪU TỪ CÁC CHUYẾN XE KHÁC NHAU")
    print("=" * 65)

    if not dataset_dir.exists():
        print(f"❌ Không tìm thấy thư mục {dataset_dir}")
        return

    # Quét toàn bộ file .mp4
    print("🔍 Đang quét toàn bộ file .mp4 trong dataset...")
    all_mp4s = glob.glob(str(dataset_dir / "**" / "*.mp4"), recursive=True)
    print(f"📊 Tìm thấy tổng cộng: {len(all_mp4s)} file .mp4")

    # Lọc các chuyến xe duy nhất (tránh copy trùng lặp cùng 1 chuyến)
    unique_trips = {}
    for p in all_mp4s:
        filename = os.path.basename(p)
        if filename not in unique_trips:
            unique_trips[filename] = p

    print(f"🚗 Tổng số chuyến xe (trips) duy nhất: {len(unique_trips)}")

    # Lấy N chuyến xe khác nhau
    selected = list(unique_trips.items())[:num_videos]
    print(f"\n📦 Bắt đầu copy {len(selected)} video vào {output_dir}:")

    total_bytes = 0
    for idx, (filename, src_path) in enumerate(selected, 1):
        dst_path = output_dir / f"sample_{idx:02d}_{filename}"
        shutil.copy2(src_path, dst_path)
        size_mb = os.path.getsize(dst_path) / (1024 * 1024)
        total_bytes += os.path.getsize(dst_path)
        print(f"  [{idx:02d}/{len(selected)}] ✅ {dst_path.name} ({size_mb:.1f} MB)")

    print("-" * 65)
    print(f"🎉 HOÀN TẤT! Đã copy {len(selected)} video (Tổng dung lượng: {total_bytes / (1024*1024):.1f} MB).")
    print(f"📁 Vị trí lưu: {output_dir}")
    print("💡 Khi pipeline kết thúc, toàn bộ thư mục này sẽ xuất hiện trên Azure Artifacts để tải về.")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sample MP4 videos")
    parser.add_argument("--num-videos", type=int, default=10, help="Số lượng video muốn trích xuất (mặc định: 10)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Thư mục lưu video (mặc định: $RUN_DIR/sample_videos)")
    args = parser.parse_args()
    extract_videos(args.num_videos, args.output_dir)
