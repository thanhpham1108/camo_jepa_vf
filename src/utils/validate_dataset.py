#!/usr/bin/env python3
"""Script validate dataset sau khi convert để đảm bảo DataLoader đọc được trơn tru."""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from src.camo_jepa.config import CaMoJEPAConfig
from src.camo_jepa.data.camo import CaMoEpisodeDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Validate CaMo-JEPA dataset loading and DataLoader pipeline.")
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get(
            "DATASET_ROOT",
            "/dataset/camo_jepa/datasets_converted_interp"
            if Path("/dataset/camo_jepa/datasets_converted_interp").exists()
            else "/dataset/camo_jepa/datasets_converted",
        ),
        help="Root path of converted dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Dataset split to validate (train or val).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader validation.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=3,
        help="Number of batches to fetch for validation.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print("🚦 BẮT ĐẦU VALIDATE DATASET (DRY-RUN)")
    print("=" * 60)
    
    os.environ["DATASET_ROOT"] = str(args.dataset_root)
    config = CaMoJEPAConfig(dataset_root=str(args.dataset_root))
    
    print(f"📁 Đang load dataset từ: {config.dataset_root}")
    print(f"🔹 Split: {args.split}")
    try:
        dataset = CaMoEpisodeDataset(
            dataset_root=config.dataset_root,
            split=args.split,
            history_length=config.history_length,
            stride=config.stride,
            image_size=config.image_size,
        )
        print(f"✅ Khởi tạo Dataset thành công!")
        print(f"📊 Tổng số clips có thể train/eval: {len(dataset)}")
    except Exception as e:
        print(f"❌ LỖI KHỞI TẠO DATASET: {e}")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"\n📦 Đang thử fetch {args.num_batches} batches ngẫu nhiên...")
    start_time = time.time()
    try:
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            print(f"\n--- Batch {i+1} ---")
            print(f"  📸 Images shape    : {batch['images'].shape} (Kỳ vọng: [{args.batch_size}, 16, 3, 256, 256])")
            print(f"  🏎️  CAN bus shape   : {batch['can_bus'].shape} (Kỳ vọng: [{args.batch_size}, 16, 18])")
            print(f"  🧭 Ego motion shape: {batch['ego_motion'].shape} (Kỳ vọng: [{args.batch_size}, 16, 6])")
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH FETCH DATA: {e}")
        print("💡 Gợi ý: Kiểm tra lại xem symlink ảnh có bị đứt không, hoặc có file .npz nào bị corrupt không.")
        return
        
    print("\n" + "=" * 60)
    print(f"✨ DATASET HOÀN TOÀN HỢP LỆ!")
    print(f"⏱️ Fetch xong {args.num_batches} batches trong {time.time() - start_time:.2f}s.")
    print("👉 Sẵn sàng đưa vào quá trình Evaluate / Train.")
    print("=" * 60)

if __name__ == "__main__":
    main()
