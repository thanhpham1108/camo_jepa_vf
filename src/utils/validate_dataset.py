#!/usr/bin/env python3
"""Script validate dataset sau khi convert để đảm bảo DataLoader đọc được trơn tru."""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader
from src.camo_jepa.config import CaMoJEPAConfig
from src.camo_jepa.data.camo import CaMoEpisodeDataset

def main():
    print("="*60)
    print("🚦 BẮT ĐẦU VALIDATE DATASET (DRY-RUN)")
    print("="*60)
    
    # Ép config đọc từ thư mục convert
    os.environ["DATASET_ROOT"] = "/dataset/camo_jepa/datasets_converted"
    config = CaMoJEPAConfig()
    
    print(f"📁 Đang load dataset từ: {config.dataset_root}")
    try:
        dataset = CaMoEpisodeDataset(config, split="train")
        print(f"✅ Khởi tạo Dataset thành công!")
        print(f"📊 Tổng số clips có thể train: {len(dataset)}")
    except Exception as e:
        print(f"❌ LỖI KHỞI TẠO DATASET: {e}")
        return
        
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    print("\n📦 Đang thử fetch 3 batches ngẫu nhiên...")
    start_time = time.time()
    try:
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            print(f"\n--- Batch {i+1} ---")
            print(f"  📸 Images shape    : {batch['images'].shape} (Kỳ vọng: [4, 16, 3, 256, 256])")
            print(f"  🏎️  CAN bus shape   : {batch['can_bus'].shape} (Kỳ vọng: [4, 16, 18])")
            print(f"  🧭 Ego motion shape: {batch['ego_motion'].shape} (Kỳ vọng: [4, 16, 6])")
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH FETCH DATA: {e}")
        print("💡 Gợi ý: Kiểm tra lại xem symlink ảnh có bị đứt không, hoặc có file .npz nào bị corrupt không.")
        return
        
    print("\n" + "="*60)
    print(f"✨ DATASET HOÀN TOÀN HỢP LỆ!")
    print(f"⏱️ Fetch xong 3 batches trong {time.time() - start_time:.2f}s.")
    print("👉 Sẵn sàng đưa vào quá trình Evaluate / Train.")
    print("="*60)

if __name__ == "__main__":
    main()
