#!/usr/bin/env python3
"""Script tách 20% scenes của data260926 từ split='train' → split='val'.

Chiến lược:
- Chỉ lấy scenes thuộc nguồn /dataset/data260926 để tách (datasets_raw 100% vào train).
- Stratified random split: 20% val, giữ tỷ lệ VF6_01_02 và VF6_03.
- Di chuyển file .npz trong episodes/train/ → episodes/val/
  (symlink ảnh trong images/train/ → images/val/ tương ứng).
- Idempotent: nếu chạy lại thì không làm hỏng gì.
- Sau khi chạy: episodes/train/  và  episodes/val/ đều sẵn sàng cho DataLoader.

Chạy:
    python3 -m src.utils.create_eval_split
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List


DEFAULT_CONVERTED_DIR = os.environ.get(
    "DATASETS_CONVERTED_DIR", "/dataset/camo_jepa/datasets_converted_interp"
)
DEFAULT_NEW_DATA_MARKER = "260926"  # scene nào đến từ data260926 thì npz có field source_raw chứa chuỗi này
EVAL_RATIO = 0.20
SEED = 42


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if path_str.startswith("/"):
        rel = Path(path_str.lstrip("/"))
        if rel.exists():
            return rel
    return p


def read_episode_source(npz_path: Path) -> str:
    """Đọc field source_raw trong .npz để biết episode đến từ thư mục nào.
    Nếu không có field source_raw (episode cũ), fallback tìm theo tên.
    """
    try:
        import numpy as np
        with np.load(npz_path, allow_pickle=False) as data:
            if "source_raw" in data.files:
                return str(data["source_raw"])
    except Exception:
        pass
    return ""


def get_episodes_by_group(
    episodes_train_dir: Path,
    converted_dir: Path,
    manifest_path: Path,
) -> Dict[str, List[Path]]:
    """Phân nhóm episode theo nguồn raw gốc.

    Returns:
        {
            "data260926/VF6_01_02": [Path, ...],
            "data260926/VF6_03": [Path, ...],
            "datasets_raw": [Path, ...],   <- không được tách ra eval
        }
    """
    npz_files = sorted(episodes_train_dir.glob("*.npz"))
    groups: Dict[str, List[Path]] = {
        "data260926/VF6_01_02": [],
        "data260926/VF6_03": [],
        "datasets_raw": [],
    }

    # Đọc manifest.jsonl để lấy source_raw nếu có
    source_map: Dict[str, str] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    ep_name = Path(row.get("episode", "")).stem
                    src = row.get("source_raw", "")
                    if ep_name and src:
                        source_map[ep_name] = src
        except Exception:
            pass

    for npz in npz_files:
        ep_name = npz.stem
        src = source_map.get(ep_name, "")

        if not src:
            src = read_episode_source(npz)

        # Fallback 2: Kiểm tra symlink ảnh thực tế trỏ về folder raw nào
        if not src:
            ep_img_dir = converted_dir / "images" / "train" / ep_name
            if ep_img_dir.exists():
                try:
                    for img in ep_img_dir.glob("*.jpg"):
                        if img.is_symlink():
                            src = str(img.resolve())
                            break
                except Exception:
                    pass

        # Phân loại
        if "data260926" in src or "VF6_01_02" in src or "VF6_03" in src:
            if "VF6_03" in src:
                groups["data260926/VF6_03"].append(npz)
            else:
                groups["data260926/VF6_01_02"].append(npz)
        elif "VF6_01_02" in ep_name:
            groups["data260926/VF6_01_02"].append(npz)
        elif "VF6_03" in ep_name:
            groups["data260926/VF6_03"].append(npz)
        else:
            groups["datasets_raw"].append(npz)

    return groups


def move_episode(
    npz_path: Path,
    episodes_train_dir: Path,
    episodes_val_dir: Path,
    images_train_dir: Path,
    images_val_dir: Path,
) -> bool:
    """Move .npz và thư mục ảnh tương ứng từ train → val."""
    ep_name = npz_path.stem

    # Di chuyển .npz
    val_npz = episodes_val_dir / npz_path.name
    if not val_npz.exists():
        shutil.move(str(npz_path), str(val_npz))

    # Di chuyển thư mục ảnh (symlink thư mục)
    img_src = images_train_dir / ep_name
    img_dst = images_val_dir / ep_name
    if img_src.exists() and not img_dst.exists():
        shutil.move(str(img_src), str(img_dst))

    return True


def create_eval_split(
    converted_dir_str: str,
    eval_ratio: float = EVAL_RATIO,
    seed: int = SEED,
    dry_run: bool = False,
) -> None:
    converted_dir = resolve_path(converted_dir_str)
    episodes_train_dir = converted_dir / "episodes" / "train"
    episodes_val_dir = converted_dir / "episodes" / "val"
    images_train_dir = converted_dir / "images" / "train"
    images_val_dir = converted_dir / "images" / "val"
    manifest_path = converted_dir / "manifest.jsonl"

    print("=" * 75)
    print("✂️  TẠO TRAIN/EVAL SPLIT CHO CAMO-JEPA")
    print("=" * 75)
    print(f"📁 Converted dir : {converted_dir}")
    print(f"📊 Eval ratio    : {eval_ratio * 100:.0f}%  (chỉ lấy từ data260926)")
    print(f"🎲 Random seed   : {seed}")
    print(f"{'🔍 DRY RUN — không thực sự move file' if dry_run else ''}")
    print("=" * 75)

    if not episodes_train_dir.exists():
        print(f"❌ Không tìm thấy {episodes_train_dir}. Hãy chạy run_batch_conversion.py trước!")
        return

    # Kiểm tra xem đã tách chưa
    val_existing = list(episodes_val_dir.glob("*.npz")) if episodes_val_dir.exists() else []
    if val_existing:
        print(f"ℹ️  Đã tồn tại {len(val_existing)} episodes trong episodes/val/")
        print("   Script sẽ chỉ tách thêm nếu còn episodes chưa được tách.")

    episodes_val_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)

    # Phân nhóm episodes
    print("\n🔍 Đang phân nhóm episodes theo nguồn raw...")
    groups = get_episodes_by_group(episodes_train_dir, converted_dir, manifest_path)

    total_new = sum(len(v) for k, v in groups.items() if "data260926" in k)
    total_raw = len(groups["datasets_raw"])
    total_all = total_new + total_raw

    print(f"  📦 Tổng episodes/train hiện tại : {total_all}")
    print(f"  └─ Từ data260926/VF6_01_02    : {len(groups['data260926/VF6_01_02'])}")
    print(f"  └─ Từ data260926/VF6_03       : {len(groups['data260926/VF6_03'])}")
    print(f"  └─ Từ datasets_raw (→ train 100%): {total_raw}")

    # Stratified sample 20% từ mỗi group của data260926
    rng = random.Random(seed)
    eval_episodes: List[Path] = []

    for group_key in ("data260926/VF6_01_02", "data260926/VF6_03"):
        pool = groups[group_key]

        # Lọc ra những episode chưa ở val (idempotent)
        pool_remaining = [p for p in pool if not (episodes_val_dir / p.name).exists()]
        n_total_group = len(pool)
        n_val_target = max(1, round(n_total_group * eval_ratio))
        n_already_val = n_total_group - len(pool_remaining)
        n_need = max(0, n_val_target - n_already_val)

        sampled = rng.sample(pool_remaining, min(n_need, len(pool_remaining)))
        eval_episodes.extend(sampled)
        print(f"\n  [{group_key}]: {n_total_group} episodes → target val={n_val_target}")
        print(f"    Đã có trong val : {n_already_val}")
        print(f"    Sẽ di chuyển thêm: {len(sampled)}")

    n_eval = len(eval_episodes)
    print(f"\n📦 Tổng số episodes sẽ di chuyển sang val: {n_eval}")

    if n_eval == 0:
        print("✅ Không cần tách thêm. Split đã hoàn chỉnh.")
    else:
        if dry_run:
            print("\n🔍 [DRY RUN] Các episodes sẽ được move sang val:")
            for ep in eval_episodes[:10]:
                print(f"   {ep.name}")
            if len(eval_episodes) > 10:
                print(f"   ... và {len(eval_episodes) - 10} episodes nữa")
        else:
            print("\n⚙️  Đang di chuyển episodes sang val/...")
            success = 0
            for ep_npz in eval_episodes:
                ok = move_episode(ep_npz, episodes_train_dir, episodes_val_dir, images_train_dir, images_val_dir)
                if ok:
                    success += 1
            print(f"✅ Đã di chuyển {success}/{n_eval} episodes sang episodes/val/")

    # Đếm kết quả cuối
    final_train = len(list(episodes_train_dir.glob("*.npz")))
    final_val = len(list(episodes_val_dir.glob("*.npz")))
    print("\n" + "=" * 75)
    print("🏁 KẾT QUẢ CUỐI CÙNG:")
    print(f"   📘 episodes/train : {final_train} episodes (DataLoader dùng split='train')")
    print(f"   📗 episodes/val   : {final_val} episodes  (DataLoader dùng split='val')")
    if final_train + final_val > 0:
        actual_eval_pct = final_val / (final_train + final_val) * 100
        print(f"   📊 Tỷ lệ eval    : {actual_eval_pct:.1f}%")
    print("=" * 75)
    print("✅ Sẵn sàng training! Cập nhật DATASET_SPLIT=train hoặc val trong extraEnv.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tách 20% data260926 → episodes/val/ cho CaMo-JEPA")
    parser.add_argument(
        "--converted-dir",
        default=os.environ.get("DATASETS_CONVERTED_DIR", DEFAULT_CONVERTED_DIR),
        help=f"Thư mục datasets_converted (mặc định: {DEFAULT_CONVERTED_DIR})",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=EVAL_RATIO,
        help=f"Tỷ lệ eval (mặc định: {EVAL_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (mặc định: {SEED})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy thử, không thực sự move file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_eval_split(
        converted_dir_str=args.converted_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )
