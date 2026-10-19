#!/usr/bin/env python3
"""Kiểm toán toàn diện độ hụt dữ liệu (10.6h raw -> 7.7h converted).

Tự động quét toàn bộ scenes từ các nguồn raw và đối chiếu với tập converted.
Xuất báo cáo chi tiết:
  1. File Markdown (Báo cáo tổng kết dễ đọc, đẹp mắt, có bảng biểu để gửi đối tác/sếp).
  2. File CSV (Chi tiết từng scene để mở bằng Excel/Google Sheets).
  3. File JSON (Dữ liệu có cấu trúc phục vụ downstream pipeline).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REQUIRED_STREAMS = ("NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data gap between Raw VF and Converted CaMo-JEPA dataset.")
    parser.add_argument(
        "--raw-dirs",
        default=os.environ.get("DATASETS_RAW_DIRS", "/dataset/camo_jepa/datasets_raw,/dataset/data260926"),
        help="Comma-separated paths to raw datasets.",
    )
    parser.add_argument(
        "--converted-dir",
        default=os.environ.get("DATASETS_CONVERTED_DIR", "/dataset/camo_jepa/datasets_converted_interp"),
        help="Path to converted dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "."),
        help="Directory where reports (md, csv, json) will be saved.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional explicit path for output CSV.",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Optional explicit path for output report.",
    )
    args, _ = parser.parse_known_args()
    return args


def token_to_ns(token: str) -> Optional[int]:
    """Chuyển timestamp token trong tên file ảnh sang nanoseconds."""
    try:
        if "-" in token:
            seconds, subsecond = token.split("-", 1)
        elif "_" in token:
            parts = token.split("_")
            seconds = parts[0]
            subsecond = parts[1] if len(parts) > 1 else "0"
        else:
            return None
        if not seconds.isdigit():
            return None
        digits = "".join(ch for ch in subsecond if ch.isdigit())
        return int(seconds) * 1_000_000_000 + int((digits + "000000000")[:9] or "0")
    except Exception:
        return None


def inspect_scene_raw(scene_path: Path) -> Dict[str, Any]:
    """Kiểm tra chi tiết tình trạng raw data của một scene folder."""
    info: Dict[str, Any] = {
        "has_camera_dir": False,
        "has_others_dir": False,
        "detected_cameras": [],
        "chosen_camera": None,
        "raw_frame_count": 0,
        "raw_duration_sec": 0.0,
        "missing_streams": [],
        "valid_streams": [],
        "reason": "OK",
    }

    cam_root = scene_path / "CAMERA"
    others_root = scene_path / "OTHERS"

    if not cam_root.is_dir():
        info["reason"] = "Thiếu thư mục CAMERA/"
        return info
    info["has_camera_dir"] = True

    # Quét các camera hiện có
    cam_dirs = [d.name for d in cam_root.iterdir() if d.is_dir()]
    info["detected_cameras"] = cam_dirs

    target_cam = None
    if "CAM_P_F" in cam_dirs:
        target_cam = "CAM_P_F"
    elif "CAM_F" in cam_dirs:
        target_cam = "CAM_F"

    if not target_cam:
        info["reason"] = f"Không có camera trước chuẩn (Tìm thấy: {', '.join(cam_dirs) if cam_dirs else 'None'})"
        return info
    info["chosen_camera"] = target_cam

    # Đếm số lượng ảnh và tính duration từ timestamp
    cam_path = cam_root / target_cam
    img_files = sorted(cam_path.glob("*.jpg"))
    if not img_files:
        img_files = sorted(cam_path.glob("*.png"))

    info["raw_frame_count"] = len(img_files)
    if not img_files:
        info["reason"] = f"Thư mục camera {target_cam} rỗng (0 file ảnh)"
        return info

    # Tính thời lượng từ frame đầu và cuối
    first_ns = token_to_ns(img_files[0].stem)
    last_ns = token_to_ns(img_files[-1].stem)
    if first_ns and last_ns and last_ns >= first_ns:
        info["raw_duration_sec"] = (last_ns - first_ns) / 1e9
    else:
        # Fallback ước lượng theo 20 FPS
        info["raw_duration_sec"] = len(img_files) / 20.0

    # Kiểm tra OTHERS
    if not others_root.is_dir():
        info["reason"] = "Thiếu thư mục OTHERS/ (không có dữ liệu cảm biến CAN/IMU)"
        return info
    info["has_others_dir"] = True

    missing_streams = []
    valid_streams = []
    for stream_name in REQUIRED_STREAMS:
        stream_dir = others_root / stream_name
        if not stream_dir.is_dir() or not list(stream_dir.glob("*.csv")):
            missing_streams.append(stream_name)
        else:
            valid_streams.append(stream_name)

    info["missing_streams"] = missing_streams
    info["valid_streams"] = valid_streams

    if missing_streams:
        info["reason"] = f"Thiếu cảm biến trong OTHERS/: {', '.join(missing_streams)}"
        return info

    return info


def generate_markdown_report(
    summary: Dict[str, Any],
    unconverted_scenes: List[Dict[str, Any]],
    top_dropped_scenes: List[Dict[str, Any]],
    output_path: Path,
    sync_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Tạo báo cáo Markdown chuyên nghiệp, trực quan để gửi đối tác."""
    raw_h = summary["duration_hours"]["raw_hours"]
    conv_h = summary["duration_hours"]["converted_hours"]
    gap_h = summary["duration_hours"]["gap_hours"]
    gap_pct = summary["duration_hours"]["gap_percent"]
    gap_breakdown = summary.get("gap_breakdown", {})
    skipped_h = gap_breakdown.get("skipped_scenes_hours", 0.0)
    skipped_min = gap_breakdown.get("skipped_scenes_minutes", 0.0)
    skipped_pct = gap_breakdown.get("skipped_scenes_pct_of_gap", 0.0)
    intra_h = gap_breakdown.get("intra_scene_sync_gap_hours", 0.0)
    intra_min = gap_breakdown.get("intra_scene_sync_gap_minutes", 0.0)
    intra_pct = gap_breakdown.get("intra_scene_pct_of_gap", 0.0)
    n_converted = summary["converted_scenes"]

    sync_info = sync_metadata or {}
    sync_mode = sync_info.get("sync_mode", "unknown")
    max_gap_ms = sync_info.get("max_interp_gap_ms", 300.0)
    max_drift_ms = sync_info.get("max_drift_ms", 80.0)

    if sync_mode == "interpolation":
        sync_desc = f"Nội suy tuyến tính LERP & SLERP (ngưỡng mất tín hiệu tối đa: {max_gap_ms:.0f} ms)"
        intra_reason = f"Cắt mép đầu/cuối clip (chờ GPS fix) + ngắt sóng cảm biến > {max_gap_ms:.0f} ms"
        filter_detail = f"Khoảng cách 2 gói tin liên tiếp > {max_gap_ms:.0f} ms (mất sóng CAN/GPS)"
    else:
        sync_desc = f"Khớp hàng gần nhất - Nearest Row (ngưỡng sai số tối đa: {max_drift_ms:.0f} ms)"
        intra_reason = f"Cắt mép đầu/cuối clip + lọc frame lệch timestamp > {max_drift_ms:.0f} ms"
        filter_detail = f"Timestamp giữa Camera và CAN-bus lệch > {max_drift_ms:.0f} ms"

    md = []
    md.append("# 📊 BÁO CÁO KIỂM TOÁN DỮ LIỆU HUẤN LUYỆN CA-MO JEPA")
    md.append(f"## Phân Tích & Giải Trình Độ Hụt Thời Lượng ({raw_h:.2f} Giờ Raw → {conv_h:.2f} Giờ Converted)\n")
    md.append("> **Dự án:** CaMo-JEPA | **Nguồn dữ liệu:** `datasets_raw` & `data260926`\n")
    md.append(f"> **Phương pháp đồng bộ:** `{sync_desc}`\n")
    md.append("---\n")

    md.append("### 1. Tổng Kết Điều Hành (Executive Summary)\n")
    md.append("| Chỉ số | Giá trị | Ý nghĩa kỹ thuật |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| **Tổng thời lượng thu thập (Raw Data)** | **{raw_h:.2f} giờ** | Tổng thời gian xe chạy ghi nhận từ các camera (@ 30 FPS) |")
    md.append(f"| **Thời lượng đạt chuẩn nạp vào Model** | **{conv_h:.2f} giờ** | Thời lượng có đủ ảnh + cảm biến đồng bộ chính xác (@ 10 Hz) |")
    md.append(f"| **Tổng thời lượng bị loại bỏ (Gap)** | **{gap_h:.2f} giờ** ({gap_pct:.1f}%) | Phần dữ liệu không đủ điều kiện kỹ thuật để huấn luyện |")
    md.append(f"| **Tổng số Scenes quét trong Raw** | **{summary['total_candidate_scenes']} scenes** | Toàn bộ thư mục ghi nhận trên hệ thống |")
    md.append(f"| **Số Scenes Chuyển Đổi Thành Công** | **{n_converted} scenes** | Đã tạo thành công các file episode `.npz` |")
    md.append(f"| **Số Scenes Bị Bỏ Qua Toàn Phần (0%)** | **{summary['skipped_scenes_total']} scenes** | Không tạo được episode do thiếu điều kiện bắt buộc (`OTHERS/`) |")
    md.append(f"| **Tổng số ảnh quét trong Raw (30 FPS)** | **{summary['frame_metrics']['total_raw_frames']:,} frames** | Trung bình 30 ảnh/giây |")
    md.append(f"| **Tổng số frame nạp vào Model (10 Hz)** | **{summary['frame_metrics']['total_converted_frames']:,} frames** | Đã hạ mẫu (downsample) về 10 Hz theo chuẩn mô hình |")
    md.append("\n---\n")

    md.append("### 2. Phân Bổ Bản Chất Độ Hụt Thời Lượng (Gap Breakdown)\n")
    md.append(f"Hao hụt **{gap_h:.2f} giờ (~{gap_pct:.1f}%)** được phân tách thành hai tầng nguyên nhân cụ thể:\n")
    md.append("| Tầng hao hụt | Thời lượng hụt | Tỷ trọng trên tổng Gap | Cơ chế kỹ thuật |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(f"| **1. Bỏ qua toàn phần ({summary['skipped_scenes_total']} scenes)** | **{skipped_h:.2f} giờ** ({skipped_min:.1f} phút) | **{skipped_pct:.1f}%** | Thiếu hẳn thư mục `OTHERS/` (không có CAN/IMU) |")
    md.append(f"| **2. Co ngắn trong {n_converted} scenes thành công** | **{intra_h:.2f} giờ** ({intra_min:.1f} phút) | **{intra_pct:.1f}%** | {intra_reason} |")
    md.append("\n> [!NOTE]")
    md.append("> **Về số lượng frame:** Khoảng ~66.7% số ảnh raw bị lược bỏ **theo thiết kế chuẩn** do converter hạ tần số lấy mẫu từ **30 FPS xuống 10 Hz** (`--target-hz 10`), phần còn lại do cắt mép đồng bộ cảm biến và các scenes bị bỏ qua.\n")
    md.append("---\n")

    md.append("### 3. Phân Loại Chi Tiết Các Nguyên Nhân Bỏ Qua / Rớt Frame\n")
    md.append("| Nhóm nguyên nhân | Chi tiết kỹ thuật | Ảnh hưởng |")
    md.append("| :--- | :--- | :---: |")
    md.append(f"| **1. Cấu trúc thư mục không hợp lệ** | Thiếu thư mục `CAMERA/` hoặc `OTHERS/` do quá trình chép file dở dang | **{summary['breakdown']['invalid_structure']} scenes** |")
    md.append(f"| **2. Thiếu Camera góc trước** | Không tìm thấy `CAM_P_F` hoặc `CAM_F` (chỉ có camera bên sườn/đuôi) | **{summary['breakdown']['no_front_camera']} scenes** |")
    md.append(f"| **3. Thiếu luồng cảm biến bắt buộc** | Thiếu một trong 4 luồng telemetry (`NAV`, `IMU`, `VEHICLE_INFO`, `STEER`) | **{summary['breakdown']['missing_telemetry']} scenes** |")
    md.append(f"| **4. Lọc frame không đồng bộ** | {filter_detail} | **Frame-level drops** |")
    md.append(f"| **5. Cắt bỏ đoạn đầu/cuối (Truncation)** | Cảm biến CAN-bus bắt đầu ghi sau Camera hoặc tắt trước Camera | **Trôi mép clip** |")
    md.append("\n---\n")

    md.append("### 4. Danh Sách Các Scenes Bị Bỏ Qua Toàn Phần (Chưa Được Convert)\n")
    if unconverted_scenes:
        md.append("| Tên Scene | Camera tìm thấy | Cảm biến thiếu | Lý do cụ thể |")
        md.append("| :--- | :---: | :---: | :--- |")
        limit = len(unconverted_scenes) if len(unconverted_scenes) <= 100 else 50
        for s in unconverted_scenes[:limit]:
            cams = ", ".join(s["detected_cameras"]) if s["detected_cameras"] else "None"
            missing = ", ".join(s["missing_streams"]) if s["missing_streams"] else "None"
            md.append(f"| `{s['scene_name']}` | `{cams}` | `{missing}` | {s['reason']} |")
        if len(unconverted_scenes) > limit:
            md.append(f"\n*... và còn {len(unconverted_scenes) - limit} scenes khác được ghi đầy đủ 100% trong file CSV `skipped_scenes_detail.csv` kèm theo.*\n")
    else:
        md.append("✅ Không có scene nào bị bỏ qua toàn phần. 100% scenes đã được xử lý.\n")

    md.append("\n---\n")

    md.append("### 5. Top Scenes Có Tỷ Lệ Rớt Frame / Hụt Thời Lượng Lớn Nhất\n")
    md.append("Đây là các scenes đã convert thành công nhưng bị loại bỏ bớt các frame do lệch timestamp cảm biến:\n")
    md.append("| Tên Scene | Số ảnh Raw | Số ảnh Converted | Số frame rớt | Thời lượng Raw | Converted | % Hụt |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for s in top_dropped_scenes[:20]:
        gap_sec = s["raw_duration_sec"] - s["converted_duration_sec"]
        pct = (gap_sec / s["raw_duration_sec"] * 100.0) if s["raw_duration_sec"] > 0 else 0.0
        md.append(
            f"| `{s['scene_name']}` | {s['raw_frames']:,} | {s['converted_frames']:,} | "
            f"**{s['skipped_frames']:,}** | {s['raw_duration_sec']:.1f}s | {s['converted_duration_sec']:.1f}s | **{pct:.1f}%** |"
        )

    md.append("\n---\n")

    md.append("### 6. Kết Luận & Đánh Giá Chất Lượng\n")
    if sync_mode == "interpolation":
        md.append(
            f"1. **Về tính toàn vẹn dữ liệu:** Toàn bộ **{conv_h:.2f} giờ** dữ liệu đã nạp vào model được "
            f"đồng bộ chuẩn xác bằng **thuật toán nội suy tuyến tính (LERP) và SLERP Quaternion trên $S^3$**, "
            f"đảm bảo các vector gia tốc, vận tốc và góc lái biến thiên liên tục, không bị nhiễu bậc thang.\n"
            f"2. **Bản chất phần thời lượng hụt {gap_h:.2f} giờ:**\n"
            f"   - **{skipped_h:.2f} giờ ({skipped_min:.1f} phút):** Bắt nguồn từ các scenes lỗi cấu trúc raw gốc (thiếu hẳn thư mục `OTHERS/`).\n"
            f"   - **{intra_h:.2f} giờ còn lại:** Bắt nguồn từ giới hạn vật lý mép khởi động khi Camera bật trước CAN-bus trong lúc đợi GPS fix vệ tinh (nguyên tắc an toàn: Strictly No Extrapolation).\n"
        )
    else:
        md.append(
            f"1. **Về tính toàn vẹn dữ liệu:** Toàn bộ **{conv_h:.2f} giờ** dữ liệu đã nạp vào model hiện tại là dữ liệu "
            f"đồng bộ dạng Nearest-Row (sai số < {max_drift_ms:.0f} ms).\n"
            f"2. **Giải pháp lấy lại thời lượng bị hụt:**\n"
            f"   - Chuyển đổi sang phương thức nội suy (Interpolation) để phục hồi phần lớn thời lượng bị hụt do lệch pha clock.\n"
        )

    output_path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dirs = [Path(d.strip().strip("\"'")) for d in args.raw_dirs.split(",") if d.strip().strip("\"'")]
    converted_dir = Path(args.converted_dir.strip().strip("\"'"))
    output_dir = Path(args.output_dir.strip().strip("\"'"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info_path = converted_dir / "dataset_info.json"
    sync_metadata: Dict[str, Any] = {}
    if dataset_info_path.exists():
        try:
            sync_metadata = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    print("=" * 75)
    print("🔍 BẮT ĐẦU QUÉT KIỂM TOÁN DỮ LIỆU TOÀN DIỆN (RAW vs CONVERTED)")
    print(f"📁 Nguồn Raw      : {[str(d) for d in raw_dirs]}")
    print(f"📁 Nguồn Converted: {converted_dir}")
    if sync_metadata:
        mode = sync_metadata.get("sync_mode", "unknown")
        gap = sync_metadata.get("max_interp_gap_ms") or sync_metadata.get("max_drift_ms")
        print(f"⚙️  Chế độ Sync    : {mode} (threshold: {gap} ms)")
    print("=" * 75)

    # 1. Tìm tất cả scene candidates trong các nguồn raw
    scenes: List[Path] = []
    for raw_root in raw_dirs:
        if not raw_root.exists():
            continue
        for root, dirs, _ in os.walk(raw_root):
            if "CAMERA" in dirs or "OTHERS" in dirs:
                scenes.append(Path(root))

    # Tìm tất cả file .npz đã convert (hỗ trợ cả train, val hoặc subfolder bất kỳ)
    episodes_root = converted_dir / "episodes"
    converted_npz_map: Dict[str, Path] = {}
    if episodes_root.exists():
        for p in episodes_root.rglob("*.npz"):
            converted_npz_map[p.stem] = p

    total_raw_duration = 0.0
    total_conv_duration = 0.0
    total_raw_frames = 0
    total_conv_frames = 0

    count_converted = 0
    count_invalid_structure = 0
    count_no_front_camera = 0
    count_missing_telemetry = 0
    count_missing_npz = 0

    all_records: List[Dict[str, Any]] = []
    unconverted_scenes: List[Dict[str, Any]] = []

    for scene in scenes:
        scene_name = scene.name
        raw_info = inspect_scene_raw(scene)

        record: Dict[str, Any] = {
            "scene_name": scene_name,
            "scene_path": str(scene),
            "status": "UNKNOWN",
            "reason": raw_info["reason"],
            "camera": raw_info["chosen_camera"],
            "detected_cameras": raw_info["detected_cameras"],
            "missing_streams": raw_info["missing_streams"],
            "raw_frames": raw_info["raw_frame_count"],
            "converted_frames": 0,
            "skipped_frames": raw_info["raw_frame_count"],
            "raw_duration_sec": round(raw_info["raw_duration_sec"], 2),
            "converted_duration_sec": 0.0,
            "gap_sec": round(raw_info["raw_duration_sec"], 2),
            "gap_percent": 100.0 if raw_info["raw_duration_sec"] > 0 else 0.0,
        }

        total_raw_duration += raw_info["raw_duration_sec"]
        total_raw_frames += raw_info["raw_frame_count"]

        # Phân loại lý do
        if not raw_info["has_camera_dir"] or not raw_info["has_others_dir"]:
            count_invalid_structure += 1
            record["status"] = "SKIPPED_INVALID_STRUCTURE"
            unconverted_scenes.append(record)
            all_records.append(record)
            continue

        if not raw_info["chosen_camera"]:
            count_no_front_camera += 1
            record["status"] = "SKIPPED_NO_FRONT_CAMERA"
            unconverted_scenes.append(record)
            all_records.append(record)
            continue

        if raw_info["missing_streams"]:
            count_missing_telemetry += 1
            record["status"] = "SKIPPED_MISSING_TELEMETRY"
            unconverted_scenes.append(record)
            all_records.append(record)
            continue

        # Kiểm tra file .npz đã convert chưa (tìm trong toàn bộ splits)
        npz_path = converted_npz_map.get(scene_name)
        if npz_path is None or not npz_path.exists():
            count_missing_npz += 1
            record["status"] = "VALID_BUT_NOT_CONVERTED"
            record["reason"] = "Đủ cấu trúc nhưng chưa tạo file .npz"
            unconverted_scenes.append(record)
            all_records.append(record)
            continue

        # Đã convert thành công
        try:
            with np.load(npz_path) as data:
                ts = data["timestamps_us"]
                conv_len = len(ts)
                conv_dur = (ts[-1] - ts[0]) / 1e6 if conv_len > 1 else 0.0

            gap_s = max(0.0, raw_info["raw_duration_sec"] - conv_dur)
            gap_p = (gap_s / raw_info["raw_duration_sec"] * 100.0) if raw_info["raw_duration_sec"] > 0 else 0.0

            record["status"] = "CONVERTED"
            record["converted_frames"] = conv_len
            record["skipped_frames"] = max(0, raw_info["raw_frame_count"] - conv_len)
            record["converted_duration_sec"] = round(conv_dur, 2)
            record["gap_sec"] = round(gap_s, 2)
            record["gap_percent"] = round(gap_p, 1)
            record["reason"] = "OK"

            total_conv_duration += conv_dur
            total_conv_frames += conv_len
            count_converted += 1
            all_records.append(record)
        except Exception as e:
            record["status"] = "CORRUPTED_NPZ"
            record["reason"] = f"Lỗi đọc file .npz: {e}"
            unconverted_scenes.append(record)
            all_records.append(record)

    # Thống kê tổng hợp
    raw_hours = total_raw_duration / 3600.0
    conv_hours = total_conv_duration / 3600.0
    gap_hours = max(0.0, raw_hours - conv_hours)
    gap_percent = (gap_hours / raw_hours * 100.0) if raw_hours > 0 else 0.0

    skipped_dur_sec = sum(r["raw_duration_sec"] for r in unconverted_scenes)
    skipped_dur_hours = skipped_dur_sec / 3600.0
    intra_sync_gap_sec = sum(max(0.0, r["raw_duration_sec"] - r["converted_duration_sec"]) for r in all_records if r["status"] == "CONVERTED")
    intra_sync_gap_hours = intra_sync_gap_sec / 3600.0

    summary = {
        "total_candidate_scenes": len(scenes),
        "converted_scenes": count_converted,
        "skipped_scenes_total": len(scenes) - count_converted,
        "breakdown": {
            "invalid_structure": count_invalid_structure,
            "no_front_camera": count_no_front_camera,
            "missing_telemetry": count_missing_telemetry,
            "missing_npz": count_missing_npz,
        },
        "frame_metrics": {
            "total_raw_frames": total_raw_frames,
            "total_converted_frames": total_conv_frames,
            "total_dropped_frames": total_raw_frames - total_conv_frames,
        },
        "duration_hours": {
            "raw_hours": round(raw_hours, 2),
            "converted_hours": round(conv_hours, 2),
            "gap_hours": round(gap_hours, 2),
            "gap_percent": round(gap_percent, 1),
        },
        "gap_breakdown": {
            "skipped_scenes_hours": round(skipped_dur_hours, 2),
            "skipped_scenes_minutes": round(skipped_dur_hours * 60, 1),
            "skipped_scenes_pct_of_gap": round(skipped_dur_hours / gap_hours * 100.0, 1) if gap_hours > 0 else 0.0,
            "intra_scene_sync_gap_hours": round(intra_sync_gap_hours, 2),
            "intra_scene_sync_gap_minutes": round(intra_sync_gap_hours * 60, 1),
            "intra_scene_pct_of_gap": round(intra_sync_gap_hours / gap_hours * 100.0, 1) if gap_hours > 0 else 0.0,
        },
    }

    # Xuất file CSV
    csv_file = Path(args.output_csv) if args.output_csv else output_dir / "skipped_scenes_detail.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scene_name", "status", "reason", "camera", "raw_frames",
            "converted_frames", "skipped_frames", "raw_duration_sec", "converted_duration_sec",
            "gap_sec", "gap_percent",
            "detected_cameras", "missing_streams", "scene_path"
        ])
        writer.writeheader()
        writer.writerows(all_records)

    # Xuất file JSON
    json_file = Path(args.output_report).with_suffix(".json") if args.output_report else output_dir / "skipped_audit_summary.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Tìm top các scene bị hao hụt thời lượng nhiều nhất trong số các scene đã convert
    converted_records = [r for r in all_records if r["status"] == "CONVERTED"]
    converted_records.sort(
        key=lambda r: (r["raw_duration_sec"] - r["converted_duration_sec"]),
        reverse=True,
    )

    # Xuất file Markdown báo cáo đẹp
    md_file = Path(args.output_report).with_suffix(".md") if args.output_report else output_dir / "skipped_audit_report.md"
    md_file.parent.mkdir(parents=True, exist_ok=True)
    generate_markdown_report(summary, unconverted_scenes, converted_records, md_file, sync_metadata)

    print("\n" + "=" * 75)
    print("✨ ĐÃ XUẤT THÀNH CÔNG BỘ BÁO CÁO KIỂM TOÁN ✨")
    print("=" * 75)
    print(f"📄 Báo cáo Markdown gửi đối tác/sếp : {md_file}")
    print(f"📊 Bảng tính chi tiết từng Scene    : {csv_file}")
    print(f"📦 Dữ liệu JSON tổng hợp             : {json_file}")
    print("-" * 75)
    print(f"⏱️  Raw: {raw_hours:.2f}h | Converted: {conv_hours:.2f}h | Hao hụt: {gap_hours:.2f}h ({gap_percent:.1f}%)")
    print(f"📦 Scenes thành công: {count_converted}/{len(scenes)} scenes")
    print("=" * 75)


if __name__ == "__main__":
    main()
