#!/usr/bin/env python3
"""Script thống kê và kiểm tra toàn diện chất lượng dữ liệu thô (raw data) cho CaMo-JEPA.

Tiêu chuẩn đạt chuẩn để đưa vào huấn luyện CaMo-JEPA:
1. Cấu trúc thư mục:
   - Có thư mục CAMERA chứa ảnh JPG (mặc định ưu tiên CAM_P_F hoặc CAM_F).
   - Có 4 luồng telemetry bắt buộc: NAV, IMU, VEHICLE_INFO, VEHICLE_STEER (trong OTHERS/ hoặc thư mục gốc scene).
2. Tính toàn vẹn dữ liệu:
   - File ảnh JPG có thể trích xuất timestamp từ tên file (<seconds>-<nanoseconds>.jpg).
   - Các stream telemetry có ít nhất 1 file .csv có dữ liệu và cột Timestamp.
3. Tính đồng bộ thời gian (Temporal Synchronization):
   - Dải thời gian của camera giao thoa với dải thời gian của cả 4 luồng telemetry.
   - Độ lệch giữa timestamp frame camera và mẫu telemetry gần nhất <= max_drift_ms (mặc định 80ms).
4. Chiều dài tối thiểu (Minimum Episode Length):
   - Số frame đạt chuẩn sau đồng bộ >= min_frames (mặc định 16 frames, bằng history_length của CaMo-JEPA).
5. Thời lượng (Duration):
   - Quy đổi ra số giờ chuẩn dựa trên target_hz (mặc định 10.0 Hz): Hours = Valid_Frames / (target_hz * 3600).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REQUIRED_STREAMS = ("NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER")
DEFAULT_SOURCE_DIRS = [
    "/dataset/camo_jepa/datasets_raw",
]


def token_to_ns(token: str) -> Optional[int]:
    """Parse timestamp string (e.g. 1772517548-432551158 or 1772517548_432551158) to nanoseconds."""
    token = token.strip()
    if not token:
        return None

    # Handle float / int string directly
    if token.replace(".", "", 1).isdigit() and "." in token:
        try:
            return int(float(token) * 1_000_000_000)
        except (ValueError, OverflowError):
            return None

    delim = "-" if "-" in token else ("_" if "_" in token else None)
    if delim:
        parts = token.split(delim, 1)
        seconds_str, subsecond_str = parts[0], parts[1]
        if seconds_str.isdigit():
            digits = "".join(c for c in subsecond_str if c.isdigit())
            sub_ns = int((digits + "000000000")[:9] or "0")
            return int(seconds_str) * 1_000_000_000 + sub_ns

    if token.isdigit():
        val = int(token)
        # If timestamp is already in microseconds (16 digits)
        if len(token) == 16:
            return val * 1_000
        # If timestamp is in nanoseconds (19 digits)
        elif len(token) >= 18:
            return val
        # If timestamp is in seconds (10 digits)
        elif len(token) == 10:
            return val * 1_000_000_000
        return val

    return None


def read_csv_timestamps(csv_path: Path, max_rows: Optional[int] = None) -> List[int]:
    """Đọc danh sách timestamp_ns từ file CSV telemetry."""
    timestamps: List[int] = []
    try:
        with csv_path.open(newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("Timestamp") or row.get("timestamp") or ""
                ts_ns = token_to_ns(ts_str)
                if ts_ns is not None:
                    timestamps.append(ts_ns)
                if max_rows and len(timestamps) >= max_rows:
                    break
    except Exception:
        pass
    return sorted(timestamps)


def read_stream_boundary(stream_dir: Path) -> Tuple[int, Optional[int], Optional[int]]:
    """Đọc nhanh số lượng timestamp, timestamp nhỏ nhất và lớn nhất của một stream."""
    all_ts: List[int] = []
    csv_files = sorted(stream_dir.glob("*.csv"))
    if not csv_files:
        return 0, None, None

    for csv_file in csv_files:
        ts_list = read_csv_timestamps(csv_file)
        if ts_list:
            all_ts.extend(ts_list)

    if not all_ts:
        return 0, None, None

    all_ts.sort()
    return len(all_ts), all_ts[0], all_ts[-1]


def find_camera_dir(scene_path: Path, preferred_cam: str = "auto") -> Optional[Tuple[str, Path]]:
    """Tìm thư mục camera trong scene."""
    cam_root = scene_path / "CAMERA"
    if not cam_root.is_dir():
        return None

    if preferred_cam != "auto":
        target = cam_root / preferred_cam
        if target.is_dir():
            return preferred_cam, target
        return None

    # Tự động ưu tiên CAM_P_F, rồi CAM_F, rồi thư mục con bất kỳ có ảnh
    candidates = ["CAM_P_F", "CAM_F"]
    for c in candidates:
        target = cam_root / c
        if target.is_dir():
            return c, target

    # Nếu không thấy tên quen thuộc, lấy thư mục con đầu tiên trong CAMERA
    subdirs = [d for d in cam_root.iterdir() if d.is_dir()]
    if subdirs:
        return subdirs[0].name, subdirs[0]

    return None


def find_telemetry_dirs(scene_path: Path) -> Tuple[Dict[str, Path], List[str]]:
    """Tìm 4 thư mục stream telemetry (ưu tiên trong OTHERS/, fallback ở thư mục gốc)."""
    others_dir = scene_path / "OTHERS"
    found_streams: Dict[str, Path] = {}
    missing_streams: List[str] = []

    for stream_name in REQUIRED_STREAMS:
        if others_dir.is_dir() and (others_dir / stream_name).is_dir():
            found_streams[stream_name] = others_dir / stream_name
        elif (scene_path / stream_name).is_dir():
            found_streams[stream_name] = scene_path / stream_name
        else:
            missing_streams.append(stream_name)

    return found_streams, missing_streams


@dataclass
class SceneReport:
    base_dir: str
    group_name: str
    scene_name: str
    camera_name: str
    is_valid: bool
    status: str
    total_camera_frames: int
    valid_frames: int
    skipped_frames: int
    duration_seconds: float
    duration_hours: float
    reason: str = ""


def check_single_scene(
    base_dir: str,
    group_name: str,
    scene_path: Path,
    camera_arg: str,
    max_drift_ms: float,
    target_hz: float,
    min_frames: int,
    full_sync: bool,
) -> SceneReport:
    scene_name = scene_path.name

    # 1. Kiểm tra camera
    cam_info = find_camera_dir(scene_path, camera_arg)
    if not cam_info:
        return SceneReport(
            base_dir=base_dir,
            group_name=group_name,
            scene_name=scene_name,
            camera_name="NONE",
            is_valid=False,
            status="MISSING_CAMERA",
            total_camera_frames=0,
            valid_frames=0,
            skipped_frames=0,
            duration_seconds=0.0,
            duration_hours=0.0,
            reason="Không có thư mục CAMERA hoặc camera yêu cầu không tồn tại",
        )

    cam_name, cam_dir = cam_info
    jpg_files = sorted(cam_dir.glob("*.jpg"))
    if not jpg_files:
        return SceneReport(
            base_dir=base_dir,
            group_name=group_name,
            scene_name=scene_name,
            camera_name=cam_name,
            is_valid=False,
            status="EMPTY_CAMERA",
            total_camera_frames=0,
            valid_frames=0,
            skipped_frames=0,
            duration_seconds=0.0,
            duration_hours=0.0,
            reason=f"Thư mục CAMERA/{cam_name} không có file ảnh JPG",
        )

    # 2. Kiểm tra telemetry
    stream_dirs, missing_streams = find_telemetry_dirs(scene_path)
    if missing_streams:
        return SceneReport(
            base_dir=base_dir,
            group_name=group_name,
            scene_name=scene_name,
            camera_name=cam_name,
            is_valid=False,
            status="MISSING_TELEMETRY",
            total_camera_frames=len(jpg_files),
            valid_frames=0,
            skipped_frames=len(jpg_files),
            duration_seconds=0.0,
            duration_hours=0.0,
            reason=f"Thiếu các stream telemetry bắt buộc: {', '.join(missing_streams)}",
        )

    # 3. Kiểm tra tính rỗng của file CSV trong telemetry
    empty_streams = []
    stream_boundaries = {}
    stream_all_ts = {}

    for st_name, st_path in stream_dirs.items():
        if full_sync:
            # Chế độ full sync: đọc toàn bộ timestamps
            all_ts = []
            for csv_file in sorted(st_path.glob("*.csv")):
                all_ts.extend(read_csv_timestamps(csv_file))
            if not all_ts:
                empty_streams.append(st_name)
            else:
                all_ts.sort()
                stream_all_ts[st_name] = all_ts
                stream_boundaries[st_name] = (len(all_ts), all_ts[0], all_ts[-1])
        else:
            count, ts_min, ts_max = read_stream_boundary(st_path)
            if count == 0 or ts_min is None or ts_max is None:
                empty_streams.append(st_name)
            else:
                stream_boundaries[st_name] = (count, ts_min, ts_max)

    if empty_streams:
        return SceneReport(
            base_dir=base_dir,
            group_name=group_name,
            scene_name=scene_name,
            camera_name=cam_name,
            is_valid=False,
            status="EMPTY_TELEMETRY_CSV",
            total_camera_frames=len(jpg_files),
            valid_frames=0,
            skipped_frames=len(jpg_files),
            duration_seconds=0.0,
            duration_hours=0.0,
            reason=f"Stream có thư mục nhưng CSV rỗng hoặc lỗi: {', '.join(empty_streams)}",
        )

    # 4. Kiểm tra timestamp và đồng bộ (Synchronization)
    cam_timestamps: List[int] = []
    for img_path in jpg_files:
        ts = token_to_ns(img_path.stem)
        if ts is not None:
            cam_timestamps.append(ts)

    if not cam_timestamps:
        return SceneReport(
            base_dir=base_dir,
            group_name=group_name,
            scene_name=scene_name,
            camera_name=cam_name,
            is_valid=False,
            status="INVALID_TIMESTAMP_NAMES",
            total_camera_frames=len(jpg_files),
            valid_frames=0,
            skipped_frames=len(jpg_files),
            duration_seconds=0.0,
            duration_hours=0.0,
            reason="Tên file ảnh không đúng định dạng timestamp (<sec>-<subsec>.jpg)",
        )

    cam_timestamps.sort()
    cam_min = cam_timestamps[0]
    cam_max = cam_timestamps[-1]

    # Kiểm tra khoảng giao thoa thời gian (Time Overlap)
    telemetry_min = max(b[1] for b in stream_boundaries.values())
    telemetry_max = min(b[2] for b in stream_boundaries.values())

    max_drift_ns = int(max_drift_ms * 1_000_000)

    # Nếu dải thời gian camera hoàn toàn nằm ngoài telemetry
    if cam_max < (telemetry_min - max_drift_ns) or cam_min > (telemetry_max + max_drift_ns):
        return SceneReport(
            base_dir=base_dir,
            group_name=group_name,
            scene_name=scene_name,
            camera_name=cam_name,
            is_valid=False,
            status="NO_TIME_OVERLAP",
            total_camera_frames=len(jpg_files),
            valid_frames=0,
            skipped_frames=len(jpg_files),
            duration_seconds=0.0,
            duration_hours=0.0,
            reason=f"Lệch pha thời gian hoàn toàn giữa camera [{cam_min}..{cam_max}] và telemetry [{telemetry_min}..{telemetry_max}]",
        )

    # 5. Đánh giá số frame đồng bộ được
    valid_frames_count = 0
    skipped_frames_count = 0

    if full_sync:
        # Full sync: kiểm tra từng frame theo thuật toán bisect giống convert_vf_to_camo.py
        for ts_cam in cam_timestamps:
            is_synced = True
            for st_name in REQUIRED_STREAMS:
                st_ts = stream_all_ts[st_name]
                idx = bisect_left(st_ts, ts_cam)
                candidates = st_ts[max(0, idx - 1):idx + 1]
                if not candidates:
                    is_synced = False
                    break
                min_diff = min(abs(c - ts_cam) for c in candidates)
                if min_diff > max_drift_ns:
                    is_synced = False
                    break
            if is_synced:
                valid_frames_count += 1
            else:
                skipped_frames_count += 1
    else:
        # Fast estimation: đếm số frame camera nằm trong dải overlap (telemetry_min - drift .. telemetry_max + drift)
        overlap_start = telemetry_min - max_drift_ns
        overlap_end = telemetry_max + max_drift_ns
        for ts_cam in cam_timestamps:
            if overlap_start <= ts_cam <= overlap_end:
                valid_frames_count += 1
            else:
                skipped_frames_count += 1

    # Resample về target_hz (10 Hz):
    # Mỗi 10 frame hợp lệ xấp xỉ 1 giây
    duration_sec = valid_frames_count / target_hz
    duration_hrs = duration_sec / 3600.0

    if valid_frames_count < min_frames:
        return SceneReport(
            base_dir=base_dir,
            group_name=group_name,
            scene_name=scene_name,
            camera_name=cam_name,
            is_valid=False,
            status="TOO_SHORT",
            total_camera_frames=len(jpg_files),
            valid_frames=valid_frames_count,
            skipped_frames=skipped_frames_count,
            duration_seconds=duration_sec,
            duration_hours=duration_hrs,
            reason=f"Số frame hợp lệ ({valid_frames_count}) ít hơn ngưỡng tối thiểu min_frames ({min_frames})",
        )

    return SceneReport(
        base_dir=base_dir,
        group_name=group_name,
        scene_name=scene_name,
        camera_name=cam_name,
        is_valid=True,
        status="OK",
        total_camera_frames=len(jpg_files),
        valid_frames=valid_frames_count,
        skipped_frames=skipped_frames_count,
        duration_seconds=duration_sec,
        duration_hours=duration_hrs,
        reason="Đạt chuẩn cho CaMo-JEPA",
    )


def scan_source_directories(
    source_dirs: List[str],
    camera_arg: str,
    max_drift_ms: float,
    target_hz: float,
    min_frames: int,
    full_sync: bool,
    verbose: bool,
) -> List[SceneReport]:
    reports: List[SceneReport] = []

    for base_str in source_dirs:
        base_path = Path(base_str)
        if not base_path.exists() and base_str.startswith("/"):
            rel_base = Path(base_str.lstrip("/"))
            if rel_base.exists():
                base_path = rel_base

        if not base_path.exists():
            print(f"⚠️  Bỏ qua (đường dẫn không tồn tại): {base_path}")
            continue

        print(f"\n🔍 Đang quét thư mục nguồn: {base_path}")

        # Tìm các thư mục scene (chứa CAMERA/ hoặc OTHERS/)
        for root, dirs, _ in os.walk(base_path):
            root_path = Path(root)
            has_camera = "CAMERA" in dirs
            has_others = "OTHERS" in dirs
            has_direct_streams = all(st in dirs for st in REQUIRED_STREAMS)

            if has_camera or has_others or has_direct_streams:
                # Xác định group_name (ví dụ: nhóm xe VF6_01_02 hoặc thư mục cha)
                rel_parts = root_path.relative_to(base_path).parts
                group_name = rel_parts[0] if len(rel_parts) > 1 else "Root"

                report = check_single_scene(
                    base_dir=base_str,
                    group_name=group_name,
                    scene_path=root_path,
                    camera_arg=camera_arg,
                    max_drift_ms=max_drift_ms,
                    target_hz=target_hz,
                    min_frames=min_frames,
                    full_sync=full_sync,
                )
                reports.append(report)

                if verbose:
                    status_icon = "✅" if report.is_valid else "❌"
                    print(
                        f"  {status_icon} [{report.group_name}] {report.scene_name}: "
                        f"{report.status} | {report.valid_frames}/{report.total_camera_frames} frames "
                        f"({report.duration_hours:.2f}h) - {report.reason}"
                    )

                # Dừng walk sâu vào bên trong scene này
                dirs.clear()

    return reports


def format_duration(seconds: float) -> str:
    """Định dạng giây thành H:M:S."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}h {mins:02d}m {secs:02d}s"


def print_summary_tables(reports: List[SceneReport], target_hz: float):
    print("\n" + "=" * 90)
    print("📊 BÁO CÁO THỐNG KÊ CHI TIẾT THEO TỪNG TẬP DỮ LIỆU / NHÓM XE (DATASET GROUPS)")
    print("=" * 90)

    # Gom nhóm theo (base_dir, group_name)
    grouped: Dict[Tuple[str, str], List[SceneReport]] = defaultdict(list)
    for r in reports:
        grouped[(r.base_dir, r.group_name)].append(r)

    header = f"{'Nguồn dữ liệu / Nhóm':<35} | {'Tổng':<5} | {'Chuẩn':<5} | {'Lỗi':<5} | {'Frames Chuẩn':<12} | {'Thời lượng (H:M:S)':<18} | {'Giờ (h)':<7}"
    print(header)
    print("-" * 90)

    total_all_scenes = 0
    total_valid_scenes = 0
    total_failed_scenes = 0
    total_valid_frames = 0
    total_seconds = 0.0

    for (b_dir, g_name), r_list in sorted(grouped.items()):
        cnt_scenes = len(r_list)
        cnt_valid = sum(1 for r in r_list if r.is_valid)
        cnt_failed = cnt_scenes - cnt_valid
        grp_valid_frames = sum(r.valid_frames for r in r_list if r.is_valid)
        grp_seconds = sum(r.duration_seconds for r in r_list if r.is_valid)
        grp_hours = grp_seconds / 3600.0

        label = f"{Path(b_dir).name}/{g_name}" if g_name != "Root" else Path(b_dir).name
        label = label[:34]
        dur_str = format_duration(grp_seconds)

        print(
            f"{label:<35} | {cnt_scenes:<5} | {cnt_valid:<5} | {cnt_failed:<5} | {grp_valid_frames:<12,d} | {dur_str:<18} | {grp_hours:<7.2f}"
        )

        total_all_scenes += cnt_scenes
        total_valid_scenes += cnt_valid
        total_failed_scenes += cnt_failed
        total_valid_frames += grp_valid_frames
        total_seconds += grp_seconds

    print("=" * 90)
    grand_hours = total_seconds / 3600.0
    grand_dur_str = format_duration(total_seconds)
    total_row = (
        f"{'TỔNG CỘNG TOÀN BỘ DATASET':<35} | {total_all_scenes:<5} | {total_valid_scenes:<5} | "
        f"{total_failed_scenes:<5} | {total_valid_frames:<12,d} | {grand_dur_str:<18} | {grand_hours:<7.2f}"
    )
    print(total_row)
    print("=" * 90)

    # Phân tích nguyên nhân lỗi nếu có
    failed_reports = [r for r in reports if not r.is_valid]
    if failed_reports:
        print("\n" + "=" * 90)
        print("🔴 PHÂN LOẠI CÁC SCENES KHÔNG ĐẠT CHUẨN:")
        print("=" * 90)
        reason_counts: Dict[str, int] = defaultdict(int)
        for r in failed_reports:
            reason_counts[f"[{r.status}] {r.reason}"] += 1

        for reason_desc, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   * {count:3d} scenes : {reason_desc}")

        print(f"\n💡 Tỷ lệ dữ liệu đạt chuẩn (Yield Rate): "
              f"{(total_valid_scenes / total_all_scenes * 100):.1f}% số scenes "
              f"({grand_hours:.2f} giờ khả dụng cho CaMo-JEPA ở tần số {target_hz:.0f}Hz)")
    else:
        print(f"\n🎉 100% CÁC SCENES ĐỀU ĐẠT CHUẨN! Tổng cộng {grand_hours:.2f} giờ dữ liệu.")

    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thống kê các tập raw data và tính tổng số giờ đạt chuẩn cho CaMo-JEPA."
    )
    env_sources = os.environ.get("RAW_DATA_DIRS")
    default_dirs = [s.strip() for s in env_sources.split(",")] if env_sources else DEFAULT_SOURCE_DIRS

    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=default_dirs,
        help=f"Danh sách các thư mục chứa raw data (mặc định: {default_dirs})",
    )
    parser.add_argument(
        "--camera",
        default="auto",
        help="Tên camera phía trước (mặc định 'auto' ưu tiên CAM_P_F rồi CAM_F)",
    )
    parser.add_argument(
        "--max-drift-ms",
        type=float,
        default=80.0,
        help="Độ lệch thời gian tối đa cho phép giữa camera và telemetry (mặc định: 80.0 ms)",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=10.0,
        help="Tần số chuẩn hóa của CaMo-JEPA (mặc định: 10.0 Hz)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=16,
        help="Số frame hợp lệ tối thiểu của 1 episode để train CaMo-JEPA (mặc định: 16)",
    )
    parser.add_argument(
        "--full-sync",
        action="store_true",
        help="Bật so khớp chính xác từng frame bằng bisect (mặc định dùng range check nhanh)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="In chi tiết từng scene trong lúc quét",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Lưu báo cáo thống kê ra file JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 90)
    print("🕵️  CÔNG CỤ KIỂM TRA & THỐNG KÊ GIỜ DỮ LIỆU ĐẠT CHUẨN CHO CAMO-JEPA")
    print("=" * 90)
    print(f"📌 Thư mục nguồn quét : {args.source_dirs}")
    print(f"📌 Camera chỉ định    : {args.camera}")
    print(f"📌 Ngưỡng drift tối đa: {args.max_drift_ms} ms")
    print(f"📌 Tần số đích        : {args.target_hz} Hz")
    print(f"📌 Độ dài tối thiểu   : {args.min_frames} frames (history_length)")
    print(f"📌 Chế độ đồng bộ     : {'Full-Sync (từng frame)' if args.full_sync else 'Fast Range Check'}")
    print("=" * 90)

    reports = scan_source_directories(
        source_dirs=args.source_dirs,
        camera_arg=args.camera,
        max_drift_ms=args.max_drift_ms,
        target_hz=args.target_hz,
        min_frames=args.min_frames,
        full_sync=args.full_sync,
        verbose=args.verbose,
    )

    if not reports:
        print("\n⚠️ Không tìm thấy scene nào trong các thư mục được quét!")
        print("💡 Gợi ý: Hãy kiểm tra lại đường dẫn với tham số '--source-dirs <đường_dẫn>'")
        return

    print_summary_tables(reports, args.target_hz)

    if args.output_json:
        data_to_save = {
            "config": {
                "source_dirs": args.source_dirs,
                "camera": args.camera,
                "max_drift_ms": args.max_drift_ms,
                "target_hz": args.target_hz,
                "min_frames": args.min_frames,
                "full_sync": args.full_sync,
            },
            "total_scenes": len(reports),
            "valid_scenes": sum(1 for r in reports if r.is_valid),
            "total_valid_frames": sum(r.valid_frames for r in reports if r.is_valid),
            "total_valid_hours": sum(r.duration_seconds for r in reports if r.is_valid) / 3600.0,
            "scenes": [asdict(r) for r in reports],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        print(f"💾 Đã lưu kết quả chi tiết ra: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
