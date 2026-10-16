#!/usr/bin/env python3
"""Script thống kê và kiểm tra toàn diện chất lượng dữ liệu thô (raw data) và dữ liệu đã convert (converted data) cho CaMo-JEPA.

Chức năng:
1. Thống kê DỮ LIỆU ĐÃ CONVERT (/dataset/camo_jepa/datasets_converted):
   - Đọc manifest.jsonl hoặc các file episodes/*.npz
   - Tính tổng số giờ đã convert (ở 10Hz: 36,000 frames = 1 giờ)
   - Phân loại xem TỪNG TẬP DATARAW gốc đã đóng góp bao nhiêu giờ vào bộ converted
2. Thống kê TOÀN BỘ DATA RAW (/dataset/camo_jepa/datasets_raw):
   - Quét toàn bộ 1124 scenes trong raw data
   - Đếm số scenes đạt chuẩn (1120 scenes) và số giờ tiềm năng của từng tập
   - Phân tích 4 scenes bị thiếu OTHERS
   - Tính tỷ lệ tiến độ đã convert (Converted Hours vs Raw Hours)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import zipfile
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

REQUIRED_STREAMS = ("NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER")
DEFAULT_SOURCE_DIRS = [
    "/dataset/camo_jepa/datasets_raw",
    "/dataset/data260926",
]
DEFAULT_CONVERTED_DIR = "/dataset/camo_jepa/datasets_converted"


def resolve_existing_path(path_str: str) -> Path:
    """Xử lý đường dẫn linh hoạt (hỗ trợ cả đường dẫn tuyệt đối /dataset... và tương đối dataset...)."""
    p = Path(path_str)
    if p.exists():
        return p
    if path_str.startswith("/"):
        rel_p = Path(path_str.lstrip("/"))
        if rel_p.exists():
            return rel_p
    return p


def token_to_ns(token: str) -> Optional[int]:
    """Parse timestamp string sang nanoseconds."""
    token = token.strip()
    if not token:
        return None

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
        if len(token) == 16:
            return val * 1_000
        elif len(token) >= 18:
            return val
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

    candidates = ["CAM_P_F", "CAM_F"]
    for c in candidates:
        target = cam_root / c
        if target.is_dir():
            return c, target

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

    empty_streams = []
    stream_boundaries = {}
    stream_all_ts = {}

    for st_name, st_path in stream_dirs.items():
        if full_sync:
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

    telemetry_min = max(b[1] for b in stream_boundaries.values())
    telemetry_max = min(b[2] for b in stream_boundaries.values())
    max_drift_ns = int(max_drift_ms * 1_000_000)

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
            reason=f"Lệch pha thời gian hoàn toàn giữa camera và telemetry",
        )

    valid_frames_count = 0
    skipped_frames_count = 0

    if full_sync:
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
        overlap_start = telemetry_min - max_drift_ns
        overlap_end = telemetry_max + max_drift_ns
        for ts_cam in cam_timestamps:
            if overlap_start <= ts_cam <= overlap_end:
                valid_frames_count += 1
            else:
                skipped_frames_count += 1

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
            reason=f"Số frame hợp lệ ({valid_frames_count}) ít hơn ngưỡng tối thiểu ({min_frames})",
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
) -> Tuple[List[SceneReport], Dict[str, str]]:
    """Quét các thư mục raw data, trả về danh sách SceneReport và bảng map scene_name -> group_name."""
    reports: List[SceneReport] = []
    scene_to_group: Dict[str, str] = {}

    for base_str in source_dirs:
        base_path = resolve_existing_path(base_str)
        if not base_path.exists():
            print(f"⚠️  Bỏ qua (đường dẫn raw data không tồn tại): {base_path}")
            continue

        print(f"🔍 Đang quét raw data: {base_path}")

        for root, dirs, _ in os.walk(base_path):
            root_path = Path(root)
            has_camera = "CAMERA" in dirs
            has_others = "OTHERS" in dirs
            has_direct_streams = all(st in dirs for st in REQUIRED_STREAMS)

            if has_camera or has_others or has_direct_streams:
                base_name = base_path.name
                rel_parts = root_path.relative_to(base_path).parts
                if len(rel_parts) > 1:
                    sub_grp = "/".join(rel_parts[:-1])
                    group_name = f"{base_name}/{sub_grp}"
                else:
                    group_name = f"{base_name}/Root"

                scene_name = root_path.name
                scene_to_group[scene_name] = group_name

                report = check_single_scene(
                    base_dir=str(base_path),
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

                dirs.clear()

    return reports, scene_to_group


def read_npz_frame_count(npz_path: Path) -> int:
    """Đọc số lượng frame của 1 file .npz nhanh chóng (bằng numpy hoặc parse header zip)."""
    if HAS_NUMPY:
        try:
            with np.load(npz_path) as data:
                if "timestamps_us" in data:
                    return len(data["timestamps_us"])
                elif "can_bus" in data:
                    return len(data["can_bus"])
        except Exception:
            pass

    try:
        with zipfile.ZipFile(npz_path, "r") as z:
            for item in z.namelist():
                if item.startswith("timestamps_us"):
                    header_bytes = z.read(item)[:256]
                    match = re.search(rb"'shape':\s*\(([^)]*)\)", header_bytes)
                    if match:
                        shape_str = match.group(1).decode("utf-8").strip().rstrip(",")
                        return int(shape_str)
    except Exception:
        pass
    return 0


def audit_converted_dataset(
    converted_dir: Path,
    scene_to_group: Dict[str, str],
    target_hz: float = 10.0,
) -> Dict[str, Any]:
    """Kiểm tra chi tiết thư mục datasets_converted, đếm số giờ và phân loại theo nguồn raw data."""
    result = {
        "exists": False,
        "total_episodes": 0,
        "total_frames": 0,
        "total_seconds": 0.0,
        "total_hours": 0.0,
        "group_breakdown": defaultdict(lambda: {"episodes": 0, "frames": 0, "hours": 0.0}),
    }

    if not converted_dir.exists():
        return result

    result["exists"] = True
    manifest_path = converted_dir / "manifest.jsonl"
    episodes_dir = converted_dir / "episodes"

    ep_frame_counts: Dict[str, int] = defaultdict(int)

    # Cách 1: Đọc từ manifest.jsonl (cực nhanh nếu có)
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ep_rel = row.get("episode", "")
                    ep_name = Path(ep_rel).stem
                    ep_frame_counts[ep_name] += 1
        except Exception:
            ep_frame_counts.clear()

    # Cách 2: Quét trực tiếp các file .npz trong episodes/ nếu manifest chưa có hoặc lỗi
    if not ep_frame_counts and episodes_dir.is_dir():
        npz_files = list(episodes_dir.rglob("*.npz"))
        for ep_path in npz_files:
            ep_name = ep_path.stem
            frames = read_npz_frame_count(ep_path)
            ep_frame_counts[ep_name] = frames

    total_episodes = len(ep_frame_counts)
    total_frames = sum(ep_frame_counts.values())
    total_seconds = total_frames / target_hz
    total_hours = total_seconds / 3600.0

    result["total_episodes"] = total_episodes
    result["total_frames"] = total_frames
    result["total_seconds"] = total_seconds
    result["total_hours"] = total_hours

    # Phân bổ từng episode về raw group
    for ep_name, frames in ep_frame_counts.items():
        group = scene_to_group.get(ep_name, "Chưa xác định / Khác")
        hours = (frames / target_hz) / 3600.0
        result["group_breakdown"][group]["episodes"] += 1
        result["group_breakdown"][group]["frames"] += frames
        result["group_breakdown"][group]["hours"] += hours

    return result


def format_duration(seconds: float) -> str:
    """Định dạng giây thành H:M:S."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}h {mins:02d}m {secs:02d}s"


def print_comprehensive_report(
    raw_reports: List[SceneReport],
    converted_info: Dict[str, Any],
    target_hz: float,
):
    # Gom nhóm raw data theo group_name
    raw_grouped: Dict[str, List[SceneReport]] = defaultdict(list)
    for r in raw_reports:
        raw_grouped[r.group_name].append(r)

    # 1. BÁO CÁO DỮ LIỆU ĐÃ CONVERT
    print("\n" + "=" * 92)
    print("📦 1. THỐNG KÊ DỮ LIỆU ĐÃ CONVERT (/dataset/camo_jepa/datasets_converted)")
    print("=" * 92)

    if not converted_info["exists"] or converted_info["total_episodes"] == 0:
        print("⚠️  Chưa tìm thấy tập dữ liệu đã convert trong datasets_converted (hoặc thư mục rỗng).")
        print("👉 Dữ liệu hiện tại đang ở dạng raw data, chưa chạy lệnh batch convert.")
    else:
        tot_ep = converted_info["total_episodes"]
        tot_fr = converted_info["total_frames"]
        tot_hr = converted_info["total_hours"]
        tot_dur = format_duration(converted_info["total_seconds"])

        header_c = f"{'Nguồn Raw gốc':<40} | {'Episodes':<9} | {'Frames Converted':<17} | {'Thời lượng':<12} | {'Giờ (h)':<7} | {'Tỷ trọng'}"
        print(header_c)
        print("-" * 92)

        for grp, data in sorted(converted_info["group_breakdown"].items()):
            dur_str = format_duration(data["hours"] * 3600.0)
            share = (data["hours"] / tot_hr * 100) if tot_hr > 0 else 0.0
            print(
                f"{grp:<40} | {data['episodes']:<9} | {data['frames']:<17,d} | {dur_str:<12} | {data['hours']:<7.2f} | {share:5.1f}%"
            )

        print("-" * 92)
        print(
            f"{'TỔNG CỘNG ĐÃ CONVERT':<40} | {tot_ep:<9} | {tot_fr:<17,d} | {tot_dur:<12} | {tot_hr:<7.2f} | 100.0%"
        )

    # 2. BÁO CÁO TỔNG THỂ DỮ LIỆU THÔ (RAW DATA) VÀ ĐÓNG GÓP TIỀM NĂNG
    print("\n" + "=" * 92)
    print("📊 2. THỐNG KÊ TOÀN BỘ DATA THÔ (/dataset/camo_jepa/datasets_raw) & ĐÓNG GÓP TỪNG TẬP")
    print("=" * 92)

    header_r = f"{'Tập dữ liệu thô (Raw Group)':<38} | {'Scenes':<7} | {'Chuẩn':<6} | {'Frames Chuẩn':<13} | {'Giờ Tiềm Năng':<14} | {'Đã Convert'}"
    print(header_r)
    print("-" * 92)

    tot_raw_scenes = len(raw_reports)
    tot_valid_scenes = sum(1 for r in raw_reports if r.is_valid)
    tot_raw_frames = sum(r.valid_frames for r in raw_reports if r.is_valid)
    tot_raw_seconds = sum(r.duration_seconds for r in raw_reports if r.is_valid)
    tot_raw_hours = tot_raw_seconds / 3600.0

    all_groups = set(raw_grouped.keys()).union(converted_info.get("group_breakdown", {}).keys())

    for grp in sorted(all_groups):
        if grp == "Chưa xác định / Khác":
            continue
        r_list = raw_grouped.get(grp, [])
        cnt_scenes = len(r_list)
        cnt_valid = sum(1 for r in r_list if r.is_valid)
        grp_valid_frames = sum(r.valid_frames for r in r_list if r.is_valid)
        grp_hours = (grp_valid_frames / target_hz) / 3600.0
        dur_str = f"{grp_hours:.2f}h"

        conv_hours = converted_info.get("group_breakdown", {}).get(grp, {}).get("hours", 0.0)
        pct_done = f"{(conv_hours / grp_hours * 100):.1f}%" if grp_hours > 0 else "0.0%"
        conv_str = f"{conv_hours:.2f}h ({pct_done})" if conv_hours > 0 else "Chưa convert"

        print(
            f"{grp:<38} | {cnt_scenes:<7} | {cnt_valid:<6} | {grp_valid_frames:<13,d} | {dur_str:<14} | {conv_str}"
        )

    print("-" * 92)
    grand_conv_hours = converted_info.get("total_hours", 0.0)
    grand_pct = f"{(grand_conv_hours / tot_raw_hours * 100):.1f}%" if tot_raw_hours > 0 else "0.0%"
    grand_conv_str = f"{grand_conv_hours:.2f}h ({grand_pct})" if grand_conv_hours > 0 else "Chưa convert"
    print(
        f"{'TỔNG CỘNG TOÀN BỘ DATASET':<38} | {tot_raw_scenes:<7} | {tot_valid_scenes:<6} | "
        f"{tot_raw_frames:<13,d} | {tot_raw_hours:<14.2f}h | {grand_conv_str}"
    )
    print("=" * 92)

    # 3. PHÂN LOẠI CÁC SCENES LỖI THIẾU OTHERS HOẶC THIẾU DỮ LIỆU
    failed_reports = [r for r in raw_reports if not r.is_valid]
    print("\n" + "=" * 92)
    if failed_reports:
        print(f"🔴 3. CHI TIẾT CÁC SCENES CHƯA ĐẠT CHUẨN ({len(failed_reports)} SCENES):")
        print("=" * 92)
        for r in failed_reports:
            print(f"   ❌ [{r.group_name}] {r.scene_name}")
            print(f"      👉 Lý do: {r.reason}")

        print(f"\n💡 QUAN TRỌNG:")
        print(f"   * Tổng số {tot_valid_scenes} scenes đạt chuẩn ({tot_raw_hours:.2f} giờ) HOÀN TOÀN SẴN SÀNG ĐỂ CONVERT!")
        print(f"   * Chỉ có {len(failed_reports)} scenes bị thiếu OTHERS/ hoặc lỗi, không ảnh hưởng đến các scenes còn lại.")
        print(f"   * Bạn có thể chạy lệnh convert ngay để trích xuất {tot_raw_hours:.2f} giờ dữ liệu cho CaMo-JEPA.")
    else:
        print("🎉 100% TẤT CẢ SCENES ĐỀU ĐẠT CHUẨN ĐỂ HUẤN LUYỆN CAMO-JEPA!")
    print("=" * 92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thống kê dữ liệu đã convert và dữ liệu raw cho CaMo-JEPA."
    )
    env_sources = os.environ.get("DATASETS_RAW_DIRS") or os.environ.get("DATASETS_RAW_DIR")
    default_dirs = [s.strip() for s in env_sources.split(",")] if env_sources else DEFAULT_SOURCE_DIRS
    converted_env = os.environ.get("DATASETS_CONVERTED_DIR", DEFAULT_CONVERTED_DIR)

    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=default_dirs,
        help=f"Danh sách các thư mục chứa raw data (mặc định: {default_dirs})",
    )
    parser.add_argument(
        "--converted-dir",
        default=converted_env,
        help=f"Thư mục datasets_converted (mặc định: {converted_env})",
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
        help="Độ lệch thời gian tối đa cho phép (mặc định: 80.0 ms)",
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
        help="Số frame hợp lệ tối thiểu của 1 episode (mặc định: 16)",
    )
    parser.add_argument(
        "--full-sync",
        action="store_true",
        help="Bật so khớp chính xác từng frame bằng bisect",
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

    print("=" * 92)
    print("🕵️  AUDIT DỮ LIỆU: THỐNG KÊ SỐ GIỜ ĐÃ CONVERT & TIỀM NĂNG DATA RAW CHO CAMO-JEPA")
    print("=" * 92)
    print(f"📌 Raw data dirs     : {args.source_dirs}")
    print(f"📌 Converted data dir: {args.converted_dir}")
    print(f"📌 Camera chỉ định   : {args.camera}")
    print(f"📌 Tần số chuẩn (Hz) : {args.target_hz} Hz (1 giờ = {int(args.target_hz * 3600):,} frames)")
    print("=" * 92)

    # 1. Quét dữ liệu thô
    raw_reports, scene_to_group = scan_source_directories(
        source_dirs=args.source_dirs,
        camera_arg=args.camera,
        max_drift_ms=args.max_drift_ms,
        target_hz=args.target_hz,
        min_frames=args.min_frames,
        full_sync=args.full_sync,
        verbose=args.verbose,
    )

    # 2. Quét dữ liệu đã convert
    converted_path = resolve_existing_path(args.converted_dir)
    converted_info = audit_converted_dataset(converted_path, scene_to_group, args.target_hz)

    # 3. Xuất bảng báo cáo tổng hợp
    print_comprehensive_report(raw_reports, converted_info, args.target_hz)

    if args.output_json:
        data_to_save = {
            "config": {
                "source_dirs": args.source_dirs,
                "converted_dir": args.converted_dir,
                "target_hz": args.target_hz,
                "max_drift_ms": args.max_drift_ms,
            },
            "converted_dataset": converted_info,
            "raw_dataset": {
                "total_scenes": len(raw_reports),
                "valid_scenes": sum(1 for r in raw_reports if r.is_valid),
                "total_valid_hours": sum(r.duration_seconds for r in raw_reports if r.is_valid) / 3600.0,
                "scenes": [asdict(r) for r in raw_reports],
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Đã lưu kết quả chi tiết ra: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
