#!/usr/bin/env python3
"""Convert a VF recording into a synchronized CaMo-JEPA dataset.

Input
-----
``--vf-root`` must contain the selected camera under
``CAMERA/<camera>/*.jpg`` and these telemetry streams under ``OTHERS/``:
``NAV``, ``IMU``, ``VEHICLE_INFO``, and ``VEHICLE_STEER``.

Synchronization
---------------
The default mode is **interpolation**: CAN-bus values at each camera
timestamp are computed by linearly interpolating between the two bracketing
telemetry rows.  IMU quaternions use SLERP (Spherical Linear Interpolation)
so that orientation is correctly interpolated on SO(3).  A frame is skipped
only when the two bracketing rows are more than ``--max-interp-gap-ms``
apart (default 300 ms) or when the camera timestamp falls outside the
recorded CAN-bus window.

Output
------
The converter writes to ``--output-root`` using the CaMo-JEPA v1 layout:

* ``images/<split>/<log-name>/<timestamp>.jpg``
* ``episodes/<split>/<log-name>.npz``
* ``manifest.jsonl``
* ``dataset_info.json``

The episode stores timestamps, source timestamps, image paths, an 18-value
``can_bus`` vector, and ``ego_motion`` as
``[yaw_rate_radps, longitudinal_acceleration_mps2]``. Output is downsampled
to at most ``--target-hz`` (10 Hz by default) without changing retained
source timestamps. Images are symlinked by default; use ``--image-mode`` to
select hard links or copies.

Usage
-----
From the repository root, validate synchronization without writing output:

    python3 src/utils/convert_vf_to_camo.py --vf-root dataset_vf/data --dry-run

Convert the recording and write the default dataset to
``dataset_camo/vf``:

    python3 src/utils/convert_vf_to_camo.py --vf-root dataset_vf/data

Use ``python src/utils/convert_vf_to_camo.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VF_ROOT = REPO_ROOT / "dataset_vf" / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "dataset_camo" / "vf"
EGO_MOTION_NAMES = ("yaw_rate_radps", "longitudinal_acceleration_mps2")
REQUIRED_STREAMS = ("NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER")


@dataclass(frozen=True)
class TelemetryRow:
    timestamp_ns: int
    values: Mapping[str, str]


@dataclass(frozen=True)
class FrameSample:
    timestamp_us: int
    source_timestamp_us: int
    token: str
    source_image: Path
    image_path: Path
    can_bus: np.ndarray
    ego_motion: np.ndarray


@dataclass
class SkippedFrame:
    token: str
    timestamp_us: int
    reason: str  # "out_of_bounds" | "gap_exceeded"
    stream_gaps_ms: dict = field(default_factory=dict)  # {stream_name: gap_ms or None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert synchronized VF images and CAN-bus telemetry into CaMo-JEPA episodes."
    )
    parser.add_argument(
        "--vf-root",
        type=Path,
        default=DEFAULT_VF_ROOT,
        help=f"VF data root containing CAMERA and OTHERS (default: {DEFAULT_VF_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination CaMo dataset root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--camera", default="CAM_P_F", help="VF camera directory (default: CAM_P_F)")
    parser.add_argument("--split", default="train", help="Output split name (default: train)")
    parser.add_argument(
        "--log-name",
        default="vf_recording_000",
        help="Output episode name for this VF recording (default: vf_recording_000)",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("interpolation", "nearest"),
        default="interpolation",
        help="Telemetry synchronization mode: 'interpolation' (default, LERP+SLERP) or 'nearest' (legacy row matching)",
    )
    parser.add_argument(
        "--max-drift-ms",
        type=float,
        default=80.0,
        help="[legacy nearest-row mode only] Maximum timestamp delta (default: 80 ms)",
    )
    parser.add_argument(
        "--max-interp-gap-ms",
        type=float,
        default=300.0,
        help=(
            "Maximum allowed gap between two bracketing CAN-bus rows when "
            "interpolating. Frames whose camera timestamp falls inside a gap "
            "larger than this are skipped as a CAN dropout (default: 300 ms)."
        ),
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How output image files refer to source images (default: symlink)",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=10.0,
        help="Maximum output frequency in Hz; source timestamps are preserved (default: 10)",
    )
    parser.add_argument("--max-frames", type=int, help="Convert at most this many frames")
    parser.add_argument("--dry-run", action="store_true", help="Validate synchronization without writing files")
    parser.add_argument(
        "--skip-log",
        action="store_true",
        help="Write skip_log.jsonl alongside the episode .npz listing every skipped frame and the reason",
    )
    return parser.parse_args()


def token_to_ns(token: str) -> int:
    try:
        seconds, subsecond = token.split("-", 1)
    except ValueError as error:
        raise ValueError(f"invalid VF timestamp token: {token}") from error
    if not seconds.isdigit():
        raise ValueError(f"invalid VF seconds component: {token}")
    digits = "".join(character for character in subsecond if character.isdigit())
    return int(seconds) * 1_000_000_000 + int((digits + "000000000")[:9] or "0")


def read_stream(stream_dir: Path) -> list[TelemetryRow]:
    rows: list[TelemetryRow] = []
    for csv_path in sorted(stream_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                timestamp = (row.get("Timestamp") or "").strip()
                if timestamp:
                    rows.append(TelemetryRow(token_to_ns(timestamp), row))
    rows.sort(key=lambda row: row.timestamp_ns)
    return rows


def nearest_row(
    rows: list[TelemetryRow], timestamps: list[int], target_ns: int
) -> tuple[TelemetryRow, int] | None:
    """Legacy nearest-row lookup; kept for backward-compatibility."""
    index = bisect_left(timestamps, target_ns)
    candidates = rows[max(0, index - 1):index + 1]
    if not candidates:
        return None
    selected = min(candidates, key=lambda row: abs(row.timestamp_ns - target_ns))
    return selected, abs(selected.timestamp_ns - target_ns)


def bracket_rows(
    rows: list[TelemetryRow],
    timestamps: list[int],
    target_ns: int,
    max_gap_ns: int,
) -> tuple[TelemetryRow, TelemetryRow, float] | None:
    """Return (row_before, row_after, fraction) for interpolation at target_ns.

    Follows the standard boundary and dropout rules (Waymo Open Dataset, ROS 2 tf2):
    * Returns None if target_ns is outside recorded range (strictly no extrapolation).
    * Returns None if the two bracketing rows are > max_gap_ns apart (sensor dropout).
    * Returns (row, row, 0.0) on exact match or identical timestamps.
    * ``fraction`` is in [0.0, 1.0]: 0.0 == row_before, 1.0 == row_after.
    """
    if not rows or not timestamps:
        return None

    # Bounds check — strictly no extrapolation
    if target_ns < timestamps[0] or target_ns > timestamps[-1]:
        return None

    idx = bisect_left(timestamps, target_ns)

    # Exact match
    if timestamps[idx] == target_ns:
        row = rows[idx]
        return row, row, 0.0

    # General case: rows[idx-1].timestamp_ns < target_ns < rows[idx].timestamp_ns
    row0 = rows[idx - 1]
    row1 = rows[idx]
    gap_ns = row1.timestamp_ns - row0.timestamp_ns

    if gap_ns <= 0:
        return row0, row1, 0.0

    if gap_ns > max_gap_ns:
        return None  # CAN dropout — do not interpolate across large missing data gap

    fraction = (target_ns - row0.timestamp_ns) / gap_ns
    return row0, row1, fraction


def slerp_quaternion(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    """Spherical linear interpolation between two unit quaternions (qw, qx, qy, qz).

    Follows the nuScenes / ROS 2 tf2 convention:
    * Rotation always takes the shortest arc on S^3.
    * Falls back to normalised LERP when the two quaternions are nearly parallel
      (dot > 0.9995) to avoid numerical division by zero.
    * Output quaternion is always unit-length.
    """
    # Unpack to arrays for arithmetic
    a = [q0[0], q0[1], q0[2], q0[3]]
    b = [q1[0], q1[1], q1[2], q1[3]]

    dot = sum(a[i] * b[i] for i in range(4))

    # Shortest-arc: negate b if dot is negative
    if dot < 0.0:
        b = [-x for x in b]
        dot = -dot

    # Near-parallel fallback to normalised LERP
    if dot > 0.9995:
        interp = [a[i] + fraction * (b[i] - a[i]) for i in range(4)]
        norm = math.sqrt(sum(x * x for x in interp))
        if norm < 1e-9:
            return (1.0, 0.0, 0.0, 0.0)
        return tuple(x / norm for x in interp)  # type: ignore[return-value]

    # Full SLERP
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta_t = theta_0 * fraction
    s0 = math.sin(theta_0 - theta_t) / sin_theta_0
    s1 = math.sin(theta_t) / sin_theta_0
    interp = [s0 * a[i] + s1 * b[i] for i in range(4)]
    norm = math.sqrt(sum(x * x for x in interp))
    return tuple(x / norm for x in interp)  # type: ignore[return-value]


def value(row: Mapping[str, str], name: str, default: float = 0.0) -> float:
    try:
        result = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def lerp(v0: float, v1: float, t: float) -> float:
    """Unclamped 1-D linear interpolation between v0 and v1 at fraction t."""
    return v0 + t * (v1 - v0)


def quaternion_from_row(imu: Mapping[str, str]) -> tuple[float, float, float, float]:
    """Read and normalise a quaternion from an IMU telemetry row."""
    result = (value(imu, "qw", 1.0), value(imu, "qx"), value(imu, "qy"), value(imu, "qz"))
    norm = math.sqrt(sum(component * component for component in result))
    return tuple(component / norm for component in result) if norm else (1.0, 0.0, 0.0, 0.0)  # type: ignore[return-value]


def local_position(nav: Mapping[str, str], origin: tuple[float, float]) -> tuple[float, float, float]:
    latitude = value(nav, "Latitude")
    longitude = value(nav, "Longitude")
    origin_latitude, origin_longitude = origin
    earth_radius_m = 6_378_137.0
    northing = math.radians(latitude - origin_latitude) * earth_radius_m
    easting = math.radians(longitude - origin_longitude) * earth_radius_m * math.cos(math.radians(origin_latitude))
    return easting, northing, value(nav, "Altitude")


def local_position_interp(
    nav0: Mapping[str, str],
    nav1: Mapping[str, str],
    fraction: float,
    origin: tuple[float, float],
) -> tuple[float, float, float]:
    """Linearly interpolate GPS position (easting, northing, altitude) at *fraction*."""
    p0 = local_position(nav0, origin)
    p1 = local_position(nav1, origin)
    return (
        lerp(p0[0], p1[0], fraction),
        lerp(p0[1], p1[1], fraction),
        lerp(p0[2], p1[2], fraction),
    )


def build_can_bus(
    nav: Mapping[str, str],
    imu: Mapping[str, str],
    vehicle: Mapping[str, str],
    steer: Mapping[str, str],
    origin: tuple[float, float],
) -> np.ndarray:
    """Build can_bus vector from a single (nearest-row) telemetry snapshot."""
    position = local_position(nav, origin)
    rotation = quaternion_from_row(imu)
    return np.asarray(
        (
            *position,
            *rotation,
            value(vehicle, "acceleration_x"),
            value(vehicle, "acceleration_y"),
            value(vehicle, "yaw_rate"),
            value(nav, "Ve"),
            value(nav, "Vn"),
            value(imu, "ax"),
            value(imu, "ay"),
            value(steer, "steer_angle"),
            value(steer, "steer_speed"),
            0.0,
            0.0,
        ),
        dtype=np.float32,
    )


def build_can_bus_interp(
    nav0: Mapping[str, str],
    nav1: Mapping[str, str],
    nav_fraction: float,
    imu0: Mapping[str, str],
    imu1: Mapping[str, str],
    imu_fraction: float,
    vehicle0: Mapping[str, str],
    vehicle1: Mapping[str, str],
    vehicle_fraction: float,
    steer0: Mapping[str, str],
    steer1: Mapping[str, str],
    steer_fraction: float,
    origin: tuple[float, float],
) -> np.ndarray:
    """Build can_bus vector [18] using per-stream linear interpolation (LERP/SLERP).

    Follows standard multi-sensor fusion specifications:
    * LERP for 3D position (easting, northing, altitude) and velocities (Ve, Vn) with nav_fraction
      (nuScenes Devkit Caesar et al. 2020 Sec 3.3, Argoverse 2 av2.geometry.interpolate).
    * SLERP for IMU orientation quaternion [qw, qx, qy, qz] with imu_fraction (Shoemake 1985, ROS 2 tf2).
    * LERP for IMU linear accelerations (ax, ay) with imu_fraction.
    * LERP for vehicle accelerations (accel_x, accel_y) and yaw_rate with vehicle_fraction (Comma.ai openpilot).
    * LERP for steering angle and speed with steer_fraction (Comma.ai openpilot).
    """
    # --- Position: LERP (easting, northing, altitude) [3] ---
    position = local_position_interp(nav0, nav1, nav_fraction, origin)

    # --- Orientation: SLERP (qw, qx, qy, qz) [4] ---
    q0 = quaternion_from_row(imu0)
    q1 = quaternion_from_row(imu1)
    rotation = slerp_quaternion(q0, q1, imu_fraction)

    # --- Vehicle kinematics: LERP [3] ---
    accel_x = lerp(value(vehicle0, "acceleration_x"), value(vehicle1, "acceleration_x"), vehicle_fraction)
    accel_y = lerp(value(vehicle0, "acceleration_y"), value(vehicle1, "acceleration_y"), vehicle_fraction)
    yaw_rate = lerp(value(vehicle0, "yaw_rate"), value(vehicle1, "yaw_rate"), vehicle_fraction)

    # --- Velocities: LERP [2] ---
    vel_e = lerp(value(nav0, "Ve"), value(nav1, "Ve"), nav_fraction)
    vel_n = lerp(value(nav0, "Vn"), value(nav1, "Vn"), nav_fraction)

    # --- IMU accelerations: LERP [2] ---
    imu_ax = lerp(value(imu0, "ax"), value(imu1, "ax"), imu_fraction)
    imu_ay = lerp(value(imu0, "ay"), value(imu1, "ay"), imu_fraction)

    # --- Steering: LERP [2] ---
    steer_angle = lerp(value(steer0, "steer_angle"), value(steer1, "steer_angle"), steer_fraction)
    steer_speed = lerp(value(steer0, "steer_speed"), value(steer1, "steer_speed"), steer_fraction)

    # Output can_bus shape: [18]
    return np.asarray(
        (
            *position,      # [3] easting, northing, altitude
            *rotation,      # [4] qw, qx, qy, qz
            accel_x,        # [1] longitudinal accel
            accel_y,        # [1] lateral accel
            yaw_rate,       # [1] yaw rate
            vel_e,          # [1] velocity east
            vel_n,          # [1] velocity north
            imu_ax,         # [1] IMU accel x
            imu_ay,         # [1] IMU accel y
            steer_angle,    # [1] steering angle
            steer_speed,    # [1] steering angular velocity
            0.0,            # [1] padding
            0.0,            # [1] padding
        ),
        dtype=np.float32,
    )


def resample_samples(samples: list[FrameSample], target_hz: float) -> list[FrameSample]:
    """Downsample while retaining original image, CAN-bus, and timestamp values."""
    if len(samples) < 2:
        return samples
    interval_us = round(1_000_000 / target_hz)
    selected = [samples[0]]
    next_timestamp_us = samples[0].timestamp_us + interval_us
    for sample in samples[1:]:
        if sample.timestamp_us >= next_timestamp_us:
            selected.append(sample)
            next_timestamp_us = sample.timestamp_us + interval_us
    return selected


def build_samples(
    args: argparse.Namespace,
) -> tuple[list[FrameSample], int, list[SkippedFrame]]:
    camera_root = args.vf_root / "CAMERA" / args.camera
    streams = {name: read_stream(args.vf_root / "OTHERS" / name) for name in REQUIRED_STREAMS}
    missing = [name for name, rows in streams.items() if not rows]
    if missing:
        raise ValueError(f"required VF telemetry streams are empty: {', '.join(missing)}")
    stream_timestamps = {
        name: [row.timestamp_ns for row in rows]
        for name, rows in streams.items()
    }
    image_paths = sorted(camera_root.glob("*.jpg"), key=lambda path: token_to_ns(path.stem))
    if not image_paths:
        raise ValueError(f"no JPG files found for camera: {camera_root}")

    first_nav = streams["NAV"][0].values
    origin = (value(first_nav, "Latitude"), value(first_nav, "Longitude"))
    sync_mode = getattr(args, "sync_mode", "interpolation")
    max_gap_ns = int(getattr(args, "max_interp_gap_ms", 300.0) * 1_000_000)
    max_drift_ns = int(getattr(args, "max_drift_ms", 80.0) * 1_000_000)
    image_root = args.output_root / "images" / args.split / args.log_name
    samples: list[FrameSample] = []
    skipped_frames: list[SkippedFrame] = []
    skipped = 0
    record_skips = getattr(args, "skip_log", False)

    for source_image in image_paths:
        timestamp_ns = token_to_ns(source_image.stem)

        if sync_mode == "interpolation":
            # Find bracketing rows for each stream independently
            brackets: dict[str, tuple[TelemetryRow, TelemetryRow, float] | None] = {
                name: bracket_rows(rows, stream_timestamps[name], timestamp_ns, max_gap_ns)
                for name, rows in streams.items()
            }

            # Determine skip reason for any stream that failed
            failed_streams: dict[str, str] = {}
            gap_info: dict[str, Optional[float]] = {}
            for name, bracket in brackets.items():
                ts = stream_timestamps[name]
                if bracket is None:
                    if timestamp_ns < ts[0] or timestamp_ns > ts[-1]:
                        failed_streams[name] = "out_of_bounds"
                        gap_info[name] = None
                    else:
                        # Inside bounds but gap too large → dropout
                        idx = bisect_left(ts, timestamp_ns)
                        gap_ms = round((ts[idx] - ts[idx - 1]) / 1_000_000, 1)
                        failed_streams[name] = "gap_exceeded"
                        gap_info[name] = gap_ms
                else:
                    row0, row1, _ = bracket
                    gap_info[name] = round((row1.timestamp_ns - row0.timestamp_ns) / 1_000_000, 1)

            if failed_streams:
                skipped += 1
                if record_skips:
                    reason = (
                        "out_of_bounds"
                        if any(v == "out_of_bounds" for v in failed_streams.values())
                        else "gap_exceeded"
                    )
                    skipped_frames.append(
                        SkippedFrame(
                            token=source_image.stem,
                            timestamp_us=timestamp_ns // 1_000,
                            reason=reason,
                            stream_gaps_ms=gap_info,
                        )
                    )
                continue

            # All streams have valid brackets — interpolate with independent fractions
            nav0, nav1, nav_t   = brackets["NAV"]  # type: ignore[misc]
            imu0, imu1, imu_t   = brackets["IMU"]  # type: ignore[misc]
            veh0, veh1, veh_t   = brackets["VEHICLE_INFO"]  # type: ignore[misc]
            str0, str1, str_t   = brackets["VEHICLE_STEER"]  # type: ignore[misc]

            can_bus = build_can_bus_interp(
                nav0=nav0.values,
                nav1=nav1.values,
                nav_fraction=nav_t,
                imu0=imu0.values,
                imu1=imu1.values,
                imu_fraction=imu_t,
                vehicle0=veh0.values,
                vehicle1=veh1.values,
                vehicle_fraction=veh_t,
                steer0=str0.values,
                steer1=str1.values,
                steer_fraction=str_t,
                origin=origin,
            )
        else:
            # Legacy nearest-row synchronization mode
            matches = {
                name: nearest_row(rows, stream_timestamps[name], timestamp_ns)
                for name, rows in streams.items()
            }
            if any(match is None or match[1] > max_drift_ns for match in matches.values()):
                skipped += 1
                if record_skips:
                    drifts = {
                        name: (round(m[1] / 1_000_000, 1) if m else None)
                        for name, m in matches.items()
                    }
                    skipped_frames.append(
                        SkippedFrame(
                            token=source_image.stem,
                            timestamp_us=timestamp_ns // 1_000,
                            reason="drift_exceeded",
                            stream_gaps_ms=drifts,
                        )
                    )
                continue
            nav, imu, vehicle, steer = (matches[name][0].values for name in REQUIRED_STREAMS)  # type: ignore[index]
            can_bus = build_can_bus(nav, imu, vehicle, steer, origin)

        # can_bus shape: [18]
        # ego_motion shape: [2] -> [yaw_rate_radps, longitudinal_acceleration_mps2]
        ego_motion = np.asarray((can_bus[9], can_bus[7]), dtype=np.float32)

        samples.append(
            FrameSample(
                timestamp_us=timestamp_ns // 1_000,
                source_timestamp_us=timestamp_ns // 1_000,
                token=source_image.stem,
                source_image=source_image,
                image_path=image_root / source_image.name,
                can_bus=can_bus,
                ego_motion=ego_motion,
            )
        )
        if args.max_frames is not None and len(samples) >= args.max_frames:
            break
    return samples, skipped, skipped_frames


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def write_skip_log(
    args: argparse.Namespace,
    skipped_frames: list[SkippedFrame],
) -> None:
    """Write skip_log.jsonl alongside the episode file — one JSON object per skipped frame."""
    if not skipped_frames:
        return
    skip_log_path = args.output_root / "episodes" / args.split / f"{args.log_name}_skip_log.jsonl"
    skip_log_path.parent.mkdir(parents=True, exist_ok=True)
    with skip_log_path.open("w", encoding="utf-8") as f:
        for sf in skipped_frames:
            row = {
                "token": sf.token,
                "timestamp_us": sf.timestamp_us,
                "reason": sf.reason,
                "stream_gaps_ms": sf.stream_gaps_ms,
            }
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"[SKIP LOG] {len(skipped_frames)} skipped frames → {skip_log_path}")


def write_output(args: argparse.Namespace, samples: list[FrameSample]) -> None:
    output_root = args.output_root
    episode_path = output_root / "episodes" / args.split / f"{args.log_name}.npz"
    for sample in samples:
        materialize_image(sample.source_image, sample.image_path, args.image_mode)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        episode_path,
        timestamps_us=np.asarray([sample.timestamp_us for sample in samples], dtype=np.int64),
        source_timestamps_us=np.asarray(
            [sample.source_timestamp_us for sample in samples], dtype=np.int64
        ),
        tokens=np.asarray([sample.token for sample in samples]),
        image_paths=np.asarray([sample.image_path.relative_to(output_root).as_posix() for sample in samples]),
        can_bus=np.stack([sample.can_bus for sample in samples]),
        ego_motion=np.stack([sample.ego_motion for sample in samples]),
        ego_motion_names=np.asarray(EGO_MOTION_NAMES),
    )
    with (output_root / "manifest.jsonl").open("a", encoding="utf-8") as manifest_file:
        for index, sample in enumerate(samples):
            row = {
                "episode": episode_path.relative_to(output_root).as_posix(),
                "frame_index": index,
                "timestamp_us": sample.timestamp_us,
                "source_timestamp_us": sample.source_timestamp_us,
                "token": sample.token,
                "image_path": sample.image_path.relative_to(output_root).as_posix(),
                "ego_motion": sample.ego_motion.tolist(),
                "ego_motion_names": list(EGO_MOTION_NAMES),
            }
            manifest_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    dataset_info = {
        "format": "camo-jepa-v1",
        "source_dataset": "vf",
        "camera": args.camera,
        "ego_motion_names": list(EGO_MOTION_NAMES),
        "ego_motion_source": "can_bus[9], can_bus[7]",
        "can_bus_shape": [18],
        "image_mode": args.image_mode,
        "target_hz": args.target_hz,
        "resampling": "downsample_preserve_source_timestamp",
        "sync_mode": getattr(args, "sync_mode", "interpolation"),
        "max_interp_gap_ms": (
            args.max_interp_gap_ms if getattr(args, "sync_mode", "interpolation") == "interpolation" else None
        ),
        "max_drift_ms": (
            args.max_drift_ms if getattr(args, "sync_mode", "interpolation") == "nearest" else None
        ),
    }
    (output_root / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.vf_root = args.vf_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.max_drift_ms < 0:
        raise SystemExit("--max-drift-ms must be non-negative")
    if args.max_interp_gap_ms <= 0:
        raise SystemExit("--max-interp-gap-ms must be positive")
    if args.target_hz <= 0:
        raise SystemExit("--target-hz must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be positive")
    if not (args.vf_root / "CAMERA").is_dir() or not (args.vf_root / "OTHERS").is_dir():
        raise SystemExit("VF root must contain both CAMERA/ and OTHERS/")

    sync_mode = getattr(args, "sync_mode", "interpolation")
    if sync_mode == "interpolation":
        print(
            f"[SYNC] interpolation mode | max_interp_gap={args.max_interp_gap_ms:.0f} ms"
        )
    else:
        print(
            f"[SYNC] nearest-row mode | max_drift={args.max_drift_ms:.0f} ms"
        )

    samples, skipped, skipped_frames = build_samples(args)
    samples = resample_samples(samples, args.target_hz)
    if not samples:
        raise SystemExit("no camera frames met the telemetry synchronization tolerance")
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_output(args, samples)
        if getattr(args, "skip_log", False):
            write_skip_log(args, skipped_frames)
    mode = "validated" if args.dry_run else "converted"
    print(f"[DONE] {mode} {len(samples)} synchronized frames, {skipped} skipped frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())