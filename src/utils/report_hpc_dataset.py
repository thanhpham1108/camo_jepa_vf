#!/usr/bin/env python3
"""Inventory CaMo-JEPA data and package VF visualization videos for a pipeline run.

Run on the HPC through the Azure pipeline with noGpu=true and entryCommand:
    python3 src/utils/report_hpc_dataset.py

The default output is $RUN_DIR/dataset_report. A local audit CSV can also be
rendered with --audit-csv, but that mode is explicitly marked as a snapshot.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
REQUIRED_STREAMS = ("NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER")
SOURCE_NAMES = ("datasets_raw", "data260926")


def args_parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", action="append", type=Path,
                   help="Raw root (repeat for multiple sources). Defaults to the two HPC roots.")
    p.add_argument("--converted-root", type=Path,
                   default=Path("/dataset/camo_jepa/datasets_converted_interp"))
    p.add_argument("--run-dir", type=Path,
                   default=Path(os.environ.get("RUN_DIR", "/workspaces/result/run_dir")))
    p.add_argument("--video-search-root", action="append", type=Path,
                   help="Additional folder to search for VF videos (repeatable).")
    p.add_argument("--video", action="append", type=Path,
                   help="Exact visualization video to copy (repeatable).")
    p.add_argument("--audit-csv", type=Path,
                   help="Render a saved audit CSV instead of scanning HPC paths.")
    p.add_argument("--no-copy-videos", action="store_true")
    return p.parse_args()


def source_and_group(path: Path, roots: list[Path]) -> tuple[str, str]:
    for idx, root in enumerate(roots):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        source = SOURCE_NAMES[idx] if idx < len(SOURCE_NAMES) else root.name
        parts = rel.parts[:-1]  # everything above the scene folder
        return source, "/".join(parts) if parts else "(root)"
    return "unknown", "unknown"


def timestamp_ns(stem: str) -> int | None:
    match = re.match(r"^(\d{9,11})[-_](\d+)", stem)
    if not match:
        return None
    return int(match.group(1)) * 1_000_000_000 + int((match.group(2) + "000000000")[:9])


def camera_measure(scene: Path) -> tuple[str | None, int, float, list[str]]:
    camera_dir = scene / "CAMERA"
    names = sorted(p.name for p in camera_dir.iterdir() if p.is_dir()) if camera_dir.is_dir() else []
    chosen = next((c for c in ("CAM_P_F", "CAM_F") if c in names), None)
    if chosen is None:
        return None, 0, 0.0, names
    files = [p for p in (camera_dir / chosen).iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".png")]
    if not files:
        return chosen, 0, 0.0, names
    first, last = min(files, key=lambda p: p.name), max(files, key=lambda p: p.name)
    start, end = timestamp_ns(first.stem), timestamp_ns(last.stem)
    duration = (end - start) / 1e9 if start is not None and end is not None and end >= start else len(files) / 30.0
    return chosen, len(files), duration, names


def scene_record(scene: Path, roots: list[Path]) -> dict:
    source, group = source_and_group(scene, roots)
    camera, frames, duration, cameras = camera_measure(scene)
    others = scene / "OTHERS"
    streams = {name: (others / name).is_dir() and any((others / name).glob("*.csv"))
               for name in REQUIRED_STREAMS}
    return {
        "name": scene.name, "path": str(scene), "source": source, "group": group,
        "camera": camera, "cameras": cameras, "raw_frames": frames,
        "raw_duration_sec": round(duration, 3),
        "has_camera": (scene / "CAMERA").is_dir(), "has_others": others.is_dir(),
        "streams": streams,
        "raw_valid": camera is not None and frames > 0 and all(streams.values()),
    }


def scan_tree(root: Path, scenes: set[Path], videos: set[Path],
              folder_stats: dict[str, dict], errors: list[str], manifest=None) -> None:
    for directory, dirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        dirs.sort()
        if "CAMERA" in dirs or "OTHERS" in dirs:
            scenes.add(current)
        stat = folder_stats.setdefault(str(current), {"files_direct": 0, "bytes_direct": 0, "extensions": {}})
        ext = Counter()
        for name in files:
            path = current / name
            try:
                size = path.stat().st_size
            except OSError as exc:
                errors.append(f"stat {path}: {exc}")
                continue
            stat["files_direct"] += 1
            stat["bytes_direct"] += size
            suffix = path.suffix.lower() or "[none]"
            ext[suffix] += 1
            if manifest is not None:
                manifest.write(f"{path}\t{size}\t{suffix}\t{int(path.is_symlink())}\n")
            if suffix in VIDEO_EXTS:
                videos.add(path)
        stat["extensions"] = dict(ext)


def converted_records(root: Path, errors: list[str]) -> tuple[dict, dict[str, dict]]:
    try:
        import numpy as np
    except ImportError:
        errors.append("numpy is unavailable; converted timestamp spans could not be read")
        np = None
    splits = {}
    episode_index = {}
    for split in ("train", "val"):
        folder = root / "episodes" / split
        files = sorted(folder.glob("*.npz")) if folder.is_dir() else []
        frames = 0
        duration = 0.0
        for path in files:
            count = None
            span = None
            if np is not None:
                try:
                    with np.load(path, allow_pickle=False) as episode:
                        stamps = episode["timestamps_us"]
                        count = len(stamps)
                        span = float((stamps[-1] - stamps[0]) / 1e6) if count > 1 else 0.0
                    frames += count
                    duration += span
                except Exception as exc:
                    errors.append(f"read {path}: {exc}")
            if path.stem in episode_index:
                errors.append(f"duplicate episode name across splits: {path.stem}")
            episode_index[path.stem] = {"split": split, "frames": count, "duration_sec": span}
        splits[split] = {"episodes": len(files), "frames": frames if np is not None else None,
                         "duration_sec": round(duration, 3) if np is not None else None}
    return splits, episode_index


def group_summary(scenes: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for s in scenes:
        grouped[(s["source"], s["group"])].append(s)
    result = []
    for (source, group), items in sorted(grouped.items()):
        result.append({
            "source": source, "group": group, "scenes": len(items),
            "raw_valid": sum(s["raw_valid"] for s in items),
            "converted": sum(s.get("converted", False) for s in items),
            "raw_frames": sum(s["raw_frames"] for s in items),
            "raw_duration_sec": round(sum(s["raw_duration_sec"] for s in items), 3),
            "converted_frames": sum(s.get("converted_frames") or 0 for s in items),
            "converted_duration_sec": round(sum(s.get("converted_duration_sec") or 0 for s in items), 3),
            "cameras": sorted({c for s in items for c in s.get("cameras", [])}),
        })
    return result


def load_snapshot(path: Path, roots: list[Path]) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scenes = []
    for r in rows:
        scene = Path(r["scene_path"])
        source, group = source_and_group(scene, roots)
        scenes.append({
            "name": r["scene_name"], "path": str(scene), "source": source, "group": group,
            "camera": r["camera"], "cameras": [], "raw_frames": int(r["raw_frames"]),
            "raw_duration_sec": float(r["raw_duration_sec"]),
            "raw_valid": r["status"] == "CONVERTED", "converted": r["status"] == "CONVERTED",
            "split": None, "converted_frames": int(r["converted_frames"]),
            "converted_duration_sec": float(r["converted_duration_sec"]),
            "status": r["status"], "reason": r["reason"],
        })
    return scenes, []


def video_resource(video: Path, roots: list[Path]) -> str:
    for idx, root in enumerate(roots):
        try:
            rel = video.relative_to(root)
        except ValueError:
            continue
        source = SOURCE_NAMES[idx] if idx < len(SOURCE_NAMES) else root.name
        name = video.name.lower()
        if source == "data260926":
            for group in ("VF6_01_02", "VF6_03"):
                if group in rel.parts or group.lower() in name or (group == "VF6_01_02" and "vf6_01" in name):
                    return f"data260926/data/{group}"
        if source == "datasets_raw" and len(rel.parts) > 2 and rel.parts[0] == "rawdata_260924":
            return "/".join((source, rel.parts[0], rel.parts[1]))
        return source
    name = video.name.lower()
    if "vf6_03" in name:
        return "data260926/data/VF6_03 (inferred from filename)"
    if "vf6_01" in name or "vf6_02" in name:
        return "VF6_01/02 (inferred from filename)"
    return "unassigned"


def select_and_copy_videos(videos: set[Path], explicit: list[Path], roots: list[Path],
                           output: Path, no_copy: bool, errors: list[str]) -> tuple[list[dict], str]:
    candidates = sorted(videos | set(explicit))
    tagged = [p for p in candidates if re.search(r"visual|viz|overview|demo|preview", str(p), re.I)]
    selected = sorted(set(explicit) | set(tagged))
    if not selected and len(candidates) == 1:
        selected = candidates
    status = "copied" if selected else ("ambiguous_candidates" if candidates else "not_found")
    records = []
    for path in candidates:
        record = {"source": str(path), "resource": video_resource(path, roots), "selected": path in selected}
        try:
            record["bytes"] = path.stat().st_size
            if path in selected and not no_copy:
                import hashlib
                digest = hashlib.sha256(str(path).encode()).hexdigest()[:10]
                target = output / "visualizations" / f"{path.stem}_{digest}{path.suffix.lower()}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                if target.stat().st_size != record["bytes"]:
                    raise OSError("copied size differs from source")
                record["copied_to"] = str(target)
        except OSError as exc:
            errors.append(f"video {path}: {exc}")
            record["error"] = str(exc)
        records.append(record)
    if no_copy:
        status = "copy_disabled"
    elif any("error" in r for r in records):
        status = "copy_error"
    return records, status


def file_rollup(stats: dict[str, dict], root: Path) -> dict:
    items = [s for p, s in stats.items() if p == str(root) or p.startswith(str(root) + os.sep)]
    extensions = Counter()
    for item in items:
        extensions.update(item["extensions"])
    return {"directories": len(items), "files": sum(i["files_direct"] for i in items),
            "bytes": sum(i["bytes_direct"] for i in items), "extensions": dict(extensions.most_common())}


def md_report(data: dict) -> str:
    groups = data["groups"]
    lines = ["# CaMo-JEPA dataset inventory", "",
             f"Generated: {data['generated_utc']} · Mode: **{data['mode']}**", "",
             "## Scope and evidence", "",
             f"Raw roots: `{data['raw_roots'][0]}`, `{data['raw_roots'][1]}`. "
             f"Converted root: `{data['converted_root']}`.", ""]
    if data["mode"] == "saved_audit_snapshot":
        lines += ["This report is reconstructed from the saved per-scene audit CSV. It is **not** a live HPC scan. "
                  "File inventory, current train/val split, labels, and video availability require the HPC run.", ""]
    lines += ["## What the data contains", "",
              f"There are **{len(data['raw_roots'])} primary raw source roots** and "
              f"**{len(groups)} leaf resource groups** in this inventory. "
              "The groups below reflect the actual scene paths in the audit, including three subgroups "
              "inside `datasets_raw`.", "",
              "Raw scene folders contain `CAMERA/<camera>/` image sequences and `OTHERS/` telemetry "
              "(`NAV`, `IMU`, `VEHICLE_INFO`, `VEHICLE_STEER`). "
              "`data260926/label/` holds CVAT 3D annotations; its file count is in the live inventory. "
              "The Phase 2 converted folder has `episodes/{train,val}/*.npz`, `images/{train,val}/`, "
              "`manifest.jsonl`, and `dataset_info.json`. Episode arrays include camera timestamps, "
              "image paths, 18-value CAN bus, and ego motion.", "", "## Dataset tree", "", "```text",
              "/dataset/", "├── camo_jepa/", "│   ├── datasets_raw/"]
    raw_groups = [g for g in groups if g["source"] == "datasets_raw"]
    for i, g in enumerate(raw_groups):
        branch = "└──" if i == len(raw_groups) - 1 else "├──"
        lines.append(f"│   │   {branch} {g['group']}/  [{g['scenes']} scenes, {g['raw_duration_sec']/3600:.3f} raw h]")
    lines += ["│   └── datasets_converted_interp/", "│       ├── episodes/{train,val}/",
              "│       ├── images/{train,val}/", "│       ├── manifest.jsonl", "│       └── dataset_info.json",
              "└── data260926/"]
    for g in groups:
        if g["source"] == "data260926":
            lines.append(f"    ├── {g['group']}/  [{g['scenes']} scenes, {g['raw_duration_sec']/3600:.3f} raw h]")
    lines += ["    └── label/  [CVAT annotations; count in live file inventory]", "```", "",
              "## Primary raw sources", "",
              "| Source root | Scenes | Converted | Raw hours | Converted span hours |",
              "| :--- | ---: | ---: | ---: | ---: |"]
    for source in SOURCE_NAMES:
        items = [g for g in groups if g["source"] == source]
        lines.append(f"| `{source}` | {sum(g['scenes'] for g in items):,} | "
                     f"{sum(g['converted'] for g in items):,} | "
                     f"{sum(g['raw_duration_sec'] for g in items)/3600:.3f} | "
                     f"{sum(g['converted_duration_sec'] for g in items)/3600:.3f} |")
    lines += ["", 
              "## Scenes and duration by raw resource", "",
              "| Raw resource | Scenes | Converted | Raw hours | Converted span hours | Raw camera frames |",
              "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        lines.append(f"| `{g['source']}/{g['group']}` | {g['scenes']:,} | {g['converted']:,} | "
                     f"{g['raw_duration_sec']/3600:.3f} | {g['converted_duration_sec']/3600:.3f} | {g['raw_frames']:,} |")
    total = data["totals"]
    lines += [f"| **Total** | **{total['scenes']:,}** | **{total['converted']:,}** | "
              f"**{total['raw_duration_sec']/3600:.3f}** | **{total['converted_duration_sec']/3600:.3f}** | "
              f"**{total['raw_frames']:,}** |", "",
              "Durations sum each scene's first-to-last selected front-camera timestamp and each converted "
              "episode's first-to-last timestamp. They are recording spans, not continuous driving time. "
              "The raw and converted frame counts are at different sampling rates and should not be subtracted "
              "to infer lost duration.", ""]
    if data["mode"] == "live_hpc_scan":
        lines += ["## File inventory", "",
                  "Counts include files under each scanned root. Directory symlinks are not followed; "
                  "size is apparent file size and can count targets of file symlinks more than once. "
                  "`file_manifest.tsv.gz` lists every scanned file path, size, extension, and symlink flag; "
                  "`folder_inventory.tsv.gz` lists per-folder counts.", "",
                  "| Root | Directories | Files | Size (GiB) | Top extensions |",
                  "| :--- | ---: | ---: | ---: | :--- |"]
        for path, stat in data["file_inventory"].items():
            exts = ", ".join(f"{k}: {v:,}" for k, v in list(stat["extensions"].items())[:6])
            lines.append(f"| `{path}` | {stat['directories']:,} | {stat['files']:,} | "
                         f"{stat['bytes']/1024**3:.2f} | {exts} |")
        lines += ["", "## Converted split", "",
                  "| Split | Episodes | Frames | Sum of episode spans (h) |", "| :--- | ---: | ---: | ---: |"]
        for split, stat in data["converted_splits"].items():
            frames = f"{stat['frames']:,}" if stat["frames"] is not None else "unavailable"
            hours = f"{stat['duration_sec']/3600:.3f}" if stat["duration_sec"] is not None else "unavailable"
            lines.append(f"| {split} | {stat['episodes']:,} | {frames} | {hours} |")
        lines.append("")
    lines += ["## Visualization videos", ""]
    if data["videos"]:
        lines += ["| Source video | Raw resource | Copied artifact |", "| :--- | :--- | :--- |"]
        for v in data["videos"]:
            lines.append(f"| `{v['source']}` | {v['resource']} | "
                         f"`{v.get('copied_to', 'not copied')}` |")
    elif data["mode"] == "saved_audit_snapshot":
        lines.append("The saved audit CSV contains no video inventory. Run this script on HPC to search "
                     "the project folders, or provide `--video /absolute/path/to/file.mp4`.")
    else:
        lines.append("No video was found in the scanned paths. Provide `--video /absolute/path/to/file.mp4` "
                     "or `--video-search-root /path/to/VF/materials` on the HPC run.")
    lines += ["", f"Video copy status: **{data['video_status']}**. "
              "If several unrelated videos are found, only names or paths containing visual/viz/overview/demo/preview "
              "are copied automatically; use `--video` to select the intended video.", "",
              "| Raw resource | Visualization found |", "| :--- | :--- |"]
    for g in groups:
        key = f"{g['source']}/{g['group']}"
        matches = [v for v in data["videos"] if key in v["source"] or key in v["resource"]]
        state = ", ".join(f"`{Path(v['source']).name}`" for v in matches) if matches else "Not yet verified"
        lines.append(f"| `{key}` | {state} |")
    lines += ["",
              "## Data quality and limitations", ""]
    skipped = [s for s in data["scenes"] if not s["converted"]]
    lines.append(f"{len(skipped)} scene(s) have no matching converted episode.")
    for s in skipped[:20]:
        lines.append(f"- `{s['path']}` — {s.get('reason', 'missing/invalid raw data or episode')}")
    if not skipped:
        lines.append("All discovered raw scenes have matching converted episode names.")
    lines += ["", "The older README quotes 11.31 converted hours, while the saved audit CSV sums to 9.05 "
              "hours from episode timestamp spans. Its per-source raw figures also conflict with the audit "
              "CSV: `VF6_01_02` is 5.717 h (README 5.56 h) and `VF6_03` is 3.185 h "
              "(README 2.34 h). The audited source figures sum to 8.902 h, consistent with the "
              "8.90 h source total. Use the live HPC report before presenting either converted-hours "
              "figure as current.", ""]
    if data["errors"]:
        lines += ["## Scan errors", ""] + [f"- {e}" for e in data["errors"][:50]]
        if len(data["errors"]) > 50:
            lines.append(f"- ... {len(data['errors'])-50} more errors in JSON")
        lines.append("")
    lines += ["## Reproduce on HPC", "",
              "Set Azure pipeline `noGpu=true` and `entryCommand` to:", "",
              "```bash", "python3 src/utils/report_hpc_dataset.py", "```", "",
              "The live report, JSON inventory, and selected visualization videos are written under "
              "`$RUN_DIR/dataset_report/` and packaged with the run artifact. If VF supplied a video "
              "outside the scanned project folders, add `--video /absolute/path/to/video.mp4` "
              "or `--video-search-root /absolute/path/to/folder` to the entry command.", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = args_parse()
    roots = args.raw_root or [Path("/dataset/camo_jepa/datasets_raw"), Path("/dataset/data260926")]
    roots = [p.resolve() for p in roots]
    output = args.run_dir / "dataset_report"
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    videos: set[Path] = set()
    stats: dict[str, dict] = {}
    if args.audit_csv:
        scenes, errors = load_snapshot(args.audit_csv, roots)
        mode = "saved_audit_snapshot"
        splits = {}
    else:
        mode = "live_hpc_scan"
        scene_paths: set[Path] = set()
        # Scan the full relevant project trees once. This includes label/, models/,
        # legacy data and any visualization video outside the two raw roots.
        project_roots = [roots[0].parent, roots[1]] if len(roots) >= 2 else roots
        for root in roots + [args.converted_root]:
            if not root.is_dir():
                errors.append(f"required root missing: {root}")
        with gzip.open(output / "file_manifest.tsv.gz", "wt", encoding="utf-8") as manifest:
            manifest.write("path\tbytes\textension\tis_symlink\n")
            for root in project_roots:
                if not root.is_dir():
                    continue
                found: set[Path] = set()
                scan_tree(root, found, videos, stats, errors, manifest)
                scene_paths.update(p for p in found if any(p == raw or raw in p.parents for raw in roots))
            for root in args.video_search_root or []:
                if root.is_dir():
                    scan_tree(root, set(), videos, stats, errors, manifest)
                else:
                    errors.append(f"video search root missing: {root}")
        with gzip.open(output / "folder_inventory.tsv.gz", "wt", encoding="utf-8") as folders:
            folders.write("path\tfiles_direct\tbytes_direct\textensions_json\n")
            for path, stat in sorted(stats.items()):
                folders.write(f"{path}\t{stat['files_direct']}\t{stat['bytes_direct']}\t"
                              f"{json.dumps(stat['extensions'], separators=(',', ':'))}\n")
        scenes = [scene_record(p, roots) for p in sorted(scene_paths)]
        splits, index = converted_records(args.converted_root, errors)
        for scene in scenes:
            ep = index.get(scene["name"])
            scene["converted"] = ep is not None
            scene["split"] = ep["split"] if ep else None
            scene["converted_frames"] = ep["frames"] if ep else 0
            scene["converted_duration_sec"] = ep["duration_sec"] if ep else 0.0
            if not scene["raw_valid"]:
                scene["reason"] = "missing front camera/images or required OTHERS telemetry"
            elif ep is None:
                scene["reason"] = "no matching .npz episode"
        unmatched = set(index) - {s["name"] for s in scenes}
        if unmatched:
            errors.append(f"{len(unmatched)} converted episode names have no discovered raw scene")
    groups = group_summary(scenes)
    video_records, video_status = select_and_copy_videos(videos, args.video or [], roots, output,
                                                         args.no_copy_videos, errors)
    if mode == "saved_audit_snapshot" and not args.video:
        video_status = "not_scanned"
    totals = {"scenes": len(scenes), "converted": sum(s["converted"] for s in scenes),
              "raw_frames": sum(s["raw_frames"] for s in scenes),
              "raw_duration_sec": round(sum(s["raw_duration_sec"] for s in scenes), 3),
              "converted_duration_sec": round(sum(s.get("converted_duration_sec") or 0 for s in scenes), 3)}
    data = {"generated_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "raw_roots": [str(r) for r in roots], "converted_root": str(args.converted_root),
            "groups": groups, "totals": totals, "converted_splits": splits,
            "file_inventory": {str(p): file_rollup(stats, p) for p in
                               (project_roots + roots + [args.converted_root] +
                                ([roots[1] / "label"] if len(roots) > 1 else []))
                               if p.is_dir()} if stats else {},
            "videos": video_records, "video_status": video_status,
            "scenes": scenes, "errors": errors}
    (output / "dataset_inventory.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    (output / "dataset_report.md").write_text(md_report(data))
    print(f"Report: {output / 'dataset_report.md'}")
    print(f"JSON: {output / 'dataset_inventory.json'}")
    print(f"Scenes: {totals['scenes']}; converted: {totals['converted']}; raw hours: {totals['raw_duration_sec']/3600:.3f}")
    print(f"Videos: {len(video_records)} found; status: {video_status}")
    if errors:
        print(f"Warnings: {len(errors)} (see JSON)", file=sys.stderr)
    return 2 if any(e.startswith("required root missing") for e in errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
