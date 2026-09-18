import argparse
import glob
from pathlib import Path
import numpy as np

def audit_dataset(dataset_dir: Path):
    episodes = glob.glob(str(dataset_dir / "episodes" / "train" / "*.npz"))
    if not episodes:
        print(f"No .npz episodes found in {dataset_dir}/episodes/train/")
        return

    total_frames = 0
    speeds_kmh = []
    accel_x = []
    accel_y = []
    yaw_rates = []
    steer_angles = []
    time_diffs = []
    
    # CAN Bus indices based on convert_vf_to_camo.py
    IDX_ACCEL_X = 7
    IDX_ACCEL_Y = 8
    IDX_YAW_RATE = 9
    IDX_VE = 10
    IDX_VN = 11
    IDX_STEER = 14

    print(f"Scanning {len(episodes)} episodes...")

    for ep_path in episodes:
        with np.load(ep_path) as data:
            can_bus = data['can_bus']
            ts = data['timestamps_us']
            
            total_frames += len(ts)
            
            # Compute speed (km/h)
            ve = can_bus[:, IDX_VE]
            vn = can_bus[:, IDX_VN]
            speed_mps = np.sqrt(ve**2 + vn**2)
            speeds_kmh.extend(speed_mps * 3.6)
            
            # Accelerations and Steering
            accel_x.extend(can_bus[:, IDX_ACCEL_X])
            accel_y.extend(can_bus[:, IDX_ACCEL_Y])
            yaw_rates.extend(can_bus[:, IDX_YAW_RATE])
            steer_angles.extend(can_bus[:, IDX_STEER])
            
            # Time diffs
            if len(ts) > 1:
                diffs = np.diff(ts) / 1e6 # Convert to seconds
                time_diffs.extend(diffs)

    # Calculate statistics
    speeds = np.array(speeds_kmh)
    steers = np.array(steer_angles)
    
    stationary_ratio = np.mean(speeds < 1.0) * 100
    sharp_turn_ratio = np.mean(np.abs(steers) > 15.0) * 100
    
    print("\n" + "="*50)
    print("📊 DATASET AUDIT REPORT")
    print("="*50)
    print(f"Total Frames Analyzed: {total_frames:,} (~{total_frames / 10 / 3600:.2f} hours at 10Hz)")
    
    print("\n🏎️  1. VELOCITY & ACCELERATION")
    print(f"  - Speed: Mean = {np.mean(speeds):.1f} km/h | Max = {np.max(speeds):.1f} km/h")
    print(f"  - Stationary Time (< 1 km/h): {stationary_ratio:.1f}%")
    print(f"  - Accel X (Longitudinal): Mean = {np.mean(accel_x):.2f} m/s² | Std = {np.std(accel_x):.2f}")
    print(f"  - Accel Y (Lateral):      Mean = {np.mean(accel_y):.2f} m/s² | Std = {np.std(accel_y):.2f}")
    
    print("\n🔄 2. STEERING & YAW")
    print(f"  - Steering Angle: Mean Abs = {np.mean(np.abs(steers)):.1f}° | Max Abs = {np.max(np.abs(steers)):.1f}°")
    print(f"  - Sharp Turns (> 15°):    {sharp_turn_ratio:.1f}%")
    print(f"  - Yaw Rate:       Mean Abs = {np.mean(np.abs(yaw_rates)):.2f} rad/s")
    
    print("\n⏱️  3. TEMPORAL SYNCHRONIZATION")
    time_diffs = np.array(time_diffs)
    print(f"  - Delta T: Mean = {np.mean(time_diffs)*1000:.1f} ms | Target = 100.0 ms")
    print(f"  - Max Frame Gap:  {np.max(time_diffs)*1000:.1f} ms")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit CaMo-JEPA dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/dataset/camo_jepa"), help="Path to converted dataset")
    args = parser.parse_args()
    audit_dataset(args.dataset_dir)
