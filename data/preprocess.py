import os
import sys
import pathlib
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scipy.io
import numpy as np
import pandas as pd
import torch

from utils.paths import Paths

CELL_RADIUS = 0.009    # 18650 cell radius (9 mm)
CELL_HEIGHT = 0.065    # 18650 cell height (65 mm)
CELL_VOLUME = np.pi * (CELL_RADIUS ** 2) * CELL_HEIGHT  # ~1.654e-5 m^3
EOL_CAPACITY_THRESHOLD = 1.4  # Rated 2.0 Ahr -> 30% fade threshold (1.4 Ahr)


def parse_date_vector(date_vec: np.ndarray) -> str:
    """Converts MATLAB date vector [Y, M, D, H, M, S] into ISO format string."""
    try:
        vec = date_vec.flatten()
        if len(vec) >= 6:
            return f"{int(vec[0]):04d}-{int(vec[1]):02d}-{int(vec[2]):02d} {int(vec[3]):02d}:{int(vec[4]):02d}:{int(vec[5]):02d}"
    except Exception:
        pass
    return ""


def load_battery_mat(mat_path: str) -> Tuple[str, List[Any]]:
    """Loads a NASA battery .mat file and extracts the cycle structure."""
    mat = scipy.io.loadmat(mat_path)
    battery_id = [k for k in mat.keys() if not k.startswith("__")][0]
    cycles = list(mat[battery_id][0, 0]["cycle"][0])
    return battery_id, cycles


def extract_cycle_summary(cycles: List[Any], battery_id: str) -> pd.DataFrame:
    """Extracts cycle-level metadata, capacity fade, thermal stats, and defect labels."""
    records = []
    discharge_idx = 0
    charge_idx = 0
    impedance_idx = 0

    for cycle_idx, c in enumerate(cycles):
        ctype = str(c["type"][0])
        ambient_temp = float(c["ambient_temperature"][0][0])
        start_time_str = parse_date_vector(c["time"][0])
        data_struct = c["data"][0, 0]

        sub_idx = 0
        if ctype == "discharge":
            discharge_idx += 1
            sub_idx = discharge_idx
        elif ctype == "charge":
            charge_idx += 1
            sub_idx = charge_idx
        elif ctype == "impedance":
            impedance_idx += 1
            sub_idx = impedance_idx

        capacity = np.nan
        duration_s = 0.0
        min_temp_c = np.nan
        max_temp_c = np.nan
        delta_temp_c = np.nan
        avg_voltage_v = np.nan
        avg_current_a = np.nan
        re_ohm = np.nan
        rct_ohm = np.nan
        num_samples = 0

        names = data_struct.dtype.names or ()

        if "Time" in names and data_struct["Time"].size > 0:
            times = data_struct["Time"].flatten()
            duration_s = float(times[-1]) if len(times) > 0 else 0.0
            num_samples = len(times)

        if "Capacity" in names and data_struct["Capacity"].size > 0:
            capacity = float(data_struct["Capacity"][0, 0])

        if "Temperature_measured" in names and data_struct["Temperature_measured"].size > 0:
            temps = data_struct["Temperature_measured"].flatten()
            min_temp_c = float(np.min(temps))
            max_temp_c = float(np.max(temps))
            delta_temp_c = max_temp_c - min_temp_c

        if "Voltage_measured" in names and data_struct["Voltage_measured"].size > 0:
            avg_voltage_v = float(np.mean(data_struct["Voltage_measured"].flatten()))

        if "Current_measured" in names and data_struct["Current_measured"].size > 0:
            avg_current_a = float(np.mean(data_struct["Current_measured"].flatten()))

        if "Re" in names and data_struct["Re"].size > 0:
            re_ohm = float(data_struct["Re"][0, 0])

        if "Rct" in names and data_struct["Rct"].size > 0:
            rct_ohm = float(data_struct["Rct"][0, 0])

        # Defect/degraded state labeling:
        # A cell is considered degraded/defective when its capacity drops below 1.4 Ahr (EOL threshold)
        is_defective = 1 if (not np.isnan(capacity) and capacity < EOL_CAPACITY_THRESHOLD) else 0

        records.append({
            "battery_id": battery_id,
            "cycle_id": cycle_idx,
            "cycle_type": ctype,
            "operation_idx": sub_idx,
            "start_time": start_time_str,
            "ambient_temp_c": ambient_temp,
            "duration_s": duration_s,
            "capacity_ah": capacity,
            "min_temp_c": min_temp_c,
            "max_temp_c": max_temp_c,
            "delta_temp_c": delta_temp_c,
            "avg_voltage_v": avg_voltage_v,
            "avg_current_a": avg_current_a,
            "re_ohm": re_ohm,
            "rct_ohm": rct_ohm,
            "num_samples": num_samples,
            "is_defective": is_defective
        })

    df = pd.DataFrame(records)
    # Forward-fill capacity and defect status to impedance and charge cycles
    df["capacity_interpolated"] = df["capacity_ah"].interpolate(method="linear").bfill().ffill()
    df["is_defective"] = (df["capacity_interpolated"] < EOL_CAPACITY_THRESHOLD).astype(int)
    return df


def extract_timeseries_data(cycles: List[Any], battery_id: str,
                            cycle_types: Tuple[str, ...] = ("discharge",)) -> pd.DataFrame:
    """Extracts continuous time-series sensor samples across cycles."""
    dfs = []
    discharge_idx = 0

    for cycle_idx, c in enumerate(cycles):
        ctype = str(c["type"][0])
        if ctype not in cycle_types:
            continue

        if ctype == "discharge":
            discharge_idx += 1

        d = c["data"][0, 0]
        names = d.dtype.names or ()
        if "Time" not in names or "Temperature_measured" not in names:
            continue

        times = d["Time"].flatten()
        temps_c = d["Temperature_measured"].flatten()
        volts = d["Voltage_measured"].flatten() if "Voltage_measured" in names else np.zeros_like(times)
        currs = d["Current_measured"].flatten() if "Current_measured" in names else np.zeros_like(times)
        cap = float(d["Capacity"][0, 0]) if ("Capacity" in names and d["Capacity"].size > 0) else np.nan

        cycle_df = pd.DataFrame({
            "battery_id": battery_id,
            "cycle_id": cycle_idx,
            "discharge_cycle_idx": discharge_idx,
            "cycle_type": ctype,
            "time_s": times,
            "voltage_v": volts,
            "current_a": currs,
            "temp_c": temps_c,
            "temp_k": temps_c + 273.15,
            "capacity_ah": cap,
            "is_defective": 1 if (not np.isnan(cap) and cap < EOL_CAPACITY_THRESHOLD) else 0
        })
        dfs.append(cycle_df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def format_pinn_surface_dataset(timeseries_df: pd.DataFrame,
                                radius: float = CELL_RADIUS,
                                height: float = CELL_HEIGHT,
                                n_azimuth_sensors: int = 4,
                                downsample_stride: int = 5) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """
    Transforms 1D surface temperature time-series into 4D spatial-temporal inputs [x, y, z, t]
    for the PINN Teacher model.
    
    Sensors are arranged around the outer mantle of the 18650 cylinder:
      - Radial position r = radius (0.009 m)
      - Azimuthal positions theta_i = 2 * pi * i / n_azimuth_sensors
      - Height z = height / 2 = 0.0325 m
      - Coordinates: x = r * cos(theta), y = r * sin(theta), z, t
      - Target: Temperature in Kelvin T_K
    """
    df_sub = timeseries_df.iloc[::downsample_stride].copy()
    theta_angles = [2.0 * np.pi * i / n_azimuth_sensors for i in range(n_azimuth_sensors)]
    z_pos = height / 2.0  # Mid-height thermocouple mounting

    surface_records = []
    for theta in theta_angles:
        x_val = radius * np.cos(theta)
        y_val = radius * np.sin(theta)

        # Estimate volumetric heat generation Q_gen ~ I^2 * R_int / V_cell
        r_int_nominal = 0.1  # ~100 mOhm
        q_gen_est = (np.abs(df_sub["current_a"]) ** 2 * r_int_nominal) / CELL_VOLUME

        sub = pd.DataFrame({
            "battery_id": df_sub["battery_id"].values,
            "cycle_id": df_sub["cycle_id"].values,
            "discharge_cycle_idx": df_sub["discharge_cycle_idx"].values,
            "is_defective": df_sub["is_defective"].values,
            "x": x_val,
            "y": y_val,
            "z": z_pos,
            "t": df_sub["time_s"].values,
            "temp_k": df_sub["temp_k"].values,
            "temp_c": df_sub["temp_c"].values,
            "voltage_v": df_sub["voltage_v"].values,
            "current_a": df_sub["current_a"].values,
            "q_gen_w_m3": q_gen_est.values
        })
        surface_records.append(sub)

    pinn_surface_df = pd.concat(surface_records, ignore_index=True)

    # Tensor representations: coords (N, 4) and targets (N, 1)
    coords_tensor = torch.tensor(
        pinn_surface_df[["x", "y", "z", "t"]].values, dtype=torch.float32
    )
    targets_tensor = torch.tensor(
        pinn_surface_df[["temp_k"]].values, dtype=torch.float32
    )

    return pinn_surface_df, coords_tensor, targets_tensor


def generate_collocation_dataset(n_points: int = 50000,
                                 radius: float = CELL_RADIUS,
                                 height: float = CELL_HEIGHT,
                                 t_max: float = 3600.0,
                                 seed: int = 42) -> torch.Tensor:
    """
    Generates interior collocation points (x, y, z, t) within the 18650 cylinder volume
    for physics-informed residual evaluation during training.
    """
    np.random.seed(seed)
    r = np.sqrt(np.random.uniform(0, radius**2, n_points))
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(0, height, n_points)
    t = np.random.uniform(0, t_max, n_points)

    coll_coords = np.stack([x, y, z, t], axis=-1)
    return torch.tensor(coll_coords, dtype=torch.float32)


def process_all_nasa_batteries(raw_dir: Optional[str] = None,
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Full preprocessing pipeline:
      1. Parses all 4 NASA battery datasets (B0005, B0006, B0007, B0018).
      2. Extracts cycle summaries with capacity fade and defect labels.
      3. Extracts continuous discharge time-series.
      4. Projects surface measurements to 4D spatial-temporal coordinates for PINN.
      5. Generates interior collocation dataset.
      6. Saves all processed artifacts to output_dir.
    """
    if raw_dir is None:
        p1 = Paths.data_raw() / "1. BatteryAgingARC-FY08Q4"
        if (p1 / "B0005.mat").exists():
            raw_dir = str(p1)
        elif (Paths.data_raw() / "B0005.mat").exists():
            raw_dir = str(Paths.data_raw())
        else:
            raw_dir = str(p1)
    if output_dir is None:
        output_dir = str(Paths.data_processed())
    os.makedirs(output_dir, exist_ok=True)

    battery_files = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]
    all_summaries = []
    all_timeseries = []

    print(f"Loading and processing {len(battery_files)} battery files from: {raw_dir}")

    for fname in battery_files:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: File {fpath} not found. Skipping.")
            continue

        bid, cycles = load_battery_mat(fpath)
        print(f" -> Processed {bid}: {len(cycles)} total cycles")

        summary_df = extract_cycle_summary(cycles, bid)
        timeseries_df = extract_timeseries_data(cycles, bid, cycle_types=("discharge",))

        all_summaries.append(summary_df)
        all_timeseries.append(timeseries_df)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_timeseries = pd.concat(all_timeseries, ignore_index=True)

    # 4D PINN surface dataset
    pinn_surf_df, surf_coords, surf_temps = format_pinn_surface_dataset(
        combined_timeseries, downsample_stride=5
    )

    # 4D PINN interior collocation points
    t_max_observed = float(combined_timeseries["time_s"].max())
    coll_coords = generate_collocation_dataset(
        n_points=50000, t_max=t_max_observed
    )

    # Save processed outputs
    summary_path = os.path.join(output_dir, "battery_cycle_summary.csv")
    timeseries_path = os.path.join(output_dir, "battery_discharge_timeseries.csv")
    pinn_surf_csv_path = os.path.join(output_dir, "pinn_surface_dataset.csv")
    pinn_surf_pt_path = os.path.join(output_dir, "pinn_surface_dataset.pt")
    pinn_coll_pt_path = os.path.join(output_dir, "pinn_collocation_dataset.pt")

    combined_summary.to_csv(summary_path, index=False)
    combined_timeseries.to_csv(timeseries_path, index=False)
    pinn_surf_df.to_csv(pinn_surf_csv_path, index=False)

    torch.save({
        "coords": surf_coords,
        "temp_k": surf_temps,
        "metadata": {
            "num_samples": len(surf_coords),
            "columns": ["x", "y", "z", "t"],
            "target": "temp_k",
            "cell_radius": CELL_RADIUS,
            "cell_height": CELL_HEIGHT
        }
    }, pinn_surf_pt_path)

    torch.save({
        "coords": coll_coords,
        "metadata": {
            "num_samples": len(coll_coords),
            "columns": ["x", "y", "z", "t"],
            "cell_radius": CELL_RADIUS,
            "cell_height": CELL_HEIGHT,
            "t_max": t_max_observed
        }
    }, pinn_coll_pt_path)

    print("\nPreprocessing complete! Artifacts saved:")
    print(f"  - Cycle summary: {summary_path} ({len(combined_summary)} rows)")
    print(f"  - Discharge timeseries: {timeseries_path} ({len(combined_timeseries)} rows)")
    print(f"  - PINN surface CSV: {pinn_surf_csv_path} ({len(pinn_surf_df)} points)")
    print(f"  - PINN surface Tensor: {pinn_surf_pt_path} {tuple(surf_coords.shape)}")
    print(f"  - PINN collocation Tensor: {pinn_coll_pt_path} {tuple(coll_coords.shape)}")

    return {
        "summary_df": combined_summary,
        "timeseries_df": combined_timeseries,
        "pinn_surface_df": pinn_surf_df,
        "surf_coords": surf_coords,
        "surf_temps": surf_temps,
        "coll_coords": coll_coords
    }


if __name__ == "__main__":
    process_all_nasa_batteries()
