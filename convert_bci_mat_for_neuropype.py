"""
Convert BCI Competition IV Dataset 1 .mat files into simple NeuroPype-friendly CSV files.

Output folder structure, one folder per recording:
  eeg.csv        continuous EEG, columns: time_sec, AF3, AF4, ...
  markers.csv    calibration cue events, if available
  channels.csv   channel labels and 2D electrode coordinates
  metadata.json  source file, sampling rates, units, labels
  true_y.csv      evaluation sample labels, if a matching true_y file exists

Default:
  Converts subject e calibration and evaluation data from 1000 Hz to 250 Hz.

Examples:
  python convert_bci_mat_for_neuropype.py
  python convert_bci_mat_for_neuropype.py --subject d --kind calib
  python convert_bci_mat_for_neuropype.py --all-subjects --kind both --target-fs 250
  python convert_bci_mat_for_neuropype.py --subject e --target-fs 1000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat


ROOT = Path(r"C:\Users\user\Desktop\example")
CALIB_DIR = ROOT / "BCICIV_1calib_1000Hz_mat"
EVAL_DIR = ROOT / "BCICIV_1eval_1000Hz_mat"
TRUE_DIR = ROOT / "true_labels"
OUTPUT_DIR = ROOT / "neuropype_ready"


def mat_field(struct: np.ndarray, name: str) -> np.ndarray:
    return struct[name][0, 0]


def clean_mat_string(value) -> str:
    arr = np.asarray(value).ravel()
    if len(arr) == 0:
        return ""
    item = arr[0]
    if isinstance(item, np.ndarray):
        return str(item.ravel()[0])
    return str(item)


def load_recording(subject: str, kind: str) -> tuple[Path, dict]:
    if kind == "calib":
        path = CALIB_DIR / f"BCICIV_calib_ds1{subject}_1000Hz.mat"
    elif kind == "eval":
        path = EVAL_DIR / f"BCICIV_eval_ds1{subject}_1000Hz.mat"
    else:
        raise ValueError("kind must be calib or eval")
    return path, loadmat(path)


def get_fs(mat: dict) -> int:
    return int(np.asarray(mat_field(mat["nfo"], "fs")).ravel()[0])


def get_channel_names(mat: dict) -> list[str]:
    return [clean_mat_string(x) for x in np.asarray(mat_field(mat["nfo"], "clab")).ravel()]


def get_classes(mat: dict) -> list[str]:
    return [clean_mat_string(x) for x in np.asarray(mat_field(mat["nfo"], "classes")).ravel()]


def get_positions(mat: dict) -> tuple[np.ndarray, np.ndarray]:
    xpos = np.asarray(mat_field(mat["nfo"], "xpos")).ravel().astype(float)
    ypos = np.asarray(mat_field(mat["nfo"], "ypos")).ravel().astype(float)
    return xpos, ypos


def resample_data(data: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    if source_fs == target_fs:
        return data.astype(np.float32, copy=False)

    gcd = math.gcd(source_fs, target_fs)
    up = target_fs // gcd
    down = source_fs // gcd
    return signal.resample_poly(data, up, down, axis=0).astype(np.float32, copy=False)


def write_eeg_csv(path: Path, data: np.ndarray, channel_names: list[str], fs: int, chunk_rows: int) -> None:
    columns = ["time_sec"] + channel_names
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")

    for start in range(0, data.shape[0], chunk_rows):
        end = min(start + chunk_rows, data.shape[0])
        time_sec = np.arange(start, end, dtype=np.float64) / fs
        chunk = np.column_stack([time_sec, data[start:end]])
        frame = pd.DataFrame(chunk, columns=columns)
        frame.to_csv(path, mode="a", index=False, header=False, float_format="%.7g")


def write_eeg_with_markers_csv(
    path: Path,
    data: np.ndarray,
    channel_names: list[str],
    fs: int,
    marker_rows: list[dict],
    chunk_rows: int,
) -> None:
    columns = ["time_sec", "marker"] + channel_names
    marker_by_sample = {int(row["sample"]): row["label_name"] for row in marker_rows}
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")

    for start in range(0, data.shape[0], chunk_rows):
        end = min(start + chunk_rows, data.shape[0])
        time_sec = np.arange(start, end, dtype=np.float64) / fs
        markers = [marker_by_sample.get(sample, "") for sample in range(start, end)]
        chunk = pd.DataFrame(data[start:end], columns=channel_names)
        chunk.insert(0, "marker", markers)
        chunk.insert(0, "time_sec", time_sec)
        chunk.to_csv(path, mode="a", index=False, header=False, float_format="%.7g")


def write_channels_csv(path: Path, channel_names: list[str], xpos: np.ndarray, ypos: np.ndarray) -> None:
    frame = pd.DataFrame(
        {
            "channel": channel_names,
            "x": xpos,
            "y": ypos,
            "z": np.zeros(len(channel_names), dtype=float),
        }
    )
    frame.to_csv(path, index=False)


def build_marker_rows(mat: dict, source_fs: int, target_fs: int, classes: list[str]) -> list[dict]:
    if "mrk" not in mat:
        return []

    positions = np.asarray(mat_field(mat["mrk"], "pos")).ravel().astype(int)
    labels = np.asarray(mat_field(mat["mrk"], "y")).ravel().astype(int)
    label_names = {
        -1: classes[0] if len(classes) > 0 else "-1",
        1: classes[1] if len(classes) > 1 else "1",
    }
    rows = []
    for event_id, (source_sample, label) in enumerate(zip(positions, labels), start=1):
        target_sample = int(round(source_sample * target_fs / source_fs))
        duration_samples = int(round(4.0 * target_fs))
        rows.append(
            {
                "event_id": len(rows) + 1,
                "source_sample": source_sample,
                "sample": target_sample,
                "time_sec": target_sample / target_fs,
                "label": label,
                "label_name": label_names.get(int(label), str(label)),
                "duration_sec": 4.0,
                "duration_samples": duration_samples,
                "event_type": "trial_start",
            }
        )
        rows.append(
            {
                "event_id": len(rows) + 1,
                "source_sample": source_sample + int(round(4.0 * source_fs)),
                "sample": target_sample + duration_samples,
                "time_sec": (target_sample + duration_samples) / target_fs,
                "label": 0,
                "label_name": "rest",
                "duration_sec": 0.0,
                "duration_samples": 0,
                "event_type": "trial_end",
            }
        )
    return sorted(rows, key=lambda row: (row["sample"], row["event_type"]))


def write_markers_csv(path: Path, marker_rows: list[dict]) -> int:
    if not marker_rows:
        return 0

    pd.DataFrame(marker_rows).to_csv(path, index=False)
    simple_path = path.with_name("markers_simple.csv")
    pd.DataFrame(
        {
            "label_name": [row["label_name"] for row in marker_rows],
            "time_sec": [row["time_sec"] for row in marker_rows],
        }
    ).to_csv(simple_path, index=False)
    return len(marker_rows)


def write_true_y_csv(path: Path, subject: str, n_samples: int, source_fs: int, target_fs: int, chunk_rows: int) -> bool:
    true_path = TRUE_DIR / f"BCICIV_eval_ds1{subject}_1000Hz_true_y.mat"
    if not true_path.exists():
        return False

    true_y = np.asarray(loadmat(true_path)["true_y"]).ravel()[:n_samples]
    if source_fs != target_fs:
        source_idx = np.round(np.arange(0, len(true_y), source_fs / target_fs)).astype(int)
        source_idx = source_idx[source_idx < len(true_y)]
        true_y = true_y[source_idx]

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("time_sec,true_y\n")

    for start in range(0, len(true_y), chunk_rows):
        end = min(start + chunk_rows, len(true_y))
        time_sec = np.arange(start, end, dtype=np.float64) / target_fs
        frame = pd.DataFrame({"time_sec": time_sec, "true_y": true_y[start:end]})
        frame.to_csv(path, mode="a", index=False, header=False, float_format="%.7g")

    return True


def convert_one(subject: str, kind: str, target_fs: int, output_dir: Path, chunk_rows: int) -> Path:
    source_path, mat = load_recording(subject, kind)
    source_fs = get_fs(mat)
    channel_names = get_channel_names(mat)
    classes = get_classes(mat)
    xpos, ypos = get_positions(mat)

    raw_cnt = np.asarray(mat["cnt"], dtype=np.float32)
    eeg_microvolts = raw_cnt * 0.1
    eeg_out = resample_data(eeg_microvolts, source_fs, target_fs)

    out_dir = output_dir / f"ds1{subject}_{kind}_{target_fs}Hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_rows = build_marker_rows(mat, source_fs, target_fs, classes)
    marker_count = write_markers_csv(out_dir / "markers.csv", marker_rows)
    write_channels_csv(out_dir / "channels.csv", channel_names, xpos, ypos)
    write_eeg_csv(out_dir / "eeg.csv", eeg_out, channel_names, target_fs, chunk_rows)
    if marker_rows:
        write_eeg_with_markers_csv(out_dir / "eeg_with_markers.csv", eeg_out, channel_names, target_fs, marker_rows, chunk_rows)

    has_true_y = False
    if kind == "eval":
        has_true_y = write_true_y_csv(out_dir / "true_y.csv", subject, raw_cnt.shape[0], source_fs, target_fs, chunk_rows)

    metadata = {
        "subject": f"ds1{subject}",
        "kind": kind,
        "source_file": str(source_path),
        "source_fs_hz": source_fs,
        "target_fs_hz": target_fs,
        "unit": "microvolts",
        "n_samples": int(eeg_out.shape[0]),
        "n_channels": int(eeg_out.shape[1]),
        "classes": classes,
        "files": {
            "eeg": "eeg.csv",
            "eeg_with_markers": "eeg_with_markers.csv" if marker_count else None,
            "channels": "channels.csv",
            "markers": "markers.csv" if marker_count else None,
            "markers_simple": "markers_simple.csv" if marker_count else None,
            "true_y": "true_y.csv" if has_true_y else None,
        },
        "notes": [
            "eeg.csv contains one time_sec column followed by EEG channels.",
            "eeg_with_markers.csv embeds marker labels in the marker column to avoid a separate marker merge step.",
            "markers.csv contains cue onset events for calibration recordings.",
            "markers_simple.csv contains only label_name and time_sec for marker-only CSV import.",
            "Marker labels are original BCI labels: -1 and 1.",
            "true_y.csv is exported only for evaluation recordings when true_labels is available.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BCI .mat recordings to NeuroPype-friendly CSV folders.")
    parser.add_argument("--subject", choices=list("abcdefg"), default="e", help="Subject to convert. Default: e.")
    parser.add_argument("--all-subjects", action="store_true", help="Convert subjects a-g.")
    parser.add_argument("--kind", choices=["calib", "eval", "both"], default="both", help="Recording type to convert.")
    parser.add_argument("--target-fs", type=int, default=250, help="Output sampling rate. Default: 250 Hz.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--chunk-rows", type=int, default=50000, help="Rows per CSV write chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = list("abcdefg") if args.all_subjects else [args.subject]
    kinds = ["calib", "eval"] if args.kind == "both" else [args.kind]

    for subject in subjects:
        for kind in kinds:
            print(f"Converting ds1{subject} {kind} to {args.target_fs} Hz...")
            out_dir = convert_one(subject, kind, args.target_fs, args.output_dir, args.chunk_rows)
            print(f"  wrote {out_dir}")


if __name__ == "__main__":
    main()
