"""
Evaluate BCI Competition IV Dataset 1 1000Hz files with two pipelines.

This script compares:
  1. minimal: raw EEG window log-variance features, no band-pass/spatial/CSP
  2. preprocessed: 8-30 Hz band-pass + high Laplacian + CSP log-variance features

It trains on each subject's calibration file, predicts the corresponding
continuous evaluation file as sample-level labels {-1, 0, 1}, and evaluates
against true_y using MSE and accuracy on non-NaN true_y samples.

Run:
  python evaluate_1000hz_pipeline.py
  python evaluate_1000hz_pipeline.py --subject a
  python evaluate_1000hz_pipeline.py --binary-only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg, signal
from scipy.io import loadmat
from sklearn import ensemble, metrics


ROOT = Path(r"C:\Users\user\Desktop\example")
CALIB_DIR = ROOT / "BCICIV_1calib_1000Hz_mat"
EVAL_DIR = ROOT / "BCICIV_1eval_1000Hz_mat"
TRUE_DIR = ROOT / "true_labels"
LABELS = np.array([-1, 0, 1], dtype=int)
LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}


@dataclass
class SubjectData:
    subject: str
    fs: int
    train_data: np.ndarray
    train_y_sample: np.ndarray
    eval_data: np.ndarray
    true_y: np.ndarray
    positions: np.ndarray


def mat_field(struct: np.ndarray, name: str) -> np.ndarray:
    return struct[name][0, 0]


def load_subject(subject: str) -> SubjectData:
    calib_path = CALIB_DIR / f"BCICIV_calib_ds1{subject}_1000Hz.mat"
    eval_path = EVAL_DIR / f"BCICIV_eval_ds1{subject}_1000Hz.mat"
    true_path = TRUE_DIR / f"BCICIV_eval_ds1{subject}_1000Hz_true_y.mat"

    calib = loadmat(calib_path)
    eval_mat = loadmat(eval_path)
    true_mat = loadmat(true_path)

    train_data = np.asarray(calib["cnt"], dtype=np.float64) * 0.1
    eval_data = np.asarray(eval_mat["cnt"], dtype=np.float64) * 0.1
    fs = int(np.asarray(mat_field(calib["nfo"], "fs")).ravel()[0])

    positions = np.vstack(
        [
            np.asarray(mat_field(calib["nfo"], "xpos")).ravel().astype(float),
            np.asarray(mat_field(calib["nfo"], "ypos")).ravel().astype(float),
        ]
    )

    train_y_sample = np.zeros(train_data.shape[0], dtype=int)
    cue_positions = np.asarray(mat_field(calib["mrk"], "pos")).ravel().astype(int)
    cue_labels = np.asarray(mat_field(calib["mrk"], "y")).ravel().astype(int)
    trial_len = 4 * fs
    for pos, label in zip(cue_positions, cue_labels):
        end = min(pos + trial_len, len(train_y_sample))
        train_y_sample[pos:end] = label

    true_y = np.asarray(true_mat["true_y"]).ravel()
    true_y = true_y[: eval_data.shape[0]]

    return SubjectData(subject, fs, train_data, train_y_sample, eval_data, true_y, positions)


def bandpass(data: np.ndarray, fs: int, low_hz: float = 8.0, high_hz: float = 30.0) -> np.ndarray:
    sos = signal.butter(3, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0)


def high_laplacian(data: np.ndarray, positions: np.ndarray) -> np.ndarray:
    n_channels = data.shape[1]
    out = np.zeros_like(data)
    for ch in range(n_channels):
        distances = np.linalg.norm(positions[:, ch, None] - positions, axis=0)
        sorted_idx = np.argsort(distances)
        candidate_idx = sorted_idx[9:21]
        candidate_distances = distances[candidate_idx]

        x_dist = np.abs(positions[0, ch] - positions[0, candidate_idx])
        y_dist = np.abs(positions[1, ch] - positions[1, candidate_idx])
        local_idx = np.concatenate([np.argsort(x_dist)[:2], np.argsort(y_dist)[:2]])

        selected_idx = candidate_idx[local_idx]
        selected_distances = candidate_distances[local_idx]
        weights = (1 / selected_distances) / np.sum(1 / selected_distances)
        out[:, ch] = data[:, ch] - data[:, selected_idx].dot(weights)
    return out


def window_starts(n_samples: int, window: int, step: int) -> np.ndarray:
    return np.arange(0, n_samples - window + 1, step, dtype=int)


def window_labels(sample_y: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    centers = starts + window // 2
    return sample_y[centers]


def logvar_features(data: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    csum = np.vstack([np.zeros((1, data.shape[1])), np.cumsum(data, axis=0)])
    csum2 = np.vstack([np.zeros((1, data.shape[1])), np.cumsum(data * data, axis=0)])
    sums = csum[starts + window] - csum[starts]
    sums2 = csum2[starts + window] - csum2[starts]
    mean = sums / window
    var = np.maximum(sums2 / window - mean * mean, 1e-12)
    return np.log(var)


def fit_csp(data: np.ndarray, starts: np.ndarray, y: np.ndarray, window: int, n_pairs: int) -> np.ndarray:
    covariances = []
    for label in (-1, 1):
        label_starts = starts[y == label]
        cov_sum = np.zeros((data.shape[1], data.shape[1]), dtype=np.float64)
        for start in label_starts:
            segment = data[start : start + window].T
            segment = segment - np.mean(segment, axis=1, keepdims=True)
            cov = segment.dot(segment.T)
            trace = np.trace(cov)
            if trace > 0:
                cov_sum += cov / trace
        covariances.append(cov_sum / max(len(label_starts), 1))

    eig_values, eig_vectors = linalg.eig(covariances[0], covariances[1])
    order = np.argsort(eig_values.real)[::-1]
    eig_vectors = eig_vectors.real[:, order]
    return np.concatenate([eig_vectors[:, :n_pairs], eig_vectors[:, -n_pairs:]], axis=1)


def csp_features(data: np.ndarray, starts: np.ndarray, window: int, filters: np.ndarray) -> np.ndarray:
    features = np.zeros((len(starts), filters.shape[1]), dtype=np.float64)
    for i, start in enumerate(starts):
        segment = data[start : start + window].T
        projected = filters.T.dot(segment)
        var = np.maximum(np.var(projected, axis=1), 1e-12)
        features[i] = np.log(var / np.sum(var))
    return features


def make_model(random_state: int) -> ensemble.RandomForestClassifier:
    return ensemble.RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=1,
        random_state=random_state,
    )


def window_predictions_to_samples(
    starts: np.ndarray,
    predictions: np.ndarray,
    n_samples: int,
    window: int,
    labels: np.ndarray,
) -> np.ndarray:
    label_to_index = {int(label): index for index, label in enumerate(labels)}
    votes = np.zeros((n_samples, len(labels)), dtype=np.int16)
    for start, pred in zip(starts, predictions):
        votes[start : start + window, label_to_index[int(pred)]] += 1
    no_vote = votes.sum(axis=1) == 0
    sample_pred = labels[np.argmax(votes, axis=1)]
    sample_pred[no_vote] = 0 if 0 in labels else labels[0]
    return sample_pred


def evaluate_prediction(true_y: np.ndarray, pred_y: np.ndarray, binary_only: bool) -> dict:
    mask = ~np.isnan(true_y)
    labels = LABELS
    if binary_only:
        mask &= np.isin(true_y, [-1, 1])
        labels = np.array([-1, 1], dtype=int)

    y_true = true_y[mask].astype(int)
    y_pred = pred_y[mask].astype(int)
    return {
        "mse": metrics.mean_squared_error(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "n_eval": len(y_true),
        "confusion": metrics.confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
    }


def run_pipeline_for_subject(data: SubjectData, mode: str, args: argparse.Namespace) -> dict:
    window = int(args.window_seconds * data.fs)
    step = int(args.step_seconds * data.fs)

    train_data = data.train_data
    eval_data = data.eval_data

    if mode == "preprocessed":
        train_data = high_laplacian(bandpass(train_data, data.fs), data.positions)
        eval_data = high_laplacian(bandpass(eval_data, data.fs), data.positions)

    train_starts = window_starts(train_data.shape[0], window, step)
    eval_starts = window_starts(eval_data.shape[0], window, step)
    y_train = window_labels(data.train_y_sample, train_starts, window)

    if mode == "minimal":
        x_train = logvar_features(train_data, train_starts, window)
        x_eval = logvar_features(eval_data, eval_starts, window)
    elif mode == "preprocessed":
        csp = fit_csp(train_data, train_starts, y_train, window, args.csp_pairs)
        x_train = csp_features(train_data, train_starts, window, csp)
        x_eval = csp_features(eval_data, eval_starts, window, csp)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if args.binary_only:
        train_mask = y_train != 0
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]

    model = make_model(args.random_state)
    model.fit(x_train, y_train)
    eval_window_pred = model.predict(x_eval)
    pred_labels = np.array([-1, 1], dtype=int) if args.binary_only else LABELS
    eval_sample_pred = window_predictions_to_samples(eval_starts, eval_window_pred, eval_data.shape[0], window, pred_labels)
    result = evaluate_prediction(data.true_y, eval_sample_pred, args.binary_only)

    pred_mask = ~np.isnan(data.true_y)
    if args.binary_only:
        pred_mask &= np.isin(data.true_y, [-1, 1])
    unique, counts = np.unique(eval_sample_pred[pred_mask], return_counts=True)
    result["pred_counts"] = dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))
    result["train_counts"] = dict(zip(*np.unique(y_train, return_counts=True)))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=list("abcdefg"), default=None, help="Run one subject only.")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--step-seconds", type=float, default=0.25)
    parser.add_argument("--csp-pairs", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--binary-only", action="store_true", help="Train/evaluate only labels -1 and 1, ignoring class 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = [args.subject] if args.subject else list("abcdefg")
    rows = []

    for subject in subjects:
        print(f"\n=== Subject {subject} ===")
        data = load_subject(subject)
        for mode in ("minimal", "preprocessed"):
            result = run_pipeline_for_subject(data, mode, args)
            rows.append((subject, mode, result["mse"], result["accuracy"], result["n_eval"]))
            print(
                f"{mode:12s} | MSE={result['mse']:.4f} | "
                f"accuracy={result['accuracy']:.4f} | n={result['n_eval']} | "
                f"pred_counts={result['pred_counts']}"
            )
            label_text = ", ".join(str(int(label)) for label in result["labels"])
            print(f"confusion rows true [{label_text}], cols pred [{label_text}]:")
            print(result["confusion"])

    print("\n=== Summary ===")
    print("subject,mode,mse,accuracy,n_eval")
    for row in rows:
        print(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]}")

    for mode in ("minimal", "preprocessed"):
        mode_rows = [r for r in rows if r[1] == mode]
        print(
            f"{mode:12s} mean MSE={np.mean([r[2] for r in mode_rows]):.6f}, "
            f"mean accuracy={np.mean([r[3] for r in mode_rows]):.6f}"
        )

    best_by_accuracy = max(rows, key=lambda row: row[3])
    best_by_mse = min(rows, key=lambda row: row[2])
    print(
        f"best accuracy: subject {best_by_accuracy[0]}, {best_by_accuracy[1]}, "
        f"accuracy={best_by_accuracy[3]:.6f}, MSE={best_by_accuracy[2]:.6f}"
    )
    print(
        f"best MSE     : subject {best_by_mse[0]}, {best_by_mse[1]}, "
        f"MSE={best_by_mse[2]:.6f}, accuracy={best_by_mse[3]:.6f}"
    )


if __name__ == "__main__":
    main()
