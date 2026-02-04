#!/usr/bin/env python3
"""
AI-powered Speech Therapy Assistant
===================================

This module provides an end-to-end, **non-diagnostic** analysis pipeline for
speech therapy progress tracking. It is designed to work with:

- The UCLASS-derived dataset in this folder (e.g. `audio/`, `clips/`), and
- Any arbitrary dataset directory containing nested `.wav` files
  (e.g. `speaker_01/file1.wav`, `speaker_02/file2.wav`, ...).

Each `.wav` file is treated as **one speech session sample**. For each file the
pipeline will:

1. Load and lightly preprocess the audio
2. Detect speech vs. silence regions
3. Extract fluency / energy features (Librosa)
4. Extract pitch / voice features (Parselmouth)
5. Build a feature dataset (CSV)
6. Compute simple improvement scores per speaker over time
7. Generate human-readable feedback and visualizations

The system is intended to support therapists, not replace clinical judgement.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

try:
    import parselmouth  
except ImportError as exc:  
    raise ImportError(
        "The 'parselmouth' package is required for pitch analysis.\n"
        "Install it with: pip install praat-parselmouth"
    ) from exc




TARGET_SR = 16_000
MIN_PAUSE_SECONDS = 0.25  
NON_SILENCE_TOP_DB = 30.0  
FRAME_LENGTH = 2048
HOP_LENGTH = 512


TARGET_SPEECH_RATE_EVENTS_PER_SEC = 3.0




@dataclass
class AudioSessionFeatures:
    file_path: str
    speaker: str
    rel_path: str

    duration_sec: float
    pause_count: int
    speech_rate: float  
    energy_var: float
    pitch_mean: float
    pitch_var: float


@dataclass
class SessionImprovement:
    speaker: str
    rel_path: str
    session_index: int  
    improvement_score: float  
    pause_change_pct: Optional[float]
    pitch_var_change_pct: Optional[float]
    energy_var_change_pct: Optional[float]
    speech_rate_towards_target_pct: Optional[float]
    explanation: str



def infer_speaker_label(root: Path, wav_path: Path) -> str:
    """
    Infer a speaker/session label from the file path.

    Strategy:
    - If the file lives in a subfolder of `root`, use the first folder name
      under root (e.g. root/speaker_01/file1.wav -> speaker_01).
    - Otherwise, fall back to the stem prefix before the first underscore,
      or the full stem if there is no underscore.
    """
    rel = wav_path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]

    stem = rel.stem
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def scan_wav_files(dataset_root: Path) -> List[Tuple[Path, str, str]]:
    """
    Recursively find all `.wav` files under `dataset_root`.

    Returns a list of tuples: (absolute_path, speaker_label, relative_path_str).
    """
    wav_paths = sorted(dataset_root.rglob("*.wav"))
    results: List[Tuple[Path, str, str]] = []

    for wav in wav_paths:
        speaker = infer_speaker_label(dataset_root, wav)
        rel_path = str(wav.relative_to(dataset_root))
        results.append((wav, speaker, rel_path))

    return results


# --------------------
# Audio preprocessing
# --------------------

def load_and_preprocess_audio(path: Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file, resample, normalize volume, and return (y, sr).

    Normalization:
    - Scales audio so that the peak absolute amplitude is 1.0.
    - If the file is silent or nearly so, returns it unchanged.
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)

    peak = np.max(np.abs(y)) if y.size > 0 else 0.0
    if peak > 0:
        y = y / peak

    return y, sr


def detect_speech_segments(y: np.ndarray, sr: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Detect non-silent (speech) segments using librosa.effects.split.

    Returns:
    - y (unchanged, for convenience)
    - intervals: list of (start_sample, end_sample) for non-silent regions
    """
    if y.size == 0:
        return y, []
    intervals = librosa.effects.split(y, top_db=NON_SILENCE_TOP_DB)
    return y, [(int(start), int(end)) for start, end in intervals]


def count_pauses(
    intervals: List[Tuple[int, int]],
    total_len: int,
    sr: int,
    min_pause_sec: float = MIN_PAUSE_SECONDS,
) -> int:
    """
    Count pauses as silence gaps between non-silent intervals longer than
    `min_pause_sec`.
    """
    if not intervals:
        return 0

    pauses = 0
    prev_end = 0

    for start, end in intervals:
        if start > prev_end:
            gap_sec = (start - prev_end) / float(sr)
            if gap_sec >= min_pause_sec:
                pauses += 1
        prev_end = end

    # We typically ignore trailing silence after the last speech segment
    return pauses


def estimate_speech_rate(y: np.ndarray, sr: int) -> float:
    """
    Estimate a simple speech rate proxy using onset detection.

    Returns:
    - syllable-like events per second (not a clinical measure of WPM).
    """
    duration_sec = len(y) / float(sr) if len(y) > 0 else 0.0
    if duration_sec <= 0.0:
        return 0.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    num_events = len(onset_frames)

    return num_events / duration_sec


def compute_energy_variation(y: np.ndarray) -> float:
    """
    Compute variance of frame-wise RMS energy as a proxy for energy stability.
    """
    if y.size == 0:
        return 0.0
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    return float(np.var(rms)) if rms.size > 0 else 0.0


# -----------------
# Pitch extraction
# -----------------

def extract_pitch_stats(audio_path: Path) -> Tuple[float, float]:
    """
    Extract mean and variance of the fundamental frequency (pitch) using
    Parselmouth / Praat.

    Unvoiced frames (0 Hz) are excluded from the statistics. If no voiced
    frames are detected, returns (NaN, NaN).
    """
    snd = parselmouth.Sound(str(audio_path))
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    voiced = pitch_values[pitch_values > 0]

    if voiced.size == 0:
        return float("nan"), float("nan")

    return float(np.mean(voiced)), float(np.var(voiced))


# --------------------
# Feature computation
# --------------------

def extract_features_for_file(
    dataset_root: Path,
    wav_path: Path,
    speaker: str,
    rel_path: str,
) -> AudioSessionFeatures:
    """
    Run preprocessing + feature extraction for a single WAV file.
    """
    y, sr = load_and_preprocess_audio(wav_path, target_sr=TARGET_SR)
    duration_sec = len(y) / float(sr) if len(y) > 0 else 0.0

    _, intervals = detect_speech_segments(y, sr)
    pause_count = count_pauses(intervals, total_len=len(y), sr=sr)
    speech_rate = estimate_speech_rate(y, sr)
    energy_var = compute_energy_variation(y)

    pitch_mean, pitch_var = extract_pitch_stats(wav_path)

    return AudioSessionFeatures(
        file_path=str(wav_path.resolve()),
        speaker=speaker,
        rel_path=rel_path,
        duration_sec=duration_sec,
        pause_count=pause_count,
        speech_rate=speech_rate,
        energy_var=energy_var,
        pitch_mean=pitch_mean,
        pitch_var=pitch_var,
    )


def build_feature_dataset(dataset_root: Path) -> pd.DataFrame:
    """
    Scan the dataset directory and build a feature DataFrame for all WAV files.
    """
    items = scan_wav_files(dataset_root)
    if not items:
        raise FileNotFoundError(f"No .wav files found under: {dataset_root}")

    records: List[Dict[str, object]] = []

    for wav_path, speaker, rel_path in items:
        feats = extract_features_for_file(dataset_root, wav_path, speaker, rel_path)
        records.append(asdict(feats))

    df = pd.DataFrame(records)
    return df


# ------------------------
# Progress / trend scoring
# ------------------------

def _percent_change(baseline: float, current: float) -> Optional[float]:
    """
    Compute percentage change from baseline to current.
    Positive value means a decrease when we call with (baseline, current)
    in the form (old, new) and then interpret appropriately.
    """
    if not np.isfinite(baseline) or baseline == 0:
        return None
    return 100.0 * (baseline - current) / abs(baseline)


def _towards_target_pct(
    value_baseline: float,
    value_current: float,
    target: float,
) -> Optional[float]:
    """
    Measure how much the current value has moved towards a target value,
    expressed as a percentage improvement over baseline.
    """
    if not (np.isfinite(value_baseline) and np.isfinite(value_current)):
        return None

    baseline_dist = abs(value_baseline - target)
    current_dist = abs(value_current - target)

    if baseline_dist == 0:
        return None

    improvement = baseline_dist - current_dist
    return 100.0 * improvement / baseline_dist


def compute_improvement_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[SessionImprovement]]:
    """
    For each speaker, treat the first session (by relative path sort order)
    as baseline and compute simple improvement scores for all sessions.

    Returns:
    - df_with_scores: original df + `speaker`, `session_index`, `improvement_score`
    - improvements: list of SessionImprovement objects with textual explanations
    """
    if df.empty:
        raise ValueError("Feature DataFrame is empty.")

    df = df.copy()
    df.sort_values(["speaker", "rel_path"], inplace=True)
    df["session_index"] = df.groupby("speaker").cumcount()

    improvements: List[SessionImprovement] = []
    scores: List[float] = []

    for speaker, group in df.groupby("speaker"):
        group = group.sort_values("session_index")
        baseline = group.iloc[0]

        for _, row in group.iterrows():
            idx = int(row["session_index"])

            if idx == 0:
                # Baseline: define neutral mid-score
                score = 50.0
                explanation = (
                    f"Baseline session for speaker '{speaker}'. "
                    "Subsequent sessions will be compared against this recording."
                )
                pause_change_pct = None
                pitch_var_change_pct = None
                energy_var_change_pct = None
                sr_towards_target_pct = None
            else:
                pause_change_pct = _percent_change(
                    float(baseline["pause_count"]), float(row["pause_count"])
                )
                pitch_var_change_pct = _percent_change(
                    float(baseline["pitch_var"]), float(row["pitch_var"])
                )
                energy_var_change_pct = _percent_change(
                    float(baseline["energy_var"]), float(row["energy_var"])
                )
                sr_towards_target_pct = _towards_target_pct(
                    float(baseline["speech_rate"]),
                    float(row["speech_rate"]),
                    target=TARGET_SPEECH_RATE_EVENTS_PER_SEC,
                )

                # Aggregate into a simple overall improvement score
                component_values: List[float] = []
                for val in [
                    pause_change_pct,
                    pitch_var_change_pct,
                    energy_var_change_pct,
                    sr_towards_target_pct,
                ]:
                    if val is not None:
                        component_values.append(val)

                if component_values:
                    # Clamp to [-100, 100] then rescale to [0, 100]
                    mean_improvement = float(np.mean(component_values))
                    mean_improvement = max(-100.0, min(100.0, mean_improvement))
                    score = 50.0 + 0.5 * mean_improvement
                else:
                    score = 50.0

                explanation_parts: List[str] = []

                if pause_change_pct is not None:
                    if pause_change_pct > 5:
                        explanation_parts.append(
                            f"Pause frequency reduced by {pause_change_pct:.1f}% compared to baseline."
                        )
                    elif pause_change_pct < -5:
                        explanation_parts.append(
                            f"Pause frequency increased by {abs(pause_change_pct):.1f}% compared to baseline."
                        )

                if pitch_var_change_pct is not None:
                    if pitch_var_change_pct > 5:
                        explanation_parts.append(
                            f"Pitch variation is {pitch_var_change_pct:.1f}% more stable, "
                            "suggesting improved vocal control."
                        )
                    elif pitch_var_change_pct < -5:
                        explanation_parts.append(
                            f"Pitch variation is {abs(pitch_var_change_pct):.1f}% higher than baseline, "
                            "indicating less stable vocal control."
                        )

                if energy_var_change_pct is not None:
                    if energy_var_change_pct > 5:
                        explanation_parts.append(
                            f"Energy levels are {energy_var_change_pct:.1f}% more consistent than baseline."
                        )
                    elif energy_var_change_pct < -5:
                        explanation_parts.append(
                            f"Energy levels are {abs(energy_var_change_pct):.1f}% more variable than baseline."
                        )

                if sr_towards_target_pct is not None:
                    if sr_towards_target_pct > 5:
                        explanation_parts.append(
                            f"Speech rate is {sr_towards_target_pct:.1f}% closer to a typical conversational pace."
                        )
                    elif sr_towards_target_pct < -5:
                        explanation_parts.append(
                            f"Speech rate moved {abs(sr_towards_target_pct):.1f}% away from a typical "
                            "conversational pace."
                        )

                if not explanation_parts:
                    explanation_parts.append(
                        "No substantial change detected across the tracked features relative to baseline."
                    )

                explanation = " ".join(explanation_parts)

            improvements.append(
                SessionImprovement(
                    speaker=speaker,
                    rel_path=row["rel_path"],
                    session_index=idx,
                    improvement_score=score,
                    pause_change_pct=pause_change_pct,
                    pitch_var_change_pct=pitch_var_change_pct,
                    energy_var_change_pct=energy_var_change_pct,
                    speech_rate_towards_target_pct=sr_towards_target_pct,
                    explanation=explanation,
                )
            )
            scores.append(score)

    df["improvement_score"] = scores
    return df, improvements


# ----------------
# Visualisations
# ----------------

def plot_pitch_contour(
    audio_path: Path,
    out_dir: Path,
    speaker: str,
    rel_path: str,
) -> Path:
    """
    Generate a pitch contour plot (pitch vs. time) for a single file.
    """
    snd = parselmouth.Sound(str(audio_path))
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    times = pitch.xs()

    voiced = pitch_values > 0

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = rel_path.replace(os.sep, "_").replace("/", "_")
    out_path = out_dir / f"{speaker}__{safe_name}__pitch.png"

    plt.figure(figsize=(8, 3))
    plt.plot(times[voiced], pitch_values[voiced], label="Pitch (Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Pitch contour – {speaker} – {rel_path}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def plot_pause_timeline(
    y: np.ndarray,
    sr: int,
    intervals: List[Tuple[int, int]],
    out_dir: Path,
    speaker: str,
    rel_path: str,
) -> Path:
    """
    Plot a simple speech vs. silence timeline based on non-silent intervals.
    """
    total_duration = len(y) / float(sr) if len(y) > 0 else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = rel_path.replace(os.sep, "_").replace("/", "_")
    out_path = out_dir / f"{speaker}__{safe_name}__speech_timeline.png"

    plt.figure(figsize=(8, 1.8))

    # Background: silence
    plt.axhspan(0, 1, xmin=0, xmax=1, facecolor="lightgray", alpha=0.5, label="Silence")

    # Overlay: speech segments
    for start, end in intervals:
        start_t = start / float(sr)
        end_t = end / float(sr)
        plt.axvspan(start_t, end_t, ymin=0.05, ymax=0.95, facecolor="tab:green", alpha=0.8)

    plt.xlim(0, max(total_duration, 1e-3))
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.title(f"Speech vs. silence – {speaker} – {rel_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def generate_visualisations_for_sample(
    dataset_root: Path,
    wav_path: Path,
    speaker: str,
    rel_path: str,
    vis_root: Path,
) -> Tuple[Path, Path]:
    """
    Convenience helper: generate both pitch contour and pause timeline
    for a given audio file.
    """
    y, sr = load_and_preprocess_audio(wav_path, target_sr=TARGET_SR)
    _, intervals = detect_speech_segments(y, sr)

    pitch_dir = vis_root / "pitch_contours"
    timeline_dir = vis_root / "speech_timelines"

    pitch_path = plot_pitch_contour(wav_path, pitch_dir, speaker, rel_path)
    timeline_path = plot_pause_timeline(y, sr, intervals, timeline_dir, speaker, rel_path)

    return pitch_path, timeline_path


# -------------
# CLI / Runner
# -------------

def run_pipeline(
    dataset_root: Path,
    features_csv: Path,
    scored_features_csv: Path,
    report_path: Path,
    vis_root: Path,
    max_visualisations_per_speaker: int = 3,
) -> None:
    """
    High-level orchestration of the full pipeline on a dataset root.
    """
    print(f"Scanning dataset under: {dataset_root}")
    df_features = build_feature_dataset(dataset_root)
    print(f"Extracted features for {len(df_features)} sessions.")

    features_csv.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(features_csv, index=False)
    print(f"Saved raw feature dataset to: {features_csv}")

    df_scored, improvements = compute_improvement_scores(df_features)
    df_scored.to_csv(scored_features_csv, index=False)
    print(f"Saved scored feature dataset to: {scored_features_csv}")

    print("\n===== SESSION IMPROVEMENT RESULTS =====\n")

    for imp in improvements:
        print(f"Speaker: {imp.speaker}")
        print(f"Session: {imp.rel_path}")
        print(f"Score: {imp.improvement_score:.1f}")
        print(f"Explanation: {imp.explanation}")
        print("-" * 40)


    # Write a simple text report with the explanations
    with open(report_path, "w", encoding="utf-8") as f:
        for imp in improvements:
            f.write(
                f"Speaker: {imp.speaker} | Session: {imp.rel_path} | "
                f"Index: {imp.session_index} | Score: {imp.improvement_score:.1f}\n"
            )
            f.write(f"  -> {imp.explanation}\n\n")
    print(f"Wrote textual progress report to: {report_path}")

    # Generate a few example visualisations per speaker
    print("Generating example visualisations (pitch contour + pause timeline)...")
    grouped = df_features.groupby("speaker")
    for speaker, group in grouped:
        subset = group.sort_values("rel_path").head(max_visualisations_per_speaker)
        for _, row in subset.iterrows():
            wav_path = Path(row["file_path"])
            rel_path = str(row["rel_path"])
            generate_visualisations_for_sample(
                dataset_root=dataset_root,
                wav_path=wav_path,
                speaker=speaker,
                rel_path=rel_path,
                vis_root=vis_root,
            )
    print(f"Visualisations saved under: {vis_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the AI-powered Speech Therapy Assistant pipeline on a folder of WAV files.\n\n"
            "Examples:\n"
            "  python speech_therapy_assistant.py --data-root ./audio\n"
            "  python speech_therapy_assistant.py --data-root ./clips/clips\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to the main dataset folder containing (nested) .wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory to store CSVs, reports, and visualisations.",
    )
    parser.add_argument(
        "--max-vis-per-speaker",
        type=int,
        default=3,
        help="Maximum number of sessions per speaker to visualise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found or not a directory: {dataset_root}")

    features_csv = output_dir / "session_features.csv"
    scored_features_csv = output_dir / "session_features_scored.csv"
    report_path = output_dir / "progress_report.txt"
    vis_root = output_dir / "visualisations"

    run_pipeline(
        dataset_root=dataset_root,
        features_csv=features_csv,
        scored_features_csv=scored_features_csv,
        report_path=report_path,
        vis_root=vis_root,
        max_visualisations_per_speaker=int(args.max_vis_per_speaker),
    )


if __name__ == "__main__":
    main()

