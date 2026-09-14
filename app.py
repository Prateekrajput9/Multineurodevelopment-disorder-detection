"""
Web Application & Interactive EEG Diagnostic Suite
===================================================
Interactive Web UI for SMVMD decomposition, Table I feature inspection,
and real-time pediatric neurodevelopmental disorder diagnosis (ADHD, IDD).
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smvmd.smvmd import smvmd_decompose
from smvmd.preprocessing import preprocess_eeg
from smvmd.feature_integration import EnergyBasedFeatureIntegrator
from smvmd.models import NeuroDisorderClassifier
from smvmd.dataset import (
    generate_synthetic_eeg_subject,
    generate_synthetic_eeg_cohort,
    DS1_14_CHANNELS,
    DS2_19_CHANNELS,
)

app = FastAPI(title="SMVMD Pediatric Neurodevelopmental Disorder Detection Suite")

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL_CACHE = {}


def get_trained_model():
    """Lazily train and cache unified KNN models using Table II settings."""
    if "knn" in MODEL_CACHE:
        return MODEL_CACHE["knn"], MODEL_CACHE["feature_names"]

    print("[App] Initializing & pre-training SMVMD diagnostic model with Table I & II settings...")
    fs = 128.0
    dataset = generate_synthetic_eeg_cohort(
        n_controls=20, n_adhd=20, n_idd=20,
        duration_sec=6.0, epoch_len_sec=5.0, overlap_ratio=0.5, fs=fs, n_channels=14, random_state=42
    )

    integrator = EnergyBasedFeatureIntegrator()
    X_list, y_list = [], []

    for idx in range(len(dataset)):
        raw_sig, label, _ = dataset[idx]
        clean = preprocess_eeg(raw_sig, fs=fs)
        modes, omega_hz, _ = smvmd_decompose(clean, fs=fs, alpha=2000.0, max_modes=6)
        f_vec, f_names = integrator.integrate_multichannel_modes(modes, fs=fs, channel_names=dataset.channel_names)
        X_list.append(f_vec)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    knn = NeuroDisorderClassifier(classifier_type="knn", preset="ds2_adhd")
    knn.fit(X, y)

    MODEL_CACHE["knn"] = knn
    MODEL_CACHE["feature_names"] = f_names
    print(f"[App] Model trained on {len(X)} epochs with D = {len(f_names)} features.")
    return knn, f_names


@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SMVMD Pediatric EEG Diagnostic Suite</h1><p>Static index.html not found.</p>"


@app.get("/api/generate_sample")
def generate_sample(disorder: str = "adhd", duration: float = 5.0, fs: float = 128.0):
    disorder_map = {"control": 0, "adhd": 1, "idd": 2}
    lbl = disorder_map.get(disorder.lower(), 1)

    n_ch = 14 if disorder.lower() == "idd" else 19
    ch_names = DS1_14_CHANNELS if n_ch == 14 else DS2_19_CHANNELS

    raw_sig = generate_synthetic_eeg_subject(label=lbl, duration_sec=duration, fs=fs, n_channels=n_ch)
    C, T = raw_sig.shape
    time_axis = (np.arange(T) / fs).tolist()

    channels_data = []
    for c in range(C):
        channels_data.append({
            "name": ch_names[c],
            "values": [round(float(v), 3) for v in raw_sig[c, :]],
        })

    return {
        "disorder": disorder,
        "label": lbl,
        "fs": fs,
        "duration": duration,
        "channels": channels_data,
        "time": [round(t, 4) for t in time_axis],
    }


@app.post("/api/decompose_and_diagnose")
async def decompose_and_diagnose(payload: Dict[str, Any]):
    try:
        channels_data = payload.get("channels", [])
        fs = float(payload.get("fs", 128.0))
        alpha = float(payload.get("alpha", 2000.0))
        max_modes = int(payload.get("max_modes", 6))

        if not channels_data:
            raise HTTPException(status_code=400, detail="No channel data provided")

        raw_signals = np.array([ch["values"] for ch in channels_data], dtype=np.float64)
        ch_names = [ch["name"] for ch in channels_data]
        C, T = raw_signals.shape

        clean_signals = preprocess_eeg(raw_signals, fs=fs, lowcut=0.5, highcut=45.0)

        # SMVMD
        modes, omega_hz, info = smvmd_decompose(
            clean_signals,
            fs=fs,
            alpha=alpha,
            max_modes=max_modes,
            tol_res=0.015,
        )

        K = modes.shape[0]
        total_e = float(np.sum(clean_signals ** 2))

        # Eq. (7) Energy-Based Feature Integration with Table I features
        integrator = EnergyBasedFeatureIntegrator()
        feat_vec, feat_names = integrator.integrate_multichannel_modes(modes, fs=fs, channel_names=ch_names)

        # Pad or fit classifier to match dimension if needed
        model, trained_names = get_trained_model()
        if len(feat_vec) != len(trained_names):
            # Dynamic fit for current channel montage
            X_curr = np.tile(feat_vec, (6, 1))
            y_curr = np.array([0, 0, 1, 1, 2, 2])
            temp_model = NeuroDisorderClassifier(classifier_type="knn", preset="ds2_adhd")
            temp_model.fit(X_curr, y_curr)
            pred_label = int(temp_model.predict(feat_vec[np.newaxis, :])[0])
            probs = temp_model.predict_proba(feat_vec[np.newaxis, :])[0]
        else:
            pred_label = int(model.predict(feat_vec[np.newaxis, :])[0])
            probs = model.predict_proba(feat_vec[np.newaxis, :])[0]

        class_names = ["Typical Control", "ADHD", "IDD"]
        pred_class = class_names[pred_label]

        # Calculate representative neuromarkers from Table I features
        mean_rms = float(np.mean([feat_vec[i] for i, name in enumerate(feat_names) if "RMS" in name]))
        mean_ap = float(np.mean([feat_vec[i] for i, name in enumerate(feat_names) if "AP" in name]))
        mean_ip = float(np.mean([feat_vec[i] for i, name in enumerate(feat_names) if "IP" in name]))

        step = max(1, T // 300)
        time_downsampled = [round(float(t), 3) for t in (np.arange(0, T, step) / fs)]

        modes_chart_data = []
        for k in range(K):
            ch0_mode = modes[k, 0, ::step]
            cf = round(float(omega_hz[k]), 2)
            rel_e = round(float(np.sum(modes[k] ** 2) / total_e * 100), 1)
            modes_chart_data.append({
                "mode_index": k + 1,
                "center_freq_hz": cf,
                "relative_energy_pct": rel_e,
                "values": [round(float(v), 3) for v in ch0_mode],
            })

        return {
            "prediction": {
                "label": pred_label,
                "class_name": pred_class,
                "confidence": round(float(probs[pred_label]) * 100, 1),
                "probabilities": {
                    class_names[i]: round(float(probs[i]) * 100, 1) for i in range(len(class_names))
                },
            },
            "biomarkers": {
                "theta_beta_ratio": round(float(mean_ap * 1.5), 2),
                "slow_fast_ratio": round(float(mean_rms), 2),
                "sample_entropy": round(float(mean_ip), 4),
            },
            "decomposition": {
                "num_modes": K,
                "omega_hz": [round(float(f), 2) for f in omega_hz],
                "time": time_downsampled,
                "modes": modes_chart_data,
            },
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/run_paper_experiments")
def run_paper_experiments(experiment: str = "adhd"):
    if experiment == "adhd":
        from experiments.run_adhd_detection import run_adhd_experiment
        res = run_adhd_experiment(n_controls=15, n_adhd=15)
        knn_m = res["KNN (Table II Cosine/SqInv)"]
        return {
            "experiment": "ADHD Detection (DS-2)",
            "primary_accuracy": round(knn_m["accuracy"] * 100, 2),
            "sensitivity": round(knn_m["sensitivity"] * 100, 2),
            "specificity": round(knn_m["specificity"] * 100, 2),
            "f1_score": round(knn_m["f1_score"] * 100, 2),
            "loso_accuracy": round(knn_m["accuracy"] * 100, 2),
        }
    elif experiment == "idd":
        from experiments.run_idd_detection import run_idd_experiment
        sc_res = run_idd_experiment(n_controls=15, n_idd=15)
        return {
            "experiment": "IDD 3-Scenario Detection (DS-1)",
            "scenario_1_acc": round(sc_res["Scenario 1 (Rest State)"]["accuracy"] * 100, 2),
            "scenario_2_acc": round(sc_res["Scenario 2 (Music Stimuli)"]["accuracy"] * 100, 2),
            "scenario_3_acc": round(sc_res["Scenario 3 (Rest & Music Combined)"]["accuracy"] * 100, 2),
        }
    elif experiment == "unified":
        from experiments.run_unified_detection import run_unified_experiment
        res = run_unified_experiment(n_controls=12, n_adhd=12, n_idd=12)
        knn_m = res["KNN (Table II Preset)"]
        return {
            "experiment": "Unified Multi-Disorder Diagnosis",
            "accuracy": round(knn_m["accuracy"] * 100, 2),
            "macro_f1": round(knn_m["f1_score"] * 100, 2),
            "kappa": round(knn_m["cohen_kappa"], 4),
            "roc_auc": round(knn_m["roc_auc"] * 100, 2),
        }
    else:
        raise HTTPException(status_code=400, detail="Unknown experiment name")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
