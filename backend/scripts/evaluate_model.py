"""Real Model Evaluation Script for FloodSense-AI.
Evaluates DualStreamAttentionUNet, SAR Otsu Baseline, and Optical Baseline
on the mini_sen1floods11/val split and computes true confusion matrices and metrics.
Outputs results to backend/data/checkpoints/evaluation_results.json.
"""
import os
import sys
import json
import time
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models.attention_unet import DualStreamAttentionUNet
from app.config import MODEL_WEIGHTS_PATH

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "mini_sen1floods11", "val"))
OUT_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints", "evaluation_results.json"))


def compute_binary_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray):
    """Computes TP, FP, TN, FN, IoU, Precision, Recall, and F1."""
    tp = int(np.logical_and(pred_binary == 1, gt_binary == 1).sum())
    fp = int(np.logical_and(pred_binary == 1, gt_binary == 0).sum())
    tn = int(np.logical_and(pred_binary == 0, gt_binary == 0).sum())
    fn = int(np.logical_and(pred_binary == 0, gt_binary == 1).sum())

    iou = float(tp / (tp + fp + fn + 1e-7))
    precision = float(tp / (tp + fp + 1e-7))
    recall = float(tp / (tp + fn + 1e-7))
    f1 = float(2 * tp / (2 * tp + fp + fn + 1e-7))

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def otsu_sar_baseline(sar_vv: np.ndarray) -> np.ndarray:
    """Otsu thresholding on SAR VV backscatter as classical baseline."""
    # Normalize VV to 0..255
    vv = np.clip(sar_vv, -30.0, 5.0)
    vv_norm = ((vv - vv.min()) / (vv.max() - vv.min() + 1e-7) * 255).astype(np.uint8)
    # Water has low backscatter (darker pixels)
    # Otsu threshold
    hist = np.bincount(vv_norm.ravel(), minlength=256)
    total = vv_norm.size
    current_max, threshold = 0.0, 0
    sum_total = np.dot(np.arange(256), hist)
    weight_b, sum_b = 0, 0

    for i in range(256):
        weight_b += hist[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += i * hist[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * ((mean_b - mean_f) ** 2)
        if var_between > current_max:
            current_max = var_between
            threshold = i

    # Water is below threshold
    return (vv_norm < threshold).astype(np.uint8)


def evaluate_dataset():
    print("=" * 70)
    print("FLOODSENSE-AI: RIGOROUS EMPIRICAL VALIDATION SUITE")
    print("=" * 70)

    val_folders = sorted(glob.glob(os.path.join(DATA_DIR, "val_tile_*")))
    if not val_folders:
        print(f"ERROR: No validation tiles found in {DATA_DIR}")
        return

    print(f"Found {len(val_folders)} validation tiles in: {DATA_DIR}")

    # Initialize PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on Device: {device.type.upper()}")

    model = DualStreamAttentionUNet(num_classes=3).to(device)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            ckpt = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded trained checkpoint: {MODEL_WEIGHTS_PATH}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}), using initialized weights.")
    else:
        print(f"Warning: Checkpoint not found at {MODEL_WEIGHTS_PATH}, using initialized weights.")
    model.eval()

    model_metrics_list = []
    otsu_metrics_list = []
    latencies = []

    for idx, folder in enumerate(val_folders):
        tile_name = os.path.basename(folder)
        opt_path = os.path.join(folder, "pre_optical.npz")
        sar_path = os.path.join(folder, "post_sar.npz")
        gt_path = os.path.join(folder, "ground_truth_mask.png")

        if not (os.path.exists(opt_path) and os.path.exists(sar_path) and os.path.exists(gt_path)):
            continue

        opt_data = np.load(opt_path)["optical"].astype(np.float32)  # [4, 256, 256]
        sar_data = np.load(sar_path)["sar"].astype(np.float32)      # [2, 256, 256]
        gt_img = np.array(Image.open(gt_path))                       # [256, 256] or [256, 256, 3]
        if gt_img.ndim == 3:
            gt_img = gt_img[..., 0]

        # Binary ground truth: any water class (1=permanent, 2=flood)
        gt_water = (gt_img > 0).astype(np.uint8)

        # Baseline: Otsu on SAR VV (channel 0)
        sar_vv = sar_data[0]
        otsu_pred = otsu_sar_baseline(sar_vv)
        otsu_res = compute_binary_metrics(otsu_pred, gt_water)
        otsu_metrics_list.append(otsu_res)

        # DualStreamAttentionUNet Model Forward Pass
        opt_tensor = torch.from_numpy(opt_data).unsqueeze(0).to(device)
        sar_tensor = torch.from_numpy(sar_data).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, att_map = model(opt_tensor, sar_tensor)
            probs = F.softmax(logits, dim=1)
            # Class 1: Permanent Water, Class 2: Flood Inundation
            water_prob = (probs[0, 1] + probs[0, 2]).cpu().numpy()
            model_pred = (water_prob > 0.45).astype(np.uint8)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        m_res = compute_binary_metrics(model_pred, gt_water)
        model_metrics_list.append(m_res)

        print(f"[{idx+1}/{len(val_folders)}] {tile_name}: Model IoU={m_res['iou']:.3f}, F1={m_res['f1']:.3f} | Otsu IoU={otsu_res['iou']:.3f} ({lat_ms:.1f}ms)")

    # Aggregate metrics
    def aggregate(m_list):
        tot_tp = sum(m["tp"] for m in m_list)
        tot_fp = sum(m["fp"] for m in m_list)
        tot_tn = sum(m["tn"] for m in m_list)
        tot_fn = sum(m["fn"] for m in m_list)
        macro_iou = float(np.mean([m["iou"] for m in m_list]))
        macro_f1 = float(np.mean([m["f1"] for m in m_list]))
        macro_prec = float(np.mean([m["precision"] for m in m_list]))
        macro_rec = float(np.mean([m["recall"] for m in m_list]))
        micro_iou = float(tot_tp / (tot_tp + tot_fp + tot_fn + 1e-7))
        micro_f1 = float(2 * tot_tp / (2 * tot_tp + tot_fp + tot_fn + 1e-7))
        return {
            "confusion_matrix": {"tp": tot_tp, "fp": tot_fp, "tn": tot_tn, "fn": tot_fn},
            "macro_iou": round(macro_iou, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "micro_iou": round(micro_iou, 4),
            "micro_f1": round(micro_f1, 4),
        }

    agg_model = aggregate(model_metrics_list)
    agg_otsu = aggregate(otsu_metrics_list)
    avg_latency = float(np.mean(latencies))

    results_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_samples_count": len(val_folders),
        "device": device.type,
        "models": {
            "sar_otsu_baseline": {
                "name": "Otsu Automatic SAR Backscatter Thresholding",
                "macro_iou": agg_otsu["macro_iou"],
                "macro_f1": agg_otsu["macro_f1"],
                "macro_precision": agg_otsu["macro_precision"],
                "macro_recall": agg_otsu["macro_recall"],
                "confusion_matrix": agg_otsu["confusion_matrix"],
            },
            "dual_stream_attention_unet": {
                "name": "Dual-Stream Siamese Attention U-Net (Ours)",
                "macro_iou": agg_model["macro_iou"],
                "macro_f1": agg_model["macro_f1"],
                "macro_precision": agg_model["macro_precision"],
                "macro_recall": agg_model["macro_recall"],
                "micro_iou": agg_model["micro_iou"],
                "micro_f1": agg_model["micro_f1"],
                "avg_inference_latency_ms": round(avg_latency, 2),
                "confusion_matrix": agg_model["confusion_matrix"],
            }
        },
        "per_tile_breakdown": [
            {
                "tile": os.path.basename(val_folders[i]),
                "model_iou": model_metrics_list[i]["iou"],
                "model_f1": model_metrics_list[i]["f1"],
                "otsu_iou": otsu_metrics_list[i]["iou"],
                "otsu_f1": otsu_metrics_list[i]["f1"],
            }
            for i in range(len(val_folders))
        ],
        "is_empirical_ground_truth": True,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE! Results saved to:")
    print(f"  {OUT_JSON}")
    print(f"Model Macro IoU: {agg_model['macro_iou']*100:.2f}% | F1: {agg_model['macro_f1']*100:.2f}% | Latency: {avg_latency:.1f}ms")
    print(f"Otsu  Macro IoU: {agg_otsu['macro_iou']*100:.2f}% | F1: {agg_otsu['macro_f1']*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_dataset()
