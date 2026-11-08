"""GPU Technical Benchmark for LightweightDualStreamAttentionUNet on NVIDIA RTX 4070.

Runs GPU-only benchmarks with:
- Batch sizes 1, 2, and 4
- FP32 Full Precision and FP16 Automatic Mixed Precision (AMP)
- 5 warm-up iterations and 10 measured iterations
- Exact torch.cuda.synchronize() latency timing
- Peak allocated and reserved GPU VRAM telemetry
- NaN / Inf checks for outputs and gradients
"""

import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import torch
import torch.nn as nn
from app.models.lightweight_attention_unet import LightweightDualStreamAttentionUNet
from app.models.losses import CombinedFloodLoss


def benchmark_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Refusing to run CPU fallback in GPU benchmark.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    compute_cap = torch.cuda.get_device_capability(0)

    print("=" * 80)
    print("FLOODSENSE-AI: NVIDIA RTX 4070 GPU MODEL BENCHMARK")
    print(f"Device: {gpu_name} | Compute Cap: {compute_cap} | VRAM: {total_vram_gb:.2f} GB")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print("=" * 80)

    model = LightweightDualStreamAttentionUNet().to(device)
    info = model.get_parameter_count()
    print(f"Model Parameters: {info['total_parameters']:,} ({info['total_parameters']/1e6:.2f} M)")
    print(f"Model Weight Size (FP32): {info['model_size_float32_mb']} MB")
    print("-" * 80)

    criterion = CombinedFloodLoss(num_classes=3, ignore_index=255).to(device)
    batch_sizes = [1, 2, 4]
    precision_modes = [
        {"name": "FP32 (Single Precision)", "amp": False},
        {"name": "FP16 AMP (Mixed Precision)", "amp": True},
    ]

    benchmark_records = []

    for mode in precision_modes:
        amp_enabled = mode["amp"]
        mode_name = mode["name"]
        print(f"\n>>> Running Precision Mode: {mode_name}")
        print(f"{'Batch':<7} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Total (ms)':<11} | {'Alloc VRAM':<12} | {'Resv VRAM':<12} | {'Finite'}")
        print("-" * 80)

        for b in batch_sizes:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
            scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

            # Generate dummy inputs and targets with 15% NoData pixels
            opt = torch.randn(b, 4, 256, 256, device=device)
            sar = torch.randn(b, 2, 256, 256, device=device)
            target = torch.randint(0, 3, (b, 256, 256), device=device, dtype=torch.long)
            target[torch.rand(b, 256, 256, device=device) < 0.15] = 255

            # 5 Warm-Up Passes
            model.train()
            for _ in range(5):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    out, _ = model(opt, sar)
                    loss = criterion(out, target)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize(device)

            # Reset Peak Memory Stats
            torch.cuda.reset_peak_memory_stats(device)

            # 10 Measured Passes
            fwd_times, bwd_times, total_times = [], [], []

            for _ in range(10):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)

                # Forward Timing
                t0 = time.perf_counter()
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    out, att = model(opt, sar)
                    loss = criterion(out, target)
                torch.cuda.synchronize(device)
                t_fwd = time.perf_counter() - t0

                # Backward Timing
                t1 = time.perf_counter()
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize(device)
                t_bwd = time.perf_counter() - t1

                fwd_times.append(t_fwd * 1000)
                bwd_times.append(t_bwd * 1000)
                total_times.append((t_fwd + t_bwd) * 1000)

            # Check Finite Outputs and Gradients
            out_finite = torch.isfinite(out).all().item()
            loss_finite = torch.isfinite(loss).item()
            grads_finite = all(
                torch.isfinite(p.grad).all().item()
                for p in model.parameters()
                if p.grad is not None
            )

            if not (out_finite and loss_finite and grads_finite):
                raise ValueError("Encountered NaN or Inf in outputs or gradients!")

            peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_resv_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

            fwd_mean = sum(fwd_times) / len(fwd_times)
            bwd_mean = sum(bwd_times) / len(bwd_times)
            tot_mean = sum(total_times) / len(total_times)

            record = {
                "mode": mode_name,
                "amp": amp_enabled,
                "batch_size": b,
                "fwd_mean_ms": round(fwd_mean, 2),
                "fwd_min_ms": round(min(fwd_times), 2),
                "fwd_max_ms": round(max(fwd_times), 2),
                "bwd_mean_ms": round(bwd_mean, 2),
                "bwd_min_ms": round(min(bwd_times), 2),
                "bwd_max_ms": round(max(bwd_times), 2),
                "total_step_ms": round(tot_mean, 2),
                "peak_alloc_mb": round(peak_alloc_mb, 2),
                "peak_resv_mb": round(peak_resv_mb, 2),
                "finite": out_finite and loss_finite and grads_finite,
            }
            benchmark_records.append(record)

            print(
                f"{b:<7} | {fwd_mean:>6.2f} ms  | {bwd_mean:>6.2f} ms  | {tot_mean:>7.2f} ms  | "
                f"{peak_alloc_mb:>8.2f} MB  | {peak_resv_mb:>8.2f} MB  | PASS (Finite)"
            )

    print("\n" + "=" * 80)
    print("ALL GPU BENCHMARK PASSES COMPLETED SUCCESSFULLY ON NVIDIA RTX 4070")
    print("=" * 80)
    return benchmark_records


if __name__ == "__main__":
    benchmark_gpu()
