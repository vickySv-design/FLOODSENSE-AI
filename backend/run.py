"""
Entrypoint script for launching the FloodSense-AI FastAPI Backend Server.
Automatically initializes synthetic Sentinel-1/2 demo presets and PyTorch weights on startup.
"""
import uvicorn
from scripts.init_demo_data import ensure_demo_data_initialized


if __name__ == "__main__":
    print("=" * 72)
    print("  FLOODSENSE-AI // DUAL-STREAM SAR & OPTICAL DISASTER ENGINE")
    print("  Initializing Demo Satellite Tiles, Road Graphs & PyTorch Weights...")
    print("=" * 72)
    summary = ensure_demo_data_initialized()
    print(f"  [OK] Presets Ready: {', '.join(summary['presets_available'])}")
    print(f"  [OK] Checkpoint:    {summary['checkpoint_path']}")
    print("  [OK] Starting FastAPI Server at http://localhost:8000")
    print("=" * 72)

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

