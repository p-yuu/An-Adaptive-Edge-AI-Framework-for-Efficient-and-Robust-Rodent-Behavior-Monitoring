from pathlib import Path
from datetime import datetime
import argparse
import json
import subprocess
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

NAMES = ("top_left", "top_mid", "top_right", "mid_left", "mid",
         "mid_right", "bottom_left", "bottom_mid", "bottom_right")
ROOT = Path(__file__).resolve().parent / "new_rtsp_9points"

def run(args, timeout=30):
    return subprocess.run(args, capture_output=True, text=True,timeout=timeout, check=True)

def read_thermal():
    command = ["adb", "shell", "i2ctransfer -y 2 w2@0x33 0x04 0x00 r1536"]
    t0 = time.monotonic()
    result = run(command, timeout=30)
    t1 = time.monotonic()
    tokens = result.stdout.split()
    if len(tokens) != 1536:
        raise RuntimeError(f"Expected 1536 I2C bytes, got {len(tokens)}. "
                           f"Output: {result.stdout[:250]!r} {result.stderr[:250]!r}")
    values = np.array([int(token, 16) for token in tokens], dtype=np.uint8)
    raw_u16 = values[0::2].astype(np.uint16) | (values[1::2].astype(np.uint16) << 8)
    raw = raw_u16.view(np.int16).reshape(24, 32).copy()
    return raw, t0, t1

def thermal_preview(raw):
    """Match camera_view.py processing; output is a display-only 160x120 array."""
    frame = raw.astype(float)
    vmin, vmax = np.percentile(frame, [1, 99])
    if vmax > vmin:
        norm = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(frame, dtype=np.uint8)
    norm = np.rot90(np.flipud(norm), k=1)
    enlarged = cv2.resize(norm, (160, 120), interpolation=cv2.INTER_NEAREST)
    # clean = cv2.bilateralFilter(enlarged, d=9, sigmaColor=75, sigmaSpace=75)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # return clahe.apply(clean)
    return enlarged

def capture_rgb(path):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-rtsp_transport", "tcp",
               "-i", "rtsp://127.0.0.1:8554/live/0",
               "-map", "0:v:0", "-frames:v", "1", str(path)]
    t0 = time.monotonic()
    run(command, timeout=35)
    t1 = time.monotonic()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not save an RGB image")
    return t0, t1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", required=True, choices=NAMES)
    args = parser.parse_args()
    run(["adb", "get-state"], timeout=10)
    run(["adb", "forward", "tcp:8554", "tcp:554"], timeout=10)

    folder = ROOT / args.point
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rgb_path = folder / f"{stamp}_rgb.png"
    thermal_path = folder / f"{stamp}_thermal_raw.npy"
    preview_path = folder / f"{stamp}_thermal_oriented.npy"
    preview_png = folder / f"{stamp}_thermal_preview.png"
    meta_path = folder / f"{stamp}_metadata.json"

    print("Keep the calibration target STILL until capture finishes.")
    rgb_start, rgb_end = capture_rgb(rgb_path)
    thermal, thermal_start, thermal_end = read_thermal()
    np.save(thermal_path, thermal)
    np.save(preview_path, np.rot90(np.flipud(thermal), k=1))
    preview = thermal_preview(thermal)
    try:
        colored_rgb = (plt.get_cmap("magma")(preview / 255.0)[:, :, :3] * 255).astype(np.uint8)
        cv2.imwrite(str(preview_png), cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR))
    except ImportError:
        cv2.imwrite(str(preview_png), preview)

    meta = {
        "point": args.point,
        "rgb_path": rgb_path.name,
        "thermal_raw_path": thermal_path.name,
        "thermal_oriented_path": preview_path.name,
        "thermal_preview_path": preview_png.name,
        "thermal_raw_shape": list(thermal.shape),
        "thermal_oriented_shape": [32, 24],
        "rgb_command_interval_monotonic_s": [rgb_start, rgb_end],
        "thermal_command_interval_monotonic_s": [thermal_start, thermal_end],
        "note": "Sequential capture. RGB timestamps are host command boundaries, "
                "not camera exposure timestamps. Thermal read is a raw register snapshot, "
                "not verified as a complete synchronized MLX90640 frame."
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved RGB: {rgb_path}")
    print(f"Saved thermal raw (24x32): {thermal_path}")
    print(f"Saved thermal oriented (32x24): {preview_path}")
    print(f"Saved thermal display preview (160x120): {preview_png}")
    print("WARNING: Not frame-synchronized; hold target still.")

if __name__ == "__main__":
    main()