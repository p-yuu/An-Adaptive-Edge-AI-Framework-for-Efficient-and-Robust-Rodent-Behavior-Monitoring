import csv
import time
from pathlib import Path

import cv2
import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Change this to the session you want to view
SESSION_DIR = SCRIPT_DIR / "result" / "20260910_162702"

THERMAL_DIR = SESSION_DIR / "thermal"
THERMAL_TIMESTAMP_CSV = SESSION_DIR / "thermal_timestamps.csv"

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Playback options
USE_REAL_TIMING = True

# If USE_REAL_TIMING = False, use this FPS instead
PLAYBACK_FPS = 10


# ==============================================================================
# LOAD TIMESTAMPS
# ==============================================================================

def load_thermal_timestamps():
    timestamps = {}

    if not THERMAL_TIMESTAMP_CSV.exists():
        print("[WARNING] thermal_timestamps.csv not found.")
        return timestamps

    with open(THERMAL_TIMESTAMP_CSV, mode="r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row["filename"]
            timestamp = float(row["timestamp_s"])

            timestamps[filename] = timestamp

    return timestamps


# ==============================================================================
# CONVERT THERMAL FRAME FOR DISPLAY
# ==============================================================================

def create_thermal_image(frame):

    # Normalize each thermal frame to 0~255 for visualization
    frame_norm = cv2.normalize(frame,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    # Resize 32x24 -> 640x480
    frame_resized = cv2.resize(frame_norm,(DISPLAY_WIDTH, DISPLAY_HEIGHT),interpolation=cv2.INTER_CUBIC)

    # Smooth the enlarged image
    frame_clean = cv2.bilateralFilter(frame_resized,d=7,sigmaColor=30,sigmaSpace=30)

    # Apply thermal colormap
    frame_colored = cv2.applyColorMap(frame_clean,cv2.COLORMAP_MAGMA)

    return frame_colored


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print("======================================")
    print(" Thermal Session Viewer")
    print("======================================")

    if not THERMAL_DIR.exists():
        print(f"[ERROR] Thermal directory not found:")
        print(THERMAL_DIR)
        return

    frame_files = sorted(THERMAL_DIR.glob("frame_*.npy"))

    if not frame_files:
        print("[ERROR] No .npy thermal frames found.")
        return

    timestamps = load_thermal_timestamps()

    print(f"Session: {SESSION_DIR.name}")
    print(f"Thermal frames: {len(frame_files)}")
    print()
    print("Controls:")
    print("  q = quit")
    print("  space = pause / resume")
    print()

    paused = False

    cv2.namedWindow("Thermal Session",cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Thermal Session",DISPLAY_WIDTH,DISPLAY_HEIGHT)

    for i, frame_file in enumerate(frame_files):
        frame = np.load(frame_file)
        thermal_image = create_thermal_image(frame)

        # ------------------------------------------------------
        # Information overlay
        # ------------------------------------------------------

        timestamp = timestamps.get(frame_file.name)

        info_frame = f"Frame: {i + 1}/{len(frame_files)}"

        cv2.putText(thermal_image,info_frame,(15, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

        if timestamp is not None:

            cv2.putText(thermal_image,f"Time: {timestamp:.3f} s",(15, 60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

        cv2.imshow("Thermal Session",thermal_image)

        # ------------------------------------------------------
        # Calculate playback delay
        # ------------------------------------------------------

        if (USE_REAL_TIMING and timestamp is not None and i < len(frame_files) - 1):
            next_timestamp = timestamps.get(frame_files[i + 1].name)

            if next_timestamp is not None:
                delay_s = next_timestamp - timestamp
                delay_ms = max(1, int(delay_s * 1000))

            else:
                delay_ms = int(1000 / PLAYBACK_FPS)

        else:
            delay_ms = int(1000 / PLAYBACK_FPS)

        # ------------------------------------------------------
        # Keyboard control
        # ------------------------------------------------------

        key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            paused = True

            while paused:
                key = cv2.waitKey(30) & 0xFF

                if key == ord(" "):
                    paused = False

                elif key == ord("q"):
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()

    print()
    print("[DONE] Playback finished.")


if __name__ == "__main__":
    main()