import os
import csv
import time
import threading
import subprocess

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULT_ROOT = SCRIPT_DIR.parent / "result"

ADB_EXE = PROJECT_ROOT / "2_tools" / "adb_fastboot" / "adb.exe"

# Luckfox RGB capture
RGB_SCRIPT_LUCKFOX = "/root/capture_with_time.py"
RGB_FILE_LUCKFOX = "/tmp/video20.yuv"
RGB_FRAME_TIMESTAMP_FILE_LUCKFOX = "/tmp/rgb_frame_timestamps.csv"

# Luckfox Thermal capture
THERMAL_SCRIPT_LUCKFOX = "/root/thermal_read_with_time.py"
THERMAL_RAW_LUCKFOX = "/tmp/thermal_frames.raw"
THERMAL_TIMESTAMP_FILE_LUCKFOX = "/tmp/thermal_frame_timestamps.csv"

# ==============================================================================
# RGB CONFIGURATION
# ==============================================================================
RGB_WIDTH = 640
RGB_HEIGHT = 480
RGB_FRAME_COUNT = 30
RGB_OUTPUT_FPS = 15

# ==============================================================================
# THERMAL CONFIGURATION
# ==============================================================================
THERMAL_FRAME_COUNT = 30
THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24
THERMAL_FRAME_BYTES = 1536

# ==============================================================================
# CREATE SESSION DIRECTORY
# ==============================================================================

SESSION_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

SESSION_DIR = RESULT_ROOT / SESSION_TIME

RGB_DIR = SESSION_DIR / "rgb"
THERMAL_DIR = SESSION_DIR / "thermal"

RGB_DIR.mkdir(parents=True, exist_ok=True)
THERMAL_DIR.mkdir(parents=True, exist_ok=True)

RGB_YUV_LOCAL = RGB_DIR / "video20.yuv"
RGB_MP4_LOCAL = RGB_DIR / "rgb.mp4"

RGB_TIMESTAMP_CSV = RGB_DIR / "rgb_timestamps.csv"
RGB_FRAME_TIMESTAMP_CSV = (RGB_DIR / "rgb_frame_timestamps.csv")
RGB_LOG_FILE = RGB_DIR / "capture_log.txt"

THERMAL_RAW_LOCAL = THERMAL_DIR / "thermal_frames.raw"
THERMAL_TIMESTAMP_CSV = (THERMAL_DIR / "thermal_frame_timestamps.csv")

# ==============================================================================
# THREAD SYNCHRONIZATION
# ==============================================================================
start_event = threading.Event()

experiment_start_ns = None

# ==============================================================================
# THERMAL FUNCTIONS
# ==============================================================================
def capture_thermal_remote():
    print("[THERMAL] Triggering ""/root/thermal_read_with_time.py...")
    command = [str(ADB_EXE),"shell",f"python3 {THERMAL_SCRIPT_LUCKFOX}"]

    try:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8")

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                print(f"[THERMAL] {line.strip()}")

        return process.returncode == 0

    except Exception as e:
        print(f"[THERMAL] Capture error: {e}")
        return False

def pull_thermal_raw_file():
    print("[THERMAL] Pulling RAW file...")
    command = [str(ADB_EXE),"pull",THERMAL_RAW_LUCKFOX,str(THERMAL_RAW_LOCAL)]

    try:
        result = subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)

        if (result.returncode == 0 and THERMAL_RAW_LOCAL.exists()):
            print("[THERMAL] RAW transferred successfully.")
            return True

        print("[THERMAL] Failed to pull RAW file.")

        if result.stderr:
            print(result.stderr)

        return False

    except Exception as e:
        print(f"[THERMAL] RAW pull error: {e}")
        return False

def pull_thermal_timestamp_file():
    print("[THERMAL] Pulling timestamp CSV...")
    command = [str(ADB_EXE),"pull",THERMAL_TIMESTAMP_FILE_LUCKFOX,str(THERMAL_TIMESTAMP_CSV)]

    try:
        result = subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)

        if (result.returncode == 0 and THERMAL_TIMESTAMP_CSV.exists()):
            print("[THERMAL] Timestamp CSV transferred successfully.")
            return True

        print("[THERMAL] Failed to pull timestamp CSV.")

        if result.stderr:
            print(result.stderr)

        return False

    except Exception as e:
        print(f"[THERMAL] Timestamp pull error: {e}")
        return False

def convert_thermal_raw():
    if not THERMAL_RAW_LOCAL.exists():
        print("[THERMAL] RAW file does not exist.")
        return False

    frame_bytes = THERMAL_FRAME_BYTES
    file_size = os.path.getsize(THERMAL_RAW_LOCAL)

    if file_size == 0:
        print("[THERMAL] RAW file is empty.")
        return False

    if file_size % frame_bytes != 0:
        print("[THERMAL] Warning: RAW file size is not a multiple of 1536.")
    frame_count = file_size // frame_bytes

    print(f"[THERMAL] RAW size: {file_size} bytes")
    print(f"[THERMAL] Frames detected: {frame_count}")

    with open(THERMAL_RAW_LOCAL,"rb") as file:
        for frame_id in range(frame_count):
            raw = file.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            pixels = np.frombuffer(raw,dtype="<i2")
            frame = pixels.reshape((THERMAL_HEIGHT, THERMAL_WIDTH))

            # Same canonical orientation established
            # in the previous thermal tests.
            frame = np.flipud(frame)
            filename = (f"frame_{frame_id:04d}.npy")
            np.save(THERMAL_DIR / filename,frame)

    print(f"[THERMAL] Converted {frame_count} frames to NPY.")
    return True

# ==============================================================================
# RGB FUNCTIONS
# ==============================================================================
def capture_rgb_remote():
    print("[RGB] Triggering /root/capture_with_time.py...")
    command = [str(ADB_EXE), "shell", f"python3 {RGB_SCRIPT_LUCKFOX}"]

    try:
        with open(RGB_LOG_FILE,mode="w",encoding="utf-8") as log_file:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    line = line.strip()
                    print(f"[RGB] {line}")
                    log_file.write(line + "\n")

        return process.returncode == 0

    except Exception as e:
        print(f"[RGB] Capture error: {e}")
        return False


def pull_rgb_file():
    print("[RGB] Pulling YUV file...")

    command = [str(ADB_EXE), "pull", RGB_FILE_LUCKFOX, str(RGB_YUV_LOCAL)]

    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if (result.returncode == 0 and RGB_YUV_LOCAL.exists()):
            print("[RGB] YUV transferred successfully.")
            return True
        print("[RGB] Failed to pull YUV.")
        return False

    except Exception as e:
        print(f"[RGB] Pull error: {e}")
        return False

def pull_rgb_frame_timestamp_file():
    print("[RGB] Pulling per-frame timestamp CSV...")

    command = [str(ADB_EXE), "pull", RGB_FRAME_TIMESTAMP_FILE_LUCKFOX, str(RGB_FRAME_TIMESTAMP_CSV)]

    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        if (result.returncode == 0 and RGB_FRAME_TIMESTAMP_CSV.exists()):
            print("[RGB] Per-frame timestamps transferred successfully.")
            return True

        print("[RGB] Failed to pull per-frame timestamp CSV.")

        if result.stderr:
            print(result.stderr)

        return False

    except Exception as e:
        print(f"[RGB] Timestamp pull error: {e}")
        return False

def convert_rgb_yuv():
    if not RGB_YUV_LOCAL.exists():
        return False

    frame_size = int(RGB_WIDTH * RGB_HEIGHT * 1.5)
    file_size = os.path.getsize(RGB_YUV_LOCAL)

    if file_size == 0:
        print("[RGB] YUV file is empty.")
        return False

    print(f"[RGB] YUV size: {file_size} bytes")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(RGB_MP4_LOCAL), fourcc, RGB_OUTPUT_FPS, (RGB_WIDTH, RGB_HEIGHT))

    with open(RGB_YUV_LOCAL, "rb") as file:
        for i in range(RGB_FRAME_COUNT):
            raw = file.read(frame_size)
            if len(raw) < frame_size:
                break

            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape((int(RGB_HEIGHT * 1.5), RGB_WIDTH))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

            writer.write(bgr)

    writer.release()
    print(f"[RGB] MP4 created: {RGB_MP4_LOCAL}")
    return True

# ==============================================================================
# RGB THREAD
# ==============================================================================
def rgb_worker():
    print("[RGB] Thread ready.")

    start_event.wait()
    capture_start_ns = time.perf_counter_ns()

    capture_start_s = (capture_start_ns - experiment_start_ns) / 1e9
    print(f"[RGB] Capture start = " f"{capture_start_s:.6f} s")

    success = capture_rgb_remote()

    capture_end_ns = time.perf_counter_ns()
    capture_end_s = (capture_end_ns - experiment_start_ns) / 1e9
    print(f"[RGB] Capture end = " f"{capture_end_s:.6f} s")

    duration_s = (capture_end_ns - capture_start_ns) / 1e9

    with open(RGB_TIMESTAMP_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "capture_start_s",
            "capture_end_s",
            "duration_s",
            "expected_frames",
            "success"
        ])

        writer.writerow([
            f"{capture_start_s:.9f}",
            f"{capture_end_s:.9f}",
            f"{duration_s:.9f}",
            RGB_FRAME_COUNT,
            success
        ])

    if not success:
        print("[RGB] Remote capture failed.")
        return

    if not pull_rgb_file():
        print("[RGB] YUV pull failed.")
        return

    if not pull_rgb_frame_timestamp_file():
        print("[RGB] Frame timestamp pull failed.")
        return

    convert_rgb_yuv()

# ==============================================================================
# THERMAL THREAD
# ==============================================================================
def thermal_worker():
    print("[THERMAL] Thread ready.")
    start_event.wait()
    print("[THERMAL] Capture starting...")

    success = capture_thermal_remote()
    if not success:
        print("[THERMAL] Remote capture failed.")
        return

    if not pull_thermal_raw_file():
        return

    if not pull_thermal_timestamp_file():
        return

    if not convert_thermal_raw():
        return

    print("[THERMAL] Processing completed.")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global experiment_start_ns
    print("======================================")
    print(" RGB + MLX90640 Temporal Test")
    print("======================================")

    print(f"Session directory:")
    print(SESSION_DIR)
    print()

    if not ADB_EXE.exists():
        print(f"ADB executable not found:\n"f"{ADB_EXE}")
        return

    rgb_thread = threading.Thread(target=rgb_worker, name="RGBThread")
    thermal_thread = threading.Thread(target=thermal_worker, name="ThermalThread")

    rgb_thread.start()
    thermal_thread.start()

    # Give both threads time to reach start_event.wait()
    time.sleep(0.2)
    experiment_start_ns = time.perf_counter_ns()

    print()
    print("[MAIN] Starting both cameras...")
    print()

    start_event.set()
    rgb_thread.join()
    thermal_thread.join()

    print()
    print("======================================")
    print(" Test finished")
    print("======================================")
    print(f"Results:")
    print(SESSION_DIR)

    print()
    print("======================================")
    print(" Generating dual playback")
    print("======================================")

    PLAYBACK_SCRIPT = SCRIPT_DIR / "dual_playback.py"

    if not PLAYBACK_SCRIPT.exists():
        print("[ERROR] dual_playback.py not found:")
        print(PLAYBACK_SCRIPT)
        return

    command = ["python",str(PLAYBACK_SCRIPT),str(SESSION_DIR)]
    result = subprocess.run(command)

    if result.returncode != 0:
        print("[ERROR] Dual playback generation failed.")
        return

    print("======================================")
    print(" All processing completed")
    print("======================================")
    print("Session:")
    print(SESSION_DIR)
    print("Playback:")
    print(SESSION_DIR / "dual_playback_actual_timestamps.mp4")

if __name__ == "__main__":
    main()