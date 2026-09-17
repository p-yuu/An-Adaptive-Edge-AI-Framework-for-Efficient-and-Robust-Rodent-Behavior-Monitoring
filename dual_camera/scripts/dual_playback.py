import csv
from pathlib import Path
import cv2
import numpy as np
import sys


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent / "result"

# ==============================================================================
# SESSION DIRECTORY
# ==============================================================================

if len(sys.argv) < 2:
    print("[ERROR] No session directory provided.")
    print("Usage:")
    print("python dual_playback.py <session_directory>")
    sys.exit(1)

session_argument = Path(sys.argv[1])
SESSION_DIR = (
    session_argument
    if session_argument.is_absolute()
    else RESULT_ROOT / session_argument
).resolve()

print(f"[PLAYBACK] Session directory:")
print(SESSION_DIR)

RGB_VIDEO = SESSION_DIR / "rgb" / "rgb.mp4"

# Change this to the session you want to inspect
# SESSION_DIR = RESULT_ROOT / "20260913_232401"
# RGB_VIDEO = SESSION_DIR / "rgb" / "rgb.mp4"

# Real per-frame RGB timestamps from Luckfox V4L2
RGB_FRAME_TIMESTAMP_CSV = (SESSION_DIR / "rgb" / "rgb_frame_timestamps.csv")
THERMAL_DIR = SESSION_DIR / "thermal"

# Real per-frame thermal timestamps from Luckfox CLOCK_MONOTONIC
THERMAL_TIMESTAMP_CSV = (THERMAL_DIR / "thermal_frame_timestamps.csv")

# Output
OUTPUT_VIDEO = SESSION_DIR / "dual_playback_actual_timestamps.mp4"
PAIRING_CSV = SESSION_DIR / "timestamp_pairing.csv"

# Each camera occupies one 640x480 panel
PANEL_WIDTH = 640
PANEL_HEIGHT = 480

OUTPUT_WIDTH = PANEL_WIDTH * 2
OUTPUT_HEIGHT = PANEL_HEIGHT

# Slower output for visual inspection.
# This affects playback speed only. It does NOT affect timestamp pairing.
OUTPUT_FPS = 15.0

# ==============================================================================
# LOAD RGB PER-FRAME TIMESTAMPS
# ==============================================================================

def load_rgb_timestamps():
    records = []
    with open(RGB_FRAME_TIMESTAMP_CSV,mode="r",newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append({
                "frame_id": int(row["frame_id"]),
                "buffer_id": int(row["buffer_id"]),
                "sequence": int(row["sequence"]),
                "timestamp_s": float(
                    row["timestamp_monotonic_s"]
                )
            })

    return records

# ==============================================================================
# LOAD THERMAL PER-FRAME TIMESTAMPS
# ==============================================================================
def load_thermal_timestamps():
    records = []
    with open(THERMAL_TIMESTAMP_CSV,mode="r",newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            frame_id = int(row["frame_id"])

            records.append({
                "frame_id": frame_id,
                "subpage_id": int(row["subpage_id"]),
                "timestamp_s": float(row["new_data_timestamp_monotonic_s"]),
                "read_start_s": float(row["read_start_monotonic_s"]),
                "read_midpoint_s": float(row["timestamp_monotonic_s"]),
                "read_end_s": float(row["read_end_monotonic_s"]),
                "read_latency_ms": float(row["read_latency_ms"]),
                "filename": (f"frame_{frame_id:04d}.npy")
            })

    return records

# ==============================================================================
# RESIZE WITHOUT CROPPING
# ==============================================================================
def fit_with_letterbox(image,target_width,target_height):
    height, width = image.shape[:2]

    scale = min(target_width / width,target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(image,(new_width, new_height),interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height,target_width,3),dtype=np.uint8)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[y_offset:y_offset + new_height,x_offset:x_offset + new_width] = resized

    return canvas

# ==============================================================================
# THERMAL VISUALIZATION
# ==============================================================================
def create_thermal_image(frame):
    # Visualization only.
    # The original NPY thermal values are not modified.
    frame_norm = cv2.normalize(frame,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    frame_resized = cv2.resize(frame_norm,(640, 480),interpolation=cv2.INTER_CUBIC)
    frame_clean = cv2.bilateralFilter(frame_resized,d=7,sigmaColor=30,sigmaSpace=30)
    frame_colored = cv2.applyColorMap(frame_clean,cv2.COLORMAP_MAGMA)

    return frame_colored

# ==============================================================================
# FIND NEAREST THERMAL FRAME
# ==============================================================================
def find_nearest_thermal(rgb_time_s,thermal_records):
    nearest = min(thermal_records,key=lambda record:abs(record["timestamp_s"] - rgb_time_s))

    return nearest

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("======================================")
    print(" RGB + Thermal Dual Playback Generator")
    print("   Using real Luckfox timestamps")
    print("======================================")

    # --------------------------------------------------------------------------
    # Check required files
    # --------------------------------------------------------------------------
    required_files = [RGB_VIDEO,RGB_FRAME_TIMESTAMP_CSV,THERMAL_TIMESTAMP_CSV]

    for path in required_files:
        if not path.exists():
            print("[ERROR] Required file not found:")
            print(path)
            return

    # --------------------------------------------------------------------------
    # Load real timestamps
    # --------------------------------------------------------------------------
    rgb_records = load_rgb_timestamps()
    thermal_records = load_thermal_timestamps()

    if not rgb_records:
        print("[ERROR] No RGB timestamps found.")
        return

    if not thermal_records:
        print("[ERROR] No thermal timestamps found.")
        return

    # Verify thermal NPY files exist
    valid_thermal_records = []

    for record in thermal_records:
        thermal_file = (THERMAL_DIR / record["filename"])

        if thermal_file.exists():
            valid_thermal_records.append(record)
        else:
            print("[WARNING] Missing thermal frame:")
            print(thermal_file)

    thermal_records = valid_thermal_records

    if not thermal_records:
        print("[ERROR] No valid thermal NPY frames found.")
        return

    # --------------------------------------------------------------------------
    # Open RGB video
    # --------------------------------------------------------------------------
    cap = cv2.VideoCapture(str(RGB_VIDEO))

    if not cap.isOpened():
        print("[ERROR] Cannot open RGB video.")
        return

    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    usable_rgb_frames = min(video_frame_count,len(rgb_records))

    print(f"RGB video frames     : {video_frame_count}")
    print(f"RGB timestamp records: {len(rgb_records)}")
    print(f"Usable RGB frames    : {usable_rgb_frames}")
    print(f"RGB encoded FPS      : {original_fps:.2f}")
    print(f"Thermal frames       : {len(thermal_records)}")
    print(f"Output FPS           : {OUTPUT_FPS:.2f}")

    rgb_first = rgb_records[0]["timestamp_s"]
    rgb_last = rgb_records[usable_rgb_frames - 1]["timestamp_s"]

    thermal_first = thermal_records[0]["timestamp_s"]
    thermal_last = thermal_records[-1]["timestamp_s"]

    overlap_start = max(rgb_first,thermal_first)
    overlap_end = min(rgb_last,thermal_last)

    print()
    print(f"RGB range     : "f"{rgb_first:.6f} -> {rgb_last:.6f}")
    print(f"Thermal range : {thermal_first:.6f} -> {thermal_last:.6f}")

    if overlap_start <= overlap_end:
        print(f"Overlap       : {overlap_start:.6f} -> {overlap_end:.6f}")
    else:
        print("[WARNING] RGB and thermal timestamp ranges do not overlap.")
    print()

    # --------------------------------------------------------------------------
    # Create output MP4
    # --------------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO),fourcc,OUTPUT_FPS,(OUTPUT_WIDTH,OUTPUT_HEIGHT))

    if not writer.isOpened():
        print("[ERROR] Cannot create output video.")
        cap.release()
        return

    # --------------------------------------------------------------------------
    # Pairing CSV
    # --------------------------------------------------------------------------
    pairing_file = open(PAIRING_CSV,mode="w",newline="")
    pairing_writer = csv.writer(pairing_file)

    pairing_writer.writerow([
        "rgb_frame_id",
        "rgb_sequence",
        "rgb_timestamp_monotonic_s",
        "thermal_frame_id",
        "thermal_subpage_id",
        "thermal_new_data_timestamp_monotonic_s",
        "thermal_read_midpoint_monotonic_s",
        "thermal_read_latency_ms",
        "delta_ms",
        "abs_delta_ms",
        "rgb_inside_thermal_range"
    ])

    # --------------------------------------------------------------------------
    # Process every usable RGB frame
    # --------------------------------------------------------------------------
    abs_deltas_ms = []

    for video_frame_id in range(usable_rgb_frames):
        success, rgb_frame = cap.read()
        if not success:
            print("[WARNING] RGB video ended early.")
            break

        rgb_record = rgb_records[video_frame_id]
        rgb_time_s = rgb_record["timestamp_s"]

        thermal_record = (find_nearest_thermal(rgb_time_s,thermal_records))
        thermal_time_s = thermal_record["timestamp_s"]
        thermal_file = (THERMAL_DIR / thermal_record["filename"])
        thermal_frame = np.load(thermal_file)
        thermal_image = (create_thermal_image(thermal_frame))

        rgb_panel = fit_with_letterbox(rgb_frame,PANEL_WIDTH,PANEL_HEIGHT)
        thermal_panel = (fit_with_letterbox(thermal_image,PANEL_WIDTH,PANEL_HEIGHT))

        # Thermal - RGB
        delta_ms = (thermal_time_s - rgb_time_s) * 1000.0
        abs_delta_ms = abs(delta_ms)
        abs_deltas_ms.append(abs_delta_ms)
        inside_thermal_range = (thermal_first <= rgb_time_s <= thermal_last)

        # ----------------------------------------------------------------------
        # RGB overlay
        # ----------------------------------------------------------------------
        cv2.putText(rgb_panel,"RGB",(20, 35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2)
        cv2.putText(rgb_panel,f"Frame: {rgb_record['frame_id']}",(20, 70),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 255),2)
        cv2.putText(rgb_panel,f"Seq: {rgb_record['sequence']}",(20, 100),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 255),2)
        cv2.putText(rgb_panel,f"t: {rgb_time_s:.6f} s",(20, 130),cv2.FONT_HERSHEY_SIMPLEX,0.60,(255, 255, 255),2)

        # ----------------------------------------------------------------------
        # Thermal overlay
        # ----------------------------------------------------------------------
        cv2.putText(thermal_panel,"THERMAL",(20, 35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2)
        cv2.putText(
            thermal_panel,
            f"Frame: {thermal_record['frame_id']}  Subpage: {thermal_record['subpage_id']}",
            (20, 70),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 255),2
        )
        cv2.putText(
            thermal_panel,
            f"new-data t: {thermal_time_s:.6f} s",
            (20, 100),cv2.FONT_HERSHEY_SIMPLEX,0.60,(255, 255, 255),2
        )
        cv2.putText(thermal_panel,f"dt: {delta_ms:+.1f} ms",(20, 130),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 255),2)
        cv2.putText(thermal_panel,("Range: "+ ("overlap" if inside_thermal_range else "outside")),(20, 160),cv2.FONT_HERSHEY_SIMPLEX,0.60,(255, 255, 255),2)

        combined = np.hstack((rgb_panel,thermal_panel))
        writer.write(combined)

        pairing_writer.writerow([
            rgb_record["frame_id"],
            rgb_record["sequence"],
            f"{rgb_time_s:.9f}",
            thermal_record["frame_id"],
            thermal_record["subpage_id"],
            f"{thermal_time_s:.9f}",
            f"{thermal_record['read_midpoint_s']:.9f}",
            f"{thermal_record['read_latency_ms']:.3f}",
            f"{delta_ms:.3f}",
            f"{abs_delta_ms:.3f}",
            inside_thermal_range
        ])

        print(
            f"[{video_frame_id + 1:02d}/"
            f"{usable_rgb_frames:02d}] "
            f"RGB frame "
            f"{rgb_record['frame_id']:02d} "
            f"{rgb_time_s:.6f}s "
            f"<-> "
            f"Thermal frame "
            f"{thermal_record['frame_id']:02d} "
            f"{thermal_time_s:.6f}s "
            f"(dt={delta_ms:+.1f}ms)"
        )

    # --------------------------------------------------------------------------
    # Finish
    # --------------------------------------------------------------------------

    pairing_file.close()
    cap.release()
    writer.release()

    print()
    print("======================================")
    print(" Done")
    print("======================================")
    print("Output video:")
    print(OUTPUT_VIDEO)
    print()
    print("Pairing CSV:")
    print(PAIRING_CSV)

    if abs_deltas_ms:
        abs_deltas = np.array(abs_deltas_ms)
        print()
        print("Timestamp pairing statistics")
        print(f"Mean |dt|   : {np.mean(abs_deltas):.2f} ms")
        print(f"Median |dt| : {np.median(abs_deltas):.2f} ms")
        print(f"Max |dt|    : {np.max(abs_deltas):.2f} ms")

if __name__ == "__main__":
    main()
