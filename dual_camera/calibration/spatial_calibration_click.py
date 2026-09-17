import csv
from pathlib import Path

import cv2
import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALIBRATION_ROOT = PROJECT_ROOT / "calibration"

# Put the extracted "location" folder beside this script.
DATA_ROOT = CALIBRATION_ROOT / "9-points_test"

SESSION_NAMES = [
    "top_left",
    "top_mid",
    "top_right",
    "mid_left",
    "mid",
    "mid_right",
    "bottom_left",
    "bottom_mid",
    "bottom_right",
]

OUTPUT_CSV = CALIBRATION_ROOT / "calibration_points.csv"

THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24

# Thermal is enlarged with nearest-neighbor interpolation so the original
# 32x24 pixel boundaries remain easy to see.
THERMAL_SCALE = 20
THERMAL_DISPLAY_WIDTH = THERMAL_WIDTH * THERMAL_SCALE   # 640
THERMAL_DISPLAY_HEIGHT = THERMAL_HEIGHT * THERMAL_SCALE # 480

RGB_DISPLAY_WIDTH = 640
RGB_DISPLAY_HEIGHT = 480


# ==============================================================================
# TIMESTAMP LOADERS
# ==============================================================================

def load_rgb_timestamps(csv_path):
    records = []

    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            records.append({
                "frame_id": int(row["frame_id"]),
                "sequence": int(row["sequence"]),
                "timestamp_s": float(row["timestamp_monotonic_s"]),
            })

    return records


def load_thermal_timestamps(csv_path):
    records = []

    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            records.append({
                "frame_id": int(row["frame_id"]),
                "subpage_id": int(row["subpage_id"]),
                "timestamp_s": float(row["new_data_timestamp_monotonic_s"]),
            })

    return records


# ==============================================================================
# FRAME SELECTION
# ==============================================================================

def choose_stable_pair(rgb_records, thermal_records):
    """
    Choose a synchronized pair near the middle of the temporal overlap.

    The target was held still during each calibration session, so choosing
    near the middle avoids startup/end motion while still using real timestamps.
    """

    rgb_first = rgb_records[0]["timestamp_s"]
    rgb_last = rgb_records[-1]["timestamp_s"]

    thermal_first = thermal_records[0]["timestamp_s"]
    thermal_last = thermal_records[-1]["timestamp_s"]

    overlap_start = max(rgb_first, thermal_first)
    overlap_end = min(rgb_last, thermal_last)

    if overlap_start > overlap_end:
        raise RuntimeError("RGB and thermal timestamps do not overlap.")

    target_time = (overlap_start + overlap_end) / 2.0

    rgb_record = min(rgb_records,key=lambda record: abs(record["timestamp_s"] - target_time))
    thermal_record = min(thermal_records,key=lambda record: abs(record["timestamp_s"] - rgb_record["timestamp_s"]))
    delta_ms = (thermal_record["timestamp_s"] - rgb_record["timestamp_s"]) * 1000.0

    return rgb_record, thermal_record, delta_ms


def read_rgb_frame(video_path, frame_id):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Cannot read RGB frame {frame_id} from {video_path}")

    return frame


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def thermal_to_display(frame):
    frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(frame_norm, cv2.COLORMAP_MAGMA)

    # IMPORTANT: nearest-neighbor keeps each original thermal pixel as a block.
    enlarged = cv2.resize(colored,(THERMAL_DISPLAY_WIDTH, THERMAL_DISPLAY_HEIGHT),interpolation=cv2.INTER_NEAREST,)

    # Draw the 32x24 pixel grid.
    for x in range(0, THERMAL_DISPLAY_WIDTH + 1, THERMAL_SCALE):
        cv2.line(
            enlarged,
            (x, 0),
            (x, THERMAL_DISPLAY_HEIGHT - 1),
            (80, 80, 80),
            1,
        )

    for y in range(0, THERMAL_DISPLAY_HEIGHT + 1, THERMAL_SCALE):
        cv2.line(
            enlarged,
            (0, y),
            (THERMAL_DISPLAY_WIDTH - 1, y),
            (80, 80, 80),
            1,
        )

    return enlarged


def resize_rgb_for_display(frame):
    original_h, original_w = frame.shape[:2]

    display = cv2.resize(
        frame,
        (RGB_DISPLAY_WIDTH, RGB_DISPLAY_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    return display, original_w, original_h


# ==============================================================================
# CLICK UI
# ==============================================================================

def collect_click(window_name, image, prompt):
    clicked = {"point": None}
    display = image.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["point"] = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        canvas = display.copy()

        cv2.rectangle(canvas,(0, 0),(canvas.shape[1], 42),(0, 0, 0),-1,)
        cv2.putText(canvas,prompt,(10, 28),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255, 255, 255),2,)

        if clicked["point"] is not None:
            x, y = clicked["point"]

            cv2.drawMarker(canvas,(x, y),(255, 255, 255),cv2.MARKER_CROSS,22,2,)
            cv2.putText(
                canvas,
                f"({x}, {y})  ENTER=accept  R=retry",
                (10, canvas.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt

        if key in (ord("r"), ord("R")):
            clicked["point"] = None

        if key in (13, 10) and clicked["point"] is not None:
            point = clicked["point"]
            cv2.destroyWindow(window_name)
            return point


# ==============================================================================
# SESSION PROCESSING
# ==============================================================================

def process_session(session_name):
    session_dir = DATA_ROOT / session_name

    rgb_video = session_dir / "rgb" / "rgb.mp4"
    rgb_csv = session_dir / "rgb" / "rgb_frame_timestamps.csv"

    thermal_dir = session_dir / "thermal"
    thermal_csv = thermal_dir / "thermal_frame_timestamps.csv"

    required = [rgb_video, rgb_csv, thermal_csv]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    rgb_records = load_rgb_timestamps(rgb_csv)
    thermal_records = load_thermal_timestamps(thermal_csv)

    rgb_record, thermal_record, delta_ms = choose_stable_pair(rgb_records, thermal_records)
    rgb_frame = read_rgb_frame(rgb_video,rgb_record["frame_id"])
    thermal_file = thermal_dir / (f"frame_{thermal_record['frame_id']:04d}.npy")
    if not thermal_file.exists():
        raise FileNotFoundError(f"Missing thermal frame: {thermal_file}")

    thermal_frame = np.load(thermal_file)

    if thermal_frame.shape != (THERMAL_HEIGHT, THERMAL_WIDTH):
        raise RuntimeError(
            f"Unexpected thermal shape {thermal_frame.shape}; "
            f"expected {(THERMAL_HEIGHT, THERMAL_WIDTH)}"
        )

    rgb_display, rgb_original_w, rgb_original_h = (resize_rgb_for_display(rgb_frame))
    thermal_display = thermal_to_display(thermal_frame)

    print()
    print("--------------------------------------")
    print(f"Session       : {session_name}")
    print(f"RGB frame     : {rgb_record['frame_id']} t={rgb_record['timestamp_s']:.6f}")
    print(f"Thermal frame : {thermal_record['frame_id']} subpage={thermal_record['subpage_id']} t={thermal_record['timestamp_s']:.6f}")
    print(f"Thermal - RGB : {delta_ms:+.2f} ms")
    print("--------------------------------------")

    rgb_click = collect_click(
        f"{session_name} - RGB",
        rgb_display,
        "Click the SAME target point in RGB, then press ENTER",
    )

    thermal_click = collect_click(
        f"{session_name} - THERMAL",
        thermal_display,
        "Click the SAME target point in THERMAL, then press ENTER",
    )

    # Convert RGB display coordinates back to the original RGB coordinates.
    rgb_x = rgb_click[0] * rgb_original_w / RGB_DISPLAY_WIDTH
    rgb_y = rgb_click[1] * rgb_original_h / RGB_DISPLAY_HEIGHT

    # Convert enlarged thermal coordinates back to continuous 32x24 coordinates.
    # Pixel centers are approximately at n + 0.5.
    thermal_x = (thermal_click[0] + 0.5) / THERMAL_SCALE
    thermal_y = (thermal_click[1] + 0.5) / THERMAL_SCALE

    return {
        "session": session_name,
        "rgb_frame_id": rgb_record["frame_id"],
        "rgb_timestamp_s": rgb_record["timestamp_s"],
        "thermal_frame_id": thermal_record["frame_id"],
        "thermal_subpage_id": thermal_record["subpage_id"],
        "thermal_timestamp_s": thermal_record["timestamp_s"],
        "delta_ms": delta_ms,
        "thermal_x": thermal_x,
        "thermal_y": thermal_y,
        "rgb_x": rgb_x,
        "rgb_y": rgb_y,
    }


# ==============================================================================
# SAVE
# ==============================================================================

def save_records(records):
    fields = [
        "session",
        "rgb_frame_id",
        "rgb_timestamp_s",
        "thermal_frame_id",
        "thermal_subpage_id",
        "thermal_timestamp_s",
        "delta_ms",
        "thermal_x",
        "thermal_y",
        "rgb_x",
        "rgb_y",
    ]

    with open(OUTPUT_CSV, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for record in records:
            writer.writerow({
                **record,
                "rgb_timestamp_s": f"{record['rgb_timestamp_s']:.9f}",
                "thermal_timestamp_s": (
                    f"{record['thermal_timestamp_s']:.9f}"
                ),
                "delta_ms": f"{record['delta_ms']:.3f}",
                "thermal_x": f"{record['thermal_x']:.4f}",
                "thermal_y": f"{record['thermal_y']:.4f}",
                "rgb_x": f"{record['rgb_x']:.2f}",
                "rgb_y": f"{record['rgb_y']:.2f}",
            })


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("======================================")
    print(" RGB + Thermal Spatial Calibration")
    print(" Manual correspondence collector")
    print("======================================")
    print()
    print(f"Data root : {DATA_ROOT}")
    print(f"Output    : {OUTPUT_CSV}")
    print()
    print("Controls:")
    print("  Left click : select target point")
    print("  ENTER      : accept point")
    print("  R          : clear/retry")
    print("  ESC        : stop")
    print()
    print(
        "IMPORTANT: click the same physical feature "
        "of the target in both images."
    )

    records = []

    try:
        for index, session_name in enumerate(SESSION_NAMES, start=1):
            print()
            print(
                f"[{index}/{len(SESSION_NAMES)}] "
                f"Processing {session_name}"
            )

            record = process_session(session_name)
            records.append(record)

            # Save after every completed session, so progress is not lost.
            save_records(records)

            print(
                "Saved correspondence: "
                f"Thermal ({record['thermal_x']:.2f}, "
                f"{record['thermal_y']:.2f}) -> "
                f"RGB ({record['rgb_x']:.1f}, "
                f"{record['rgb_y']:.1f})"
            )

    except KeyboardInterrupt:
        print()
        print("[STOPPED] Calibration stopped by user.")

    except Exception as error:
        print()
        print(f"[ERROR] {error}")

    finally:
        cv2.destroyAllWindows()

        if records:
            save_records(records)
            print()
            print(f"Saved {len(records)} correspondence(s):")
            print(OUTPUT_CSV)

    if len(records) == len(SESSION_NAMES):
        print()
        print("======================================")
        print(" All 9 calibration points completed")
        print("======================================")


if __name__ == "__main__":
    main()
