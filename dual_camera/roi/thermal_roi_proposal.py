import csv
from pathlib import Path

import cv2
import numpy as np


# ==============================================================================
# CONFIG
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROI_ROOT = SCRIPT_DIR
CALIBRATION_ROOT = SCRIPT_DIR.parent / "calibration"

# Extract 16.zip and make DATA_ROOT point to the folder containing 1,2,...,17.
DATA_ROOT = ROI_ROOT / "ROI_test"

CALIBRATION_CSV = CALIBRATION_ROOT / "calibration_points.csv"
BACKGROUND_SESSION = "14"

OUTPUT_DIR = ROI_ROOT / "thermal_roi_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_W = 640
RGB_H = 480

# Current .npy files were created from MLX90640 bytes using little-endian int16.
# MLX90640 RAM words arrive MSB-first, so the saved arrays need byte correction.
CORRECT_SAVED_ENDIANNESS = True

# Background-adaptive threshold:
# threshold = high percentile of background temporal noise + margin.
BACKGROUND_NOISE_PERCENTILE = 99.5
THRESHOLD_EXTRA = 3.0

# Ignore tiny isolated blobs.
MIN_COMPONENT_AREA = 4

TEST_SESSIONS = [str(i) for i in range(1, 18) if str(i) != BACKGROUND_SESSION]

# ==============================================================================
# THERMAL
# ==============================================================================

def load_thermal_stack(session_name):
    thermal_dir = DATA_ROOT / session_name / "thermal"
    files = sorted(thermal_dir.glob("frame_*.npy"))

    if not files:
        raise FileNotFoundError(f"No thermal frames: {thermal_dir}")

    stack = np.stack([np.load(path) for path in files])

    if CORRECT_SAVED_ENDIANNESS:
        # Undo the byte-order mistake in the current saved NPY pipeline.
        stack = stack.astype(np.int16).byteswap().astype(np.int16)

    return stack.astype(np.float32)


def build_background():
    stack = load_thermal_stack(BACKGROUND_SESSION)

    background = np.median(stack, axis=0)

    # Measure how much an empty scene fluctuates around its median.
    abs_noise = np.abs(stack - background)
    noise_level = np.percentile(abs_noise,BACKGROUND_NOISE_PERCENTILE)
    threshold = float(noise_level + THRESHOLD_EXTRA)

    return background, threshold


def detect_warm_roi(session_name, background, threshold):
    stack = load_thermal_stack(session_name)

    # Hand is stationary in each short session, so median aggregation suppresses
    # temporal noise and subpage staggering.
    frame = np.median(stack, axis=0)

    difference = frame - background

    mask = (difference > threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,connectivity=8)
    candidates = []

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= MIN_COMPONENT_AREA:
            candidates.append(label)

    if not candidates:
        return frame, difference, mask, None

    largest = max(candidates,key=lambda label: stats[label, cv2.CC_STAT_AREA])

    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    area = int(stats[largest, cv2.CC_STAT_AREA])

    cx, cy = centroids[largest]

    roi = {
        "x1": x,
        "y1": y,
        "x2": x + w,
        "y2": y + h,
        "cx": float(cx),
        "cy": float(cy),
        "area": area,
    }

    return frame, difference, mask, roi


# ==============================================================================
# AFFINE: THERMAL -> RGB
# ==============================================================================
def load_affine():
    thermal_points = []
    rgb_points = []

    with open(CALIBRATION_CSV, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            thermal_points.append([float(row["thermal_x"]),float(row["thermal_y"])])
            rgb_points.append([float(row["rgb_x"]),float(row["rgb_y"])])

    thermal_points = np.asarray(thermal_points, dtype=np.float64)
    rgb_points = np.asarray(rgb_points, dtype=np.float64)

    design = np.column_stack([thermal_points[:, 0],thermal_points[:, 1],np.ones(len(thermal_points))])
    coeff, _, _, _ = np.linalg.lstsq(design,rgb_points,rcond=None)

    return coeff.T


def transform_point(matrix, x, y):
    point = matrix @ np.array([x, y, 1.0])
    return float(point[0]), float(point[1])


def thermal_box_to_rgb(matrix, roi):
    # Map all four corners; do not map only the center.
    corners_t = [
        (roi["x1"], roi["y1"]),
        (roi["x2"], roi["y1"]),
        (roi["x1"], roi["y2"]),
        (roi["x2"], roi["y2"]),
    ]

    corners_r = [transform_point(matrix, x, y) for x, y in corners_t]
    xs = [p[0] for p in corners_r]
    ys = [p[1] for p in corners_r]

    return {
        "x1": max(0.0, min(xs)),
        "y1": max(0.0, min(ys)),
        "x2": min(float(RGB_W - 1), max(xs)),
        "y2": min(float(RGB_H - 1), max(ys)),
        "corners": corners_r,
    }


# ==============================================================================
# RGB FRAME
# ==============================================================================
def read_middle_rgb_frame(session_name):
    video_path = DATA_ROOT / session_name / "rgb" / "rgb.mp4"
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle = max(0, frame_count // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot read frame from {video_path}")

    return frame


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def normalize_heat(image):
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.applyColorMap(normalized,cv2.COLORMAP_MAGMA)


def make_debug_image(session_name,rgb,difference,mask,thermal_roi,rgb_roi,threshold):
    rgb = cv2.resize(rgb, (RGB_W, RGB_H))

    thermal_vis = normalize_heat(difference)
    thermal_vis = cv2.resize(thermal_vis,(RGB_W, RGB_H),interpolation=cv2.INTER_NEAREST)
    mask_vis = cv2.resize(mask * 255,(RGB_W, RGB_H),interpolation=cv2.INTER_NEAREST)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    if thermal_roi is not None:
        scale_x = RGB_W / 32.0
        scale_y = RGB_H / 24.0

        tx1 = int(thermal_roi["x1"] * scale_x)
        ty1 = int(thermal_roi["y1"] * scale_y)
        tx2 = int(thermal_roi["x2"] * scale_x)
        ty2 = int(thermal_roi["y2"] * scale_y)

        cv2.rectangle(thermal_vis,(tx1, ty1),(tx2, ty2),(255, 255, 255),2)

        if rgb_roi is not None:
            rx1 = int(round(rgb_roi["x1"]))
            ry1 = int(round(rgb_roi["y1"]))
            rx2 = int(round(rgb_roi["x2"]))
            ry2 = int(round(rgb_roi["y2"]))

            cv2.rectangle(rgb,(rx1, ry1),(rx2, ry2),(255, 255, 255),2)

    for panel, label in [
        (rgb, "RGB: predicted ROI"),
        (thermal_vis, "Thermal difference + blob"),
        (mask_vis, f"Binary mask (threshold={threshold:.1f})"),
    ]:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel,label,(8, 24),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255, 255, 255),2)

    combined = np.hstack([rgb, thermal_vis, mask_vis])
    cv2.putText(combined,f"session {session_name}",(10, combined.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255, 255, 255),2)

    return combined


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"DATA_ROOT not found: {DATA_ROOT}\n"
            "Edit DATA_ROOT near the top of this script."
        )

    if not CALIBRATION_CSV.exists():
        raise FileNotFoundError(
            f"Calibration CSV not found: {CALIBRATION_CSV}"
        )

    background, threshold = build_background()
    affine = load_affine()

    print("==========================================")
    print(" Thermal-guided RGB ROI proposal")
    print("==========================================")
    print(f"Background session : {BACKGROUND_SESSION}")
    print(f"Adaptive threshold : {threshold:.2f}")
    print()
    print("Affine Thermal -> RGB:")
    print(affine)
    print()

    result_rows = []

    for session_name in TEST_SESSIONS:
        frame, difference, mask, thermal_roi = detect_warm_roi(
            session_name,
            background,
            threshold
        )

        rgb = read_middle_rgb_frame(session_name)

        if thermal_roi is None:
            print(f"session {session_name:>2}: NO WARM REGION")

            rgb_roi = None

            result_rows.append({
                "session": session_name,
                "thermal_x1": "",
                "thermal_y1": "",
                "thermal_x2": "",
                "thermal_y2": "",
                "thermal_area_px": 0,
                "rgb_x1": "",
                "rgb_y1": "",
                "rgb_x2": "",
                "rgb_y2": "",
            })

        else:
            rgb_roi = thermal_box_to_rgb(
                affine,
                thermal_roi
            )

            print(
                f"session {session_name:>2}: "
                f"T=({thermal_roi['x1']},{thermal_roi['y1']})-"
                f"({thermal_roi['x2']},{thermal_roi['y2']}) "
                f"area={thermal_roi['area']:3d}  ->  "
                f"RGB=({rgb_roi['x1']:.1f},{rgb_roi['y1']:.1f})-"
                f"({rgb_roi['x2']:.1f},{rgb_roi['y2']:.1f})"
            )

            result_rows.append({
                "session": session_name,
                "thermal_x1": thermal_roi["x1"],
                "thermal_y1": thermal_roi["y1"],
                "thermal_x2": thermal_roi["x2"],
                "thermal_y2": thermal_roi["y2"],
                "thermal_area_px": thermal_roi["area"],
                "rgb_x1": f"{rgb_roi['x1']:.2f}",
                "rgb_y1": f"{rgb_roi['y1']:.2f}",
                "rgb_x2": f"{rgb_roi['x2']:.2f}",
                "rgb_y2": f"{rgb_roi['y2']:.2f}",
            })

        debug = make_debug_image(session_name,rgb,difference,mask,thermal_roi,rgb_roi,threshold)

        cv2.imwrite(str(OUTPUT_DIR / f"session_{int(session_name):02d}.png"),debug)

    csv_path = OUTPUT_DIR / "roi_proposals.csv"

    with open(csv_path, "w", newline="") as file:
        fields = [
            "session",
            "thermal_x1",
            "thermal_y1",
            "thermal_x2",
            "thermal_y2",
            "thermal_area_px",
            "rgb_x1",
            "rgb_y1",
            "rgb_x2",
            "rgb_y2",
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)

    print()
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Open session_XX.png files first.")
    print(
        "At this stage the RGB rectangle is the raw mapped ROI proposal "
        "without an added safety margin."
    )


if __name__ == "__main__":
    main()
