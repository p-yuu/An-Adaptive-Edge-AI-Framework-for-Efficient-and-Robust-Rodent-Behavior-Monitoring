from pathlib import Path
import csv
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "new_rtsp_9points"
OUTPUT_CSV = SCRIPT_DIR / "calibration_points_rtsp_fixed.csv"
SESSION_NAMES = (
    "top_left", "top_mid", "top_right",
    "mid_left", "mid", "mid_right",
    "bottom_left", "bottom_mid", "bottom_right"
)
THERMAL_DISPLAY_WIDTH = 800
THERMAL_DISPLAY_HEIGHT = 600
THERMAL_COORD_WIDTH = 24
THERMAL_COORD_HEIGHT = 32
RGB_DISPLAY_MAX_WIDTH = 900
RGB_DISPLAY_MAX_HEIGHT = 650

def latest_pair(folder):
    metadata_files = sorted(folder.glob("*_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No *_metadata.json found in {folder}")
    meta_file = metadata_files[-1]
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    rgb = folder / meta["rgb_path"]
    thermal = folder / meta["thermal_raw_path"]
    if not rgb.is_file() or not thermal.is_file():
        raise FileNotFoundError(f"Missing RGB or thermal file in {folder}")
    return rgb, thermal, meta_file

def orient_thermal(raw):
    if raw.shape != (24, 32):
        raise ValueError(f"Expected raw thermal 24x32, got {raw.shape}")
    return np.rot90(np.flipud(raw), k=1)  # output height 32, width 24

def make_thermal_display(raw):
    oriented = orient_thermal(raw)
    f = oriented.astype(np.float32)
    lo, hi = np.percentile(f, [1, 99])
    if hi <= lo:
        norm = np.zeros_like(oriented, dtype=np.uint8)
    else:
        norm = (np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    # Match the camera_view.py landscape appearance, enlarged for accurate clicking.
    enlarged = cv2.resize(norm, (THERMAL_DISPLAY_WIDTH, THERMAL_DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
    enlarged = cv2.bilateralFilter(enlarged, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enlarged = clahe.apply(enlarged)
    preview = enlarged
    try:
        rgb = (plt.get_cmap("magma")(preview / 255.0)[:, :, :3] * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

def make_rgb_display(rgb):
    h, w = rgb.shape[:2]
    factor = min(RGB_DISPLAY_MAX_WIDTH / w, RGB_DISPLAY_MAX_HEIGHT / h, 1.0)
    new_w, new_h = max(1, round(w*factor)), max(1, round(h*factor))
    display = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return display, (w/new_w, h/new_h)

def click_point(title, image):
    state = {"xy": None}
    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["xy"] = (x, y)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            canvas = image.copy()
            if state["xy"] is not None:
                x, y = state["xy"]
                cv2.drawMarker(canvas, (x,y), (255,255,255),
                               cv2.MARKER_CROSS, 16, 2)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(30) & 0xff
            if key in (13, 10) and state["xy"] is not None:
                return state["xy"]
            if key in (ord('r'), ord('R')):
                state["xy"] = None
            if key == 27:
                raise KeyboardInterrupt
    finally:
        cv2.destroyWindow(title)

FIELDS = ["session", "rgb_image", "thermal_file", "thermal_x", "thermal_y",
          "rgb_x", "rgb_y"]

def save(records):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

def main():
    print("Calibration data:", DATA_ROOT)
    print("Output:", OUTPUT_CSV)
    print("Click the SAME physical target point in RGB and thermal.")
    print("Left-click to choose, Enter to confirm, R to retry, Esc to stop.")
    records = []
    if OUTPUT_CSV.exists():
        answer = input("Existing output CSV found. Overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Stopped without changing existing CSV.")
            return
    try:
        for index, name in enumerate(SESSION_NAMES, 1):
            folder = DATA_ROOT / name
            rgb_path, thermal_path, meta_path = latest_pair(folder)
            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                raise RuntimeError(f"Cannot read {rgb_path}")
            thermal = np.load(thermal_path, allow_pickle=False)
            rgb_display, (sx, sy) = make_rgb_display(rgb)
            thermal_display = make_thermal_display(thermal)
            print(f"[{index}/9] {name}: {rgb_path.name}, {thermal_path.name}")
            rgb_px, rgb_py = click_point(f"{name} - RGB", rgb_display)
            th_px, th_py = click_point(f"{name} - Thermal", thermal_display)
            record = {
                "session": name,
                "rgb_image": str(rgb_path.relative_to(DATA_ROOT)),
                "thermal_file": str(thermal_path.relative_to(DATA_ROOT)),
                "thermal_x": round((th_px + 0.5) * THERMAL_COORD_WIDTH / THERMAL_DISPLAY_WIDTH - 0.5, 4),
                "thermal_y": round((th_py + 0.5) * THERMAL_COORD_HEIGHT / THERMAL_DISPLAY_HEIGHT - 0.5, 4),
                "rgb_x": round((rgb_px + 0.5) * sx - 0.5, 2),
                "rgb_y": round((rgb_py + 0.5) * sy - 0.5, 2),
            }
            records.append(record)
            save(records)
            print(f"  Saved: thermal ({record['thermal_x']}, {record['thermal_y']}) "
                  f"-> RGB ({record['rgb_x']}, {record['rgb_y']})")
    except (KeyboardInterrupt, Exception) as error:
        print(f"Stopped: {error}")
        if records:
            save(records)
    finally:
        cv2.destroyAllWindows()
    print(f"Saved {len(records)}/9 points to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
