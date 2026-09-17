import csv
from pathlib import Path

import cv2


# ==============================================================================
# CONFIG
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROI_ROOT = SCRIPT_DIR

# Folder containing session folders 1, 2, ..., 17.
DATA_ROOT = ROI_ROOT / "ROI_test"
OUTPUT_CSV = ROI_ROOT / "rgb_ground_truth.csv"
BACKGROUND_SESSION = "14"
SESSIONS = [str(i) for i in range(1, 18) if str(i) != BACKGROUND_SESSION]


# ==============================================================================
# HELPERS
# ==============================================================================
def read_middle_frame(session_name):
    video_path = DATA_ROOT / session_name / "rgb" / "rgb.mp4"
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = max(0, frame_count // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot read frame: {video_path}")

    return frame_id, frame


def load_existing():
    records = {}
    if not OUTPUT_CSV.exists():
        return records

    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records[row["session"]] = row

    return records


def save_records(records):
    fields = [
        "session",
        "frame_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "touches_image_border",
    ]

    ordered = sorted(records.values(),key=lambda r: int(r["session"]))
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"DATA_ROOT not found: {DATA_ROOT}\n"
            "Edit DATA_ROOT near the top of this script if necessary."
        )

    records = load_existing()

    print("====================================================")
    print(" RGB Ground-Truth Bounding Box Annotation")
    print("====================================================")
    print("Draw ONE box around the visible hand.")
    print()
    print("Mouse drag : draw bounding box")
    print("ENTER/SPACE: accept")
    print("C          : cancel selection / redraw")
    print("ESC        : stop; completed labels remain saved")
    print()
    print("Important:")
    print("- Box the visible hand as tightly as reasonably possible.")
    print("- Include fingers and palm that are visible.")
    print("- Do NOT enlarge the box to compensate for thermal/affine error.")
    print("- If the hand is cut by the RGB image edge, box only the visible part.")
    print("====================================================")
    print()

    for session in SESSIONS:
        if session in records:
            print(f"session {session}: already labeled, skipping")
            continue

        frame_id, frame = read_middle_frame(session)
        display = frame.copy()

        cv2.putText(
            display,
            f"Session {session} | Frame {frame_id} | Draw visible hand GT",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4
        )
        cv2.putText(
            display,
            f"Session {session} | Frame {frame_id} | Draw visible hand GT",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        print(f"\nSession {session}: draw bounding box...")

        x, y, w, h = cv2.selectROI(
            f"RGB Ground Truth - Session {session}",
            display,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyAllWindows()

        # selectROI returns zero-sized ROI if ESC/cancel is used.
        if w == 0 or h == 0:
            print("Annotation stopped. Existing labels were saved.")
            break

        height, width = frame.shape[:2]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        # A GT box touching the image border may indicate that the true hand
        # extends outside the observable RGB frame. Keep this information for
        # later evaluation instead of silently treating it as a full-object GT.
        touches_border = int(
            x1 <= 0 or
            y1 <= 0 or
            x2 >= width or
            y2 >= height
        )

        records[session] = {
            "session": session,
            "frame_id": frame_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": int(w),
            "height": int(h),
            "touches_image_border": touches_border,
        }

        save_records(records)

        print(
            f"saved: session={session}, "
            f"box=({x1},{y1})-({x2},{y2}), "
            f"touches_border={touches_border}"
        )

    save_records(records)

    print()
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Completed: {len(records)} / {len(SESSIONS)} sessions")


if __name__ == "__main__":
    main()
