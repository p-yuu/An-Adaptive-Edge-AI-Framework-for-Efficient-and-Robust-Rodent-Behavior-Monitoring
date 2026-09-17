import csv
from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "ROI_test"
ROI_CSV = SCRIPT_DIR / "thermal_roi_results" / "roi_proposals.csv"

BACKGROUND_SESSION = "14"
RGB_W, RGB_H = 640, 480
THERMAL_W, THERMAL_H = 32, 24
MARGIN = 50     # 修改 margin 大小
CORRECT_SAVED_ENDIANNESS = True
WINDOW_NAME = "Thermal-guided RGB ROI Demo"
DEMO_SESSIONS = [str(i) for i in range(1, 18) if str(i) != BACKGROUND_SESSION]

def load_roi_proposals():
    out = {}
    with open(ROI_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = str(int(r["session"]))
            if not r["rgb_x1"]:
                continue
            out[s] = {
                "thermal": tuple(float(r[k]) for k in ("thermal_x1","thermal_y1","thermal_x2","thermal_y2")),
                "rgb": tuple(float(r[k]) for k in ("rgb_x1","rgb_y1","rgb_x2","rgb_y2"))
            }
    return out

def thermal_stack(session):
    files = sorted((DATA_ROOT / session / "thermal").glob("frame_*.npy"))
    if not files:
        raise FileNotFoundError(f"No thermal frames for session {session}")
    a = np.stack([np.load(p) for p in files])
    if CORRECT_SAVED_ENDIANNESS:
        a = a.astype(np.int16).byteswap().astype(np.int16)
    return a.astype(np.float32)

def rgb_middle(session):
    path = DATA_ROOT / session / "rgb" / "rgb.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fid = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path}")
    return fid, cv2.resize(frame, (RGB_W, RGB_H))

def thermal_vis(session, background):
    target = np.median(thermal_stack(session), axis=0)
    diff = target - background
    v = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    v = cv2.applyColorMap(v, cv2.COLORMAP_MAGMA)
    return cv2.resize(v, (RGB_W, RGB_H), interpolation=cv2.INTER_NEAREST)

def margin_box(b):
    x1,y1,x2,y2 = b
    return (int(max(0,x1-MARGIN)), int(max(0,y1-MARGIN)),
            int(min(RGB_W-1,x2+MARGIN)), int(min(RGB_H-1,y2+MARGIN)))

def thermal_display_box(b):
    sx, sy = RGB_W/THERMAL_W, RGB_H/THERMAL_H
    x1,y1,x2,y2 = b
    return tuple(map(int, (x1*sx,y1*sy,x2*sx,y2*sy)))

def title(img, text):
    cv2.rectangle(img,(0,0),(img.shape[1],46),(0,0,0),-1)
    cv2.putText(img,text,(14,31),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA)

def arrow_panel():
    p = np.zeros((RGB_H,190,3),np.uint8)
    cv2.putText(p,"Affine",(54,210),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
    cv2.arrowedLine(p,(20,245),(170,245),(255,255,255),3,cv2.LINE_AA,tipLength=.12)
    cv2.putText(p,"Thermal to RGB",(31,290),cv2.FONT_HERSHEY_SIMPLEX,.48,(180,180,180),1,cv2.LINE_AA)
    return p

def make_frame(session, proposals, background):
    t = thermal_vis(session, background)
    tx1,ty1,tx2,ty2 = thermal_display_box(proposals[session]["thermal"])
    cv2.rectangle(t,(tx1,ty1),(tx2,ty2),(255,255,255),3)
    title(t,"Detected thermal region")

    fid, rgb = rgb_middle(session)
    x1,y1,x2,y2 = margin_box(proposals[session]["rgb"])
    cv2.rectangle(rgb,(x1,y1),(x2,y2),(255,255,255),3)
    title(rgb,f"Mapped ROI + {MARGIN}px margin")

    demo = np.hstack((t,arrow_panel(),rgb))
    bar = np.zeros((50,demo.shape[1],3),np.uint8)
    cv2.putText(bar,f"Session {session} | RGB frame {fid} | A/D: previous/next | ESC: exit",
                (15,32),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2,cv2.LINE_AA)
    return np.vstack((demo,bar))

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing {DATA_ROOT}")
    if not ROI_CSV.exists():
        raise FileNotFoundError(f"Missing {ROI_CSV}")

    proposals = load_roi_proposals()
    sessions = [s for s in DEMO_SESSIONS if s in proposals]
    background = np.median(thermal_stack(BACKGROUND_SESSION), axis=0)

    print(f"Demo ready: {len(sessions)} sessions, margin={MARGIN}px")
    print("D/right = next, A/left = previous, ESC = exit")

    i = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(WINDOW_NAME, make_frame(sessions[i], proposals, background))
        k = cv2.waitKeyEx(0)
        if k == 27:
            break
        if k in (ord("d"),ord("D"),2555904):
            i = (i+1) % len(sessions)
        elif k in (ord("a"),ord("A"),2424832):
            i = (i-1) % len(sessions)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
