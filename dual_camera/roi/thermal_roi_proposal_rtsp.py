"""Thermal-guided RGB ROI proposal for RTSP RGB (1280x720) + MLX90640.

Input layout (relative to this script):
  ROI_test_rtsp/background/thermal/frame_000.npy ...
  ROI_test_rtsp/background/rgb/rgb.png (optional)
  ROI_test_rtsp/01/thermal/frame_000.npy ...
  ROI_test_rtsp/01/rgb/rgb.png (or rgb.mp4)

Each NPY must contain unrotated MLX90640 sensor grid (24,32),
not a colorized/resized preview. Thermal values must be comparable across sessions.
"""
from pathlib import Path
import csv
import re
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'ROI_test_rtsp'
MATRIX_FILE = ROOT.parent / 'calibration' / 'spatial_alignment_evaluation_rtsp' / 'affine_matrix.txt'
# Adjust MATRIX_FILE to your actual file location, e.g. ROOT / 'affine_matrix.txt'.
OUTPUT_DIR = ROOT / 'thermal_roi_results_rtsp'
BACKGROUND_SESSION = 'background'
RGB_W, RGB_H = 1280, 720
THERMAL_BYTE_SWAP = False  # Only True if independently verified for this capture pipeline.
BACKGROUND_NOISE_PERCENTILE = 99.5
THRESHOLD_EXTRA = 3.0  # RAW sensor units, NOT degrees Celsius.
MIN_COMPONENT_AREA = 4
DEBUG_PANEL_W, DEBUG_PANEL_H = 640, 360

FIELDS = ['session','thermal_x1','thermal_y1','thermal_x2','thermal_y2',
          'thermal_area_px','rgb_x1','rgb_y1','rgb_x2','rgb_y2']


def load_affine(path):
    if not path.exists():
        raise FileNotFoundError(f'Affine matrix file not found: {path}')
    text = path.read_text(encoding='utf-8')
    match = re.search(r'\[\[\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*\]\s*\[\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*\]\]', text)
    if match is None:
        raise ValueError('Cannot parse 2x3 matrix from affine_matrix.txt')
    matrix = np.array([float(v) for v in match.groups()], dtype=np.float64).reshape(2,3)
    if not np.all(np.isfinite(matrix)):
        raise ValueError('Affine matrix contains non-finite values')
    return matrix


def orient_thermal(raw):
    # EXACT calibration convention: raw (24,32) -> oriented (32,24).
    return np.rot90(np.flipud(raw), k=1)


def load_thermal_stack(session):
    folder = DATA_ROOT / session / 'thermal'
    files = sorted(folder.glob('frame_*.npy'))
    if not files:
        raise FileNotFoundError(f'No frame_*.npy found in {folder}')
    frames = []
    for path in files:
        arr = np.load(path, allow_pickle=False)
        if arr.shape != (24,32):
            raise ValueError(f'{path}: expected raw shape (24,32), got {arr.shape}')
        if THERMAL_BYTE_SWAP:
            if arr.dtype.kind not in 'iu' or arr.dtype.itemsize != 2:
                raise ValueError(f'{path}: byteswap requires 16-bit integer input')
            arr = arr.byteswap()
        frames.append(orient_thermal(arr.astype(np.float32)))
    stack = np.stack(frames)
    if not np.isfinite(stack).all():
        raise ValueError(f'Non-finite thermal data in {session}')
    return stack


def read_rgb(session):
    folder = DATA_ROOT / session / 'rgb'
    for name in ('rgb.png','rgb.jpg','rgb.jpeg'):
        path = folder / name
        if path.exists():
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f'Cannot read {path}')
            break
    else:
        path = folder / 'rgb.mp4'
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f'Expected rgb.png or rgb.mp4 in {folder}')
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,n//2))
        ok,image = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Cannot decode {path}')
    if image.shape[:2] != (RGB_H, RGB_W):
        raise ValueError(f'{session}: RGB expected 1280x720, got {image.shape[1]}x{image.shape[0]}. Do not resize without updating calibration.')
    return image


def detect(frame, background, threshold):
    difference = frame - background
    mask = (difference > threshold).astype(np.uint8)
    count,labels,stats,_ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA]
    if not candidates:
        return difference,mask,None
    i = max(candidates,key=lambda j: stats[j,cv2.CC_STAT_AREA])
    x,y,w,h,area = [int(v) for v in stats[i]]
    return difference,mask,(x,y,x+w,y+h,area)


def project_bbox(matrix, box):
    x1,y1,x2,y2,_ = box
    corners = np.array([[x1,y1,1],[x2,y1,1],[x1,y2,1],[x2,y2,1]], dtype=np.float64)
    points = corners @ matrix.T
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    x1r,y1r = np.maximum(lo,[0,0])
    x2r,y2r = np.minimum(hi,[RGB_W,RGB_H])
    if x2r <= x1r or y2r <= y1r:
        return None
    return (float(x1r),float(y1r),float(x2r),float(y2r))


def make_debug(rgb,diff,mask,tbox,rbox,session,threshold):
    rgb = rgb.copy()
    if rbox is not None:
        a,b,c,d = map(lambda v:int(round(v)),rbox)
        cv2.rectangle(rgb,(a,b),(c,d),(0,255,0),3)
    rgb_small = cv2.resize(rgb,(DEBUG_PANEL_W,DEBUG_PANEL_H),interpolation=cv2.INTER_AREA)
    norm = cv2.normalize(diff,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm,cv2.COLORMAP_MAGMA)
    heat = cv2.resize(heat,(DEBUG_PANEL_W,DEBUG_PANEL_H))
    binary = cv2.cvtColor(cv2.resize(mask*255,(DEBUG_PANEL_W,DEBUG_PANEL_H),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
    if tbox is not None:
        x1,y1,x2,y2,_=tbox
        cv2.rectangle(heat,(round(x1*DEBUG_PANEL_W/24),round(y1*DEBUG_PANEL_H/32)),
                      (round(x2*DEBUG_PANEL_W/24),round(y2*DEBUG_PANEL_H/32)),(255,255,255),2)
    for panel,label in ((rgb_small,'RGB proposed ROI'),(heat,'Thermal difference (oriented)'),(binary,'Thermal mask')):
        cv2.rectangle(panel,(0,0),(DEBUG_PANEL_W,32),(0,0,0),-1)
        cv2.putText(panel,label,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    result = np.hstack((rgb_small,heat,binary))
    cv2.putText(result,f'Session {session} | threshold={threshold:.2f}',(10,DEBUG_PANEL_H-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
    return result


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f'Data folder not found: {DATA_ROOT}')
    matrix = load_affine(MATRIX_FILE)
    background_stack = load_thermal_stack(BACKGROUND_SESSION)
    if len(background_stack) < 5:
        raise ValueError('Collect at least 5 background thermal frames (20+ recommended).')
    background = np.median(background_stack,axis=0)
    noise = np.abs(background_stack-background)
    threshold = float(np.percentile(noise,BACKGROUND_NOISE_PERCENTILE)+THRESHOLD_EXTRA)
    sessions = sorted((p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name != BACKGROUND_SESSION),key=lambda s:(not s.isdigit(),int(s) if s.isdigit() else s))
    if not sessions:
        raise ValueError('No target session folders found')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]
    print('Affine (oriented thermal -> 1280x720 RGB):\n',matrix)
    print(f'Background frames={len(background_stack)}, noise threshold={threshold:.3f}')
    for session in sessions:
        frame = np.median(load_thermal_stack(session),axis=0)
        rgb = read_rgb(session)
        difference,mask,tbox = detect(frame,background,threshold)
        rbox = project_bbox(matrix,tbox) if tbox is not None else None
        row=dict.fromkeys(FIELDS,'')
        row['session']=session
        row['thermal_area_px']=tbox[4] if tbox else 0
        if tbox is not None:
            for key,value in zip(('thermal_x1','thermal_y1','thermal_x2','thermal_y2'),tbox[:4]):row[key]=value
        if rbox is not None:
            for key,value in zip(('rgb_x1','rgb_y1','rgb_x2','rgb_y2'),rbox):row[key]=f'{value:.2f}'
        rows.append(row)
        cv2.imwrite(str(OUTPUT_DIR/f'session_{session}.png'),make_debug(rgb,difference,mask,tbox,rbox,session,threshold))
        print(f'{session}: thermal={tbox}, rgb={rbox}')
    with (OUTPUT_DIR/'roi_proposals.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    print('Saved:',OUTPUT_DIR)
    print('WARNING: raw MLX90640 register values are not calibrated temperatures; verify sensor decoding and background stability before interpreting warm regions.')

if __name__ == '__main__':
    main()
