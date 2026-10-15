from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import subprocess
import time
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent / 'ROI_test_rtsp'
RTSP_URL = 'rtsp://127.0.0.1:8554/live/0'

def run(args, timeout=30):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)

def read_thermal():
    t0 = time.monotonic()
    result = run(['adb', 'shell', 'i2ctransfer -y 2 w2@0x33 0x04 0x00 r1536'], timeout=30)
    t1 = time.monotonic()
    tokens = result.stdout.split()
    if len(tokens) != 1536:
        raise RuntimeError(f'Expected 1536 I2C bytes, got {len(tokens)}: {result.stdout[:160]!r}')
    values = np.array([int(token,16) for token in tokens], dtype=np.uint8)
    # Preserve the same decoding convention as the supplied calibration capture.
    raw_u16 = values[0::2].astype(np.uint16) | (values[1::2].astype(np.uint16) << 8)
    raw = raw_u16.view(np.int16).reshape(24,32).copy()
    return raw, t0, t1

def capture_rgb(path):
    t0 = time.monotonic()
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-rtsp_transport','tcp',
         '-i',RTSP_URL,'-map','0:v:0','-frames:v','1',str(path)],timeout=35)
    t1 = time.monotonic()
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f'RGB capture failed: {path}')
    if img.shape[:2] != (720,1280):
        raise RuntimeError(f'RGB shape {img.shape[:2]} is not (720,1280); check RTSP stream and calibration')
    return t0,t1

def save_preview(raw, path):
    from matplotlib import colormaps
    f = raw.astype(np.float32)
    lo,hi = np.percentile(f,[1,99])
    norm = np.zeros((24,32),dtype=np.uint8) if hi<=lo else (np.clip((f-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    oriented = np.rot90(np.flipud(norm),k=1)
    # Display only: match the landscape 160x120 convention from calibration capture.
    display = cv2.resize(oriented,(160,120),interpolation=cv2.INTER_NEAREST)
    colored = (colormaps['magma'](display/255.0)[:,:,:3]*255).astype(np.uint8)
    cv2.imwrite(str(path),cv2.cvtColor(colored,cv2.COLOR_RGB2BGR))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--session',required=True,help='background or 01..12')
    p.add_argument('--frames',type=int,default=40)
    args=p.parse_args()
    valid={'background'}|{f'{i:02d}' for i in range(1,13)}
    if args.session not in valid:
        p.error('session must be background or 01..12')
    if args.frames<1:
        p.error('--frames must be positive')
    folder=ROOT/args.session
    if folder.exists() and any(folder.rglob('*')):
        raise FileExistsError(f'{folder} already has data. Move/rename it before recapturing; existing data was not overwritten.')
    rgb_dir=folder/'rgb'; thermal_dir=folder/'thermal'
    rgb_dir.mkdir(parents=True,exist_ok=True); thermal_dir.mkdir(parents=True,exist_ok=True)
    run(['adb','get-state'],timeout=10)
    run(['adb','forward','tcp:8554','tcp:554'],timeout=10)
    print(f'Session {args.session}: capture 1 RGB + {args.frames} thermal register snapshots.')
    print('Hold scene/target STILL until capture completes; capture duration depends on adb/I2C speed.')
    input('Press Enter when ready...')
    rgb_start,rgb_end=capture_rgb(rgb_dir/'rgb.png')
    rows=[]
    for i in range(args.frames):
        raw,t0,t1=read_thermal()
        np.save(thermal_dir/f'frame_{i:03d}.npy',raw)
        rows.append({'frame':i,'file':f'frame_{i:03d}.npy','read_start_monotonic_s':f'{t0:.6f}',
                     'read_end_monotonic_s':f'{t1:.6f}','min_raw':int(raw.min()),'max_raw':int(raw.max())})
        print(f'  thermal {i+1}/{args.frames} ({t1-t0:.3f}s)',flush=True)
    with (folder/'thermal_timestamps.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_preview(raw,folder/'thermal_preview_last.png')
    metadata={'session':args.session,'created_local':datetime.now().isoformat(),
              'rgb_path':'rgb/rgb.png','rgb_size':[1280,720],
              'rgb_command_interval_monotonic_s':[rgb_start,rgb_end],
              'thermal_count':len(rows),'thermal_raw_shape':[24,32],
              'thermal_decoding':'little-endian int16 as in provided calibration capture; not verified temperature',
              'note':'Sequential captures, not exposure-synchronized; MLX90640 RAM read is not verified as a coherent full frame.'}
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Done: {folder}')
    print('Inspect rgb/rgb.png and thermal_preview_last.png before capturing more sessions.')

if __name__=='__main__':
    main()
