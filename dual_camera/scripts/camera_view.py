import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import time
from pathlib import Path


ADB_EXE = Path(__file__).resolve().parent.parent.parent / "2_tools" / "adb_fastboot" / "adb.exe"

def call_adb(command):
    cmd = [str(ADB_EXE), "shell", command]
    try:
        res = subprocess.check_output(cmd).decode().split()
        if res and "i2ctransfer" in res[0]:
            return None
        return [int(x, 16) for x in res]
    except Exception as e:
        print(f"ADB Error: {e}")
        return None

print("1. Camera initialization and factory fault check...")
test_read = call_adb("i2ctransfer -y 2 w2@0x33 0x04 0x00 r2")
if not test_read:
    print("Sensor is not responding.")
    exit()


plt.ion()
fig, ax = plt.subplots(figsize=(10, 7))
# Initialize with bilinear interpolation
im = ax.imshow(np.zeros((120, 160)), cmap='magma', vmin=0, vmax=255)
plt.colorbar(im).set_label("Relative Temperature")

print("2. Optimized thermal stream enabled. Move your hand!")

while True:
    raw = call_adb("i2ctransfer -y 2 w2@0x33 0x04 0x00 r1536")
    
    if raw and len(raw) == 1536:
    
        u_pixels = np.array(raw[0::2], dtype=np.uint16) + (np.array(raw[1::2], dtype=np.uint16) << 8)
        pixels = u_pixels.astype(np.int16)
        frame = pixels.reshape((24, 32)).astype(float)
        
      
        vmin, vmax = np.percentile(frame, [1, 99])
        if vmax - vmin > 0:
            frame_norm = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        else:
            frame_norm = np.zeros_like(frame, dtype=np.uint8)
            
        # orientation for natural motion reflection
        # frame_norm = np.fliplr(frame_norm)
        frame_norm = np.flipud(frame_norm)
        
        # Initial Bilinear Resize (x5) from (24, 32) to (120, 160)
        frame_resized = cv2.resize(frame_norm, (160, 120), interpolation=cv2.INTER_LINEAR)
        
        # Bilateral Filtering to preserve edges while smoothing surfaces
        frame_clean = cv2.bilateralFilter(frame_resized, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Contrast Enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame_final = clahe.apply(frame_clean)
        
        # Display update
        im.set_array(frame_final)
            
        plt.pause(0.01)
    else:
        print("Transmission error...")
        time.sleep(0.1)