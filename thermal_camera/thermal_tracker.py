import subprocess
import os
import numpy as np
from scipy import ndimage
import cv2
import time
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


ADB_EXE = PROJECT_ROOT / "2_tools" / "adb_fastboot" / "adb.exe"


FICHIER_CSV = SCRIPT_DIR / "donnees_thermiques.csv"

# Reset CSV file on launch
os.makedirs(os.path.dirname(FICHIER_CSV), exist_ok=True)
with open(FICHIER_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp_s", "Temp_Relative_Max", "Pos_Hotspot_X", "Pos_Hotspot_Y"])

def call_adb(command):
    cmd = [str(ADB_EXE), "shell", command]
    try:
        res = subprocess.check_output(cmd, creationflags=0x08000000).decode().split()
        if res and "i2ctransfer" in res[0]:
            return None
        return [int(x, 16) for x in res]
    except Exception as e:
        print(f"ADB Error: {e}")
        return None

print("1. Sensor initialization...")
test_read = call_adb("i2ctransfer -y 2 w2@0x33 0x04 0x00 r2")
if not test_read:
    print("Sensor is not responding.")
    exit()


cv2.namedWindow("Live Thermal Stream (Zero Latency)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Thermal Stream (Zero Latency)", 640, 480)

print("2. Real-time capture enabled! Press 'q' to quit.")
temps_debut = time.time()


SEUIL_FOND_NOIR = 25000   # Values below this threshold are absolute black
GAIN_SENSIBILITE = 2.5    # Multiplies hot spot contrast

while True:
    raw = call_adb("i2ctransfer -y 2 w2@0x33 0x04 0x00 r1536")
    
    if raw and len(raw) == 1536:
        # 16-bit decoding
        bytes_data = np.array(raw, dtype=np.uint8).tobytes()
        pixels = np.frombuffer(bytes_data, dtype='<i2').astype(float)
        frame = pixels.reshape((24, 32))
        
        # Cleaning and cropping (H=18, W=26)
        frame = ndimage.median_filter(frame, size=3)
        frame = frame[3:21, 3:29]

        # Correct thermal camera orientation
        frame = np.flipud(frame)
        
        # Hotspot detection
        frame_smoothed = cv2.blur(frame, (3, 3))
        mask_centre = np.zeros_like(frame_smoothed, dtype=bool)
        mask_centre[2:-2, 2:-2] = True
        
        frame_pour_hotspot = np.where(mask_centre, frame_smoothed, -99999)
        temp_max_brute = float(np.max(frame_pour_hotspot))
        y_hot, x_hot = np.unravel_index(np.argmax(frame_pour_hotspot, axis=None), frame_pour_hotspot.shape)
        
        # Black background normalization with high sensitivity
        frame_sub = np.maximum(0, frame - SEUIL_FOND_NOIR)
        frame_norm = np.clip(frame_sub * (GAIN_SENSIBILITE / 10.0), 0, 255).astype(np.uint8)
        
        # Mirror effect
        # frame_norm = np.fliplr(frame_norm)
        # x_hot_mirror = frame.shape[1] - 1 - x_hot
        
        # Bicubic Resize + Bilateral Filter
        frame_resized = cv2.resize(frame_norm, (320, 240), interpolation=cv2.INTER_CUBIC)
        frame_clean = cv2.bilateralFilter(frame_resized, d=7, sigmaColor=30, sigmaSpace=30)
        
        # Thermal Colorization (Magma Palette)
        frame_colored = cv2.applyColorMap(frame_clean, cv2.COLORMAP_MAGMA)
        
        # Draw target crosshair
        cx = int((x_hot / frame.shape[1]) * 320)
        cy = int((y_hot / frame.shape[0]) * 240)
        
        if temp_max_brute > SEUIL_FOND_NOIR + 1000:
            cv2.circle(frame_colored, (cx, cy), 6, (0, 255, 0), -1)  # Green dot
        
        cv2.putText(frame_colored, f"Max: {int(temp_max_brute)}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save to CSV if hotspot detected
        if temp_max_brute > SEUIL_FOND_NOIR + 1000:
            timestamp = round(time.time() - temps_debut, 2)
            with open(FICHIER_CSV, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, round(temp_max_brute, 2), int(x_hot), int(y_hot)])
        
     
        cv2.imshow("Live Thermal Stream (Zero Latency)", frame_colored)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else:
        print("Transmission error...")
        time.sleep(0.02)

cv2.destroyAllWindows()