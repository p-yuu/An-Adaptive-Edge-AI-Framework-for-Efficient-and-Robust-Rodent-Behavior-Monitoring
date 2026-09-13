import os
import subprocess
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

# ==============================================================================
# DYNAMIC PATHS CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Path to ADB executable in 2_tools/adb_fastboot/
ADB_EXE = PROJECT_ROOT / "2_tools" / "adb_fastboot" / "adb.exe"

# Path to YOLO model inside 5_code/ (5_code/yolov8n.pt)
MODEL_PATH = SCRIPT_DIR.parent / "yolov8n.pt"

# Remote path on Luckfox board
SCRIPT_CAPTURE_LUCKFOX = "/root/capture.py"
FICHIER_LUCKFOX = "/tmp/video20.yuv"        

# Local output paths inside 5_code/imx_camera_yolo/
FICHIER_YUV_LOCAL = SCRIPT_DIR / "video20.yuv"
VIDEO_SORTIE = SCRIPT_DIR / "video_finale_yolo.mp4" 

WIDTH = 640
HEIGHT = 480
NOMBRE_IMAGES = 60  
FPS = 15            
# ==============================================================================

def executer_capture_distance():
    print(" Triggering remote capture on Luckfox...")
    commande_remote = [str(ADB_EXE), "shell", f"python3 {SCRIPT_CAPTURE_LUCKFOX}"]
    try:
        process = subprocess.Popen(
            commande_remote, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
        )
        while True:
            ligne = process.stdout.readline()
            if not ligne and process.poll() is not None:
                break
            if ligne:
                print(f"   ➔ {ligne.strip()}")
        return process.returncode == 0
    except Exception as e:
        print(f" ❌ Failed to communicate with Luckfox via ADB: {e}")
        return False

def recuperer_fichier_via_adb():
    print("  Connecting ADB to retrieve YUV file...")
    os.makedirs(os.path.dirname(FICHIER_YUV_LOCAL), exist_ok=True)
    
    commande_adb = [str(ADB_EXE), "pull", FICHIER_LUCKFOX, str(FICHIER_YUV_LOCAL)]
    try:
        resultat = subprocess.run(commande_adb, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if resultat.returncode == 0 and os.path.exists(FICHIER_YUV_LOCAL):
            print(" ✅ YUV file successfully transferred to PC!")
            return True
        else:
            print(" ❌ ADB Error: File could not be copied.")
            return False
    except Exception as e:
        print(f" ❌ System error during adb pull: {e}")
        return False

def extraire_convertir_et_analyser_yolo():
    if not executer_capture_distance():
        print(" Stopped: Remote capture failed.")
        return

    time.sleep(1)

    if not recuperer_fichier_via_adb():
        print(" Stopped: Unable to retrieve file.")
        return

    taille_image_octets = int(WIDTH * HEIGHT * 1.5)
    taille_reelle = os.path.getsize(FICHIER_YUV_LOCAL)
    
    if taille_reelle == 0:
        print(" ❌ FAILURE: Transferred file is empty (0 bytes).")
        return
        
    print(f" Local file validated ({taille_reelle} bytes).")
    print(" Loading YOLOv8 model...")
    
    # Load YOLO model from 5_code/yolov8n.pt
    model = YOLO(str(MODEL_PATH))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(VIDEO_SORTIE), fourcc, FPS, (WIDTH, HEIGHT))

    print(" Analyzing frames and writing video...")
    with open(FICHIER_YUV_LOCAL, "rb") as f:
        for i in range(NOMBRE_IMAGES):
            donnees_brutes = f.read(taille_image_octets)
            if not donnees_brutes or len(donnees_brutes) < taille_image_octets:
                break

            # Decode YUV image to BGR
            tableau_yuv = np.frombuffer(donnees_brutes, dtype=np.uint8)
            matrice_yuv = tableau_yuv.reshape((int(HEIGHT * 1.5), WIDTH))
            image_bgr = cv2.cvtColor(matrice_yuv, cv2.COLOR_YUV2BGR_NV12)
            
            # Run YOLO prediction on frame
            results = model.predict(source=image_bgr, verbose=False)
            
            # Retrieve frame with bounding boxes
            annotated_frame = results[0].plot()
            
            # Live display during conversion
            cv2.imshow("Live YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Write annotated frame to video
            video_writer.write(annotated_frame)

    video_writer.release()
    cv2.destroyAllWindows()
    print(f" Final video with YOLO detections is ready: {VIDEO_SORTIE}")

if __name__ == "__main__":
    extraire_convertir_et_analyser_yolo()