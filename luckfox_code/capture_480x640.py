import subprocess
import os


DEVICE = "/dev/video12"
WIDTH = 480
HEIGHT = 640
NOMBRE_IMAGES = 30
YUV_PATH = "/tmp/video20.yuv"

if os.path.exists(YUV_PATH):
    os.remove(YUV_PATH)

print("[Luckfox] Starting capture on main channel...")

# Standard validated command for clean Rockchip firmware
cmd = f"v4l2-ctl --device={DEVICE} --set-fmt-video=width={WIDTH},height={HEIGHT},pixelformat=NV12 --stream-mmap --stream-to={YUV_PATH} --stream-count={NOMBRE_IMAGES}"

PROCES_CAPTURE = subprocess.run(cmd, shell=True)

if PROCES_CAPTURE.returncode == 0:
    print("[Luckfox] Capture completed successfully!")
else:
    print(f"[Luckfox] v4l2-ctl error (Code: {PROCES_CAPTURE.returncode})")