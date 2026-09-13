import subprocess
import os
import re
import csv


DEVICE = "/dev/video12"
WIDTH = 640
HEIGHT = 480
NOMBRE_IMAGES = 30

YUV_PATH = "/tmp/video20.yuv"
TIMESTAMP_PATH = "/tmp/rgb_frame_timestamps.csv"


# ==============================================================================
# CLEAN OLD FILES
# ==============================================================================

if os.path.exists(YUV_PATH):
    os.remove(YUV_PATH)

if os.path.exists(TIMESTAMP_PATH):
    os.remove(TIMESTAMP_PATH)

print("[Luckfox] Starting RGB capture...")


# ==============================================================================
# V4L2 COMMAND
# ==============================================================================

cmd = [
    "v4l2-ctl",
    f"--device={DEVICE}",
    f"--set-fmt-video=width={WIDTH},height={HEIGHT},pixelformat=NV12",
    "--stream-mmap",
    f"--stream-count={NOMBRE_IMAGES}",
    f"--stream-to={YUV_PATH}",
    "--verbose",
    "--stream-show-delta-now"
]

# ==============================================================================
# START CAPTURE
# ==============================================================================

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)


# Example line:
#
# cap dqbuf: 0 seq: 28412 bytesused: 460800
# ts: 1670.903713 delta now: +93.256 ms
#
pattern = re.compile(
    r"cap dqbuf:\s*(\d+)"
    r"\s+seq:\s*(\d+)"
    r".*?"
    r"ts:\s*([0-9]+\.[0-9]+)"
)


frame_records = []


for line in process.stdout:
    line = line.strip()
    print(line)
    match = pattern.search(line)

    if match:
        buffer_id = int(match.group(1))
        sequence = int(match.group(2))
        timestamp_s = float(match.group(3))

        frame_id = len(frame_records)
        frame_records.append({
            "frame_id": frame_id,
            "buffer_id": buffer_id,
            "sequence": sequence,
            "timestamp_s": timestamp_s
        })

process.wait()


# ==============================================================================
# SAVE TIMESTAMPS
# ==============================================================================

with open(TIMESTAMP_PATH,mode="w",newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "frame_id",
        "buffer_id",
        "sequence",
        "timestamp_monotonic_s"
    ])

    for record in frame_records:
        writer.writerow([
            record["frame_id"],
            record["buffer_id"],
            record["sequence"],
            f"{record['timestamp_s']:.6f}"
        ])


# ==============================================================================
# RESULT
# ==============================================================================

if process.returncode == 0:
    print(f"[Luckfox] Capture completed successfully.")
    print(f"[Luckfox] Frames timestamped: "f"{len(frame_records)}")
    print(f"[Luckfox] Timestamp CSV: "f"{TIMESTAMP_PATH}")

else:
    print(f"[Luckfox] v4l2-ctl error "f"(Code: {process.returncode})")