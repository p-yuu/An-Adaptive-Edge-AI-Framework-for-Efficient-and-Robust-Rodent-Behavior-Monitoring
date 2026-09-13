import subprocess
import time
import csv
import os

FRAME_COUNT = 30
RAW_OUTPUT_PATH = "/tmp/thermal_frames.raw"
TIMESTAMP_OUTPUT_PATH = "/tmp/thermal_frame_timestamps.csv"
I2C_BUS = "2"
I2C_ADDRESS = "0x33"
FRAME_BYTES = 1536

STATUS_REGISTER_HIGH = "0x80"
STATUS_REGISTER_LOW = "0x00"

RAM_START_HIGH = "0x04"
RAM_START_LOW = "0x00"

POLL_INTERVAL_S = 0.005

def read_status():
    cmd = ["i2ctransfer","-y",I2C_BUS,f"w2@{I2C_ADDRESS}",STATUS_REGISTER_HIGH,STATUS_REGISTER_LOW,"r2"]

    try:
        result = (subprocess.check_output(cmd,stderr=subprocess.DEVNULL).decode().split())
    except subprocess.CalledProcessError:
        return None

    if len(result) != 2:
        return None

    try:
        high = int(result[0], 16)
        low = int(result[1], 16)
    except ValueError:
        return None

    return (high << 8) | low

def reset_status():
    cmd = ["i2ctransfer","-y",I2C_BUS,f"w4@{I2C_ADDRESS}",STATUS_REGISTER_HIGH,STATUS_REGISTER_LOW,"0x00","0x30"]

    try:
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return True

    except subprocess.CalledProcessError:
        return False

def wait_for_new_data():
    while True:
        status = read_status()

        if status is None:
            time.sleep(POLL_INTERVAL_S)
            continue

        new_data = (status >> 3) & 0x01
        subpage = status & 0x01

        if new_data:
            timestamp = time.clock_gettime(time.CLOCK_MONOTONIC)
            return timestamp, subpage, status

        time.sleep(POLL_INTERVAL_S)

def read_thermal_frame():
    start_t = time.clock_gettime(time.CLOCK_MONOTONIC)
    cmd = ["i2ctransfer","-y",I2C_BUS,f"w2@{I2C_ADDRESS}",RAM_START_HIGH,RAM_START_LOW,f"r{FRAME_BYTES}"]
    try:
        result = subprocess.check_output(cmd,stderr=subprocess.DEVNULL).decode().split()
    except subprocess.CalledProcessError:
        return None

    end_t = time.clock_gettime(time.CLOCK_MONOTONIC)

    try:
        raw = bytes(int(value, 16) for value in result)
    except ValueError:
        return None

    if len(raw) != FRAME_BYTES:
        return None

    midpoint_t = (start_t + end_t) / 2.0

    return (raw,start_t,midpoint_t,end_t)

def main():
    for path in [RAW_OUTPUT_PATH,TIMESTAMP_OUTPUT_PATH]:
        if os.path.exists(path):
            os.remove(path)

    print("[THERMAL] Starting capture...")
    print(f"[THERMAL] Target frames: "f"{FRAME_COUNT}")

    successful_frames = 0

    with open(RAW_OUTPUT_PATH,"wb") as raw_file, open(TIMESTAMP_OUTPUT_PATH,"w",newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_id",
            "subpage_id",
            "new_data_timestamp_monotonic_s",
            "read_start_monotonic_s",
            "timestamp_monotonic_s",
            "read_end_monotonic_s",
            "read_latency_ms"
        ])

        while successful_frames < FRAME_COUNT:
            (new_data_t,subpage,status) = wait_for_new_data()

            if not reset_status():
                print("[THERMAL] ""Failed to reset status.")
                continue

            result = read_thermal_frame()
            if result is None:
                print("[THERMAL] ""RAM read failed.")
                continue

            (raw,start_t,midpoint_t,end_t) = result
            raw_file.write(raw)
            latency_ms = (end_t - start_t) * 1000.0

            writer.writerow([
                successful_frames,
                subpage,
                f"{new_data_t:.9f}",
                f"{start_t:.9f}",
                f"{midpoint_t:.9f}",
                f"{end_t:.9f}",
                f"{latency_ms:.3f}"
            ])

            print(
                f"[THERMAL] "
                f"frame={successful_frames:04d} "
                f"subpage={subpage} "
                f"new_data={new_data_t:.6f} "
                f"read_mid={midpoint_t:.6f} "
                f"latency={latency_ms:.2f} ms"
            )

            successful_frames += 1

    print()
    print("[THERMAL] Capture finished.")
    print(f"[THERMAL] Successful frames: {successful_frames}")
    print(f"[THERMAL] Raw file: {RAW_OUTPUT_PATH}")
    print(f"[THERMAL] Timestamp CSV: {TIMESTAMP_OUTPUT_PATH}")


if __name__ == "__main__":
    main()