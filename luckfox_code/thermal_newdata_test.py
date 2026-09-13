import subprocess
import time

BUS = "2"
ADDR = "0x33"

def read_status():
    cmd = ["i2ctransfer","-y", BUS,f"w2@{ADDR}","0x80", "0x00","r2"]
    result = subprocess.check_output(cmd).decode().split()

    if len(result) != 2:
        raise RuntimeError(f"Invalid status response: {result}")

    high = int(result[0], 16)
    low = int(result[1], 16)

    return (high << 8) | low


print("Waiting for MLX90640 new data...")

last_subpage = None
count = 0

while count < 20:
    status = read_status()

    new_data = (status >> 3) & 0x01
    subpage = status & 0x01

    if new_data:
        timestamp = time.clock_gettime(time.CLOCK_MONOTONIC)

        print(
            f"t={timestamp:.6f}  "
            f"status=0x{status:04X}  "
            f"new_data={new_data}  "
            f"subpage={subpage}"
        )

        count += 1

        # Reference acquisition flow resets the status
        subprocess.run(["i2ctransfer","-y", BUS,f"w4@{ADDR}","0x80", "0x00","0x00", "0x30"], check=True)
    time.sleep(0.005)