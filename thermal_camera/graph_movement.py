import csv
import matplotlib.pyplot as plt
from pathlib import Path



CSV_PATH = Path(__file__).resolve().parent / "donnees_thermiques.csv"

timestamps = []
temp_max = []
pos_x = []
pos_y = []

BODY_TEMP_THRESHOLD = 25000  

try:
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            temp = float(row["Temp_Relative_Max"])
            x = int(row["Pos_Hotspot_X"])
            y = int(row["Pos_Hotspot_Y"])
            
            # Filter real movement points within bounds
            if temp > BODY_TEMP_THRESHOLD and (2 <= x <= 24) and (2 <= y <= 16):
                timestamps.append(float(row["Timestamp_s"]))
                temp_max.append(temp)
                pos_x.append(x)
                pos_y.append(y)
                
    print(f"Filtering successful! {len(pos_x)} valid hand points retained from capture.")
except Exception as e:
    print(f"Read error: {e}")
    exit()

if not pos_x:
    print("No valid points found. Check the threshold value.")
    exit()


def smooth(data_list, window=3):
    return [sum(data_list[max(0, i-window):i+1]) / len(data_list[max(0, i-window):i+1]) for i in range(len(data_list))]

pos_x_smoothed = smooth(pos_x, window=3)
pos_y_smoothed = smooth(pos_y, window=3)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


ax1.plot(pos_x_smoothed, pos_y_smoothed, color='darkviolet', linestyle='-', linewidth=2.5, label='Actual Motion')
ax1.scatter(pos_x_smoothed[0], pos_y_smoothed[0], color='limegreen', s=150, label='Start', zorder=5, edgecolors='black')
ax1.scatter(pos_x_smoothed[-1], pos_y_smoothed[-1], color='red', s=150, label='End', zorder=5, edgecolors='black')

ax1.set_xlim(0, 26)
ax1.set_ylim(18, 0) 
ax1.set_title("Movement Trajectory")
ax1.set_xlabel("X Position (pixels)")
ax1.set_ylabel("Y Position (pixels)")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()


ax2.plot(timestamps, temp_max, color='firebrick', linewidth=2)
ax2.set_title("Temperature During Motion")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Relative Temperature")
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()