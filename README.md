## 1. Overview
The current prototype supports:

- 從 IMX415 擷取 RGB 影像 
- PC 端 YOLOv8 目標偵測 
- MLX90640 熱成像
- Thermal hotspot tracking
- Thermal 運動記錄和軌跡視覺化

Luckfox 板負責 RGB 相機擷取和與 MLX90640 的通訊。影像處理和 YOLO 推理目前在 Windows 主機 PC 上執行。

## 2. System Architecture
YOLO inference is not currently executed on the Luckfox NPU.
```
                Windows PC
                    │
                USB / ADB
                    │
            Luckfox Pico Zero
             /              \
            /                \
        MIPI-CSI              I2C
            │                  │
        IMX415            MLX90640
            │                  │
        RGB frames          Thermal data
            │                  │
            └───── ADB ────────┘
                    ↓
                Windows PC
                /         \
            OpenCV/YOLO    Thermal Processing
```

## 3. Hardware
| Component         | Role                                   | Interface |
| ----------------- | -------------------------------------- | --------- |
| Luckfox Pico Zero | Edge board / camera & sensor interface | USB/ADB   |
| Sony IMX415       | RGB camera                             | MIPI-CSI  |
| MLX90640          | Thermal sensor, 32×24                  | I2C       |


## 4. Software Environment
### Host PC

- Windows
- Conda environment: `clara_rat`
- Python: 3.9.12

Create the environment:

```powershell
conda create -n clara_rat python=3.9.12
conda activate clara_rat
pip install -r requirements.txt
```

## 5. Project Structure
```text id="ebwzsg"
Clara_MiceMonitoring/
│
├── 2_tools/
│   └── adb_fastboot/
│       └── adb.exe
│
└── 5_code/
    ├── README.md
    ├── requirements.txt
    ├── yolov8n.pt
    │
    ├── imx_camera/
    │   ├── capture_convert.py
    │   ├── video_finale.mp4
    │   └── video20.yuv
    │
    ├── imx_camera_yolo/
    │   ├── capture_yolo.py
    │   ├── video_finale_yolo.mp4
    │   └── video20.yuv
    │
    ├── luckfox_code/
    │   └── capture.py
    │
    ├── thermal_camera/
    │   ├── camera_view.py
    │   ├── thermal_tracker.py
    │   ├── graph_movement.py
    │   └── donnees_thermiques.csv
    │
    └── yolo_image/
```

## Luckfox
``5_code/luckfox_code/capture.py`` 編譯完後目前部屬到 Luckfox: ``/root/capture.py``
``capture_480x640``：比例為 480x640 的影像 (廢棄)  
``capture_with_time``：捕捉影像與 frame 之間的時間差的 CSV  
``thermal_newdata_test.py``：測試 8 HZ refresh rate 是否能正確捕捉到資訊 (廢棄 / debug) 
``thermal_read_with_time.py``：拍攝 30 frame thermal information 並記錄時間

## 6. RGB Camera Pipeline
### 6.1 RGB Capture
Remote capture script: `/root/capture.py`

當前配置：
- Device: `/dev/video12`
- Resolution: 640 × 480
- Pixel format: NV12
- Frames per capture: 30
- Output: `/tmp/video20.yuv`

pipeline：執行 ``imx_camera/capture_covert.py``
```
capture_convert.py
      ↓
ADB
      ↓
/root/capture.py
      ↓
v4l2-ctl
      ↓
/dev/video12
      ↓
/tmp/video20.yuv
      ↓
ADB pull
      ↓
OpenCV NV12 → BGR
      ↓
video_finale.mp4
```
### 6.2 RGB + YOLO
pipeline：執行 ``imx_camera_yolo/capture_yolo.py``
```
capture_yolo.py
      ↓
capture RGB
      ↓
ADB pull
      ↓
NV12 → BGR
      ↓
YOLOv8n
      ↓
bounding boxes
      ↓
video_finale_yolo.mp4
```
目前 yolo 是跑在 PC 上的

## 7. Thermal Camera Pipeline
當前配置
- I2C Bus: 2
- Address: 0x33
- Resolution: 32×24
- Read size: 1536 bytes/frame

``camera_view.py``：即時轉換為可視化的熱像圖

``thermal_tracker.py``：即時轉換為可視化的熱像圖，並即時 Hotspot 偵測，並將熱點的時間、相對熱度與位置記錄至 CSV

``graph_movement.py``：將 ``thermal_tracker.py`` 生成的 CSV 轉成軌跡圖與溫度變動圖

## 8. Usage
### 1. Connect the Luckfox
- 將 Luckfox Pico Zero 以 USB Type-C 接到電腦作為電源供應
- 確認 ADB 連線
    ```
    adb devices
    ```
    輸出 ``<device-id>    device``

## 9. dual_camera

用於 RGB camera 與 MLX90640 thermal camera 的同步擷取、
時間對齊、空間校正，以及 thermal-guided RGB ROI 測試。

### 1. Capture & Playback

位於 `/scripts`

- `dual_capture_test.py`
  - 同時觸發 RGB 與 Thermal capture。
  - 儲存兩個 sensor 的影像與 timestamp。

- `dual_playback.py`
  - 根據 timestamp 配對 RGB 與 Thermal frames。
  - 產生雙攝影機同步播放結果。

- `camera_view.py`
  - Thermal camera 即時顯示。

- `view_npy.py`
  - 查看儲存的 thermal `.npy` frame。


### 2. Spatial Calibration

位於 `/calibration`

- `spatial_calibration_click.py`
  - 人工點選 RGB 與 Thermal 中相同位置。
  - 建立 Thermal → RGB correspondence。

- `calibration_points.csv`
  - 9-point calibration 的 correspondence。

- `spatial_alignment_evaluate.py`
  - 建立 Affine transformation。
  - 計算 Full-fit 與 Leave-One-Out spatial error。

- `/9-points_test`
  - 9-point calibration 原始測試資料。

- `/spatial_alignment_evaluation`
  - Affine transformation 與 spatial error 分析結果。


### 3. Thermal-guided RGB ROI

位於 `/roi`

- `thermal_roi_proposal.py`
  - 比較 background 與 target thermal frames。
  - 偵測 Thermal warm region。
  - 將 Thermal bounding box 經 Affine transformation 映射至 RGB。

- `rgb_ground_truth_annotator.py`
  - 手動標註 RGB 中可見手部 Ground Truth bounding box。

- `rgb_ground_truth.csv`
  - Ground Truth 標註結果。

- `roi_margin_sweep.py`
  - 比較 Thermal-generated ROI 與 RGB Ground Truth。
  - 評估不同 safety margin 下的 target containment 與 ROI area。

- `/ROI_test`
  - ROI 測試原始資料。

- `/thermal_roi_results`
  - Thermal → RGB ROI proposal 結果。

- `/roi_margin_evaluation`
  - Safety margin 與 ROI area / containment trade-off 分析結果。

- `roi_demo.py`
  - demo raw ROI + margin (50px)
  - D: 下一張圖, A:上一張圖, esc: 結束 