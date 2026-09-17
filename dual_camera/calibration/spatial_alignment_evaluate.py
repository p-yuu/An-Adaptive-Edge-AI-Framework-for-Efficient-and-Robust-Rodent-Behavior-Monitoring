import csv
from pathlib import Path

import cv2
import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALIBRATION_ROOT = PROJECT_ROOT / "calibration"
INPUT_CSV = CALIBRATION_ROOT / "calibration_points.csv"

OUTPUT_DIR = CALIBRATION_ROOT / "spatial_alignment_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_FIT_CSV = OUTPUT_DIR / "affine_full_fit.csv"
LOO_CSV = OUTPUT_DIR / "affine_leave_one_out.csv"
MATRIX_TXT = OUTPUT_DIR / "affine_matrix.txt"
FULL_FIT_MAP = OUTPUT_DIR / "affine_full_fit_map.png"
LOO_MAP = OUTPUT_DIR / "affine_loo_map.png"

RGB_WIDTH = 640
RGB_HEIGHT = 480

# Draw an error circle to make the magnitude easier to interpret.
POINT_RADIUS = 6
TEXT_SCALE = 0.45


# ==============================================================================
# LOAD CALIBRATION POINTS
# ==============================================================================

def load_points():
    records = []

    with open(INPUT_CSV, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            records.append({
                "session": row["session"],
                "thermal_x": float(row["thermal_x"]),
                "thermal_y": float(row["thermal_y"]),
                "rgb_x": float(row["rgb_x"]),
                "rgb_y": float(row["rgb_y"]),
            })

    if len(records) < 4:
        raise RuntimeError(
            "At least 4 correspondence points are recommended for this evaluation."
        )

    return records


# ==============================================================================
# AFFINE FITTING
# ==============================================================================

def fit_affine(thermal_points, rgb_points):
    """
    Least-squares affine fit:

        [x_R]   [a b c] [x_T]
        [y_R] = [d e f] [y_T]
                        [ 1 ]

    Using np.linalg.lstsq keeps the evaluation deterministic and uses all
    supplied calibration points.
    """
    thermal_points = np.asarray(thermal_points, dtype=np.float64)
    rgb_points = np.asarray(rgb_points, dtype=np.float64)

    design = np.column_stack([
        thermal_points[:, 0],
        thermal_points[:, 1],
        np.ones(len(thermal_points))
    ])

    # Solve design @ coeff ~= RGB, where coeff has shape (3, 2)
    coeff, _, _, _ = np.linalg.lstsq(design, rgb_points, rcond=None)

    # Convert to OpenCV-style 2x3 affine matrix.
    matrix = coeff.T
    return matrix


def apply_affine(matrix, thermal_point):
    x, y = thermal_point

    predicted = matrix @ np.array([x, y, 1.0], dtype=np.float64)

    return predicted


# ==============================================================================
# FULL-FIT EVALUATION
# ==============================================================================

def evaluate_full_fit(records):
    thermal_points = np.array([
        [r["thermal_x"], r["thermal_y"]]
        for r in records
    ], dtype=np.float64)

    rgb_points = np.array([
        [r["rgb_x"], r["rgb_y"]]
        for r in records
    ], dtype=np.float64)

    matrix = fit_affine(thermal_points, rgb_points)

    results = []

    for record in records:
        predicted = apply_affine(
            matrix,
            (record["thermal_x"], record["thermal_y"])
        )

        actual = np.array(
            [record["rgb_x"], record["rgb_y"]],
            dtype=np.float64
        )

        error_vector = predicted - actual
        error_px = np.linalg.norm(error_vector)

        results.append({
            **record,
            "predicted_rgb_x": float(predicted[0]),
            "predicted_rgb_y": float(predicted[1]),
            "error_dx": float(error_vector[0]),
            "error_dy": float(error_vector[1]),
            "error_px": float(error_px),
        })

    return matrix, results


# ==============================================================================
# LEAVE-ONE-OUT EVALUATION
# ==============================================================================

def evaluate_leave_one_out(records):
    results = []

    for test_index, test_record in enumerate(records):
        train_records = [
            record
            for index, record in enumerate(records)
            if index != test_index
        ]

        thermal_train = np.array([
            [r["thermal_x"], r["thermal_y"]]
            for r in train_records
        ], dtype=np.float64)

        rgb_train = np.array([
            [r["rgb_x"], r["rgb_y"]]
            for r in train_records
        ], dtype=np.float64)

        matrix = fit_affine(thermal_train, rgb_train)

        predicted = apply_affine(
            matrix,
            (test_record["thermal_x"], test_record["thermal_y"])
        )

        actual = np.array(
            [test_record["rgb_x"], test_record["rgb_y"]],
            dtype=np.float64
        )

        error_vector = predicted - actual
        error_px = np.linalg.norm(error_vector)

        results.append({
            **test_record,
            "predicted_rgb_x": float(predicted[0]),
            "predicted_rgb_y": float(predicted[1]),
            "error_dx": float(error_vector[0]),
            "error_dy": float(error_vector[1]),
            "error_px": float(error_px),
        })

    return results


# ==============================================================================
# CSV OUTPUT
# ==============================================================================

def save_results_csv(path, results):
    fields = [
        "session",
        "thermal_x",
        "thermal_y",
        "actual_rgb_x",
        "actual_rgb_y",
        "predicted_rgb_x",
        "predicted_rgb_y",
        "error_dx",
        "error_dy",
        "error_px",
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "session": result["session"],
                "thermal_x": f"{result['thermal_x']:.4f}",
                "thermal_y": f"{result['thermal_y']:.4f}",
                "actual_rgb_x": f"{result['rgb_x']:.2f}",
                "actual_rgb_y": f"{result['rgb_y']:.2f}",
                "predicted_rgb_x": f"{result['predicted_rgb_x']:.2f}",
                "predicted_rgb_y": f"{result['predicted_rgb_y']:.2f}",
                "error_dx": f"{result['error_dx']:.2f}",
                "error_dy": f"{result['error_dy']:.2f}",
                "error_px": f"{result['error_px']:.2f}",
            })


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def draw_map(results, title, output_path):
    canvas = np.zeros(
        (RGB_HEIGHT, RGB_WIDTH, 3),
        dtype=np.uint8
    )

    # Subtle coordinate grid.
    for x in range(0, RGB_WIDTH, 80):
        cv2.line(canvas, (x, 0), (x, RGB_HEIGHT - 1), (45, 45, 45), 1)

    for y in range(0, RGB_HEIGHT, 80):
        cv2.line(canvas, (0, y), (RGB_WIDTH - 1, y), (45, 45, 45), 1)

    cv2.putText(
        canvas,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        canvas,
        "circle = actual, cross = predicted, line = error vector",
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )

    for result in results:
        actual = (
            int(round(result["rgb_x"])),
            int(round(result["rgb_y"]))
        )

        predicted = (
            int(round(result["predicted_rgb_x"])),
            int(round(result["predicted_rgb_y"]))
        )

        # Error vector.
        cv2.line(
            canvas,
            actual,
            predicted,
            (180, 180, 180),
            2
        )

        # Actual point.
        cv2.circle(
            canvas,
            actual,
            POINT_RADIUS,
            (255, 255, 255),
            2
        )

        # Predicted point.
        cv2.drawMarker(
            canvas,
            predicted,
            (255, 255, 255),
            cv2.MARKER_CROSS,
            14,
            2
        )

        label = (
            f"{result['session']} "
            f"{result['error_px']:.1f}px"
        )

        label_x = min(max(actual[0] + 8, 5), RGB_WIDTH - 170)
        label_y = min(max(actual[1] - 8, 65), RGB_HEIGHT - 8)

        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(str(output_path), canvas)


# ==============================================================================
# STATISTICS
# ==============================================================================

def print_statistics(name, results):
    errors = np.array(
        [result["error_px"] for result in results],
        dtype=np.float64
    )

    print()
    print(name)
    print("-" * len(name))

    for result in results:
        print(
            f"{result['session']:12s} "
            f"error={result['error_px']:7.2f} px "
            f"dx={result['error_dx']:+7.2f} "
            f"dy={result['error_dy']:+7.2f}"
        )

    print()
    print(f"Mean error   : {np.mean(errors):.2f} px")
    print(f"Median error : {np.median(errors):.2f} px")
    print(f"Max error    : {np.max(errors):.2f} px")
    print(f"RMSE         : {np.sqrt(np.mean(errors ** 2)):.2f} px")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("======================================")
    print(" Spatial Alignment Evaluation")
    print(" Thermal -> RGB affine mapping")
    print("======================================")
    print()
    print(f"Input : {INPUT_CSV}")
    print(f"Output: {OUTPUT_DIR}")

    if not INPUT_CSV.exists():
        print()
        print("[ERROR] calibration_points.csv not found.")
        return

    records = load_points()

    print(f"Loaded {len(records)} correspondence points.")

    # Full fit.
    matrix, full_results = evaluate_full_fit(records)

    # Leave-one-out validation.
    loo_results = evaluate_leave_one_out(records)

    save_results_csv(FULL_FIT_CSV, full_results)
    save_results_csv(LOO_CSV, loo_results)

    draw_map(
        full_results,
        "Affine full-fit error map",
        FULL_FIT_MAP
    )

    draw_map(
        loo_results,
        "Affine leave-one-out validation map",
        LOO_MAP
    )

    with open(MATRIX_TXT, "w") as file:
        file.write("Thermal -> RGB affine matrix\n")
        file.write("============================\n\n")
        file.write(np.array2string(matrix, precision=9))
        file.write("\n\n")
        file.write(
            "x_rgb = "
            f"{matrix[0,0]:.9f} * x_thermal + "
            f"{matrix[0,1]:.9f} * y_thermal + "
            f"{matrix[0,2]:.9f}\n"
        )
        file.write(
            "y_rgb = "
            f"{matrix[1,0]:.9f} * x_thermal + "
            f"{matrix[1,1]:.9f} * y_thermal + "
            f"{matrix[1,2]:.9f}\n"
        )

    print()
    print("Affine matrix:")
    print(matrix)

    print_statistics("Full-fit error", full_results)
    print_statistics("Leave-one-out validation", loo_results)

    print()
    print("Generated files:")
    print(f"  {MATRIX_TXT}")
    print(f"  {FULL_FIT_CSV}")
    print(f"  {LOO_CSV}")
    print(f"  {FULL_FIT_MAP}")
    print(f"  {LOO_MAP}")
    print()
    print(
        "Interpretation: full-fit shows how well one affine model fits the "
        "nine supplied points; leave-one-out better estimates how well that "
        "model predicts an unseen calibration location."
    )


if __name__ == "__main__":
    main()
