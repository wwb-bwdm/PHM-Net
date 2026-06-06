import os
import cv2
import numpy as np
from glob import glob
from scipy.spatial.distance import cdist
import time
import re
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def detect_stripe_centers(mask, min_width=10):
    h, w = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    white_pixels = np.where(binary == 255)
    if len(white_pixels[0]) == 0:
        raise ValueError("未检测到白色像素（条纹）")

    x_coords = sorted(white_pixels[1])
    stripe_groups = []
    current_group = [x_coords[0]] if x_coords else []
    for x in x_coords[1:]:
        if x - current_group[-1] <= 1:
            current_group.append(x)
        else:
            stripe_groups.append(current_group)
            current_group = [x]
    if current_group:
        stripe_groups.append(current_group)

    stripe_centers = []
    for group in stripe_groups:
        left, right = min(group), max(group)
        if (right - left + 1) >= min_width:
            stripe_centers.append((left + right) / 2)
    return stripe_centers

def select_16_benchmark_stripes_center(benchmark_centers):
    total_benchmark = len(benchmark_centers)
    if total_benchmark < 16:
        raise ValueError(f"基准图片总条纹数不足16条")
    start_idx = (total_benchmark - 16) // 2
    end_idx = start_idx + 16
    benchmark_16 = benchmark_centers[start_idx:end_idx]
    benchmark_16_rightmost = benchmark_16[-1]
    return benchmark_16, benchmark_16_rightmost

def match_benchmark_rightmost_to_target_left(benchmark_16_rightmost, target_centers, max_dist_threshold=50):
    total_target = len(target_centers)
    if total_target < 16:
        raise ValueError(f"目标图片总条纹数不足16条")

    target_left_mask = np.array(target_centers) <= benchmark_16_rightmost
    target_left_centers = np.array(target_centers)[target_left_mask].tolist()
    target_left_indices = np.where(target_left_mask)[0].tolist()
    if not target_left_centers:
        raise ValueError("目标中无左侧条纹")

    target_left_array = np.array(target_left_centers).reshape(-1, 1)
    dist_matrix = cdist(np.array([[benchmark_16_rightmost]]), target_left_array)
    min_dist_idx_in_left = np.argmin(dist_matrix)
    min_dist = dist_matrix[0][min_dist_idx_in_left]

    if min_dist > max_dist_threshold:
        raise ValueError("匹配距离超限")
    return target_left_indices[min_dist_idx_in_left]

def target_select_16_from_rightmost(target_centers, target_match_rightmost_idx):
    target_16_start_idx = target_match_rightmost_idx - 15
    if target_16_start_idx < 0:
        raise ValueError("向左不足16条")
    return target_centers[target_16_start_idx : target_match_rightmost_idx + 1]

def calculate_filtered_displacement(diffs, conversion_factor):
    stripes = np.array(diffs)
    max_idx = np.argmax(stripes)
    stripes_after_max = np.delete(stripes, max_idx)
    min_idx = np.argmin(stripes_after_max)
    filtered_stripes = np.delete(stripes_after_max, min_idx)
    img_mean = np.mean(filtered_stripes)
    return img_mean * conversion_factor

def load_image_groups(folder):
    pattern = re.compile(r"(\d+)_(\d+)_pred")
    groups = {}
    for img_path in glob(os.path.join(folder, "*.png")):
        name = os.path.basename(img_path)
        match = pattern.search(name)
        if not match:
            continue
        group_id = int(match.group(1))
        sub_id = int(match.group(2))
        if group_id not in groups:
            groups[group_id] = {}
        groups[group_id][sub_id] = img_path
    return groups

def run_single_folder_pipeline(folder):
    groups = load_image_groups(folder)
    if not groups:
        print("❌ 未找到符合 x_x_pred 格式的图片")
        return

    print(f"\n===== 共找到 {len(groups)} 组图片 =====")
    index = 1

    for group in sorted(groups.keys()):
        pair = groups[group]
        if 1 not in pair or 2 not in pair:
            continue

        try:
            bench_path = pair[1]
            target_path = pair[2]
            bench_img = cv2.imread(bench_path, cv2.IMREAD_GRAYSCALE)
            target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            h, w = bench_img.shape[:2]

            CONVERSION = (800 * 4.5) / w
            MIN_WIDTH = 10 / 800 * w
            MAX_DIST = 50 / 800 * w

            bench_centers = detect_stripe_centers(bench_img, MIN_WIDTH)
            bench16, right_x = select_16_benchmark_stripes_center(bench_centers)

            target_centers = detect_stripe_centers(target_img, MIN_WIDTH)
            match_idx = match_benchmark_rightmost_to_target_left(right_x, target_centers, MAX_DIST)
            target16 = target_select_16_from_rightmost(target_centers, match_idx)

            diffs = [b - t for b, t in zip(bench16, target16)]


            displacement = calculate_filtered_displacement(diffs, CONVERSION)


            print(f"{index} {displacement:.6f}")
            index += 1

        except Exception:
            continue


if __name__ == "__main__":
    IMAGE_FOLDER = r"test_masks"
    run_single_folder_pipeline(IMAGE_FOLDER)