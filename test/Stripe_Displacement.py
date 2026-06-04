import os
import cv2
import numpy as np
from glob import glob
from scipy.spatial.distance import cdist
import re


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
        raise ValueError(f"基准图片总条纹数不足16条（实际{total_benchmark}条）")

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

    target_match_original_idx = target_left_indices[min_dist_idx_in_left]
    return target_match_original_idx


def target_select_16_from_rightmost(target_centers, target_match_rightmost_idx):
    target_16_start_idx = target_match_rightmost_idx - 15
    target_16_end_idx = target_match_rightmost_idx + 1

    if target_16_start_idx < 0:
        raise ValueError("向左不足16条")

    target_16 = target_centers[target_16_start_idx:target_16_end_idx]
    return target_16


def calculate_diffs_and_stats(benchmark_16, target_16):
    diffs = [benchmark_16[i] - target_16[i] for i in range(16)]
    abs_diffs = [abs(d) for d in diffs]
    all_abs_avg = np.mean(abs_diffs)
    return all_abs_avg


def get_group_pairs(image_folder):
    img_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob(os.path.join(image_folder, ext)))

    pattern = re.compile(r'(\d+)_(\d+)_pred')
    group_map = {}

    for path in img_paths:
        filename = os.path.basename(path)
        match = pattern.search(filename)
        if not match:
            continue

        group_num = int(match.group(1))
        img_num = int(match.group(2))

        if group_num not in group_map:
            group_map[group_num] = {}
        group_map[group_num][img_num] = path

    pairs = []
    for group_num in sorted(group_map.keys()):
        item = group_map[group_num]
        if 1 in item and 2 in item:
            bench_path = item[1]
            target_path = item[2]
            pairs.append((bench_path, target_path, group_num))

    if not pairs:
        raise ValueError("未找到匹配的图片组")
    return pairs


def process_single_folder(image_folder):
    if not os.path.exists(image_folder):
        print(f"文件夹不存在：{image_folder}")
        return

    try:
        image_pairs = get_group_pairs(image_folder)
    except Exception as e:
        print(e)
        return

    for bench_path, target_path, group_num in image_pairs:
        try:
            bench_img = cv2.imread(bench_path, cv2.IMREAD_GRAYSCALE)
            bench_img = cv2.resize(bench_img, (800, 550))
            if bench_img is None:
                continue

            h, w = bench_img.shape[:2]
            conversion_factor = (800 * 4.5) / w
            min_width = 10 / 800 * w
            max_dist_threshold = 50 / 800 * w

            bench_centers = detect_stripe_centers(bench_img, min_width)
            bench_16, bench_16_rightmost = select_16_benchmark_stripes_center(bench_centers)

            target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            if target_img is None:
                continue
            target_centers = detect_stripe_centers(target_img, min_width)
            target_match_idx = match_benchmark_rightmost_to_target_left(bench_16_rightmost, target_centers,
                                                                        max_dist_threshold)
            target_16 = target_select_16_from_rightmost(target_centers, target_match_idx)

            all_abs_avg = calculate_diffs_and_stats(bench_16, target_16) * conversion_factor

            print(f"{group_num} {all_abs_avg:.6f}")

        except Exception:
            continue


if __name__ == "__main__":

    IMAGE_FOLDER = r"E:\spy\800" 



    process_single_folder(
        image_folder=IMAGE_FOLDER
    )