
import os
import cv2
import random
import shutil
import numpy as np
from pathlib import Path
from skimage.exposure import match_histograms

# ===== 配置 =====
DATA_ROOT = r"C:/Users/91819/Desktop/da/NEU-DET"
RATIO     = 2.0
SEED      = 42
OUT_NAME  = "train_cp_poisson_v2_2_0x"

# ===== 策略①：各类缺陷真实尺寸分布（与第二组完全一致）=====
CLASS_SIZE_RANGE = {
    0: (0.42, 0.63, 0.24, 0.52),  # crazing
    1: (0.07, 0.21, 0.15, 0.70),  # inclusion
    2: (0.18, 0.38, 0.19, 0.60),  # patches
    3: (0.37, 0.60, 0.60, 0.60),  # pitted_surface
    4: (0.23, 0.50, 0.21, 0.56),  # rolled_in_scale
    5: (0.10, 0.50, 0.10, 0.50),  # scratches
}

CLASS_NAMES = ['crazing', 'inclusion', 'patches',
               'pitted_surface', 'rolled-in_scale', 'scratches']

# ===== 策略②：梯度位置选择参数（与第二组完全一致）=====
N_CANDIDATE = 10

# ===== 泊松融合参数 =====
MORPH_DILATE_ITER   = 3     # 形态学膨胀迭代次数
FOREGROUND_MIN_RATIO = 0.10  # Otsu 前景占比低于此值时切换为椭圆 Mask


# ==================== 工具函数（与第二组相同）====================

def load_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return boxes


def save_labels(label_path, boxes):
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def yolo_to_pixel(box, img_w, img_h):
    cls, cx, cy, w, h = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    return cls, x1, y1, x2, y2


def pixel_to_yolo(cls, x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return [cls,
            min(max(cx, 0), 1),
            min(max(cy, 0), 1),
            min(max(w,  0), 1),
            min(max(h,  0), 1)]


def compute_gradient_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sx ** 2 + sy ** 2)


def select_low_gradient_position(grad_map, paste_w, paste_h, existing_boxes, img_w, img_h):
    map_h, map_w = grad_map.shape
    best_pos  = None
    best_grad = float('inf')

    for _ in range(N_CANDIDATE * 3):
        if map_w - paste_w <= 0 or map_h - paste_h <= 0:
            px = random.randint(0, max(0, img_w - paste_w - 1))
            py = random.randint(0, max(0, img_h - paste_h - 1))
            return px, py

        px = random.randint(0, map_w - paste_w - 1)
        py = random.randint(0, map_h - paste_h - 1)

        overlap = False
        for box in existing_boxes:
            _, bx1, by1, bx2, by2 = yolo_to_pixel(box, img_w, img_h)
            if not (px + paste_w < bx1 or px > bx2 or
                    py + paste_h < by1 or py > by2):
                overlap = True
                break
        if overlap:
            continue

        region_grad = grad_map[py:py + paste_h, px:px + paste_w].mean()
        if region_grad < best_grad:
            best_grad = region_grad
            best_pos  = (px, py)

    if best_pos is None:
        px = random.randint(0, max(0, img_w - paste_w - 1))
        py = random.randint(0, max(0, img_h - paste_h - 1))
        best_pos = (px, py)

    return best_pos


# ==================== 新增①：亮度对齐 ====================

def align_brightness(instance: np.ndarray, bg_region: np.ndarray) -> np.ndarray:
    """
    直方图匹配：把 instance 的亮度/对比度分布对齐到 bg_region。
    解决粘贴缺陷和背景亮度差异过大导致的矩形色块问题。
    bg_region 太小或为空时直接返回原 instance。
    """
    if bg_region.size == 0 or bg_region.shape != instance.shape:
        return instance
    try:
        matched = match_histograms(instance, bg_region, channel_axis=-1)
        return matched.astype(np.uint8)
    except Exception:
        return instance


# ==================== 新增②：椭圆 Mask 兜底 ====================

def build_ellipse_mask(instance: np.ndarray) -> np.ndarray:
    """
    为离散点状缺陷（如 pitted_surface）生成椭圆 Mask。
    比矩形 Mask 边缘更自然，比 Otsu 二值化更稳定。
    """
    h, w = instance.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    axes = (max(cx - 2, 1), max(cy - 2, 1))  # 留 2px 边距给泊松融合
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


# ==================== 改进版：形态学 Mask（加占比判断）====================

def build_morphological_mask(instance: np.ndarray,
                              dilate_iter: int = MORPH_DILATE_ITER) -> np.ndarray:
    """
    Otsu 二值化提取缺陷轮廓 + 形态学膨胀生成过渡带。
    若前景占比 < FOREGROUND_MIN_RATIO（点状离散缺陷），自动切换为椭圆 Mask。
    """
    gray = cv2.cvtColor(instance, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_ratio = (binary > 0).sum() / binary.size
    if foreground_ratio < FOREGROUND_MIN_RATIO:
        # 点状离散缺陷，Otsu 提取效果差，切换椭圆 Mask
        return build_ellipse_mask(instance)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.dilate(binary, kernel, iterations=dilate_iter)

    if mask.max() == 0:
        return build_ellipse_mask(instance)

    return mask


# ==================== 泊松融合（与 v1 相同）====================

def poisson_paste(bg_img: np.ndarray, instance: np.ndarray,
                  paste_x: int, paste_y: int) -> np.ndarray:
    ih, iw = instance.shape[:2]
    bh, bw = bg_img.shape[:2]

    max_w = bw - paste_x
    max_h = bh - paste_y
    if max_w <= 0 or max_h <= 0:
        return bg_img

    crop_w   = min(iw, max_w)
    crop_h   = min(ih, max_h)
    instance = instance[:crop_h, :crop_w]
    ih, iw   = instance.shape[:2]

    if iw < 8 or ih < 8:
        result = bg_img.copy()
        result[paste_y:paste_y + ih, paste_x:paste_x + iw] = instance
        return result

    mask = build_morphological_mask(instance)

    cx = paste_x + iw // 2
    cy = paste_y + ih // 2
    cx = max(iw // 2 + 1, min(cx, bw - iw // 2 - 1))
    cy = max(ih // 2 + 1, min(cy, bh - ih // 2 - 1))

    try:
        result = cv2.seamlessClone(
            instance, bg_img, mask,
            (cx, cy),
            cv2.NORMAL_CLONE
        )
    except cv2.error:
        result = bg_img.copy()
        result[paste_y:paste_y + ih, paste_x:paste_x + iw] = instance

    return result


# ==================== 主增强函数 ====================

def copy_paste_poisson(bg_img, bg_boxes, src_img, src_boxes, img_w, img_h):
    if not src_boxes:
        return bg_img, bg_boxes

    result_img   = bg_img.copy()
    result_boxes = bg_boxes.copy()

    box = random.choice(src_boxes)
    cls, x1, y1, x2, y2 = yolo_to_pixel(box, img_w, img_h)

    inst_w = x2 - x1
    inst_h = y2 - y1
    if inst_w <= 2 or inst_h <= 2:
        return bg_img, bg_boxes

    instance = src_img[y1:y2, x1:x2]

    # ===== 策略①：自适应缩放（与第二组相同）=====
    min_w, max_w, min_h, max_h = CLASS_SIZE_RANGE.get(cls, (0.05, 0.5, 0.05, 0.5))
    target_w = max(8, min(int(random.uniform(min_w, max_w) * img_w), img_w - 1))
    target_h = max(8, min(int(random.uniform(min_h, max_h) * img_h), img_h - 1))
    instance = cv2.resize(instance, (target_w, target_h))

    # ===== 策略②：低梯度位置选择（与第二组相同）=====
    grad_map = compute_gradient_map(result_img)
    paste_x, paste_y = select_low_gradient_position(
        grad_map, target_w, target_h, result_boxes, img_w, img_h
    )
    paste_x = min(max(0, paste_x), img_w - target_w - 1)
    paste_y = min(max(0, paste_y), img_h - target_h - 1)

    # ===== 新增：亮度对齐（v2 核心修复）=====
    bg_region = result_img[paste_y:paste_y + target_h,
                            paste_x:paste_x + target_w]
    instance = align_brightness(instance, bg_region)

    # ===== 形态学泊松融合 =====
    result_img = poisson_paste(result_img, instance, paste_x, paste_y)

    actual_h = min(target_h, img_h - paste_y)
    actual_w = min(target_w, img_w - paste_x)
    new_box  = pixel_to_yolo(cls, paste_x, paste_y,
                              paste_x + actual_w, paste_y + actual_h,
                              img_w, img_h)
    result_boxes.append(new_box)
    return result_img, result_boxes


# ==================== main ====================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    data_root     = Path(DATA_ROOT)
    train_img_dir = data_root / "train" / "images"
    train_lbl_dir = data_root / "train" / "labels"

    out_img_dir = data_root / OUT_NAME / "images"
    out_lbl_dir = data_root / OUT_NAME / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(train_img_dir.glob("*.jpg")) +
                       list(train_img_dir.glob("*.png")) +
                       list(train_img_dir.glob("*.bmp")))

    print(f"原始 train 图像数量: {len(img_paths)}")

    print("复制原始数据...")
    for img_path in img_paths:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        shutil.copy(str(img_path), str(out_img_dir / img_path.name))
        if lbl_path.exists():
            shutil.copy(str(lbl_path), str(out_lbl_dir / lbl_path.name))

    n_orig   = len(img_paths)
    n_target = int(n_orig * RATIO)
    n_aug    = n_target - n_orig
    print(f"目标总数: {n_target}，需生成增强图: {n_aug}")

    print("开始生成增强图像（亮度对齐 + 形态学泊松融合）...")
    aug_count = 0
    for i in range(n_aug):
        bg_path     = random.choice(img_paths)
        bg_lbl_path = train_lbl_dir / (bg_path.stem + ".txt")
        bg_img      = cv2.imread(str(bg_path))
        if bg_img is None:
            continue
        img_h, img_w = bg_img.shape[:2]
        bg_boxes     = load_labels(bg_lbl_path)

        src_path     = random.choice(img_paths)
        src_lbl_path = train_lbl_dir / (src_path.stem + ".txt")
        src_img      = cv2.imread(str(src_path))
        if src_img is None:
            continue
        src_boxes = load_labels(src_lbl_path)

        new_img, new_boxes = copy_paste_poisson(
            bg_img, bg_boxes, src_img, src_boxes, img_w, img_h
        )

        out_stem = f"aug_{i:05d}"
        cv2.imwrite(str(out_img_dir / f"{out_stem}.jpg"), new_img)
        save_labels(out_lbl_dir / f"{out_stem}.txt", new_boxes)
        aug_count += 1

        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{n_aug}")

    print(f"\n完成！共生成增强图像: {aug_count}")
    print(f"输出目录: {out_img_dir.parent}")
    print(f"\n训练时 data.yaml 的 train 路径改为:")
    print(f"  train: C:/Users/91819/Desktop/data/NEU-DET/{OUT_NAME}/images")


if __name__ == "__main__":
    main()