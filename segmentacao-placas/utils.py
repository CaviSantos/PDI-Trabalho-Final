import numpy as np


def pos_processar_bbox(box):
    """
    Garante que os valores permaneçam entre 0 e 1.
    """
    box = np.asarray(box, dtype=np.float32)

    box[0] = np.clip(box[0], 0, 1)
    box[1] = np.clip(box[1], 0, 1)
    box[2] = np.clip(box[2], 0.001, 1)
    box[3] = np.clip(box[3], 0.001, 1)

    return box


def yolo_to_xyxy_norm(box):
    box = pos_processar_bbox(box)

    cx, cy, w, h = box

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.array([
        np.clip(x1, 0, 1),
        np.clip(y1, 0, 1),
        np.clip(x2, 0, 1),
        np.clip(y2, 0, 1)
    ])


def bbox_yolo_para_pixel(box, img_w, img_h):

    x1, y1, x2, y2 = yolo_to_xyxy_norm(box)

    x1 = int(x1 * img_w)
    y1 = int(y1 * img_h)
    x2 = int(x2 * img_w)
    y2 = int(y2 * img_h)

    x1 = max(0, min(x1, img_w - 2))
    y1 = max(0, min(y1, img_h - 2))
    x2 = max(x1 + 1, min(x2, img_w - 1))
    y2 = max(y1 + 1, min(y2, img_h - 1))

    return x1, y1, x2, y2


def expandir_bbox_pixel(x1, y1, x2, y2,
                         img_w, img_h,
                         padding=0.06):

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w - 1, x2 + pad_x)
    y2 = min(img_h - 1, y2 + pad_y)

    return x1, y1, x2, y2