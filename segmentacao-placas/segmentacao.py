import cv2
import numpy as np

from models import ResultadoSegmentacao

from utils import (
    bbox_yolo_para_pixel,
    expandir_bbox_pixel
)


def segmentar_roi_placa_otsu(
    img_rgb,
    bbox_yolo,
    padding=0.06,
    escala=4
):
    """
    Segmenta a ROI da placa destacando principalmente os caracteres.

    Estratégia:
      1. Recorta a placa usando a bbox;
      2. Aumenta a ROI para melhorar a segmentação visual;
      3. Converte para cinza;
      4. Aplica CLAHE;
      5. Usa BlackHat para destacar regiões escuras sobre fundo claro;
      6. Aplica threshold adaptativo;
      7. Remove bordas externas da placa;
      8. Filtra componentes pequenos ou muito grandes.

    Retorna um objeto ResultadoSegmentacao.
    """

    img_h, img_w = img_rgb.shape[:2]

    x1, y1, x2, y2 = bbox_yolo_para_pixel(
        bbox_yolo,
        img_w,
        img_h
    )

    x1, y1, x2, y2 = expandir_bbox_pixel(
        x1, y1, x2, y2,
        img_w,
        img_h,
        padding=padding
    )

    roi = img_rgb[y1:y2, x1:x2].copy()

    if roi.size == 0:
        mask_vazia = np.zeros((10, 10), dtype=np.uint8)

        return ResultadoSegmentacao(
            roi=roi,
            mascara=mask_vazia,
            segmentada=roi,
            bbox_pixel=(x1, y1, x2, y2)
        )

    # --------------------------------------------------------
    # 1. Aumentar ROI para melhorar a visualização/segmentação
    # --------------------------------------------------------
    h_roi, w_roi = roi.shape[:2]

    roi_up = cv2.resize(
        roi,
        (w_roi * escala, h_roi * escala),
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        roi_up,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------------
    # 2. Melhorar contraste local
    # --------------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(4, 4)
    )

    gray_eq = clahe.apply(gray)

    gray_eq = cv2.bilateralFilter(
        gray_eq,
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )

    # --------------------------------------------------------
    # 3. BlackHat
    # --------------------------------------------------------
    h, w = gray_eq.shape[:2]

    kernel_w = max(9, w // 6)
    kernel_h = max(3, h // 4)

    kernel_blackhat = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_w, kernel_h)
    )

    blackhat = cv2.morphologyEx(
        gray_eq,
        cv2.MORPH_BLACKHAT,
        kernel_blackhat
    )

    # --------------------------------------------------------
    # 4. Threshold
    # --------------------------------------------------------
    _, mask_otsu = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask_adapt = cv2.adaptiveThreshold(
        gray_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7
    )

    mask = cv2.bitwise_or(
        mask_otsu,
        mask_adapt
    )

    # --------------------------------------------------------
    # 5. Remover bordas
    # --------------------------------------------------------
    margem_x = int(0.04 * w)
    margem_y = int(0.16 * h)

    mask[:margem_y, :] = 0
    mask[h - margem_y:, :] = 0
    mask[:, :margem_x] = 0
    mask[:, w - margem_x:] = 0

    # --------------------------------------------------------
    # 6. Morfologia
    # --------------------------------------------------------
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # --------------------------------------------------------
    # 7. Componentes conectados
    # --------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    mask_filtrada = np.zeros_like(mask)

    area_total = h * w

    for i in range(1, num_labels):

        x, y, ww, hh, area = stats[i]

        area_frac = area / area_total
        altura_frac = hh / h
        largura_frac = ww / w

        toca_borda = (
            x <= 1 or
            y <= 1 or
            x + ww >= w - 1 or
            y + hh >= h - 1
        )

        if toca_borda:
            continue

        if area_frac < 0.001:
            continue

        if area_frac > 0.20:
            continue

        if altura_frac < 0.12:
            continue

        if largura_frac > 0.45:
            continue

        mask_filtrada[labels == i] = 255

    if np.sum(mask_filtrada > 0) < 0.01 * area_total:
        mask_filtrada = mask

    # --------------------------------------------------------
    # 8. Aplicar máscara
    # --------------------------------------------------------
    segmentada = cv2.bitwise_and(
        roi_up,
        roi_up,
        mask=mask_filtrada
    )

    return ResultadoSegmentacao(
        roi=roi_up,
        mascara=mask_filtrada,
        segmentada=segmentada,
        bbox_pixel=(x1, y1, x2, y2)
    )