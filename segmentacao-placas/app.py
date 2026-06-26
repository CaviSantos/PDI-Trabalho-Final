import os
import sys

import cv2
import matplotlib.pyplot as plt

from segmentacao import segmentar_roi_placa_otsu


def processar_array(imagem_rgb, bbox):
    """
    Processa uma imagem já carregada na memória.

    Parâmetros:
        imagem_rgb (numpy.ndarray): imagem em RGB.
        bbox (list): bounding box no formato YOLO [cx, cy, w, h].

    Retorno:
        ResultadoSegmentacao
    """
    return segmentar_roi_placa_otsu(imagem_rgb, bbox)


def processar_imagem(caminho_imagem, bbox):
    """
    Carrega uma imagem do disco e executa a segmentação.

    Parâmetros:
        caminho_imagem (str): caminho da imagem.
        bbox (list): bounding box no formato YOLO [cx, cy, w, h].

    Retorno:
        ResultadoSegmentacao
    """

    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho_imagem}"
        )

    imagem = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2RGB
    )

    return processar_array(imagem, bbox)


def mostrar_resultado(resultado):
    """
    Exibe o resultado da segmentação.
    """

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(resultado.roi)
    plt.title("ROI")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(resultado.mascara, cmap="gray")
    plt.title("Máscara")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(resultado.segmentada)
    plt.title("Segmentada")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def salvar_resultado(resultado, pasta_saida="resultados"):
    """
    Salva todas as imagens geradas pela segmentação.
    """

    os.makedirs(pasta_saida, exist_ok=True)

    cv2.imwrite(
        os.path.join(pasta_saida, "roi.png"),
        cv2.cvtColor(resultado.roi, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(pasta_saida, "mascara.png"),
        resultado.mascara
    )

    cv2.imwrite(
        os.path.join(pasta_saida, "segmentada.png"),
        cv2.cvtColor(resultado.segmentada, cv2.COLOR_RGB2BGR)
    )

    print(f"Resultados salvos em '{pasta_saida}'")


if __name__ == "__main__":

    if len(sys.argv) != 6:
        print(
            "Uso:\n"
            "python app.py <imagem> <cx> <cy> <w> <h>"
        )
        sys.exit(1)

    caminho = sys.argv[1]

    bbox = [
        float(sys.argv[2]),
        float(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
    ]

    resultado = processar_imagem(
        caminho,
        bbox
    )

    mostrar_resultado(resultado)
    salvar_resultado(resultado)