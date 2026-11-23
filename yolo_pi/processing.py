"""Pré-traitement (letterbox + normalisation) et post-traitement (décodage + NMS).

Tout est vectorisé avec NumPy / OpenCV : pas de boucle Python par boîte ou par
classe, sinon le post-traitement deviendrait non négligeable sur le Pi.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LetterboxInfo:
    """Ce qu'il faut pour remettre les boîtes dans le repère de l'image d'origine."""

    scale_x: float  # facteur de redimensionnement horizontal
    scale_y: float  # facteur vertical (= scale_x en letterbox, différent en stretch)
    pad_x: int      # bande ajoutée à gauche (pixels)
    pad_y: int      # bande ajoutée en haut (pixels)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


def preprocess(frame_bgr: np.ndarray, size: int, mode: str = "letterbox") -> tuple[np.ndarray, LetterboxInfo]:
    """Image BGR (H, W, 3) uint8 -> tenseur (1, 3, size, size) float32 dans [0, 1].

    mode="letterbox" : on redimensionne en gardant le ratio et on complète avec du
        gris (114), comme à l'entraînement COCO d'Ultralytics. À utiliser avec les
        poids d'origine.
    mode="stretch" : on déforme l'image en carré, sans bandes. À utiliser quand le
        dataset d'entraînement a été préparé ainsi (Roboflow "Resize ... (Stretch)",
        ex. HERIDAL) : les objets ont alors à l'inférence la même forme qu'à l'entraînement.
    """
    h, w = frame_bgr.shape[:2]
    if mode == "stretch":
        canvas = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        info = LetterboxInfo(size / w, size / h, 0, 0)
    elif mode == "letterbox":
        scale = min(size / h, size / w)
        new_w, new_h = round(w * scale), round(h * scale)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        info = LetterboxInfo(scale, scale, pad_x, pad_y)
    else:
        raise ValueError(f"mode de redimensionnement inconnu : {mode}")

    # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0,1], ajout de la dimension batch.
    # blobFromImage fait tout ça en une passe C++ (plus rapide que 4 ops NumPy).
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, swapRB=True)
    return blob, info


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Non-Maximum Suppression glouton. boxes: (N, 4) en xyxy. Renvoie les indices gardés."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        # IoU de la meilleure boîte avec toutes les autres restantes (vectorisé)
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def postprocess(
    output: np.ndarray,
    info: LetterboxInfo,
    orig_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int = 100,
) -> list[Detection]:
    """Décode la sortie brute de YOLOv8 et applique la NMS.

    Sortie YOLOv8 : (1, 4 + nb_classes, N) avec N = 8400 en 640x640 (2100 en 320).
    Pour chaque ancre : cx, cy, w, h (en pixels de l'entrée), puis un score par
    classe (déjà passé dans une sigmoïde, pas d'"objectness" séparée en v8).
    """
    preds = output[0].T  # (N, 84)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    # Filtre sur le seuil AVANT la NMS : on passe de 8400 candidats à quelques dizaines.
    mask = scores >= conf_threshold
    if not mask.any():
        return []
    preds, scores, class_ids = preds[mask], scores[mask], class_ids[mask]

    # cx, cy, w, h -> x1, y1, x2, y2
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    # NMS par classe en une seule passe : on décale les boîtes de chaque classe
    # d'un offset énorme pour qu'elles ne se chevauchent jamais entre classes.
    offsets = class_ids[:, None].astype(np.float32) * 4096.0
    keep = nms(boxes + offsets, scores, iou_threshold)[:max_detections]
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    # Annule le letterbox : retour dans le repère de l'image caméra.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.scale_y
    h, w = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

    return [
        Detection(*map(float, b), float(s), int(c))
        for b, s, c in zip(boxes, scores, class_ids)
    ]
