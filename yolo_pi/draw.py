"""Dessin des boîtes, labels et FPS sur une image.

`names` = noms des classes du modèle (Detector.names) : COCO pour les poids
d'origine, tes classes pour un modèle fine-tuné.
"""

import cv2
import numpy as np

from .coco import COCO_CLASSES
from .processing import Detection

def _color(class_id: int) -> tuple[int, int, int]:
    # Couleur stable par classe (même classe -> même couleur d'une frame à l'autre)
    rng = np.random.default_rng(class_id)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _label(names: list[str], class_id: int) -> str:
    return names[class_id] if class_id < len(names) else str(class_id)


def draw_detections(frame: np.ndarray, dets: list[Detection], fps: float | None = None,
                    names: list[str] = COCO_CLASSES) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        color = _color(d.class_id)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)

        ## Affichage de la position ajouté
        label = f"{_label(names, d.class_id)} {d.score:.2f} \n Position : {abs(d.x1 + d.x2)/2:.2f}, {abs(d.y1 + d.y2)/2:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(p1[1], th + 4)
        cv2.rectangle(out, (p1[0], y - th - 4), (p1[0] + tw + 2, y), color, -1)
        cv2.putText(out, label, (p1[0] + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(out, f"FPS {fps:.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def format_console(dets: list[Detection], names: list[str] = COCO_CLASSES) -> str:
    if not dets:
        return "-"
    return ", ".join(
        f"{_label(names, d.class_id)} {d.score:.2f} [{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]"
        for d in dets
    )
