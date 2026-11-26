import cv2
import numpy as np

from yolo_pi.detector import Detector, Timings
from yolo_pi.processing import Detection

import filter

def build_rotations(shape: tuple, n_rotations: int) -> list:
    """Précalcule, pour chaque angle 360/n * i, la matrice M de cv2.warpAffine, la
    taille du canvas élargi (équivalent du reshape=True de scipy, image jamais rognée)
    et Minv = invertAffineTransform(M).

    Par défaut warpAffine(src, M, size) traite M comme la transformation DIRECTE
    origine -> tournée (il l'inverse en interne pour échantillonner) : pour projeter
    une boîte détectée dans l'image TOURNÉE vers l'image d'origine, on applique Minv.
    """
    h, w = shape[:2]
    mats = []
    step = 360.0 / n_rotations
    for i in range(n_rotations):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), step * i, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(round(h * sin + w * cos))
        nh = int(round(h * cos + w * sin))
        # Recentre le contenu dans le canvas élargi.
        M[0, 2] += (nw - w) / 2.0
        M[1, 2] += (nh - h) / 2.0
        mats.append((M, (nw, nh), cv2.invertAffineTransform(M)))
    return mats

def merge_detections(detections_per_angle: list, mats: list, shape: tuple) -> list[Detection]:
    """Fusion : garde l'angle qui a détecté le plus de personnes, puis projette ses
    boîtes dans le repère de l'image d'origine en appliquant Minv (inverse de la
    matrice de rotation : l'image tournée -> l'image d'origine).

    Les coins sont remis dans l'ordre (min/max) et recadrés sur l'image : les boîtes
    ne peuvent plus sortir du cadre ni avoir x1 > x2.
    """
    best_dets, best_Minv = detections_per_angle[0], mats[0][2]
    for dets, (_, _, Minv) in zip(detections_per_angle[1:], mats[1:]):
        if len(dets) > len(best_dets):
            best_dets, best_Minv = dets, Minv
    if not best_dets:
        return []
    h, w = shape[:2]
    out = []
    for d in best_dets:
        # (2, 3) @ (3, 2) -> (2, 2) : ligne = composante, colonne = coin ; on transpose
        # pour avoir un point par ligne : corners = [(x1, y1), (x2, y2)]
        corners = (best_Minv @ np.array([[d.x1, d.y1, 1.0], [d.x2, d.y2, 1.0]]).T).T
        x1, y1 = corners[0]
        x2, y2 = corners[1]
        out.append(Detection(
            float(np.clip(min(x1, x2), 0, w)), float(np.clip(min(y1, y2), 0, h)),
            float(np.clip(max(x1, x2), 0, w)), float(np.clip(max(y1, y2), 0, h)),
            d.score, d.class_id,
        ))
    return out


def detect_rotations(pool, detector: Detector, frame0: np.ndarray, mats: list, workers: int):
    """Lance une inférence par rotation (en parallèle si un pool est fourni) et renvoie
    (liste des détections filtrées par angle, Timings CUMULÉS sur toutes les rotations)."""
    def one(i):
        M, size, _ = mats[i]
        rotated = cv2.warpAffine(frame0, M, size, flags=cv2.INTER_LINEAR,
                                 borderValue=(114, 114, 114))
        dets, t = detector(rotated)
        return filter.filter(dets), t  # personnes uniquement

    if pool is not None and workers > 1:
        results = list(pool.map(one, range(len(mats))))
    else:
        results = [one(i) for i in range(len(mats))]

    all_dets = [r[0] for r in results]
    t_total = Timings(
        sum(r[1].pre_ms for r in results),
        sum(r[1].infer_ms for r in results),
        sum(r[1].post_ms for r in results),
    )
    return all_dets, t_total