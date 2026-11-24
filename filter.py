"""Filtre les données transmises par le modèle (fichier ajouté)
    filter -> renvoie une liste de détections sans occurence humaine"""

from yolo_pi.processing import Detection

def filter(detections: list[Detection]) -> list[Detection]:
    """Filtre detections pour supprimer les objets non humains détectés"""
    new_detection = []
    for det in detections:
        if det.class_id == 0:
            new_detection.append(det)
    return new_detection