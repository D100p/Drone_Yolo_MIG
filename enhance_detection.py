"""Améliore la détection de personnes par utilisation de rotations à 45° successives
"""
from yolo_pi.processing import Detection
import numpy as np

def global_detection(detections_per_frame: list[list[Detection]]) -> list[Detection]:
    """Renvoie une liste de détections signalant au moins autant de personnes qu'il y a
    de détections"""

    ## Sélection du plus grand nombre de détections en une image
    main_det, idx = np.array(detections_per_frame[0]), 0
    for i in range(len(detections_per_frame)):
        if len(detections_per_frame[i]) > len(main_det):
            main_det = np.array(detections_per_frame[i]).copy()
            idx = i

    ## Création de la liste de détections finale
    if idx != 0:
        rotation_matrix = np.array([[np.cos(np.pi*45*idx/180), np.sin(np.pi*45*idx/180)],
                                    [-np.sin(np.pi*45*idx/180), np.cos(np.pi*45*idx/180)]])
        for detection in main_det:
            vec1 = np.dot(rotation_matrix, np.array([[detection.x1], [detection.y1]]))
            vec2 = np.dot(rotation_matrix, np.array([[detection.x2], [detection.y2]]))
            
            detection.x1, detection.y1 = vec1[0, 0], vec1[1, 0]
            detection.x2, detection.y2 = vec2[0, 0], vec2[1, 0]

    return main_det.tolist()