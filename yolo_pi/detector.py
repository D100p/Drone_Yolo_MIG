"""Pipeline complet : pré-traitement -> inférence -> post-traitement, avec chrono par étape."""

import time
from dataclasses import dataclass

import numpy as np

from .backends import load_backend
from .processing import Detection, postprocess, preprocess


@dataclass
class Timings:
    pre_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.pre_ms + self.infer_ms + self.post_ms


class Detector:
    def __init__(self, model: str, imgsz: int | None = None, conf: float = 0.25,
                 iou: float = 0.45, threads: int | None = None, resize: str = "letterbox"):
        self.backend = load_backend(model, threads)
        self.resize = resize  # "letterbox" (poids COCO) ou "stretch" (dataset Roboflow étiré)
        fixed = self.backend.input_size
        if fixed and imgsz and imgsz != fixed:
            raise ValueError(
                f"Le modèle a été exporté en {fixed}x{fixed} mais --imgsz={imgsz}. "
                f"Ré-exporte avec imgsz={imgsz} ou retire --imgsz."
            )
        self.imgsz = fixed or imgsz or 640
        self.conf = conf
        self.iou = iou
        self.names = self.backend.names    # noms des classes lus dans le modèle
        self.family = self.backend.family  # ex. "yolo26n", "yolov8n"

    def __call__(self, frame_bgr: np.ndarray) -> tuple[list[Detection], Timings]:
        t0 = time.perf_counter()
        blob, info = preprocess(frame_bgr, self.imgsz, self.resize)
        t1 = time.perf_counter()
        out = self.backend.infer(blob)
        t2 = time.perf_counter()
        dets = postprocess(out, info, frame_bgr.shape[:2], self.conf, self.iou)
        t3 = time.perf_counter()
        return dets, Timings((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
