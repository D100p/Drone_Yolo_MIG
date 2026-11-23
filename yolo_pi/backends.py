"""Moteurs d'inférence. Même interface : infer(blob) -> sortie (1, 4 + nb_classes, N).

Chaque backend expose aussi ce qu'il lit dans les métadonnées écrites par Ultralytics
à l'export : `names` (noms des classes : COCO ou ton dataset fine-tuné), `family`
(ex. "yolo26n") et `input_size`.
"""

import ast
import os
import re
from pathlib import Path

import numpy as np

from .coco import COCO_CLASSES


def _family(description: str) -> str:
    """'Ultralytics YOLO26n model trained on ...' -> 'yolo26n'."""
    m = re.search(r"YOLO(v?\d+)([nsmlx])", description or "")
    return f"yolo{m.group(1)}{m.group(2)}".lower() if m else "unknown"


def _names_from_dict(d) -> list[str]:
    return [d[i] for i in sorted(d)] if d else list(COCO_CLASSES)


class OnnxBackend:
    """ONNX Runtime CPU. Moteur par défaut : simple à installer, bien optimisé ARM."""

    name = "onnx"

    def __init__(self, model_path: str, threads: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = threads  # 4 = un thread par cœur du Pi 5
        opts.inter_op_num_threads = 1        # graphe séquentiel : rien à paralléliser ici
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Taille d'entrée figée dans le modèle exporté (640 ou 320)
        self.input_size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else None

        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = _names_from_dict(ast.literal_eval(meta["names"]) if "names" in meta else None)
        self.family = _family(meta.get("description", ""))

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: blob})[0]


class NcnnBackend:
    """NCNN (Tencent). Souvent plus rapide qu'ONNX Runtime sur CPU ARM.

    Attend le dossier produit par `yolo export model=yolo26n.pt format=ncnn`
    (contient model.ncnn.param, model.ncnn.bin et metadata.yaml).
    """

    name = "ncnn"

    def __init__(self, model_dir: str, threads: int):
        import ncnn

        d = Path(model_dir)
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False  # pas de GPU Vulkan utile sur le Pi
        self.net.load_param(str(d / "model.ncnn.param"))
        self.net.load_model(str(d / "model.ncnn.bin"))
        self._ncnn = ncnn

        meta = self._read_metadata(d / "metadata.yaml")
        self.names = _names_from_dict(meta["names"])
        self.family = _family(meta["description"])
        self.input_size = meta["imgsz"]

    @staticmethod
    def _read_metadata(path: Path) -> dict:
        """Lecture minimale de metadata.yaml (évite d'ajouter PyYAML sur le Pi)."""
        meta = {"names": {}, "description": "", "imgsz": None}
        if not path.exists():
            return meta
        section = None
        for line in path.read_text().splitlines():
            if not line.startswith(" ") and ":" in line:
                key, val = line.split(":", 1)
                section, val = key.strip(), val.strip()
                if section == "description":
                    meta["description"] = val.strip("'\"")
            elif section == "names" and ":" in line:
                k, v = line.split(":", 1)
                meta["names"][int(k)] = v.strip().strip("'\"")
            elif section == "imgsz" and line.strip().startswith("-") and meta["imgsz"] is None:
                meta["imgsz"] = int(line.strip()[1:])
        return meta

    def infer(self, blob: np.ndarray) -> np.ndarray:
        ex = self.net.create_extractor()
        ex.input("in0", self._ncnn.Mat(np.ascontiguousarray(blob[0])))
        _, out = ex.extract("out0")
        return np.asarray(out)[None]  # (84, N) -> (1, 84, N)


def load_backend(model: str, threads: int | None = None):
    threads = threads or min(4, os.cpu_count() or 4)
    if Path(model).is_dir():
        return NcnnBackend(model, threads)
    if model.endswith(".onnx"):
        return OnnxBackend(model, threads)
    raise ValueError(f"Modèle non reconnu : {model} (attendu : .onnx ou dossier *_ncnn_model)")
