"""Émulation des performances d'un Raspberry Pi 5 (8 Go) sur une machine plus rapide.

Principe : on ne peut pas rendre un Mac "identique" à un Cortex-A76 (architecture,
bande passante mémoire et noyaux ONNX différents), donc un simple coefficient serait
faux. On impose plutôt le TEMPS PAR FRAME du Pi : chaque étape tourne normalement,
puis on attend la différence jusqu'au temps que le Pi aurait mis. FPS, latence et
frames perdues deviennent ceux du Pi.

Les temps cibles viennent d'un profil :
  - par défaut : estimations tirées de benchmarks publiés (voir DEFAULT_PROFILE) ;
  - mieux : un profil mesuré sur TON Pi avec
        python benchmark.py --model ... --save-profile pi_profile.json
    puis  python mac_demo.py --pi-profile pi_profile.json   -> timing réel du Pi.
"""

import json
import time
from pathlib import Path

from .detector import Detector, Timings

# Estimations Raspberry Pi 5, FP32, 4 threads, caméra 640x480. Clé : famille_backend_taille.
# YOLO26n 640 : chiffres officiels Ultralytics (docs "Raspberry Pi", Pi 5) ONNX 125.99 ms,
#   NCNN 67.03 ms (inférence seule).
# YOLOv8n 640 : pas de chiffre officiel ; YOLO11n ONNX ~147 ms (6.8 FPS) et YOLOv8n fait
#   ~8.7 GFLOPs contre 6.5 -> ~190 ms ONNX, ~100 ms NCNN (cohérent avec 173-198 / 85-106 ms
#   mesurés par la communauté).
# 320 : 4x moins de calcul mais ~3.5x plus rapide seulement (surcoûts fixes) -> estimé.
# Pré/post : temps mesurés sur Mac Apple Silicon x ~4 (écart mono-cœur A76 vs M-series).
# Un modèle fine-tuné (ex. HERIDAL) garde l'architecture, donc les mêmes temps.
DEFAULT_PROFILE = {
    "yolo26n_onnx_640": {"pre_ms": 5.0, "infer_ms": 126.0, "post_ms": 5.0},
    "yolo26n_onnx_320": {"pre_ms": 1.5, "infer_ms": 36.0, "post_ms": 1.3},
    "yolo26n_ncnn_640": {"pre_ms": 5.0, "infer_ms": 67.0, "post_ms": 5.0},
    "yolo26n_ncnn_320": {"pre_ms": 1.5, "infer_ms": 19.0, "post_ms": 1.3},
    "yolov8n_onnx_640": {"pre_ms": 5.0, "infer_ms": 190.0, "post_ms": 5.0},
    "yolov8n_onnx_320": {"pre_ms": 1.5, "infer_ms": 55.0, "post_ms": 1.3},
    "yolov8n_ncnn_640": {"pre_ms": 5.0, "infer_ms": 100.0, "post_ms": 5.0},
    "yolov8n_ncnn_320": {"pre_ms": 1.5, "infer_ms": 30.0, "post_ms": 1.3},
}

PI_THREADS = 4    # 4 cœurs Cortex-A76
PI_RAM_GB = 8.0


def profile_key(detector: Detector) -> str:
    return f"{detector.family}_{detector.backend.name}_{detector.imgsz}"


def save_profile_entry(path: str, key: str, pre_ms: float, infer_ms: float, post_ms: float) -> None:
    """Ajoute/remplace une entrée dans le fichier profil (les autres entrées sont gardées)."""
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[key] = {"pre_ms": round(pre_ms, 2), "infer_ms": round(infer_ms, 2), "post_ms": round(post_ms, 2)}
    p.write_text(json.dumps(data, indent=2) + "\n")


class PiEmulator:
    def __init__(self, profile_path: str | None = None):
        self.profile = dict(DEFAULT_PROFILE)
        self.calibrated: set[str] = set()
        if profile_path:
            measured = json.loads(Path(profile_path).read_text())
            self.profile.update(measured)
            self.calibrated = set(measured)
        self.too_slow_frames = 0  # frames où la machine hôte a été PLUS lente que le Pi
        self._deadline: float | None = None  # instant où la frame suivante peut sortir

    def source(self, key: str) -> str:
        return "mesuré sur Pi" if key in self.calibrated else "estimation"

    def target(self, key: str) -> Timings:
        if key not in self.profile:
            raise KeyError(f"Pas de temps Pi pour '{key}'. Calibre-le avec "
                           f"benchmark.py --save-profile, ou ajoute-le à DEFAULT_PROFILE.")
        t = self.profile[key]
        return Timings(t["pre_ms"], t["infer_ms"], t["post_ms"])

    def reset(self) -> None:
        """À appeler quand on change de modèle ou qu'on réactive l'émulation."""
        self._deadline = None

    def throttle(self, key: str, actual: Timings) -> Timings:
        """Cadence la boucle au rythme du Pi. Renvoie les temps Pi de la frame.

        On rythme d'une frame à la suivante (et pas seulement le calcul) : tout le
        reste de la boucle (affichage, fenêtre...) est absorbé dans l'attente, donc
        les FPS obtenus sont exactement 1000 / temps_Pi_par_frame.
        """
        target = self.target(key)
        period = target.total_ms / 1e3
        now = time.perf_counter()
        if self._deadline is None:  # première frame : on n'a que le temps de calcul
            self._deadline = now + max(0.0, period - actual.total_ms / 1e3)
        wait_s = self._deadline - now
        if wait_s > 0:
            time.sleep(wait_s)
        else:
            self.too_slow_frames += 1
        # Échéance suivante calée sur la précédente (pas sur "maintenant") : les retards de
        # sleep sont rattrapés à la frame suivante. Si on a plus d'une frame de retard
        # (Mac surchargé), on repart de maintenant au lieu d'enchaîner les frames sans attendre.
        now = time.perf_counter()
        self._deadline = (now if now - self._deadline > period else self._deadline) + period
        return target
