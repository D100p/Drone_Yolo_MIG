"""Capture caméra dans un thread : la boucle principale récupère toujours la frame
la plus récente au lieu d'attendre la caméra (et sans accumuler de retard).

Deux types de caméra :
  - CSI (nappe plate, Pi Camera Module) : via Picamera2 / libcamera. Sur le Pi 5,
    OpenCV ne sait PAS lire ces caméras directement.
  - USB / webcam du Mac / fichier vidéo : via OpenCV.

source :
  "auto" (défaut) -> caméra CSI si Picamera2 en trouve une, sinon caméra OpenCV n°0
  "csi"           -> caméra CSI (erreur si absente)
  "0", "1", ...   -> caméra OpenCV n° N ; "/dev/videoN" ou fichier vidéo aussi acceptés
"""

import sys
import threading
import time

import cv2
import numpy as np


def _csi_available() -> bool:
    try:
        from picamera2 import Picamera2
        return len(Picamera2.global_camera_info()) > 0
    except Exception:  # Picamera2 absent (Mac, venv sans --system-site-packages) ou erreur libcamera
        return False


class _CsiGrabber:
    """Caméra CSI via Picamera2. Renvoie des images BGR uint8 comme OpenCV."""

    def __init__(self, width: int, height: int, fps: int):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        # "RGB888" chez libcamera = octets dans l'ordre B, G, R : directement utilisable par OpenCV.
        config = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=4,
            controls={"FrameRate": fps},
        )
        self.cam.configure(config)
        # Autofocus continu si le capteur en a un (Camera Module 3)
        if "AfMode" in self.cam.camera_controls:
            from libcamera import controls
            self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.cam.start()
        self.name = self.cam.camera_properties.get("Model", "csi")

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.cam.capture_array("main")  # bloque jusqu'à la frame suivante

    def release(self):
        self.cam.stop()
        self.cam.close()


class _OpenCvGrabber:
    """Webcam USB, webcam du Mac ou fichier vidéo via OpenCV."""

    def __init__(self, source: int | str, width: int, height: int, fps: int):
        # V4L2 explicite sous Linux (Pi) : évite que OpenCV essaie GStreamer d'abord.
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") and isinstance(source, int) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra {source!r}")
        # MJPG : en YUYV (non compressé) beaucoup de webcams USB plafonnent à 5-10 FPS.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # pas de file d'images périmées
        self.name = f"opencv:{source}"

    def read(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()

    def release(self):
        self.cap.release()


def _make_grabber(source, width, height, fps):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if source == "auto":
        if _csi_available():
            return _CsiGrabber(width, height, fps)
        return _OpenCvGrabber(0, width, height, fps)
    if source == "csi":
        if not _csi_available():
            raise RuntimeError(
                "Aucune caméra CSI trouvée. Vérifie `rpicam-hello --list-cameras` et que le venv "
                "a été créé avec --system-site-packages (Picamera2 vient d'apt, pas de pip).")
        return _CsiGrabber(width, height, fps)
    return _OpenCvGrabber(source, width, height, fps)


class Camera:
    def __init__(self, source: int | str = "auto", width: int = 640, height: int = 480, fps: int = 30):
        self._grabber = _make_grabber(source, width, height, fps)
        ok, frame = self._grabber.read()
        if not ok:
            self._grabber.release()
            raise RuntimeError("La caméra s'ouvre mais ne renvoie aucune image")
        self.name = self._grabber.name
        self._frame = frame
        self._frame_id = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def resolution(self) -> tuple[int, int]:
        h, w = self._frame.shape[:2]
        return w, h

    def _loop(self):
        while self._running:
            ok, frame = self._grabber.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()

    def read(self, last_id: int = -1, timeout: float = 1.0) -> tuple[int, np.ndarray]:
        """Renvoie (id, frame) en attendant une frame plus récente que last_id."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id, timeout=timeout)
            return self._frame_id, self._frame

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._grabber.release()
