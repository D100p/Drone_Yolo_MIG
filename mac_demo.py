"""Interface légère pour tester en temps réel sur Mac, aux performances d'un Raspberry Pi 5.

Fenêtre OpenCV avec :
  - curseur "conf %"   : seuil de confiance en direct
  - touche m           : bascule entre les modèles (ex. 640 <-> 320)
  - touche p           : active / désactive l'émulation Pi 5 en direct
  - touche s           : capture d'écran annotée dans captures/
  - touche q ou Échap  : quitter

Par défaut le Mac est bridé comme un Pi 5 8 Go : 4 threads, 8 Go de RAM max, caméra
640x480, et temps par frame égal à celui du Pi (voir yolo_pi/pi_emulation.py).

Lancement :
    python mac_demo.py                                   # émulation Pi 5 (temps estimés)
    python mac_demo.py --pi-profile pi_profile.json      # temps mesurés sur ton Pi
    python mac_demo.py --no-pi                           # pleine vitesse du Mac

Au premier lancement, macOS demande l'autorisation d'accès à la caméra pour le
Terminal / VS Code : accepte puis relance.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import datetime

import cv2
import numpy as np

from yolo_pi.camera import Camera
from yolo_pi.detector import Detector
from yolo_pi.draw import draw_detections
from yolo_pi.memory import MemoryGuard
from yolo_pi.pi_emulation import PI_RAM_GB, PI_THREADS, PiEmulator, profile_key

import filter
from scipy.ndimage import rotate
from enhance_detection import global_detection

WIN = "YOLO - temps reel  (m: modele | p: mode Pi | s: capture | q: quitter)"


def parse_args():
    p = argparse.ArgumentParser(description="Démo temps réel YOLO avec fenêtre")
    p.add_argument("--models", nargs="+",
                   default=["models/yolo26n_640.onnx", 
                            "models/yolo26n_320.onnx"],
                   help="un ou plusieurs modèles ; la touche m passe de l'un à l'autre")
    p.add_argument("--camera", default="auto",
                   help="auto (sur Mac = caméra 0, FaceTime / webcam), index caméra ou fichier vidéo")
    p.add_argument("--cam-width", type=int, default=640, help="640x480 comme sur le Pi")
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--resize", choices=["letterbox", "stretch"], default="letterbox",
                   help="stretch pour un modèle entraîné sur un dataset Roboflow étiré (HERIDAL)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--pi", action=argparse.BooleanOptionalAction, default=True,
                   help="émule les performances d'un Pi 5 8 Go (défaut : oui ; --no-pi pour désactiver)")
    p.add_argument("--pi-profile", default=None,
                   help="JSON de temps mesurés sur le Pi (benchmark.py --save-profile)")
    p.add_argument("--max-ram-gb", type=float, default=PI_RAM_GB,
                   help="plafond de RAM du processus en Go (défaut : 8, comme le Pi 5)")
    p.add_argument("--max-frames", type=int, default=0, help="(test) s'arrête après N frames")
    return p.parse_args()


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    """Panneau d'infos semi-transparent en haut à gauche."""
    h = 22 * len(lines) + 12
    w = 12 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t in lines)
    roi = img[0:h, 0:w]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (6, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    # Installé en premier pour que la limite noyau (Linux) couvre aussi le chargement des modèles
    guard = MemoryGuard(args.max_ram_gb)
    print(guard.describe())
    models = [m for m in args.models if Path(m).exists()]
    if not models:
        raise SystemExit(f"Aucun modèle trouvé parmi {args.models}. Voir README (export).")

    emulator = PiEmulator(args.pi_profile)
    pi_mode = args.pi
    # 4 threads comme les 4 cœurs du Pi (fixé au chargement, reste valable en --no-pi)
    threads = PI_THREADS if args.pi else None

    # Les détecteurs sont chargés une seule fois : la bascule (touche m) est instantanée.
    detectors = [Detector(m, conf=args.conf, iou=args.iou, threads=threads, resize=args.resize)
                 for m in models]
    for d, m in zip(detectors, models):
        k = profile_key(d)
        if args.pi:
            t = emulator.target(k)  # échoue tout de suite si aucun temps Pi pour ce modèle
            print(f"  {m}: Pi 5 = {t.total_ms:.0f} ms/frame -> {1000 / t.total_ms:.1f} FPS max "
                  f"({emulator.source(k)})")
    idx = 0

    src = int(args.camera) if args.camera.isdigit() else args.camera
    cam = Camera(src, args.cam_width, args.cam_height)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("conf %", WIN, int(args.conf * 100), 95, lambda v: None)
    cv2.setTrackbarMin("conf %", WIN, 5)

    capture_dir = Path("captures")
    frame_times = deque(maxlen=30)  # FPS = moyenne glissante sur 30 frames
    infer_ms = deque(maxlen=30)
    fid, n, fps = -1, 0, 0.0
    try:
        while True:
            fid, frame0 = cam.read(fid)
            detectors_for_each_frame = []
            for i in range(8):
                frame = rotate(frame0, angle=45*i, reshape=True)
                det = detectors[idx]
                det.conf = cv2.getTrackbarPos("conf %", WIN) / 100.0

                dets, t = det(frame)

                ## Prise en compte des personnes uniquement
                dets = filter.filter(dets)
                detectors_for_each_frame.append(dets)

            frame = frame0
            dets = global_detection(detectors_for_each_frame)

            key_name = profile_key(det)
            if pi_mode:
                t = emulator.throttle(key_name, t)  # attend jusqu'au temps Pi de la frame
            frame_times.append(time.perf_counter())
            infer_ms.append(t.infer_ms)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
            ram_mb = guard.check()  # MemoryError (arrêt propre) si > --max-ram-gb

            mode = (f"mode : Raspberry Pi 5 ({emulator.source(key_name)})" if pi_mode
                    else "mode : pleine vitesse Mac")
            out = draw_detections(frame, dets, names=det.names)
            draw_hud(out, [
                f"{fps:5.1f} FPS",
                mode,
                f"modele : {Path(models[idx]).name} ({det.imgsz}x{det.imgsz})",
                f"inference : {np.mean(infer_ms):5.1f} ms",
                f"pre/post : {t.pre_ms:.1f} / {t.post_ms:.1f} ms",
                f"conf >= {det.conf:.2f} | {len(dets)} objets",
                f"RAM : {ram_mb:.0f} Mo / {args.max_ram_gb:.0f} Go",
                f"{datetime.datetime.now().replace(microsecond=0)}",
            ])
            cv2.imshow(WIN, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m") and len(detectors) > 1:
                idx = (idx + 1) % len(detectors)
                emulator.reset()
                frame_times.clear()
                infer_ms.clear()
            if key == ord("p"):
                pi_mode = not pi_mode
                emulator.reset()
                frame_times.clear()
                infer_ms.clear()
            if key == ord("s"):
                capture_dir.mkdir(exist_ok=True)
                path = capture_dir / time.strftime("capture_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(str(path), out)
                print(f"Capture enregistrée : {path}")
            # Fermeture de la fenêtre avec la croix rouge
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    except MemoryError as e:
        print(f"\nArrêt : {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print(f"Pic de RAM : {guard.peak_bytes / 1024**2:.0f} Mo")
        if emulator.too_slow_frames:
            print(f"Attention : {emulator.too_slow_frames} frames ont été plus lentes que le Pi "
                  f"lui-même (machine hôte chargée ?) : l'émulation n'a pas pu les égaliser.")
        print(f"{n} frames, FPS final affiché : {fps:.1f}")


if __name__ == "__main__":
    main()