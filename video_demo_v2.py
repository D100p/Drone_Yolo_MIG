#!/usr/bin/env python3
"""Version v2 de video_demo_mp4.py : même lecture de fichier vidéo avec YOLO
(rotations successives, filtre personnes, fusion des détections, émulation Pi 5),
mais plus fluide et avec moins de bugs.

Améliorations par rapport à video_demo_mp4.py :
  - rotations faites avec cv2.warpAffine (matrices précalculées une fois par taille
    d'image) au lieu de scipy.ndimage.rotate (interpolation spline order=3, très lent
    sur CPU) : la rotation elle-même passe de dizaines de ms à ~1 ms ;
  - les inférences des rotations tournent en parallèle dans un pool de threads
    (--no-parallel pour du séquentiel, --workers N pour régler la taille du pool) ;
  - la projection des boîtes vers l'image d'origine applique l'inverse de la matrice
    de rotation (cv2.invertAffineTransform) : plus d'erreur de sens de rotation que
    dans enhance_detection.global_detection, coins remis dans l'ordre et recadrés ;
  - le temps cumulé de TOUTES les rotations est transmis à l'émulation Pi (dans
    video_demo_mp4.py seule la dernière rotation était comptabilisée : FPS affichés
    faux et émulation qui ne cadence pas la boucle) ;
  - le curseur "conf %" est lu une seule fois par frame ;
  - le délai d'affichage n'ajoute plus les 1000/FPS complets à chaque frame quand le
    traitement a déjà pris du temps : on attend seulement le TEMPS RESTANT du budget
    de la frame vidéo -> lecture beaucoup plus fluide quand la machine peine ;
  - --rotations N : nombre de rotations (défaut 8, soit des pas de 45° comme avant).

Fenêtre OpenCV avec :
  - curseur "conf %"   : seuil de confiance en direct
  - touche m           : bascule entre les modèles (ex. 640 <-> 320)
  - touche p           : active / désactive l'émulation Pi 5 en direct
  - touche s           : capture d'écran annotée dans captures/
  - touche q ou Échap  : quitter

Par défaut le PC est bridé comme un Pi 5 8 Go : 4 threads, 8 Go de RAM max,
et temps par frame égal à celui du Pi (voir yolo_pi/pi_emulation.py).

Lancement :
    python video_demo_v2.py --video ma_video.mp4                          # émulation Pi 5 (temps estimés)
    python video_demo_v2.py --video ma_video.mp4 --pi-profile pi_profile.json  # temps mesurés sur ton Pi
    python video_demo_v2.py --video ma_video.mp4 --no-pi                 # pleine vitesse du PC
    python video_demo_v2.py --video ma_video.mp4 --loop                   # boucle infinie
"""


import argparse
import datetime
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from yolo_pi.detector import Detector, Timings
from yolo_pi.draw import draw_detections
from yolo_pi.memory import MemoryGuard
from yolo_pi.pi_emulation import PI_RAM_GB, PI_THREADS, PiEmulator, profile_key
from yolo_pi.processing import Detection

import filter

WIN = "YOLO v2 - lecture MP4  (m: modele | p: mode Pi | s: capture | q: quitter)"


def parse_args():
    p = argparse.ArgumentParser(description="Démo YOLO v2 (plus fluide) sur fichier vidéo MP4 avec fenêtre")
    p.add_argument("--models", nargs="+",
                   default=["models/yolo26n_640.onnx",
                            "models/yolo26n_320.onnx"],
                   help="un ou plusieurs modèles ; la touche m passe de l'un à l'autre")
    p.add_argument("--video", required=True,
                   help="chemin vers le fichier vidéo à analyser (mp4, avi, mov, mkv, ...)")
    p.add_argument("--resize", choices=["letterbox", "stretch"], default="letterbox",
                   help="stretch pour un modèle entraîné sur un dataset Roboflow étiré (HERIDAL)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--rotations", type=int, default=8,
                   help="nombre de rotations de l'image (défaut 8 = pas de 45°) ; "
                        "moins de rotations = plus de FPS")
    p.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True,
                   help="lance les inférences des rotations en parallèle (défaut : oui)")
    p.add_argument("--workers", type=int, default=0,
                   help="taille du pool de threads (0 = auto : min(8, rotations, nb cœurs))")
    p.add_argument("--pi", action=argparse.BooleanOptionalAction, default=True,
                   help="émule les performances d'un Pi 5 8 Go (défaut : oui ; --no-pi pour désactiver)")
    p.add_argument("--pi-profile", default=None,
                   help="JSON de temps mesurés sur le Pi (benchmark.py --save-profile)")
    p.add_argument("--max-ram-gb", type=float, default=PI_RAM_GB,
                   help="plafond de RAM du processus en Go (défaut : 8, comme le Pi 5)")
    p.add_argument("--max-frames", type=int, default=0, help="(test) s'arrête après N frames")
    p.add_argument("--loop", action="store_true",
                   help="boucle la vidéo en boucle infinie")
    args = p.parse_args()
    if args.rotations < 1:
        p.error("--rotations doit être >= 1")
    return args


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
    matrice de rotation : image tournée -> image d'origine).

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
    guard = MemoryGuard(args.max_ram_gb)
    print(guard.describe())

    models = [m for m in args.models if Path(m).exists()]
    if not models:
        raise SystemExit(f"Aucun modèle trouvé parmi {args.models}. Voir README (export).")

    emulator = PiEmulator(args.pi_profile)
    pi_mode = args.pi
    threads = PI_THREADS if args.pi else None

    detectors = [Detector(m, conf=args.conf, iou=args.iou, threads=threads, resize=args.resize)
                 for m in models]
    for d, m in zip(detectors, models):
        k = profile_key(d)
        if args.pi:
            t = emulator.target(k)
            print(f"  {m}: Pi 5 = {t.total_ms:.0f} ms/frame -> {1000 / t.total_ms:.1f} FPS max "
                  f"({emulator.source(k)})")
    idx = 0

    # Pool de threads pour paralléliser les inférences des rotations (ONNX Runtime
    # libère le GIL pendant l'inférence : vrai gain sur les machines multi-cœurs).
    workers = args.workers if args.workers > 0 else min(8, args.rotations, os.cpu_count() or 4)
    if not args.parallel:
        workers = 1
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    step = 360.0 / args.rotations
    print(f"rotations : {args.rotations} (pas de {step:g}°) | inférences en parallèle : "
          f"{'oui' if pool else 'non'} ({workers} thread(s))")

    # ============================================================
    # Ouverture du fichier vidéo (cv2.VideoCapture, pas Camera qui
    # est fait pour une webcam temps réel avec thread)
    # ============================================================
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Fichier vidéo introuvable : {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la vidéo : {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Vidéo : {video_path.name} | {video_w}x{video_h} | {video_fps:.1f} FPS | {total_frames} frames")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("conf %", WIN, int(args.conf * 100), 95, lambda v: None)
    cv2.setTrackbarMin("conf %", WIN, 5)

    capture_dir = Path("captures")
    frame_times = deque(maxlen=30)
    infer_ms = deque(maxlen=30)
    mats_cache = {}  # matrices précalculées par taille d'image
    n, fps = 0, 0.0
    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame0 = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_times.clear()
                    infer_ms.clear()
                    continue
                print("Fin de la vidéo.")
                break

            mats = mats_cache.get(frame0.shape)
            if mats is None:
                mats = build_rotations(frame0.shape, args.rotations)
                mats_cache[frame0.shape] = mats

            det = detectors[idx]
            det.conf = cv2.getTrackbarPos("conf %", WIN) / 100.0  # lu une seule fois par frame

            all_dets, t_total = detect_rotations(pool, det, frame0, mats, workers)
            dets = merge_detections(all_dets, mats, frame0.shape)

            key_name = profile_key(det)
            if pi_mode:
                # Temps CUMULÉ des rotations : l'émulation cadence toute la boucle,
                # pas seulement la dernière inférence.
                t = emulator.throttle(key_name, t_total)
            else:
                t = t_total
            frame_times.append(time.perf_counter())
            infer_ms.append(t.infer_ms)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
            ram_mb = guard.check()

            mode = (f"mode : Raspberry Pi 5 ({emulator.source(key_name)})" if pi_mode
                    else "mode : pleine vitesse PC")

            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress = f"frame {pos}/{total_frames}" if total_frames > 0 else f"frame {pos}"

            out = draw_detections(frame0, dets, names=det.names)
            draw_hud(out, [
                f"{fps:5.1f} FPS",
                mode,
                f"modele : {Path(models[idx]).name} ({det.imgsz}x{det.imgsz})",
                f"inference : {np.mean(infer_ms):5.1f} ms (cumul {args.rotations} rotations)",
                f"pre/post : {t.pre_ms:.1f} / {t.post_ms:.1f} ms",
                f"conf >= {det.conf:.2f} | {len(dets)} objets",
                f"rotations : {args.rotations} | threads : {workers}",
                f"RAM : {ram_mb:.0f} Mo / {args.max_ram_gb:.0f} Go",
                f"video : {video_path.name} | {progress}",
                f"{datetime.datetime.now().replace(microsecond=0)}",
            ])
            cv2.imshow(WIN, out)

            # Cadence de lecture : en mode Pi c'est l'émulation qui a déjà attendu,
            # sinon on attend seulement le TEMPS RESTANT du budget de la frame vidéo
            # (au lieu d'ajouter les 1000/FPS complets à chaque frame).
            if pi_mode:
                delay_ms = 1
            else:
                budget_ms = 1000.0 / video_fps if video_fps > 0 else 0.0
                elapsed_ms = (time.perf_counter() - loop_start) * 1e3
                delay_ms = max(1, int(budget_ms - elapsed_ms)) if budget_ms > 0 else 1
            key = cv2.waitKey(delay_ms) & 0xFF
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
        if pool is not None:
            pool.shutdown(wait=True)
        cap.release()
        cv2.destroyAllWindows()
        print(f"Pic de RAM : {guard.peak_bytes / 1024**2:.0f} Mo")
        if emulator.too_slow_frames:
            print(f"Attention : {emulator.too_slow_frames} frames ont été plus lentes que le Pi "
                  f"lui-même (machine hôte chargée ?) : l'émulation n'a pas pu les égaliser.")
        print(f"{n} frames, FPS final affiché : {fps:.1f}")


if __name__ == "__main__":
    main()
