#!/usr/bin/env python3
"""Détection temps réel YOLO (YOLO26n par défaut, YOLOv8n ou modèle fine-tuné) sur webcam USB.

Exemples :
    python detect.py
    python detect.py --model models/yolo26n_320.onnx --conf 0.4 --save-dir out --save-every 10
    python detect.py --model models/yolo26n_ncnn_model_320
    python detect.py --model models/heridal_640.onnx        # modèle fine-tuné (train.py)
"""

import argparse
import signal
import time
from pathlib import Path

import cv2

from yolo_pi.camera import Camera
from yolo_pi.detector import Detector
from yolo_pi.draw import draw_detections, format_console


def parse_args():
    p = argparse.ArgumentParser(description="Détection d'objets YOLO temps réel (Raspberry Pi 5)")
    p.add_argument("--model", default="models/yolo26n_640.onnx",
                   help="fichier .onnx ou dossier *_ncnn_model (défaut : models/yolo26n_640.onnx)")
    p.add_argument("--imgsz", type=int, default=None,
                   help="taille d'entrée du réseau, 640 ou 320 (défaut : lue dans le modèle)")
    p.add_argument("--resize", choices=["letterbox", "stretch"], default="letterbox",
                   help="letterbox pour les poids COCO, stretch pour un modèle entraîné sur un "
                        "dataset Roboflow étiré (ex. HERIDAL)")
    p.add_argument("--conf", type=float, default=0.25, help="seuil de confiance (défaut : 0.25)")
    p.add_argument("--iou", type=float, default=0.45, help="seuil IoU de la NMS (défaut : 0.45)")
    p.add_argument("--camera", default="auto",
                   help="auto (CSI si présente, sinon caméra 0), csi, index OpenCV (0, 1...), "
                        "/dev/videoN ou fichier vidéo")
    p.add_argument("--cam-width", type=int, default=640, help="largeur demandée à la caméra")
    p.add_argument("--cam-height", type=int, default=480, help="hauteur demandée à la caméra")
    p.add_argument("--threads", type=int, default=None, help="threads d'inférence (défaut : 4)")
    p.add_argument("--save-dir", default=None, help="dossier où sauvegarder les images annotées")
    p.add_argument("--save-every", type=int, default=1,
                   help="sauvegarde une image sur N (défaut : 1 = toutes)")
    p.add_argument("--save-only-detections", action="store_true",
                   help="ne sauvegarde que les images contenant au moins une détection")
    p.add_argument("--show", action="store_true",
                   help="affiche une fenêtre vidéo (nécessite un écran ; q pour quitter)")
    p.add_argument("--max-frames", type=int, default=0, help="s'arrête après N frames (0 = infini)")
    p.add_argument("--quiet", action="store_true", help="n'affiche que les FPS, pas les détections")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera

    detector = Detector(args.model, args.imgsz, args.conf, args.iou, args.threads, args.resize)
    cam = Camera(source, args.cam_width, args.cam_height)
    print(f"Modèle {args.model} ({detector.family}, {len(detector.names)} classes) | "
          f"entrée {detector.imgsz}x{detector.imgsz} | "
          f"caméra {cam.name} {cam.resolution[0]}x{cam.resolution[1]} | conf {args.conf}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    stop = False

    def on_sigint(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)  # Ctrl+C -> sortie propre (caméra libérée)

    frame_id, n, fps = -1, 0, 0.0
    t_prev = time.perf_counter()
    try:
        while not stop:
            frame_id, frame = cam.read(frame_id)
            dets, t = detector(frame)
            n += 1

            # FPS lissé (moyenne mobile exponentielle) pour un affichage lisible
            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            fps = inst if n == 1 else 0.9 * fps + 0.1 * inst

            line = (f"[{n:6d}] {fps:5.1f} FPS | pre {t.pre_ms:5.1f} ms  inf {t.infer_ms:6.1f} ms  "
                    f"post {t.post_ms:4.1f} ms | {len(dets)} obj")
            print(line if args.quiet else f"{line} | {format_console(dets, detector.names)}", flush=True)

            want_save = save_dir and n % args.save_every == 0 and (dets or not args.save_only_detections)
            if want_save or args.show:
                annotated = draw_detections(frame, dets, fps, detector.names)
                if want_save:
                    cv2.imwrite(str(save_dir / f"frame_{n:06d}.jpg"), annotated)
                if args.show:
                    cv2.imshow("YOLOv8n", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cam.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"\nTerminé : {n} frames.")


if __name__ == "__main__":
    main()
