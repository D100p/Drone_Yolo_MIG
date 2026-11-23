#!/usr/bin/env python3
"""Benchmark : FPS moyens sur N frames (100 par défaut), avec détail par étape.

Deux modes :
  --source camera  : pipeline réel (capture + détection), ce que tu auras en vrai
  --source image   : une image fixe en boucle -> mesure le calcul seul, sans la caméra
                     (utile pour comparer 640 vs 320 ou ONNX vs NCNN de façon reproductible)

Exemples :
    python benchmark.py --model yolov8n.onnx --source image
    python benchmark.py --model yolov8n_320.onnx --source camera --frames 100
"""

import argparse
import statistics
import time

import cv2
import numpy as np

from yolo_pi.detector import Detector


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FPS YOLOv8n")
    p.add_argument("--model", default="models/yolo26n_640.onnx")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--resize", choices=["letterbox", "stretch"], default="letterbox")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--frames", type=int, default=100, help="nombre de frames mesurées (défaut : 100)")
    p.add_argument("--warmup", type=int, default=10,
                   help="frames de chauffe non comptées (allocations, caches, fréquence CPU)")
    p.add_argument("--source", choices=["image", "camera"], default="image")
    p.add_argument("--image", default=None, help="image à utiliser en mode image (défaut : bruit aléatoire 640x480)")
    p.add_argument("--camera", default="auto", help="auto, csi, index OpenCV ou fichier vidéo")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--save-profile", default=None,
                   help="(sur le Pi) enregistre les temps moyens dans ce JSON, pour mac_demo.py --pi-profile")
    return p.parse_args()


def pct(values, q):
    return float(np.percentile(values, q))


def main():
    args = parse_args()
    det = Detector(args.model, args.imgsz, args.conf, threads=args.threads, resize=args.resize)

    cam = None
    if args.source == "camera":
        from yolo_pi.camera import Camera
        src = int(args.camera) if args.camera.isdigit() else args.camera
        cam = Camera(src, args.cam_width, args.cam_height)
        get_frame = lambda last: cam.read(last)  # noqa: E731
    else:
        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                raise SystemExit(f"Impossible de lire {args.image}")
        else:
            img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        get_frame = lambda last: (last + 1, img)  # noqa: E731

    print(f"Modèle {args.model} ({det.family}) | entrée {det.imgsz}x{det.imgsz} | source {args.source} | "
          f"{args.warmup} frames de chauffe + {args.frames} mesurées")

    fid = -1
    for _ in range(args.warmup):
        fid, frame = get_frame(fid)
        det(frame)

    pre, inf, post, total, n_obj = [], [], [], [], 0
    t_start = time.perf_counter()
    for _ in range(args.frames):
        t0 = time.perf_counter()
        fid, frame = get_frame(fid)
        dets, t = det(frame)
        total.append((time.perf_counter() - t0) * 1e3)  # inclut l'attente caméra
        pre.append(t.pre_ms)
        inf.append(t.infer_ms)
        post.append(t.post_ms)
        n_obj += len(dets)
    elapsed = time.perf_counter() - t_start
    if cam:
        cam.release()

    print("\n=== Résultats ===")
    print(f"FPS moyen        : {args.frames / elapsed:6.2f}  ({args.frames} frames en {elapsed:.2f} s)")
    print(f"Latence / frame  : moy {statistics.mean(total):6.1f} ms | p50 {pct(total, 50):6.1f} | p95 {pct(total, 95):6.1f}")
    print(f"  pré-traitement : moy {statistics.mean(pre):6.2f} ms")
    print(f"  inférence      : moy {statistics.mean(inf):6.2f} ms | p95 {pct(inf, 95):6.1f}")
    print(f"  post-traitement: moy {statistics.mean(post):6.2f} ms")
    print(f"Objets détectés  : {n_obj / args.frames:.1f} par frame en moyenne")

    if args.save_profile:
        from yolo_pi.pi_emulation import profile_key, save_profile_entry
        key = profile_key(det)
        save_profile_entry(args.save_profile, key, statistics.mean(pre),
                           statistics.mean(inf), statistics.mean(post))
        print(f"Profil '{key}' enregistré dans {args.save_profile}")


if __name__ == "__main__":
    main()
