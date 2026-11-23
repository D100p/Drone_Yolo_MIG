#!/usr/bin/env python3
"""Vérifie la caméra : type détecté, résolution, débit d'images, et enregistre une photo.

    python check_camera.py                  # auto : CSI si présente, sinon caméra 0
    python check_camera.py --camera csi     # force la caméra CSI (nappe plate)
    python check_camera.py --width 1280 --height 720
"""

import argparse
import time

import cv2

from yolo_pi.camera import Camera


def main():
    p = argparse.ArgumentParser(description="Test de la caméra")
    p.add_argument("--camera", default="auto")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--out", default="test_camera.jpg")
    args = p.parse_args()

    cam = Camera(args.camera, args.width, args.height)
    print(f"Caméra   : {cam.name}")
    print(f"Résolution obtenue : {cam.resolution[0]}x{cam.resolution[1]} "
          f"(demandée {args.width}x{args.height})")
    try:
        fid, frame = cam.read()
        t0 = time.perf_counter()
        for _ in range(args.frames):
            fid, frame = cam.read(fid)
        fps = args.frames / (time.perf_counter() - t0)
    finally:
        cam.release()
    print(f"Capture  : {fps:.1f} FPS sur {args.frames} images")
    print(f"Luminosité moyenne : {frame.mean():.0f}/255 "
          + ("(image noire ? cache d'objectif, nappe ?)" if frame.mean() < 10 else "(OK)"))
    cv2.imwrite(args.out, frame)
    print(f"Photo    : {args.out}")


if __name__ == "__main__":
    main()
