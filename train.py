#!/usr/bin/env python3
"""Fine-tuning YOLO26n sur ton dataset (ex. HERIDAL exporté de Roboflow au format YOLOv8),
puis export ONNX direct dans models/ pour detect.py / mac_demo.py.

Pendant l'entraînement (voir train_monitor.py) :
  - TensorBoard avec courbes + images d'exemple à chaque epoch :
        tensorboard --logdir runs/tensorboard     puis http://localhost:6006
  - checkpoints/<nom>_best.pt = toujours le meilleur modèle (critère --best-metric).

À lancer sur le Mac (GPU Apple "mps") ou sur Colab (GPU NVIDIA), PAS sur le Pi.

Exemples :
    python train.py --data datasets/heridal/data.yaml
    python train.py --data datasets/heridal/data.yaml --epochs 150 --imgsz 640 --name heridal
    python train.py --data datasets/heridal/data.yaml --export-sizes 640 320 --ncnn
"""

import argparse
import shutil
from pathlib import Path


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"            # GPU NVIDIA (Colab, PC)
    if torch.backends.mps.is_available():
        return "mps"          # GPU Apple Silicon
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning YOLO26n + export ONNX")
    p.add_argument("--data", required=True, help="data.yaml du dataset (format YOLOv8 Roboflow)")
    p.add_argument("--model", default="yolo26n.pt", help="poids de départ (pré-entraînés COCO)")
    p.add_argument("--imgsz", type=int, default=640, help="taille d'entraînement (défaut : 640)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16, help="baisse à 8 si manque de mémoire")
    p.add_argument("--patience", type=int, default=30,
                   help="arrêt anticipé si pas d'amélioration pendant N epochs")
    p.add_argument("--device", default=None, help="0 / mps / cpu (défaut : auto)")
    p.add_argument("--name", default="heridal", help="nom du run et préfixe des modèles exportés")
    p.add_argument("--export-sizes", type=int, nargs="+", default=[640],
                   help="tailles d'export ONNX (défaut : 640 ; 320 inutile si les objets sont petits)")
    p.add_argument("--ncnn", action="store_true", help="exporte aussi en NCNN (plus rapide sur le Pi)")
    p.add_argument("--best-metric", choices=["map50", "map50-95", "recall"], default="map50",
                   help="critère du meilleur checkpoint (défaut : map50 = trouver les personnes, "
                        "sans exiger des boîtes parfaitement ajustées)")
    p.add_argument("--samples", type=int, default=6, help="images d'exemple dans TensorBoard")

    # Réglages anti-surapprentissage (petit dataset). Valeurs Ultralytics par défaut si omis.
    g = p.add_argument_group("régularisation / optimisation")
    g.add_argument("--freeze", type=int, default=None,
                   help="gèle les N premières couches (10 = backbone gardé de COCO, seule la tête apprend)")
    g.add_argument("--optimizer", default="auto",
                   help="SGD, AdamW... ATTENTION : avec 'auto', Ultralytics ignore --lr0")
    g.add_argument("--lr0", type=float, default=None, help="learning rate initial (défaut Ultralytics 0.01)")
    g.add_argument("--cos-lr", action="store_true", help="décroissance cosinus du learning rate")
    g.add_argument("--scale", type=float, default=None,
                   help="augmentation de zoom +/- (défaut 0.5 ; plus bas = petits objets moins écrasés)")
    args = p.parse_args()
    if args.lr0 is not None and args.optimizer == "auto":
        p.error("--lr0 est ignoré avec --optimizer auto : précise --optimizer SGD (ou AdamW)")
    return args


def main():
    args = parse_args()
    from ultralytics import YOLO

    device = args.device or pick_device()
    print(f"Entraînement sur : {device}")

    from train_monitor import TrainMonitor
    monitor = TrainMonitor(args.data, args.name, n_samples=args.samples, best_metric=args.best_metric)

    model = YOLO(args.model)
    model.add_callback("on_train_epoch_end", monitor.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)
    model.add_callback("on_train_end", monitor.on_train_end)
    # Seuls les réglages fournis sont transmis ; sinon Ultralytics garde ses défauts.
    extra = {k: v for k, v in {"freeze": args.freeze, "lr0": args.lr0, "scale": args.scale}.items()
             if v is not None}
    model.train(
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        **extra,
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=str(Path("runs").resolve()),  # chemin absolu : évite runs/detect/runs/...
        name=args.name,
        exist_ok=True,
    )
    # Meilleur checkpoint selon --best-metric (Ultralytics garde aussi le sien,
    # runs/<nom>/weights/best.pt, choisi sur le mAP50-95)
    best = monitor.best_path if monitor.best_path.exists() else Path(model.trainer.best)
    print(f"\nMeilleurs poids : {best}  ({args.best_metric} = {monitor.best_value:.4f})")

    # Évaluation finale (mAP) sur le jeu de validation
    metrics = YOLO(best).val(data=args.data, imgsz=args.imgsz, device=device, verbose=False)
    print(f"mAP50 = {metrics.box.map50:.3f} | mAP50-95 = {metrics.box.map:.3f}")

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    for size in args.export_sizes:
        exported = Path(YOLO(best).export(format="onnx", imgsz=size))
        dest = out_dir / f"{args.name}_{size}.onnx"
        shutil.move(exported, dest)
        print(f"Exporté : {dest}")
        if args.ncnn:
            exported = Path(YOLO(best).export(format="ncnn", imgsz=size))
            dest = out_dir / f"{args.name}_ncnn_model_{size}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(exported, dest)
            print(f"Exporté : {dest}")

    print(f"\nTest sur le Mac : python mac_demo.py --resize stretch --models "
          + " ".join(f"models/{args.name}_{s}.onnx" for s in args.export_sizes)
          + "\n(--resize stretch si les images du dataset ont été étirées en carré par Roboflow)")


if __name__ == "__main__":
    main()
