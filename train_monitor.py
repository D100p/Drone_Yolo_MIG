"""Suivi d'entraînement branché sur les callbacks Ultralytics (utilisé par train.py).

À chaque fin d'epoch :
  1. TensorBoard : pertes (train/val), métriques (mAP, précision, rappel), learning rate.
  2. TensorBoard : images d'exemple sur des images de validation FIXES (les mêmes à
     chaque epoch, pour voir le modèle progresser) :
       - "exemples/vue_complete" : vérité terrain en VERT, prédictions en ROUGE ;
       - "exemples/zoom_personnes" : zoom x4 autour de chaque personne annotée
         (indispensable : sur HERIDAL elles font ~11 px).
  3. Checkpoint : si la métrique choisie bat le record, les poids de cette epoch sont
     copiés dans checkpoints/<nom>_best.pt (+ un .json avec l'epoch et les scores).
     Ce fichier est donc TOUJOURS le meilleur modèle vu jusqu'ici.

Ultralytics sauvegarde last.pt juste AVANT d'appeler on_fit_epoch_end : c'est
exactement les poids de l'epoch qu'on vient d'évaluer.
"""

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

METRIC_KEYS = {
    "map50": "metrics/mAP50(B)",
    "map50-95": "metrics/mAP50-95(B)",
    "recall": "metrics/recall(B)",
}
GREEN, RED = (0, 200, 0), (0, 0, 255)  # BGR


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre deux ensembles de boîtes xyxy : (N,4) x (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _grid(tiles: list[np.ndarray], cols: int, pad: int = 4) -> np.ndarray:
    """Assemble des images de même taille en grille (fond gris foncé)."""
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        out[y:y + h, x:x + w] = t
    return out


class TrainMonitor:
    def __init__(self, data_yaml: str, name: str, log_dir: str = "runs/tensorboard",
                 ckpt_dir: str = "checkpoints", n_samples: int = 6, best_metric: str = "map50",
                 conf: float = 0.15, crop: int = 48, zoom: int = 4, max_crops: int = 24):
        # conf 0.15 (et pas 0.25) : en début d'entraînement le modèle est peu confiant,
        # un seuil bas permet de voir les détections émerger dès les premières epochs.
        from torch.utils.tensorboard import SummaryWriter
        from ultralytics.data.utils import check_det_dataset

        self.name = name
        self.metric_key = METRIC_KEYS[best_metric]
        self.best_metric = best_metric
        self.best_value = -1.0
        self.conf, self.crop, self.zoom, self.max_crops = conf, crop, zoom, max_crops

        run_dir = Path(log_dir) / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(str(run_dir))
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.ckpt_dir / f"{name}_best.pt"
        print(f"TensorBoard : {run_dir}  |  meilleur checkpoint ({best_metric}) : {self.best_path}")

        val_dir = Path(check_det_dataset(data_yaml)["val"])
        self.samples = self._pick_samples(val_dir, n_samples)

    # ------------------------------------------------------------------ données
    @staticmethod
    def _read_gt(img_path: Path, w: int, h: int) -> np.ndarray:
        lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lbl.exists():
            return np.zeros((0, 4))
        rows = [list(map(float, l.split()[1:5])) for l in lbl.read_text().splitlines() if l.strip()]
        if not rows:
            return np.zeros((0, 4))
        cx, cy, bw, bh = np.array(rows).T
        return np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)

    def _pick_samples(self, val_dir: Path, n: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """n images de validation contenant des personnes, réparties sur tout le jeu
        (choix déterministe : ce sont les mêmes à chaque epoch et à chaque run)."""
        candidates = []
        for p in sorted(val_dir.glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(p))
            gt = self._read_gt(p, img.shape[1], img.shape[0])
            if len(gt):
                candidates.append((p.name, img, gt))
        idx = np.linspace(0, len(candidates) - 1, min(n, len(candidates))).round().astype(int)
        return [candidates[i] for i in idx]

    # ---------------------------------------------------------------- callbacks
    def on_train_epoch_end(self, trainer):
        self._epoch_trained = True  # une vraie epoch d'entraînement vient de se terminer

    def on_fit_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        w = self.writer

        # Ultralytics rappelle on_fit_epoch_end une dernière fois dans final_eval() pour
        # ré-évaluer SON best.pt (critère mAP50-95), avec epoch+1. Ce n'est pas une
        # epoch : on journalise juste ses scores à part, sans toucher au checkpoint.
        if not getattr(self, "_epoch_trained", False):
            for k, v in (trainer.metrics or {}).items():
                w.add_scalar(f"final_ultralytics_best/{k}", float(v), epoch)
            w.flush()
            return
        self._epoch_trained = False

        # 1. Courbes
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            w.add_scalar(k, float(v), epoch)
        for k, v in (trainer.metrics or {}).items():
            w.add_scalar(k, float(v), epoch)
        for k, v in (trainer.lr or {}).items():
            w.add_scalar(k, float(v), epoch)

        # 2. Meilleur checkpoint (poids de CETTE epoch = last.pt qui vient d'être écrit)
        value = float((trainer.metrics or {}).get(self.metric_key, -1))
        if value > self.best_value and Path(trainer.last).exists():
            self.best_value = value
            shutil.copy2(trainer.last, self.best_path)
            info = {"epoch": epoch, "best_metric": self.best_metric, "value": round(value, 4),
                    "metrics": {k: round(float(v), 4) for k, v in trainer.metrics.items()}}
            self.best_path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
            w.add_text("checkpoint", f"epoch {epoch} : nouveau meilleur {self.best_metric} = {value:.4f}", epoch)
            print(f"  -> nouveau meilleur checkpoint (epoch {epoch}, {self.best_metric} = {value:.4f})")
        w.add_scalar(f"best/{self.best_metric}", self.best_value, epoch)

        # 3. Images d'exemple
        try:
            self._log_examples(trainer, epoch)
        except Exception as e:  # une image ratée ne doit jamais tuer un entraînement d'1 h
            print(f"  (images TensorBoard ignorées à l'epoch {epoch} : {e})")
        w.flush()

    def on_train_end(self, trainer):
        self.writer.close()

    # ------------------------------------------------------------------ images
    def _log_examples(self, trainer, epoch: int):
        from ultralytics import YOLO

        model = YOLO(str(trainer.last))
        results = model.predict([s[1] for s in self.samples], imgsz=trainer.args.imgsz,
                                conf=self.conf, device=str(trainer.device), verbose=False)
        full, crops = [], []
        total_gt = total_found = total_pred = 0
        for (name, img, gt), r in zip(self.samples, results):
            pred = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
            scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
            found = (_iou(gt, pred).max(1) >= 0.5) if len(pred) else np.zeros(len(gt), bool)
            total_gt += len(gt); total_found += int(found.sum()); total_pred += len(pred)

            vis = img.copy()
            for b in gt.astype(int):
                cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), GREEN, 1)
            for b, s in zip(pred.astype(int), scores):
                cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), RED, 1)
                cv2.putText(vis, f"{s:.2f}", (b[0], max(b[1] - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, RED, 1, cv2.LINE_AA)

            # Zooms autour de chaque personne annotée (fond = image déjà annotée)
            H, W = vis.shape[:2]
            half = self.crop // 2
            for b, ok in zip(gt, found):
                if len(crops) >= self.max_crops:
                    break
                cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
                x0, y0 = min(max(cx - half, 0), W - self.crop), min(max(cy - half, 0), H - self.crop)
                c = cv2.resize(vis[y0:y0 + self.crop, x0:x0 + self.crop], None, fx=self.zoom,
                               fy=self.zoom, interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(c, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), GREEN if ok else RED, 3)
                crops.append(c)

            header = f"{name[:40]} | GT {len(gt)} | trouvees {int(found.sum())} | pred {len(pred)}"
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 18), (0, 0, 0), -1)
            cv2.putText(vis, header, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            full.append(vis)

        to_rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # noqa: E731
        self.writer.add_image("exemples/vue_complete", to_rgb(_grid(full, cols=3)), epoch, dataformats="HWC")
        if crops:
            self.writer.add_image("exemples/zoom_personnes", to_rgb(_grid(crops, cols=8)), epoch,
                                  dataformats="HWC")
        # Rappel "à l'oeil" sur les exemples (indicatif : 6 images seulement)
        self.writer.add_scalar("exemples/personnes_trouvees_pct", 100 * total_found / max(total_gt, 1), epoch)
        self.writer.add_scalar("exemples/nb_predictions", total_pred, epoch)
        