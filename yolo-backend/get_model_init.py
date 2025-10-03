from ultralytics import YOLO
import shutil, os
m = YOLO("yolov8n.pt")   # or yolov8m.pt
# Ultralytics exposes the local checkpoint path on the model object:
ckpt_path = getattr(m, "ckpt_path", None) or getattr(m, "ckpt", None)
if not ckpt_path or not os.path.isfile(ckpt_path):
    raise SystemExit(f"Couldn't locate downloaded checkpoint: {ckpt_path}")
os.makedirs("weights", exist_ok=True)
dst = "weights/best.pt"
shutil.copy2(ckpt_path, dst)
print("Saved:", dst)