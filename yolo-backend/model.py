from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_local_path
from ultralytics import YOLO
import os

import os, glob, shutil, uuid, json
from pathlib import Path
from PIL import Image
from label_studio_ml.utils import get_local_path

# Env var so you can change weights without editing code:
#   export YOLO_WEIGHTS="weights/best.pt"
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "weights/best.pt")
print("IM RUNNING")

def _ls_rect_to_yolo(rect, iw, ih, cls_id=0):
    # LS gives percent coords. Convert to normalized YOLO [cx, cy, w, h] in 0..1
    x = rect["value"]["x"] / 100.0 * iw
    y = rect["value"]["y"] / 100.0 * ih
    w = rect["value"]["width"]  / 100.0 * iw
    h = rect["value"]["height"] / 100.0 * ih
    cx = (x + w / 2.0) / iw
    cy = (y + h / 2.0) / ih
    return cls_id, cx, cy, w / iw, h / ih


class LogoDetector(LabelStudioMLBase):
    # Must match your Label Studio labeling config:
    FROM_NAME = "label"   # <RectangleLabels name="label" ...>
    TO_NAME   = "image"   # <Image name="image" ...>
    LABEL     = "logo"    # <Label value="logo" ...>

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = YOLO(YOLO_WEIGHTS)  # loads your YOLOv8/YOLO11 .pt
        self.imgsz = int(os.getenv("YOLO_IMGSZ", "1280"))
        self.conf  = float(os.getenv("YOLO_CONF", "0.25"))
        self.iou   = float(os.getenv("YOLO_IOU", "0.6"))

    def process_event(self, *args, event=None, data=None, job_id=None, **kwargs):
        """
        Handle Label Studio webhook/model events.
        LS often calls: process_event(event, data, job_id, additional_params)
        We normalize args → (event, data, job_id) and ALWAYS return a dict.
        """
        # Normalize positional args if keywords weren't supplied
        if event is None and len(args) > 0:
            event = args[0]
        if data is None and len(args) > 1:
            data = args[1]
        if job_id is None and len(args) > 2:
            job_id = args[2]

        etype = (str(event) if event is not None else "").upper()

        # Events we want to acknowledge but not act on
        ignore = {
            "ANNOTATION_CREATED",
            "ANNOTATION_UPDATED",
            "ANNOTATION_DELETED",
            "TASKS_CREATED",
            "TASKS_DELETED",
            "DATA_IMPORTED",
            "DATA_UPDATED",
            "PROJECT_UPDATED",
            "WEBHOOK_TEST"
        }
        if etype in ignore:
            return {"ok": True, "ignored_event": etype}

        # Known training triggers (varies by LS version)
        train_events = {"MODEL_TRAIN", "START_TRAINING", "TRAIN"}
        if etype in train_events:
            # Delegate to fit(); make sure it returns a dict
            return self.fit(event=event, data=data, job_id=job_id, **kwargs)

        # Default fallback
        return {"ok": True, "message": f"No handler for event '{etype}'"}

    def predict(self, tasks, **kwargs):
        """Return Label Studio-style rectangle predictions (percent coords)."""
        predictions = []
        for t in tasks:
            # Resolve local path for the image (handles remote URLs too)
            image_path = get_local_path(t["data"].get("image") or t["data"].get("image_url"))
            res = self.model.predict(
                source=image_path,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )[0]

            h, w = res.orig_shape  # (H, W)
            results = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), sc in zip(xyxy, confs):
                    results.append({
                        "from_name": self.FROM_NAME,
                        "to_name": self.TO_NAME,
                        "type": "rectanglelabels",
                        "original_width": int(w),
                        "original_height": int(h),
                        "value": {
                            "x": 100.0 * x1 / w,
                            "y": 100.0 * y1 / h,
                            "width": 100.0 * (x2 - x1) / w,
                            "height": 100.0 * (y2 - y1) / h,
                            "rotation": 0,
                            "rectanglelabels": [self.LABEL],
                        },
                        "score": float(sc)
                    })

            predictions.append({
                "result": results,
                "score": float(sum([r["score"] for r in results])/len(results)) if results else 0.0,
                "model_version": "yolo-prelabel-v1"
            })
        print('RETURNING PREDICTIONS')
        return {"predictions": predictions}
    

    def fit(self, *args, event=None, data=None, **kwargs):
        """
        Called by Label Studio when you click 'Start training' or enable
        'Start model training on annotation submission'.
        Expects `data` to contain annotated tasks.
        """
        # Where we’ll stage a small YOLO dataset for fine-tuning
        workdir = Path(os.getenv("YOLO_TRAIN_DIR", "ls_train")) / str(uuid.uuid4())  # unique run
        img_train = workdir / "images" / "train"
        lbl_train = workdir / "labels" / "train"
        img_val   = workdir / "images" / "val"
        lbl_val   = workdir / "labels" / "val"
        for p in [img_train, lbl_train, img_val, lbl_val]:
            p.mkdir(parents=True, exist_ok=True)

        def normalize_items(payload):
            items = []

            if isinstance(payload, dict):
                # Shape A: {"tasks":[...], "annotations":[...]}
                tasks = payload.get("tasks") or payload.get("data") or []
                anns  = payload.get("annotations") or []
                if isinstance(tasks, list) and tasks:
                    if anns:
                        by_tid = {}
                        for a in anns:
                            tid = a.get("task") or a.get("task_id")
                            if tid is not None:
                                by_tid.setdefault(tid, []).append(a)
                        for t in tasks:
                            tid = t.get("id") or t.get("task_id")
                            items.append((t, by_tid.get(tid, [])))
                    else:
                        # tasks may have nested 'annotations'
                        for t in tasks:
                            items.append((t, t.get("annotations", [])))

                # Shape B: {"task": {...}, "annotation": {...}}
                elif payload.get("task") and payload.get("annotation"):
                    t = payload["task"]
                    a = payload["annotation"]
                    items.append((t, [a]))

            elif isinstance(payload, list):
                # Some versions send list of records each with {task, annotation}
                for rec in payload:
                    if isinstance(rec, dict) and rec.get("task") and rec.get("annotation"):
                        items.append((rec["task"], [rec["annotation"]]))

            return items

        items = normalize_items(data or {})
        # -------------------------------------------------------------------------------

        # --- your existing loop, unchanged except it now iterates "items" ---
        counter = 0
        for task, ann_list in items:
            img_url = (
                task.get("data", {}).get("image")
                or task.get("data", {}).get("image_url")
            )
            if not img_url:
                continue
            local_img = get_local_path(img_url)
            if not os.path.isfile(local_img):
                continue

            with Image.open(local_img) as im:
                iw, ih = im.size

            # gather rectangles from first completed annotation
            rects = []
            for ann in ann_list or []:
                for r in (ann.get("result") or []):
                    if r.get("type") == "rectanglelabels" and r.get("value"):
                        rects.append(r)
                if rects:
                    break
            if not rects:
                continue

            is_val = (counter % 10 == 0)
            img_out_dir = img_val if is_val else img_train
            lbl_out_dir = lbl_val if is_val else lbl_train

            stem = Path(local_img).stem
            ext  = Path(local_img).suffix.lower() or ".jpg"
            dst_img = img_out_dir / f"{stem}{ext}"
            if str(Path(local_img).resolve()) != str(dst_img.resolve()):
                shutil.copy2(local_img, dst_img)

            with open(lbl_out_dir / f"{stem}.txt", "w") as f:
                for r in rects:
                    cls_id, cx, cy, w, h = _ls_rect_to_yolo(r, iw, ih, cls_id=0)
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            counter += 1

            # Pull tasks/annotations from payload (LS sends a few shapes; we handle common ones)
            tasks = data.get("tasks") or data.get("data") or []
            # If LS sends {"project":"", "tasks":[...], "annotations":[...]} we map annotations by task id
            anns_by_tid = {}
            for ann in (data.get("annotations") or []):
                tid = ann.get("task") or ann.get("task_id")
                if tid is not None:
                    anns_by_tid.setdefault(tid, []).append(ann)

            # Helper to iterate (task, annotation_list)
            def _iter_items():
                if tasks and anns_by_tid:
                    for t in tasks:
                        tid = t.get("id") or t.get("task_id")
                        yield t, anns_by_tid.get(tid, [])
                elif tasks:
                    # Some LS versions include annotations nested in each task
                    for t in tasks:
                        yield t, t.get("annotations", [])
                else:
                    return {}

            # Build a tiny train/val split (e.g., 90/10)
            counter = 0
            for task, ann_list in _iter_items():
                # Resolve image path
                img_url = task.get("data", {}).get("image") or task.get("data", {}).get("image_url")
                if not img_url:
                    continue
                local_img = get_local_path(img_url)
                if not os.path.isfile(local_img):
                    continue

                # Open to get intrinsic size
                with Image.open(local_img) as im:
                    iw, ih = im.size

                # Gather rectangles from the first completed annotation per task
                rects = []
                for ann in ann_list:
                    if ann.get("result"):
                        for r in ann["result"]:
                            if r.get("type") == "rectanglelabels" and r.get("value"):
                                rects.append(r)
                        if rects:
                            break
                if not rects:
                    continue

                # Decide train or val
                is_val = (counter % 10 == 0)  # every 10th sample to val
                img_out_dir = img_val if is_val else img_train
                lbl_out_dir = lbl_val if is_val else lbl_train

                # Copy image
                stem = Path(local_img).stem
                ext  = Path(local_img).suffix.lower() or ".jpg"
                dst_img = img_out_dir / f"{stem}{ext}"
                if str(Path(local_img).resolve()) != str(dst_img.resolve()):
                    shutil.copy2(local_img, dst_img)

                # Write YOLO label txt
                with open(lbl_out_dir / f"{stem}.txt", "w") as f:
                    for r in rects:
                        cls_id, cx, cy, w, h = _ls_rect_to_yolo(r, iw, ih, cls_id=0)
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                counter += 1

            if counter == 0:
                print("No annotated tasks found; nothing to train on.")
                return {"ok": False, "message": "No annotated rectangles found; nothing to train on."}

            # data.yaml for Ultralytics
            data_yaml = workdir / "data.yaml"
            data_yaml.write_text(
                f"path: {workdir}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"names:\n"
                f"  0: logo\n"
            )

            # Training knobs (lightweight fine-tune)
            epochs = int(os.getenv("YOLO_EPOCHS", "5"))           # keep small for on-save updates
            imgsz  = int(os.getenv("YOLO_IMGSZ", str(self.imgsz)))
            device = os.getenv("YOLO_DEVICE", "cpu")              # "cpu" or "0" for GPU 0

            # Fine-tune current model
            # (Ultralytics saves to runs/detect/train*/weights/best.pt)
            results = self.model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                batch=int(os.getenv("YOLO_BATCH", "8")),
                patience=3,
                verbose=False
            )

            # Find the newest best.pt
            run_dirs = sorted(Path("runs/detect").glob("train*"), key=os.path.getmtime)
            best_ckpt = None
            for rd in reversed(run_dirs):
                cand = rd / "weights" / "best.pt"
                if cand.exists() and cand.stat().st_size > 1024:
                    best_ckpt = cand
                    break


            if not best_ckpt:
                print("WARNING: No best.pt found in runs/detect/train*")
                return {"ok": False, "message": "Training finished but best.pt was not found."}

            # Replace serving weights
            out_dir = Path(os.getenv("YOLO_SERVE_WEIGHTS_DIR", "weights"))
            out_dir.mkdir(parents=True, exist_ok=True)
            serve_path = out_dir / "best.pt"
            shutil.copy2(best_ckpt, serve_path)

            # Reload model in memory so subsequent /predict uses the new weights
            self.model = YOLO(str(serve_path))

            # Optional: cleanup small temp datasets to avoid disk growth
            if os.getenv("YOLO_KEEP_TRAIN_ARTIFACTS", "0") != "1":
                shutil.rmtree(workdir, ignore_errors=True)

            print("Training completed successfully.")
            return {
                "ok": True,
                "trained_on_images": counter,
                "epochs": epochs,
                "weights": str(serve_path)
            }