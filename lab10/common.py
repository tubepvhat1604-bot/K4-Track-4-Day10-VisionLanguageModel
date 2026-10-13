"""Bước 2: nhãn YOLO -> JSON (thang 0–1000 hoặc pixel), prompt, đọc split."""
import json
import random
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

PROMPT = (
    "Detect all road damage in this dashcam image. Classes: pothole, crack, manhole.\n"
    "Return ONLY a JSON list, no other text, in the form "
    '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}].\n'
    "Coordinates are integers on a 0-1000 scale relative to the image width and height "
    "(0 = left/top edge, 1000 = right/bottom edge). Return [] if there is no damage."
)

PROMPT_PIXEL = (
    "Detect all road damage in this dashcam image. Classes: pothole, crack, manhole.\n"
    "Return ONLY a JSON list, no other text, in the form "
    '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}].\n'
    "Coordinates are integer pixel positions in the {w}x{h} image "
    "(x from 0 to {w}, y from 0 to {h}). Return [] if there is no damage."
)


def prompt_for(coord, width=640, height=360):
    if coord == "norm1000":
        return PROMPT
    return PROMPT_PIXEL.replace("{w}", str(width)).replace("{h}", str(height))


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (xc, yc, w, h chuẩn hoá [0,1]) -> {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    box = [(xc - w / 2) * sx, (yc - h / 2) * sy, (xc + w / 2) * sx, (yc + h / 2) * sy]
    lim = [sx, sy, sx, sy]
    box = [min(max(int(round(v)), 0), m) for v, m in zip(box, lim)]
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixels(bbox, coord="norm1000", width=640, height=360):
    """Đổi bbox_2d về pixel để vẽ."""
    if coord == "pixel":
        return [float(v) for v in bbox]
    x1, y1, x2, y2 = bbox
    return [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]


def load_split(name, splits_dir="data/splits", limit=None, fraction=None, seed=42):
    rows = read_jsonl(Path(splits_dir) / f"{name}.jsonl")
    if fraction is not None and fraction < 1.0:
        rng = random.Random(seed)
        rows = rng.sample(rows, max(1, int(round(len(rows) * fraction))))
    if limit is not None:
        rows = rows[:limit]
    return rows


def split_stats(splits_dir="data/splits"):
    out = {}
    for s in ("train", "val", "test"):
        rows = read_jsonl(Path(splits_dir) / f"{s}.jsonl")
        boxes = {c: 0 for c in CLASSES}
        areas = {c: [] for c in CLASSES}
        for r in rows:
            for o in r["objects"]:
                boxes[o["label"]] += 1
                areas[o["label"]].append(o["yolo"][2] * o["yolo"][3])
        med = {c: (sorted(a)[len(a) // 2] if a else 0.0) for c, a in areas.items()}
        out[s] = {"images": len(rows), "boxes": boxes,
                  "sessions": len({r["session"] for r in rows}),
                  "empty_images": sum(1 for r in rows if not r["objects"]),
                  "median_box_area_frac": med}
    sess = {s: {r["session"] for r in read_jsonl(Path(splits_dir) / f"{s}.jsonl")}
            for s in ("train", "val", "test")}
    out["session_overlap"] = {
        "train&val": len(sess["train"] & sess["val"]),
        "train&test": len(sess["train"] & sess["test"]),
        "val&test": len(sess["val"] & sess["test"]),
    }
    return out
