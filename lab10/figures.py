"""Ảnh minh hoạ: box đúng (xanh lá) và box dự đoán (đỏ), chọn ca đúng và ca sai."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, parse_output

from .common import box_to_pixels

GT_COLOR = (0, 200, 0)
PRED_COLOR = (230, 30, 30)


def _per_image(entries, preds):
    rows = []
    for e in entries:
        raw = preds.get(e["image_id"])
        parsed = parse_output(raw) if raw is not None else None
        per, _ = _match(_gt_boxes(e), parsed or [])
        tp = sum(v[0] for v in per.values())
        fp = sum(v[1] for v in per.values())
        fn = sum(v[2] for v in per.values())
        rows.append({"entry": e, "raw": raw, "parsed": parsed, "tp": tp, "fp": fp, "fn": fn})
    return rows


def draw(entry, parsed, data_dir, out_path, title):
    im = Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")
    W, H = im.size
    canvas = Image.new("RGB", (W, H + 22), "white")
    canvas.paste(im, (0, 22))
    d = ImageDraw.Draw(canvas)
    d.text((4, 4), title[:110], fill=(0, 0, 0))
    for g in _gt_boxes(entry):
        x1, y1, x2, y2 = box_to_pixels(g["bbox"], "norm1000", W, H)
        d.rectangle([x1, y1 + 22, x2, y2 + 22], outline=GT_COLOR, width=3)
        d.text((x1 + 2, y2 + 22 - 12), "GT " + g["label"], fill=GT_COLOR)
    for p in parsed or []:
        x1, y1, x2, y2 = box_to_pixels(p["bbox"], "norm1000", W, H)
        d.rectangle([x1, y1 + 22, x2, y2 + 22], outline=PRED_COLOR, width=2)
        d.text((x1 + 2, y1 + 22 + 2), p["label"], fill=PRED_COLOR)
    canvas.save(out_path)


def make_figures(out, splits_dir, data_dir, split, best):
    out = Path(out)
    entries = read_jsonl(Path(splits_dir) / f"{split}.jsonl")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    captions = []

    def load(exp):
        p = out / "preds" / f"{exp}.jsonl"
        return {r["image_id"]: r["raw"] for r in read_jsonl(p)} if p.exists() else None

    preds = load(best)
    if preds:
        rows = [r for r in _per_image(entries, preds) if r["raw"] is not None]
        good = sorted([r for r in rows if r["tp"] > 0],
                      key=lambda r: (r["fp"] + r["fn"], -r["tp"]))[:3]
        used = {r["entry"]["image_id"] for r in good}
        bad_fp = sorted([r for r in rows if r["entry"]["image_id"] not in used],
                        key=lambda r: -r["fp"])[:2]
        used |= {r["entry"]["image_id"] for r in bad_fp}
        bad_fn = sorted([r for r in rows if r["entry"]["image_id"] not in used],
                        key=lambda r: -r["fn"])[:2]
        for kind, group in (("dung", good), ("sai_FP", bad_fp), ("sai_FN", bad_fn)):
            for r in group:
                e = r["entry"]
                name = f"{best}_{kind}_{e['image_id']}.png"
                title = f"{best.upper()} {kind}: TP={r['tp']} FP={r['fp']} FN={r['fn']}"
                draw(e, r["parsed"], data_dir, fig_dir / name, title)
                captions.append({"file": f"figures/{name}", "exp": best.upper(), "kind": kind,
                                 "image_id": e["image_id"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                                 "gt": [o["label"] for o in e["objects"]], "raw": r["raw"]})
    e0 = load("e0")
    if e0:
        rows = [r for r in _per_image(entries, e0) if r["raw"] is not None]
        broken = [r for r in rows if r["parsed"] is None][:1]
        for r in broken + [r for r in rows if r["parsed"]][:1]:
            e = r["entry"]
            kind = "json_hong" if r["parsed"] is None else "zero_shot"
            name = f"e0_{kind}_{e['image_id']}.png"
            draw(e, r["parsed"], data_dir, fig_dir / name,
                 f"E0 {kind}: TP={r['tp']} FP={r['fp']} FN={r['fn']}")
            captions.append({"file": f"figures/{name}", "exp": "E0", "kind": kind,
                             "image_id": e["image_id"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                             "gt": [o["label"] for o in e["objects"]], "raw": r["raw"]})
    (fig_dir / "captions.json").write_text(json.dumps(captions, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return captions
