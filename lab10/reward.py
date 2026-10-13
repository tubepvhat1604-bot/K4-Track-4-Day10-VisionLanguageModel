"""Bước 9 (E5): reward IoU kiểm chứng được cho GRPO. Python thuần, test không cần GPU."""
from scripts.score import iou_xyxy, parse_output

IOU_THR = 0.5


def iou_reward(completion, gt_boxes):
    """gt_boxes: list[{"label", "bbox": (x1,y1,x2,y2)}] thang 0–1000.

    0.0 nếu JSON hỏng; 1.0 nếu cả GT và dự đoán đều rỗng; còn lại
    0.2 + 0.8 * (tổng IoU các cặp khớp) / max(n_gt, n_pred, 1).
    Ghép đôi tham lam theo IoU giảm dần, cùng lớp, chỉ nhận IoU >= 0.5.
    Box bịa (thừa) bị phạt qua mẫu số, nên trả [] hay rải nhiều box đều không có lợi.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)
