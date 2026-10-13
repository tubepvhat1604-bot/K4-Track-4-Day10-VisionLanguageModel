import json

import pytest

from lab10.reward import iou_reward

GT = [{"label": "pothole", "bbox": (100, 100, 300, 300)},
      {"label": "crack", "bbox": (500, 500, 700, 600)}]


def _out(*boxes):
    return json.dumps([{"label": l, "bbox_2d": list(b)} for l, b in boxes])


def test_perfect_prediction_gets_one():
    assert iou_reward(_out(("pothole", (100, 100, 300, 300)), ("crack", (500, 500, 700, 600))), GT) == pytest.approx(1.0)


def test_order_does_not_matter():
    assert iou_reward(_out(("crack", (500, 500, 700, 600)), ("pothole", (100, 100, 300, 300))), GT) == pytest.approx(1.0)


def test_broken_json_gets_zero():
    assert iou_reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert iou_reward("I see a pothole", GT) == 0.0


def test_fenced_json_is_accepted():
    s = "```json\n" + _out(("pothole", (100, 100, 300, 300)), ("crack", (500, 500, 700, 600))) + "\n```"
    assert iou_reward(s, GT) == pytest.approx(1.0)


def test_wrong_class_gets_only_format_reward():
    assert iou_reward(_out(("manhole", (100, 100, 300, 300)), ("manhole", (500, 500, 700, 600))), GT) == pytest.approx(0.2)


def test_low_iou_is_not_matched():
    assert iou_reward(_out(("pothole", (250, 250, 450, 450))), GT) == pytest.approx(0.2)


def test_extra_boxes_are_penalised():
    one = iou_reward(_out(("pothole", (100, 100, 300, 300))), GT)
    extra = iou_reward(_out(("pothole", (100, 100, 300, 300)),
                            ("pothole", (800, 800, 900, 900)),
                            ("crack", (0, 0, 50, 50)),
                            ("manhole", (600, 0, 700, 100))), GT)
    assert one == pytest.approx(0.2 + 0.8 * 1 / 2)
    assert extra == pytest.approx(0.2 + 0.8 * 1 / 4)
    assert extra < one


def test_empty_prediction_on_nonempty_gt_is_not_rewarded_much():
    assert iou_reward("[]", GT) == pytest.approx(0.2)


def test_empty_gt_and_empty_prediction_gets_one():
    assert iou_reward("[]", []) == 1.0


def test_boxes_on_empty_image_are_penalised():
    assert iou_reward(_out(("crack", (0, 0, 100, 100))), []) == pytest.approx(0.2)


def test_one_huge_box_cannot_hack_the_reward():
    assert iou_reward(_out(("pothole", (0, 0, 1000, 1000))), GT) == pytest.approx(0.2)


def test_partial_iou_counts_fractionally():
    r = iou_reward(_out(("pothole", (100, 100, 300, 260)), ("crack", (500, 500, 700, 600))), GT)
    assert r == pytest.approx(0.2 + 0.8 * (0.8 + 1.0) / 2)


def test_each_gt_matched_once():
    r = iou_reward(_out(("pothole", (100, 100, 300, 300)), ("pothole", (100, 100, 300, 300))),
                   [GT[0]])
    assert r == pytest.approx(0.2 + 0.8 * 1 / 2)
