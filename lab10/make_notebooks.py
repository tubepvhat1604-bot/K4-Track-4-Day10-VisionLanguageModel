"""Sinh lab_day10.ipynb (bài làm, từng bước) và kaggle_lab10.ipynb (chạy hết trên Kaggle).

Chạy: python -m lab10.make_notebooks
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def nb(cells, kaggle=False):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}, "accelerator": "GPU"}
    if kaggle:
        meta["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                          "isGpuEnabled": True, "language": "python"}
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


SETUP_ENV = r'''
# Chạy được trên cả Kaggle lẫn Colab. Trên Colab: Runtime → Change runtime type → T4 GPU.
import os, subprocess, sys
REPO_URL = "https://github.com/tubepvhat1604-bot/K4-Track-4-Day10-VisionLanguageModel.git"
BRANCH = "main"
if not os.path.exists("scripts/score.py"):                 # chưa ở trong repo → clone về
    if not os.path.exists("lab"):
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, "lab"], check=True)
    os.chdir("lab")
print("thư mục làm việc:", os.getcwd())
!nvidia-smi -L
!pip -q install -r requirements-colab.txt
'''

FIND_DATA = r'''
# Tìm dataset: Kaggle (kagglehub, không cần khoá) → /kaggle/input → Kaggle CLI (Colab Secrets).
import glob, os, subprocess

def find_dataset():
    try:
        import kagglehub
        p = kagglehub.dataset_download("lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes")
        if glob.glob(os.path.join(p, "**", "labels-YOLO"), recursive=True):
            return p
    except Exception as e:
        print("kagglehub không dùng được:", e)
    hits = glob.glob("/kaggle/input/**/labels-YOLO", recursive=True)
    return os.path.dirname(hits[0]) if hits else None

if not os.path.exists("data/splits/meta.json"):
    src = find_dataset()
    if src:
        subprocess.run(["python", "scripts/prepare_data.py", "--source", src], check=True)
    else:   # Colab: KAGGLE_USERNAME / KAGGLE_KEY trong Colab Secrets, KHÔNG viết khoá vào notebook
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        subprocess.run(["python", "scripts/prepare_data.py", "--kaggle"], check=True)
!cat data/splits/meta.json | head -5
'''

LAB_CELLS = [
    md("# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường\n"
       "Bài làm. Mã nằm trong thư mục `lab10/` (notebook này gọi lại các hàm đó theo từng bước của `GUIDE.md`).\n"
       "Muốn chạy hết một lần trên Kaggle thì dùng `kaggle_lab10.ipynb` (xem `KAGGLE.md`)."),
    md("## 0. Môi trường (Colab T4 hoặc Kaggle T4)"),
    code(SETUP_ENV),
    code("!python -m pytest -q"),
    md("## 1. Dữ liệu: tải và chia split"),
    code(FIND_DATA),
    code(r'''
import json
from lab10.common import split_stats
print(json.dumps(split_stats(), indent=1))   # số ảnh/box từng split, session_overlap phải toàn 0
'''),
    md("## 2. Nhãn: YOLO → JSON thang 0–1000"),
    code(r'''
from PIL import Image, ImageDraw
from lab10.common import load_split, objects_to_answer, box_to_pixels, yolo_to_box
train = load_split("train")
e = next(r for r in train if len(r["objects"]) >= 2)
ans = objects_to_answer(e["objects"])
print(ans)
# Kiểm tra: JSON 0–1000 → pixel → vẽ (đỏ) phải trùng box YOLO gốc (xanh)
im = Image.open("data/" + e["image_path"]).convert("RGB"); d = ImageDraw.Draw(im)
for o in e["objects"]:
    xc, yc, w, h = o["yolo"]
    d.rectangle([(xc - w/2) * 640, (yc - h/2) * 360, (xc + w/2) * 640, (yc + h/2) * 360], outline="lime", width=5)
for b in json.loads(ans):
    d.rectangle(box_to_pixels(b["bbox_2d"]), outline="red", width=2)
im
'''),
    md("## 3. E0: zero-shot"),
    code("!python -m lab10.run stats e0     # nạp model 4-bit, suy luận 311 ảnh test, chấm bằng scripts/score.py"),
    code(r'''
from scripts.labcommon import read_jsonl
for r in read_jsonl("outputs/preds/e0.jsonl")[:5]:
    print(repr(r["raw"][:300]), "\n")
'''),
    md("## 4. Dựng dữ liệu huấn luyện: chat template và mask loss"),
    code(r'''
print(json.dumps(json.load(open("outputs/stats.json"))["example"], indent=1, ensure_ascii=False))
# image_tokens phải là 220; decoded_loss_part phải đúng là JSON đáp án + <|im_end|>
'''),
    md("## 5. Setup LoRA / đóng băng cho E1, E2, E3"),
    code(r'''
print(json.dumps(json.load(open("outputs/stats.json"))["param_counts"], indent=1))
# Số module LoRA (252 / 252 / 348) và số tham số huấn luyện được in ra khi chạy bước 6.
'''),
    md("## 6. Vòng huấn luyện"),
    code("!python -m lab10.run e1 e2 e3     # mỗi setup: train 300 bước → lưu adapter (+ projector) → suy luận → chấm"),
    md("## 7. Suy luận và chấm E1–E3"),
    code(r'''
for e in ["e0", "e1", "e2", "e3"]:
    d = json.load(open(f"outputs/evidence/{e}.json"))
    print(e, round(d["micro"]["f1"], 4), [round(x, 4) for x in d["f1_ci95"]], d["parse_valid_rate"], d.get("meta"))
'''),
    md("## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)"),
    code("!python -m lab10.run e4 extras e5"),
    code(r'''
!mkdir -p report/evidence && cp outputs/evidence/*.json report/evidence/
!python scripts/check_submission.py
'''),
]

KAGGLE_CELLS = [
    md(r'''
# Lab Day 10 trên Kaggle: chạy toàn bộ bằng một nút

**Trước khi chạy** (bảng bên phải → *Session options*):
1. **Accelerator** = `GPU T4 x2`.
2. **Internet** = On (cần xác minh số điện thoại cho tài khoản Kaggle).

**Chạy thử nhanh (≈30 phút):** để `SMOKE_TEST = True`, bấm *Run All*, chờ dòng `XONG, không có lỗi`.
**Chạy thật (≈5–7 giờ):** đổi thành `SMOKE_TEST = False`, bấm *Save Version* → *Save & Run All (Commit)*.
Xong thì vào tab *Output* của phiên bản đó, tải file **`lab10_outputs.zip`** và gửi lại.
'''),
    code(r'''
SMOKE_TEST = True          # True = chạy thử nhanh; False = chạy thật để nộp bài
RUN_E5 = True              # bonus GRPO (+10 điểm); đặt False nếu muốn tiết kiệm giờ GPU
REPO_URL = "https://github.com/tubepvhat1604-bot/K4-Track-4-Day10-VisionLanguageModel.git"
BRANCH = "claude/project-thread-llnd8q"
'''),
    code(r'''
import os, subprocess
os.chdir("/kaggle/working")
if not os.path.exists("lab"):
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, "lab"], check=True)
os.chdir("/kaggle/working/lab")
subprocess.run(["git", "pull", "-q", "origin", BRANCH], check=True)   # luôn lấy mã mới nhất
subprocess.run(["git", "log", "--oneline", "-1"])
!nvidia-smi -L
!pip -q install -r requirements-colab.txt
'''),
    code(FIND_DATA),
    code(r'''
# Nếu phiên trước bị ngắt giữa chừng: thêm Output của phiên trước làm Input, ô này chép lại để chạy tiếp.
import glob, shutil
for prev in glob.glob("/kaggle/input/**/lab/outputs/evidence", recursive=True):
    base = os.path.dirname(os.path.dirname(prev))
    for d in ("outputs", "adapters"):
        if os.path.exists(os.path.join(base, d)):
            shutil.copytree(os.path.join(base, d), d, dirs_exist_ok=True)
    print("đã chép kết quả cũ từ", base)
'''),
    code(r'''
steps = ["all"] if RUN_E5 else ["stats", "e0", "e1", "e2", "e3", "e4", "extras"]
cmd = " ".join(steps) + (" --smoke" if SMOKE_TEST else "")
!python -m lab10.run {cmd} 2>&1 | grep --line-buffered -v "Loading weights"
'''),
    code(r'''
# Gom kết quả để gửi lại (không chứa trọng số, không chứa ảnh dataset)
import shutil, json
out = "outputs_smoke" if SMOKE_TEST else "outputs"
shutil.copy("data/splits/meta.json", f"{out}/meta.json")
shutil.make_archive("/kaggle/working/lab10_outputs" + ("_smoke" if SMOKE_TEST else ""), "zip", out)
for f in sorted(glob.glob(f"{out}/evidence/*.json")):
    d = json.load(open(f))
    print(os.path.basename(f), "F1=%.4f" % d["micro"]["f1"], "CI95=", [round(x, 4) for x in d["f1_ci95"]],
          "parse_valid=%.3f" % d["parse_valid_rate"], d.get("meta", {}))
print("\nFile để tải về:", "/kaggle/working/lab10_outputs" + ("_smoke" if SMOKE_TEST else "") + ".zip")
'''),
]


def main():
    (ROOT / "lab_day10.ipynb").write_text(json.dumps(nb(LAB_CELLS), ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8")
    (ROOT / "kaggle_lab10.ipynb").write_text(json.dumps(nb(KAGGLE_CELLS, True), ensure_ascii=False, indent=1)
                                             + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
