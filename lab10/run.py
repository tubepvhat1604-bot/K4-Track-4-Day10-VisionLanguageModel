"""Chạy cả lab theo từng thí nghiệm. Ví dụ:

    python -m lab10.run stats
    python -m lab10.run e0
    python -m lab10.run e1 e2 e3 e4 extras
    python -m lab10.run all            # = stats e0 e1 e2 e3 e4 extras e5
    python -m lab10.run all --smoke    # chạy thử thật nhanh để bắt lỗi trước

Kết quả nằm trong --out (mặc định outputs/): evidence/*.json (do scripts/score.py sinh),
preds/*.jsonl, runs/<exp>/run.json + loss_log.json, figures/*.png, stats.json.
Adapter + projector nằm trong --adapters (không đưa vào git).
Thí nghiệm nào đã có evidence thì bỏ qua, nên chạy lại sau khi bị ngắt là an toàn.
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

from .common import MODEL_ID, load_split, split_stats

ROOT = Path(__file__).resolve().parents[1]
SETUPS = {}
for _n in (1, 2, 3):
    _c = yaml.safe_load((ROOT / "configs" / f"e{_n}.yaml").read_text(encoding="utf-8"))
    SETUPS[f"e{_n}"] = {"targets": _c["lora"]["targets"], "train_projector": _c["train_projector"],
                        "r": _c["lora"]["r"], "alpha": _c["lora"]["alpha"],
                        "dropout": _c["lora"]["dropout"], "lr": _c["lr"],
                        "batch_size": _c["batch_size"], "grad_accum": _c["grad_accum"],
                        "max_steps": _c["max_steps"], "seed": _c["seed"]}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Lab:
    def __init__(self, a):
        self.a = a
        suffix = "_smoke" if a.smoke else ""
        self.out = Path(a.out + suffix)
        self.adapters = Path(a.adapters + suffix)
        for d in ("evidence", "preds", "runs", "figures", "extra"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.split = "val" if a.smoke else "test"
        self._processor = None

    # ---------- tiện ích ----------
    @property
    def processor(self):
        if self._processor is None:
            from .modeling import load_processor
            self._processor = load_processor(self.a.model_id)
        return self._processor

    def eval_entries(self):
        return load_split(self.split, self.a.splits_dir, limit=8 if self.a.smoke else None)

    def score(self, pred, name, out_json, meta=None, coord="norm1000"):
        cmd = [sys.executable, str(ROOT / "scripts" / "score.py"), "--pred", str(pred),
               "--name", name, "--out", str(out_json), "--split", self.split,
               "--splits-dir", self.a.splits_dir, "--coord", coord]
        for k, v in (meta or {}).items():
            cmd += ["--meta", f"{k}={v}"]
        subprocess.run(cmd, check=True)
        d = json.loads(Path(out_json).read_text(encoding="utf-8"))
        log(f"{name}: F1={d['micro']['f1']:.4f} CI95={[round(x, 4) for x in d['f1_ci95']]} "
            f"parse_valid_rate={d['parse_valid_rate']:.3f}")
        return d

    def evidence(self, name):
        return self.out / "evidence" / f"{name}.json"

    def load_base(self):
        from .modeling import load_model
        return load_model(self.a.model_id, quantize=not self.a.no_quant,
                          dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                          device_map=None if torch.cuda.is_available() else "cpu")

    # ---------- các bước ----------
    def stats(self):
        from transformers import AutoConfig

        from .data import describe_example, encode_entry
        from .modeling import block_param_counts
        s = {"split_stats": split_stats(self.a.splits_dir)}
        meta = json.loads((Path(self.a.splits_dir) / "meta.json").read_text(encoding="utf-8"))
        s["dataset_version"] = meta["dataset_version"]
        s["param_counts"] = block_param_counts(AutoConfig.from_pretrained(self.a.model_id))
        ex = load_split("train", self.a.splits_dir, limit=1)[0]
        s["example"] = {"image_id": ex["image_id"],
                        **describe_example(self.processor, encode_entry(self.processor, ex, data_dir=self.a.data_dir))}
        (self.out / "stats.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"stats: {json.dumps(s, ensure_ascii=False)[:1500]}")

    def e0(self):
        from .infer import predict
        if self.evidence("e0").exists():
            return log("E0 đã có evidence, bỏ qua")
        model = self.load_base()
        pred = self.out / "preds" / "e0.jsonl"
        t = time.time()
        predict(model, self.processor, self.eval_entries(), pred, batch_size=self.a.batch_infer,
                max_new_tokens=256, data_dir=self.a.data_dir)
        log(f"E0 suy luận xong trong {(time.time() - t) / 60:.1f} phút")
        del model
        free()
        self.score(pred, "E0", self.evidence("e0"))

    def train_and_eval(self, exp, setup, name, ev_name, fraction=None, coord="norm1000"):
        from .infer import predict
        from .modeling import apply_setup
        from .train import train
        if self.evidence(ev_name).exists():
            return log(f"{name} đã có evidence, bỏ qua")
        adir = self.adapters / exp
        run_dir = self.out / "runs" / exp
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (adir / "run.json").exists():
            entries = load_split("train", self.a.splits_dir, fraction=fraction, seed=setup["seed"])
            model = self.load_base()
            model, info = apply_setup(model, setup["targets"], setup["train_projector"],
                                      setup["r"], setup["alpha"], setup["dropout"])
            log(f"{name}: {info}")
            steps = 5 if self.a.smoke else setup["max_steps"]
            run, _ = train(model, self.processor, entries, adir, setup["train_projector"], coord,
                           lr=setup["lr"], batch_size=setup["batch_size"],
                           grad_accum=setup["grad_accum"], max_steps=steps, seed=setup["seed"],
                           data_dir=self.a.data_dir, extra_info={**info, "setup": setup,
                                                                 "train_fraction": fraction or 1.0,
                                                                 "coord": coord})
            del model
            free()
        for f in ("run.json", "loss_log.json"):
            shutil.copy(adir / f, run_dir / f)
        run = json.loads((adir / "run.json").read_text(encoding="utf-8"))
        log(f"{name}: train xong {run['train_minutes']} phút, VRAM đỉnh {run['peak_vram_gb']} GB, "
            f"loss {run['first_loss']:.3f} -> {run['last_loss']:.3f}")
        model = self.load_inference_model(adir, setup["train_projector"])
        pred = self.out / "preds" / f"{exp}.jsonl"
        predict(model, self.processor, self.eval_entries(), pred, coord=coord,
                batch_size=self.a.batch_infer, max_new_tokens=256, data_dir=self.a.data_dir)
        del model
        free()
        meta = {"trainable_params": run["trainable_params"],
                "peak_vram_gb": run["peak_vram_gb"] if run["peak_vram_gb"] > 0 else 0.01,
                "train_minutes": run["train_minutes"] if run["train_minutes"] > 0 else 0.1}
        return self.score(pred, name, self.evidence(ev_name), meta, coord)

    def load_inference_model(self, adir, train_projector):
        from peft import PeftModel

        from .modeling import load_projector_state, patch_old_torchao
        patch_old_torchao()
        model = self.load_base()
        model = PeftModel.from_pretrained(model, str(adir))
        if train_projector:
            state = torch.load(adir / "projector.pt", map_location="cpu")
            n = load_projector_state(model, state, dtype=torch.float16
                                     if torch.cuda.is_available() else torch.float32)
            log(f"đã nạp lại {n} tensor projector")
        model.eval()
        return model

    def e123(self, exp):
        return self.train_and_eval(exp, SETUPS[exp], exp.upper(), exp)

    def best_setup(self):
        best, f1 = "e1", -1.0
        for e in ("e1", "e2", "e3"):
            p = self.evidence(e)
            if p.exists():
                v = json.loads(p.read_text(encoding="utf-8"))["micro"]["f1"]
                if v > f1:
                    best, f1 = e, v
        return best

    def e4(self):
        var = self.a.e4_var
        base = self.best_setup()
        log(f"E4: biến={var}, setup gốc={base.upper()} (F1 cao nhất trong E1–E3)")
        setup = dict(SETUPS[base])
        if var == "data_size":
            runs = [("0.1", {"fraction": 0.1}), ("1.0", None)]
        elif var == "rank":
            runs = [("4", {"r": 4}), ("64", {"r": 64})]
        elif var == "coord_scale":
            runs = [("pixel", {"coord": "pixel"}), ("norm1000", None)]
        else:
            raise ValueError(var)
        for val, change in runs:
            ev = f"e4_{var}_{val}"
            if change is None:              # giá trị này chính là setup gốc: chấm lại dự đoán của nó
                if self.evidence(ev).exists() or not self.evidence(base).exists():
                    continue
                run = json.loads((self.out / "runs" / base / "run.json").read_text(encoding="utf-8"))
                meta = {"trainable_params": run["trainable_params"],
                        "peak_vram_gb": max(run["peak_vram_gb"], 0.01),
                        "train_minutes": max(run["train_minutes"], 0.1), "same_run_as": base.upper()}
                self.score(self.out / "preds" / f"{base}.jsonl", f"E4_{var}_{val}", self.evidence(ev), meta)
                continue
            s = dict(setup)
            fraction, coord = None, "norm1000"
            if "r" in change:
                s["r"], s["alpha"] = change["r"], 2 * change["r"]
            fraction = change.get("fraction")
            coord = change.get("coord", "norm1000")
            self.train_and_eval(f"e4_{var}_{val}", s, f"E4_{var}_{val}", ev, fraction, coord)

    def e5(self):
        from peft import PeftModel

        from .grpo import run_grpo
        from .infer import predict
        from .modeling import load_projector_state, patch_old_torchao
        patch_old_torchao()
        if self.evidence("e5").exists():
            return log("E5 đã có evidence, bỏ qua")
        base = self.best_setup()
        src = self.adapters / base
        adir = self.adapters / "e5"
        train_projector = SETUPS[base]["train_projector"]
        log(f"E5: GRPO khởi tạo từ adapter {base.upper()}")
        if not (adir / "run.json").exists():
            model = self.load_base()
            model = PeftModel.from_pretrained(model, str(src), is_trainable=True)
            if train_projector:
                load_projector_state(model, torch.load(src / "projector.pt", map_location="cpu"),
                                     dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                for n, p in model.named_parameters():
                    if "visual." in n and "lora_" not in n:
                        p.requires_grad = False
            model.enable_input_require_grads()
            run, _ = run_grpo(model, self.processor, adir, self.a.splits_dir, self.a.data_dir,
                              max_steps=3 if self.a.smoke else 60)
            if train_projector:
                shutil.copy(src / "projector.pt", adir / "projector.pt")
            run["init_from"] = base.upper()
            (adir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
            del model
            free()
        run_dir = self.out / "runs" / "e5"
        run_dir.mkdir(parents=True, exist_ok=True)
        for f in ("run.json", "loss_log.json"):
            shutil.copy(adir / f, run_dir / f)
        run = json.loads((adir / "run.json").read_text(encoding="utf-8"))
        model = self.load_inference_model(adir, train_projector)
        pred = self.out / "preds" / "e5.jsonl"
        predict(model, self.processor, self.eval_entries(), pred, batch_size=self.a.batch_infer,
                max_new_tokens=256, data_dir=self.a.data_dir)
        del model
        free()
        meta = {"trainable_params": run["trainable_params"],
                "peak_vram_gb": max(run["peak_vram_gb"], 0.01),
                "train_minutes": max(run["train_minutes"], 0.1), "init_from": base.upper()}
        return self.score(pred, "E5", self.evidence("e5"), meta)

    def extras(self):
        from .figures import make_figures
        # Câu 3: chấm dự đoán thang 0–1000 của E1 như thể là pixel
        for e in ("e1", self.best_setup()):
            p = self.out / "preds" / f"{e}.jsonl"
            if p.exists():
                self.score(p, f"{e.upper()}_scored_as_pixel",
                           self.out / "extra" / f"{e}_scored_as_pixel.json", coord="pixel")
        make_figures(self.out, self.a.splits_dir, self.a.data_dir, self.split, self.best_setup())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+",
                    choices=["all", "stats", "e0", "e1", "e2", "e3", "e4", "extras", "e5"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--e4-var", default="data_size", choices=["data_size", "rank", "coord_scale"])
    ap.add_argument("--batch-infer", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="5 bước train, 8 ảnh val: chỉ để bắt lỗi")
    ap.add_argument("--no-quant", action="store_true", help="(chỉ dùng để thử trên CPU)")
    a = ap.parse_args(argv)
    steps = ["stats", "e0", "e1", "e2", "e3", "e4", "extras", "e5"] if "all" in a.steps else a.steps
    lab = Lab(a)
    failed = []
    for s in steps:
        log(f"===== {s.upper()} =====")
        try:
            if s in ("e1", "e2", "e3"):
                lab.e123(s)
            else:
                getattr(lab, s)()
        except Exception:
            traceback.print_exc()
            failed.append(s)
            free()
    log("XONG" + (f", lỗi ở: {failed}" if failed else ", không có lỗi"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
