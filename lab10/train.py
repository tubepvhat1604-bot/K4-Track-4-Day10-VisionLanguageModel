"""Bước 6: vòng huấn luyện fp16 + GradScaler + tích luỹ gradient."""
import contextlib
import json
import math
import random
import time
from pathlib import Path

import torch

from .data import collate, encode_entry
from .modeling import projector_state


def save_checkpoint(model, out_dir, train_projector):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)                    # chỉ lưu adapter LoRA
    if train_projector:                               # projector không nằm trong adapter
        torch.save(projector_state(model), out_dir / "projector.pt")


def train(model, processor, entries, out_dir, train_projector=False, coord="norm1000",
          lr=2e-4, batch_size=2, grad_accum=4, max_steps=300, warmup_frac=0.05, seed=42,
          save_every=100, data_dir="data", amp_dtype=torch.float16, log_every=10, extra_info=None):
    from transformers import get_cosine_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    device = model.device
    use_amp = device.type == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(math.ceil(warmup_frac * max_steps)), max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    def batches():
        rng = random.Random(seed)
        while True:
            order = entries[:]
            rng.shuffle(order)
            for i in range(0, len(order) - batch_size + 1, batch_size):
                yield order[i:i + batch_size]

    it = batches()
    model.train()
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    log = []
    t0 = time.time()
    for step in range(1, max_steps + 1):
        total = 0.0
        for _ in range(grad_accum):
            batch = collate([encode_entry(processor, e, coord, data_dir) for e in next(it)], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
            with ctx:
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn ở bước {step}: {loss.item()} "
                                         "(tràn số fp16? thử giảm lr hoặc batch_size=1)")
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        log.append({"step": step, "loss": total, "lr": sched.get_last_lr()[0],
                    "grad_norm": float(gn), "minutes": (time.time() - t0) / 60})
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  bước {step}/{max_steps}  loss={total:.4f}  "
                  f"{(time.time() - t0) / 60:.1f} phút", flush=True)
        if save_every and step % save_every == 0 and step != max_steps:
            save_checkpoint(model, Path(out_dir) / "latest", train_projector)
    minutes = (time.time() - t0) / 60
    save_checkpoint(model, out_dir, train_projector)
    run = {
        "trainable_params": int(sum(p.numel() for p in params)),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if use_amp else 0.0,
        "train_minutes": round(minutes, 1),
        "max_steps": max_steps, "batch_size": batch_size, "grad_accum": grad_accum, "lr": lr,
        "n_train_images": len(entries), "first_loss": log[0]["loss"], "last_loss": log[-1]["loss"],
        **(extra_info or {}),
    }
    (Path(out_dir) / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (Path(out_dir) / "loss_log.json").write_text(json.dumps(log), encoding="utf-8")
    return run, log
