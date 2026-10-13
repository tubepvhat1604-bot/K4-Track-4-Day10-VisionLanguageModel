"""Bước 9 (E5, tuỳ chọn): GRPO với reward IoU, khởi tạo từ adapter SFT tốt nhất."""
import json
import time
from pathlib import Path

import torch

from scripts.score import _gt_boxes

from .common import PROMPT, load_split
from .reward import iou_reward


def _text(completion):
    if isinstance(completion, list):          # dạng hội thoại: [{"role": "assistant", "content": ...}]
        c = completion[-1]["content"] if completion else ""
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c
    return completion


def reward_fn(completions, gt, **kwargs):
    return [iou_reward(_text(c), json.loads(g)) for c, g in zip(completions, gt)]


def build_dataset(splits_dir, data_dir, n_images, seed=42):
    from datasets import Dataset, Image
    rows = load_split("train", splits_dir, seed=seed)
    import random
    rows = random.Random(seed).sample(rows, min(n_images, len(rows)))
    ds = Dataset.from_dict({
        "image": [str(Path(data_dir) / r["image_path"]) for r in rows],
        "prompt": [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
                   for _ in rows],
        "gt": [json.dumps([{"label": b["label"], "bbox": list(b["bbox"])} for b in _gt_boxes(r)])
               for r in rows],
    })
    return ds.cast_column("image", Image())


def run_grpo(model, processor, out_dir, splits_dir, data_dir, max_steps=60, num_generations=4,
             max_completion_length=128, learning_rate=5e-6, beta=0.0, seed=42):
    from trl import GRPOConfig, GRPOTrainer

    cuda = torch.cuda.is_available()
    per_device = num_generations
    tok = processor.tokenizer
    # Không cho sinh token ảnh trong câu trả lời (nếu có, forward tính loss sẽ lệch số token ảnh).
    banned = [[tok.convert_tokens_to_ids(t)] for t in
              ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")]
    args = GRPOConfig(
        output_dir=str(Path(out_dir) / "trainer"), max_steps=max_steps, learning_rate=learning_rate,
        beta=beta, num_generations=num_generations, max_completion_length=max_completion_length,
        per_device_train_batch_size=per_device, gradient_accumulation_steps=1,
        use_vllm=False, fp16=cuda, bf16=False, logging_steps=1, save_strategy="no", report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, generation_kwargs={"bad_words_ids": banned}, seed=seed, dataloader_num_workers=0, log_completions=False,
    )
    n_images = max_steps * per_device // num_generations + 8
    ds = build_dataset(splits_dir, data_dir, n_images, seed)
    processor.tokenizer.padding_side = "left"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=args, train_dataset=ds,
                          processing_class=processor)
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    hist = trainer.state.log_history
    run = {
        "trainable_params": int(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0,
        "train_minutes": round(minutes, 1), "max_steps": max_steps,
        "num_generations": num_generations, "max_completion_length": max_completion_length,
        "learning_rate": learning_rate, "beta": beta,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out_dir / "loss_log.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    return run, hist
