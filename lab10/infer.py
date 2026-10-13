"""Bước 3 + 7: suy luận theo batch, ghi {"image_id","raw"} mỗi dòng."""
import contextlib
import json
import time
from pathlib import Path

import torch

from .common import prompt_for
from .data import open_image, user_message


def _autocast(device, dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model, processor, entries, out_path, coord="norm1000", batch_size=4,
            max_new_tokens=256, data_dir="data", amp_dtype=torch.float16, log_every=10):
    """Suy luận toàn bộ entries. Nếu file đã có một phần (session bị ngắt), chạy tiếp phần còn lại."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["image_id"])
    todo = [e for e in entries if e["image_id"] not in done]
    model.eval()
    processor.tokenizer.padding_side = "left"
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for bi in range(0, len(todo), batch_size):
            batch = todo[bi:bi + batch_size]
            convs = [[user_message(open_image(e, data_dir),
                                   prompt_for(coord, e["width"], e["height"]))] for e in batch]
            inputs = processor.apply_chat_template(
                convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", processor_kwargs={"padding": True}).to(model.device)
            with _autocast(model.device, amp_dtype):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen = out[:, inputs["input_ids"].shape[1]:]
            texts = processor.batch_decode(gen, skip_special_tokens=True)
            for e, t in zip(batch, texts):
                f.write(json.dumps({"image_id": e["image_id"], "raw": t}, ensure_ascii=False) + "\n")
            f.flush()
            nb = bi // batch_size + 1
            if log_every and nb % log_every == 0:
                print(f"  {min(bi + batch_size, len(todo))}/{len(todo)} ảnh, "
                      f"{(time.time() - t0) / 60:.1f} phút", flush=True)
    return out_path
