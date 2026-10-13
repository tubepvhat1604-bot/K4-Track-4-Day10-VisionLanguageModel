"""Bước 4: dựng ví dụ huấn luyện bằng chat template, mask loss, collate."""
from pathlib import Path

import torch
from PIL import Image

from .common import objects_to_answer, prompt_for

IGNORE = -100


def open_image(entry, data_dir="data"):
    return Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")


def user_message(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def encode_example(processor, image, prompt, answer):
    user = user_message(image, prompt)
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        # Template chèn thêm gì đó vào lượt assistant: tự ghép chuỗi để prompt chắc chắn là tiền tố.
        text = processor.apply_chat_template([user], add_generation_prompt=True, tokenize=False)
        full_enc = processor(text=[text + answer + "<|im_end|>\n"], images=[image], return_tensors="pt")
        if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
            raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: kiểm tra chat template")
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = IGNORE
    out = {k: v for k, v in full_enc.items()}
    out["labels"] = labels
    return out


def encode_entry(processor, entry, coord="norm1000", data_dir="data"):
    image = open_image(entry, data_dir)
    answer = objects_to_answer(entry["objects"], coord, entry["width"], entry["height"])
    return encode_example(processor, image, prompt_for(coord, entry["width"], entry["height"]), answer)


def collate(batch, pad_token_id):
    """Pad bên phải cho huấn luyện. pixel_values / image_grid_thw nối theo chiều 0."""
    L = max(b["input_ids"].shape[1] for b in batch)
    seq_keys = {"input_ids": pad_token_id, "attention_mask": 0, "labels": IGNORE,
                "mm_token_type_ids": 0}
    out = {}
    for k, pad in seq_keys.items():
        if k not in batch[0]:
            continue
        rows = []
        for b in batch:
            t = b[k][0]
            rows.append(torch.cat([t, torch.full((L - t.shape[0],), pad, dtype=t.dtype)]))
        out[k] = torch.stack(rows)
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], 0)
    out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], 0)
    return out


def describe_example(processor, enc):
    """Số token ảnh, số token được tính loss, và phần giải mã của labels (kiểm tra mask)."""
    ids = enc["input_ids"][0]
    lab = enc["labels"][0]
    img_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return {
        "total_tokens": int(ids.shape[0]),
        "image_tokens": int((ids == img_id).sum()),
        "loss_tokens": int((lab != IGNORE).sum()),
        "masked_tokens": int((lab == IGNORE).sum()),
        "decoded_loss_part": processor.tokenizer.decode(lab[lab != IGNORE]),
    }
