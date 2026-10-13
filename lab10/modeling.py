"""Bước 3 + 5: nạp model 4-bit, gắn LoRA, mở projector, lưu/nạp tham số projector."""
import torch

from .common import MODEL_ID

LLM_TARGETS = (r"model\.language_model\.layers\.\d+\."
               r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")
VIT_TARGETS = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")


def load_processor(model_id=MODEL_ID):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    proc.tokenizer.padding_side = "left"
    return proc


def load_model(model_id=MODEL_ID, quantize=True, dtype=torch.float16, device_map=None):
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    kw = {"dtype": dtype}
    if quantize:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            # transformers 5.x so khớp tên module từ ĐẦU tên đầy đủ ("model.visual.blocks..."),
            # nên chỉ ghi "visual" như GUIDE thì ViT vẫn bị lượng tử hoá. Thêm "model.visual".
            llm_int8_skip_modules=["visual", "model.visual", "lm_head"])
        kw["device_map"] = device_map or {"": 0}
    elif device_map is not None:
        kw["device_map"] = device_map
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kw)
    if quantize:
        check_vit_not_quantized(model)
    return model


def check_vit_not_quantized(model):
    """ViT + projector phải là Linear fp16 thường; LLM phải là 4-bit."""
    bad = [n for n, m in model.named_modules()
           if "visual" in n and "4bit" in type(m).__name__]
    if bad:
        raise RuntimeError(f"ViT/projector bị lượng tử hoá 4-bit ({len(bad)} module, ví dụ {bad[:2]})")
    if not any("4bit" in type(m).__name__ for n, m in model.named_modules() if "language_model" in n):
        raise RuntimeError("LLM không được lượng tử hoá 4-bit")


def block_param_counts(model_or_config):
    """Số tham số từng khối (đếm trên model đầy đủ, không lượng tử, dựng trên device 'meta')."""
    from transformers import Qwen3VLForConditionalGeneration
    model = model_or_config
    if not isinstance(model_or_config, torch.nn.Module):
        with torch.device("meta"):
            model = Qwen3VLForConditionalGeneration(model_or_config)
    c = {"vit_blocks_and_patch_embed": 0, "merger": 0, "deepstack_merger_list": 0,
         "llm_layers_and_norm": 0, "embed_tokens": 0, "lm_head": 0}
    for n, p in model.named_parameters():
        k = p.numel()
        if "visual.merger" in n:
            c["merger"] += k
        elif "visual.deepstack_merger_list" in n:
            c["deepstack_merger_list"] += k
        elif "visual" in n:
            c["vit_blocks_and_patch_embed"] += k
        elif "embed_tokens" in n:
            c["embed_tokens"] += k
        elif n.startswith("lm_head"):
            c["lm_head"] += k
        else:
            c["llm_layers_and_norm"] += k
    c["vit_total"] = c["vit_blocks_and_patch_embed"] + c["merger"] + c["deepstack_merger_list"]
    c["projector_total"] = c["merger"] + c["deepstack_merger_list"]
    c["llm_total"] = c["llm_layers_and_norm"] + c["embed_tokens"] + c["lm_head"]
    c["total"] = c["vit_total"] + c["llm_total"]
    return c


def lora_target_regex(targets):
    if targets == "llm":
        return LLM_TARGETS
    if targets == "llm+vit":
        return f"({LLM_TARGETS})|({VIT_TARGETS})"
    raise ValueError(targets)


def apply_setup(model, targets="llm", train_projector=False, r=16, alpha=32, dropout=0.05,
                gradient_checkpointing=True):
    """E1: targets=llm; E2: + train_projector; E3: targets=llm+vit + train_projector."""
    import re

    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    regex = lora_target_regex(targets)
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=regex, bias="none"))
    if train_projector:
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                if not p.is_floating_point():
                    raise RuntimeError(f"{n} không phải số thực (bị lượng tử hoá?), không train được")
                p.data = p.data.float()
                p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    projector = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and any(k in n for k in PROJECTOR_KEYS))
    info = {"matched_lora_modules": len(matched), "trainable_params": trainable,
            "trainable_projector_params": projector, "trainable_lora_params": trainable - projector}
    return model, info


def projector_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in PROJECTOR_KEYS)}


def load_projector_state(model, state, dtype=None):
    params = dict(model.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise KeyError(f"không tìm thấy {len(missing)} tham số projector, ví dụ {missing[:2]}")
    with torch.no_grad():
        for n, t in state.items():
            p = params[n]
            p.data = t.to(device=p.device, dtype=dtype or p.dtype)
    return len(state)
