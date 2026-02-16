"""Adversarial prefix search for target responses.

Described algorithm: discrete prefix tokens are greedily replaced based on
gradient direction to steer the model toward a desired target text output.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from python.model_load import get_model_device, load_llm


DEFAULT_PREFIX_LEN = 3
DEFAULT_ITERATIONS = 6
MAX_PREFIX_LEN = 32
MAX_ITERATIONS = 64


def _clean_token_ids(value: Any, vocab_size: int, prefix_len: int) -> List[int]:
    if not value:
        return []
    ids: List[int] = []
    iterable: Iterable[Any]
    if isinstance(value, str):
        iterable = [tok for tok in value.replace("/", " ").replace(";", " ").split()]
    elif isinstance(value, Iterable):
        iterable = value
    else:
        iterable = [value]
    for raw in iterable:
        if isinstance(raw, str) and raw.strip() == "":
            continue
        try:
            num = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= num < vocab_size:
            ids.append(num)
            if len(ids) >= prefix_len:
                break
    return ids


def _build_initial_prefix(
    prefix_len: int,
    vocab_size: int,
    device: torch.device,
    seed_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    seed = seed_ids or []
    clean = [int(x) for x in seed if isinstance(x, int) or (isinstance(x, str) and x.isdigit())]
    clean = [x for x in clean if 0 <= x < vocab_size]
    clean = clean[:prefix_len]
    if len(clean) < prefix_len:
        rand = torch.randint(0, vocab_size, (prefix_len - len(clean),), device=device)
        clean.extend(int(x) for x in rand.tolist())
    tensor = torch.tensor(clean[:prefix_len], dtype=torch.long, device=device)
    return tensor


def _token_mask(tokenizer, vocab_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    special_ids = getattr(tokenizer, "all_special_ids", []) or []
    for sid in special_ids:
        if isinstance(sid, int) and 0 <= sid < vocab_size:
            mask[sid] = False
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_id, int) and 0 <= pad_id < vocab_size:
        mask[pad_id] = False
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int) and 0 <= eos_id < vocab_size:
        mask[eos_id] = False
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if isinstance(unk_id, int) and 0 <= unk_id < vocab_size:
        mask[unk_id] = False
    return mask


def find_adversarial_prefix(
    model_key: str,
    input_string: str,
    target_text: str,
    prefix_length: int = DEFAULT_PREFIX_LEN,
    iterations: int = DEFAULT_ITERATIONS,
    initial_prefix_ids: Optional[Sequence[int]] = None,
    initial_prefix_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Greedy discrete attack that replaces one token at a time by the gradient."""

    tokenizer, model = load_llm(model_key)
    device = get_model_device(model)
    model.eval()

    prefix_length = max(1, min(prefix_length, MAX_PREFIX_LEN))
    iterations = max(1, min(iterations, MAX_ITERATIONS))

    prompt_input = (input_string or "").strip()
    target_input = (target_text or "").strip()
    if not target_input:
        raise ValueError("target_text is required")

    prompt_enc = tokenizer(prompt_input, return_tensors="pt", add_special_tokens=True)
    prompt_ids = prompt_enc["input_ids"][0].to(device)
    target_enc = tokenizer(target_input, return_tensors="pt", add_special_tokens=False)
    target_ids = target_enc["input_ids"][0].to(device)
    if target_ids.numel() == 0:
        raise ValueError("target_text must contain at least one token")

    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.num_embeddings
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("invalid vocabulary size")

    text_seed_ids: List[int] = []
    if initial_prefix_text:
        try:
            seed_enc = tokenizer(initial_prefix_text, return_tensors="pt", add_special_tokens=False)
            if seed_enc["input_ids"].numel() > 0:
                text_seed_ids = seed_enc["input_ids"][0].tolist()
        except Exception:
            text_seed_ids = []
    combined_prefix_ids: List[int] = []
    if initial_prefix_ids:
        combined_prefix_ids.extend([int(x) for x in initial_prefix_ids if isinstance(x, int)])
    combined_prefix_ids.extend([x for x in text_seed_ids if isinstance(x, int)])
    prefix_ids = _build_initial_prefix(prefix_length, vocab_size, device, combined_prefix_ids)
    token_mask = _token_mask(tokenizer, vocab_size, device)
    embed_weights = embed_layer.weight.detach()

    best_prefix: torch.Tensor = prefix_ids.clone()
    best_loss = float("inf")
    loss_history: List[float] = []

    for iter_idx in range(iterations):
        context_ids = torch.cat([prefix_ids, prompt_ids], dim=0)
        full_ids = torch.cat([context_ids, target_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids, dtype=torch.long)

        inputs_embeds = embed_layer(full_ids).detach()
        inputs_embeds.requires_grad_(True)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = full_ids[:, 1:]

        start_idx = prefix_length + prompt_ids.size(0)
        if start_idx <= 0:
            start_idx = 1
        target_len = target_ids.size(0)
        slice_logits = shift_logits[:, start_idx - 1 : start_idx - 1 + target_len, :]
        slice_labels = shift_labels[:, start_idx - 1 : start_idx - 1 + target_len]

        loss = F.cross_entropy(slice_logits.reshape(-1, slice_logits.size(-1)), slice_labels.reshape(-1), reduction="sum")
        loss_value = loss.item()
        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_prefix = prefix_ids.clone()

        model.zero_grad()
        loss.backward()

        grads = inputs_embeds.grad
        if grads is None:
            break
        prefix_grad = grads[0, :prefix_length]
        if prefix_grad.numel() == 0:
            break
        grad_norm = prefix_grad.norm(dim=-1)
        if grad_norm.numel() == 0:
            break
        best_pos = int(torch.argmax(grad_norm).item())
        direction = -prefix_grad[best_pos]
        if direction.norm().item() == 0:
            break

        scores = torch.matmul(embed_weights.to(device), direction)
        allowed = token_mask.clone()
        current_id = int(prefix_ids[best_pos].item())
        if 0 <= current_id < vocab_size:
            allowed[current_id] = False
        scores = torch.where(allowed, scores, torch.full_like(scores, float("-inf")))

        next_token = int(torch.argmax(scores).item())
        prefix_ids[best_pos] = next_token

    final_ids = best_prefix if best_loss < float("inf") else prefix_ids
    prefix_strings = tokenizer.convert_ids_to_tokens(final_ids.tolist())
    prefix_text = tokenizer.decode(final_ids.tolist(), skip_special_tokens=True)

    return {
        "status": "ok",
        "input_string": prompt_input,
        "target_text": target_input,
        "prefix_ids": final_ids.tolist(),
        "prefix_tokens": prefix_strings,
        "prefix_text": prefix_text,
        "loss": best_loss if best_loss < float("inf") else None,
        "iterations": iterations,
        "loss_history": loss_history,
    }
