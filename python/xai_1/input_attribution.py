"""
Input attribution for response: which input tokens contributed to the generated output.
Uses Chat Template (system + user). Attribution is over all input tokens before generation
(system instruction + input string). input_grad: gradient of output w.r.t. input embeddings, abs, normalize.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from python.model_load import get_model_device, load_llm
from python.model_generation import chat_completion, _simple_chat_prompt_to_ids


def _messages_to_input_ids(tokenizer, messages: List[Dict[str, str]]):
    """Same tokenization as chat_completion: apply_chat_template or fallback."""
    if getattr(tokenizer, "apply_chat_template", None):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            input_ids = _simple_chat_prompt_to_ids(tokenizer, messages)
    else:
        input_ids = _simple_chat_prompt_to_ids(tokenizer, messages)
    return input_ids


def _get_input_tokens_and_scores_input_grad_chat(
    model_key: str,
    messages: List[Dict[str, str]],
    generated_text: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
) -> Tuple[List[str], List[float], torch.Tensor]:
    """
    Attribution over full prompt (system + user tokens) using chat template.
    Returns (token_strings, normalized_scores_0_1, output_ids_used).
    """
    tokenizer, model = load_llm(model_key)
    device = get_model_device(model)
    model.eval()

    input_ids = _messages_to_input_ids(tokenizer, messages)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    prompt_length = input_ids.shape[1]

    do_sample = temperature > 0
    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kw["temperature"] = temperature
        if top_p < 1.0:
            gen_kw["top_p"] = top_p
        if top_k > 0:
            gen_kw["top_k"] = top_k

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kw)
    output_ids = out[0][prompt_length:].unsqueeze(0)
    gen_len = output_ids.shape[1]
    if gen_len == 0:
        token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])
        return token_strs, [0.0] * len(token_strs), output_ids

    embed_layer = model.get_input_embeddings()
    prompt_embeds = embed_layer(input_ids).detach().clone().requires_grad_(True)
    output_embeds = embed_layer(output_ids)
    full_embeds = torch.cat([prompt_embeds, output_embeds], dim=1)
    seq_len = full_embeds.shape[1]
    attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

    outputs = model(inputs_embeds=full_embeds, attention_mask=attention_mask)
    logits = outputs.logits
    logits_for_gen = logits[0, prompt_length - 1 : prompt_length - 1 + gen_len]
    target = output_ids[0]
    loss = -F.cross_entropy(logits_for_gen, target, reduction="sum")
    loss.backward()

    grad = prompt_embeds.grad
    if grad is None:
        token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])
        return token_strs, [0.0] * len(token_strs), output_ids

    importance = grad.abs().sum(dim=-1).squeeze(0)
    importance = importance.detach().cpu().float()
    min_val = importance.min().item()
    max_val = importance.max().item()
    if max_val > min_val:
        scores = ((importance - min_val) / (max_val - min_val)).tolist()
    else:
        scores = [0.0] * importance.shape[0]

    token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])
    return token_strs, scores, output_ids


def compute_input_attribution(
    model_key: str,
    input_string: str,
    system_instruction: str = "",
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    top_p: float = 1.0,
    top_k: int = 50,
    attribution_method: str = "input_grad",
) -> Dict[str, Any]:
    """
    Run chat completion (Chat Template: system + user) then compute input token attribution
    over all input tokens before generation (system instruction + input string).

    Returns:
        generated_text, input_tokens, token_scores, system_instruction, input_string, ...
    """
    system_instruction = (system_instruction or "").strip()
    input_string = (input_string or "").strip()
    messages: List[Dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": input_string})

    do_sample = temperature > 0
    generated_text = chat_completion(
        model_key=model_key,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    if not attribution_method or attribution_method.lower() == "none":
        tokenizer, _ = load_llm(model_key)
        input_ids = _messages_to_input_ids(tokenizer, messages)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])
        zeros = [0.0] * len(token_strs)
        return {
            "generated_text": generated_text,
            "input_tokens": token_strs,
            "token_scores": zeros,
            "token_scores_drop_special": zeros,
            "attribution_method": "none",
            "system_instruction": system_instruction,
            "input_string": input_string,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }

    if attribution_method.lower() == "input_grad":
        token_strs, scores, _ = _get_input_tokens_and_scores_input_grad_chat(
            model_key=model_key,
            messages=messages,
            generated_text=generated_text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        # Special tokens → min score for "dropped" visualization
        tokenizer, _ = load_llm(model_key)
        special_set = set(getattr(tokenizer, "all_special_tokens", []) or [])
        min_score = min(scores) if scores else 0.0
        token_scores_drop_special = [
            min_score if t in special_set else scores[i]
            for i, t in enumerate(token_strs)
        ]
        return {
            "generated_text": generated_text,
            "input_tokens": token_strs,
            "token_scores": scores,
            "token_scores_drop_special": token_scores_drop_special,
            "attribution_method": "input_grad",
            "system_instruction": system_instruction,
            "input_string": input_string,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }

    raise ValueError(f"Unknown attribution_method: {attribution_method}")
