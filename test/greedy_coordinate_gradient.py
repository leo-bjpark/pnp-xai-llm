import torch 
import torch.nn.functional as F

def format_chat(
    tokenizer,
    prompt: str,
    response: str | None = None,
    add_generation_prompt: bool = False,
):
    messages = [{"role": "user", "content": prompt}]

    if response is not None:
        messages.append(
            {"role": "assistant", "content": response}
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def find_subsequence_span(input_ids: torch.Tensor,
                          target_ids: torch.Tensor):
    """
    Find (start, end) indices such that:
    input_ids[start:end] == target_ids

    end is exclusive.
    """
    input_len = input_ids.size(0)
    target_len = target_ids.size(0)
    
    for i in range(input_len - target_len + 1):
        if torch.equal(input_ids[i:i + target_len], target_ids):
            return i, i + target_len

    target_ids = target_ids[1:-1]
    target_len = target_ids.size(0)
    for i in range(input_len - target_len + 1):
        if torch.equal(input_ids[i:i + target_len], target_ids):
            return i, i + target_len

    raise ValueError("target_ids not found in input_ids", target_ids, input_ids)



def batch_gcg_input_prepare(tokenizer, prefix_prompts, update_strings, postfix_prompts, contexts, target_strings):
    full_input_ids = [] 
    generation_input_ids = []
    update_spans = [] 
    generation_update_spans = [] 
    target_spans = []
    for prefix_prompt, update_string, postfix_prompt, context, target_string in zip(prefix_prompts, update_strings, postfix_prompts, contexts, target_strings):
        user_prompt = f"{prefix_prompt}{update_string}{postfix_prompt}{context}"
        full_text = format_chat(
            tokenizer,
            prompt=user_prompt,
            response=target_string,
            add_generation_prompt=True
        )
        
        current_input_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
        current_update_token_ids = tokenizer.encode(update_string, add_special_tokens=False, return_tensors="pt")[0]
        current_target_token_ids = tokenizer.encode(target_string, add_special_tokens=False, return_tensors="pt")[0]

        full_input_ids.append(current_input_ids)
        
        
        generation_text = format_chat(
            tokenizer,
            prompt=user_prompt,
            response=None,
            add_generation_prompt=False
        )
        current_generation_input_ids = tokenizer(generation_text, return_tensors="pt").input_ids[0]
        generation_input_ids.append(current_generation_input_ids)
        
        start, end = find_subsequence_span(current_input_ids, current_update_token_ids)
        update_spans.append((start, end))
        start, end = find_subsequence_span(current_input_ids, current_target_token_ids)
        target_spans.append((start, end))
        
        start, end = find_subsequence_span(current_generation_input_ids, current_update_token_ids)
        generation_update_spans.append((start, end))
        
    # batchify by left padding 
    max_length = max([len(v) for v in full_input_ids])
    max_length_for_generation = max([len(v) for v in generation_input_ids])
    for i in range(len(full_input_ids)):
        left_padding = max_length - len(full_input_ids[i])
        padding = torch.zeros(left_padding).long().fill_(tokenizer.pad_token_id)
        full_input_ids[i] = torch.cat([padding, full_input_ids[i]])
        update_spans[i] = (update_spans[i][0] + left_padding, update_spans[i][1] + left_padding)
        target_spans[i] = (target_spans[i][0] + left_padding, target_spans[i][1] + left_padding)
        
        # left padding for generation
        left_padding = max_length_for_generation - len(generation_input_ids[i])
        padding = torch.zeros(left_padding).long().fill_(tokenizer.pad_token_id)
        generation_input_ids[i] = torch.cat([padding, generation_input_ids[i]])
        generation_update_spans[i] = (generation_update_spans[i][0] + left_padding, generation_update_spans[i][1] + left_padding)
        
        tokens = full_input_ids[i][update_spans[i][0]:update_spans[i][1]]
        decoded_tokens = tokenizer.decode(tokens)
        if decoded_tokens != update_strings[i]:
            print(f"[Warning] update_string is not correct: ORIGINAL: {update_strings[i]} | DECODED: {decoded_tokens}")
        
        tokens = full_input_ids[i][target_spans[i][0]:target_spans[i][1]]
        decoded_tokens = tokenizer.decode(tokens)
        if decoded_tokens != target_strings[i]:
            print(f"[Warning] target_string is not correct: ORIGINAL: {target_strings[i]} | DECODED: {decoded_tokens}")
            
    full_input_ids = torch.stack(full_input_ids)
    generation_input_ids = torch.stack(generation_input_ids)
    update_spans = torch.tensor(update_spans)
    target_spans = torch.tensor(target_spans)
    generation_update_spans = torch.tensor(generation_update_spans)
    return full_input_ids, generation_input_ids, update_spans, target_spans, generation_update_spans



def masked_ce_loss_on_target_spans(logits: torch.Tensor,
                                  input_ids: torch.Tensor,
                                  target_spans: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    device = logits.device

    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = input_ids[:, 1:].clone()        # [B, T-1]

    loss_mask = torch.zeros((B, T - 1), dtype=torch.bool, device=device)

    for b in range(B):
        s, e = target_spans[b].tolist()
        # shifted indices 적용: [s-1, e-1)
        ss = max(s - 1, 0)
        ee = max(e - 1, 0)
        ss = min(ss, T - 1)
        ee = min(ee, T - 1)
        if ee > ss:
            loss_mask[b, ss:ee] = True

    masked_labels = shift_labels.clone()
    masked_labels[~loss_mask] = -100

    loss = F.cross_entropy(
        shift_logits.view(-1, V),
        masked_labels.view(-1),
        ignore_index=-100
    )
    return loss

def is_clean_text(token_id, tokenizer):
    text = tokenizer.decode([token_id])
    return text.isalnum() or text.isspace() or any(c in ".,!?" for c in text)



from tqdm import tqdm
def batch_gcg_chat_update(model,
                          tokenizer,
                          full_input_ids: torch.Tensor,
                          update_spans: torch.Tensor,
                          target_spans: torch.Tensor,
                          num_steps: int = 40,
                          top_k: int = 32, 
                          verbose: bool = True):
    
    
    vocab_size = model.get_input_embeddings().weight.shape[0]
    clean_indices = []
    for i in range(vocab_size):
        if is_clean_text(i, tokenizer):
            clean_indices.append(i)
    
    clean_indices = torch.tensor(clean_indices, device=model.device)
    
    
    device = model.device
    input_ids = full_input_ids.clone().to(device)
    update_spans = update_spans.to(device)
    target_spans = target_spans.to(device)

    emb_layer = model.get_input_embeddings()
    emb_weight = emb_layer.weight  # [V, d]

    if verbose:
        pbar = tqdm(range(num_steps))
    else:
        pbar = range(num_steps)
    losses = []
    for step in pbar:
        input_ids.requires_grad = False
        inputs_embeds = emb_layer(input_ids).detach().requires_grad_(True)
        
        outputs = model(inputs_embeds=inputs_embeds)
        logits = outputs.logits
        loss = masked_ce_loss_on_target_spans(logits, input_ids, target_spans)
        
        model.zero_grad()
        loss.backward()
        grad = inputs_embeds.grad  # [B, T, d]

        with torch.no_grad():
            new_input_ids = input_ids.clone()
            
            for b in range(input_ids.size(0)):
                us, ue = update_spans[b].tolist()
                span_grad = grad[b, us:ue]
                pos_in_span = torch.randint(0, ue - us, (1,)).item()
                pos = us + pos_in_span
                
                g = grad[b, pos]
                scores = torch.matmul(emb_weight, -g) # [V]
                
                
                mask = torch.ones_like(scores, dtype=torch.bool)
                mask[clean_indices] = False
                scores[mask] = -float('inf')
                
                top_indices = torch.topk(scores, top_k).indices # [batch_size]
                
                candidate_ids = input_ids[b:b+1].repeat(top_k, 1)
                candidate_ids[:, pos] = top_indices
                
                b_target_spans = target_spans[b:b+1].repeat(top_k, 1)
                
                candidate_logits = model(input_ids=candidate_ids).logits
                
                best_loss = float('inf')
                best_token = input_ids[b, pos].item()
                
                for i in range(top_k):
                    tmp_loss = masked_ce_loss_on_target_spans(
                        candidate_logits[i:i+1], candidate_ids[i:i+1], b_target_spans[i:i+1]
                    ).item()
                    
                    if tmp_loss < best_loss:
                        best_loss = tmp_loss
                        best_token = top_indices[i].item()
                
                new_input_ids[b, pos] = best_token

            input_ids = new_input_ids
        losses.append(loss.item())
        if verbose:
            decoded_updates = [tokenizer.decode(input_ids[b, update_spans[b, 0]:update_spans[b, 1]]) 
                               for b in range(input_ids.size(0))]
            pbar.set_postfix({"loss": loss.item(), "step": step, "suffix": decoded_updates})
    return input_ids, losses
