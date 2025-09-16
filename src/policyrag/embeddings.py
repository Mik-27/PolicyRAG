import torch
from typing import Tuple
from torch import Tensor


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def generate_embeddings(tokenizer, model, text, stats_obj=None) -> Tuple[list, tuple]:
    """
    Tokenize `text` (a list of strings) and produce embeddings using `model`.
    `stats_obj` may be an object with `record_count` and `total_token_length` attributes to be incremented.
    Returns (embeddings_numpy, shape)
    """
    doc_batch_dict = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    attention_mask = doc_batch_dict['attention_mask']
    token_counts = attention_mask.sum(dim=1).tolist()
    total_tokens = attention_mask.sum().item()
    if stats_obj is not None:
        stats_obj.record_count += len(token_counts)
        stats_obj.total_token_length += total_tokens

    with torch.no_grad():
        doc_outputs = model(**doc_batch_dict)
        doc_embeddings = last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])

    return doc_embeddings.squeeze().cpu().numpy(), doc_embeddings.shape
