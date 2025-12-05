from __future__ import annotations
import json
from typing import Dict, Tuple, Optional

import torch
from torch import nn

from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab


def load_code_embeddings(json_fp: str) -> Dict[str, torch.Tensor]:
    """
    Load MedTok code2embeddings.json -> {code: [float...]} as torch tensors.
    """
    with open(json_fp, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        out[str(k)] = torch.as_tensor(v, dtype=torch.float32)
    return out


def build_category_embedding_matrix(
    vocab: CategoryVocab,
    code2embeddings_fp: str,
    *,
    normalize: bool = False,
    init_std: float = 0.02,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Build an embedding matrix indexed by *raw* category ids (0..max_id) and a mask
    indicating which rows were filled from pretrained MedTok embeddings.

    Returns
    -------
    weight : torch.Tensor [vocab_size, d_emb]
    mask   : torch.BoolTensor [vocab_size] True if row came from MedTok
    """
    code2emb = load_code_embeddings(code2embeddings_fp)
    if not code2emb:
        raise ValueError(f"No embeddings found in {code2embeddings_fp}")

    # Use the first vector to infer dimension
    first_vec = next(iter(code2emb.values()))
    d_emb = first_vec.numel()
    size = max(vocab.code2id.values()) + 1

    weight = torch.empty(size, d_emb, dtype=torch.float32)
    torch.nn.init.normal_(weight, mean=0.0, std=init_std)
    mask = torch.zeros(size, dtype=torch.bool)

    for code, raw_id in vocab.code2id.items():
        vec = code2emb.get(code)
        if vec is None or vec.numel() != d_emb:
            continue
        if normalize:
            norm = vec.norm()
            if norm > 0:
                vec = vec / norm
        weight[raw_id] = vec
        mask[raw_id] = True

    return weight, mask


def apply_pretrained_to_embedding(
    emb: nn.Embedding,
    vocab: CategoryVocab,
    code2embeddings_fp: str,
    *,
    normalize: bool = False,
    projector: Optional[nn.Module] = None,
) -> torch.BoolTensor:
    """
    In-place initialize rows of a *global* embedding table using MedTok pretrained
    embeddings for a given category. The embedding must be large enough to index
    vocab.offset + raw_id.

    Args
    ----
    emb : nn.Embedding
        Global embedding table indexed by global token ids.
    vocab : CategoryVocab
        Category vocabulary with offset + code2id.
    code2embeddings_fp : str
        Path to code2embeddings.json.
    normalize : bool
        L2-normalize pretrained vectors before injection.
    projector : nn.Module
        Optional projector mapping pretrained_dim -> emb.embedding_dim. If None,
        dimensions must match exactly.

    Returns
    -------
    mask_global : torch.BoolTensor
        Boolean mask over emb.weight rows that were overwritten.
    """
    weight_local, mask_local = build_category_embedding_matrix(
        vocab, code2embeddings_fp, normalize=normalize
    )
    device = emb.weight.device
    weight_local = weight_local.to(device)
    if projector is not None:
        projected = projector(weight_local)
    else:
        if weight_local.size(1) != emb.embedding_dim:
            raise ValueError(
                f"Dim mismatch: pretrained {weight_local.size(1)} vs emb {emb.embedding_dim}. "
                "Pass a projector to map dimensions."
            )
        projected = weight_local

    idx_local = torch.nonzero(mask_local, as_tuple=False).squeeze(1)
    if idx_local.numel() == 0:
        return torch.zeros_like(emb.weight, dtype=torch.bool)

    idx_global = idx_local + vocab.offset
    if idx_global.max().item() >= emb.num_embeddings:
        raise ValueError(
            f"Embedding too small: needs at least {int(idx_global.max())+1} rows for offset {vocab.offset}."
        )

    mask_global = torch.zeros(emb.num_embeddings, dtype=torch.bool, device=device)
    mask_global[idx_global] = True
    with torch.no_grad():
        emb.weight[idx_global] = projected[idx_local]
    return mask_global
