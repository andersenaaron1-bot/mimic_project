import json
from pathlib import Path

import torch
from torch import nn
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ehr_hier.tokenizers.medtok_loader import build_vocab_from_code2embeddings
from src.ehr_hier.tokenizers.medtok_canonicalize import diagnosis_filter
from src.ehr_hier.tokenizers.medtok_embeddings import (
    build_category_embedding_matrix,
    apply_pretrained_to_embedding,
)


def test_build_category_embedding_matrix(tmp_path: Path):
    # Use tiny diag vocab
    code2emb_fp = tmp_path / "code2embeddings.json"
    payload = {
        "<UNK>": [0.0, 0.0, 0.0],
        "ICD10CM//A123": [1.0, 2.0, 3.0],
        "ICD9CM//25000": [4.0, 5.0, 6.0],
    }
    code2emb_fp.write_text(json.dumps(payload))
    vocab = build_vocab_from_code2embeddings(
        str(code2emb_fp), offset=1_000_000, name="diag", filter_fn=diagnosis_filter
    )

    weight, mask = build_category_embedding_matrix(vocab, str(code2emb_fp))
    assert weight.shape == (max(vocab.code2id.values()) + 1, 3)
    assert mask.sum().item() == 3  # includes <UNK>
    assert torch.allclose(weight[vocab.code2id["ICD10CM//A123"]], torch.tensor([1.0, 2.0, 3.0]))


def test_apply_pretrained_to_global_embedding(tmp_path: Path):
    code2emb_fp = tmp_path / "code2embeddings.json"
    payload = {
        "<UNK>": [0.0, 0.0],
        "ICD10CM//A123": [0.1, 0.2],
    }
    code2emb_fp.write_text(json.dumps(payload))
    vocab = build_vocab_from_code2embeddings(
        str(code2emb_fp), offset=1_000_000, name="diag", filter_fn=diagnosis_filter
    )

    # Embedding with room for offset + max_id
    emb = nn.Embedding(num_embeddings=vocab.offset + max(vocab.code2id.values()) + 1, embedding_dim=2)
    mask_global = apply_pretrained_to_embedding(emb, vocab, str(code2emb_fp))
    gid = vocab.offset + vocab.code2id["ICD10CM//A123"]
    assert torch.allclose(emb.weight[gid], torch.tensor([0.1, 0.2]))
    assert mask_global[gid]


def test_apply_with_projector(tmp_path: Path):
    code2emb_fp = tmp_path / "code2embeddings.json"
    payload = {"ICD10CM//A123": [1.0, 2.0, 3.0], "<UNK>": [0.0, 0.0, 0.0]}
    code2emb_fp.write_text(json.dumps(payload))
    vocab = build_vocab_from_code2embeddings(
        str(code2emb_fp), offset=1_000_000, name="diag", filter_fn=diagnosis_filter
    )

    emb = nn.Embedding(num_embeddings=vocab.offset + max(vocab.code2id.values()) + 1, embedding_dim=2)
    projector = nn.Linear(3, 2, bias=False)
    apply_pretrained_to_embedding(emb, vocab, str(code2emb_fp), projector=projector)
    gid = vocab.offset + vocab.code2id["ICD10CM//A123"]
    # Verify projection applied
    expected = projector(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(emb.weight[gid], expected)
