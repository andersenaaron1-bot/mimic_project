import argparse
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ehr_hier.models.value_encoders.cvae import ValueCVAE, ValueCVAEConfig
from src.ehr_hier.tokenizers.value_tokenizer import ValueDiscretizer, TokenizerConfig
from src.ehr_hier.data.meds_value_dataset import ValueEventsDataset, collate_value_batch

Sample = Dict[str, Any]



def estimate_var_stats_from_dataset(
    dataset: Iterable[Sample],
    max_samples: Optional[int] = 100_000,
    min_std: float = 1e-3,
    min_count: int = 10,
) -> (torch.Tensor, torch.Tensor):
    """
    Estimate per-var_id mean/std for value, with a safeguard for rare vars.

    - For vars with count >= min_count: per-var mean/std.
    - For vars with count <  min_count: use global mean/std.

    Returns
    -------
    mean : torch.Tensor [max_var_id+1]
    std  : torch.Tensor [max_var_id+1]
        Index 0 is unused. Var_ids are expected to be >= 1.
    """
    sum_by_var = defaultdict(float)
    sumsq_by_var = defaultdict(float)
    count_by_var = defaultdict(int)

    # Global stats
    g_sum = 0.0
    g_sumsq = 0.0
    g_count = 0

    for i, s in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        v = s.get("value", None)
        vid = s.get("var_id", None)
        if v is None or vid is None:
            continue

        try:
            v = float(v)
            vid = int(vid)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(v) or vid <= 0:
            continue

        sum_by_var[vid] += v
        sumsq_by_var[vid] += v * v
        count_by_var[vid] += 1

        g_sum += v
        g_sumsq += v * v
        g_count += 1

    if g_count == 0:
        mean = torch.tensor([0.0], dtype=torch.float32)
        std = torch.tensor([1.0], dtype=torch.float32)
        return mean, std

    g_mean = g_sum / g_count
    g_var = (g_sumsq / g_count) - g_mean * g_mean
    if g_var < 0:
        g_var = 0.0
    g_std = max(math.sqrt(g_var), min_std)

    max_vid = max(count_by_var.keys()) if count_by_var else 0
    mean = torch.zeros(max_vid + 1, dtype=torch.float32)
    std = torch.ones(max_vid + 1, dtype=torch.float32)

    for vid in range(1, max_vid + 1):
        c = count_by_var.get(vid, 0)
        if c >= min_count:
            s = sum_by_var[vid]
            ss = sumsq_by_var[vid]
            m = s / c
            var = (ss / c) - m * m
            if var < 0:
                var = 0.0
            sd = math.sqrt(var)
            if sd < min_std:
                sd = min_std
            mean[vid] = m
            std[vid] = sd
        else:
            mean[vid] = g_mean
            std[vid] = g_std

    return mean, std


def main():
    ap = argparse.ArgumentParser()
    # meds_reader DB, not raw MEDS root
    ap.add_argument("--meds_reader_db", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--splits_parquet", type=str, default=None)
    ap.add_argument("--codes_parquet", type=str, default=None)

    ap.add_argument("--cvae_ckpt", type=str, required=True)

    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)

    # tokenizer cfg
    ap.add_argument("--d_val", type=int, default=64)
    ap.add_argument("--num_codebooks", type=int, default=2)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--beta_commit", type=float, default=0.25)
    ap.add_argument("--attn_topk", type=int, default=8)
    ap.add_argument("--attn_temp", type=float, default=0.5)
    ap.add_argument("--aux_soft_weight", type=float, default=0.05)
    ap.add_argument("--aux_entropy_weight", type=float, default=0.01)

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", type=str, default="./value_tokenizer.pt")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) Frozen cVAE encoder
    ckpt = torch.load(args.cvae_ckpt, map_location="cpu")
    cvae_cfg = ValueCVAEConfig(**ckpt["cfg"])
    assert cvae_cfg.z_dim == args.d_val, f"cVAE z_dim={cvae_cfg.z_dim} != d_val={args.d_val}"

    # infer n_vars from checkpoint
    n_vars = ckpt["state_dict"]["var_emb.weight"].shape[0]
    print("Loaded cVAE with n_vars =", n_vars)

    cvae = ValueCVAE(n_vars=n_vars, cfg=cvae_cfg).to(device).eval()
    cvae.load_state_dict(ckpt["state_dict"], strict=False)
    for p in cvae.parameters():
        p.requires_grad = False

    # 2) Dataset: stats + training versions
    ds_stats = ValueEventsDataset(
        meds_reader_db=args.meds_reader_db,
        split=args.split,
        splits_parquet=args.splits_parquet,
        codes_parquet=args.codes_parquet,
        shuffle_subjects=False,
    )
    ds = ValueEventsDataset(
        meds_reader_db=args.meds_reader_db,
        split=args.split,
        splits_parquet=args.splits_parquet,
        codes_parquet=args.codes_parquet,
        shuffle_subjects=True,
    )

    print(f"Dataset split={args.split}, subjects={len(ds.subject_ids)}")

    print("Estimating per-variable mean/std for value (tokenizer)...")
    mean_by_var, std_by_var = estimate_var_stats_from_dataset(
        dataset=ds_stats,
        max_samples=100_000,
        min_std=1e-3,
        min_count=10,
    )
    print("  max var_id seen:", mean_by_var.shape[0] - 1)

    mean_by_var = mean_by_var.to(device)
    std_by_var = std_by_var.to(device)

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.workers,
        collate_fn=collate_value_batch,
    )

    # 3) Tokenizer model
    tok_cfg = TokenizerConfig(
        d_val=args.d_val,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        beta_commit=args.beta_commit,
        attn_topk=args.attn_topk,
        attn_temp=args.attn_temp,
        aux_soft_weight=args.aux_soft_weight,
        aux_entropy_weight=args.aux_entropy_weight,
    )
    tok = ValueDiscretizer(tok_cfg).to(device)
    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr)

    step = 0
    for ep in range(args.epochs):
        t0 = time.time()
        for b in dl:
            # move to device
            b = {k: v.to(device) for k, v in b.items()}
            var_id = b["var_id"].long()
            value = b["value"].float()
            dt_prev = b["dt_prev"].float()
            age = b["age"].float()
            sex = b["sex"].float()

            # ---- same normalization as CVAE training ----

            # per-var z-score, with rare-vars fallback & clipping
            max_idx = min(var_id.max().item(), mean_by_var.shape[0] - 1)
            vid_clamped = var_id.clamp(max=max_idx)
            mean = mean_by_var[vid_clamped]
            std = std_by_var[vid_clamped]
            value_z = (value - mean) / std
            value_z = value_z.clamp(min=-8.0, max=8.0)

            # dt_prev: clamp and log1p
            dt_prev_hours = dt_prev.clamp(min=0.0, max=24.0 * 7 * 4)
            dt_prev_norm = torch.log1p(dt_prev_hours)

            # age: clamp 0..100, scale to [0,1]
            age_norm = age.clamp(min=0.0, max=100.0) / 100.0

            # sex: already 0/1
            sex_norm = sex

            # ---- encode with frozen cVAE ----
            with torch.no_grad():
                # value_std: [N], var_id: [N], etc. -> z: [N, D]
                z = cvae.encode_mu(
                    value_std=value_z,
                    var_id=var_id,
                    dt_prev=dt_prev_norm,
                    age=age_norm,
                    sex=sex_norm,
                )

            # Treat as [B=1, T=N, D]
            z = z.unsqueeze(0)  # [1, N, D]

            # Forward through tokenizer
            out = tok(z)  # {indices, z_hat, loss, stats}

            # Extra recon loss between z and z_hat (helps early training)
            rec = F.mse_loss(out["z_hat"], z.detach())
            loss = out["loss"] + rec

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                s = out["stats"]
                print(
                    f"ep {ep} step {step} "
                    f"loss {loss.item():.4f} | "
                    f"vq {s['vq_loss']:.4f} soft {s['soft_rec']:.4f} | "
                    f"uniq_lvl {s.get('avg_uniq_codes_per_level', -1):.1f} | "
                    f"attn_H {s['attn_entropy']:.3f}"
                )
            step += 1

        print(f"Epoch {ep} in {time.time()-t0:.1f}s")

    torch.save(
        {
            "cfg": {
                "d_val": args.d_val,
                "num_codebooks": args.num_codebooks,
                "codebook_size": args.codebook_size,
                "beta_commit": args.beta_commit,
                "attn_topk": args.attn_topk,
                "attn_temp": args.attn_temp,
                "aux_soft_weight": args.aux_soft_weight,
                "aux_entropy_weight": args.aux_entropy_weight,
            },
            "state_dict": tok.state_dict(),
        },
        args.out,
    )
    print(f"Saved tokenizer → {args.out}")


if __name__ == "__main__":
    main()
