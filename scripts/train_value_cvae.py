import argparse
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ehr_hier.data.meds_value_dataset import  (
    ValueEventsDataset,
    collate_value_batch,
)
from src.ehr_hier.models.value_encoders.cvae import (
    ValueCVAE,
    ValueCVAEConfig,
)


Sample = Dict[str, Any]


# ---------------------------------------------------------------------------
# 1. Estimate per-variable mean/std for value
# ---------------------------------------------------------------------------

def estimate_var_stats_from_dataset(
    dataset: Iterable[Sample],
    max_samples: Optional[int] = 100_000,
    min_std: float = 1e-3,
) -> (torch.Tensor, torch.Tensor):
    """
    One-pass estimator for per-var_id mean/std, using samples from ValueEventsDataset.

    Returns
    -------
    mean : torch.Tensor [max_var_id+1]
    std  : torch.Tensor [max_var_id+1]
        Index 0 is unused. Var_ids are expected to be >= 1.
    """
    sum_by_var = defaultdict(float)
    sumsq_by_var = defaultdict(float)
    count_by_var = defaultdict(int)

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

    if not count_by_var:
        # Fallback: no samples -> trivial stats
        mean = torch.tensor([0.0], dtype=torch.float32)
        std = torch.tensor([1.0], dtype=torch.float32)
        return mean, std

    max_vid = max(count_by_var.keys())
    mean = torch.zeros(max_vid + 1, dtype=torch.float32)
    std = torch.ones(max_vid + 1, dtype=torch.float32)

    for vid in range(1, max_vid + 1):
        c = count_by_var.get(vid, 0)
        if c == 0:
            mean[vid] = 0.0
            std[vid] = 1.0
            continue

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

    return mean, std


# ---------------------------------------------------------------------------
# 2. Trainer
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--meds_reader_db",
        type=str,
        required=True,
        help="Path to meds_reader SubjectDatabase (output of meds_reader_convert)",
    )
    ap.add_argument("--split", type=str, default="train")

    ap.add_argument(
        "--n_vars",
        type=int,
        required=True,
        help=(
            "Size of MEDS vocab (max var_id). "
            "E.g., max(code_id)+1 from codes.parquet, or max(ds.code2id.values())+1."
        ),
    )
    ap.add_argument(
        "--splits_parquet",
        type=str,
        default=None,
        help="Path to MEDS metadata/subject_splits.parquet",
    )
    ap.add_argument(
        "--codes_parquet",
        type=str,
        default=None,
        help="Path to MEDS metadata/codes.parquet",
    )

    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--var_emb_dim", type=int, default=64)
    ap.add_argument("--beta_kl", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="./cvae_ckpt.pt")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ds = ValueEventsDataset(
        meds_reader_db=args.meds_reader_db,
        split=args.split,
        splits_parquet=args.splits_parquet,
        codes_parquet=args.codes_parquet,
        include_code_fn=None,
        allowed_var_ids=None,
        shuffle_subjects=True,
    )

    print(f"Dataset split={args.split}, subjects={len(ds.subject_ids)}")

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.workers,
        collate_fn=collate_value_batch,
    )

    # Move stats to device
    mean_by_var = mean_by_var.to(device)
    std_by_var = std_by_var.to(device)

    # ---------------- Model ----------------

    cfg = ValueCVAEConfig(
        z_dim=args.z_dim,
        hidden=args.hidden,
        var_emb_dim=args.var_emb_dim,
        beta_kl=args.beta_kl,
    )
    model = ValueCVAE(n_vars=args.n_vars, cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---------------- Training loop ----------------

    model.train()
    step = 0
    for epoch in range(args.epochs):
        t0 = time.time()

        for batch in dl:
            # batch keys: value, var_id, dt_prev, age, sex
            batch = {k: v.to(device) for k, v in batch.items()}

            # 1) Standardize value per var_id: value_z
            var_id = batch["var_id"].long()
            value = batch["value"].float()

            # Guard against var_ids beyond known stats
            max_idx = min(var_id.max().item(), mean_by_var.shape[0] - 1)
            vid_clamped = var_id.clamp(max=max_idx)

            mean = mean_by_var[vid_clamped]
            std = std_by_var[vid_clamped]

            value_z = (value - mean) / std
            value_z = value_z.clamp(min=-8.0, max=8.0)

            # 2) Normalize dt_prev: clamp long gaps, then log1p
            dt_prev_hours = batch["dt_prev"].clamp(min=0.0, max=24.0 * 7 * 4)
            dt_prev_norm = torch.log1p(dt_prev_hours)

            # 3) Normalize age: clamp 0..100, scale to [0,1]
            age_norm = batch["age"].clamp(min=0.0, max=100.0) / 100.0

            # 4) Sex: already 0.0 (non-male) / 1.0 (male)
            sex_norm = batch["sex"]

            # 5) Forward pass
            out = model(
                value_z,       # standardized numeric value
                var_id,        # categorical ID
                dt_prev_norm,
                age_norm,
                sex_norm,
            )

            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(
                    f"ep {epoch} step {step} "
                    f"loss {out['loss'].item():.4f} | "
                    f"recon {out['recon_nll'].item():.4f} | "
                    f"kl {out['kl'].item():.4f}"
                    f"value_z mean {value_z.mean().item():.3f} std {value_z.std().item():.3f} | "
                    f"dt_prev_norm mean {dt_prev_norm.mean().item():.3f} max {dt_prev_norm.max().item():.3f} | "
                    f"age_norm mean {age_norm.mean().item():.3f}"
                )
            step += 1

        print(f"Epoch {epoch} done in {time.time() - t0:.1f}s")

    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
