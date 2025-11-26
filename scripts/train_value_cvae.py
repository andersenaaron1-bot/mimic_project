import argparse, sys, time, math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ehr_hier.data.meds_value_dataset import ValueEventsDataset, collate_value_batch
from src.ehr_hier.models.value_encoders.cvae import ValueCVAE, ValueCVAEConfig

Sample = Dict[str, Any]

def estimate_var_stats_from_dataset(
    dataset: Iterable[Sample],
    max_samples: Optional[int] = 100_000,
    min_std: float = 1e-3,
) -> (torch.Tensor, torch.Tensor):
    sum_by_var = defaultdict(float); sumsq_by_var = defaultdict(float); count_by_var = defaultdict(int)
    for i, s in enumerate(dataset):
        if max_samples is not None and i >= max_samples: break
        v, vid = s.get("value"), s.get("var_id")
        if v is None or vid is None: continue
        try:
            v = float(v); vid = int(vid)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or vid <= 0: continue
        sum_by_var[vid] += v; sumsq_by_var[vid] += v * v; count_by_var[vid] += 1

    if not count_by_var:
        return torch.tensor([0.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32)

    max_vid = max(count_by_var.keys())
    mean = torch.zeros(max_vid + 1, dtype=torch.float32)
    std  = torch.ones (max_vid + 1, dtype=torch.float32)
    for vid in range(1, max_vid + 1):
        c = count_by_var.get(vid, 0)
        if c == 0: mean[vid] = 0.0; std[vid] = 1.0; continue
        s = sum_by_var[vid]; ss = sumsq_by_var[vid]
        m = s / c; var = max(0.0, (ss / c) - m * m)
        sd = max(min_std, math.sqrt(var))
        mean[vid] = m; std[vid] = sd
    return mean, std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meds_reader_db", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    # NEW: frozen mapping + optional precomputed stats
    ap.add_argument("--code2id_pt", type=str, required=True, help="torch.save()'d dict {code->var_id} (measurement-only)")
    ap.add_argument("--stats_pt", type=str, default=None, help="optional torch file with {'mean_by_var','std_by_var'}")
    # keep n_vars but allow auto if omitted
    ap.add_argument("--n_vars", type=int, default=-1)
    ap.add_argument("--splits_parquet", type=str, default=None)
    ap.add_argument("--codes_parquet", type=str, default=None)  # unused when fixed mapping is provided

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

    # ---- load frozen measurement mapping ----
    code2id_meas: Dict[str, int] = torch.load(args.code2id_pt)
    n_vars = (max(code2id_meas.values()) + 1) if args.n_vars <= 0 else args.n_vars
    print(f"n_vars = {n_vars} (from mapping)")

    # ---- dataset (uses frozen mapping; skips unknowns) ----
    ds = ValueEventsDataset(
        meds_reader_db=args.meds_reader_db,
        split=args.split,
        splits_parquet=args.splits_parquet,
        codes_parquet=None,              # ignore dynamic mapping
        include_code_fn=None,
        allowed_var_ids=None,
        shuffle_subjects=True,
        fixed_code2id=code2id_meas,      # <<< NEW
        strict_codes=True,               # <<< NEW (skip unknown)
    )
    print(f"Dataset split={args.split}, subjects={len(ds.subject_ids)}")

    # ---- stats: load or compute on the same split (non-shuffled pass) ----
    if args.stats_pt:
        stats = torch.load(args.stats_pt, map_location="cpu")
        mean_by_var = stats["mean_by_var"]; std_by_var = stats["std_by_var"]
        print(f"Loaded stats from {args.stats_pt} (max_vid={mean_by_var.shape[0]-1})")
    else:
        print("Estimating per-variable mean/std (one pass)...")
        ds_stats = ValueEventsDataset(
            meds_reader_db=args.meds_reader_db,
            split=args.split,
            splits_parquet=args.splits_parquet,
            codes_parquet=None,
            include_code_fn=None,
            allowed_var_ids=None,
            shuffle_subjects=False,
            fixed_code2id=code2id_meas,
            strict_codes=True,
        )
        mean_by_var, std_by_var = estimate_var_stats_from_dataset(ds_stats, max_samples=100_000, min_std=1e-3)
        print("  max var_id seen:", mean_by_var.shape[0] - 1)

    mean_by_var = mean_by_var.to(device)
    std_by_var  = std_by_var.to(device)

    dl = DataLoader(ds, batch_size=args.batch, num_workers=0, collate_fn=collate_value_batch)

    # ---- model ----
    cfg = ValueCVAEConfig(
        z_dim=args.z_dim, hidden=args.hidden, var_emb_dim=args.var_emb_dim,
        use_dt_prev=True, use_age=True, use_sex=True, beta_kl=args.beta_kl,
    )
    model = ValueCVAE(n_vars=n_vars, cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train(); step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            var_id = batch["var_id"].long()
            value  = batch["value"].float()

            # per-var z-score (clamped to known stats)
            max_idx = min(var_id.max().item(), mean_by_var.shape[0] - 1)
            vid = var_id.clamp(max=max_idx)
            mean = mean_by_var[vid]; std = std_by_var[vid]
            value_z = (value - mean) / std
            value_z = value_z.clamp(min=-8.0, max=8.0)

            dt_prev_norm = torch.log1p(batch["dt_prev"].clamp(min=0.0, max=24.0 * 7 * 4))
            age_norm     = batch["age"].clamp(min=0.0, max=100.0) / 100.0
            sex_norm     = batch["sex"].clamp(0.0, 1.0)

            out = model(value_z, var_id, dt_prev_norm, age_norm, sex_norm)

            opt.zero_grad(set_to_none=True); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

            if step % 50 == 0:
                print(
                    f"ep {epoch} step {step} "
                    f"| loss {out['loss'].item():.4f} "
                    f"| recon {out['recon_nll'].item():.4f} "
                    f"| kl {out['kl'].item():.4f} "
                    f"| value_z μ {value_z.mean().item():.3f} σ {value_z.std().item():.3f}"
                )
            step += 1
        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s")

    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
