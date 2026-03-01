#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.ehr_hier.data.meds_value_dataset import ValueEventsDataset, collate_value_batch
from src.ehr_hier.models.value_encoders.cvae import ValueCVAE, ValueCVAEConfig


def _quantile(sorted_vals: List[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(round((len(sorted_vals) - 1) * q))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return int(sorted_vals[idx])


def summarize_train_sparsity(count_by_var: torch.Tensor) -> Dict[str, object]:
    vals = [int(x) for x in count_by_var.tolist()[1:] if int(x) > 0]
    vals.sort()
    n = len(vals)
    total = int(sum(vals))
    thr = [1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000]

    out: Dict[str, object] = {
        "n_vars_with_events": n,
        "n_events": total,
        "quantiles": {
            "p10": _quantile(vals, 0.10),
            "p25": _quantile(vals, 0.25),
            "p50": _quantile(vals, 0.50),
            "p75": _quantile(vals, 0.75),
            "p90": _quantile(vals, 0.90),
            "p99": _quantile(vals, 0.99),
        },
    }
    for t in thr:
        c = sum(1 for v in vals if v <= t)
        out[f"vars_le_{t}"] = c
        out[f"vars_le_{t}_frac"] = (c / n) if n > 0 else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CVAE reconstruction quality overall and per measurement code.")
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--stats_pt", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--split", default="tuning", help="Evaluation split: train|tuning|held_out")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max_steps", type=int, default=-1, help="Limit number of eval batches; -1 for full split")
    ap.add_argument("--min_eval_count", type=int, default=100, help="Min eval count per var for top lists")
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--train_batch_for_steps", type=int, default=8192)
    ap.add_argument("--per_var_csv", default=None, help="Optional CSV path for per-var metrics")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    code2id: Dict[str, int] = torch.load(args.code2id_pt, map_location="cpu")
    id2code = {v: k for k, v in code2id.items()}

    stats_obj = torch.load(args.stats_pt, map_location="cpu")
    mean_by_var: torch.Tensor = stats_obj["mean_by_var"]
    std_by_var: torch.Tensor = stats_obj["std_by_var"]
    count_by_var: torch.Tensor = stats_obj["count_by_var"]

    train_events = int(count_by_var.sum().item())
    steps_per_epoch = int(math.ceil(train_events / args.train_batch_for_steps))
    sparsity = summarize_train_sparsity(count_by_var)

    ckpt = torch.load(args.cvae_ckpt, map_location="cpu")
    cfg = ValueCVAEConfig(**ckpt["cfg"])
    n_vars_ckpt = int(ckpt["state_dict"]["var_emb.weight"].shape[0])
    n_vars_map = int(max(code2id.values()) + 1) if code2id else 1
    if n_vars_ckpt != n_vars_map:
        raise ValueError(f"n_vars mismatch: ckpt={n_vars_ckpt}, mapping={n_vars_map}")

    ds = ValueEventsDataset(
        meds_reader_db=args.meds_reader_db,
        split=args.split,
        splits_parquet=args.splits_parquet,
        codes_parquet=None,
        include_code_fn=None,
        allowed_var_ids=None,
        shuffle_subjects=False,
        fixed_code2id=code2id,
        strict_codes=True,
    )
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate_value_batch)

    model = ValueCVAE(n_vars=n_vars_ckpt, cfg=cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    mean_by_var = mean_by_var.to(device)
    std_by_var = std_by_var.to(device)

    total_n = 0
    total_abs = 0.0
    total_sq = 0.0
    total_nll_z = 0.0
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    per: Dict[int, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "abs": 0.0, "sq": 0.0, "nll_z": 0.0})

    with torch.no_grad():
        for step, batch in enumerate(dl):
            if args.max_steps >= 0 and step >= args.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            var_id = batch["var_id"].long()
            value = batch["value"].float()
            dt_prev_norm = torch.log1p(batch["dt_prev"].clamp(min=0.0, max=24.0 * 7 * 4))
            age_norm = batch["age"].clamp(min=0.0, max=100.0) / 100.0
            sex_norm = batch["sex"].clamp(0.0, 1.0)

            max_idx = mean_by_var.shape[0] - 1
            vid = var_id.clamp(max=max_idx)
            mean = mean_by_var[vid]
            std = std_by_var[vid]
            value_z = ((value - mean) / std).clamp(min=-8.0, max=8.0)

            out = model(value_z, var_id, dt_prev_norm, age_norm, sex_norm)
            mu_hat_z = out["mu_hat"].squeeze(-1)
            log_sigma = out["log_sigma"].squeeze(-1)
            pred = (mu_hat_z * std) + mean

            abs_err = (pred - value).abs()
            sq_err = (pred - value).pow(2)
            nll_z = 0.5 * (2.0 * log_sigma + ((value_z - mu_hat_z).pow(2) / torch.exp(2.0 * log_sigma)))

            bsz = int(value.shape[0])
            total_n += bsz
            total_abs += float(abs_err.sum().item())
            total_sq += float(sq_err.sum().item())
            total_nll_z += float(nll_z.sum().item())
            total_loss += float(out["loss"].item()) * bsz
            total_recon += float(out["recon_nll"].item()) * bsz
            total_kl += float(out["kl"].item()) * bsz

            vid_cpu = var_id.detach().cpu().tolist()
            abs_cpu = abs_err.detach().cpu().tolist()
            sq_cpu = sq_err.detach().cpu().tolist()
            nll_cpu = nll_z.detach().cpu().tolist()
            for i, v in enumerate(vid_cpu):
                pv = per[int(v)]
                pv["n"] += 1.0
                pv["abs"] += float(abs_cpu[i])
                pv["sq"] += float(sq_cpu[i])
                pv["nll_z"] += float(nll_cpu[i])

    if total_n == 0:
        raise RuntimeError("No evaluation samples yielded; check DB/split/mapping.")

    print(f"device={device}")
    print(f"eval_split={args.split}")
    print(f"eval_samples={total_n}")
    print(f"train_events={train_events}")
    print(f"train_steps_per_epoch(batch={args.train_batch_for_steps})={steps_per_epoch}")
    print(
        "train_sparsity_quantiles="
        f"{sparsity['quantiles']}"
    )
    print(
        "overall "
        f"mae={total_abs/total_n:.6f} "
        f"rmse={math.sqrt(total_sq/total_n):.6f} "
        f"nll_z={total_nll_z/total_n:.6f} "
        f"loss={total_loss/total_n:.6f} "
        f"recon={total_recon/total_n:.6f} "
        f"kl={total_kl/total_n:.6f}"
    )

    rows: List[Tuple[int, str, int, int, float, float, float]] = []
    for vid, d in per.items():
        n = int(d["n"])
        if n <= 0:
            continue
        mae = d["abs"] / n
        rmse = math.sqrt(d["sq"] / n)
        nll = d["nll_z"] / n
        code = id2code.get(int(vid), f"VID_{vid}")
        train_count = int(count_by_var[vid].item()) if vid < len(count_by_var) else 0
        rows.append((int(vid), code, n, train_count, mae, rmse, nll))

    if args.per_var_csv:
        out_csv = Path(args.per_var_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["var_id", "code", "eval_count", "train_count", "mae", "rmse", "nll_z"])
            for r in rows:
                w.writerow(r)
        print(f"wrote_per_var_csv={out_csv}")

    eligible = [r for r in rows if r[2] >= args.min_eval_count]
    by_mae = sorted(eligible, key=lambda x: x[4], reverse=True)[: args.top_k]
    by_nll = sorted(eligible, key=lambda x: x[6], reverse=True)[: args.top_k]

    print(f"top_{args.top_k}_worst_by_mae(min_eval_count={args.min_eval_count}):")
    for r in by_mae:
        print(f"  vid={r[0]} n={r[2]} train_n={r[3]} mae={r[4]:.5f} rmse={r[5]:.5f} nll_z={r[6]:.5f} code={r[1]}")

    print(f"top_{args.top_k}_worst_by_nll_z(min_eval_count={args.min_eval_count}):")
    for r in by_nll:
        print(f"  vid={r[0]} n={r[2]} train_n={r[3]} mae={r[4]:.5f} rmse={r[5]:.5f} nll_z={r[6]:.5f} code={r[1]}")


if __name__ == "__main__":
    main()
