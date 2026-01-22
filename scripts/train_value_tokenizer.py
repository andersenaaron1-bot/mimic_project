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
    from collections import defaultdict
    sum_by_var = defaultdict(float); sumsq_by_var = defaultdict(float); count_by_var = defaultdict(int)
    g_sum = g_sumsq = 0.0; g_count = 0

    for i, s in enumerate(dataset):
        if max_samples is not None and i >= max_samples: break
        v, vid = s.get("value"), s.get("var_id")
        if v is None or vid is None: continue
        try:
            v = float(v); vid = int(vid)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or vid <= 0: continue
        sum_by_var[vid] += v; sumsq_by_var[vid] += v*v; count_by_var[vid] += 1
        g_sum += v; g_sumsq += v*v; g_count += 1

    if g_count == 0:
        return torch.tensor([0.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32)

    g_mean = g_sum / g_count
    g_var  = max(0.0, g_sumsq / g_count - g_mean*g_mean)
    g_std  = max(min_std, math.sqrt(g_var))

    max_vid = max(count_by_var.keys()) if count_by_var else 0
    mean = torch.zeros(max_vid + 1, dtype=torch.float32)
    std  = torch.ones (max_vid + 1, dtype=torch.float32)
    for vid in range(1, max_vid + 1):
        c = count_by_var.get(vid, 0)
        if c >= min_count:
            m = sum_by_var[vid] / c
            var = max(0.0, (sumsq_by_var[vid] / c) - m*m)
            sd = max(min_std, math.sqrt(var))
        else:
            m, sd = g_mean, g_std
        mean[vid] = m; std[vid] = sd
    return mean, std

def audit_filters(db_path, splits_parquet, split, code2id, limit_subjects=80):
    import meds_reader as mr, pandas as pd, math
    from collections import Counter

    keep = set(pd.read_parquet(splits_parquet)
                 .query("split == @split")["subject_id"].astype(int))
    db = mr.SubjectDatabase(db_path)
    c = Counter()
    visited = 0

    for sid in db:
        sid = int(sid)
        if sid not in keep:
            continue
        visited += 1
        subj = db[sid]
        for ev in subj.events:
            # time filter (your dataset requires a proper datetime with .timestamp())
            t = getattr(ev, "time", None)
            if t is None or not hasattr(t, "timestamp"):
                c["drop:no_time"] += 1
                continue
            try:
                _ = t.timestamp()
            except Exception:
                c["drop:bad_time"] += 1
                continue

            code = getattr(ev, "code", None)
            if code is None:
                c["drop:no_code"] += 1
                continue
            code = str(code)
            if code not in code2id:
                c["drop:unknown_code"] += 1
                continue

            v = getattr(ev, "numeric_value", None)
            if v is None:
                c["drop:no_numeric"] += 1
                continue
            try:
                v = float(v)
            except Exception:
                c["drop:bad_numeric_cast"] += 1
                continue
            if not math.isfinite(v):
                c["drop:nonfinite"] += 1
                continue

            c["ok"] += 1

        if visited >= limit_subjects:
            break

    print(f"visited subjects: {visited} / {len(keep)}")
    print(c)

def main():
    ap = argparse.ArgumentParser()
    # meds_reader DB and split
    ap.add_argument("--meds_reader_db", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--splits_parquet", type=str, required=True)

    # NEW: frozen mapping + stats (match cVAE training!)
    ap.add_argument("--code2id_pt", type=str, required=True, help="torch.save'd dict {code->var_id} used by cVAE")
    ap.add_argument("--stats_pt", type=str, default=None, help="torch file with {'mean_by_var','std_by_var'}")

    # cVAE ckpt
    ap.add_argument("--cvae_ckpt", type=str, required=True)

    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)  # set 0 on Windows if dataset not worker-safe

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

    # ---- frozen mapping
    code2id = torch.load(args.code2id_pt)
    n_vars_map = max(code2id.values()) + 1

    # ---- frozen cVAE encoder
    ckpt = torch.load(args.cvae_ckpt, map_location="cpu")
    cvae_cfg = ValueCVAEConfig(**ckpt["cfg"])
    assert cvae_cfg.z_dim == args.d_val, f"cVAE z_dim={cvae_cfg.z_dim} != d_val={args.d_val}"

    n_vars_cvae = ckpt["state_dict"]["var_emb.weight"].shape[0]
    assert n_vars_cvae == n_vars_map, f"mapping n_vars={n_vars_map} != cVAE n_vars={n_vars_cvae}"
    print("Loaded cVAE with n_vars =", n_vars_cvae)

    cvae = ValueCVAE(n_vars=n_vars_cvae, cfg=cvae_cfg).to(device).eval()
    cvae.load_state_dict(ckpt["state_dict"], strict=True)
    for p in cvae.parameters(): p.requires_grad = False

    print("mapping file:", args.code2id_pt)
    code2id = torch.load(args.code2id_pt, map_location="cpu")
    print("mapping size:", len(code2id))
    print("split:", args.split, "splits_parquet:", args.splits_parquet)

    for i, k in enumerate(list(code2id.keys())[:5]):
        print("map sample:", i, k, "->", code2id[k])

    # ---- datasets (must use the same frozen mapping + strict_codes)
    ds_stats = ValueEventsDataset(
        meds_reader_db=args.meds_reader_db,
        split=args.split,
        splits_parquet=args.splits_parquet,
        codes_parquet=None,
        shuffle_subjects=False,
        fixed_code2id=code2id,
        strict_codes=True,
    )
    ds = ValueEventsDataset(
        meds_reader_db=args.meds_reader_db,
        split=args.split,
        splits_parquet=args.splits_parquet,
        codes_parquet=None,
        shuffle_subjects=True,
        fixed_code2id=code2id,
        strict_codes=True,
    )
    print(f"subjects in split: {len(ds.subject_ids)}")

    it = iter(ds)
    try:
        first = next(it)
        print("first sample:", first)
    except StopIteration:
        print("DATASET EMPTY")

    audit_filters(args.meds_reader_db, args.splits_parquet, args.split, code2id)

    # ---- stats: load (preferred) or compute on the same split+mapping
    if args.stats_pt:
        st = torch.load(args.stats_pt, map_location="cpu")
        mean_by_var, std_by_var = st["mean_by_var"], st["std_by_var"]
        print(f"Loaded stats from {args.stats_pt} (max_vid={mean_by_var.shape[0]-1})")
    else:
        print("Estimating per-variable mean/std for value (tokenizer, train split)...")
        mean_by_var, std_by_var = estimate_var_stats_from_dataset(
            dataset=ds_stats,
            max_samples=100_000,
            min_std=1e-3,
            min_count=10,
        )
        print("  max var_id seen:", mean_by_var.shape[0] - 1)

    mean_by_var = mean_by_var.to(device)
    std_by_var  = std_by_var.to(device)

    from collections import Counter

    cnt = Counter()
    # iterate raw subjects without DataLoader
    for sid in ds.subject_ids[:50]:  # sample some subjects
        import meds_reader as mr
        subj = mr.SubjectDatabase(args.meds_reader_db)[int(sid)]
        for ev in subj.events:
            v = getattr(ev, "numeric_value", None)
            if v is None:
                continue
            cnt["total_numeric"] += 1
            code = getattr(ev, "code", None)
            if code is None:
                continue
            if str(code) in code2id:  # your loaded frozen mapping
                cnt["mapped"] += 1
            else:
                cnt["unmapped"] += 1

    print(cnt)

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.workers,
        collate_fn=collate_value_batch,
        # persistent_workers=False  # optional
    )
    # Peek one batch
    b = next(iter(dl))
    max_vid_batch = int(b["var_id"].max())
    print("sanity: max var_id in first batch =", max_vid_batch,
          "| cVAE n_vars =", n_vars_cvae,
          "| stats max_vid =", mean_by_var.shape[0] - 1)

    assert max_vid_batch < n_vars_cvae, "var_id exceeds cVAE var_emb size → mapping mismatch"
    assert max_vid_batch < mean_by_var.shape[0], "var_id exceeds stats range → stats mismatch"

    # ---- tokenizer model
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

    # ---- training
    step = 0
    for ep in range(args.epochs):
        t0 = time.time()
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            var_id = b["var_id"].long()
            value  = b["value"].float()
            dt_prev = b["dt_prev"].float()
            age    = b["age"].float()
            sex    = b["sex"].float()

            # same normalization as cVAE
            max_idx = min(var_id.max().item(), mean_by_var.shape[0] - 1)
            vid = var_id.clamp_max(max_idx)
            mean = mean_by_var[vid]; std = std_by_var[vid]
            value_z = (value - mean) / std
            value_z = value_z.clamp(-8.0, 8.0)
            dt_prev_norm = torch.log1p(dt_prev.clamp(0.0, 24.0 * 7 * 4))
            age_norm     = age.clamp(0.0, 100.0) / 100.0
            sex_norm     = sex.clamp(0.0, 1.0)

            with torch.no_grad():
                z = cvae.encode_mu(value_z, var_id, dt_prev_norm, age_norm, sex_norm)
                if z.dim() == 2:  # [N,D] -> [1,N,D]
                    z = z.unsqueeze(0)

            out = tok(z)  # {'indices','z_hat','loss','stats'}
            rec = F.mse_loss(out["z_hat"], z.detach())
            loss = out["loss"] + rec

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                s = out["stats"]
                print(f"ep {ep} step {step} loss {loss.item():.4f} "
                      f"| vq {s['vq_loss']:.4f} soft {s['soft_rec']:.4f} "
                      f"| uniq_lvl {s.get('avg_uniq_codes_per_level', -1):.1f} "
                      f"| attn_H {s['attn_entropy']:.3f}")
            step += 1
        print(f"Epoch {ep} in {time.time()-t0:.1f}s")

    torch.save({"cfg": tok_cfg.__dict__, "state_dict": tok.state_dict()}, args.out)
    print(f"Saved tokenizer → {args.out}")

if __name__ == "__main__":
    main()



