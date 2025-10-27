import argparse, time, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.ehr_hier.models.value_encoders.cvae import ValueCVAE, ValueCVAEConfig
from src.ehr_hier.tokenizers.value_tokenizer import ValueDiscretizer, TokenizerConfig
from src.ehr_hier.data.meds_value_dataset import ValueEventsDataset, collate_value_batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meds_root', type=str, required=True)
    ap.add_argument('--split', type=str, default='train')
    ap.add_argument('--cvae_ckpt', type=str, required=True)
    ap.add_argument('--n_vars', type=int, required=True)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--workers', type=int, default=8)
    # tokenizer cfg
    ap.add_argument('--d_val', type=int, default=64)
    ap.add_argument('--num_codebooks', type=int, default=2)
    ap.add_argument('--codebook_size', type=int, default=256)
    ap.add_argument('--beta_commit', type=float, default=0.25)
    ap.add_argument('--attn_topk', type=int, default=8)
    ap.add_argument('--attn_temp', type=float, default=0.5)
    ap.add_argument('--aux_soft_weight', type=float, default=0.05)
    ap.add_argument('--aux_entropy_weight', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--out', type=str, default='./value_tokenizer.pt')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Frozen cVAE encoder
    ckpt = torch.load(args.cvae_ckpt, map_location='cpu')
    cvae_cfg = ValueCVAEConfig(**ckpt['cfg'])
    assert cvae_cfg.z_dim == args.d_val, f"cVAE z_dim={cvae_cfg.z_dim} != d_val={args.d_val}"
    cvae = ValueCVAE(n_vars=args.n_vars, cfg=cvae_cfg).to(device).eval()
    cvae.load_state_dict(ckpt['state_dict'], strict=False)
    for p in cvae.parameters(): p.requires_grad = False

    # 2) Tokenizer model
    tok = ValueDiscretizer(TokenizerConfig(
        d_val=args.d_val,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        beta_commit=args.beta_commit,
        attn_topk=args.attn_topk,
        attn_temp=args.attn_temp,
        aux_soft_weight=args.aux_soft_weight,
        aux_entropy_weight=args.aux_entropy_weight,
    )).to(device)
    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr)

    # 3) Dataset streaming numeric events; -> batch and feed through cVAE then tokenizer
    ds = ValueEventsDataset(args.meds_root, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate_value_batch)

    step = 0
    for ep in range(args.epochs):
        t0 = time.time()
        for b in dl:
            for k in b: b[k] = b[k].to(device)
            # prepare [B,T,D]: here we have a flat batch [N], so fake B=1,T=N
            z = cvae.encode_mu(b['value'].unsqueeze(0), b['var_id'].unsqueeze(0),
                               b['dt_prev'].unsqueeze(0), b['age'].unsqueeze(0), b['sex'].unsqueeze(0))  # [1,N,D]
            out = tok(z)  # {'indices','z_hat','loss','stats'}
            # add simple reconstruction loss of z vs z_hat (helps early training)
            rec = F.mse_loss(out['z_hat'], z.detach())
            loss = out['loss'] + rec

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                s = out['stats']
                print(f"ep {ep} step {step} loss {loss.item():.4f} "
                      f"| vq {s['vq_loss']:.4f} soft {s['soft_rec']:.4f} "
                      f"| uniq_lvl {s['avg_uniq_codes_per_level']:.1f} "
                      f"| attn_H {s['attn_entropy']:.3f}")
            step += 1
        print(f"Epoch {ep} in {time.time()-t0:.1f}s")

    torch.save({
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
    }, args.out)
    print(f"Saved tokenizer → {args.out}")


if __name__ == "__main__":
    main()
