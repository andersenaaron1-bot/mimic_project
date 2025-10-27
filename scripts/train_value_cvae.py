import argparse, time, torch
from torch.utils.data import DataLoader
from src.ehr_hier.models.value_encoders.cvae import ValueCVAE, ValueCVAEConfig
from src.ehr_hier.data.meds_value_dataset import ValueEventsDataset, collate_value_batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meds_root', type=str, required=True)
    ap.add_argument('--split', type=str, default='train')
    ap.add_argument('--n_vars', type=int, required=True)   # size of MEDS vocab
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--z_dim', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--var_emb_dim', type=int, default=64)
    ap.add_argument('--beta_kl', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=str, default='./cvae_ckpt.pt')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = ValueEventsDataset(args.meds_root, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate_value_batch)

    cfg = ValueCVAEConfig(z_dim=args.z_dim, hidden=args.hidden, var_emb_dim=args.var_emb_dim, beta_kl=args.beta_kl)
    model = ValueCVAE(n_vars=args.n_vars, cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch['value'], batch['var_id'], batch['dt_prev'], batch['age'], batch['sex'])
            opt.zero_grad(set_to_none=True)
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                print(f"ep {epoch} step {step} loss {out['loss'].item():.4f} | recon {out['recon_nll'].item():.4f} | kl {out['kl'].item():.4f}")
            step += 1
        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s")
    torch.save({'cfg': cfg.__dict__, 'state_dict': model.state_dict()}, args.out)
    print(f"Saved {args.out}")

if __name__ == '__main__':
    main()
