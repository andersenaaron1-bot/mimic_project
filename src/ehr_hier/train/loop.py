import torch, torch.optim as optim
from ..utils.checkpoint import save_ckpt
from ..utils.logging import log
from ..objectives.discriminative import mortality_loss

def train_one(cfg, model, loader, device):
    model.to(device); model.train()
    opt = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    step = 0
    for batch in loader:
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        loss = mortality_loss(model(batch))
        loss.backward(); opt.step(); opt.zero_grad(); step += 1
        if step % 10 == 0: log(f"[step {step}] loss={loss.item():.4f}")
        if step % cfg.trainer.ckpt_every == 0: save_ckpt(model, opt, step, cfg.out.dir)
        if step >= cfg.trainer.max_steps: break
