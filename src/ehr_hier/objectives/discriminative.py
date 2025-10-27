import torch, torch.nn.functional as F
def mortality_loss(logits: torch.Tensor, targets: torch.Tensor | None = None):
    if targets is None:
        targets = (torch.sigmoid(logits.detach()) > 0.5).float()
    return F.binary_cross_entropy_with_logits(logits, targets)
