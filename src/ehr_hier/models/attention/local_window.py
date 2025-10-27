import torch
#Stub only
def local_window_mask(attn_mask: torch.Tensor, window: int):
    B,T,_ = attn_mask.shape
    idx = torch.arange(T, device=attn_mask.device)
    band = (idx[None,:,None] - idx[None,None,:]).abs() <= window
    band = band.expand(B, T, T)
    return attn_mask & band